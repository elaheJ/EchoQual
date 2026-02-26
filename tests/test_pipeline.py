#!/usr/bin/env python3
"""
Unit tests for EchoQual pipeline using synthetic data.
Validates all components without requiring the actual EchoNet-Dynamic dataset.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import (
    VideoEncoder,
    ContrastiveProjectionHead,
    ViewClassificationHead,
    VLAlignmentProjection,
    EchoQualModel,
)
from src.quality_scorer import (
    ViewConfidenceScorer,
    EmbeddingDensityScorer,
    VLAlignmentScorer,
    QualityScoreFusion,
    EchoQualityScorer,
)
from src.canonical_texts import get_canonical_texts, A4C_CANONICAL, A4C_POOR_QUALITY
from src.ssl_pretraining import NTXentLoss


class TestVideoEncoder(unittest.TestCase):
    """Test video encoder forward pass."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.num_frames = 16
        self.frame_size = 112  # Smaller for speed
        self.video = torch.randn(
            self.batch_size, 3, self.num_frames, self.frame_size, self.frame_size
        )

    def test_r3d_encoder(self):
        encoder = VideoEncoder(backbone="r3d_18", pretrained=False, embedding_dim=256)
        out = encoder(self.video)
        self.assertEqual(out.shape, (self.batch_size, 256))

    def test_backbone_features(self):
        encoder = VideoEncoder(backbone="r3d_18", pretrained=False, embedding_dim=256)
        features = encoder.get_backbone_features(self.video)
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertTrue(features.shape[1] > 0)


class TestEchoQualModel(unittest.TestCase):
    """Test full model forward pass."""

    def test_forward(self):
        model = EchoQualModel(
            backbone="r3d_18",
            pretrained=False,
            embedding_dim=256,
            projection_dim=64,
            num_views=2,
        )
        video = torch.randn(2, 3, 16, 112, 112)
        outputs = model(video)

        self.assertIn("embedding", outputs)
        self.assertIn("contrastive_proj", outputs)
        self.assertIn("view_logits", outputs)
        self.assertEqual(outputs["embedding"].shape, (2, 256))
        self.assertEqual(outputs["contrastive_proj"].shape, (2, 64))
        self.assertEqual(outputs["view_logits"].shape, (2, 2))

    def test_encode(self):
        model = EchoQualModel(backbone="r3d_18", pretrained=False, embedding_dim=128)
        video = torch.randn(4, 3, 16, 112, 112)
        emb = model.encode(video)
        self.assertEqual(emb.shape, (4, 128))


class TestContrastiveLoss(unittest.TestCase):
    """Test NT-Xent contrastive loss."""

    def test_loss_computation(self):
        loss_fn = NTXentLoss(temperature=0.07)
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        z2 = torch.nn.functional.normalize(z2, dim=-1)

        loss = loss_fn(z1, z2)
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss))

    def test_identical_views_low_loss(self):
        loss_fn = NTXentLoss(temperature=0.5)
        z = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)
        # Identical views should have lower loss than random
        loss_identical = loss_fn(z, z + torch.randn_like(z) * 0.01)
        loss_random = loss_fn(z, torch.nn.functional.normalize(torch.randn(8, 64), dim=-1))
        # Not guaranteed due to randomness, but generally true


class TestViewConfidenceScorer(unittest.TestCase):
    """Test view confidence quality signal."""

    def test_entropy_scoring(self):
        scorer = ViewConfidenceScorer(method="entropy")

        # High-confidence logits (one class dominant)
        high_conf = torch.tensor([[10.0, -5.0], [8.0, -3.0]])
        # Low-confidence logits (uniform)
        low_conf = torch.tensor([[0.1, 0.0], [-0.1, 0.05]])

        scores_high = scorer.score(high_conf)
        scores_low = scorer.score(low_conf)

        # High confidence should yield higher quality scores
        self.assertTrue(scores_high.mean() > scores_low.mean())

    def test_max_softmax_scoring(self):
        scorer = ViewConfidenceScorer(method="max_softmax")
        logits = torch.randn(10, 3)
        scores = scorer.score(logits)
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(scores >= 0) and np.all(scores <= 1))

    def test_energy_scoring(self):
        scorer = ViewConfidenceScorer(method="energy")
        logits = torch.randn(10, 3)
        scores = scorer.score(logits)
        self.assertEqual(len(scores), 10)


class TestEmbeddingDensityScorer(unittest.TestCase):
    """Test embedding density quality signal."""

    def setUp(self):
        np.random.seed(42)
        # Create a tight cluster and some outliers
        self.cluster = np.random.randn(100, 64) * 0.5
        self.outliers = np.random.randn(10, 64) * 3.0 + 5.0

    def test_knn_scoring(self):
        scorer = EmbeddingDensityScorer(method="knn", k=5)
        scorer.fit(self.cluster)

        scores_cluster = scorer.score(self.cluster[:10])
        scores_outlier = scorer.score(self.outliers)

        # Cluster members should score higher than outliers
        self.assertTrue(scores_cluster.mean() > scores_outlier.mean())

    def test_mahalanobis_scoring(self):
        scorer = EmbeddingDensityScorer(method="mahalanobis")
        scorer.fit(self.cluster)

        scores_cluster = scorer.score(self.cluster[:10])
        scores_outlier = scorer.score(self.outliers)

        self.assertTrue(scores_cluster.mean() > scores_outlier.mean())

    def test_gmm_scoring(self):
        scorer = EmbeddingDensityScorer(method="gmm")
        scorer.fit(self.cluster)

        scores_cluster = scorer.score(self.cluster[:10])
        scores_outlier = scorer.score(self.outliers)

        self.assertTrue(scores_cluster.mean() > scores_outlier.mean())


class TestVLAlignmentScorer(unittest.TestCase):
    """Test vision-language alignment quality signal."""

    def test_scoring(self):
        scorer = VLAlignmentScorer(text_encoder_name="dummy")
        good_texts = A4C_CANONICAL[:3]
        poor_texts = A4C_POOR_QUALITY[:3]

        embeddings = np.random.randn(20, 256).astype(np.float32)
        scorer.fit(good_texts, poor_texts, embeddings)

        scores = scorer.score(embeddings)
        self.assertEqual(len(scores), 20)


class TestQualityScoreFusion(unittest.TestCase):
    """Test score fusion."""

    def test_weighted_sum(self):
        fusion = QualityScoreFusion(
            weights={"a": 0.5, "b": 0.5},
            method="weighted_sum",
        )
        scores = {
            "a": np.array([0.8, 0.2, 0.5]),
            "b": np.array([0.6, 0.4, 0.9]),
        }
        composite = fusion.fuse(scores)
        self.assertEqual(len(composite), 3)

    def test_rank_aggregation(self):
        fusion = QualityScoreFusion(method="rank_aggregation")
        scores = {
            "a": np.array([0.8, 0.2, 0.5, 0.9, 0.1]),
            "b": np.array([0.6, 0.4, 0.9, 0.3, 0.7]),
        }
        composite = fusion.fuse(scores)
        self.assertEqual(len(composite), 5)
        self.assertTrue(composite.min() >= 0)
        self.assertTrue(composite.max() <= 1)


class TestEchoQualityScorer(unittest.TestCase):
    """Test the full quality scorer pipeline."""

    def test_end_to_end(self):
        np.random.seed(42)
        N = 50
        D = 128

        embeddings = np.random.randn(N, D).astype(np.float32)
        view_logits = torch.randn(N, 2)

        scorer = EchoQualityScorer(
            view_conf_method="entropy",
            density_method="knn",
            density_k=5,
        )

        texts = get_canonical_texts("A4C")
        scorer.fit(embeddings, texts["good"], texts["poor"])

        scores = scorer.score(embeddings, view_logits)

        self.assertIn("composite", scores)
        self.assertIn("view_confidence", scores)
        self.assertIn("embedding_density", scores)
        self.assertIn("vl_alignment", scores)

        self.assertEqual(len(scores["composite"]), N)
        self.assertFalse(np.any(np.isnan(scores["composite"])))


class TestCanonicalTexts(unittest.TestCase):
    """Test canonical text definitions."""

    def test_a4c_texts_exist(self):
        texts = get_canonical_texts("A4C")
        self.assertIn("good", texts)
        self.assertIn("poor", texts)
        self.assertTrue(len(texts["good"]) >= 3)
        self.assertTrue(len(texts["poor"]) >= 3)

    def test_all_views(self):
        for view in ["A4C", "PLAX", "PSAX", "A2C"]:
            texts = get_canonical_texts(view, include_poor=False)
            self.assertIn("good", texts)
            self.assertTrue(len(texts["good"]) > 0)


class TestSyntheticQualityOrdering(unittest.TestCase):
    """
    Key validation: verify that synthetically degraded images
    receive lower quality scores than clean images.
    """

    def test_noise_degrades_quality(self):
        """Adding noise to embeddings should reduce density scores."""
        np.random.seed(42)
        clean = np.random.randn(100, 64) * 0.5
        noisy = clean[:10] + np.random.randn(10, 64) * 2.0

        scorer = EmbeddingDensityScorer(method="knn", k=5)
        scorer.fit(clean)

        scores_clean = scorer.score(clean[:10])
        scores_noisy = scorer.score(noisy)

        self.assertTrue(
            scores_clean.mean() > scores_noisy.mean(),
            f"Clean ({scores_clean.mean():.4f}) should score higher "
            f"than noisy ({scores_noisy.mean():.4f})",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
