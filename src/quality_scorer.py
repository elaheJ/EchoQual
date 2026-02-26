"""
Self-supervised quality scoring: three proxy signals + fusion.

Signal 1: View Classification Confidence
  - High softmax confidence / low entropy → canonical view → likely high quality
  - Uses a view classifier trained with SSL pseudo-labels

Signal 2: Embedding Density Scoring
  - Compute distance to view-specific centroids in embedding space
  - High-quality images cluster tightly; degraded images are outliers
  - Methods: k-NN distance, Mahalanobis distance, GMM likelihood

Signal 3: Vision-Language Alignment
  - Cosine similarity between video embeddings and canonical text descriptions
  - Higher alignment with "good quality" texts → better quality
  - Differential: similarity(good) - similarity(poor) for robustness

Fusion: weighted combination of normalized scores.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler


class ViewConfidenceScorer:
    """
    Signal 1: Quality proxy from view classification confidence.

    Intuition: A model trained to recognize A4C views will be less confident
    on poorly acquired images where expected anatomical structures are
    missing or distorted.
    """

    def __init__(self, method: str = "entropy"):
        """
        Args:
            method: 'max_softmax', 'entropy', or 'energy'
        """
        self.method = method

    def score(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute confidence score from view classification logits.

        Args:
            logits: [N, num_classes] raw logits from view classifier
        Returns:
            scores: [N] quality scores (higher = better quality)
        """
        logits = logits.detach().cpu()

        if self.method == "max_softmax":
            probs = F.softmax(logits, dim=-1)
            scores = probs.max(dim=-1).values.numpy()

        elif self.method == "entropy":
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            # Lower entropy = higher confidence = higher quality
            scores = (1.0 - entropy / np.log(logits.shape[-1])).numpy()

        elif self.method == "energy":
            # Energy-based OOD detection: -T * logsumexp(logits / T)
            T = 1.0
            energy = T * torch.logsumexp(logits / T, dim=-1)
            scores = energy.numpy()  # Higher energy = more in-distribution

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores


class EmbeddingDensityScorer:
    """
    Signal 2: Quality proxy from embedding space density.

    High-quality images form tight clusters in the learned representation
    space. Poor-quality images are outliers with high distance to centroids.
    """

    def __init__(
        self,
        method: str = "knn",
        k: int = 10,
        normalize: bool = True,
    ):
        self.method = method
        self.k = k
        self.normalize = normalize
        self.centroid = None
        self.covariance_inv = None
        self.knn_model = None
        self.gmm = None
        self.reference_embeddings = None

    def fit(self, embeddings: np.ndarray):
        """
        Fit density model on reference (training) embeddings.

        Args:
            embeddings: [N, D] reference embeddings from training set
        """
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        self.reference_embeddings = embeddings
        self.centroid = embeddings.mean(axis=0)

        if self.method == "knn":
            self.knn_model = NearestNeighbors(n_neighbors=self.k, metric="cosine")
            self.knn_model.fit(embeddings)

        elif self.method == "mahalanobis":
            cov = np.cov(embeddings.T) + 1e-6 * np.eye(embeddings.shape[1])
            self.covariance_inv = np.linalg.inv(cov)

        elif self.method == "gmm":
            self.gmm = GaussianMixture(
                n_components=min(5, len(embeddings) // 10),
                covariance_type="full",
                random_state=42,
            )
            self.gmm.fit(embeddings)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute density-based quality scores.

        Args:
            embeddings: [N, D] query embeddings
        Returns:
            scores: [N] quality scores (higher = closer to distribution = better)
        """
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        if self.method == "knn":
            distances, _ = self.knn_model.kneighbors(embeddings)
            avg_dist = distances.mean(axis=1)
            # Invert: lower distance = higher quality
            scores = 1.0 / (1.0 + avg_dist)

        elif self.method == "mahalanobis":
            scores = np.array([
                -mahalanobis(emb, self.centroid, self.covariance_inv)
                for emb in embeddings
            ])
            # Negate so higher = better quality
            scores = -scores

        elif self.method == "gmm":
            log_likelihood = self.gmm.score_samples(embeddings)
            scores = log_likelihood  # Higher log-likelihood = better quality

        else:
            # Fallback: cosine distance to centroid
            cos_sim = embeddings @ self.centroid / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(self.centroid) + 1e-8
            )
            scores = cos_sim

        return scores


class VLAlignmentScorer:
    """
    Signal 3: Quality proxy from vision-language alignment.

    Measures how well a video embedding aligns with canonical text
    descriptions of properly acquired views. Uses a differential score:
    similarity(good_texts) - similarity(poor_texts) for robustness.
    """

    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.text_encoder_name = text_encoder_name
        self.good_text_embeddings = None
        self.poor_text_embeddings = None
        self.text_encoder = None
        self.video_projection = None
        self.text_projection = None

    def _load_text_encoder(self):
        """Lazy-load sentence transformer."""
        if self.text_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.text_encoder = SentenceTransformer(self.text_encoder_name)
            except ImportError:
                print("WARNING: sentence-transformers not available. "
                      "Using random text embeddings for demonstration.")
                self.text_encoder = None

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text descriptions to embeddings."""
        self._load_text_encoder()
        if self.text_encoder is not None:
            return self.text_encoder.encode(texts, normalize_embeddings=True)
        else:
            # Fallback: deterministic pseudo-random embeddings based on text hash
            np.random.seed(42)
            return np.random.randn(len(texts), 384).astype(np.float32)

    def fit(
        self,
        good_texts: List[str],
        poor_texts: List[str],
        video_embeddings: np.ndarray,
        video_projection: Optional[nn.Module] = None,
    ):
        """
        Precompute text embeddings and optionally learn projection.

        Args:
            good_texts: Descriptions of well-acquired views
            poor_texts: Descriptions of poorly-acquired views
            video_embeddings: [N, D] training video embeddings for projection alignment
            video_projection: Optional learned projection layer
        """
        self.good_text_embeddings = self.encode_texts(good_texts)
        self.poor_text_embeddings = self.encode_texts(poor_texts)
        self.video_projection = video_projection

    def score(
        self,
        video_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute VL alignment quality scores.

        Strategy: Measure cosine similarity between video embeddings and
        canonical "good" texts, minus similarity to "poor" texts.

        When dimensions mismatch (video vs text), we use a simple linear
        projection or CKA-style comparison.

        Args:
            video_embeddings: [N, D_video]
        Returns:
            scores: [N] differential alignment scores
        """
        # Normalize video embeddings
        v_norm = video_embeddings / (
            np.linalg.norm(video_embeddings, axis=1, keepdims=True) + 1e-8
        )

        if self.good_text_embeddings is None:
            return np.zeros(len(video_embeddings))

        # If dimensions differ, use PCA alignment or learned projection
        if v_norm.shape[1] != self.good_text_embeddings.shape[1]:
            # Simple approach: project both to shared dim via SVD
            shared_dim = min(v_norm.shape[1], self.good_text_embeddings.shape[1], 128)

            from sklearn.decomposition import PCA

            pca_v = PCA(n_components=shared_dim, random_state=42)
            v_proj = pca_v.fit_transform(v_norm)

            pca_t = PCA(n_components=shared_dim, random_state=42)
            good_t_proj = pca_t.fit_transform(self.good_text_embeddings)
            poor_t_proj = pca_t.transform(self.poor_text_embeddings)
        else:
            v_proj = v_norm
            good_t_proj = self.good_text_embeddings
            poor_t_proj = self.poor_text_embeddings

        # Normalize projections
        v_proj = v_proj / (np.linalg.norm(v_proj, axis=1, keepdims=True) + 1e-8)
        good_t_proj = good_t_proj / (np.linalg.norm(good_t_proj, axis=1, keepdims=True) + 1e-8)
        poor_t_proj = poor_t_proj / (np.linalg.norm(poor_t_proj, axis=1, keepdims=True) + 1e-8)

        # Average similarity to good texts
        sim_good = (v_proj @ good_t_proj.T).mean(axis=1)

        # Average similarity to poor texts
        sim_poor = (v_proj @ poor_t_proj.T).mean(axis=1)

        # Differential score
        scores = sim_good - sim_poor

        return scores


class QualityScoreFusion:
    """
    Fuses three quality proxy signals into a single composite score.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted_sum",
        normalize: bool = True,
    ):
        self.weights = weights or {
            "view_confidence": 0.3,
            "embedding_density": 0.4,
            "vl_alignment": 0.3,
        }
        self.method = method
        self.normalize = normalize
        self.scalers = {}

    def fuse(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine multiple quality signals into a single score.

        Args:
            scores: Dict mapping signal name to [N] score arrays
        Returns:
            composite: [N] fused quality scores
        """
        normalized = {}
        for name, s in scores.items():
            if self.normalize:
                scaler = MinMaxScaler()
                s_reshaped = s.reshape(-1, 1)
                s_norm = scaler.fit_transform(s_reshaped).flatten()
                self.scalers[name] = scaler
                normalized[name] = s_norm
            else:
                normalized[name] = s

        if self.method == "weighted_sum":
            composite = np.zeros_like(list(normalized.values())[0])
            for name, s in normalized.items():
                w = self.weights.get(name, 1.0 / len(normalized))
                composite += w * s

        elif self.method == "rank_aggregation":
            # Borda count: rank each signal, average ranks
            ranks = {}
            N = len(list(normalized.values())[0])
            for name, s in normalized.items():
                order = np.argsort(-s)  # Descending
                rank = np.empty_like(order)
                rank[order] = np.arange(N)
                ranks[name] = rank

            avg_rank = np.mean(list(ranks.values()), axis=0)
            composite = 1.0 - avg_rank / N  # Convert rank to score

        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        return composite


class EchoQualityScorer:
    """
    Main quality scoring class that orchestrates all three signals.
    """

    def __init__(
        self,
        view_conf_method: str = "entropy",
        density_method: str = "knn",
        density_k: int = 10,
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        fusion_weights: Optional[Dict[str, float]] = None,
        fusion_method: str = "weighted_sum",
    ):
        self.view_scorer = ViewConfidenceScorer(method=view_conf_method)
        self.density_scorer = EmbeddingDensityScorer(method=density_method, k=density_k)
        self.vl_scorer = VLAlignmentScorer(text_encoder_name=text_encoder)
        self.fusion = QualityScoreFusion(
            weights=fusion_weights, method=fusion_method
        )

    def fit(
        self,
        reference_embeddings: np.ndarray,
        good_texts: List[str],
        poor_texts: List[str],
    ):
        """
        Fit scorers on reference (training) data.
        No quality labels needed — only embeddings and canonical texts.
        """
        self.density_scorer.fit(reference_embeddings)
        self.vl_scorer.fit(good_texts, poor_texts, reference_embeddings)

    def score(
        self,
        embeddings: np.ndarray,
        view_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all quality scores for a set of video embeddings.

        Args:
            embeddings: [N, D] video embeddings
            view_logits: [N, C] optional view classifier logits
        Returns:
            dict with individual signal scores and composite score
        """
        scores = {}

        # Signal 1: View confidence
        if view_logits is not None:
            scores["view_confidence"] = self.view_scorer.score(view_logits)
        else:
            scores["view_confidence"] = np.ones(len(embeddings))

        # Signal 2: Embedding density
        scores["embedding_density"] = self.density_scorer.score(embeddings)

        # Signal 3: VL alignment
        scores["vl_alignment"] = self.vl_scorer.score(embeddings)

        # Fusion
        scores["composite"] = self.fusion.fuse(scores)

        return scores
