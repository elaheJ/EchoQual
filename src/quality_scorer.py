"""
Self-supervised quality scoring: three proxy signals + fusion.
Now VIEW-AWARE: builds per-view centroids and canonical text anchors.

Signal 1: View Classification Confidence (entropy-based)
Signal 2: Embedding Density (per-view k-NN / Mahalanobis)
Signal 3: Vision-Language Alignment (per-view canonical text matching)
Fusion: weighted combination of normalized scores.
"""

from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler


class ViewConfidenceScorer:
    """Signal 1: Quality from view classification confidence."""

    def __init__(self, method: str = "entropy"):
        self.method = method

    def score(self, logits: torch.Tensor) -> np.ndarray:
        logits = logits.detach().cpu()
        if self.method == "entropy":
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            scores = (1.0 - entropy / np.log(logits.shape[-1])).numpy()
        elif self.method == "max_softmax":
            scores = F.softmax(logits, dim=-1).max(dim=-1).values.numpy()
        elif self.method == "energy":
            scores = torch.logsumexp(logits, dim=-1).numpy()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return scores


class EmbeddingDensityScorer:
    """
    Signal 2: Per-view embedding density.
    Builds separate centroids/kNN models per view so that quality is
    measured relative to same-view neighbors, not cross-view distance.
    """

    def __init__(self, method: str = "knn", k: int = 10, normalize: bool = True):
        self.method = method
        self.k = k
        self.normalize = normalize
        self.per_view_models: Dict = {}
        self.global_model = None

    def _norm(self, emb: np.ndarray) -> np.ndarray:
        if self.normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / (norms + 1e-8)
        return emb

    def fit(self, embeddings: np.ndarray, views: Optional[List[str]] = None):
        """
        Fit density models. If views provided, build per-view models.
        Falls back to a single global model if views is None.
        """
        embeddings = self._norm(embeddings)

        if views is not None:
            view_set = sorted(set(views))
            for v in view_set:
                mask = np.array([vv == v for vv in views])
                v_emb = embeddings[mask]
                if len(v_emb) < max(self.k, 3):
                    continue
                model = NearestNeighbors(
                    n_neighbors=min(self.k, len(v_emb) - 1), metric="cosine"
                )
                model.fit(v_emb)
                self.per_view_models[v] = {
                    "knn": model,
                    "centroid": v_emb.mean(axis=0),
                    "count": len(v_emb),
                }
            print(f"  [EmbeddingDensity] Built per-view models for {list(self.per_view_models.keys())}")

        # Always build a global fallback
        k_global = min(self.k, len(embeddings) - 1)
        self.global_model = NearestNeighbors(n_neighbors=k_global, metric="cosine")
        self.global_model.fit(embeddings)
        self.global_centroid = embeddings.mean(axis=0)

    def score(self, embeddings: np.ndarray, views: Optional[List[str]] = None) -> np.ndarray:
        embeddings = self._norm(embeddings)
        scores = np.zeros(len(embeddings))

        if views is not None and self.per_view_models:
            for i, (emb, v) in enumerate(zip(embeddings, views)):
                if v in self.per_view_models:
                    dists, _ = self.per_view_models[v]["knn"].kneighbors(emb.reshape(1, -1))
                    scores[i] = 1.0 / (1.0 + dists.mean())
                else:
                    dists, _ = self.global_model.kneighbors(emb.reshape(1, -1))
                    scores[i] = 1.0 / (1.0 + dists.mean())
        else:
            dists, _ = self.global_model.kneighbors(embeddings)
            scores = 1.0 / (1.0 + dists.mean(axis=1))

        return scores


class VLAlignmentScorer:
    """
    Signal 3: Per-view vision-language alignment.
    Each view has its own canonical good/poor text anchors.
    """

    def __init__(self, text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.text_encoder_name = text_encoder_name
        self.text_encoder = None
        self.per_view_texts: Dict = {}

    def _load_text_encoder(self):
        if self.text_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.text_encoder = SentenceTransformer(self.text_encoder_name)
            except ImportError:
                print("WARNING: sentence-transformers not available. Using random embeddings.")
                self.text_encoder = None

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        self._load_text_encoder()
        if self.text_encoder is not None:
            return self.text_encoder.encode(texts, normalize_embeddings=True)
        else:
            np.random.seed(hash(texts[0]) % 2**31)
            return np.random.randn(len(texts), 384).astype(np.float32)

    def fit(
        self,
        view_texts: Dict[str, Dict[str, List[str]]],
        video_embeddings: np.ndarray,
    ):
        """
        Precompute text embeddings for each view.

        Args:
            view_texts: {view_name: {"good": [...], "poor": [...]}}
            video_embeddings: reference embeddings for PCA alignment
        """
        for view, texts in view_texts.items():
            self.per_view_texts[view] = {
                "good": self.encode_texts(texts["good"]),
                "poor": self.encode_texts(texts["poor"]),
            }
        print(f"  [VLAlignment] Encoded texts for views: {list(self.per_view_texts.keys())}")

        # Store reference embedding dim for projection
        self._video_dim = video_embeddings.shape[1]
        self._text_dim = list(self.per_view_texts.values())[0]["good"].shape[1]

    def score(self, video_embeddings: np.ndarray, views: Optional[List[str]] = None) -> np.ndarray:
        v_norm = video_embeddings / (np.linalg.norm(video_embeddings, axis=1, keepdims=True) + 1e-8)

        # Handle dimension mismatch with PCA
        sample_good = list(self.per_view_texts.values())[0]["good"]
        if v_norm.shape[1] != sample_good.shape[1]:
            from sklearn.decomposition import PCA
            shared_dim = min(v_norm.shape[1], sample_good.shape[1], 128, len(v_norm) - 1)
            shared_dim = max(shared_dim, 2)
            pca_v = PCA(n_components=shared_dim, random_state=42)
            v_proj = pca_v.fit_transform(v_norm)
            v_proj = v_proj / (np.linalg.norm(v_proj, axis=1, keepdims=True) + 1e-8)
        else:
            v_proj = v_norm

        scores = np.zeros(len(video_embeddings))

        for i in range(len(video_embeddings)):
            view = views[i] if views is not None else None
            texts = self.per_view_texts.get(view) if view else None
            if texts is None:
                # Use first available view's texts as fallback
                texts = list(self.per_view_texts.values())[0]

            good_t = texts["good"]
            poor_t = texts["poor"]

            # Match dimensions: truncate to smaller dim for dot product
            vi = v_proj[i:i+1]
            d_v = vi.shape[1]
            d_t = good_t.shape[1]
            if d_v != d_t:
                shared = min(d_v, d_t)
                vi = vi[:, :shared]
                vi = vi / (np.linalg.norm(vi, axis=1, keepdims=True) + 1e-8)
                good_t_s = good_t[:, :shared]
                good_t_s = good_t_s / (np.linalg.norm(good_t_s, axis=1, keepdims=True) + 1e-8)
                poor_t_s = poor_t[:, :shared]
                poor_t_s = poor_t_s / (np.linalg.norm(poor_t_s, axis=1, keepdims=True) + 1e-8)
            else:
                good_t_s = good_t
                poor_t_s = poor_t

            sim_good = (vi @ good_t_s.T).mean()
            sim_poor = (vi @ poor_t_s.T).mean()
            scores[i] = sim_good - sim_poor

        return scores


class QualityScoreFusion:
    """Fuses three quality proxy signals into a composite score."""

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
        normalized = {}
        for name, s in scores.items():
            if self.normalize and s.std() > 1e-8:
                scaler = MinMaxScaler()
                s_norm = scaler.fit_transform(s.reshape(-1, 1)).flatten()
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
            N = len(list(normalized.values())[0])
            ranks = {}
            for name, s in normalized.items():
                order = np.argsort(-s)
                rank = np.empty_like(order)
                rank[order] = np.arange(N)
                ranks[name] = rank
            avg_rank = np.mean(list(ranks.values()), axis=0)
            composite = 1.0 - avg_rank / N
        else:
            raise ValueError(f"Unknown fusion: {self.method}")
        return composite


class EchoQualityScorer:
    """
    Main quality scoring class — view-aware orchestration of all 3 signals.
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
        self.fusion = QualityScoreFusion(weights=fusion_weights, method=fusion_method)

    def fit(
        self,
        reference_embeddings: np.ndarray,
        view_texts: Dict[str, Dict[str, List[str]]],
        views: Optional[List[str]] = None,
    ):
        """
        Fit scorers on reference data. No quality labels needed.

        Args:
            reference_embeddings: [N, D] embeddings from training set
            view_texts: {view: {"good": [...], "poor": [...]}}
            views: [N] view labels for per-view density modeling
        """
        print("\nFitting quality scorers...")
        self.density_scorer.fit(reference_embeddings, views=views)
        self.vl_scorer.fit(view_texts, reference_embeddings)

    def score(
        self,
        embeddings: np.ndarray,
        view_logits: Optional[torch.Tensor] = None,
        views: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        scores = {}

        # Signal 1
        if view_logits is not None:
            scores["view_confidence"] = self.view_scorer.score(view_logits)
        else:
            scores["view_confidence"] = np.ones(len(embeddings))

        # Signal 2 (per-view)
        scores["embedding_density"] = self.density_scorer.score(embeddings, views=views)

        # Signal 3 (per-view)
        scores["vl_alignment"] = self.vl_scorer.score(embeddings, views=views)

        # Fusion
        scores["composite"] = self.fusion.fuse(scores)

        return scores
