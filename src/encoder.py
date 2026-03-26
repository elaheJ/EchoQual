"""
Video encoder with support for:
  1. EchoPrime pretrained backbone (MViT, 512-dim, multi-view aware)
  2. Torchvision R3D/R(2+1)D fallback (for training from scratch)

Plus projection heads for contrastive SSL, view classification, and VL alignment.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models.video as video_models


# ---------------------------------------------------------------------------
# EchoPrime backbone wrapper
# ---------------------------------------------------------------------------

class EchoPrimeEncoder(nn.Module):
    """
    Wrapper around EchoPrime's pretrained MViT video encoder.

    Loads the encoder weights from the EchoPrime release. The encoder
    expects [B, 3, 16, 224, 224] input and produces [B, 512] embeddings.

    Setup:
        1. git clone https://github.com/echonet/EchoPrime
        2. Download model_data.zip from the release
        3. Point echoprime_dir to the cloned repo root
    """

    def __init__(
        self,
        echoprime_dir: str = "EchoPrime",
        freeze: bool = False,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._freeze = freeze

        # Import EchoPrime's model
        import sys
        sys.path.insert(0, str(Path(echoprime_dir).resolve()))

        try:
            from echo_prime import EchoPrime
            ep = EchoPrime(model_data_dir=os.path.join(echoprime_dir, "model_data"))
            self.backbone = ep.video_encoder
            self.backbone.eval()

            # EchoPrime's video encoder outputs 512-dim
            self._backbone_dim = 512

            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                print("[EchoPrimeEncoder] Backbone frozen (feature extraction mode)")
            else:
                # Unfreeze last 2 blocks for fine-tuning (per OpenReview findings)
                for p in self.backbone.parameters():
                    p.requires_grad = False
                # Unfreeze last 2 transformer blocks + head
                unfrozen = 0
                for name, p in self.backbone.named_parameters():
                    if any(k in name for k in ["blocks.15", "blocks.14", "head", "norm"]):
                        p.requires_grad = True
                        unfrozen += 1
                print(f"[EchoPrimeEncoder] Unfroze {unfrozen} params in last 2 blocks")

            print(f"[EchoPrimeEncoder] Loaded successfully from {echoprime_dir}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load EchoPrime encoder from {echoprime_dir}. "
                f"Error: {e}\n"
                f"Make sure you've cloned EchoPrime and downloaded model_data.zip. "
                f"See: https://github.com/echonet/EchoPrime"
            )

        # Optional projection if output dim differs
        if embedding_dim != self._backbone_dim:
            self.proj = nn.Sequential(
                nn.Linear(self._backbone_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, T, H, W] video tensor (T=16, H=W=224)
        Returns:
            embeddings: [B, embedding_dim]
        """
        features = self.backbone(x)
        return self.proj(features)

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Standard torchvision backbone (R3D / R(2+1)D)
# ---------------------------------------------------------------------------

class VideoEncoder(nn.Module):
    """
    3D CNN video encoder: [B, C, T, H, W] -> [B, embedding_dim].
    Supports r3d_18, r2plus1d_18, mc3_18 from torchvision.
    """

    def __init__(
        self,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        if backbone == "r3d_18":
            weights = "DEFAULT" if pretrained else None
            self.backbone = video_models.r3d_18(weights=weights)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "r2plus1d_18":
            weights = "DEFAULT" if pretrained else None
            self.backbone = video_models.r2plus1d_18(weights=weights)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "mc3_18":
            weights = "DEFAULT" if pretrained else None
            self.backbone = video_models.mc3_18(weights=weights)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.embed_proj = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.embed_proj(features)

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Projection heads
# ---------------------------------------------------------------------------

class ContrastiveProjectionHead(nn.Module):
    """MLP projection head for SimCLR-style contrastive learning."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViewClassificationHead(nn.Module):
    """
    N-way view classifier. For multi-view datasets this classifies
    which standard echo view a video belongs to. Confidence serves
    as quality proxy: canonical views -> high confidence -> good quality.
    """

    def __init__(self, input_dim: int = 512, num_views: int = 5, hidden_dim: int = 256):
        super().__init__()
        output_dim = max(num_views, 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VLAlignmentProjection(nn.Module):
    """
    Projects video embeddings into a joint vision-language space.
    Designed to align with EchoPrime's 512-dim text encoder or
    sentence-transformers (384-dim).
    """

    def __init__(self, video_dim: int = 512, text_dim: int = 384, projection_dim: int = 256):
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, video_emb: torch.Tensor, text_emb: torch.Tensor) -> tuple:
        v = nn.functional.normalize(self.video_proj(video_emb), dim=-1)
        t = nn.functional.normalize(self.text_proj(text_emb), dim=-1)
        similarity = self.logit_scale.exp() * v @ t.T
        return v, t, similarity


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class EchoQualModel(nn.Module):
    """
    Full model combining encoder + all heads.

    Supports two modes:
      1. backbone="echoprime" -> load EchoPrime pretrained MViT
      2. backbone="r3d_18" etc -> train from scratch with torchvision
    """

    def __init__(
        self,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        num_views: int = 5,
        text_encoder_dim: int = 384,
        vl_projection_dim: int = 256,
        echoprime_dir: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if backbone == "echoprime":
            if echoprime_dir is None:
                echoprime_dir = "EchoPrime"
            self.encoder = EchoPrimeEncoder(
                echoprime_dir=echoprime_dir,
                freeze=freeze_backbone,
                embedding_dim=embedding_dim,
            )
        else:
            self.encoder = VideoEncoder(backbone, pretrained, embedding_dim)

        self.contrastive_head = ContrastiveProjectionHead(
            embedding_dim, embedding_dim // 2, projection_dim
        )
        self.view_head = ViewClassificationHead(embedding_dim, num_views)
        self.vl_head = VLAlignmentProjection(
            embedding_dim, text_encoder_dim, vl_projection_dim
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict:
        emb = self.encoder(x)
        return {
            "embedding": emb,
            "contrastive_proj": self.contrastive_head(emb),
            "view_logits": self.view_head(emb),
        }
