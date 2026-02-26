"""
Video encoder backbone with projection heads for:
  1. Contrastive SSL embedding space
  2. View classification head
  3. Vision-language alignment projection

Supports R3D-18, R(2+1)D-18, and MC3-18 backbones from torchvision.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


class VideoEncoder(nn.Module):
    """
    3D CNN video encoder that produces a fixed-size embedding from
    echocardiogram video clips [B, C, T, H, W] -> [B, embedding_dim].
    """

    def __init__(
        self,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load backbone
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

        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] video tensor
        Returns:
            embeddings: [B, embedding_dim]
        """
        features = self.backbone(x)  # [B, backbone_dim]
        embeddings = self.embed_proj(features)  # [B, embedding_dim]
        return embeddings

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw backbone features before projection."""
        return self.backbone(x)


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
    Head for echocardiographic view classification.

    For EchoNet-Dynamic (single A4C view), this acts as a "view quality"
    discriminator: it learns what a canonical A4C view looks like.
    High confidence = well-acquired; low confidence = poor quality.

    For multi-view datasets, this generalizes to N-way view classification.
    """

    def __init__(self, input_dim: int = 512, num_views: int = 1, hidden_dim: int = 256):
        super().__init__()
        # For single-view (A4C), we use a binary classifier:
        # "is this a proper A4C view?" treated as 2-class
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
    Projects video embeddings into a joint vision-language space
    for computing cosine similarity with text embeddings.
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

    def forward(
        self, video_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> tuple:
        """
        Args:
            video_emb: [B, video_dim]
            text_emb: [N_texts, text_dim]
        Returns:
            video_proj: [B, projection_dim] L2-normalized
            text_proj: [N, projection_dim] L2-normalized
            similarity: [B, N] scaled cosine similarity
        """
        v = self.video_proj(video_emb)
        t = self.text_proj(text_emb)

        v = nn.functional.normalize(v, dim=-1)
        t = nn.functional.normalize(t, dim=-1)

        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * v @ t.T

        return v, t, similarity


class EchoQualModel(nn.Module):
    """
    Full model combining encoder + all heads for quality assessment.
    """

    def __init__(
        self,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        num_views: int = 1,
        text_encoder_dim: int = 384,
        vl_projection_dim: int = 256,
    ):
        super().__init__()
        self.encoder = VideoEncoder(backbone, pretrained, embedding_dim)
        self.contrastive_head = ContrastiveProjectionHead(
            embedding_dim, embedding_dim // 2, projection_dim
        )
        self.view_head = ViewClassificationHead(embedding_dim, num_views)
        self.vl_head = VLAlignmentProjection(
            embedding_dim, text_encoder_dim, vl_projection_dim
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get video embeddings."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass returning all outputs."""
        emb = self.encoder(x)
        return {
            "embedding": emb,
            "contrastive_proj": self.contrastive_head(emb),
            "view_logits": self.view_head(emb),
        }
