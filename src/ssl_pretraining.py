"""
Self-supervised contrastive pretraining for echocardiogram video encoder.

Implements:
  1. SimCLR-style contrastive learning with echo-specific augmentations
  2. Temporal frame reordering pretext task (inspired by EchoCLR)
  3. Joint training objective combining both signals

The trained encoder produces embeddings used by the quality scoring module.
"""

import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .encoder import VideoEncoder, ContrastiveProjectionHead


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: [B, D] L2-normalized projection vectors from two views
        Returns:
            loss: scalar
        """
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat(
            [torch.arange(B, 2 * B), torch.arange(0, B)], dim=0
        ).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


class FrameReorderingHead(nn.Module):
    """
    Pretext task: predict the correct temporal order of shuffled frames.
    Encourages the encoder to learn temporal/cardiac cycle dynamics.
    """

    def __init__(self, embedding_dim: int = 512, num_frames: int = 16):
        super().__init__()
        self.num_frames = num_frames
        # Predict a permutation score for each frame position
        self.order_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_frames),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict ordering scores from video embedding."""
        return self.order_predictor(embedding)


class SSLPretrainer:
    """
    Orchestrates self-supervised pretraining of the video encoder.

    Training loop:
      1. Sample two augmented views of each video
      2. Encode both views -> embeddings -> projections
      3. Compute NT-Xent contrastive loss
      4. Optionally compute frame reordering loss
      5. Backpropagate combined loss
    """

    def __init__(
        self,
        encoder: VideoEncoder,
        projection_head: ContrastiveProjectionHead,
        device: torch.device,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        reorder_weight: float = 0.5,
        num_frames: int = 16,
    ):
        self.device = device
        self.encoder = encoder.to(device)
        self.projection_head = projection_head.to(device)

        # Frame reordering head
        self.reorder_head = FrameReorderingHead(
            encoder.embedding_dim, num_frames
        ).to(device)

        # Loss
        self.contrastive_loss = NTXentLoss(temperature)
        self.reorder_weight = reorder_weight

        # Optimizer
        params = (
            list(encoder.parameters())
            + list(projection_head.parameters())
            + list(self.reorder_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )

    def _shuffle_frames(
        self, video: torch.Tensor
    ) -> tuple:
        """
        Shuffle temporal frames of video and return shuffled video + true order.
        Args:
            video: [B, C, T, H, W]
        Returns:
            shuffled: [B, C, T, H, W]
            order: [B, T] ground truth frame indices
        """
        B, C, T, H, W = video.shape
        orders = []
        shuffled_videos = []

        for i in range(B):
            perm = torch.randperm(T)
            orders.append(perm)
            shuffled_videos.append(video[i, :, perm, :, :])

        shuffled = torch.stack(shuffled_videos, dim=0)
        order = torch.stack(orders, dim=0).to(video.device)

        return shuffled, order

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.projection_head.train()
        self.reorder_head.train()

        total_loss = 0.0
        total_contrastive = 0.0
        total_reorder = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"SSL Epoch {epoch}")
        for batch in pbar:
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)

            # Encode both views
            emb1 = self.encoder(view1)
            emb2 = self.encoder(view2)

            # Project for contrastive loss
            z1 = F.normalize(self.projection_head(emb1), dim=-1)
            z2 = F.normalize(self.projection_head(emb2), dim=-1)

            # Contrastive loss
            loss_contrast = self.contrastive_loss(z1, z2)

            # Frame reordering pretext task
            shuffled, true_order = self._shuffle_frames(view1)
            emb_shuffled = self.encoder(shuffled)
            order_pred = self.reorder_head(emb_shuffled)
            loss_reorder = F.cross_entropy(
                order_pred, true_order[:, :order_pred.shape[1]].argmax(dim=-1)
                if true_order.dim() > 1
                else true_order[:, 0]
            )

            # Combined loss
            loss = loss_contrast + self.reorder_weight * loss_reorder

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_contrastive += loss_contrast.item()
            total_reorder += loss_reorder.item()
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                contrast=f"{loss_contrast.item():.4f}",
                reorder=f"{loss_reorder.item():.4f}",
            )

        self.scheduler.step()

        return {
            "total_loss": total_loss / max(num_batches, 1),
            "contrastive_loss": total_contrastive / max(num_batches, 1),
            "reorder_loss": total_reorder / max(num_batches, 1),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
    ) -> list:
        """Full pretraining loop."""
        history = []

        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(dataloader, epoch)
            history.append(metrics)

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"Contrastive: {metrics['contrastive_loss']:.4f} | "
                f"Reorder: {metrics['reorder_loss']:.4f}"
            )

            # Save checkpoint
            if checkpoint_dir and epoch % 10 == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "projection_head_state_dict": self.projection_head.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    Path(checkpoint_dir) / f"ssl_checkpoint_epoch{epoch}.pt",
                )

        return history

    def save_encoder(self, path: str):
        """Save just the encoder weights for downstream use."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
