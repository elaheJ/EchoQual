"""
EchoNet-Dynamic dataset loader with SSL-friendly augmentations.

Loads AVI echocardiogram videos, samples fixed-length clips, and applies
spatial/temporal augmentations for contrastive pretraining.
"""

import os
import math
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EchoNetDynamicDataset(Dataset):
    """
    Dataset for EchoNet-Dynamic echocardiogram videos.

    Each item returns a dict with:
        - video: Tensor [C, T, H, W] normalized to [0, 1]
        - filename: str
        - ef: float (ejection fraction)
        - split: str (TRAIN/VAL/TEST)
    """

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        num_frames: int = 16,
        frame_size: int = 224,
        max_videos: Optional[int] = None,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.video_dir = self.root_dir / "Videos"
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

        # Load file list
        filelist_path = self.root_dir / "FileList.csv"
        self.filelist = pd.read_csv(filelist_path)

        # Filter by split if specified
        if split is not None:
            self.filelist = self.filelist[self.filelist["Split"] == split.upper()]

        # Subset for rapid prototyping
        if max_videos is not None:
            self.filelist = self.filelist.head(max_videos)

        self.filelist = self.filelist.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.filelist)

    def _load_video(self, filepath: str) -> np.ndarray:
        """Load video frames from AVI file. Returns [T, H, W, C] uint8 array."""
        cap = cv2.VideoCapture(str(filepath))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not read video: {filepath}")

        return np.stack(frames, axis=0)  # [T, H, W, C]

    def _sample_frames(self, video: np.ndarray) -> np.ndarray:
        """Uniformly sample num_frames from the video."""
        total_frames = video.shape[0]

        if total_frames >= self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Repeat frames if video is too short
            indices = np.arange(total_frames)
            while len(indices) < self.num_frames:
                indices = np.concatenate([indices, np.arange(total_frames)])
            indices = indices[: self.num_frames]

        return video[indices]

    def _resize_frames(self, video: np.ndarray) -> np.ndarray:
        """Resize all frames to (frame_size, frame_size)."""
        resized = []
        for frame in video:
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            resized.append(frame)
        return np.stack(resized, axis=0)

    def __getitem__(self, idx: int) -> Dict:
        row = self.filelist.iloc[idx]
        filename = row["FileName"]

        # Handle .avi extension
        if not filename.endswith(".avi"):
            filename = filename + ".avi"

        video_path = self.video_dir / filename

        # Load and preprocess video
        video = self._load_video(video_path)
        video = self._sample_frames(video)
        video = self._resize_frames(video)

        # Convert to tensor: [T, H, W, C] -> [C, T, H, W], float [0, 1]
        video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0

        # Apply optional transforms
        if self.transform is not None:
            video_tensor = self.transform(video_tensor)

        return {
            "video": video_tensor,
            "filename": row["FileName"],
            "ef": float(row.get("EF", 0.0)),
            "esv": float(row.get("ESV", 0.0)),
            "edv": float(row.get("EDV", 0.0)),
            "split": str(row.get("Split", "TRAIN")),
            "num_frames_orig": int(row.get("NumberOfFrames", video.shape[0])),
        }


class ContrastiveEchoDataset(Dataset):
    """
    Wraps EchoNetDynamicDataset to return two augmented views of the same video
    for contrastive self-supervised learning.
    """

    def __init__(
        self,
        base_dataset: EchoNetDynamicDataset,
        augmentation_strength: float = 0.5,
    ):
        self.base_dataset = base_dataset
        self.augmentation_strength = augmentation_strength

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _augment_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial and temporal augmentations to a video tensor [C, T, H, W].
        These are echo-specific augmentations designed for ultrasound data.
        """
        C, T, H, W = video.shape

        # 1. Random horizontal flip (anatomical flip)
        if random.random() < 0.5:
            video = torch.flip(video, dims=[3])

        # 2. Random crop and resize
        if random.random() < 0.8:
            crop_ratio = random.uniform(0.75, 1.0)
            crop_h = int(H * crop_ratio)
            crop_w = int(W * crop_ratio)
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            video = video[:, :, top : top + crop_h, left : left + crop_w]
            video = torch.nn.functional.interpolate(
                video.permute(1, 0, 2, 3),  # [T, C, H, W]
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)  # back to [C, T, H, W]

        # 3. Brightness/contrast jitter (ultrasound-appropriate)
        if random.random() < 0.7:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            video = video * contrast + (brightness - 1.0)
            video = video.clamp(0.0, 1.0)

        # 4. Gaussian noise (simulates speckle)
        if random.random() < 0.5:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(video) * noise_std
            video = (video + noise).clamp(0.0, 1.0)

        # 5. Temporal jitter: randomly drop and repeat frames
        if random.random() < 0.3:
            drop_idx = random.randint(0, T - 1)
            replace_idx = random.randint(0, T - 1)
            video[:, drop_idx] = video[:, replace_idx]

        # 6. Temporal reversal
        if random.random() < 0.2:
            video = torch.flip(video, dims=[1])

        return video

    def __getitem__(self, idx: int) -> Dict:
        sample = self.base_dataset[idx]
        video = sample["video"]

        view1 = self._augment_video(video.clone())
        view2 = self._augment_video(video.clone())

        return {
            "view1": view1,
            "view2": view2,
            "filename": sample["filename"],
            "ef": sample["ef"],
        }


def create_dataloaders(
    root_dir: str,
    num_frames: int = 16,
    frame_size: int = 224,
    batch_size: int = 16,
    max_videos: Optional[int] = None,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders."""
    loaders = {}
    for split in ["TRAIN", "VAL", "TEST"]:
        ds = EchoNetDynamicDataset(
            root_dir=root_dir,
            split=split,
            num_frames=num_frames,
            frame_size=frame_size,
            max_videos=max_videos if split == "TRAIN" else min(max_videos or 9999, 200),
        )
        loaders[split.lower()] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "TRAIN"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "TRAIN"),
        )
    return loaders
