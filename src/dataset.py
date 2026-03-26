"""
Multi-view echocardiogram dataset loaders.

Supports:
  - EchoNet-Dynamic (A4C only, 10,030 videos)
  - CAMUS (A4C + A2C, 500 patients × 2 views)
  - Sample/synthetic multi-view data (for testing)

All loaders return a unified dict format:
    video: Tensor [C, T, H, W]
    filename: str
    view: str (e.g. "A4C", "A2C", "PLAX")
    view_label: int
    ef: float
    split: str
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .canonical_texts import VIEW_LABELS


# ---------------------------------------------------------------------------
# Base dataset with shared preprocessing
# ---------------------------------------------------------------------------

class BaseEchoDataset(Dataset):
    """Shared video loading and preprocessing logic."""

    def __init__(self, num_frames: int = 16, frame_size: int = 224, transform=None):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.samples: List[Dict] = []  # subclasses populate this

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, filepath: str) -> np.ndarray:
        """Load video frames from AVI/MHD file. Returns [T, H, W, C] uint8."""
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
        return np.stack(frames, axis=0)

    def _load_nifti_sequence(self, filepath: str) -> np.ndarray:
        """Load CAMUS .mhd/.nii sequence. Returns [T, H, W, C] uint8."""
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(str(filepath))
            arr = sitk.GetArrayFromImage(img)  # [T, H, W] or [H, W]
            if arr.ndim == 2:
                arr = arr[np.newaxis]  # single frame -> [1, H, W]
            # Convert grayscale to 3-channel
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = np.stack([arr, arr, arr], axis=-1)  # [T, H, W, 3]
            return arr
        except ImportError:
            raise ImportError(
                "SimpleITK required for CAMUS data. "
                "Install with: pip install SimpleITK"
            )

    def _sample_frames(self, video: np.ndarray) -> np.ndarray:
        """Uniformly sample num_frames from the video."""
        total = video.shape[0]
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(total)
            while len(indices) < self.num_frames:
                indices = np.concatenate([indices, np.arange(total)])
            indices = indices[: self.num_frames]
        return video[indices]

    def _resize_frames(self, video: np.ndarray) -> np.ndarray:
        """Resize all frames to (frame_size, frame_size)."""
        resized = []
        for frame in video:
            resized.append(cv2.resize(frame, (self.frame_size, self.frame_size)))
        return np.stack(resized, axis=0)

    def _to_tensor(self, video: np.ndarray) -> torch.Tensor:
        """Convert [T, H, W, C] uint8 -> [C, T, H, W] float [0, 1]."""
        tensor = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor


# ---------------------------------------------------------------------------
# EchoNet-Dynamic (A4C only)
# ---------------------------------------------------------------------------

class EchoNetDynamicDataset(BaseEchoDataset):
    """
    EchoNet-Dynamic: 10,030 A4C echocardiography videos.
    https://echonet.github.io/dynamic/
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
        super().__init__(num_frames, frame_size, transform)
        self.root_dir = Path(root_dir)
        self.video_dir = self.root_dir / "Videos"

        filelist = pd.read_csv(self.root_dir / "FileList.csv")
        if split is not None:
            filelist = filelist[filelist["Split"] == split.upper()]
        if max_videos is not None:
            filelist = filelist.head(max_videos)
        filelist = filelist.reset_index(drop=True)

        for _, row in filelist.iterrows():
            fname = row["FileName"]
            if not fname.endswith(".avi"):
                fname = fname + ".avi"
            self.samples.append({
                "video_path": str(self.video_dir / fname),
                "filename": row["FileName"],
                "view": "A4C",
                "view_label": VIEW_LABELS.get("A4C", 2),
                "ef": float(row.get("EF", 0.0)),
                "esv": float(row.get("ESV", 0.0)),
                "edv": float(row.get("EDV", 0.0)),
                "split": str(row.get("Split", "TRAIN")),
                "num_frames_orig": int(row.get("NumberOfFrames", 0)),
            })

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        video = self._load_video(s["video_path"])
        video = self._sample_frames(video)
        video = self._resize_frames(video)
        return {
            "video": self._to_tensor(video),
            "filename": s["filename"],
            "view": s["view"],
            "view_label": s["view_label"],
            "ef": s["ef"],
            "esv": s["esv"],
            "edv": s["edv"],
            "split": s["split"],
        }


# ---------------------------------------------------------------------------
# CAMUS (A4C + A2C)
# ---------------------------------------------------------------------------

class CAMUSDataset(BaseEchoDataset):
    """
    CAMUS: 500 patients × 2 views (A4C, A2C) × full-cycle sequences.
    https://www.creatis.insa-lyon.fr/Challenge/camus/

    Expected layout:
        data/CAMUS/
        ├── patient0001/
        │   ├── patient0001_4CH_sequence.mhd   (A4C full cycle)
        │   ├── patient0001_2CH_sequence.mhd   (A2C full cycle)
        │   ├── patient0001_4CH_ED.mhd         (end-diastole frame)
        │   └── ...
        ├── patient0002/
        └── ...
    """

    VIEW_MAP = {"4CH": "A4C", "2CH": "A2C"}

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        num_frames: int = 16,
        frame_size: int = 224,
        max_patients: Optional[int] = None,
        views: Optional[List[str]] = None,
        transform=None,
    ):
        super().__init__(num_frames, frame_size, transform)
        self.root_dir = Path(root_dir)
        views = views or ["4CH", "2CH"]

        # Find patient directories
        patient_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith("patient")
        ])

        # Simple split: 70/15/15
        if split is not None:
            n = len(patient_dirs)
            if split.upper() == "TRAIN":
                patient_dirs = patient_dirs[: int(0.7 * n)]
            elif split.upper() == "VAL":
                patient_dirs = patient_dirs[int(0.7 * n): int(0.85 * n)]
            elif split.upper() == "TEST":
                patient_dirs = patient_dirs[int(0.85 * n):]

        if max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]

        for pdir in patient_dirs:
            patient_id = pdir.name
            for view_key in views:
                seq_file = pdir / f"{patient_id}_{view_key}_sequence.mhd"
                if not seq_file.exists():
                    continue
                view_name = self.VIEW_MAP.get(view_key, view_key)
                self.samples.append({
                    "video_path": str(seq_file),
                    "filename": f"{patient_id}_{view_key}",
                    "view": view_name,
                    "view_label": VIEW_LABELS.get(view_name, -1),
                    "ef": 0.0,  # CAMUS provides LV volumes, not EF directly
                    "esv": 0.0,
                    "edv": 0.0,
                    "split": split.upper() if split else "TRAIN",
                    "loader": "nifti",
                })

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        video = self._load_nifti_sequence(s["video_path"])
        video = self._sample_frames(video)
        video = self._resize_frames(video)
        return {
            "video": self._to_tensor(video),
            "filename": s["filename"],
            "view": s["view"],
            "view_label": s["view_label"],
            "ef": s["ef"],
            "esv": s["esv"],
            "edv": s["edv"],
            "split": s["split"],
        }


# ---------------------------------------------------------------------------
# Generic multi-view dataset from FileList.csv (for synthetic/sample data)
# ---------------------------------------------------------------------------

class MultiViewEchoDataset(BaseEchoDataset):
    """
    Generic multi-view loader from a FileList.csv that includes a View column.
    Works with synthetic data from generate_sample_data.py and any custom dataset
    that follows the same schema.
    """

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        num_frames: int = 16,
        frame_size: int = 224,
        max_videos: Optional[int] = None,
        views: Optional[List[str]] = None,
        transform=None,
    ):
        super().__init__(num_frames, frame_size, transform)
        self.root_dir = Path(root_dir)
        self.video_dir = self.root_dir / "Videos"

        filelist = pd.read_csv(self.root_dir / "FileList.csv")
        if split is not None:
            filelist = filelist[filelist["Split"] == split.upper()]
        if views is not None:
            filelist = filelist[filelist["View"].isin(views)]
        if max_videos is not None:
            filelist = filelist.head(max_videos)
        filelist = filelist.reset_index(drop=True)

        for _, row in filelist.iterrows():
            fname = row["FileName"]
            if not fname.endswith(".avi"):
                fname = fname + ".avi"
            view = row.get("View", "A4C")
            self.samples.append({
                "video_path": str(self.video_dir / fname),
                "filename": row["FileName"],
                "view": view,
                "view_label": VIEW_LABELS.get(view, row.get("ViewLabel", -1)),
                "ef": float(row.get("EF", 0.0)),
                "esv": float(row.get("ESV", 0.0)),
                "edv": float(row.get("EDV", 0.0)),
                "split": str(row.get("Split", "TRAIN")),
                "quality_proxy": float(row.get("QualityProxy", -1)),
            })

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        video = self._load_video(s["video_path"])
        video = self._sample_frames(video)
        video = self._resize_frames(video)
        result = {
            "video": self._to_tensor(video),
            "filename": s["filename"],
            "view": s["view"],
            "view_label": s["view_label"],
            "ef": s["ef"],
            "split": s["split"],
        }
        if s.get("quality_proxy", -1) >= 0:
            result["quality_proxy"] = s["quality_proxy"]
        return result


# ---------------------------------------------------------------------------
# Contrastive wrapper (view-aware)
# ---------------------------------------------------------------------------

class ContrastiveEchoDataset(Dataset):
    """
    Wraps any BaseEchoDataset to return two augmented views for
    contrastive self-supervised learning.
    """

    def __init__(self, base_dataset: BaseEchoDataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _augment(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape

        if random.random() < 0.5:
            video = torch.flip(video, dims=[3])

        if random.random() < 0.8:
            ratio = random.uniform(0.75, 1.0)
            ch, cw = int(H * ratio), int(W * ratio)
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            video = video[:, :, top:top+ch, left:left+cw]
            video = torch.nn.functional.interpolate(
                video.permute(1, 0, 2, 3), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(1, 0, 2, 3)

        if random.random() < 0.7:
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            video = (video * c + (b - 1.0)).clamp(0.0, 1.0)

        if random.random() < 0.5:
            video = (video + torch.randn_like(video) * random.uniform(0.01, 0.05)).clamp(0.0, 1.0)

        return video

    def __getitem__(self, idx: int) -> Dict:
        sample = self.base_dataset[idx]
        video = sample["video"]
        return {
            "view1": self._augment(video.clone()),
            "view2": self._augment(video.clone()),
            "filename": sample["filename"],
            "view": sample["view"],
            "view_label": sample["view_label"],
            "ef": sample["ef"],
        }


# ---------------------------------------------------------------------------
# Combined multi-source loader
# ---------------------------------------------------------------------------

def create_multiview_dataset(
    sources: Dict,
    split: Optional[str] = None,
    num_frames: int = 16,
    frame_size: int = 224,
    max_videos_per_source: Optional[int] = None,
) -> ConcatDataset:
    """
    Create a combined dataset from multiple sources.

    Args:
        sources: dict mapping source name to config, e.g.:
            {
                "echonet": {"type": "echonet", "root_dir": "data/EchoNet-Dynamic"},
                "camus":   {"type": "camus",   "root_dir": "data/CAMUS"},
                "sample":  {"type": "multiview", "root_dir": "data/sample_multiview"},
            }
    Returns:
        ConcatDataset combining all sources
    """
    datasets = []

    for name, cfg in sources.items():
        dtype = cfg["type"]
        root = cfg["root_dir"]

        if dtype == "echonet":
            ds = EchoNetDynamicDataset(
                root, split=split, num_frames=num_frames,
                frame_size=frame_size, max_videos=max_videos_per_source,
            )
        elif dtype == "camus":
            ds = CAMUSDataset(
                root, split=split, num_frames=num_frames,
                frame_size=frame_size, max_patients=max_videos_per_source,
                views=cfg.get("views", ["4CH", "2CH"]),
            )
        elif dtype == "multiview":
            ds = MultiViewEchoDataset(
                root, split=split, num_frames=num_frames,
                frame_size=frame_size, max_videos=max_videos_per_source,
                views=cfg.get("views"),
            )
        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

        if len(ds) > 0:
            datasets.append(ds)
            print(f"  [{name}] Loaded {len(ds)} videos")
        else:
            print(f"  [{name}] WARNING: 0 videos found at {root}")

    if not datasets:
        raise RuntimeError("No data loaded from any source")

    return ConcatDataset(datasets)


def create_dataloaders(
    sources: Dict,
    num_frames: int = 16,
    frame_size: int = 224,
    batch_size: int = 16,
    max_videos_per_source: Optional[int] = None,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders from multiple sources."""
    loaders = {}
    for split_name in ["TRAIN", "VAL", "TEST"]:
        ds = create_multiview_dataset(
            sources, split=split_name, num_frames=num_frames,
            frame_size=frame_size, max_videos_per_source=max_videos_per_source,
        )
        loaders[split_name.lower()] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split_name == "TRAIN"),
            num_workers=num_workers, pin_memory=True, drop_last=(split_name == "TRAIN"),
        )
    return loaders
