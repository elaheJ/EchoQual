#!/usr/bin/env python3
"""
Generate synthetic sample data for EchoQual multi-view testing.

Creates synthetic echocardiogram-like AVI videos and a FileList.csv
for 5 standard views:
  1. PLAX  — Parasternal Long Axis
  2. A3C   — Apical 3-Chamber
  3. A4C   — Apical 4-Chamber
  4. Doppler_A3C — Doppler Apical 3-Chamber (Aortic Valve flow)
  5. Doppler_PLAX — Doppler Parasternal Long Axis (IVS flow)

Each video is a 112×112 grayscale clip (32 frames) with simple geometric
shapes simulating the ultrasound fan, chamber outlines, and (for Doppler)
a color-flow region. Videos vary in "quality" via noise level, contrast,
and geometric distortion to test the quality scoring pipeline.

Usage:
    python scripts/generate_sample_data.py --output_dir data/sample_multiview
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Drawing helpers — simulate echo-like frames for each view
# ---------------------------------------------------------------------------

def _draw_fan_mask(img, cx, cy, radius, angle_start, angle_end):
    """Draw a sector (ultrasound fan) mask."""
    overlay = img.copy()
    cv2.ellipse(overlay, (cx, cy), (radius, radius),
                0, angle_start, angle_end, 40, -1)
    return cv2.addWeighted(img, 0.5, overlay, 0.5, 0)


def _add_speckle(img, level=0.12):
    """Add speckle noise typical of ultrasound."""
    noise = np.random.rayleigh(level * 255, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _draw_chamber(img, center, axes, angle, color):
    """Draw an elliptical chamber."""
    cv2.ellipse(img, center, axes, angle, 0, 360, color, 1, cv2.LINE_AA)


def _draw_doppler_region(img, rect, phase):
    """Overlay a simulated color-flow Doppler region."""
    x, y, w, h = rect
    doppler = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        # Aliasing-like color pattern
        val = int(127 + 127 * np.sin(2 * np.pi * (row / h * 3 + phase)))
        doppler[row, :, 2] = val            # red channel
        doppler[row, :, 0] = 255 - val       # blue channel
    roi = img[y:y+h, x:x+w]
    blended = cv2.addWeighted(roi, 0.5, doppler, 0.5, 0)
    img[y:y+h, x:x+w] = blended
    return img


# ---------------------------------------------------------------------------
# Per-view frame generators
# ---------------------------------------------------------------------------

def generate_plax_frame(t, size=112, quality=1.0):
    """Parasternal Long Axis: LV, LA, aortic valve, longitudinal."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img = _draw_fan_mask(img, size//2, 5, size-10, 30, 150)

    # LV cavity
    lv_cx = size//2 + int(5 * np.sin(2*np.pi*t/32))  # beating
    _draw_chamber(img, (lv_cx, size//2+10), (30, 18), 10, (180, 180, 180))
    # LA
    _draw_chamber(img, (size//2+25, size//2+30), (15, 12), 0, (160, 160, 160))
    # Aortic root
    cv2.line(img, (size//2-5, 20), (size//2-5, size//2-5), (140, 140, 140), 1)
    cv2.line(img, (size//2+5, 20), (size//2+5, size//2-5), (140, 140, 140), 1)
    # IVS line
    cv2.line(img, (size//2-18, 25), (size//2-12, size//2+25), (170, 170, 170), 1)

    if quality < 0.7:
        img = _add_speckle(img, 0.25)
    else:
        img = _add_speckle(img, 0.08)
    return img


def generate_a3c_frame(t, size=112, quality=1.0):
    """Apical 3-Chamber: LA, LV, Aortic Outflow from apex."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img = _draw_fan_mask(img, size//2, size-5, size-10, 200, 340)

    beat = int(4 * np.sin(2*np.pi*t/32))
    # LV
    _draw_chamber(img, (size//2, size//2+10+beat), (22, 35), 0, (180, 180, 180))
    # LA
    _draw_chamber(img, (size//2+20, 30), (16, 12), 0, (160, 160, 160))
    # Aortic outflow
    cv2.line(img, (size//2+8, size//2-10), (size//2+15, 25), (150, 150, 150), 1)
    cv2.line(img, (size//2+18, size//2-10), (size//2+25, 25), (150, 150, 150), 1)

    img = _add_speckle(img, 0.10 if quality > 0.5 else 0.22)
    return img


def generate_a4c_frame(t, size=112, quality=1.0):
    """Apical 4-Chamber: all four chambers from apex."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img = _draw_fan_mask(img, size//2, size-5, size-10, 200, 340)

    beat = int(4 * np.sin(2*np.pi*t/32))
    # LV (left side of image = anatomical right)
    _draw_chamber(img, (size//2-15, size//2+beat), (18, 30), 0, (180, 180, 180))
    # RV
    _draw_chamber(img, (size//2+15, size//2+beat), (20, 28), 0, (170, 170, 170))
    # LA
    _draw_chamber(img, (size//2-15, 30), (14, 12), 0, (160, 160, 160))
    # RA
    _draw_chamber(img, (size//2+15, 30), (14, 12), 0, (155, 155, 155))
    # Septum
    cv2.line(img, (size//2, size-15), (size//2, 20), (170, 170, 170), 1)

    img = _add_speckle(img, 0.08 if quality > 0.5 else 0.20)
    return img


def generate_doppler_a3c_frame(t, size=112, quality=1.0):
    """Doppler Apical 3-Chamber: color flow across Aortic Valve."""
    img = generate_a3c_frame(t, size, quality)
    # Doppler overlay on aortic outflow region
    phase = t / 32.0
    img = _draw_doppler_region(img, (size//2+5, 20, 20, 35), phase)
    return img


def generate_doppler_plax_frame(t, size=112, quality=1.0):
    """Doppler PLAX: color flow across interventricular septum."""
    img = generate_plax_frame(t, size, quality)
    # Doppler overlay on IVS region
    phase = t / 32.0
    img = _draw_doppler_region(img, (size//2-22, 22, 15, 40), phase)
    return img


# ---------------------------------------------------------------------------
# Map view names to generators
# ---------------------------------------------------------------------------

VIEW_GENERATORS = {
    "PLAX":          generate_plax_frame,
    "A3C":           generate_a3c_frame,
    "A4C":           generate_a4c_frame,
    "Doppler_A3C":   generate_doppler_a3c_frame,
    "Doppler_PLAX":  generate_doppler_plax_frame,
}

VIEW_LABELS = {
    "PLAX": 0,
    "A3C": 1,
    "A4C": 2,
    "Doppler_A3C": 3,
    "Doppler_PLAX": 4,
}

# ---------------------------------------------------------------------------
# Video + CSV generation
# ---------------------------------------------------------------------------

def generate_video(filepath, view, num_frames=32, size=112, quality=1.0):
    """Write a synthetic AVI video for a given view."""
    gen = VIEW_GENERATORS[view]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(filepath), fourcc, 30.0, (size, size))

    for t in range(num_frames):
        frame = gen(t, size, quality)
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Generate multi-view sample data")
    parser.add_argument("--output_dir", type=str, default="data/sample_multiview",
                        help="Root directory for sample dataset")
    parser.add_argument("--videos_per_view", type=int, default=10,
                        help="Number of videos per view")
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Frames per video")
    parser.add_argument("--frame_size", type=int, default=112,
                        help="Frame height/width")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output = Path(args.output_dir)
    video_dir = output / "Videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    vid_id = 0

    for view in VIEW_GENERATORS:
        for i in range(args.videos_per_view):
            vid_id += 1
            # Quality varies: ~60% good, ~20% medium, ~20% poor
            r = random.random()
            if r < 0.6:
                quality = random.uniform(0.75, 1.0)
                quality_label = "good"
            elif r < 0.8:
                quality = random.uniform(0.45, 0.74)
                quality_label = "medium"
            else:
                quality = random.uniform(0.1, 0.44)
                quality_label = "poor"

            filename = f"{view}_{vid_id:04d}"
            avi_path = video_dir / f"{filename}.avi"

            generate_video(avi_path, view,
                           num_frames=args.num_frames,
                           size=args.frame_size,
                           quality=quality)

            # Synthetic clinical metadata
            ef = round(random.uniform(25, 70), 1)
            esv = round(random.uniform(30, 120), 1)
            edv = round(random.uniform(80, 200), 1)

            rows.append({
                "FileName": filename,
                "EF": ef,
                "ESV": esv,
                "EDV": edv,
                "NumberOfFrames": args.num_frames,
                "Split": random.choices(["TRAIN", "VAL", "TEST"],
                                        weights=[0.7, 0.15, 0.15])[0],
                "View": view,
                "ViewLabel": VIEW_LABELS[view],
                "QualityProxy": round(quality, 3),
                "QualityBin": quality_label,
            })

    df = pd.DataFrame(rows)
    csv_path = output / "FileList.csv"
    df.to_csv(csv_path, index=False)

    print(f"Created {len(rows)} sample videos across {len(VIEW_GENERATORS)} views")
    print(f"  Videos: {video_dir}")
    print(f"  FileList: {csv_path}")
    print(f"\nView distribution:")
    print(df["View"].value_counts().to_string())
    print(f"\nQuality distribution:")
    print(df["QualityBin"].value_counts().to_string())
    print(f"\nSplit distribution:")
    print(df["Split"].value_counts().to_string())
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
