#!/usr/bin/env python3
"""
EchoQual: End-to-end pipeline for self-supervised echo quality assessment.

Usage:
    python scripts/run_pipeline.py --data_dir data/EchoNet-Dynamic --output_dir outputs --max_videos 500
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import EchoNetDynamicDataset, ContrastiveEchoDataset
from src.encoder import EchoQualModel
from src.ssl_pretraining import SSLPretrainer, ContrastiveProjectionHead
from src.quality_scorer import EchoQualityScorer
from src.canonical_texts import get_canonical_texts
from src.evaluation import (
    compute_ef_proxy_quality,
    evaluate_quality_scores,
    visualize_results,
)
from src.utils import load_config, set_seed, get_device, ensure_dir, format_metrics


def step1_ssl_pretrain(
    model, train_dataset, config, device, output_dir
):
    """Step 1: Self-supervised contrastive pretraining."""
    print("\n" + "=" * 60)
    print("STEP 1: Self-Supervised Contrastive Pretraining")
    print("=" * 60)

    ssl_cfg = config["ssl_pretraining"]

    if not ssl_cfg["enabled"]:
        print("SSL pretraining disabled. Using pretrained weights only.")
        return

    contrastive_dataset = ContrastiveEchoDataset(train_dataset)
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=ssl_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    projection_head = ContrastiveProjectionHead(
        input_dim=config["encoder"]["embedding_dim"],
        output_dim=ssl_cfg["projection_dim"],
    )

    pretrainer = SSLPretrainer(
        encoder=model.encoder,
        projection_head=projection_head,
        device=device,
        learning_rate=ssl_cfg["learning_rate"],
        weight_decay=ssl_cfg["weight_decay"],
        temperature=ssl_cfg["temperature"],
        num_frames=config["data"]["num_frames"],
    )

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    history = pretrainer.train(
        contrastive_loader,
        num_epochs=ssl_cfg["epochs"],
        checkpoint_dir=checkpoint_dir,
    )

    # Save final encoder
    encoder_path = os.path.join(output_dir, "checkpoints", "encoder_final.pt")
    pretrainer.save_encoder(encoder_path)
    print(f"Encoder saved to {encoder_path}")

    return history


def step2_extract_embeddings(
    model, dataset, device, batch_size=16
):
    """Step 2: Extract embeddings for all videos."""
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Video Embeddings")
    print("=" * 60)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    all_embeddings = []
    all_view_logits = []
    all_filenames = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            videos = batch["video"].to(device)
            outputs = model(videos)

            all_embeddings.append(outputs["embedding"].cpu().numpy())
            all_view_logits.append(outputs["view_logits"].cpu())
            all_filenames.extend(batch["filename"])

    embeddings = np.concatenate(all_embeddings, axis=0)
    view_logits = torch.cat(all_view_logits, dim=0)

    print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    return embeddings, view_logits, all_filenames


def step3_compute_quality_scores(
    embeddings, view_logits, config
):
    """Step 3: Compute self-supervised quality scores."""
    print("\n" + "=" * 60)
    print("STEP 3: Computing Self-Supervised Quality Scores")
    print("=" * 60)

    qs_cfg = config["quality_scoring"]

    scorer = EchoQualityScorer(
        view_conf_method=qs_cfg["view_confidence"]["method"],
        density_method=qs_cfg["embedding_density"]["method"],
        density_k=qs_cfg["embedding_density"]["k"],
        fusion_weights={
            "view_confidence": qs_cfg["view_confidence"]["weight"],
            "embedding_density": qs_cfg["embedding_density"]["weight"],
            "vl_alignment": qs_cfg["vl_alignment"]["weight"],
        },
        fusion_method=qs_cfg["fusion"]["method"],
    )

    # Get canonical texts
    texts = get_canonical_texts("A4C", include_poor=True)

    # Fit on all embeddings (unsupervised — no quality labels used)
    scorer.fit(
        reference_embeddings=embeddings,
        good_texts=texts["good"],
        poor_texts=texts["poor"],
    )

    # Score all videos
    scores = scorer.score(embeddings, view_logits)

    print(f"Composite score range: [{scores['composite'].min():.4f}, "
          f"{scores['composite'].max():.4f}]")

    return scores, scorer


def step4_evaluate(
    scores, filenames, data_dir, output_dir, config
):
    """Step 4: Evaluate against proxy ground truth."""
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)

    filelist = pd.read_csv(os.path.join(data_dir, "FileList.csv"))

    # Match filenames to filelist rows
    filelist_subset = filelist[filelist["FileName"].isin(filenames)].copy()
    filelist_subset = filelist_subset.set_index("FileName").loc[filenames].reset_index()

    # Compute proxy quality labels (no expert labels!)
    proxy_quality = compute_ef_proxy_quality(filelist_subset)

    # Evaluate
    metrics = evaluate_quality_scores(scores, proxy_quality)

    print("\n--- Evaluation Metrics ---")
    print(format_metrics(metrics))

    # Visualize
    figures_dir = os.path.join(output_dir, "figures")
    ranked_df = visualize_results(
        scores, proxy_quality, filenames, figures_dir,
        num_examples=config["evaluation"].get("num_examples", 20),
    )

    # Save scores
    scores_dir = ensure_dir(os.path.join(output_dir, "scores"))
    scores_df = pd.DataFrame({
        "filename": filenames,
        "composite_score": scores["composite"],
        "view_confidence": scores["view_confidence"],
        "embedding_density": scores["embedding_density"],
        "vl_alignment": scores["vl_alignment"],
        "proxy_quality": proxy_quality,
    })
    scores_df.to_csv(os.path.join(scores_dir, "quality_scores.csv"), index=False)
    print(f"\nScores saved to {scores_dir}/quality_scores.csv")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(scores_dir, "metrics.csv"), index=False)

    return metrics, ranked_df


def main():
    parser = argparse.ArgumentParser(description="EchoQual: Self-supervised quality assessment")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to EchoNet-Dynamic dataset")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum videos to process (overrides config)")
    parser.add_argument("--skip_ssl", action="store_true",
                        help="Skip SSL pretraining")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    config = load_config(args.config)
    if args.max_videos:
        config["data"]["max_videos"] = args.max_videos
    if args.skip_ssl:
        config["ssl_pretraining"]["enabled"] = False

    ensure_dir(args.output_dir)

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    train_dataset = EchoNetDynamicDataset(
        root_dir=args.data_dir,
        split="TRAIN",
        num_frames=config["data"]["num_frames"],
        frame_size=config["data"]["frame_size"],
        max_videos=config["data"]["max_videos"],
    )
    print(f"Training set: {len(train_dataset)} videos")

    # Full dataset for scoring (train + val + test)
    full_dataset = EchoNetDynamicDataset(
        root_dir=args.data_dir,
        split=None,  # All splits
        num_frames=config["data"]["num_frames"],
        frame_size=config["data"]["frame_size"],
        max_videos=config["data"]["max_videos"],
    )
    print(f"Full dataset: {len(full_dataset)} videos")

    # Initialize model
    model = EchoQualModel(
        backbone=config["encoder"]["backbone"],
        pretrained=config["encoder"]["pretrained_imagenet"],
        embedding_dim=config["encoder"]["embedding_dim"],
    ).to(device)

    # Step 1: SSL Pretraining
    step1_ssl_pretrain(model, train_dataset, config, device, args.output_dir)

    # Step 2: Extract embeddings
    embeddings, view_logits, filenames = step2_extract_embeddings(
        model, full_dataset, device
    )

    # Save embeddings
    emb_dir = ensure_dir(os.path.join(args.output_dir, "embeddings"))
    np.save(os.path.join(emb_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(emb_dir, "filenames.npy"), filenames)
    torch.save(view_logits, os.path.join(emb_dir, "view_logits.pt"))

    # Step 3: Compute quality scores
    scores, scorer = step3_compute_quality_scores(embeddings, view_logits, config)

    # Step 4: Evaluate
    metrics, ranked_df = step4_evaluate(
        scores, filenames, args.data_dir, args.output_dir, config
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
