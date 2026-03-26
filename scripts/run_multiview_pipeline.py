#!/usr/bin/env python3
"""
EchoQual Multi-View Pipeline: End-to-end self-supervised quality assessment
across multiple echocardiographic views.

Supports:
  - EchoNet-Dynamic (A4C) + CAMUS (A4C, A2C) for real 2-view proof-of-concept
  - Synthetic sample data for pipeline validation
  - EchoPrime pretrained encoder or train-from-scratch R3D backbone

Usage:
    # With synthetic sample data (pipeline test):
    python scripts/run_multiview_pipeline.py \\
        --config configs/multiview.yaml \\
        --data_sources sample=data/sample_multiview \\
        --skip_ssl --max_videos 50

    # With EchoNet-Dynamic only:
    python scripts/run_multiview_pipeline.py \\
        --config configs/multiview.yaml \\
        --data_sources echonet=data/EchoNet-Dynamic \\
        --max_videos 500

    # With EchoNet-Dynamic + CAMUS (2-view real data):
    python scripts/run_multiview_pipeline.py \\
        --config configs/multiview.yaml \\
        --data_sources echonet=data/EchoNet-Dynamic camus=data/CAMUS \\
        --max_videos 500

    # With EchoPrime backbone:
    python scripts/run_multiview_pipeline.py \\
        --config configs/multiview.yaml \\
        --data_sources echonet=data/EchoNet-Dynamic \\
        --backbone echoprime --echoprime_dir ./EchoPrime \\
        --skip_ssl --max_videos 500
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    EchoNetDynamicDataset,
    CAMUSDataset,
    MultiViewEchoDataset,
    ContrastiveEchoDataset,
    create_multiview_dataset,
)
from src.encoder import EchoQualModel, ContrastiveProjectionHead
from src.ssl_pretraining import SSLPretrainer
from src.quality_scorer import EchoQualityScorer
from src.canonical_texts import get_canonical_texts, get_all_view_texts, VIEW_LABELS, NUM_VIEWS
from src.evaluation import (
    compute_ef_proxy_quality,
    evaluate_quality_scores,
    visualize_results,
)
from src.utils import load_config, set_seed, get_device, ensure_dir, format_metrics


def parse_data_sources(source_args: list, config: dict) -> dict:
    """
    Parse --data_sources arguments like 'echonet=data/EchoNet-Dynamic camus=data/CAMUS'
    into a sources dict, merging with config defaults.
    """
    sources = {}
    config_sources = config.get("data", {}).get("sources", {})

    if source_args:
        for arg in source_args:
            name, path = arg.split("=", 1)
            # Get type from config or infer from name
            cfg = config_sources.get(name, {})
            dtype = cfg.get("type", name)
            sources[name] = {
                "type": dtype,
                "root_dir": path,
                "views": cfg.get("views"),
            }
    else:
        # Use config sources that are enabled
        for name, cfg in config_sources.items():
            if cfg.get("enabled", False):
                sources[name] = cfg

    return sources


def step1_ssl_pretrain(model, dataset, config, device, output_dir):
    """Step 1: Self-supervised contrastive pretraining."""
    print("\n" + "=" * 60)
    print("STEP 1: Self-Supervised Contrastive Pretraining")
    print("=" * 60)

    ssl_cfg = config["ssl_pretraining"]
    if not ssl_cfg["enabled"]:
        print("SSL pretraining disabled.")
        return

    contrastive_ds = ContrastiveEchoDataset(dataset)
    loader = DataLoader(
        contrastive_ds,
        batch_size=ssl_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    if len(loader) == 0:
        print("WARNING: Empty dataloader, skipping SSL.")
        return

    proj_head = ContrastiveProjectionHead(
        input_dim=config["encoder"]["embedding_dim"],
        output_dim=ssl_cfg["projection_dim"],
    )

    pretrainer = SSLPretrainer(
        encoder=model.encoder,
        projection_head=proj_head,
        device=device,
        learning_rate=ssl_cfg["learning_rate"],
        weight_decay=ssl_cfg["weight_decay"],
        temperature=ssl_cfg["temperature"],
        num_frames=config["data"]["num_frames"],
    )

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    history = pretrainer.train(
        loader,
        num_epochs=ssl_cfg["epochs"],
        checkpoint_dir=ckpt_dir,
    )
    pretrainer.save_encoder(os.path.join(ckpt_dir, "encoder_final.pt"))
    return history


def step2_extract_embeddings(model, dataset, device, batch_size=8):
    """Step 2: Extract embeddings for all videos."""
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Video Embeddings")
    print("=" * 60)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model.eval()
    all_emb, all_logits, all_fnames, all_views = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding"):
            videos = batch["video"].to(device)
            out = model(videos)
            all_emb.append(out["embedding"].cpu().numpy())
            all_logits.append(out["view_logits"].cpu())
            all_fnames.extend(batch["filename"])
            all_views.extend(batch["view"])

    embeddings = np.concatenate(all_emb, axis=0)
    view_logits = torch.cat(all_logits, dim=0)

    print(f"Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    view_counts = pd.Series(all_views).value_counts()
    print(f"View distribution:\n{view_counts.to_string()}")

    return embeddings, view_logits, all_fnames, all_views


def step3_compute_quality_scores(embeddings, view_logits, views, config):
    """Step 3: Compute self-supervised quality scores (per-view aware)."""
    print("\n" + "=" * 60)
    print("STEP 3: Computing View-Aware Quality Scores")
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

    # Get per-view canonical texts
    unique_views = sorted(set(views))
    view_texts = {}
    for v in unique_views:
        texts = get_canonical_texts(v, include_poor=True)
        view_texts[v] = texts
    print(f"Canonical texts loaded for views: {unique_views}")

    # Fit (unsupervised — no quality labels)
    scorer.fit(
        reference_embeddings=embeddings,
        view_texts=view_texts,
        views=views,
    )

    # Score
    scores = scorer.score(embeddings, view_logits, views=views)

    print(f"Composite score range: [{scores['composite'].min():.4f}, "
          f"{scores['composite'].max():.4f}]")

    return scores, scorer


def step4_evaluate(scores, filenames, views, sources, output_dir, config):
    """Step 4: Evaluate against proxy ground truth."""
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation (Global + Per-View)")
    print("=" * 60)

    # Try to load proxy quality from data sources
    proxy_quality = np.zeros(len(filenames))

    for name, cfg in sources.items():
        csv_path = os.path.join(cfg["root_dir"], "FileList.csv")
        if os.path.exists(csv_path):
            fl = pd.read_csv(csv_path)

            # Check for explicit QualityProxy column
            if "QualityProxy" in fl.columns:
                fname_to_q = dict(zip(fl["FileName"].astype(str), fl["QualityProxy"].astype(float)))
                for i, fn in enumerate(filenames):
                    if fn in fname_to_q:
                        proxy_quality[i] = fname_to_q[fn]

            # Otherwise use EF-based proxy
            elif "EF" in fl.columns:
                matching = fl[fl["FileName"].isin(filenames)]
                if len(matching) > 0:
                    ef_proxy = compute_ef_proxy_quality(matching)
                    fname_to_q = dict(zip(matching["FileName"].astype(str), ef_proxy))
                    for i, fn in enumerate(filenames):
                        if fn in fname_to_q:
                            proxy_quality[i] = fname_to_q[fn]

    # Fallback: use embedding density as self-proxy
    if proxy_quality.sum() == 0:
        print("WARNING: No proxy quality found. Using embedding density as self-proxy.")
        proxy_quality = scores["embedding_density"]

    # Evaluate
    metrics = evaluate_quality_scores(scores, proxy_quality, views=views)

    print("\n--- Global Metrics ---")
    global_metrics = {k: v for k, v in metrics.items() if not any(
        k.startswith(f"spearman_{v}") or k.startswith(f"auc_{v}")
        for v in set(views)
    )}
    print(format_metrics(global_metrics))

    per_view = {k: v for k, v in metrics.items() if k not in global_metrics}
    if per_view:
        print("\n--- Per-View Metrics ---")
        print(format_metrics(per_view))

    # Visualize
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    ranked_df = visualize_results(
        scores, proxy_quality, filenames, str(fig_dir), views=views,
        num_examples=config["evaluation"].get("num_examples", 10),
    )

    # Save
    scores_dir = ensure_dir(os.path.join(output_dir, "scores"))
    scores_df = pd.DataFrame({
        "filename": filenames,
        "view": views,
        "composite_score": scores["composite"],
        "view_confidence": scores["view_confidence"],
        "embedding_density": scores["embedding_density"],
        "vl_alignment": scores["vl_alignment"],
        "proxy_quality": proxy_quality,
    })
    scores_df.to_csv(os.path.join(scores_dir, "quality_scores.csv"), index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(scores_dir, "metrics.csv"), index=False)

    print(f"\nResults saved to {output_dir}")
    return metrics, ranked_df


def main():
    parser = argparse.ArgumentParser(
        description="EchoQual: Multi-view self-supervised quality assessment"
    )
    parser.add_argument("--config", default="configs/multiview.yaml")
    parser.add_argument(
        "--data_sources", nargs="+",
        help="Data sources as name=path pairs, e.g.: echonet=data/EchoNet-Dynamic camus=data/CAMUS"
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--backbone", default=None,
                        help="Override encoder backbone (echoprime, r3d_18, etc)")
    parser.add_argument("--echoprime_dir", default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--skip_ssl", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    config = load_config(args.config)

    # Apply CLI overrides
    if args.backbone:
        config["encoder"]["backbone"] = args.backbone
    if args.echoprime_dir:
        config["encoder"]["echoprime_dir"] = args.echoprime_dir
    if args.max_videos:
        config["data"]["max_videos_per_source"] = args.max_videos
    if args.skip_ssl:
        config["ssl_pretraining"]["enabled"] = False

    output_dir = args.output_dir or config.get("output", {}).get("base_dir", "outputs/multiview")
    ensure_dir(output_dir)

    # Parse data sources
    sources = parse_data_sources(args.data_sources, config)
    if not sources:
        print("ERROR: No data sources specified. Use --data_sources or enable sources in config.")
        sys.exit(1)

    print(f"\nData sources: {list(sources.keys())}")
    for name, cfg in sources.items():
        print(f"  {name}: {cfg['root_dir']} (type={cfg['type']})")

    # Load combined dataset
    print("\nLoading datasets...")
    max_vps = config["data"].get("max_videos_per_source")
    train_dataset = create_multiview_dataset(
        sources, split="TRAIN",
        num_frames=config["data"]["num_frames"],
        frame_size=config["data"]["frame_size"],
        max_videos_per_source=max_vps,
    )
    print(f"Training set: {len(train_dataset)} videos")

    full_dataset = create_multiview_dataset(
        sources, split=None,
        num_frames=config["data"]["num_frames"],
        frame_size=config["data"]["frame_size"],
        max_videos_per_source=max_vps,
    )
    print(f"Full dataset: {len(full_dataset)} videos")

    # Initialize model
    backbone = config["encoder"]["backbone"]
    print(f"\nInitializing model (backbone={backbone})...")
    model = EchoQualModel(
        backbone=backbone,
        pretrained=config["encoder"].get("pretrained_imagenet", True),
        embedding_dim=config["encoder"]["embedding_dim"],
        num_views=NUM_VIEWS,
        echoprime_dir=config["encoder"].get("echoprime_dir"),
        freeze_backbone=config["encoder"].get("freeze_backbone", False),
    ).to(device)

    # Step 1: SSL
    step1_ssl_pretrain(model, train_dataset, config, device, output_dir)

    # Step 2: Extract embeddings
    embeddings, view_logits, filenames, views = step2_extract_embeddings(
        model, full_dataset, device, batch_size=args.batch_size,
    )

    # Save embeddings
    emb_dir = ensure_dir(os.path.join(output_dir, "embeddings"))
    np.save(os.path.join(emb_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(emb_dir, "filenames.npy"), filenames)
    np.save(os.path.join(emb_dir, "views.npy"), views)
    torch.save(view_logits, os.path.join(emb_dir, "view_logits.pt"))

    # Step 3: Quality scores
    scores, scorer = step3_compute_quality_scores(
        embeddings, view_logits, views, config,
    )

    # Step 4: Evaluate
    metrics, ranked_df = step4_evaluate(
        scores, filenames, views, sources, output_dir, config,
    )

    print("\n" + "=" * 60)
    print("MULTI-VIEW PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results: {output_dir}")
    print(f"Views processed: {sorted(set(views))}")
    print(f"Total videos scored: {len(filenames)}")


if __name__ == "__main__":
    main()
