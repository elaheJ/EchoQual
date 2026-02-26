"""
Evaluation of self-supervised quality scores against proxy ground truth.

Since we have NO expert quality labels, we use proxy signals:
  1. EF prediction confidence: videos where an EF model makes large errors
     are likely poor quality (structures hard to segment = hard to see)
  2. Segmentation consistency: temporal variance in auto-segmentation area
  3. Known perturbation ranking: synthetically degrade images and verify
     that quality scores decrease monotonically

Metrics:
  - Spearman rank correlation with proxy ground truth
  - Kendall's tau
  - AUC for binary good/poor classification
  - Visualization of score distributions
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_ef_proxy_quality(
    filelist: pd.DataFrame,
    predicted_ef: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create proxy quality labels from EF prediction confidence.

    Rationale: Videos where cardiac structures are clearly visible allow
    accurate EF estimation. High absolute EF error → likely poor quality.

    If no predicted EF is available, we use variance in reported ESV/EDV
    as a proxy (unusual values suggest measurement difficulty).

    Args:
        filelist: DataFrame with EF, ESV, EDV columns
        predicted_ef: Optional model-predicted EF values
    Returns:
        proxy_quality: [N] continuous proxy quality scores (higher = better)
    """
    if predicted_ef is not None:
        true_ef = filelist["EF"].values
        ef_error = np.abs(predicted_ef - true_ef)
        # Invert: low error = high quality
        proxy_quality = 1.0 / (1.0 + ef_error / 10.0)
    else:
        # Heuristic: use EF, ESV, EDV plausibility as proxy
        ef = filelist["EF"].values.astype(float)
        esv = filelist.get("ESV", pd.Series(np.zeros(len(filelist)))).values.astype(float)
        edv = filelist.get("EDV", pd.Series(np.zeros(len(filelist)))).values.astype(float)

        # Physiologically plausible ranges suggest good quality
        ef_plausible = np.exp(-((ef - 55) ** 2) / (2 * 15**2))  # Peak at normal EF

        # Volume ratio consistency
        if edv.sum() > 0:
            vol_ratio = esv / (edv + 1e-6)
            vol_plausible = np.exp(-((vol_ratio - 0.4) ** 2) / (2 * 0.15**2))
        else:
            vol_plausible = np.ones_like(ef_plausible)

        proxy_quality = 0.6 * ef_plausible + 0.4 * vol_plausible

    return proxy_quality


def compute_perturbation_ranking(
    model,
    videos: list,
    scorer,
    device: torch.device,
    perturbation_levels: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
) -> Dict:
    """
    Synthetic validation: add increasing noise to videos and verify
    that quality scores decrease monotonically.

    Returns:
        dict with monotonicity rate and mean score per perturbation level
    """
    import torch

    results = {level: [] for level in perturbation_levels}

    for video_data in videos:
        video = video_data["video"].unsqueeze(0).to(device)

        for level in perturbation_levels:
            if level > 0:
                noise = torch.randn_like(video) * level
                perturbed = (video + noise).clamp(0, 1)
            else:
                perturbed = video

            with torch.no_grad():
                emb = model.encode(perturbed).cpu().numpy()

            score = scorer.score(emb)["composite"][0]
            results[level].append(score)

    # Compute monotonicity: fraction of videos where scores decrease with noise
    mean_scores = {k: np.mean(v) for k, v in results.items()}
    levels_sorted = sorted(perturbation_levels)
    monotonic_count = 0
    total_pairs = 0

    for i in range(len(levels_sorted) - 1):
        for j in range(i + 1, len(levels_sorted)):
            if mean_scores[levels_sorted[i]] >= mean_scores[levels_sorted[j]]:
                monotonic_count += 1
            total_pairs += 1

    monotonicity_rate = monotonic_count / max(total_pairs, 1)

    return {
        "mean_scores_by_level": mean_scores,
        "monotonicity_rate": monotonicity_rate,
    }


def evaluate_quality_scores(
    scores: Dict[str, np.ndarray],
    proxy_quality: np.ndarray,
    binary_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate quality scores against proxy ground truth.

    Args:
        scores: Dict with 'composite' and individual signal scores
        proxy_quality: [N] proxy quality labels (continuous)
        binary_threshold: Threshold for binary good/poor classification
    Returns:
        metrics: Dict of evaluation metrics
    """
    metrics = {}

    composite = scores.get("composite", np.zeros_like(proxy_quality))

    # Spearman rank correlation
    rho, p_val = stats.spearmanr(composite, proxy_quality)
    metrics["spearman_rho"] = rho
    metrics["spearman_pval"] = p_val

    # Kendall's tau
    tau, p_val_tau = stats.kendalltau(composite, proxy_quality)
    metrics["kendall_tau"] = tau
    metrics["kendall_pval"] = p_val_tau

    # Binary AUC
    binary_labels = (proxy_quality >= np.median(proxy_quality)).astype(int)
    if len(np.unique(binary_labels)) > 1:
        auc = roc_auc_score(binary_labels, composite)
        metrics["auc_binary"] = auc
    else:
        metrics["auc_binary"] = float("nan")

    # Per-signal correlations
    for signal_name in ["view_confidence", "embedding_density", "vl_alignment"]:
        if signal_name in scores:
            rho_s, _ = stats.spearmanr(scores[signal_name], proxy_quality)
            metrics[f"spearman_{signal_name}"] = rho_s

    return metrics


def visualize_results(
    scores: Dict[str, np.ndarray],
    proxy_quality: np.ndarray,
    filenames: List[str],
    output_dir: str,
    num_examples: int = 20,
):
    """Generate evaluation visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    composite = scores.get("composite", np.zeros_like(proxy_quality))

    # 1. Score distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(composite, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Composite Quality Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Self-Supervised Quality Scores")

    # 2. Correlation scatter
    ax = axes[0, 1]
    ax.scatter(proxy_quality, composite, alpha=0.3, s=10, color="steelblue")
    rho, _ = stats.spearmanr(composite, proxy_quality)
    ax.set_xlabel("Proxy Quality (EF-based)")
    ax.set_ylabel("Predicted Quality Score")
    ax.set_title(f"Correlation (Spearman ρ = {rho:.3f})")

    # Add regression line
    z = np.polyfit(proxy_quality, composite, 1)
    p = np.poly1d(z)
    x_range = np.linspace(proxy_quality.min(), proxy_quality.max(), 100)
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)

    # 3. Per-signal comparison
    ax = axes[1, 0]
    signal_names = ["view_confidence", "embedding_density", "vl_alignment"]
    signal_correlations = []
    for name in signal_names:
        if name in scores:
            rho_s, _ = stats.spearmanr(scores[name], proxy_quality)
            signal_correlations.append(rho_s)
        else:
            signal_correlations.append(0)

    bars = ax.bar(
        ["View\nConfidence", "Embedding\nDensity", "VL\nAlignment"],
        signal_correlations,
        color=["#4C72B0", "#55A868", "#C44E52"],
        alpha=0.8,
    )
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-Signal Correlation with Proxy Quality")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Add value labels
    for bar, val in zip(bars, signal_correlations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 4. ROC curve
    ax = axes[1, 1]
    binary_labels = (proxy_quality >= np.median(proxy_quality)).astype(int)
    if len(np.unique(binary_labels)) > 1:
        fpr, tpr, _ = roc_curve(binary_labels, composite)
        auc = roc_auc_score(binary_labels, composite)
        ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC: Good vs Poor Quality Classification")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_evaluation.png"), dpi=150)
    plt.close()

    # 5. Save ranked results
    ranked_df = pd.DataFrame(
        {
            "filename": filenames,
            "composite_score": composite,
            "proxy_quality": proxy_quality,
        }
    )
    for name in signal_names:
        if name in scores:
            ranked_df[name] = scores[name]

    ranked_df = ranked_df.sort_values("composite_score", ascending=False)
    ranked_df.to_csv(os.path.join(output_dir, "ranked_quality.csv"), index=False)

    # Print top and bottom examples
    print("\n=== TOP QUALITY (best) ===")
    print(ranked_df.head(num_examples)[["filename", "composite_score", "proxy_quality"]])
    print("\n=== BOTTOM QUALITY (worst) ===")
    print(ranked_df.tail(num_examples)[["filename", "composite_score", "proxy_quality"]])

    return ranked_df
