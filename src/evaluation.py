"""
Evaluation of self-supervised quality scores — now with per-view metrics.

Proxy ground truth (no expert labels):
  1. EF prediction confidence
  2. Synthetic perturbation ranking
  3. QualityProxy column (if available in FileList)

Metrics: Spearman rho, Kendall tau, AUC (global + per-view).
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_ef_proxy_quality(filelist: pd.DataFrame, predicted_ef=None) -> np.ndarray:
    """Proxy quality from EF prediction confidence or heuristic."""
    if predicted_ef is not None:
        ef_error = np.abs(predicted_ef - filelist["EF"].values)
        return 1.0 / (1.0 + ef_error / 10.0)
    else:
        ef = filelist["EF"].values.astype(float)
        ef_plausible = np.exp(-((ef - 55) ** 2) / (2 * 15**2))
        esv = filelist.get("ESV", pd.Series(np.zeros(len(filelist)))).values.astype(float)
        edv = filelist.get("EDV", pd.Series(np.ones(len(filelist)))).values.astype(float)
        vol_ratio = esv / (edv + 1e-6)
        vol_plausible = np.exp(-((vol_ratio - 0.4) ** 2) / (2 * 0.15**2))
        return 0.6 * ef_plausible + 0.4 * vol_plausible


def evaluate_quality_scores(
    scores: Dict[str, np.ndarray],
    proxy_quality: np.ndarray,
    views: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate quality scores globally and per-view."""
    metrics = {}
    composite = scores.get("composite", np.zeros_like(proxy_quality))

    # Global metrics
    rho, p = stats.spearmanr(composite, proxy_quality)
    metrics["spearman_rho"] = rho
    metrics["spearman_pval"] = p

    tau, p_tau = stats.kendalltau(composite, proxy_quality)
    metrics["kendall_tau"] = tau

    binary_labels = (proxy_quality >= np.median(proxy_quality)).astype(int)
    if len(np.unique(binary_labels)) > 1:
        metrics["auc_binary"] = roc_auc_score(binary_labels, composite)
    else:
        metrics["auc_binary"] = float("nan")

    # Per-signal correlations
    for sig in ["view_confidence", "embedding_density", "vl_alignment"]:
        if sig in scores:
            r, _ = stats.spearmanr(scores[sig], proxy_quality)
            metrics[f"spearman_{sig}"] = r

    # Per-view metrics
    if views is not None:
        view_arr = np.array(views)
        for v in sorted(set(views)):
            mask = view_arr == v
            if mask.sum() < 5:
                continue
            r_v, _ = stats.spearmanr(composite[mask], proxy_quality[mask])
            metrics[f"spearman_{v}"] = r_v

            bl = (proxy_quality[mask] >= np.median(proxy_quality[mask])).astype(int)
            if len(np.unique(bl)) > 1:
                metrics[f"auc_{v}"] = roc_auc_score(bl, composite[mask])

    return metrics


def visualize_results(
    scores: Dict[str, np.ndarray],
    proxy_quality: np.ndarray,
    filenames: List[str],
    output_dir: str,
    views: Optional[List[str]] = None,
    num_examples: int = 20,
) -> pd.DataFrame:
    """Generate evaluation visualizations with per-view breakdowns."""
    os.makedirs(output_dir, exist_ok=True)
    composite = scores.get("composite", np.zeros_like(proxy_quality))

    n_plots = 4 if views is None else 6
    fig, axes = plt.subplots(2, 3 if views else 2, figsize=(18 if views else 14, 10))
    axes = axes.flatten()

    # 1. Score distribution
    ax = axes[0]
    ax.hist(composite, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Composite Quality Score")
    ax.set_ylabel("Count")
    ax.set_title("Quality Score Distribution")

    # 2. Correlation scatter
    ax = axes[1]
    if views:
        unique_views = sorted(set(views))
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_views)))
        for v, c in zip(unique_views, colors):
            mask = np.array([vv == v for vv in views])
            ax.scatter(proxy_quality[mask], composite[mask], alpha=0.4, s=15, c=[c], label=v)
        ax.legend(fontsize=8)
    else:
        ax.scatter(proxy_quality, composite, alpha=0.3, s=10, color="steelblue")
    rho, _ = stats.spearmanr(composite, proxy_quality)
    ax.set_xlabel("Proxy Quality")
    ax.set_ylabel("Predicted Quality")
    ax.set_title(f"Correlation (Spearman ρ = {rho:.3f})")

    # 3. Per-signal bars
    ax = axes[2]
    sig_names = ["view_confidence", "embedding_density", "vl_alignment"]
    sig_rhos = []
    for s in sig_names:
        if s in scores:
            r, _ = stats.spearmanr(scores[s], proxy_quality)
            sig_rhos.append(r)
        else:
            sig_rhos.append(0)
    bars = ax.bar(
        ["View\nConfidence", "Embedding\nDensity", "VL\nAlignment"],
        sig_rhos, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.8,
    )
    for bar, val in zip(bars, sig_rhos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-Signal Correlation")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 4. ROC
    ax = axes[3]
    bl = (proxy_quality >= np.median(proxy_quality)).astype(int)
    if len(np.unique(bl)) > 1:
        fpr, tpr, _ = roc_curve(bl, composite)
        auc = roc_auc_score(bl, composite)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC: Good vs Poor")
        ax.legend()

    # 5-6. Per-view breakdowns (if views provided)
    if views and len(axes) > 4:
        unique_views = sorted(set(views))

        # Per-view score distributions
        ax = axes[4]
        view_data = []
        for v in unique_views:
            mask = np.array([vv == v for vv in views])
            view_data.append(composite[mask])
        ax.boxplot(view_data, labels=unique_views)
        ax.set_ylabel("Composite Score")
        ax.set_title("Score Distribution by View")

        # Per-view Spearman
        ax = axes[5]
        view_rhos = []
        for v in unique_views:
            mask = np.array([vv == v for vv in views])
            if mask.sum() >= 5:
                r, _ = stats.spearmanr(composite[mask], proxy_quality[mask])
                view_rhos.append(r)
            else:
                view_rhos.append(0)
        ax.bar(unique_views, view_rhos, color=plt.cm.Set2(np.linspace(0, 1, len(unique_views))))
        for i, val in enumerate(view_rhos):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Per-View Correlation")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_evaluation.png"), dpi=150)
    plt.close()

    # Save ranked CSV
    ranked = pd.DataFrame({
        "filename": filenames,
        "composite_score": composite,
        "proxy_quality": proxy_quality,
    })
    if views:
        ranked["view"] = views
    for s in sig_names:
        if s in scores:
            ranked[s] = scores[s]
    ranked = ranked.sort_values("composite_score", ascending=False)
    ranked.to_csv(os.path.join(output_dir, "ranked_quality.csv"), index=False)

    print("\n=== TOP QUALITY ===")
    cols = ["filename", "composite_score", "proxy_quality"]
    if views:
        cols.insert(1, "view")
    print(ranked.head(num_examples)[cols].to_string(index=False))
    print("\n=== BOTTOM QUALITY ===")
    print(ranked.tail(num_examples)[cols].to_string(index=False))

    return ranked
