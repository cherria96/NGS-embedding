"""Figure generation (design-spec item 17's Figures artifact). Every figure
built from a synthetic run is titled with a "SYNTHETIC" banner."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _maybe_synthetic_title(title: str, synthetic: bool) -> str:
    return f"[SYNTHETIC DEMONSTRATION] {title}" if synthetic else title


def plot_oof_predicted_vs_observed(oof: pd.DataFrame, path: str, synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    sub = oof.dropna(subset=["prediction"])
    ax.scatter(sub["target"], sub["prediction"], alpha=0.7, s=20)
    if len(sub):
        lo = min(sub["target"].min(), sub["prediction"].min())
        hi = max(sub["target"].max(), sub["prediction"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Observed (original units)")
    ax.set_ylabel("Pooled OOF prediction")
    ax.set_title(_maybe_synthetic_title("OOF predicted vs. observed", synthetic))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals_by_site(oof: pd.DataFrame, path: str, synthetic: bool = False) -> None:
    sub = oof.dropna(subset=["prediction"]).copy()
    sub["resid"] = sub["target"] - sub["prediction"]
    order = sub.groupby("site_id")["resid"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(order)), 4))
    data = [sub.loc[sub["site_id"] == s, "resid"].to_numpy() for s in order]
    ax.boxplot(data, labels=order, showmeans=True)
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_ylabel("Residual (original units)")
    ax.set_title(_maybe_synthetic_title("OOF residuals by site", synthetic))
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compression_vs_error(search_results: pd.DataFrame, path: str, synthetic: bool = False,
                               score_label: str = "Inner site-balanced MSE") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(search_results["n_active_taxa"], search_results["mean_site_score"], alpha=0.6, s=20)
    ax.set_xlabel("Active taxa (inner-CV mean)")
    ax.set_ylabel(score_label)
    ax.set_title(_maybe_synthetic_title("Compression vs. error over the regularization path", synthetic))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(oof: pd.DataFrame, path: str, synthetic: bool = False) -> None:
    from sklearn.metrics import roc_curve, roc_auc_score

    sub = oof.dropna(subset=["prediction"])
    fig, ax = plt.subplots(figsize=(5, 5))
    if sub["target"].nunique() > 1:
        fpr, tpr, _ = roc_curve(sub["target"], sub["prediction"])
        auc = roc_auc_score(sub["target"], sub["prediction"])
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.legend()
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(_maybe_synthetic_title("Pooled OOF ROC curve", synthetic))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_graph_weight_summary(edges: pd.DataFrame, path: str, domain: str, synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(edges) and "mixed_weight" in edges.columns:
        ax.hist(edges["mixed_weight"].dropna(), bins=20)
    ax.set_xlabel("Mixed edge weight")
    ax.set_ylabel("Count")
    ax.set_title(_maybe_synthetic_title(f"{domain} graph edge weight distribution", synthetic))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
