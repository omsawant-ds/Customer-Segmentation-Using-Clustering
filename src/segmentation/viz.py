"""Figures used in the README and reports."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, PALETTE

sns.set_theme(style="whitegrid", font_scale=1.0)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def skew_before_after(raw: pd.DataFrame, logged: pd.DataFrame, cols=("BALANCE", "PURCHASES", "PAYMENTS", "MINIMUM_PAYMENTS")):
    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 6))
    for j, c in enumerate(cols):
        sns.histplot(raw[c], bins=50, ax=axes[0, j], color=PALETTE[1])
        axes[0, j].set_title(f"{c}\nskew = {raw[c].skew():.1f}")
        sns.histplot(logged[c], bins=50, ax=axes[1, j], color=PALETTE[2])
        axes[1, j].set_title(f"log1p({c})\nskew = {logged[c].skew():.1f}")
        for r in (0, 1):
            axes[r, j].set_xlabel("")
    fig.suptitle("Heavy right-skew is removed by a log transform before clustering", y=1.02, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_skew_before_after")


def k_selection(sweep: pd.DataFrame, chosen_k: int):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes, ["inertia", "silhouette", "davies_bouldin"],
                              ["Elbow (inertia)", "Silhouette (higher is better)", "Davies-Bouldin (lower is better)"]):
        ax.plot(sweep["k"], sweep[col], marker="o", color=PALETTE[0])
        ax.axvline(chosen_k, ls="--", color="grey")
        ax.set_title(title)
        ax.set_xlabel("k")
    fig.suptitle(f"Choosing the number of segments (chosen k = {chosen_k})", y=1.03, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_k_selection")


def model_comparison(table: pd.DataFrame):
    t = table.reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title in zip(axes, ["silhouette", "davies_bouldin", "stability_ari"],
                              ["Silhouette (higher better)", "Davies-Bouldin (lower better)", "Stability, ARI (higher better)"]):
        sns.barplot(data=t, y="model", x=col, ax=ax, palette=PALETTE[: len(t)], hue="model", legend=False)
        ax.set_title(title)
        ax.set_ylabel("")
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("Algorithm comparison on identical preprocessing", y=1.03, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_model_comparison")


def segments_pca(proj: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    order = proj.groupby("persona").size().sort_values(ascending=False).index
    for i, name in enumerate(order):
        sub = proj[proj["persona"] == name]
        ax.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.5, color=PALETTE[i % len(PALETTE)], label=f"{name} ({len(sub):,})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Customer segments projected onto two principal components", fontweight="bold")
    ax.legend(markerscale=3, frameon=True)
    return _save(fig, "04_segments_pca")


def persona_heatmap(z_means: pd.DataFrame, personas: dict[int, str], features: list[str]):
    data = z_means[features].copy()
    data.index = [personas[int(i)] for i in data.index]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(data, cmap="RdBu_r", center=0, annot=True, fmt=".1f", linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("What defines each segment (standardised feature means; red = above average)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    return _save(fig, "05_persona_heatmap")


def segment_sizes(profile: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    p = profile.sort_values("customers")
    ax.barh(p["persona"], p["customers"], color=PALETTE[: len(p)])
    for y, (n, s) in enumerate(zip(p["customers"], p["share_%"])):
        ax.text(n, y, f"  {n:,} ({s}%)", va="center")
    ax.set_xlim(0, p["customers"].max() * 1.25)
    ax.set_title("Segment sizes", fontweight="bold")
    return _save(fig, "06_segment_sizes")
