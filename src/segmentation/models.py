"""Clustering algorithms behind one interface."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .config import SEED


class _GMM:
    """GaussianMixture exposing ``fit_predict`` like the other estimators."""

    def __init__(self, k):
        self.model = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=SEED)

    def fit_predict(self, X):
        return self.model.fit(X).predict(X)


# Algorithms that take a number of clusters.
MODEL_FACTORIES: dict[str, Callable[[int], object]] = {
    "K-Means": lambda k: KMeans(n_clusters=k, n_init=10, random_state=SEED),
    "MiniBatch K-Means": lambda k: MiniBatchKMeans(n_clusters=k, n_init=10, batch_size=1024, random_state=SEED),
    "Agglomerative (Ward)": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
    "Gaussian Mixture": lambda k: _GMM(k),
}


def fit_predict(name: str, X: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(MODEL_FACTORIES[name](k).fit_predict(X))


def tune_dbscan(X, eps_grid=None, min_samples_grid=None, max_noise: float = 0.25, min_clusters: int = 2) -> dict:
    """Grid-search DBSCAN, keeping only settings that give a usable segmentation.

    A setting is usable when it finds at least ``min_clusters`` clusters, labels at most
    ``max_noise`` of customers as noise, and no cluster holds >90% of the non-noise points.
    Among those the highest silhouette (on non-noise points) wins. This rules out the degenerate
    "one big cluster + noise" solution, which can show an inflated silhouette while being useless.
    """
    eps_grid = eps_grid if eps_grid is not None else np.arange(0.8, 3.01, 0.2)
    min_samples_grid = min_samples_grid if min_samples_grid is not None else [10, 20, 30, 50]
    best = None
    for eps in eps_grid:
        for ms in min_samples_grid:
            labels = DBSCAN(eps=float(eps), min_samples=int(ms)).fit_predict(X)
            mask = labels != -1
            n_clusters = len(set(labels[mask]))
            noise = 1 - mask.mean()
            if n_clusters < min_clusters or noise > max_noise:
                continue
            if np.bincount(labels[mask]).max() / mask.sum() > 0.9:
                continue
            sil = silhouette_score(X[mask], labels[mask])
            if best is None or sil > best["silhouette"]:
                best = {"eps": round(float(eps), 2), "min_samples": int(ms), "n_clusters": n_clusters,
                        "noise": float(noise), "silhouette": float(sil), "labels": labels}
    if best is None:
        raise RuntimeError("No DBSCAN setting satisfied the constraints; widen the grid.")
    return best
