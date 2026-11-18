"""Internal validation metrics and stability analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from .config import K_RANGE, SEED
from .models import MODEL_FACTORIES, fit_predict


def score_labels(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz on non-noise points, plus noise share."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = {"n_clusters": n_clusters, "noise": float(1 - mask.mean())}
    if n_clusters < 2:
        return {**out, "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        **out,
        "silhouette": float(silhouette_score(X[mask], labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
        "calinski_harabasz": float(calinski_harabasz_score(X[mask], labels[mask])),
    }


def sweep_k(X: np.ndarray, name: str = "K-Means", k_range=K_RANGE) -> pd.DataFrame:
    """Metrics for each k (plus inertia for K-Means, used by the elbow plot)."""
    rows = []
    for k in k_range:
        labels = fit_predict(name, X, k)
        row = {"k": k, **score_labels(X, labels)}
        if name == "K-Means":
            row["inertia"] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X).inertia_
        rows.append(row)
    return pd.DataFrame(rows)


def stability(X: np.ndarray, name: str, k: int, n_boot: int = 10, frac: float = 0.8) -> float:
    """Mean Adjusted Rand Index between the full-data labelling and labellings of random subsamples.

    Near 1: segments are reproducible. Near 0: they are an artefact of the particular sample.
    """
    rng = np.random.default_rng(SEED)
    base = fit_predict(name, X, k)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        scores.append(adjusted_rand_score(base[idx], fit_predict(name, X[idx], k)))
    return float(np.mean(scores))


def compare_models(X: np.ndarray, k: int, dbscan: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit every algorithm at ``k`` and return a metrics table plus the labels of each."""
    rows, labels_by_model = [], {}
    for name in MODEL_FACTORIES:
        labels = fit_predict(name, X, k)
        labels_by_model[name] = labels
        rows.append({"model": name, **score_labels(X, labels), "stability_ari": stability(X, name, k)})
    if dbscan is not None:
        label = f"DBSCAN (eps={dbscan['eps']}, min_samples={dbscan['min_samples']})"
        labels_by_model[label] = dbscan["labels"]
        rows.append({"model": label, **score_labels(X, dbscan["labels"]), "stability_ari": np.nan})
    return pd.DataFrame(rows).set_index("model"), labels_by_model
