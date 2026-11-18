"""High-level API: fit on a customer table, then label new customers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import ID_COL, SEED
from .data import clean_data
from .preprocess import engineer_features, build_preprocessor
from .profile import assign_personas, build_profile, standardised_means

DEFAULT_K = 5

RAW_COLUMNS = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
    "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS",
    "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
]


class Segmenter:
    """Preprocessing + K-Means + persona naming in one fitted object."""

    def __init__(self, k: int = DEFAULT_K, seed: int = SEED):
        self.k, self.seed = k, seed

    def fit(self, raw: pd.DataFrame) -> "Segmenter":
        df, ids = clean_data(raw)
        feats = engineer_features(df)
        self.pre_ = build_preprocessor()
        X = self.pre_.fit_transform(feats)
        self.columns_ = list(feats.columns)
        self.kmeans_ = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(X)
        labels = self.kmeans_.labels_
        self.personas_ = assign_personas(standardised_means(X, labels, self.columns_))
        self.profile_ = build_profile(feats, labels, self.personas_)
        self.pca_ = PCA(n_components=2, random_state=self.seed).fit(X)
        self.X_, self.features_, self.labels_, self.ids_ = X, feats, labels, ids
        return self

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Label customers. ``raw`` needs the 17 original behavioural columns."""
        missing = [c for c in RAW_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        data = raw[RAW_COLUMNS].copy()
        data["CREDIT_LIMIT"] = data["CREDIT_LIMIT"].fillna(data["CREDIT_LIMIT"].median())
        data["MINIMUM_PAYMENTS"] = data["MINIMUM_PAYMENTS"].fillna(0.0)
        feats = engineer_features(data)[self.columns_]
        labels = self.kmeans_.predict(self.pre_.transform(feats))
        out = pd.DataFrame({"segment": labels, "persona": [self.personas_[int(i)] for i in labels]}, index=raw.index)
        if ID_COL in raw.columns:
            out.insert(0, ID_COL, raw[ID_COL].values)
        return out

    def projection(self) -> pd.DataFrame:
        """2-D PCA coordinates of the training customers, for plotting."""
        xy = self.pca_.transform(self.X_)
        return pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], "segment": self.labels_,
                             "persona": [self.personas_[int(i)] for i in self.labels_]})
