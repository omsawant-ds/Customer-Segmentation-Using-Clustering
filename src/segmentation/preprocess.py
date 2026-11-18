"""Feature engineering and the preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SKEW_THRESHOLD


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few behavioural ratios that are easier to interpret than raw amounts."""
    out = df.copy()
    out["UTILIZATION"] = (out["BALANCE"] / out["CREDIT_LIMIT"]).clip(0, 1.5)
    out["PAYMENT_RATIO"] = (out["PAYMENTS"] / (out["BALANCE"] + 1.0)).clip(0, 50)
    avg = out["PURCHASES"] / out["PURCHASES_TRX"].replace(0, np.nan)
    out["AVG_PURCHASE"] = avg.fillna(0.0)
    out["CASH_ADVANCE_SHARE"] = out["CASH_ADVANCE"] / (out["CASH_ADVANCE"] + out["PURCHASES"] + 1.0)
    return out


class SkewLogTransformer(BaseEstimator, TransformerMixin):
    """``log1p`` every non-negative column whose skewness (measured at fit time) exceeds ``threshold``."""

    def __init__(self, threshold: float = SKEW_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.log_columns_ = [c for c in self.columns_ if X[c].min() >= 0 and X[c].skew() > self.threshold]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        X[self.log_columns_] = np.log1p(X[self.log_columns_])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


def build_preprocessor() -> Pipeline:
    """Log-transform skewed features, then standardise. Fit on training data only."""
    return Pipeline([("log", SkewLogTransformer()), ("scale", StandardScaler())])


def prepare(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, Pipeline]:
    """Engineer features and return ``(features, scaled_matrix, fitted_preprocessor)``."""
    feats = engineer_features(df_clean)
    pre = build_preprocessor()
    X = pre.fit_transform(feats)
    return feats, X, pre
