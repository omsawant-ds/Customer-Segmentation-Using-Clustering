import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation.data import clean_data, load_data  # noqa: E402
from segmentation.evaluate import score_labels, stability  # noqa: E402
from segmentation.models import tune_dbscan  # noqa: E402
from segmentation.preprocess import SkewLogTransformer, engineer_features, prepare  # noqa: E402
from segmentation.profile import ARCHETYPES  # noqa: E402
from segmentation.segmenter import RAW_COLUMNS, Segmenter  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return load_data()


@pytest.fixture(scope="module")
def fitted(raw):
    return Segmenter(k=5).fit(raw)


def test_clean_data_has_no_missing_and_no_id(raw):
    df, ids = clean_data(raw)
    assert df.isna().sum().sum() == 0
    assert "CUST_ID" not in df.columns
    assert len(df) == len(ids) == len(raw) - 1  # one row has no credit limit


def test_engineered_features_are_finite(raw):
    df, _ = clean_data(raw)
    feats = engineer_features(df)
    assert np.isfinite(feats.to_numpy()).all()
    assert {"UTILIZATION", "PAYMENT_RATIO", "AVG_PURCHASE", "CASH_ADVANCE_SHARE"} <= set(feats.columns)


def test_log_transform_reduces_skew(raw):
    df, _ = clean_data(raw)
    t = SkewLogTransformer().fit(df)
    out = t.transform(df)
    for col in t.log_columns_:
        assert out[col].skew() < df[col].skew()


def test_preprocessed_matrix_is_standardised(raw):
    df, _ = clean_data(raw)
    _, X, _ = prepare(df)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1, atol=1e-6)


def test_score_labels_ignores_noise():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2)), rng.normal(20, 5, (10, 2))])
    labels = np.array([0] * 50 + [1] * 50 + [-1] * 10)
    s = score_labels(X, labels)
    assert s["n_clusters"] == 2 and s["silhouette"] > 0.9 and s["noise"] == pytest.approx(10 / 110)


def test_single_cluster_gets_nan_silhouette():
    X = np.random.default_rng(0).normal(size=(30, 2))
    s = score_labels(X, np.zeros(30, dtype=int))
    assert np.isnan(s["silhouette"])


def test_dbscan_rejects_degenerate_solutions(raw):
    df, _ = clean_data(raw)
    _, X, _ = prepare(df)
    best = tune_dbscan(X, eps_grid=[1.6], min_samples_grid=[10])
    labels = best["labels"]
    assert best["n_clusters"] >= 2 and best["noise"] <= 0.25
    assert np.bincount(labels[labels != -1]).max() / (labels != -1).sum() <= 0.9


def test_kmeans_is_stable(raw):
    df, _ = clean_data(raw)
    _, X, _ = prepare(df)
    assert stability(X, "K-Means", 5, n_boot=3) > 0.8


def test_segmenter_assigns_every_persona_once(fitted):
    names = list(fitted.personas_.values())
    assert sorted(names) == sorted(ARCHETYPES)
    assert fitted.profile_["customers"].sum() == len(fitted.labels_)


def test_predict_reproduces_training_labels(fitted, raw):
    out = fitted.predict(raw.head(500))
    clean_ids = fitted.ids_.head(500)
    # first rows are unaffected by the one dropped row (index 5203)
    assert (out["segment"].to_numpy() == fitted.labels_[:500]).mean() > 0.99
    assert list(out["CUST_ID"]) == list(clean_ids)


def test_predict_requires_columns(fitted):
    with pytest.raises(ValueError):
        fitted.predict(pd.DataFrame({"BALANCE": [1.0]}))


def test_predict_handles_single_new_customer(fitted):
    row = pd.DataFrame([{c: 0.0 for c in RAW_COLUMNS}])
    row["CREDIT_LIMIT"] = 3000.0
    row["TENURE"] = 12
    out = fitted.predict(row)
    assert len(out) == 1 and out["persona"].iloc[0] in fitted.personas_.values()
