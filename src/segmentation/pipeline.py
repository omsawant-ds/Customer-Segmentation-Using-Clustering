"""End-to-end run: ``python -m segmentation.pipeline`` regenerates every table and figure."""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from . import viz
from .config import REPORTS_DIR
from .data import clean_data, load_data
from .evaluate import compare_models, sweep_k
from .models import tune_dbscan
from .preprocess import engineer_features
from .profile import persona_info, standardised_means
from .segmenter import DEFAULT_K, Segmenter


def run(k: int = DEFAULT_K, data_path=None) -> dict:
    warnings.filterwarnings("ignore")
    REPORTS_DIR.mkdir(exist_ok=True)
    raw = load_data(data_path)
    df, ids = clean_data(raw)
    seg = Segmenter(k=k).fit(raw)
    X, feats = seg.X_, seg.features_

    print(f"{len(df):,} customers, {len(feats.columns)} features")
    logged = seg.pre_.named_steps["log"].transform(feats)
    viz.skew_before_after(feats, logged)

    print("Sweeping k ...")
    sweep = sweep_k(X)
    sweep.to_csv(REPORTS_DIR / "k_sweep.csv", index=False)
    viz.k_selection(sweep, k)

    print("Comparing algorithms ...")
    db = tune_dbscan(X)
    table, _ = compare_models(X, k, dbscan=db)
    table.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")
    viz.model_comparison(table)

    z = standardised_means(X, seg.labels_, seg.columns_)
    viz.segments_pca(seg.projection())
    viz.persona_heatmap(z, seg.personas_, seg.columns_)
    viz.segment_sizes(seg.profile_)

    seg.profile_.to_csv(REPORTS_DIR / "segment_profiles.csv")
    assignments = pd.DataFrame({"CUST_ID": ids, "segment": seg.labels_,
                                "persona": [seg.personas_[int(i)] for i in seg.labels_]})
    assignments.to_csv(REPORTS_DIR / "customer_segments.csv", index=False)

    print("\nModel comparison\n", table.round(3).to_string())
    print("\nSegment profiles (medians)\n", seg.profile_.to_string())
    for name in seg.profile_["persona"]:
        info = persona_info(name)
        print(f"\n{name}: {info['description']}\n  -> {info['action']}")
    return {"segmenter": seg, "table": table, "sweep": sweep}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of segments (default 5)")
    parser.add_argument("--data", default=None, help="path to the customer CSV")
    args = parser.parse_args()
    run(args.k, args.data)


if __name__ == "__main__":
    main()
