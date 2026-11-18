"""Turn anonymous cluster ids into business personas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Each archetype is a set of (feature, direction) cues. Clusters are matched to archetypes
# by maximising total agreement between the cue and the cluster's standardised mean
# (optimal one-to-one assignment), so naming does not depend on arbitrary cluster ids.
ARCHETYPES: dict[str, dict] = {
    "Cash-Advance Borrowers": {
        "cues": {"CASH_ADVANCE_FREQUENCY": 1.0, "CASH_ADVANCE": 1.0, "PURCHASES_FREQUENCY": -1.5, "PURCHASES": -1.0},
        "description": "Use the card mainly to withdraw cash and carry a balance; almost no purchases.",
        "action": "Offer low-rate personal loans or balance-transfer plans; monitor for credit risk.",
    },
    "Heavy Revolvers": {
        "cues": {"CASH_ADVANCE_FREQUENCY": 1.0, "BALANCE": 1.0, "UTILIZATION": 1.0, "PURCHASES": 0.5, "PAYMENTS": 1.0},
        "description": "High balances and high utilisation, mixing purchases with frequent cash advances.",
        "action": "Highest interest revenue but also highest risk: credit-line reviews, instalment-conversion offers.",
    },
    "Savvy Transactors": {
        "cues": {"PRC_FULL_PAYMENT": 1.5, "BALANCE": -1.0, "UTILIZATION": -1.0, "PURCHASES_FREQUENCY": 0.5, "INSTALLMENTS_PURCHASES": 0.5},
        "description": "Buy often, pay the statement in full and keep balances low.",
        "action": "Reward with cashback / points and premium cards; revenue comes from interchange, not interest.",
    },
    "Active Spenders": {
        "cues": {"PURCHASES_FREQUENCY": 1.5, "ONEOFF_PURCHASES": 1.0, "PURCHASES_TRX": 1.0, "PRC_FULL_PAYMENT": -0.5},
        "description": "Frequent buyers with sizeable one-off purchases who sometimes revolve.",
        "action": "Cross-sell instalment plans and spend-based rewards; good candidates for limit increases.",
    },
    "Low-Activity Customers": {
        "cues": {"PURCHASES_FREQUENCY": -1.0, "PURCHASES": -1.0, "BALANCE": -1.0, "PAYMENTS": -1.0, "CREDIT_LIMIT": -0.5},
        "description": "Small limits, low balances and rare use of the card.",
        "action": "Re-engagement campaigns, welcome bonuses, or limit-based nudges; churn-risk watch-list.",
    },
}


def standardised_means(X: np.ndarray, labels: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Mean of each (scaled) feature per cluster; noise label -1 is ignored."""
    frame = pd.DataFrame(X, columns=columns)
    frame["cluster"] = labels
    return frame[frame["cluster"] != -1].groupby("cluster")[columns].mean()


def assign_personas(z_means: pd.DataFrame) -> dict[int, str]:
    """Optimal one-to-one match of clusters to archetypes; leftovers become ``Segment <id>``."""
    names = list(ARCHETYPES)
    score = np.zeros((len(z_means), len(names)))
    for j, name in enumerate(names):
        for feat, weight in ARCHETYPES[name]["cues"].items():
            if feat in z_means.columns:
                score[:, j] += weight * z_means[feat].to_numpy()
    rows, cols = linear_sum_assignment(-score)
    mapping = {int(z_means.index[r]): names[c] for r, c in zip(rows, cols)}
    for cid in z_means.index:
        mapping.setdefault(int(cid), f"Segment {cid}")
    return mapping


def build_profile(features: pd.DataFrame, labels: np.ndarray, personas: dict[int, str]) -> pd.DataFrame:
    """Per-segment size and median of the main behavioural features (original units)."""
    key_cols = ["BALANCE", "CREDIT_LIMIT", "PURCHASES", "CASH_ADVANCE", "PAYMENTS", "PURCHASES_FREQUENCY",
                "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT", "UTILIZATION", "AVG_PURCHASE"]
    key_cols = [c for c in key_cols if c in features.columns]
    f = features.assign(cluster=labels)
    f = f[f["cluster"] != -1]
    prof = f.groupby("cluster")[key_cols].median()
    prof.insert(0, "customers", f.groupby("cluster").size())
    prof.insert(1, "share_%", (100 * prof["customers"] / prof["customers"].sum()).round(1))
    prof.insert(0, "persona", [personas[int(i)] for i in prof.index])
    return prof.round(2)


def persona_info(name: str) -> dict:
    return ARCHETYPES.get(name, {"description": "Unlabelled segment.", "action": "Inspect profile manually."})
