"""Loading and cleaning the credit-card customer dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, ID_COL


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV (defaults to ``data/Customer_Data.csv`` in the repo)."""
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(features, customer_ids)`` with missing values handled.

    * the single row without ``CREDIT_LIMIT`` is dropped,
    * missing ``MINIMUM_PAYMENTS`` are filled with 0 (no minimum payment was ever due),
    * duplicate customers are removed.
    """
    df = df.copy()
    df = df.dropna(subset=["CREDIT_LIMIT"])
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(0.0)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID_COL])
    df = df.reset_index(drop=True)
    ids = df[ID_COL] if ID_COL in df.columns else pd.Series(df.index.astype(str), name=ID_COL)
    return df.drop(columns=[ID_COL], errors="ignore"), ids
