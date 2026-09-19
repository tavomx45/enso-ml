"""Data loading and validation utilities.

The ML package starts from the processed ENSO table and the processed coastal
precipitation table. If a previously merged file exists, it is reused.
Otherwise the two source tables are merged by month.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ENSO_COLUMNS = [
    "nino12",
    "nino3",
    "nino34",
    "nino4",
    "soi",
    "tni",
]

REQUIRED_COLUMNS = [
    "date",
    "precipitation_mm",
    *ENSO_COLUMNS,
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def validate_monthly_series(df: pd.DataFrame) -> None:
    """Validate ordering, uniqueness and monthly continuity."""
    if df.empty:
        raise ValueError("The dataframe is empty.")

    if df["date"].duplicated().any():
        duplicated = df.loc[df["date"].duplicated(), "date"].tolist()
        raise ValueError(f"Duplicate dates found: {duplicated[:5]}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dates must be sorted in ascending order.")

    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")

    if len(expected) != len(df) or not np.array_equal(expected.values, df["date"].values):
        raise ValueError(
            "The merged dataset must contain exactly one observation per month "
            "with no missing calendar months."
        )


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def merge_enso_precipitation(
    enso_df: pd.DataFrame,
    precipitation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ENSO indicators and coastal precipitation by month."""
    merged = pd.merge(
        enso_df,
        precipitation_df[["date", "precipitation_mm"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.sort_values("date").reset_index(drop=True)

    validate_required_columns(merged)
    validate_monthly_series(merged)

    return merged


def load_enso_precipitation(processed_dir: str | Path) -> pd.DataFrame:
    """Load the monthly ENSO + precipitation dataset.

    Search order:
    1. enso_precip.csv
    2. enso_precipitation.csv
    3. merge enso_master.csv with ecuador_coast_precipitation.csv
    """
    processed_dir = Path(processed_dir)

    merged_candidates = [
        processed_dir / "enso_precip.csv",
        processed_dir / "enso_precipitation.csv",
    ]

    for path in merged_candidates:
        if path.exists():
            df = _read_csv(path)
            df = df.sort_values("date").reset_index(drop=True)
            validate_required_columns(df)
            validate_monthly_series(df)
            return df

    enso_path = processed_dir / "enso_master.csv"
    precip_path = processed_dir / "ecuador_coast_precipitation.csv"

    if not enso_path.exists() or not precip_path.exists():
        raise FileNotFoundError(
            "Could not find a merged ENSO/precipitation file and could not "
            "rebuild it. Expected either enso_precip.csv / "
            "enso_precipitation.csv, or both enso_master.csv and "
            "ecuador_coast_precipitation.csv in data/processed/."
        )

    enso_df = _read_csv(enso_path)
    precipitation_df = _read_csv(precip_path)

    return merge_enso_precipitation(
        enso_df=enso_df,
        precipitation_df=precipitation_df,
    )
