"""Feature engineering for the final seasonal S2 formulation.

Final target
------------
Mean precipitation anomaly over the next three months:

    (anomaly[t+1] + anomaly[t+2] + anomaly[t+3]) / 3

Final predictors
----------------
- Current ENSO indices.
- 3-month trailing mean for each ENSO index.
- Cyclic encoding of the center month of the forecast window.
- Current precipitation anomaly.
- 3-month trailing mean of precipitation anomaly.

All rolling features use only information available at time t or earlier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


ENSO_VARIABLES = [
    "nino12",
    "nino3",
    "nino34",
    "nino4",
    "soi",
    "tni",
]

TARGET_COLUMN = "target_anomaly_next3m"

MODEL_FEATURES = [
    "nino12",
    "nino12_ma3",
    "nino3",
    "nino3_ma3",
    "nino34",
    "nino34_ma3",
    "nino4",
    "nino4_ma3",
    "soi",
    "soi_ma3",
    "tni",
    "tni_ma3",
    "target_month_sin",
    "target_month_cos",
    "precip_anomaly",
    "precip_anomaly_ma3",
]

METADATA_COLUMNS = [
    "date",
    "target_start",
    "target_center",
    "target_end",
    "precip_anomaly",
    "precip_anomaly_ma3",
]


def compute_monthly_climatology(
    df: pd.DataFrame,
    end_date: str | pd.Timestamp | None = None,
) -> dict[int, float]:
    """Compute the 12 monthly precipitation means.

    During historical evaluation, pass an ``end_date`` to prevent leakage.
    For the final production fit, ``None`` can use all currently known history.
    """
    source = df

    if end_date is not None:
        cutoff = pd.Timestamp(end_date)
        source = df.loc[df["date"] <= cutoff]

    climatology = (
        source.assign(month=source["date"].dt.month)
        .groupby("month")["precipitation_mm"]
        .mean()
    )

    if len(climatology) != 12:
        raise ValueError("The climatology must contain all 12 calendar months.")

    return {int(month): float(value) for month, value in climatology.items()}


def apply_climatology(
    df: pd.DataFrame,
    climatology: Mapping[int, float],
) -> pd.DataFrame:
    """Add monthly climatology and precipitation anomaly."""
    result = df.copy()
    result["month"] = result["date"].dt.month

    normalized = {int(month): float(value) for month, value in climatology.items()}
    result["climatology_mm"] = result["month"].map(normalized)

    if result["climatology_mm"].isna().any():
        missing_months = sorted(
            result.loc[result["climatology_mm"].isna(), "month"].unique()
        )
        raise ValueError(f"Climatology is missing calendar months: {missing_months}")

    result["precip_anomaly"] = (
        result["precipitation_mm"] - result["climatology_mm"]
    )

    return result


def add_forecast_window(df: pd.DataFrame) -> pd.DataFrame:
    """Add dates and cyclic month features for the next-3-month window."""
    result = df.copy()

    result["target_start"] = result["date"] + pd.DateOffset(months=1)
    result["target_center"] = result["date"] + pd.DateOffset(months=2)
    result["target_end"] = result["date"] + pd.DateOffset(months=3)

    target_month = result["target_center"].dt.month
    result["target_month_sin"] = np.sin(2 * np.pi * target_month / 12)
    result["target_month_cos"] = np.cos(2 * np.pi * target_month / 12)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Add trailing means using current and previous observations only."""
    result = df.copy()

    for variable in ENSO_VARIABLES:
        result[f"{variable}_ma{window}"] = (
            result[variable].rolling(window=window, min_periods=window).mean()
        )

    result[f"precip_anomaly_ma{window}"] = (
        result["precip_anomaly"].rolling(window=window, min_periods=window).mean()
    )

    return result


def add_seasonal_target(
    df: pd.DataFrame,
    offsets: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Add the mean future anomaly over months t+1, t+2 and t+3."""
    if tuple(offsets) != (1, 2, 3):
        raise ValueError(
            "The current production target is fixed to months t+1, t+2, t+3."
        )

    result = df.copy()

    future_anomalies = pd.concat(
        [result["precip_anomaly"].shift(-offset) for offset in offsets],
        axis=1,
    )

    result[TARGET_COLUMN] = future_anomalies.mean(axis=1, skipna=False)
    return result


def build_feature_frame(
    df: pd.DataFrame,
    climatology: Mapping[int, float],
) -> pd.DataFrame:
    """Create all predictors required by the final model."""
    result = apply_climatology(df=df, climatology=climatology)
    result = add_forecast_window(result)
    result = add_rolling_features(result, window=3)
    return result


def build_training_frame(
    df: pd.DataFrame,
    climatology: Mapping[int, float],
) -> pd.DataFrame:
    """Create labeled rows for training or historical evaluation."""
    result = build_feature_frame(df=df, climatology=climatology)
    result = add_seasonal_target(result)

    columns = list(dict.fromkeys(METADATA_COLUMNS + MODEL_FEATURES + [TARGET_COLUMN]))

    return (
        result[columns]
        .dropna(subset=MODEL_FEATURES + [TARGET_COLUMN])
        .reset_index(drop=True)
    )


def build_inference_frame(
    df: pd.DataFrame,
    climatology: Mapping[int, float],
) -> pd.DataFrame:
    """Create predictor rows without requiring future precipitation."""
    result = build_feature_frame(df=df, climatology=climatology)
    columns = list(dict.fromkeys(METADATA_COLUMNS + MODEL_FEATURES))

    return (
        result[columns]
        .dropna(subset=MODEL_FEATURES)
        .reset_index(drop=True)
    )


def latest_inference_row(
    df: pd.DataFrame,
    climatology: Mapping[int, float],
) -> pd.DataFrame:
    """Return the most recent row for which all model features exist."""
    inference = build_inference_frame(df=df, climatology=climatology)

    if inference.empty:
        raise ValueError("No complete inference row could be constructed.")

    return inference.tail(1).copy()
