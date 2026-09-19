"""High-level orchestration for training and forecasting."""

from __future__ import annotations

from curses import raw
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .config import ProjectPaths
from .data import load_enso_precipitation
from .features import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_training_frame,
    compute_monthly_climatology,
    latest_inference_row,
)
from .model import load_model, predict, save_model, train_model


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    climatology_path: Path
    metadata_path: Path
    training_samples: int
    training_start: str
    training_end: str


@dataclass(frozen=True)
class ForecastResult:
    observation_date: str
    target_start: str
    target_end: str
    predicted_anomaly_mm_per_month: float
    persistence_baseline_mm_per_month: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


def save_climatology(
    climatology: dict[int, float],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(month): value for month, value in climatology.items()}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_climatology(path: str | Path) -> dict[int, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(month): float(value) for month, value in raw.items()}


def train_final_model(paths: ProjectPaths | None = None) -> TrainingArtifacts:
    """Fit the production model using all currently labeled history."""
    paths = paths or ProjectPaths.from_package()
    paths.ensure_output_dirs()

    raw = load_enso_precipitation(paths.data_processed)

    # Keep the climatology definition fixed to the same reference
    # period used during model development and evaluation.
    CLIMATOLOGY_END = pd.Timestamp("2010-12-01")

    climatology = compute_monthly_climatology(raw, end_date=CLIMATOLOGY_END,)
    training = build_training_frame(raw, climatology=climatology)

    model = train_model(training)

    save_model(model, paths.model_file)
    save_climatology(climatology, paths.climatology_file)

    metadata = {
        "model_name": "KernelRidge",
        "kernel": "linear",
        "alpha": 1.0,
        "preprocessing": [
            "SimpleImputer(strategy='mean')",
            "StandardScaler()",
        ],
        "target": TARGET_COLUMN,
        "target_description": (
            "Mean monthly precipitation anomaly over months t+1, t+2 and t+3."
        ),
        "features": MODEL_FEATURES,
        "training_samples": int(len(training)),
        "training_feature_start": str(training["date"].min().date()),
        "training_feature_end": str(training["date"].max().date()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    paths.metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return TrainingArtifacts(
        model_path=paths.model_file,
        climatology_path=paths.climatology_file,
        metadata_path=paths.metadata_file,
        training_samples=int(len(training)),
        training_start=str(training["date"].min().date()),
        training_end=str(training["date"].max().date()),
    )


def predict_latest(paths: ProjectPaths | None = None) -> ForecastResult:
    """Forecast the next 3-month mean anomaly from the latest known month."""
    paths = paths or ProjectPaths.from_package()

    raw = load_enso_precipitation(paths.data_processed)
    model = load_model(paths.model_file)
    climatology = load_climatology(paths.climatology_file)

    latest = latest_inference_row(raw, climatology=climatology)
    prediction = float(predict(model, latest)[0])
    row = latest.iloc[0]

    return ForecastResult(
        observation_date=str(pd.Timestamp(row["date"]).date()),
        target_start=str(pd.Timestamp(row["target_start"]).date()),
        target_end=str(pd.Timestamp(row["target_end"]).date()),
        predicted_anomaly_mm_per_month=prediction,
        persistence_baseline_mm_per_month=float(row["precip_anomaly_ma3"]),
    )
