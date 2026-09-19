"""ENSO seasonal precipitation forecasting package."""

from .config import ProjectPaths
from .features import MODEL_FEATURES, TARGET_COLUMN
from .model import build_model, regression_metrics
from .pipeline import ForecastResult, TrainingArtifacts, predict_latest, train_final_model

__all__ = [
    "ProjectPaths",
    "MODEL_FEATURES",
    "TARGET_COLUMN",
    "build_model",
    "regression_metrics",
    "ForecastResult",
    "TrainingArtifacts",
    "predict_latest",
    "train_final_model",
]

__version__ = "0.1.0"
