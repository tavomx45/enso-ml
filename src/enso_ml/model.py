"""Final ML model and evaluation helpers.

This reproduces the winning PyCaret candidate without requiring PyCaret at
inference time:

SimpleImputer(mean) -> StandardScaler -> KernelRidge(alpha=1, kernel="linear")
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURES, TARGET_COLUMN


def build_model() -> Pipeline:
    """Build the exact final Kernel Ridge pipeline."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, MODEL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "kr",
                KernelRidge(
                    alpha=1.0,
                    kernel="linear",
                    gamma=None,
                    degree=3,
                    coef0=1.0,
                ),
            ),
        ]
    )


def train_model(training_df: pd.DataFrame) -> Pipeline:
    """Fit the final model on an already engineered training dataframe."""
    model = build_model()
    model.fit(training_df[MODEL_FEATURES], training_df[TARGET_COLUMN])
    return model


def predict(model: Pipeline, feature_df: pd.DataFrame) -> np.ndarray:
    """Predict mean anomaly for the next three months."""
    return model.predict(feature_df[MODEL_FEATURES])


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return the three metrics used throughout the project."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Persist a fitted sklearn pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted sklearn pipeline."""
    return joblib.load(Path(path))
