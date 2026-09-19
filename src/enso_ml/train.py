"""Command-line entry point for the final model.

From the repository root:

    uv run python -m enso_ml.train

If the package is not yet installed by the project environment:

    PYTHONPATH=src uv run python -m enso_ml.train
"""

from __future__ import annotations

from .config import ProjectPaths
from .pipeline import predict_latest, train_final_model


def main() -> None:
    paths = ProjectPaths.from_package()
    artifacts = train_final_model(paths)

    print("\nTraining complete")
    print("-----------------")
    print(f"Samples: {artifacts.training_samples}")
    print(f"Feature dates: {artifacts.training_start} -> {artifacts.training_end}")
    print(f"Model: {artifacts.model_path}")
    print(f"Climatology: {artifacts.climatology_path}")
    print(f"Metadata: {artifacts.metadata_path}")

    forecast = predict_latest(paths)

    print("\nLatest forecast")
    print("---------------")
    print(f"Observation month: {forecast.observation_date}")
    print(f"Forecast window: {forecast.target_start} -> {forecast.target_end}")
    print(
        "Kernel Ridge anomaly: "
        f"{forecast.predicted_anomaly_mm_per_month:.2f} mm/month"
    )
    print(
        "3M persistence baseline: "
        f"{forecast.persistence_baseline_mm_per_month:.2f} mm/month"
    )


if __name__ == "__main__":
    main()
