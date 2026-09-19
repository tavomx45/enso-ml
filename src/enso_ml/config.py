"""Project paths and small configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths used by the project."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        return cls(Path(root).expanduser().resolve())

    @classmethod
    def from_package(cls) -> "ProjectPaths":
        """Infer the repository root from src/enso_ml/config.py."""
        return cls(Path(__file__).resolve().parents[2])

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def model_file(self) -> Path:
        return self.models / "kernel_ridge_seasonal.joblib"

    @property
    def climatology_file(self) -> Path:
        return self.models / "monthly_climatology.json"

    @property
    def metadata_file(self) -> Path:
        return self.models / "model_metadata.json"

    def ensure_output_dirs(self) -> None:
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
