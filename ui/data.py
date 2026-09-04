"""Read-only access to validated project artifacts for the UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_RUN = (
    PROJECT_ROOT
    / "results"
    / "mobilenet_v3_large"
    / "20260830T171150Z_94069968"
)
DEPLOYMENT_MANIFEST = PROJECT_ROOT / "pi_deployment" / "model_manifest.json"


@dataclass(frozen=True)
class DashboardData:
    selected_model: str
    deployment_format: str
    classes: tuple[str, ...]
    accuracy: float
    macro_f1: float
    venomous_recall: float
    snake_macro_recall: float
    deployment_candidates: str
    pi_benchmark_status: str


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Validated project artifact not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_dashboard_data() -> DashboardData:
    """Load dashboard values without importing or evaluating an ML model."""
    metrics = _read_json(SELECTED_RUN / "metrics.json")
    deployment = _read_json(DEPLOYMENT_MANIFEST)
    classes = tuple(deployment["class_order"])
    models = deployment["models"]
    representations = " / ".join(model["representation"] for model in models)
    return DashboardData(
        selected_model=str(metrics["model_name"]),
        deployment_format="TensorFlow Lite",
        classes=classes,
        accuracy=float(metrics["test_accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        venomous_recall=float(metrics["venomous_snake"]["recall"]),
        snake_macro_recall=float(metrics["snake_macro_recall"]),
        deployment_candidates=representations,
        pi_benchmark_status=str(deployment["raspberry_pi_benchmark_status"]),
    )
