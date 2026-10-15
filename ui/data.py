"""Read-only access to validated project artifacts for the UI."""

from __future__ import annotations

from dataclasses import dataclass


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
    test_images: int
    correct: int
    mean_confidence: float
    median_confidence: float
    accuracy_interval: tuple[float, float]


def load_dashboard_data() -> DashboardData:
    """Use the selected export's recomputed evaluation, never a historical run."""
    from .services.evaluation import load_active_evaluation
    measured = load_active_evaluation()
    classes = measured["classes"]
    recalls = {row["name"]: row["recall"] for row in measured["per_class"]}
    return DashboardData(
        selected_model="MobileNetV3 Large (15 classes)", deployment_format="TensorFlow Lite Float32",
        classes=classes, accuracy=measured["accuracy"], macro_f1=measured["macro_f1"],
        venomous_recall=recalls["Venomous_Snake"],
        snake_macro_recall=(recalls["Venomous_Snake"]+recalls["Non_Venomous_Snake"])/2,
        deployment_candidates="Float32 / Float16", pi_benchmark_status="Pending hardware benchmarking",
        test_images=measured["total"], correct=measured["correct"],
        mean_confidence=measured["mean_confidence"], median_confidence=measured["median_confidence"],
        accuracy_interval=measured["accuracy_interval"],
    )
