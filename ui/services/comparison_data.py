"""Read and validate existing multi-model comparison artifacts for the GUI."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_CSV = PROJECT_ROOT / "results" / "model_comparison_dashboard" / "model_comparison_metrics.csv"
EXPECTED_FINGERPRINT = "b103634bb6a4a01785cb7c7e3b176223325052da47228dba687edaf43a84c594"
COMPLETED_RUNS = {
    "Custom CNN": PROJECT_ROOT / "results" / "custom_cnn" / "20260830T142449Z_c23a5d39",
    "MobileNetV2": PROJECT_ROOT / "results" / "mobilenet_v2" / "20260830T161743Z_552c91eb",
    "MobileNetV3 Large": PROJECT_ROOT / "results" / "mobilenet_v3_large" / "20260830T171150Z_94069968",
}


@dataclass(frozen=True)
class ModelComparisonRecord:
    model: str
    status: str
    algorithm: str
    accuracy: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    mse: float | None
    mean_confidence: float | None
    median_confidence: float | None
    venomous_recall: float | None
    snake_macro_recall: float | None
    snake_macro_f1: float | None
    model_size_mib: float | None
    inference_time_ms: float | None


def _optional_float(value: str) -> float | None:
    return None if value.strip().upper() == "N/A" else float(value)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_comparison_records() -> tuple[ModelComparisonRecord, ...]:
    """Load validated results only; never run models or scan datasets."""
    if not COMPARISON_CSV.is_file():
        raise FileNotFoundError(f"Validated comparison table not found: {COMPARISON_CSV}")
    with COMPARISON_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_models = (*COMPLETED_RUNS, "EfficientNetB0")
    if tuple(row["Model"] for row in rows) != expected_models:
        raise ValueError("Comparison model order does not match validated artifacts")

    snake_f1: dict[str, float] = {}
    for model, run in COMPLETED_RUNS.items():
        metrics = _read_json(run / "metrics.json")
        config = _read_json(run / "config.json")
        if metrics["dataset_fingerprint"] != EXPECTED_FINGERPRINT or config["dataset_fingerprint"] != EXPECTED_FINGERPRINT:
            raise ValueError(f"Dataset fingerprint mismatch for {model}")
        snake_f1[model] = float(metrics["snake_macro_f1"])

    records: list[ModelComparisonRecord] = []
    for row in rows:
        model = row["Model"]
        if model == "EfficientNetB0":
            numeric_fields = [value for key, value in row.items() if key not in {"Model", "Status", "Algorithm"}]
            if row["Status"] != "INCOMPLETE" or any(value != "N/A" for value in numeric_fields):
                raise ValueError("EfficientNetB0 must remain INCOMPLETE with N/A metrics")
        records.append(ModelComparisonRecord(
            model=model,
            status=row["Status"],
            algorithm=row["Algorithm"],
            accuracy=_optional_float(row["Accuracy"]),
            macro_precision=_optional_float(row["Macro_Precision"]),
            macro_recall=_optional_float(row["Macro_Recall"]),
            macro_f1=_optional_float(row["Macro_F1"]),
            mse=_optional_float(row["MSE"]),
            mean_confidence=_optional_float(row["Mean_Confidence"]),
            median_confidence=_optional_float(row["Median_Confidence"]),
            venomous_recall=_optional_float(row["Venomous_Snake_Recall"]),
            snake_macro_recall=_optional_float(row["Snake_Macro_Recall"]),
            snake_macro_f1=snake_f1.get(model),
            model_size_mib=_optional_float(row["Model_Size_MiB"]),
            inference_time_ms=_optional_float(row["Inference_Time_ms"]),
        ))
    return tuple(records)
