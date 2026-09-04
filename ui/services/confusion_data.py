"""Read-only loading and validation of historical confusion matrices."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .comparison_data import COMPLETED_RUNS


CLASS_NAMES = (
    "Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake",
    "Venomous_Snake", "Wild_Boar",
)
EXPECTED_SNAKE_CONFUSIONS = {
    "Custom CNN": (54, 35),
    "MobileNetV2": (15, 20),
    "MobileNetV3 Large": (11, 8),
}


@dataclass(frozen=True)
class ConfusionMatrixRecord:
    model: str
    raw: np.ndarray
    normalized: np.ndarray
    metrics: dict[str, float]
    total: int
    correct: int
    incorrect: int
    matrix_accuracy: float
    largest_confusion: tuple[str, str, int]
    nonvenomous_to_venomous: int
    venomous_to_nonvenomous: int
    interpretation: str


def _load_matrix(path: Path, value_type: type) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Validated confusion matrix not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if tuple(rows[0][1:]) != CLASS_NAMES or tuple(row[0] for row in rows[1:]) != CLASS_NAMES:
        raise ValueError(f"Class order mismatch in {path}")
    matrix = np.asarray([[value_type(value) for value in row[1:]] for row in rows[1:]])
    if matrix.shape != (7, 7) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid 7×7 matrix in {path}")
    return matrix


def _interpret(model: str, accuracy: float, off_diagonal: int, snake_errors: int) -> str:
    if model == "Custom CNN":
        return f"The diagonal contains {accuracy:.1%} of predictions, with broader errors across several wildlife classes. Snake-type confusion accounts for {snake_errors} of {off_diagonal} errors."
    if model == "MobileNetV2":
        return f"Predictions are strongly concentrated on the diagonal ({accuracy:.1%}), showing substantially improved class separation relative to Custom CNN. Snake-type confusion remains the main error pattern ({snake_errors} cases)."
    return f"Most predictions lie on the main diagonal ({accuracy:.1%}), indicating strong classification performance. Remaining errors are concentrated mainly between venomous and non-venomous snake classes ({snake_errors} cases)."


def load_confusion_records() -> dict[str, ConfusionMatrixRecord]:
    """Load matrices and metrics only; no image access or model execution."""
    records: dict[str, ConfusionMatrixRecord] = {}
    nonvenomous = CLASS_NAMES.index("Non_Venomous_Snake")
    venomous = CLASS_NAMES.index("Venomous_Snake")
    for model, run in COMPLETED_RUNS.items():
        raw = _load_matrix(run / "confusion_matrix_raw.csv", int).astype(np.int64)
        normalized = _load_matrix(run / "confusion_matrix_normalized.csv", float).astype(np.float64)
        total = int(raw.sum())
        correct = int(np.trace(raw))
        incorrect = total - correct
        if total != 545:
            raise ValueError(f"{model} confusion matrix totals {total}, expected 545")
        if not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError(f"{model} normalized matrix rows do not sum to 100%")
        expected_normalized = raw / raw.sum(axis=1, keepdims=True)
        if not np.allclose(normalized, expected_normalized, rtol=0, atol=1e-12):
            raise ValueError(f"{model} normalized matrix does not match validated row normalization")
        with (run / "metrics.json").open(encoding="utf-8") as handle:
            historical = json.load(handle)
        matrix_accuracy = correct / total
        if not np.isclose(matrix_accuracy, historical["test_accuracy"], rtol=0, atol=1e-12):
            raise ValueError(f"{model} matrix accuracy differs from historical accuracy")
        nv_to_v = int(raw[nonvenomous, venomous])
        v_to_nv = int(raw[venomous, nonvenomous])
        if (nv_to_v, v_to_nv) != EXPECTED_SNAKE_CONFUSIONS[model]:
            raise ValueError(f"{model} snake confusion counts differ from validated values")
        off = raw.copy(); np.fill_diagonal(off, 0)
        row, column = np.unravel_index(int(np.argmax(off)), off.shape)
        largest = (CLASS_NAMES[row], CLASS_NAMES[column], int(off[row, column]))
        metrics = {
            "accuracy": float(historical["test_accuracy"]),
            "macro_precision": float(historical["macro_precision"]),
            "macro_recall": float(historical["macro_recall"]),
            "macro_f1": float(historical["macro_f1"]),
            "venomous_recall": float(historical["venomous_snake"]["recall"]),
            "snake_macro_recall": float(historical["snake_macro_recall"]),
        }
        records[model] = ConfusionMatrixRecord(
            model=model, raw=raw, normalized=normalized, metrics=metrics,
            total=total, correct=correct, incorrect=incorrect,
            matrix_accuracy=matrix_accuracy, largest_confusion=largest,
            nonvenomous_to_venomous=nv_to_v,
            venomous_to_nonvenomous=v_to_nv,
            interpretation=_interpret(model, matrix_accuracy, incorrect, nv_to_v + v_to_nv),
        )
    return records
