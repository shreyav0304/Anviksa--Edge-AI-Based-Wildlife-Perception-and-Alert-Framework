"""Training-history and cross-model reporting utilities."""

from __future__ import annotations

import csv
import importlib.util
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

COMPARISON_FIELDS = (
    "model_name",
    "algorithm",
    "training_method",
    "total_parameters",
    "saved_model_size_bytes",
    "test_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "venomous_snake_precision",
    "venomous_snake_recall",
    "venomous_snake_f1",
    "non_venomous_snake_precision",
    "non_venomous_snake_recall",
    "non_venomous_snake_f1",
    "snake_macro_recall",
    "snake_macro_f1",
    "average_inference_time_ms",
    "median_inference_time_ms",
    "approximate_fps",
)


def save_history(history: dict[str, list[float]], output_dir: Path) -> None:
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(history)
    row_count = max((len(values) for values in history.values()), default=0)
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", *fields))
        writer.writeheader()
        for index in range(row_count):
            row = {"epoch": index + 1}
            row.update(
                {
                    field: history[field][index]
                    if index < len(history[field])
                    else ""
                    for field in fields
                }
            )
            writer.writerow(row)

    _save_history_plot(
        history,
        "accuracy",
        "val_accuracy",
        "Accuracy",
        output_dir / "training_accuracy.png",
    )
    _save_history_plot(
        history,
        "loss",
        "val_loss",
        "Loss",
        output_dir / "training_loss.png",
    )


def _save_history_plot(
    history: dict[str, list[float]],
    training_key: str,
    validation_key: str,
    ylabel: str,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to save training-history plots."
        ) from exc

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(history.get(training_key, []), label="Training")
    axis.plot(history.get(validation_key, []), label="Validation")
    stage_flags = history.get("fine_tuning_stage", [])
    if 1 in stage_flags:
        boundary = stage_flags.index(1)
        axis.axvline(
            boundary - 0.5,
            color="black",
            linestyle="--",
            linewidth=1.25,
            label="Fine-tuning begins",
        )
    axis.set(xlabel="Epoch", ylabel=ylabel, title=f"Training and Validation {ylabel}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _flatten_metrics(metrics: dict) -> dict[str, object]:
    venomous = metrics["venomous_snake"]
    non_venomous = metrics["non_venomous_snake"]
    return {
        field: value
        for field, value in {
            **metrics,
            "venomous_snake_precision": venomous["precision"],
            "venomous_snake_recall": venomous["recall"],
            "venomous_snake_f1": venomous["f1"],
            "non_venomous_snake_precision": non_venomous["precision"],
            "non_venomous_snake_recall": non_venomous["recall"],
            "non_venomous_snake_f1": non_venomous["f1"],
        }.items()
        if field in COMPARISON_FIELDS
    }


def create_comparison_report(
    experiment_dirs: Sequence[Path], results_root: Path
) -> Path:
    """Compare real completed experiments; refuse missing result metrics."""
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Matplotlib is required to generate comparison graphs."
        )

    rows: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for directory in experiment_dirs:
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Completed metrics not found: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fingerprints.add(metrics["dataset_fingerprint"])
        rows.append(_flatten_metrics(metrics))

    if len(rows) < 2:
        raise ValueError("At least two completed experiments are required.")
    if len(fingerprints) != 1:
        raise ValueError("Experiments use different dataset fingerprints.")

    identifier = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    output_dir = results_root / "comparison" / identifier
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    best = {
        "highest_test_accuracy": max(rows, key=lambda row: row["test_accuracy"])[
            "model_name"
        ],
        "highest_macro_f1": max(rows, key=lambda row: row["macro_f1"])[
            "model_name"
        ],
        "highest_macro_recall": max(rows, key=lambda row: row["macro_recall"])[
            "model_name"
        ],
        "highest_venomous_snake_recall": max(
            rows, key=lambda row: row["venomous_snake_recall"]
        )["model_name"],
        "highest_snake_macro_recall": max(
            rows, key=lambda row: row["snake_macro_recall"]
        )["model_name"],
        "fastest_inference": min(
            rows, key=lambda row: row["average_inference_time_ms"]
        )["model_name"],
        "smallest_saved_model": min(
            rows, key=lambda row: row["saved_model_size_bytes"]
        )["model_name"],
    }
    (output_dir / "best_models.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )

    graph_specs = (
        ("test_accuracy", "Test Accuracy", "accuracy_comparison.png"),
        ("macro_precision", "Macro Precision", "precision_comparison.png"),
        ("macro_recall", "Macro Recall", "recall_comparison.png"),
        ("macro_f1", "Macro F1", "f1_comparison.png"),
        ("snake_macro_recall", "Snake Macro Recall", "snake_recall_comparison.png"),
        ("average_inference_time_ms", "Inference Time (ms)", "inference_time_comparison.png"),
        ("saved_model_size_bytes", "Model Size (bytes)", "model_size_comparison.png"),
        ("total_parameters", "Parameter Count", "parameter_count_comparison.png"),
    )
    for metric, ylabel, filename in graph_specs:
        _comparison_graph(rows, metric, ylabel, output_dir / filename)
    return output_dir


def _comparison_graph(
    rows: Sequence[dict[str, object]], metric: str, ylabel: str, path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to save model-comparison plots."
        ) from exc

    figure, axis = plt.subplots(figsize=(10, 6))
    names = [str(row["model_name"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    axis.bar(names, values)
    axis.set(ylabel=ylabel, title=f"Model vs {ylabel}")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
