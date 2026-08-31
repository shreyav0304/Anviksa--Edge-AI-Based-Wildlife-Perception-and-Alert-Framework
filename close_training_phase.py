"""Create final reports from the three valid completed experiments only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "comparison"
EXPERIMENTS = (
    ("Custom CNN", ROOT / "results/custom_cnn/20260830T142449Z_c23a5d39", 32),
    ("MobileNetV2", ROOT / "results/mobilenet_v2/20260830T161743Z_552c91eb", 32),
    ("MobileNetV3 Large", ROOT / "results/mobilenet_v3_large/20260830T171150Z_94069968", 32),
)

FIELDS = (
    "Model", "Algorithm", "Training Method", "Parameters", "Model Size",
    "Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
    "Weighted Precision", "Weighted Recall", "Weighted F1",
    "Venomous Snake Precision", "Venomous Snake Recall", "Venomous Snake F1",
    "Non-Venomous Snake Precision", "Non-Venomous Snake Recall",
    "Non-Venomous Snake F1", "Snake Macro Recall", "Snake Macro F1",
    "Average Inference Time", "Median Inference Time", "Approximate FPS",
)


def row(name, directory, batch_size):
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return {
        "Model": name, "Algorithm": metrics["algorithm"],
        "Training Method": metrics["training_method"],
        "Parameters": metrics["total_parameters"],
        "Model Size": metrics["saved_model_size_bytes"],
        "Test Accuracy": metrics["test_accuracy"],
        "Macro Precision": metrics["macro_precision"],
        "Macro Recall": metrics["macro_recall"], "Macro F1": metrics["macro_f1"],
        "Weighted Precision": metrics["weighted_precision"],
        "Weighted Recall": metrics["weighted_recall"], "Weighted F1": metrics["weighted_f1"],
        "Venomous Snake Precision": metrics["venomous_snake"]["precision"],
        "Venomous Snake Recall": metrics["venomous_snake"]["recall"],
        "Venomous Snake F1": metrics["venomous_snake"]["f1"],
        "Non-Venomous Snake Precision": metrics["non_venomous_snake"]["precision"],
        "Non-Venomous Snake Recall": metrics["non_venomous_snake"]["recall"],
        "Non-Venomous Snake F1": metrics["non_venomous_snake"]["f1"],
        "Snake Macro Recall": metrics["snake_macro_recall"],
        "Snake Macro F1": metrics["snake_macro_f1"],
        "Average Inference Time": metrics["average_inference_time_ms"],
        "Median Inference Time": metrics["median_inference_time_ms"],
        "Approximate FPS": metrics["approximate_fps"],
        "_batch_size": batch_size,
    }


def graph(rows, field, filename, ylabel, scale=1.0):
    figure, axis = plt.subplots(figsize=(9, 6))
    values = [item[field] * scale for item in rows]
    bars = axis.bar([item["Model"] for item in rows], values)
    axis.set(ylabel=ylabel, title=field)
    axis.grid(axis="y", alpha=0.3)
    axis.bar_label(bars, fmt="%.2f")
    figure.tight_layout(); figure.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [row(*item) for item in EXPERIMENTS]
    with (OUTPUT / "final_valid_model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    status_fields = ("Model", "Status", "Batch 32", "Batch 16", "Test evaluated", "Excluded from quantitative ranking", "Notes")
    status_rows = [
        {"Model": name, "Status": "COMPLETE", "Batch 32": "Used", "Batch 16": "Not applicable", "Test evaluated": "YES", "Excluded from quantitative ranking": "NO", "Notes": str(path)} for name, path, _ in EXPERIMENTS
    ] + [{"Model": "EfficientNetB0", "Status": "RESOURCE-LIMITED / INCOMPLETE", "Batch 32": "Confirmed TensorFlow ResourceExhaustedError", "Batch 16": "Passed diagnostic preflight; training terminated during fine-tuning with Windows access violation under severe memory pressure", "Test evaluated": "NO", "Excluded from quantitative ranking": "YES", "Notes": "Attempted — excluded from quantitative comparison because training could not complete under the available computational resources."}]
    with (OUTPUT / "model_experiment_status.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=status_fields); writer.writeheader(); writer.writerows(status_rows)
    graph(rows, "Test Accuracy", "accuracy_comparison.png", "Accuracy (%)", 100)
    graph(rows, "Macro F1", "macro_f1_comparison.png", "Macro F1 (%)", 100)
    graph(rows, "Snake Macro Recall", "snake_macro_recall_comparison.png", "Snake Macro Recall (%)", 100)
    graph(rows, "Venomous Snake Recall", "venomous_snake_recall_comparison.png", "Venomous Snake Recall (%)", 100)
    graph(rows, "Average Inference Time", "inference_time_comparison.png", "Milliseconds")
    graph(rows, "Model Size", "model_size_comparison.png", "MiB", 1 / 1048576)
    graph(rows, "Parameters", "parameter_count_comparison.png", "Parameters")
    print(OUTPUT)


if __name__ == "__main__":
    main()
