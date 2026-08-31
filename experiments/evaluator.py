"""Shared evaluation and efficiency measurement for all registered models."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import ExperimentConfig
from .model_registry import ModelSpec


def _parameter_counts(model: tf.keras.Model) -> dict[str, int]:
    trainable = sum(int(tf.size(weight)) for weight in model.trainable_weights)
    non_trainable = sum(
        int(tf.size(weight)) for weight in model.non_trainable_weights
    )
    return {
        "total_parameters": model.count_params(),
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
    }


def benchmark_inference(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_images: int = 100,
) -> dict[str, float | int]:
    """Benchmark batch-one warm inference under repeatable conditions."""
    single_images = dataset.unbatch().batch(1).take(max_images)
    iterator = iter(single_images)
    try:
        warmup_images, _ = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot benchmark an empty dataset.") from exc
    model(warmup_images, training=False)

    timings: list[float] = []
    measured = 0
    for images, _ in single_images:
        started = perf_counter()
        model(images, training=False)
        timings.append((perf_counter() - started) * 1000.0)
        measured += int(images.shape[0])

    average = statistics.fmean(timings)
    median = statistics.median(timings)
    return {
        "benchmark_images": measured,
        "average_inference_time_ms": average,
        "median_inference_time_ms": median,
        "approximate_fps": 1000.0 / average if average > 0 else 0.0,
    }


def compute_confusion_matrices(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(num_classes)
    raw = confusion_matrix(true_labels, predicted_labels, labels=labels)
    row_totals = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw,
        row_totals,
        out=np.zeros_like(raw, dtype=np.float64),
        where=row_totals != 0,
    )
    return raw, normalized


def _save_matrix_csv(
    matrix: np.ndarray, class_names: tuple[str, ...], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix, strict=True):
            writer.writerow([class_name, *row.tolist()])


def _save_matrix_plot(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    path: Path,
    normalized: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to save confusion-matrix plots."
        ) from exc

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual class",
        xlabel="Predicted class",
        title=(
            "Normalized Confusion Matrix"
            if normalized
            else "Raw Confusion Matrix"
        ),
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    value_format = ".2f" if normalized else "d"
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrices(
    raw: np.ndarray,
    normalized: np.ndarray,
    class_names: tuple[str, ...],
    output_dir: Path,
) -> None:
    _save_matrix_csv(raw, class_names, output_dir / "confusion_matrix_raw.csv")
    _save_matrix_csv(
        normalized,
        class_names,
        output_dir / "confusion_matrix_normalized.csv",
    )
    np.save(output_dir / "confusion_matrix_raw.npy", raw)
    np.save(output_dir / "confusion_matrix_normalized.npy", normalized)
    _save_matrix_plot(
        raw,
        class_names,
        output_dir / "confusion_matrix_raw.png",
        normalized=False,
    )
    _save_matrix_plot(
        normalized,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        normalized=True,
    )


def evaluate_model(
    model: tf.keras.Model,
    spec: ModelSpec,
    test_dataset: tf.data.Dataset,
    config: ExperimentConfig,
    output_dir: Path,
    saved_model_path: Path,
    benchmark_images: int = 100,
) -> dict[str, object]:
    """Evaluate one accepted checkpoint once against the frozen test set."""
    test_loss, keras_accuracy = model.evaluate(test_dataset, verbose=1)
    probabilities = model.predict(test_dataset, verbose=1)
    predicted = np.argmax(probabilities, axis=1)
    true_labels = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in test_dataset]
    )

    macro = precision_recall_fscore_support(
        true_labels, predicted, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        true_labels, predicted, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        true_labels,
        predicted,
        labels=np.arange(len(config.class_names)),
        zero_division=0,
    )
    report_text = classification_report(
        true_labels,
        predicted,
        labels=np.arange(len(config.class_names)),
        target_names=config.class_names,
        digits=4,
        zero_division=0,
    )

    per_class_rows: list[dict[str, object]] = []
    for index, class_name in enumerate(config.class_names):
        per_class_rows.append(
            {
                "class": class_name,
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
        )
    class_lookup = {row["class"]: row for row in per_class_rows}
    snake_rows = [
        class_lookup["Venomous_Snake"],
        class_lookup["Non_Venomous_Snake"],
    ]

    raw_cm, normalized_cm = compute_confusion_matrices(
        true_labels, predicted, len(config.class_names)
    )
    save_confusion_matrices(raw_cm, normalized_cm, config.class_names, output_dir)

    benchmark = benchmark_inference(model, test_dataset, benchmark_images)
    model_size = saved_model_path.stat().st_size if saved_model_path.is_file() else 0
    metrics: dict[str, object] = {
        "model_key": spec.key,
        "model_name": spec.display_name,
        "algorithm": spec.algorithm,
        "training_method": spec.training_method,
        "dataset_fingerprint": config.dataset_fingerprint,
        "test_loss": float(test_loss),
        "keras_test_accuracy": float(keras_accuracy),
        "test_accuracy": float(accuracy_score(true_labels, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "venomous_snake": class_lookup["Venomous_Snake"],
        "non_venomous_snake": class_lookup["Non_Venomous_Snake"],
        "snake_macro_recall": float(
            statistics.fmean(float(row["recall"]) for row in snake_rows)
        ),
        "snake_macro_f1": float(
            statistics.fmean(float(row["f1"]) for row in snake_rows)
        ),
        "per_class": per_class_rows,
        "saved_model_size_bytes": model_size,
        **_parameter_counts(model),
        **benchmark,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    with (output_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=per_class_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_class_rows)
    return metrics
