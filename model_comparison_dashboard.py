"""Build the Anviksa AI multi-model comparison dashboard from frozen results.

This script never trains or modifies a model.  It reads the selected historical
experiment artifacts and, when necessary, performs inference on the frozen test
manifest to derive probability MSE and top-1 confidence statistics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "model_comparison_dashboard"
CLASS_NAMES = (
    "Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake",
    "Venomous_Snake", "Wild_Boar",
)
RUNS = {
    "Custom CNN": RESULTS / "custom_cnn" / "20260830T142449Z_c23a5d39",
    "MobileNetV2": RESULTS / "mobilenet_v2" / "20260830T161743Z_552c91eb",
    "MobileNetV3 Large": RESULTS / "mobilenet_v3_large" / "20260830T171150Z_94069968",
}
SLUGS = {
    "Custom CNN": "custom_cnn",
    "MobileNetV2": "mobilenet_v2",
    "MobileNetV3 Large": "mobilenet_v3_large",
}
ALGORITHMS = {
    "Custom CNN": "Convolutional Neural Network trained from scratch",
    "MobileNetV2": "Lightweight CNN using depthwise separable convolution, inverted residuals, transfer learning and fine-tuning",
    "MobileNetV3 Large": "Optimized lightweight CNN using depthwise convolution, inverted residual blocks, squeeze-and-excitation, hard-swish, transfer learning and fine-tuning",
    "EfficientNetB0": "Efficient CNN using MBConv, depthwise separable convolution, squeeze-and-excitation and compound scaling (Status: incomplete experiment)",
}
EXPECTED = {
    "Custom CNN": (0.5045871559633027, 0.5632287401308231, 0.543071099674662, 0.50727303583959),
    "MobileNetV2": (0.9247706422018349, 0.9509036491914015, 0.9535246822194272, 0.9520287395522233),
    "MobileNetV3 Large": (0.9541284403669725, 0.9688672743760461, 0.9686971396837043, 0.9687245232484056),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_raw_matrix(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if tuple(rows[0][1:]) != CLASS_NAMES:
        raise ValueError(f"Class order mismatch in {path}")
    if tuple(row[0] for row in rows[1:]) != CLASS_NAMES:
        raise ValueError(f"Row class order mismatch in {path}")
    return np.asarray([[int(value) for value in row[1:]] for row in rows[1:]], dtype=np.int64)


def artifact_digest(run: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metrics.json", "confusion_matrix_raw.csv", "config.json", "best_model.keras"):
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment artifact missing: {path}")
        stat = path.stat()
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def frozen_test_data(config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    dataset_root = Path(config["dataset_root"])
    metadata_root = Path(config["metadata_root"])
    recorded_classes = tuple(read_json(metadata_root / "class_names.json"))
    if recorded_classes != CLASS_NAMES or tuple(config["class_names"]) != CLASS_NAMES:
        raise ValueError("Frozen class order does not match the required class order")
    fingerprint = (metadata_root / "dataset_fingerprint.sha256").read_text(encoding="utf-8").strip()
    if fingerprint != config["dataset_fingerprint"]:
        raise ValueError("Frozen dataset fingerprint metadata does not match the experiment")
    paths: list[str] = []
    labels: list[int] = []
    lookup = {name: index for index, name in enumerate(CLASS_NAMES)}
    with (metadata_root / "test_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                raise ValueError("The frozen test manifest contains a non-test row")
            image = dataset_root / row["relative_path"]
            if not image.is_file():
                raise FileNotFoundError(f"Frozen test image missing: {image}")
            paths.append(str(image))
            labels.append(lookup[row["class_name"]])
    if len(paths) != 545:
        raise ValueError(f"Frozen test count mismatch: expected 545, found {len(paths)}")
    return paths, np.asarray(labels, dtype=np.int64)


def probabilities_for(name: str, run: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cache = OUTPUT / f"prediction_probabilities_{SLUGS[name]}.npz"
    source = artifact_digest(run)
    paths, labels = frozen_test_data(config)
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if str(saved["source_digest"]) == source and np.array_equal(saved["true_labels"], labels):
            probabilities = saved["probabilities"]
            if probabilities.shape == (len(labels), len(CLASS_NAMES)):
                return probabilities, labels

    import tensorflow as tf

    image_size = tuple(config["image_size"])
    batch_size = int(config["batch_size"])
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    def load_image(path: Any) -> Any:
        content = tf.io.read_file(path)
        image = tf.io.decode_image(content, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.cast(image, tf.float32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print(f"Running inference only: {name} ({len(paths)} frozen test images)")
    model = tf.keras.models.load_model(run / "best_model.keras", compile=False)
    probabilities = np.asarray(model.predict(dataset, verbose=1), dtype=np.float64)
    del model
    if probabilities.shape != (len(labels), len(CLASS_NAMES)):
        raise ValueError(f"Unexpected probability shape for {name}: {probabilities.shape}")
    np.savez_compressed(cache, probabilities=probabilities, true_labels=labels, source_digest=np.asarray(source))
    return probabilities, labels


def validate_and_analyze(name: str, run: Path) -> dict[str, Any]:
    metrics = read_json(run / "metrics.json")
    config = read_json(run / "config.json")
    expected = EXPECTED[name]
    recorded = tuple(float(metrics[key]) for key in ("test_accuracy", "macro_precision", "macro_recall", "macro_f1"))
    if not np.allclose(recorded, expected, rtol=0, atol=1e-12):
        raise ValueError(f"Historical metric discrepancy for {name}: {recorded} != {expected}")
    probabilities, labels = probabilities_for(name, run, config)
    predicted = np.argmax(probabilities, axis=1)
    macro = precision_recall_fscore_support(labels, predicted, average="macro", zero_division=0)
    recalculated = (accuracy_score(labels, predicted), macro[0], macro[1], macro[2])
    if not np.allclose(recalculated, recorded, rtol=0, atol=1e-12):
        raise ValueError(f"Recalculated metric discrepancy for {name}: {recalculated} != {recorded}")
    calculated_cm = confusion_matrix(labels, predicted, labels=np.arange(len(CLASS_NAMES)))
    historical_cm = load_raw_matrix(run / "confusion_matrix_raw.csv")
    if not np.array_equal(calculated_cm, historical_cm):
        raise ValueError(f"Recalculated confusion matrix differs for {name}")
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels]
    confidence = probabilities.max(axis=1)
    return {
        **metrics,
        "mse": float(np.mean(np.square(one_hot - probabilities))),
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "raw_cm": historical_cm,
        "normalized_cm": np.divide(historical_cm, historical_cm.sum(axis=1, keepdims=True), dtype=np.float64),
    }


def style_axis(axis: Any, ylabel: str, title: str) -> None:
    axis.set_ylabel(ylabel, fontweight="bold")
    axis.set_title(title, fontsize=15, fontweight="bold", pad=15)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22)


def label_bars(axis: Any, percent: bool = True) -> None:
    fmt = "%.2f%%" if percent else "%.5f"
    for container in axis.containers:
        axis.bar_label(container, fmt=fmt, padding=3, fontsize=8)


def save_charts(data: dict[str, dict[str, Any]]) -> None:
    names = list(RUNS)
    colors = ["#2463a6", "#1d9a8a", "#e08b2c", "#8d5bb7"]
    values = np.asarray([[data[n][k] * 100 for k in ("test_accuracy", "macro_precision", "macro_recall", "macro_f1")] for n in names])
    fig, ax = plt.subplots(figsize=(12, 7)); x = np.arange(len(names)); width = .19
    for index, label in enumerate(("Accuracy", "Macro Precision", "Macro Recall", "Macro F1")):
        ax.bar(x + (index - 1.5) * width, values[:, index], width, label=label, color=colors[index])
    ax.set_xticks(x, names); ax.set_ylim(0, 108); ax.legend(ncols=4, loc="upper center")
    style_axis(ax, "Score (%)", "Anvīkṣa AI — Model Performance Comparison"); label_bars(ax)
    fig.tight_layout(); fig.savefig(OUTPUT / "model_metrics_comparison.png", dpi=300); plt.close(fig)

    specs = [
        ("accuracy_comparison.png", ("test_accuracy",), ("Accuracy",), "Accuracy (%)", "Model Accuracy Comparison", True),
        ("f1_score_comparison.png", ("macro_f1",), ("Macro F1",), "Macro F1 (%)", "Macro F1 Score Comparison", True),
        ("precision_recall_comparison.png", ("macro_precision", "macro_recall"), ("Macro Precision", "Macro Recall"), "Score (%)", "Macro Precision and Recall Comparison", True),
        ("confidence_comparison.png", ("mean_confidence", "median_confidence"), ("Mean Prediction Confidence", "Median Prediction Confidence"), "Confidence (%)", "Prediction Confidence Comparison", True),
        ("mse_comparison.png", ("mse",), ("Probability MSE (lower is better)",), "Mean Squared Error", "Classification Probability MSE Comparison", False),
    ]
    for filename, keys, labels, ylabel, title, percent in specs:
        fig, ax = plt.subplots(figsize=(10, 6)); x = np.arange(len(names)); width = .72 / len(keys)
        for index, (key, label) in enumerate(zip(keys, labels, strict=True)):
            vals = [data[n][key] * (100 if percent else 1) for n in names]
            ax.bar(x + (index - (len(keys)-1)/2) * width, vals, width, label=label, color=colors[index])
        ax.set_xticks(x, names); ax.legend() if len(keys) > 1 else None
        if percent: ax.set_ylim(0, 108)
        style_axis(ax, ylabel, title); label_bars(ax, percent=percent)
        fig.tight_layout(); fig.savefig(OUTPUT / filename, dpi=300); plt.close(fig)


def save_confusion_plots(data: dict[str, dict[str, Any]]) -> None:
    display_labels = [name.replace("_", "\n") for name in CLASS_NAMES]
    for name in RUNS:
        for normalized in (False, True):
            matrix = data[name]["normalized_cm" if normalized else "raw_cm"]
            fig, ax = plt.subplots(figsize=(10.5, 8.5)); image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
            fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
            ax.set(xticks=np.arange(7), yticks=np.arange(7), xticklabels=display_labels, yticklabels=display_labels,
                   xlabel="Predicted class", ylabel="Actual class", title=f"{name} — {'Row-Normalized Confusion Matrix (%)' if normalized else 'Raw Confusion Matrix'}")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            threshold = float(matrix.max()) / 2
            for row in range(7):
                for column in range(7):
                    text = f"{matrix[row, column] * 100:.1f}%" if normalized else str(int(matrix[row, column]))
                    ax.text(column, row, text, ha="center", va="center", fontsize=8, color="white" if matrix[row, column] > threshold else "#172033")
            fig.tight_layout()
            suffix = "_normalized" if normalized else ""
            fig.savefig(OUTPUT / f"confusion_matrix_{SLUGS[name]}{suffix}.png", dpi=300); plt.close(fig)


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.10f}".rstrip("0").rstrip(".")


def save_table_and_summary(data: dict[str, dict[str, Any]]) -> None:
    fields = ["Model", "Status", "Algorithm", "Parameters", "Model_Size_MiB", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1", "MSE", "Mean_Confidence", "Median_Confidence", "Venomous_Snake_Recall", "Non_Venomous_Snake_Recall", "Snake_Macro_Recall", "Inference_Time_ms", "FPS"]
    rows = []
    for name in RUNS:
        m = data[name]
        rows.append({"Model": name, "Status": "COMPLETED", "Algorithm": ALGORITHMS[name], "Parameters": m["total_parameters"], "Model_Size_MiB": fmt(m["saved_model_size_bytes"] / 1048576), "Accuracy": fmt(m["test_accuracy"]), "Macro_Precision": fmt(m["macro_precision"]), "Macro_Recall": fmt(m["macro_recall"]), "Macro_F1": fmt(m["macro_f1"]), "Weighted_F1": fmt(m["weighted_f1"]), "MSE": fmt(m["mse"]), "Mean_Confidence": fmt(m["mean_confidence"]), "Median_Confidence": fmt(m["median_confidence"]), "Venomous_Snake_Recall": fmt(m["venomous_snake"]["recall"]), "Non_Venomous_Snake_Recall": fmt(m["non_venomous_snake"]["recall"]), "Snake_Macro_Recall": fmt(m["snake_macro_recall"]), "Inference_Time_ms": fmt(m["average_inference_time_ms"]), "FPS": fmt(m["approximate_fps"])})
    rows.append({field: ("EfficientNetB0" if field == "Model" else "INCOMPLETE" if field == "Status" else ALGORITHMS["EfficientNetB0"] if field == "Algorithm" else "N/A") for field in fields})
    with (OUTPUT / "model_comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    best = lambda key, reverse=True: sorted(RUNS, key=lambda n: data[n][key], reverse=reverse)[0]
    summary = ["ANVĪKṢA AI — MULTI-MODEL COMPARISON SUMMARY", "", "Completed: Custom CNN, MobileNetV2, MobileNetV3 Large", "Incomplete / N.A.: EfficientNetB0", "", "Classification performance:", f"Highest Accuracy: {best('test_accuracy')}", f"Highest Precision (macro): {best('macro_precision')}", f"Highest Recall (macro): {best('macro_recall')}", f"Highest F1 (macro): {best('macro_f1')}", f"Lowest probability MSE: {best('mse', False)}", f"Highest Venomous Snake Recall: {max(RUNS, key=lambda n: data[n]['venomous_snake']['recall'])}", f"Highest Snake Macro Recall: {best('snake_macro_recall')}", "", "Deployment efficiency (recorded historical measurements):", f"Smallest model: {min(RUNS, key=lambda n: data[n]['saved_model_size_bytes'])}", f"Fastest recorded inference: {min(RUNS, key=lambda n: data[n]['average_inference_time_ms'])}", "", "Classification performance and deployment efficiency are separate considerations.", "EfficientNetB0 is excluded from rankings because its controlled experiment is incomplete."]
    (OUTPUT / "model_comparison_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        data = {name: validate_and_analyze(name, run) for name, run in RUNS.items()}
        save_charts(data); save_confusion_plots(data); save_table_and_summary(data)
        print(json.dumps({name: {key: data[name][key] for key in ("test_accuracy", "macro_precision", "macro_recall", "macro_f1", "mse", "mean_confidence", "median_confidence")} for name in RUNS}, indent=2))
        print(f"Dashboard created: {OUTPUT}")
        print("MODEL METRIC VALIDATION: PASS")
        print("CONFUSION MATRIX VALIDATION: PASS")
        return 0
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("MODEL METRIC VALIDATION: FAIL", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
