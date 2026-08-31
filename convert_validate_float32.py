"""Convert and validate the selected MobileNetV3 Large as Float32 TFLite."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from experiments.config import ExperimentConfig
from experiments.data_pipeline import build_dataset_bundle
from experiments.evaluator import save_confusion_matrices

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "results/mobilenet_v3_large/20260830T171150Z_94069968"
SOURCE = SOURCE_DIR / "best_model.keras"
OUTPUT = ROOT / "results/deployment/mobilenet_v3_large_float32"
TFLITE_PATH = OUTPUT / "mobilenet_v3_large_float32.tflite"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.dtype, type)):
        return str(value)
    return value


def metrics_for(true, predicted, class_names):
    macro = precision_recall_fscore_support(true, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(true, predicted, average="weighted", zero_division=0)
    per = precision_recall_fscore_support(true, predicted, labels=np.arange(len(class_names)), zero_division=0)
    rows = [{"class": name, "precision": float(per[0][i]), "recall": float(per[1][i]), "f1": float(per[2][i]), "support": int(per[3][i])} for i, name in enumerate(class_names)]
    lookup = {row["class"]: row for row in rows}
    snakes = (lookup["Venomous_Snake"], lookup["Non_Venomous_Snake"])
    return {"test_accuracy": float(accuracy_score(true, predicted)), "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]), "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]), "weighted_f1": float(weighted[2]), "per_class": rows, "venomous_snake": snakes[0], "non_venomous_snake": snakes[1], "snake_macro_recall": statistics.fmean(row["recall"] for row in snakes), "snake_macro_f1": statistics.fmean(row["f1"] for row in snakes)}


def main():
    if OUTPUT.exists() and not TFLITE_PATH.is_file():
        raise FileExistsError(f"Incomplete deployment directory exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256(SOURCE)
    source_size = SOURCE.stat().st_size
    # Batch-one streaming is deliberately used for validation to stay within
    # the development machine's constrained RAM. It does not change images.
    config = ExperimentConfig(model_key="mobilenet_v3_large", batch_size=1)
    datasets = build_dataset_bundle(config, strict_validation=True)
    model = tf.keras.models.load_model(SOURCE, compile=False)
    if model.name != "anviksa_mobilenet_v3_large" or list(model.input_shape[1:]) != [224, 224, 3] or model.output_shape[-1] != 7:
        raise ValueError("Source model identity or tensor specification mismatch.")

    if not TFLITE_PATH.is_file():
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = []
        tflite_bytes = converter.convert()
        TFLITE_PATH.write_bytes(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    tensor_details = {"input": jsonable(input_detail), "output": jsonable(output_detail)}
    (OUTPUT / "tensor_details.json").write_text(json.dumps(tensor_details, indent=2) + "\n", encoding="utf-8")

    all_true, keras_parts, tflite_parts = [], [], []
    benchmark_sample = None
    for batch_images, labels in datasets.test:
        batch = batch_images.numpy().astype(np.float32, copy=False)
        if benchmark_sample is None:
            benchmark_sample = batch[0:1].copy()
        all_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        keras_parts.append(model(batch_images, training=False).numpy())
        for image in batch:
            interpreter.set_tensor(input_detail["index"], image[None, ...])
            interpreter.invoke()
            tflite_parts.append(interpreter.get_tensor(output_detail["index"])[0].copy())
    true = np.asarray(all_true, dtype=np.int64)
    keras_probs = np.concatenate(keras_parts, axis=0)
    tflite_probs = np.asarray(tflite_parts)
    if not np.isfinite(tflite_probs).all() or not np.allclose(tflite_probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("TFLite probabilities failed finite/sum validation.")

    keras_pred = np.argmax(keras_probs, axis=1)
    tflite_pred = np.argmax(tflite_probs, axis=1)
    absolute = np.abs(keras_probs - tflite_probs)
    agreement_rows = []
    for i, entry in enumerate(datasets.test_entries):
        agreement_rows.append({"image": entry.relative_path, "true_class": config.class_names[true[i]], "keras_predicted_class": config.class_names[keras_pred[i]], "tflite_predicted_class": config.class_names[tflite_pred[i]], "keras_confidence": float(keras_probs[i, keras_pred[i]]), "tflite_confidence": float(tflite_probs[i, tflite_pred[i]]), "top1_agree": bool(keras_pred[i] == tflite_pred[i]), "maximum_absolute_probability_difference": float(absolute[i].max()), "mean_absolute_probability_difference": float(absolute[i].mean())})
    with (OUTPUT / "prediction_agreement.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=agreement_rows[0]); writer.writeheader(); writer.writerows(agreement_rows)

    result = metrics_for(true, tflite_pred, config.class_names)
    result["test_loss"] = float(np.mean(-np.log(np.clip(tflite_probs[np.arange(len(true)), true], 1e-7, 1.0))))
    result["dataset_fingerprint"] = config.dataset_fingerprint
    with (OUTPUT / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result["per_class"][0]); writer.writeheader(); writer.writerows(result["per_class"])
    raw = confusion_matrix(true, tflite_pred, labels=np.arange(7))
    normalized = np.divide(raw, raw.sum(axis=1, keepdims=True), out=np.zeros_like(raw, dtype=float), where=raw.sum(axis=1, keepdims=True) != 0)
    save_confusion_matrices(raw, normalized, config.class_names, OUTPUT)
    original_raw = np.load(SOURCE_DIR / "confusion_matrix_raw.npy")
    changed_predictions = int(np.sum(keras_pred != tflite_pred))
    result["confusion_matrix_changed_cells"] = int(np.sum(raw != original_raw))
    result["confusion_matrix_total_absolute_count_difference"] = int(np.abs(raw - original_raw).sum())
    (OUTPUT / "tflite_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    sample = benchmark_sample
    warmups, iterations = 10, 100
    for _ in range(warmups):
        interpreter.set_tensor(input_detail["index"], sample); interpreter.invoke(); interpreter.get_tensor(output_detail["index"])
    timings = []
    for _ in range(iterations):
        started = perf_counter(); interpreter.set_tensor(input_detail["index"], sample); interpreter.invoke(); interpreter.get_tensor(output_detail["index"]); timings.append((perf_counter() - started) * 1000)
    benchmark = {"benchmark_type": "model-only TFLite interpreter latency; resizing/RGB decoding excluded", "num_threads": 1, "warmup_iterations": warmups, "timed_iterations": iterations, "average_inference_time_ms": statistics.fmean(timings), "median_inference_time_ms": statistics.median(timings), "minimum_inference_time_ms": min(timings), "maximum_inference_time_ms": max(timings), "standard_deviation_ms": statistics.pstdev(timings), "approximate_fps": 1000 / statistics.fmean(timings)}
    (OUTPUT / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")

    original = json.loads((SOURCE_DIR / "metrics.json").read_text(encoding="utf-8"))
    compare_fields = ("representation", "size_bytes", "test_accuracy", "macro_f1", "venomous_snake_recall", "snake_macro_recall", "average_inference_time_ms", "median_inference_time_ms", "approximate_fps")
    compare_rows = [{"representation": "Keras", "size_bytes": source_size, "test_accuracy": original["test_accuracy"], "macro_f1": original["macro_f1"], "venomous_snake_recall": original["venomous_snake"]["recall"], "snake_macro_recall": original["snake_macro_recall"], "average_inference_time_ms": original["average_inference_time_ms"], "median_inference_time_ms": original["median_inference_time_ms"], "approximate_fps": original["approximate_fps"]}, {"representation": "Float32 TFLite", "size_bytes": TFLITE_PATH.stat().st_size, "test_accuracy": result["test_accuracy"], "macro_f1": result["macro_f1"], "venomous_snake_recall": result["venomous_snake"]["recall"], "snake_macro_recall": result["snake_macro_recall"], "average_inference_time_ms": benchmark["average_inference_time_ms"], "median_inference_time_ms": benchmark["median_inference_time_ms"], "approximate_fps": benchmark["approximate_fps"]}]
    with (OUTPUT / "keras_vs_tflite_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=compare_fields); writer.writeheader(); writer.writerows(compare_rows)
    metadata = {"source_model": str(SOURCE), "source_sha256_before": source_hash_before, "source_sha256_after": sha256(SOURCE), "source_size_bytes": source_size, "tflite_path": str(TFLITE_PATH), "tflite_sha256": sha256(TFLITE_PATH), "tflite_size_bytes": TFLITE_PATH.stat().st_size, "size_reduction_bytes": source_size - TFLITE_PATH.stat().st_size, "size_reduction_percent": (source_size - TFLITE_PATH.stat().st_size) / source_size * 100, "conversion": "standard Float32; no optimizations or quantization", "class_order": list(config.class_names), "external_preprocessing": "RGB resize nearest-neighbor to 224x224; cast float32; retain approximately [0,255]", "embedded_preprocessing": "MobileNetV3 Rescaling(scale=1/127.5, offset=-1.0)", "images_compared": len(true), "top1_agreement_count": int(np.sum(keras_pred == tflite_pred)), "top1_agreement_percentage": float(np.mean(keras_pred == tflite_pred) * 100), "mean_absolute_probability_difference": float(absolute.mean()), "maximum_observed_probability_difference": float(absolute.max()), "changed_top1_predictions": changed_predictions, "changed_images": [row for row in agreement_rows if not row["top1_agree"]]}
    (OUTPUT / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "agreement": metadata["top1_agreement_percentage"], "accuracy": result["test_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
