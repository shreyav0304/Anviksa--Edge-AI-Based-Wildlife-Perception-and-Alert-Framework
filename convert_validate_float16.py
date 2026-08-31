"""Convert and validate the selected MobileNetV3 Large as Float16 TFLite.

This is a deployment-only workflow: it never compiles, trains, or updates the
source model. Validation streams the frozen test manifest one image at a time
to keep memory use bounded on native Windows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from experiments.config import ExperimentConfig, environment_metadata
from experiments.data_pipeline import build_dataset_bundle, validate_clean_dataset
from experiments.evaluator import save_confusion_matrices


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "results/mobilenet_v3_large/20260830T171150Z_94069968"
SOURCE = SOURCE_DIR / "best_model.keras"
FLOAT32_DIR = ROOT / "results/deployment/mobilenet_v3_large_float32"
FLOAT32_PATH = FLOAT32_DIR / "mobilenet_v3_large_float32.tflite"
OUTPUT = ROOT / "results/deployment/mobilenet_v3_large_float16"
FLOAT16_PATH = OUTPUT / "mobilenet_v3_large_float16.tflite"

EXPECTED_SOURCE_SHA256 = "dc89f6b2c39a1206d5c27de218a1a3664a2ea986efe8c2cbddc54091a2661e14"
EXPECTED_FLOAT32_SHA256 = "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.dtype, type)):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def metrics_for(true: np.ndarray, predicted: np.ndarray, class_names: tuple[str, ...]) -> dict:
    macro = precision_recall_fscore_support(true, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(true, predicted, average="weighted", zero_division=0)
    per = precision_recall_fscore_support(
        true, predicted, labels=np.arange(len(class_names)), zero_division=0
    )
    rows = [
        {
            "class": name,
            "precision": float(per[0][index]),
            "recall": float(per[1][index]),
            "f1": float(per[2][index]),
            "support": int(per[3][index]),
        }
        for index, name in enumerate(class_names)
    ]
    lookup = {row["class"]: row for row in rows}
    snakes = (lookup["Venomous_Snake"], lookup["Non_Venomous_Snake"])
    return {
        "test_accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": rows,
        "venomous_snake": snakes[0],
        "non_venomous_snake": snakes[1],
        "snake_macro_recall": float(statistics.fmean(row["recall"] for row in snakes)),
        "snake_macro_f1": float(statistics.fmean(row["f1"] for row in snakes)),
    }


def interpreter_info(interpreter: tf.lite.Interpreter, input_detail: dict, output_detail: dict) -> dict:
    delegates = getattr(interpreter, "_delegates", [])
    return {
        "runtime": "tf.lite.Interpreter",
        "tensorflow_version": tf.__version__,
        "num_threads": 1,
        "delegate_configuration": (
            "Default TensorFlow Lite delegates enabled; XNNPACK is selected by the runtime "
            "when supported. No explicit delegate was loaded."
        ),
        "explicit_delegate_objects": [type(delegate).__name__ for delegate in delegates],
        "input": jsonable(input_detail),
        "output": jsonable(output_detail),
    }


def invoke(interpreter: tf.lite.Interpreter, input_detail: dict, output_detail: dict, image: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_detail["index"], image[None, ...])
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"])[0].copy()


def benchmark(interpreter, input_detail, output_detail, sample: np.ndarray) -> dict:
    warmups, iterations = 10, 100
    for _ in range(warmups):
        invoke(interpreter, input_detail, output_detail, sample[0])
    timings = []
    for _ in range(iterations):
        started = perf_counter()
        invoke(interpreter, input_detail, output_detail, sample[0])
        timings.append((perf_counter() - started) * 1000.0)
    average = statistics.fmean(timings)
    return {
        "benchmark_type": "model-only TFLite interpreter latency; resizing/RGB decoding excluded",
        "platform": "Windows CPU",
        "interpreter": "tf.lite.Interpreter with default XNNPACK delegate",
        "num_threads": 1,
        "batch_size": 1,
        "warmup_iterations": warmups,
        "timed_iterations": iterations,
        "representative_input": "first frozen-test image, identical method to Float32 benchmark",
        "average_inference_time_ms": average,
        "median_inference_time_ms": statistics.median(timings),
        "minimum_inference_time_ms": min(timings),
        "maximum_inference_time_ms": max(timings),
        "standard_deviation_ms": statistics.pstdev(timings),
        "approximate_fps": 1000.0 / average,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite deployment output: {OUTPUT}")
    if not SOURCE.is_file() or not FLOAT32_PATH.is_file():
        raise FileNotFoundError("Required Keras or validated Float32 source artifact is missing.")

    source_hash_before = sha256(SOURCE)
    float32_hash_before = sha256(FLOAT32_PATH)
    if source_hash_before != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Source integrity failure: {source_hash_before}")
    if float32_hash_before != EXPECTED_FLOAT32_SHA256:
        raise RuntimeError(f"Float32 baseline integrity failure: {float32_hash_before}")

    config = ExperimentConfig(model_key="mobilenet_v3_large", batch_size=1)
    datasets = build_dataset_bundle(config, strict_validation=True)
    model = tf.keras.models.load_model(SOURCE, compile=False)
    if (
        model.name != "anviksa_mobilenet_v3_large"
        or list(model.input_shape[1:]) != [224, 224, 3]
        or model.output_shape[-1] != 7
    ):
        raise ValueError("Source model identity or tensor specification mismatch.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    FLOAT16_PATH.write_bytes(tflite_bytes)

    float32 = tf.lite.Interpreter(model_path=str(FLOAT32_PATH), num_threads=1)
    float16 = tf.lite.Interpreter(model_path=str(FLOAT16_PATH), num_threads=1)
    float32.allocate_tensors()
    float16.allocate_tensors()
    f32_in, f32_out = float32.get_input_details()[0], float32.get_output_details()[0]
    f16_in, f16_out = float16.get_input_details()[0], float16.get_output_details()[0]
    if f16_out["shape"].tolist() != [1, 7]:
        raise ValueError(f"Float16 output is not seven values: {f16_out['shape']}")
    if f16_in["dtype"] != np.float32 or f16_out["dtype"] != np.float32:
        raise ValueError("Float16 external input/output interface is not float32.")
    tensor_payload = interpreter_info(float16, f16_in, f16_out)
    tensor_payload["float32_baseline_interface"] = interpreter_info(float32, f32_in, f32_out)
    (OUTPUT / "tensor_details.json").write_text(
        json.dumps(tensor_payload, indent=2) + "\n", encoding="utf-8"
    )

    true_parts, keras_parts, f32_parts, f16_parts = [], [], [], []
    benchmark_sample = None
    for images, labels in datasets.test:
        batch = images.numpy().astype(np.float32, copy=False)
        if benchmark_sample is None:
            benchmark_sample = batch[0:1].copy()
        true_parts.append(int(np.argmax(labels.numpy()[0])))
        keras_parts.append(model(images, training=False).numpy()[0].copy())
        f32_parts.append(invoke(float32, f32_in, f32_out, batch[0]))
        f16_parts.append(invoke(float16, f16_in, f16_out, batch[0]))

    true = np.asarray(true_parts, dtype=np.int64)
    keras_probs = np.asarray(keras_parts)
    f32_probs = np.asarray(f32_parts)
    f16_probs = np.asarray(f16_parts)
    if len(true) != 545:
        raise ValueError(f"Frozen test count changed: {len(true)}")
    for name, probabilities in (("Keras", keras_probs), ("Float32", f32_probs), ("Float16", f16_probs)):
        if probabilities.shape != (545, 7):
            raise ValueError(f"{name} probability shape is {probabilities.shape}, expected (545, 7).")
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{name} produced non-finite values.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"{name} probability sums are invalid.")

    keras_pred = np.argmax(keras_probs, axis=1)
    f32_pred = np.argmax(f32_probs, axis=1)
    f16_pred = np.argmax(f16_probs, axis=1)
    k_f16_diff = np.abs(keras_probs - f16_probs)
    f32_f16_diff = np.abs(f32_probs - f16_probs)
    rows = []
    for index, entry in enumerate(datasets.test_entries):
        rows.append(
            {
                "image": entry.relative_path,
                "true_class": config.class_names[true[index]],
                "keras_predicted_class": config.class_names[keras_pred[index]],
                "float32_predicted_class": config.class_names[f32_pred[index]],
                "float16_predicted_class": config.class_names[f16_pred[index]],
                "keras_confidence": float(keras_probs[index, keras_pred[index]]),
                "float32_confidence": float(f32_probs[index, f32_pred[index]]),
                "float16_confidence": float(f16_probs[index, f16_pred[index]]),
                "keras_vs_float16_top1_agree": bool(keras_pred[index] == f16_pred[index]),
                "float32_vs_float16_top1_agree": bool(f32_pred[index] == f16_pred[index]),
                "keras_vs_float16_mean_absolute_probability_difference": float(k_f16_diff[index].mean()),
                "keras_vs_float16_maximum_absolute_probability_difference": float(k_f16_diff[index].max()),
                "float32_vs_float16_mean_absolute_probability_difference": float(f32_f16_diff[index].mean()),
                "float32_vs_float16_maximum_absolute_probability_difference": float(f32_f16_diff[index].max()),
            }
        )
    write_csv(OUTPUT / "prediction_agreement_float16.csv", rows)

    result = metrics_for(true, f16_pred, config.class_names)
    result["test_loss"] = float(
        np.mean(-np.log(np.clip(f16_probs[np.arange(len(true)), true], 1e-7, 1.0)))
    )
    result["dataset_fingerprint"] = config.dataset_fingerprint
    write_csv(OUTPUT / "per_class_metrics.csv", result["per_class"])

    raw = confusion_matrix(true, f16_pred, labels=np.arange(7))
    normalized = np.divide(
        raw,
        raw.sum(axis=1, keepdims=True),
        out=np.zeros_like(raw, dtype=float),
        where=raw.sum(axis=1, keepdims=True) != 0,
    )
    save_confusion_matrices(raw, normalized, config.class_names, OUTPUT)
    keras_raw = np.load(SOURCE_DIR / "confusion_matrix_raw.npy")
    float32_raw = np.load(FLOAT32_DIR / "confusion_matrix_raw.npy")
    result["total_correct"] = int(np.trace(raw))
    result["total_errors"] = int(raw.sum() - np.trace(raw))
    result["snake_type_confusions"] = {
        "non_venomous_as_venomous": int(raw[4, 5]),
        "venomous_as_non_venomous": int(raw[5, 4]),
    }
    result["confusion_comparison"] = {
        "versus_keras_changed_cells": int(np.sum(raw != keras_raw)),
        "versus_keras_total_absolute_count_difference": int(np.abs(raw - keras_raw).sum()),
        "versus_float32_changed_cells": int(np.sum(raw != float32_raw)),
        "versus_float32_total_absolute_count_difference": int(np.abs(raw - float32_raw).sum()),
        "changed_cells_versus_float32": [
            {
                "actual": config.class_names[row],
                "predicted": config.class_names[column],
                "float32_count": int(float32_raw[row, column]),
                "float16_count": int(raw[row, column]),
                "difference": int(raw[row, column] - float32_raw[row, column]),
            }
            for row, column in np.argwhere(raw != float32_raw)
        ],
    }
    (OUTPUT / "tflite_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    benchmark_result = benchmark(float16, f16_in, f16_out, benchmark_sample)
    (OUTPUT / "benchmark.json").write_text(
        json.dumps(benchmark_result, indent=2) + "\n", encoding="utf-8"
    )

    keras_metrics = json.loads((SOURCE_DIR / "metrics.json").read_text(encoding="utf-8"))
    f32_metrics = json.loads((FLOAT32_DIR / "tflite_metrics.json").read_text(encoding="utf-8"))
    f32_benchmark = json.loads((FLOAT32_DIR / "benchmark.json").read_text(encoding="utf-8"))
    source_size = SOURCE.stat().st_size
    f32_size = FLOAT32_PATH.stat().st_size
    f16_size = FLOAT16_PATH.stat().st_size
    comparison_fields = (
        "representation", "file_size_bytes", "file_size_mib", "test_accuracy",
        "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "venomous_snake_recall", "non_venomous_snake_recall",
        "snake_macro_recall", "snake_macro_f1", "average_inference_time_ms",
        "median_inference_time_ms", "approximate_fps", "latency_runtime_note",
    )

    def comparison_row(name, size, metrics, timing, runtime_note):
        return {
            "representation": name,
            "file_size_bytes": size,
            "file_size_mib": size / (1024**2),
            "test_accuracy": metrics["test_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "venomous_snake_recall": metrics["venomous_snake"]["recall"],
            "non_venomous_snake_recall": metrics["non_venomous_snake"]["recall"],
            "snake_macro_recall": metrics["snake_macro_recall"],
            "snake_macro_f1": metrics["snake_macro_f1"],
            "average_inference_time_ms": timing["average_inference_time_ms"],
            "median_inference_time_ms": timing["median_inference_time_ms"],
            "approximate_fps": timing["approximate_fps"],
            "latency_runtime_note": runtime_note,
        }

    comparison_rows = [
        comparison_row("Keras", source_size, keras_metrics, keras_metrics, "Different Keras runtime benchmark methodology; not directly equivalent."),
        comparison_row("Float32 TFLite", f32_size, f32_metrics, f32_benchmark, "TFLite, XNNPACK, 1 thread, batch 1, model-only."),
        comparison_row("Float16 TFLite", f16_size, result, benchmark_result, "TFLite, XNNPACK, 1 thread, batch 1, model-only."),
    ]
    write_csv(
        OUTPUT / "keras_float32_float16_comparison.csv",
        comparison_rows,
        comparison_fields,
    )

    k_agree = keras_pred == f16_pred
    f32_agree = f32_pred == f16_pred
    changed_indices = np.flatnonzero(~k_agree | ~f32_agree)
    changed_images = [rows[index] for index in changed_indices]
    snake_changed = [
        row for row in changed_images
        if row["true_class"] in {"Venomous_Snake", "Non_Venomous_Snake"}
        or row["keras_predicted_class"] in {"Venomous_Snake", "Non_Venomous_Snake"}
        or row["float32_predicted_class"] in {"Venomous_Snake", "Non_Venomous_Snake"}
        or row["float16_predicted_class"] in {"Venomous_Snake", "Non_Venomous_Snake"}
    ]
    # Acceptance requires preserved safety metrics, no more than one changed
    # prediction, and a material storage reduction. This makes the decision
    # evidence-based without claiming Raspberry Pi speed before device tests.
    accepted = (
        int(np.sum(~f32_agree)) <= 1
        and result["test_accuracy"] >= f32_metrics["test_accuracy"] - (1 / 545)
        and result["macro_f1"] >= f32_metrics["macro_f1"] - 0.002
        and result["venomous_snake"]["recall"] >= f32_metrics["venomous_snake"]["recall"]
        and result["snake_macro_recall"] >= f32_metrics["snake_macro_recall"] - (1 / 256)
        and f16_size < f32_size
    )
    dataset_after = validate_clean_dataset(config, strict=True)
    source_hash_after = sha256(SOURCE)
    float32_hash_after = sha256(FLOAT32_PATH)
    if source_hash_after != source_hash_before or float32_hash_after != float32_hash_before:
        raise RuntimeError("A protected model artifact changed during validation.")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(),
        "source_model": str(SOURCE.resolve()),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "float32_baseline": str(FLOAT32_PATH.resolve()),
        "float32_sha256_before": float32_hash_before,
        "float32_sha256_after": float32_hash_after,
        "float16_path": str(FLOAT16_PATH.resolve()),
        "float16_sha256": sha256(FLOAT16_PATH),
        "conversion": {
            "source": "original selected Keras model (not the Float32 TFLite model)",
            "optimizations": ["tf.lite.Optimize.DEFAULT"],
            "supported_types": ["tf.float16"],
            "representative_dataset": None,
            "integer_input_output": False,
            "training_performed": False,
        },
        "class_order": list(config.class_names),
        "dataset_fingerprint_before": config.dataset_fingerprint,
        "dataset_fingerprint_after": dataset_after["fingerprint"],
        "test_images": len(true),
        "external_preprocessing": "RGB decode; nearest-neighbor resize to 224x224; float32; retain approximately [0,255]",
        "embedded_preprocessing": "MobileNetV3 Rescaling(scale=1/127.5, offset=-1.0)",
        "agreement": {
            "keras_vs_float16_count": int(np.sum(k_agree)),
            "keras_vs_float16_percentage": float(np.mean(k_agree) * 100),
            "keras_vs_float16_mean_absolute_probability_difference": float(k_f16_diff.mean()),
            "keras_vs_float16_maximum_absolute_probability_difference": float(k_f16_diff.max()),
            "keras_vs_float16_changed_top1_predictions": int(np.sum(~k_agree)),
            "float32_vs_float16_count": int(np.sum(f32_agree)),
            "float32_vs_float16_percentage": float(np.mean(f32_agree) * 100),
            "float32_vs_float16_mean_absolute_probability_difference": float(f32_f16_diff.mean()),
            "float32_vs_float16_maximum_absolute_probability_difference": float(f32_f16_diff.max()),
            "float32_vs_float16_changed_top1_predictions": int(np.sum(~f32_agree)),
            "all_disagreements": changed_images,
            "snake_related_disagreements": snake_changed,
        },
        "storage": {
            "keras_bytes": source_size,
            "keras_mib": source_size / (1024**2),
            "float32_bytes": f32_size,
            "float32_mib": f32_size / (1024**2),
            "float16_bytes": f16_size,
            "float16_mib": f16_size / (1024**2),
            "float16_reduction_vs_keras_bytes": source_size - f16_size,
            "float16_reduction_vs_keras_percent": (source_size - f16_size) / source_size * 100,
            "float16_reduction_vs_float32_bytes": f32_size - f16_size,
            "float16_reduction_vs_float32_percent": (f32_size - f16_size) / f32_size * 100,
        },
        "acceptance_rule": "At most 1 Float32 top-1 change; accuracy loss <=1/545; macro-F1 loss <=0.002; no venomous-recall loss; snake-macro-recall loss <=1/256; Float16 smaller than Float32.",
        "deployment_decision": "ACCEPTED FOR RASPBERRY PI BENCHMARKING" if accepted else "REJECTED FOR RASPBERRY PI BENCHMARKING",
        "raspberry_pi_performance_claim": "None; device performance has not yet been measured.",
        "integrity": {
            "keras_unchanged": source_hash_after == source_hash_before == EXPECTED_SOURCE_SHA256,
            "float32_unchanged": float32_hash_after == float32_hash_before == EXPECTED_FLOAT32_SHA256,
            "dataset_unchanged": dataset_after["fingerprint"] == config.dataset_fingerprint,
            "no_training": True,
            "class_order_unchanged": tuple(config.class_names) == tuple(dataset_after["class_order"]),
        },
    }
    (OUTPUT / "conversion_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "float16_sha256": metadata["float16_sha256"],
                "keras_vs_float16_agreement": metadata["agreement"]["keras_vs_float16_percentage"],
                "float32_vs_float16_agreement": metadata["agreement"]["float32_vs_float16_percentage"],
                "test_accuracy": result["test_accuracy"],
                "decision": metadata["deployment_decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
