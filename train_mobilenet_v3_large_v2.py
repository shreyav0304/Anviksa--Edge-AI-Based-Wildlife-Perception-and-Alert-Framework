#!/usr/bin/env python3
"""Run the explicitly approved, isolated ANVIKSA Dataset V2 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from experiments.config import ExperimentConfig
from experiments.data_pipeline import build_shared_augmentation
from experiments.model_registry import build_model, compile_model, configure_fine_tuning


CLASSES = ("Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake", "Venomous_Snake", "Wild_Boar")
EXPECTED_FINGERPRINT = "9dd48f6a1aeb8afad1f76d943b7baf36a58c837e50b7124cacb3f5211e580f23"
EXPECTED_COUNTS = {"train": 3819, "validation": 572, "test": 545}
BASELINE_DIR = Path("results/mobilenet_v3_large/20260830T171150Z_94069968")
BASELINE_DASHBOARD = Path("results/model_comparison_dashboard")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    protected = [root / "clean_dataset", root / "combined_dataset", root / "models", root / "pi_deployment",
                 root / "results/custom_cnn", root / "results/mobilenet_v2", root / "results/mobilenet_v3_large",
                 root / "results/mobilenet_v3_large_v2",
                 root / "results/efficientnet_b0", root / "results/deployment", root / "results/model_comparison_dashboard",
                 root / "ui", root / "snake_video_dataset"]
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for base in protected if base.exists() for path in base.rglob("*") if path.is_file()}


def verify_dataset(root: Path) -> tuple[list[dict[str, str]], dict]:
    dataset = root / "dataset_v2_candidate"
    metadata = dataset / "metadata"
    with (metadata / "dataset_v2_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    if tuple(CLASSES) != CLASSES:
        errors.append("class order mismatch")
    counts = Counter(row["split"] for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        errors.append(f"split counts mismatch: {dict(counts)}")
    if len(rows) != 4936:
        errors.append(f"total count mismatch: {len(rows)}")
    new_rows = [row for row in rows if row["is_new_candidate_image"].lower() == "true"]
    if len(new_rows) != 135 or any(row["split"] == "test" for row in new_rows):
        errors.append("new image count or frozen-test exclusion mismatch")
    missing, corrupt, mismatched = [], [], []
    seen_hash_splits: dict[str, set[str]] = defaultdict(set)
    canonical: list[tuple[str, str]] = []
    for row in rows:
        path = dataset / row["destination_path"]
        if not path.is_file():
            missing.append(row["destination_path"]); continue
        actual_hash = sha256_file(path)
        if actual_hash != row["sha256"]:
            mismatched.append(row["destination_path"])
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt.append(row["destination_path"])
        seen_hash_splits[actual_hash].add(row["split"])
        canonical.append((row["destination_path"], json.dumps({"destination_path": row["destination_path"], "split": row["split"],
                                                               "class_name": row["class_name"], "sha256": row["sha256"]},
                                                              sort_keys=True, separators=(",", ":"))))
    fingerprint = hashlib.sha256(("\n".join(line for _, line in sorted(canonical)) + "\n").encode()).hexdigest()
    cross_split = {key: sorted(value) for key, value in seen_hash_splits.items() if len(value) > 1}
    historical_test = list(csv.DictReader((metadata / "historical_frozen_test_manifest.csv").open(newline="", encoding="utf-8-sig")))
    v2_test = [row for row in rows if row["split"] == "test"]
    test_match = len(historical_test) == 545 and [r["relative_path"] for r in historical_test] == [r["destination_path"] for r in v2_test] and [r["sha256"] for r in historical_test] == [r["sha256"] for r in v2_test]
    validation = json.loads((metadata / "dataset_v2_validation.json").read_text(encoding="utf-8"))
    if fingerprint != EXPECTED_FINGERPRINT:
        errors.append(f"fingerprint mismatch: {fingerprint}")
    if missing or corrupt or mismatched or cross_split:
        errors.append("file integrity, readability, or exact cross-split duplicate check failed")
    if not test_match:
        errors.append("historical frozen-test manifest mismatch")
    if validation.get("new_source_group_split_leakage") or validation.get("new_exact_frozen_test_matches") or validation.get("new_perceptual_frozen_test_candidates"):
        errors.append("recorded leakage checks are not clean")
    result = {"status": "PASS" if not errors else "FAIL", "dataset_root": str(dataset.resolve()),
              "fingerprint": fingerprint, "expected_fingerprint": EXPECTED_FINGERPRINT,
              "class_order": list(CLASSES), "split_counts": dict(counts), "total_images": len(rows),
              "new_images": len(new_rows), "new_test_images": sum(r["split"] == "test" for r in new_rows),
              "all_files_present": not missing, "all_files_readable": not corrupt,
              "all_hashes_match": not mismatched, "exact_cross_split_duplicate_groups": cross_split,
              "historical_frozen_test_unchanged": test_match, "recorded_v2_validation": validation,
              "rights_status": "REVIEW REQUIRED", "errors": errors}
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return rows, result


def make_dataset(root: Path, rows: list[dict[str, str]], split: str, batch: int, seed: int) -> tf.data.Dataset:
    chosen = [row for row in rows if row["split"] == split]
    indices = {name: i for i, name in enumerate(CLASSES)}
    paths = [str(root / "dataset_v2_candidate" / row["destination_path"]) for row in chosen]
    labels = [indices[row["class_name"]] for row in chosen]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(len(chosen), seed=seed, reshuffle_each_iteration=True)
    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.cast(tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), tf.float32)
        return image, tf.one_hot(label, len(CLASSES))
    options = tf.data.Options(); options.experimental_deterministic = True
    return ds.with_options(options).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


class EpochLog(tf.keras.callbacks.Callback):
    def __init__(self, stage: str, csv_path: Path):
        super().__init__(); self.stage = stage; self.csv_path = csv_path; self.rows = []; self.started = 0.0
    def on_epoch_begin(self, epoch, logs=None): self.started = perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        row = {"stage": self.stage, "stage_epoch": epoch + 1, "duration_seconds": perf_counter() - self.started,
               "learning_rate": float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
               **{key: float(value) for key, value in (logs or {}).items()}}
        self.rows.append(row)


def callbacks(output: Path, stage: str, weights: Path, log: EpochLog) -> list:
    return [tf.keras.callbacks.ModelCheckpoint(weights, monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=3, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.CSVLogger(output / f"{stage}_training_log.csv"), log]


def save_matrix(matrix: np.ndarray, path: Path, normalized: bool) -> None:
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["actual\\predicted", *CLASSES])
        for name, row in zip(CLASSES, matrix): writer.writerow([name, *row.tolist()])
    fig, ax = plt.subplots(figsize=(11, 9)); im = ax.imshow(matrix, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual", title="Normalized Confusion Matrix" if normalized else "Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = float(matrix.max()) / 2
    for i in range(7):
        for j in range(7): ax.text(j, i, format(matrix[i, j], ".2f" if normalized else "d"), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout(); fig.savefig(path, dpi=250); plt.close(fig)


def graph_history(rows: list[dict], output: Path) -> None:
    for train_key, val_key, title, filename in (("accuracy", "val_accuracy", "Accuracy", "training_accuracy.png"), ("loss", "val_loss", "Loss", "training_loss.png")):
        fig, ax = plt.subplots(figsize=(10, 6)); x = range(1, len(rows) + 1)
        ax.plot(x, [r[train_key] for r in rows], label="Training"); ax.plot(x, [r[val_key] for r in rows], label="Validation")
        boundary = next((i for i, r in enumerate(rows, 1) if r["stage"] == "fine_tuning"), None)
        if boundary: ax.axvline(boundary - .5, color="black", linestyle="--", label="Fine-tuning begins")
        ax.set(xlabel="Epoch", ylabel=title, title=f"Training and Validation {title}"); ax.grid(True); ax.legend(); fig.tight_layout(); fig.savefig(output / filename, dpi=250); plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parent
    variant = "v3" if "--v3" in sys.argv[1:] else "v2"
    before = protected_snapshot(root)
    rows, verification = verify_dataset(root)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = root / f"results/mobilenet_v3_large_{variant}" / identifier
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset_verification.json", verification)
    started = datetime.now(timezone.utc); state = {"status": "IN_PROGRESS", "stage_reached": "preflight", "started_at_utc": started.isoformat(), "rights_status": "REVIEW REQUIRED"}
    write_json(output / "experiment_metadata.json", state)
    config = ExperimentConfig(model_key="mobilenet_v3_large", dataset_root=root / "dataset_v2_candidate",
                              metadata_root=root / "dataset_v2_candidate/metadata", results_root=root / "results",
                              dataset_fingerprint=EXPECTED_FINGERPRINT, batch_size=32)
    configuration = config.to_dict() | {"experiment_model_key": f"mobilenet_v3_large_{variant}", "starting_weights": "ImageNet (fresh; no historical checkpoint loaded)",
                                      "optimizer": "Adam", "loss": "categorical_crossentropy", "checkpoint_monitor": "val_accuracy",
                                      "selection_policy": "higher best validation accuracy across stage 1 and fine-tuning",
                                      "frozen_test_policy": "not constructed or accessed until selected saved model validation passes",
                                      "batch_size_rationale": "32 is the proven original controlled V1 setting on this CPU environment"}
    write_json(output / "training_configuration.json", configuration)
    random.seed(42); np.random.seed(42); tf.keras.utils.set_random_seed(42)
    try: tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError): pass
    train = make_dataset(root, rows, "train", 32, 42); validation = make_dataset(root, rows, "validation", 32, 42)
    train_counts = Counter(r["class_name"] for r in rows if r["split"] == "train")
    calculated_v2_weights = {i: 3819 / (7 * train_counts[name]) for i, name in enumerate(CLASSES)}
    v1_weights_record = json.loads((root / BASELINE_DIR / "class_weights.json").read_text(encoding="utf-8"))
    if set(v1_weights_record) != {str(i) for i in range(7)}:
        raise RuntimeError("Complete historical seven-class V1 weights could not be recovered")
    v1_weights = {i: float(v1_weights_record[str(i)]) for i in range(7)}
    class_weights = v1_weights if variant == "v3" else calculated_v2_weights
    write_json(output / "class_weights.json", class_weights)
    if variant == "v3":
        weight_comparison = {CLASSES[i]: {"v1_weight_used_by_v3": v1_weights[i], "v2_recalculated_weight": calculated_v2_weights[i],
                                          "difference_v3_minus_v2": v1_weights[i] - calculated_v2_weights[i]} for i in range(7)}
        write_json(output / "class_weight_comparison.json", {"status": "PASS", "class_order": list(CLASSES), "weights": weight_comparison})
        v2_config = json.loads((root / "results/mobilenet_v3_large_v2/20260904T120832Z_06b32daa/training_configuration.json").read_text(encoding="utf-8"))
        controlled_keys = ["model_key", "dataset_root", "metadata_root", "image_size", "batch_size", "seed", "initial_epochs", "fine_tune_epochs",
                           "initial_learning_rate", "fine_tune_learning_rate", "early_stopping_patience", "reduce_lr_patience", "reduce_lr_factor",
                           "class_names", "dataset_fingerprint", "augmentation", "starting_weights", "optimizer", "loss", "checkpoint_monitor", "selection_policy", "frozen_test_policy"]
        controls = []
        for key in controlled_keys:
            v2_value, v3_value = v2_config.get(key), configuration.get(key)
            controls.append({"setting": key, "v2": v2_value, "v3": v3_value, "status": "MATCH" if v2_value == v3_value else "DIFFERENT"})
        controls.append({"setting": "class_weights", "v2": calculated_v2_weights, "v3": v1_weights, "status": "DIFFERENT", "intentional": True})
        unexpected = [row for row in controls if row["status"] == "DIFFERENT" and row["setting"] != "class_weights"]
        control_record = {"status": "PASS" if not unexpected else "FAIL", "independent_variable": "class_weights", "settings": controls, "unexpected_differences": unexpected}
        write_json(output / "v2_v3_controlled_variables.json", control_record)
        if unexpected:
            raise RuntimeError(f"Unexpected controlled-variable differences: {unexpected}")
    try:
        bundle = build_model("mobilenet_v3_large", config, weights="imagenet")
        compile_model(bundle.model, 1e-3)
        state["stage_reached"] = "stage_1"; write_json(output / "experiment_metadata.json", state)
        stage1_log = EpochLog("stage_1", output / "training_history.csv"); stage1_weights = output / "stage1_best.weights.h5"
        h1 = bundle.model.fit(train, validation_data=validation, epochs=30, class_weight=class_weights, callbacks=callbacks(output, "stage1", stage1_weights, stage1_log))
        bundle.model.load_weights(stage1_weights); stage1_best = max(h1.history["val_accuracy"])
        fine_meta = configure_fine_tuning(bundle); compile_model(bundle.model, 1e-5)
        state["stage_reached"] = "fine_tuning"; write_json(output / "experiment_metadata.json", state)
        fine_log = EpochLog("fine_tuning", output / "training_history.csv"); fine_weights = output / "fine_tuned_best.weights.h5"
        h2 = bundle.model.fit(train, validation_data=validation, epochs=25, class_weight=class_weights, callbacks=callbacks(output, "fine_tune", fine_weights, fine_log))
        fine_best = max(h2.history["val_accuracy"])
        selected = "fine_tuning" if fine_best >= stage1_best else "stage_1"
        bundle.model.load_weights(fine_weights if selected == "fine_tuning" else stage1_weights)
        all_history = stage1_log.rows + fine_log.rows
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_history[0])); writer.writeheader(); writer.writerows(all_history)
        graph_history(all_history, output)
        best_model = output / "best_model.keras"; bundle.model.save(best_model)
        del bundle
        selected_model = tf.keras.models.load_model(best_model)
        sample_images, _ = next(iter(validation.take(1))); sample_prob = selected_model(sample_images, training=False).numpy()
        if selected_model.input_shape != (None, 224, 224, 3) or selected_model.output_shape != (None, 7) or not np.isfinite(sample_prob).all() or not np.allclose(sample_prob.sum(axis=1), 1, atol=1e-4):
            raise RuntimeError("selected saved-model validation failed")
        write_json(output / "model_validation.json", {"status": "PASS", "input_shape": selected_model.input_shape, "output_shape": selected_model.output_shape,
                                                        "probabilities_finite": True, "probability_sums_valid": True})
        state["stage_reached"] = "frozen_test_evaluation"; write_json(output / "experiment_metadata.json", state)
        test_rows = [r for r in rows if r["split"] == "test"]
        test = make_dataset(root, rows, "test", 32, 42)
        probabilities = selected_model.predict(test, verbose=1)
        truth = np.array([CLASSES.index(r["class_name"]) for r in test_rows]); predicted = probabilities.argmax(axis=1)
        loss = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-7, 1))))
        macro = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(truth, predicted, average="weighted", zero_division=0)
        per = precision_recall_fscore_support(truth, predicted, labels=range(7), zero_division=0)
        per_rows = [{"class": name, "precision": float(per[0][i]), "recall": float(per[1][i]), "f1": float(per[2][i]), "support": int(per[3][i])} for i, name in enumerate(CLASSES)]
        with (output / "classification_report.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_rows[0])); writer.writeheader(); writer.writerows(per_rows)
        (output / "classification_report.txt").write_text(classification_report(truth, predicted, target_names=CLASSES, digits=6, zero_division=0), encoding="utf-8")
        cm = confusion_matrix(truth, predicted, labels=range(7)); norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
        save_matrix(cm, output / "confusion_matrix.png", False); save_matrix(norm, output / "confusion_matrix_normalized.png", True)
        confidence = probabilities.max(axis=1); one_hot = np.eye(7)[truth]; mse = float(np.mean((probabilities - one_hot) ** 2))
        with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["relative_path", "true_class", "predicted_class", "confidence", *[f"probability_{c}" for c in CLASSES]]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for row, y, p, conf, probs in zip(test_rows, truth, predicted, confidence, probabilities):
                writer.writerow({"relative_path": row["destination_path"], "true_class": CLASSES[y], "predicted_class": CLASSES[p], "confidence": float(conf), **{f"probability_{c}": float(probs[i]) for i, c in enumerate(CLASSES)}})
        lookup = {r["class"]: r for r in per_rows}; snake = [lookup["Non_Venomous_Snake"], lookup["Venomous_Snake"]]
        metrics = {"model_name": f"MobileNetV3 Large Dataset {variant.upper()}", "dataset_fingerprint": EXPECTED_FINGERPRINT, "test_images": 545, "test_loss": loss,
                   "test_accuracy": float(accuracy_score(truth, predicted)), "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
                   "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]), "weighted_f1": float(weighted[2]),
                   "mean_confidence": float(confidence.mean()), "median_confidence": float(np.median(confidence)), "probability_mse": mse,
                   "venomous_snake": lookup["Venomous_Snake"], "non_venomous_snake": lookup["Non_Venomous_Snake"],
                   "snake_macro_precision": float(np.mean([r["precision"] for r in snake])),
                   "snake_macro_recall": float(np.mean([r["recall"] for r in snake])), "snake_macro_f1": float(np.mean([r["f1"] for r in snake])),
                   "venomous_as_non_venomous": int(cm[5, 4]), "non_venomous_as_venomous": int(cm[4, 5]), "per_class": per_rows,
                   "total_parameters": selected_model.count_params(), "saved_model_size_bytes": best_model.stat().st_size}
        write_json(output / "metrics.json", metrics)
        baseline_metrics = json.loads((root / BASELINE_DIR / "metrics.json").read_text(encoding="utf-8"))
        with (root / BASELINE_DASHBOARD / "model_comparison_metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
            dashboard = next(r for r in csv.DictReader(handle) if r["Model"] == "MobileNetV3 Large")
        baseline = {"test_accuracy": baseline_metrics["test_accuracy"], "macro_precision": baseline_metrics["macro_precision"], "macro_recall": baseline_metrics["macro_recall"], "macro_f1": baseline_metrics["macro_f1"],
                    "weighted_f1": baseline_metrics["weighted_f1"], "snake_macro_recall": baseline_metrics["snake_macro_recall"], "snake_macro_f1": baseline_metrics["snake_macro_f1"],
                    "venomous_snake_recall": baseline_metrics["venomous_snake"]["recall"], "non_venomous_snake_recall": baseline_metrics["non_venomous_snake"]["recall"],
                    "probability_mse": float(dashboard["MSE"]), "mean_confidence": float(dashboard["Mean_Confidence"]), "median_confidence": float(dashboard["Median_Confidence"])}
        current_flat = {"test_accuracy": metrics["test_accuracy"], "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
                        "snake_macro_recall": metrics["snake_macro_recall"], "snake_macro_f1": metrics["snake_macro_f1"], "venomous_snake_recall": metrics["venomous_snake"]["recall"],
                        "non_venomous_snake_recall": metrics["non_venomous_snake"]["recall"], "probability_mse": mse, "mean_confidence": metrics["mean_confidence"], "median_confidence": metrics["median_confidence"]}
        base_per = {r["class"]: r for r in baseline_metrics["per_class"]}
        non_snake = {name: {"baseline_f1": base_per[name]["f1"], "v2_f1": lookup[name]["f1"], "difference": lookup[name]["f1"] - base_per[name]["f1"],
                            "status": "improved" if lookup[name]["f1"] > base_per[name]["f1"] else "degraded" if lookup[name]["f1"] < base_per[name]["f1"] else "unchanged"}
                     for name in CLASSES if "Snake" not in name}
        comparison = {"baseline_experiment": str((root / BASELINE_DIR).resolve()), "same_frozen_test": True, "baseline": baseline, "dataset_v2": current_flat,
                      "difference_v2_minus_baseline": {key: current_flat[key] - baseline[key] for key in baseline}, "non_snake_class_f1_assessment": non_snake}
        write_json(output / "comparison_with_baseline.json", comparison)
        graph_keys = ["test_accuracy", "macro_f1", "snake_macro_recall", "venomous_snake_recall", "non_venomous_snake_recall", "probability_mse", "mean_confidence"]
        fig, axes = plt.subplots(2, 4, figsize=(18, 9)); axes = axes.ravel()
        for ax, key in zip(axes, graph_keys): ax.bar(["V1", "V2"], [baseline[key], current_flat[key]]); ax.set_title(key.replace("_", " ")); ax.grid(axis="y", alpha=.3)
        axes[-1].axis("off"); fig.suptitle("MobileNetV3 Large: Dataset V1 vs Dataset V2"); fig.tight_layout(); fig.savefig(output / "comparison_with_baseline.png", dpi=250); plt.close(fig)
        state.update({"status": "COMPLETE", "stage_reached": "complete", "ended_at_utc": datetime.now(timezone.utc).isoformat(), "selected_stage": selected,
                      "stage1_best_validation_accuracy": stage1_best, "fine_tune_best_validation_accuracy": fine_best, "fine_tuning": fine_meta,
                      "epochs_completed": len(all_history), "protected_paths_unchanged": before == protected_snapshot(root),
                      "tensorflow_version": tf.__version__, "python_version": sys.version, "platform": platform.platform()})
        write_json(output / "experiment_metadata.json", state)
        print(json.dumps({"status": "COMPLETE", "output": str(output), "metrics": metrics}, indent=2)); return 0
    except (tf.errors.ResourceExhaustedError, MemoryError) as exc:
        state.update({"status": "INCOMPLETE_RESOURCE_LIMIT", "ended_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc), "protected_paths_unchanged": before == protected_snapshot(root)})
        write_json(output / "experiment_metadata.json", state); print(json.dumps(state, indent=2), file=sys.stderr); return 2
    except Exception as exc:
        state.update({"status": "INCOMPLETE_ERROR", "ended_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc), "protected_paths_unchanged": before == protected_snapshot(root)})
        write_json(output / "experiment_metadata.json", state); raise


if __name__ == "__main__":
    raise SystemExit(main())
