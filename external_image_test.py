"""Run qualitative inference on images kept outside the frozen datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "pi_deployment/models/mobilenet_v3_large_float32.tflite"
EXTERNAL = ROOT / "external_test_images"
OUTPUT = ROOT / "results/external_image_test"
EXPECTED_MODEL_SHA256 = "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7"
CLASS_NAMES = (
    "Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake",
    "Venomous_Snake", "Wild_Boar",
)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
FROZEN_ROOTS = (ROOT / "combined_dataset", ROOT / "clean_dataset")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(roots: tuple[Path, ...]) -> dict[str, str]:
    result = {}
    for root in roots:
        if root.is_dir():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


def level(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.60:
        return "MODERATE"
    return "LOW"


def main() -> None:
    if not MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL}")
    model_hash_before = sha256(MODEL)
    if model_hash_before != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"Model SHA-256 mismatch: expected {EXPECTED_MODEL_SHA256}, got {model_hash_before}"
        )
    if not EXTERNAL.is_dir():
        raise FileNotFoundError(f"External image directory not found: {EXTERNAL}")

    images = sorted(
        (path for path in EXTERNAL.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.relative_to(EXTERNAL).as_posix().casefold(),
    )
    if not images:
        raise RuntimeError("No supported external images found.")

    frozen_before = snapshot(FROZEN_ROOTS)
    external_before = snapshot((EXTERNAL,))

    interpreter = tf.lite.Interpreter(model_path=str(MODEL), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"]) != (1, 224, 224, 3) or input_detail["dtype"] != np.float32:
        raise RuntimeError(f"Unexpected input tensor: {input_detail}")
    if tuple(output_detail["shape"]) != (1, 7):
        raise RuntimeError(f"Unexpected output tensor: {output_detail}")

    rows = []
    json_rows = []
    for path in images:
        try:
            with Image.open(path) as image:
                image.load()
                rgb = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to read supported image {path}: {exc}") from exc
        resized = rgb.resize((224, 224), resample=Image.Resampling.NEAREST)
        model_input = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

        interpreter.set_tensor(input_detail["index"], model_input)
        started = perf_counter()
        interpreter.invoke()
        inference_time_ms = (perf_counter() - started) * 1000.0
        probabilities = np.asarray(interpreter.get_tensor(output_detail["index"])[0], dtype=np.float64)
        if probabilities.shape != (7,) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"Invalid model output for {path.name}: {probabilities}")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
            raise RuntimeError(f"Probabilities do not sum to one for {path.name}: {probabilities.sum()}")

        order = np.argsort(-probabilities, kind="stable")
        top1, top2, top3 = (int(index) for index in order[:3])
        confidence = float(probabilities[top1])
        margin = confidence - float(probabilities[top2])
        confidence_level = level(confidence)
        uncertain = confidence < 0.60 or margin < 0.15
        filename = path.relative_to(EXTERNAL).as_posix()
        probability_map = {
            name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)
        }
        row = {
            "filename": filename,
            "predicted_class": CLASS_NAMES[top1],
            "confidence": confidence,
            "confidence_level": confidence_level,
            "top1_probability": confidence,
            "top2_class": CLASS_NAMES[top2],
            "top2_probability": float(probabilities[top2]),
            "top3_class": CLASS_NAMES[top3],
            "top3_probability": float(probabilities[top3]),
            "top1_top2_margin": margin,
            "uncertain": uncertain,
            "inference_time_ms": inference_time_ms,
            **{f"probability_{name}": probability for name, probability in probability_map.items()},
        }
        rows.append(row)
        json_rows.append({
            **row,
            "top3": [
                {"class": CLASS_NAMES[index], "probability": float(probabilities[index])}
                for index in (top1, top2, top3)
            ],
            "class_probabilities": probability_map,
        })

    model_hash_after = sha256(MODEL)
    frozen_after = snapshot(FROZEN_ROOTS)
    external_after = snapshot((EXTERNAL,))
    model_unchanged = model_hash_before == model_hash_after == EXPECTED_MODEL_SHA256
    dataset_unchanged = frozen_before == frozen_after
    external_unchanged = external_before == external_after
    test_paths = [key for key in frozen_before if "/test/" in f"/{key.lower()}/"]
    post_test_paths = [key for key in frozen_after if "/test/" in f"/{key.lower()}/"]
    no_test_files_added = set(post_test_paths) == set(test_paths)
    integrity = {
        "model_unchanged": model_unchanged,
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "dataset_unchanged": dataset_unchanged,
        "frozen_dataset_files_before": len(frozen_before),
        "frozen_dataset_files_after": len(frozen_after),
        "external_images_unchanged": external_unchanged,
        "no_training_occurred": True,
        "no_files_added_to_frozen_test_dataset": no_test_files_added,
        "frozen_test_files_before": len(test_paths),
        "frozen_test_files_after": len(post_test_paths),
    }
    if not all((model_unchanged, dataset_unchanged, external_unchanged, no_test_files_added)):
        raise RuntimeError(f"Post-inference integrity verification failed: {integrity}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (OUTPUT / "external_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "test_type": "external qualitative inference test",
        "model": MODEL.relative_to(ROOT).as_posix(),
        "model_sha256": model_hash_before,
        "class_order": list(CLASS_NAMES),
        "preprocessing": "RGB; nearest-neighbor resize to 224x224; float32; [0,255]; batch dimension; no external normalization",
        "predictions": json_rows,
        "integrity": integrity,
    }
    (OUTPUT / "external_predictions.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    counts = {name: sum(row["confidence_level"] == name for row in rows) for name in ("HIGH", "MODERATE", "LOW")}
    uncertain_count = sum(row["uncertain"] for row in rows)
    report_lines = [
        "ANVIKSA MOBILENETV3 LARGE - EXTERNAL IMAGE TEST",
        "=" * 52,
        "External qualitative inference only; no ground truth or accuracy reported.",
        f"Model: {MODEL.relative_to(ROOT).as_posix()}",
        f"Verified SHA-256: {model_hash_before}",
        "Preprocessing: RGB, 224x224 nearest-neighbor, float32 [0,255], batch dimension",
        "",
    ]
    for item in json_rows:
        report_lines.extend([
            f"Filename: {item['filename']}",
            f"Predicted class: {item['predicted_class']}",
            f"Confidence: {item['confidence']:.8f} ({item['confidence_level']})",
            "Top-3: " + ", ".join(f"{entry['class']}={entry['probability']:.8f}" for entry in item["top3"]),
            f"Top-1/Top-2 margin: {item['top1_top2_margin']:.8f}",
            f"Uncertain: {str(item['uncertain']).upper()}",
            f"Inference time: {item['inference_time_ms']:.3f} ms",
            "All probabilities: " + ", ".join(f"{name}={probability:.8f}" for name, probability in item["class_probabilities"].items()),
            "",
        ])
    report_lines.extend([
        "INTEGRITY",
        f"Model unchanged: {model_unchanged}",
        f"Dataset unchanged: {dataset_unchanged}",
        "No training occurred: True",
        f"No files added to frozen test dataset: {no_test_files_added}",
        f"External images unchanged: {external_unchanged}",
        "",
        f"EXTERNAL IMAGES TESTED: {len(rows)}",
        f"HIGH CONFIDENCE: {counts['HIGH']}",
        f"MODERATE CONFIDENCE: {counts['MODERATE']}",
        f"LOW CONFIDENCE: {counts['LOW']}",
        f"UNCERTAIN: {uncertain_count}",
        "",
        "EXTERNAL IMAGE INFERENCE TEST: COMPLETE",
    ])
    (OUTPUT / "external_test_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("IMAGE | PREDICTION | CONFIDENCE | TOP-2 | MARGIN | STATUS")
    for row in rows:
        status = row["confidence_level"] + (" / UNCERTAIN" if row["uncertain"] else "")
        print(
            f"{row['filename']} | {row['predicted_class']} | {row['confidence']:.1%} | "
            f"{row['top2_class']} {row['top2_probability']:.1%} | "
            f"{row['top1_top2_margin']:.1%} | {status}"
        )
    print()
    print(f"EXTERNAL IMAGES TESTED: {len(rows)}")
    print(f"HIGH CONFIDENCE: {counts['HIGH']}")
    print(f"MODERATE CONFIDENCE: {counts['MODERATE']}")
    print(f"LOW CONFIDENCE: {counts['LOW']}")
    print(f"UNCERTAIN: {uncertain_count}")
    print("EXTERNAL IMAGE INFERENCE TEST: COMPLETE")


if __name__ == "__main__":
    main()
