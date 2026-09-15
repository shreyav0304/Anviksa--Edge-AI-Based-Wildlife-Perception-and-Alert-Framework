from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_ROOT = PROJECT_ROOT / "wildlife_gate"
KERAS_PATH = GATE_ROOT / "results" / "gate_v1" / "wildlife_gate_v1.keras"
TFLITE_PATH = (
    GATE_ROOT
    / "results"
    / "gate_v1"
    / "deployment"
    / "wildlife_gate_v1_float32.tflite"
)
TEST_ROOT = GATE_ROOT / "dataset" / "test"
CLASSES = (("Non_Wildlife", 0), ("Wildlife", 1))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_SHAPE = (1, 224, 224, 3)
OUTPUT_SHAPE = (1, 1)
THRESHOLD = 0.50
MAX_ALLOWED_DIFFERENCE = 0.001


def image_paths() -> list[tuple[Path, int]]:
    images = []
    for class_name, label in CLASSES:
        directory = TEST_ROOT / class_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Test class directory missing: {directory}")
        images.extend(
            (path, label)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return images


def load_image(path: Path) -> np.ndarray:
    # imdecode handles Windows paths that cv2.imread may not handle reliably.
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode the image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype(np.float32), axis=0)


def main() -> int:
    print("ANVIKSA AI - WILDLIFE GATE V1 KERAS/TFLITE VERIFICATION")
    print(f"Keras model : {KERAS_PATH}")
    print(f"TFLite model: {TFLITE_PATH}")
    print(f"Threshold   : {THRESHOLD:.2f}")

    try:
        paths = image_paths()
        if not paths:
            raise ValueError("No test images found in the two class directories")
        model = tf.keras.models.load_model(KERAS_PATH)
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Expected one input and one output; got {len(inputs)} inputs, "
                f"{len(outputs)} outputs"
            )
        input_info, output_info = inputs[0], outputs[0]
        input_shape = tuple(int(x) for x in input_info["shape"])
        output_shape = tuple(int(x) for x in output_info["shape"])
        print(f"TFLite input : shape={input_shape}, dtype={input_info['dtype'].__name__}")
        print(f"TFLite output: shape={output_shape}, dtype={output_info['dtype'].__name__}")
        if input_shape != INPUT_SHAPE or input_info["dtype"] != np.float32:
            raise ValueError("TFLite input must be shape (1, 224, 224, 3), float32")
        if output_shape != OUTPUT_SHAPE or output_info["dtype"] != np.float32:
            raise ValueError("TFLite output must be shape (1, 1), float32")
    except Exception as exc:
        print(f"FAIL: Setup or tensor verification failed: {exc}")
        return 1

    checked = 0
    differences = []
    mismatches = 0
    keras_correct = 0
    tflite_correct = 0
    tn = fp = fn = tp = 0
    failures = []

    for path, actual in paths:
        try:
            tensor = load_image(path)
            keras_output = np.asarray(model(tensor, training=False))
            if keras_output.shape != OUTPUT_SHAPE:
                raise ValueError(f"Unexpected Keras output shape: {keras_output.shape}")
            interpreter.set_tensor(input_info["index"], tensor)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_info["index"])
            keras_prob = float(keras_output[0, 0])
            tflite_prob = float(tflite_output[0, 0])
            if not np.isfinite(keras_prob) or not np.isfinite(tflite_prob):
                raise ValueError("Non-finite model probability")
            if not 0 <= keras_prob <= 1 or not 0 <= tflite_prob <= 1:
                raise ValueError("Model probability outside [0, 1]")
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            continue

        difference = abs(keras_prob - tflite_prob)
        keras_decision = int(keras_prob >= THRESHOLD)
        tflite_decision = int(tflite_prob >= THRESHOLD)
        checked += 1
        differences.append(difference)
        mismatches += int(keras_decision != tflite_decision)
        keras_correct += int(keras_decision == actual)
        tflite_correct += int(tflite_decision == actual)
        if actual == 0:
            tn += int(tflite_decision == 0)
            fp += int(tflite_decision == 1)
        else:
            fn += int(tflite_decision == 0)
            tp += int(tflite_decision == 1)
        print(
            f"{path.relative_to(TEST_ROOT)}: actual={actual}, "
            f"Keras={keras_prob:.8f} ({keras_decision}), "
            f"TFLite={tflite_prob:.8f} ({tflite_decision}), "
            f"abs_diff={difference:.8f}"
        )

    max_difference = max(differences, default=float("nan"))
    mean_difference = float(np.mean(differences)) if differences else float("nan")
    print("\nVERIFICATION RESULT")
    print(f"Images checked                    : {checked}/{len(paths)}")
    print(f"Maximum absolute probability diff: {max_difference:.8f}")
    print(f"Mean absolute probability diff   : {mean_difference:.8f}")
    print(f"Decision mismatches              : {mismatches}")
    print(f"Keras accuracy                   : {keras_correct / checked:.2%}" if checked else "Keras accuracy                   : N/A")
    print(f"TFLite accuracy                  : {tflite_correct / checked:.2%}" if checked else "TFLite accuracy                  : N/A")
    print("TFLite confusion matrix:")
    print(f"  Actual Non-Wildlife -> Non-Wildlife: {tn}")
    print(f"  Actual Non-Wildlife -> Wildlife    : {fp}")
    print(f"  Actual Wildlife -> Non-Wildlife    : {fn}")
    print(f"  Actual Wildlife -> Wildlife        : {tp}")

    reasons = []
    if checked != len(paths):
        reasons.append(f"Only {checked} of {len(paths)} test images were evaluated")
    if mismatches:
        reasons.append(f"{mismatches} Keras/TFLite decision mismatches")
    if not differences or max_difference > MAX_ALLOWED_DIFFERENCE:
        reasons.append(
            f"Maximum absolute probability difference exceeds {MAX_ALLOWED_DIFFERENCE:.3f} "
            "or no probabilities were measured"
        )
    reasons.extend(f"Image failed: {failure}" for failure in failures)
    if reasons:
        print("FAIL: " + "; ".join(reasons))
        return 1
    print("PASS: All test images evaluated; decisions match; maximum difference <= 0.001")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
