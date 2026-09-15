from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "wildlife_gate"
    / "results"
    / "gate_v1"
    / "wildlife_gate_v1.keras"
)

CAMERA_INDEX = 0
IMG_SIZE = (224, 224)

# Conservative prototype threshold.
WILDLIFE_THRESHOLD = 0.50


def main():
    print("=" * 68)
    print("ANVIKSA AI - LIVE WILDLIFE GATE TEST")
    print("=" * 68)

    print(f"Model     : {MODEL_PATH}")
    print(f"Threshold : {WILDLIFE_THRESHOLD:.2f}")
    print()
    print("Controls:")
    print("  Q = quit")
    print("=" * 68)

    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[WARNING] Failed to read webcam frame.")
                continue

            # OpenCV = BGR
            # TensorFlow training images = RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resized = cv2.resize(
                rgb,
                IMG_SIZE,
                interpolation=cv2.INTER_AREA,
            )

            batch = np.expand_dims(
                resized.astype(np.float32),
                axis=0,
            )

            probability = float(
                model.predict(batch, verbose=0)[0][0]
            )

            if probability >= WILDLIFE_THRESHOLD:
                decision = "WILDLIFE"
                confidence = probability
            else:
                decision = "NON-WILDLIFE"
                confidence = 1.0 - probability

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            if decision == "WILDLIFE":
                display_color = (0, 0, 255)
            else:
                display_color = (0, 255, 0)

            cv2.putText(
                frame,
                f"Gate: {decision}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                display_color,
                2,
            )

            cv2.putText(
                frame,
                f"Wildlife probability: {probability * 100:.2f}%",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                display_color,
                2,
            )

            cv2.putText(
                frame,
                f"Decision confidence: {confidence * 100:.2f}%",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                display_color,
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "ANVIKSA AI - Wildlife Gate Test",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("[DONE] Live Wildlife Gate test stopped.")


if __name__ == "__main__":
    main()