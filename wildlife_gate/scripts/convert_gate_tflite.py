from __future__ import annotations

import hashlib
from pathlib import Path

import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "wildlife_gate"
    / "results"
    / "gate_v1"
    / "wildlife_gate_v1.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "wildlife_gate"
    / "results"
    / "gate_v1"
    / "deployment"
)

TFLITE_PATH = OUTPUT_DIR / "wildlife_gate_v1_float32.tflite"

HASH_PATH = OUTPUT_DIR / "wildlife_gate_v1_float32.sha256"


# ============================================================
# SHA256
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANVIKSA AI - WILDLIFE GATE V1 TFLITE CONVERSION")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify Keras model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Keras model not found:\n{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Loading Keras model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Convert Keras -> TFLite Float32
    # --------------------------------------------------------

    print()
    print("[INFO] Converting model to TFLite Float32...")

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    # No quantization yet.
    # First deployment model stays Float32 so we can compare
    # its output directly against the Keras model.
    converter.optimizations = []

    tflite_model = converter.convert()

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    TFLITE_PATH.write_bytes(
        tflite_model
    )

    # --------------------------------------------------------
    # Calculate SHA256
    # --------------------------------------------------------

    digest = sha256_file(
        TFLITE_PATH
    )

    HASH_PATH.write_text(
        f"{digest}  {TFLITE_PATH.name}\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # File information
    # --------------------------------------------------------

    size_bytes = TFLITE_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(f"TFLite : {TFLITE_PATH}")
    print(f"Size   : {size_mb:.3f} MiB")
    print(f"SHA256 : {digest}")
    print(f"Hash   : {HASH_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()