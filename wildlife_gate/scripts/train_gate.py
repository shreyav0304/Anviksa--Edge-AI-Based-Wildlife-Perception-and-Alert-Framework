from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "wildlife_gate" / "dataset"
RESULTS_DIR = PROJECT_ROOT / "wildlife_gate" / "results" / "gate_v1"

MODEL_PATH = RESULTS_DIR / "wildlife_gate_v1.keras"
HISTORY_PATH = RESULTS_DIR / "history.json"


def make_dataset(split: str, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / split,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )


def main():
    tf.keras.utils.set_random_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANVIKSA AI - WILDLIFE GATE V1 TRAINING")
    print("=" * 70)

    train_ds = make_dataset("train", shuffle=True)
    val_ds = make_dataset("validation", shuffle=False)

    print(f"\nClass order: {train_ds.class_names}")

    if train_ds.class_names != ["Non_Wildlife", "Wildlife"]:
        raise RuntimeError(
            f"Unexpected class order: {train_ds.class_names}"
        )

    # Pipeline performance
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    # Mild augmentation only.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    # MobileNetV3 includes its preprocessing by default.
    # Input must remain RGB float values in the [0, 255] range.
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )

    # Stage 1: feature extraction
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")

    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Output = probability of Wildlife.
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="wildlife_probability",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="anviksa_wildlife_gate_v1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n[STAGE 1] Training classification head...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save final restored/best model.
    model.save(MODEL_PATH)

    serializable_history = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model   : {MODEL_PATH}")
    print(f"History : {HISTORY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()