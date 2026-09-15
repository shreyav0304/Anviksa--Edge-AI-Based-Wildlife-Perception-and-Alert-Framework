from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "wildlife_gate" / "dataset" / "test"
MODEL_PATH = (
    PROJECT_ROOT
    / "wildlife_gate"
    / "results"
    / "gate_v1"
    / "wildlife_gate_v1.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
THRESHOLD = 0.50

CLASS_NAMES = ["Non_Wildlife", "Wildlife"]


def main():
    print("=" * 68)
    print("ANVIKSA AI - WILDLIFE GATE V1 TEST EVALUATION")
    print("=" * 68)

    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Class order : {test_ds.class_names}")

    if test_ds.class_names != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected class order: {test_ds.class_names}"
        )

    y_true = []
    y_prob = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0).reshape(-1)

        y_true.extend(labels.numpy().reshape(-1).astype(int))
        y_prob.extend(probabilities)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # 0 = Non_Wildlife
    # 1 = Wildlife
    y_pred = (y_prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)

    non_wildlife_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    wildlife_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    print()
    print("CONFUSION MATRIX")
    print()
    print("                     Predicted")
    print("                 Non-Wildlife   Wildlife")
    print(f"Actual Non-Wildlife      {tn:3d}          {fp:3d}")
    print(f"Actual Wildlife          {fn:3d}          {tp:3d}")

    print()
    print(f"Test accuracy        : {accuracy * 100:.2f}%")
    print(
        f"Non-Wildlife recall  : "
        f"{non_wildlife_recall * 100:.2f}%"
    )
    print(
        f"Wildlife recall      : "
        f"{wildlife_recall * 100:.2f}%"
    )

    print()
    print("Critical errors:")
    print(
        f"Non-Wildlife -> Wildlife : {fp} "
        "(false wildlife acceptance)"
    )
    print(
        f"Wildlife -> Non-Wildlife : {fn} "
        "(missed wildlife)"
    )

    print()
    print("CLASSIFICATION REPORT")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Probability ranges:")

    negative_probs = y_prob[y_true == 0]
    wildlife_probs = y_prob[y_true == 1]

    if len(negative_probs):
        print(
            "Non-Wildlife wildlife-probability: "
            f"min={negative_probs.min():.4f}, "
            f"max={negative_probs.max():.4f}, "
            f"mean={negative_probs.mean():.4f}"
        )

    if len(wildlife_probs):
        print(
            "Wildlife wildlife-probability    : "
            f"min={wildlife_probs.min():.4f}, "
            f"max={wildlife_probs.max():.4f}, "
            f"mean={wildlife_probs.mean():.4f}"
        )

    print()
    print("=" * 68)


if __name__ == "__main__":
    main()