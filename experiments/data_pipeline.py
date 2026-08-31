"""Frozen-manifest data loading shared by every compared model."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_integrity import dataset_fingerprint, inspect_dataset

from .config import CLASS_NAMES, EXPECTED_SPLIT_COUNTS, ExperimentConfig


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: str
    split: str
    class_name: str
    sha256: str
    width: int
    height: int


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    train_entries: list[ManifestEntry]
    validation_entries: list[ManifestEntry]
    test_entries: list[ManifestEntry]
    class_weights: dict[int, float]


def _load_manifest(path: Path, expected_split: str) -> list[ManifestEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen manifest not found: {path}")

    entries: list[ManifestEntry] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] != expected_split:
                raise ValueError(
                    f"Manifest {path} contains split {row['split']!r}; "
                    f"expected {expected_split!r}."
                )
            entries.append(
                ManifestEntry(
                    relative_path=row["relative_path"],
                    split=row["split"],
                    class_name=row["class_name"],
                    sha256=row["sha256"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
    return entries


def validate_clean_dataset(config: ExperimentConfig, strict: bool = True) -> dict:
    """Validate fixed labels, manifests, counts, and dataset fingerprint."""
    if not config.dataset_root.is_dir():
        raise FileNotFoundError(
            f"Clean dataset not found: {config.dataset_root.resolve()}"
        )

    class_path = config.metadata_root / "class_names.json"
    fingerprint_path = config.metadata_root / "dataset_fingerprint.sha256"
    if not class_path.is_file() or not fingerprint_path.is_file():
        raise FileNotFoundError(
            f"Clean dataset metadata is incomplete: {config.metadata_root}"
        )

    recorded_classes = tuple(json.loads(class_path.read_text(encoding="utf-8")))
    if recorded_classes != config.class_names or recorded_classes != CLASS_NAMES:
        raise ValueError(
            "Class-order mismatch. Expected "
            f"{list(CLASS_NAMES)}, received {list(recorded_classes)}."
        )

    recorded_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip()
    if recorded_fingerprint != config.dataset_fingerprint:
        raise ValueError(
            "Clean metadata fingerprint mismatch: expected "
            f"{config.dataset_fingerprint}, received {recorded_fingerprint}."
        )

    manifests = {
        split: _load_manifest(
            config.metadata_root / f"{split}_manifest.csv", split
        )
        for split in EXPECTED_SPLIT_COUNTS
    }
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        if len(manifests[split]) != expected_count:
            raise ValueError(
                f"{split} manifest count changed: expected {expected_count}, "
                f"received {len(manifests[split])}."
            )
        for entry in manifests[split]:
            if entry.class_name not in config.class_names:
                raise ValueError(
                    f"Unknown class in manifest: {entry.class_name}"
                )
            if not (config.dataset_root / entry.relative_path).is_file():
                raise FileNotFoundError(
                    "Manifest image missing: "
                    f"{config.dataset_root / entry.relative_path}"
                )

    if strict:
        audit = inspect_dataset(config.dataset_root)
        live_fingerprint = dataset_fingerprint(audit["records"])
        if live_fingerprint != config.dataset_fingerprint:
            raise ValueError(
                "Clean dataset content changed. Expected fingerprint "
                f"{config.dataset_fingerprint}, received {live_fingerprint}."
            )

    return {
        "dataset_root": str(config.dataset_root.resolve()),
        "fingerprint": recorded_fingerprint,
        "class_order": list(recorded_classes),
        "split_counts": {
            split: len(entries) for split, entries in manifests.items()
        },
        "manifests": manifests,
    }


def calculate_class_weights(
    train_entries: list[ManifestEntry], class_names: tuple[str, ...]
) -> dict[int, float]:
    """Calculate balanced weights using training labels only."""
    counts = np.zeros(len(class_names), dtype=np.int64)
    class_indices = {name: index for index, name in enumerate(class_names)}
    for entry in train_entries:
        counts[class_indices[entry.class_name]] += 1
    if np.any(counts == 0):
        raise ValueError(f"Training split contains an empty class: {counts}")
    total = int(counts.sum())
    return {
        index: float(total / (len(class_names) * count))
        for index, count in enumerate(counts)
    }


def _dataset_from_entries(
    entries: list[ManifestEntry],
    config: ExperimentConfig,
    training: bool,
) -> tf.data.Dataset:
    class_indices = {name: index for index, name in enumerate(config.class_names)}
    paths = [str(config.dataset_root / entry.relative_path) for entry in entries]
    labels = [class_indices[entry.class_name] for entry in entries]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            len(entries), seed=config.seed, reshuffle_each_iteration=True
        )

    def load_image(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(
            image, config.image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        image = tf.cast(image, tf.float32)
        label_one_hot = tf.one_hot(label, len(config.class_names))
        return image, label_one_hot

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_dataset_bundle(
    config: ExperimentConfig, strict_validation: bool = True
) -> DatasetBundle:
    validation = validate_clean_dataset(config, strict=strict_validation)
    manifests: dict[str, list[ManifestEntry]] = validation["manifests"]
    class_weights = calculate_class_weights(
        manifests["train"], config.class_names
    )
    return DatasetBundle(
        train=_dataset_from_entries(manifests["train"], config, training=True),
        validation=_dataset_from_entries(
            manifests["validation"], config, training=False
        ),
        test=_dataset_from_entries(manifests["test"], config, training=False),
        train_entries=manifests["train"],
        validation_entries=manifests["validation"],
        test_entries=manifests["test"],
        class_weights=class_weights,
    )


def build_shared_augmentation(config: ExperimentConfig) -> tf.keras.Sequential:
    """Create moderate geometry and illumination augmentation for training."""
    settings = config.augmentation
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal", seed=config.seed, name="random_horizontal_flip"
            ),
            tf.keras.layers.RandomRotation(
                float(settings["rotation_factor"]),
                fill_mode="nearest",
                seed=config.seed,
                name="random_rotation",
            ),
            tf.keras.layers.RandomTranslation(
                float(settings["translation_height"]),
                float(settings["translation_width"]),
                fill_mode="nearest",
                seed=config.seed,
                name="random_translation",
            ),
            tf.keras.layers.RandomZoom(
                float(settings["zoom_factor"]),
                fill_mode="nearest",
                seed=config.seed,
                name="random_zoom",
            ),
            tf.keras.layers.RandomContrast(
                float(settings["contrast_factor"]),
                seed=config.seed,
                name="random_contrast",
            ),
            tf.keras.layers.RandomBrightness(
                float(settings["brightness_factor"]),
                value_range=(0.0, 255.0),
                seed=config.seed,
                name="random_brightness",
            ),
        ],
        name="shared_training_augmentation",
    )
