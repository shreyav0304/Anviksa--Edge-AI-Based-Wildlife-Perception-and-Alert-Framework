"""Lightweight framework validation; never trains or evaluates real models."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from experiments.config import (
    ExperimentConfig,
    create_experiment_directory,
    set_global_determinism,
)
from experiments.data_pipeline import (
    build_dataset_bundle,
    build_shared_augmentation,
)
from experiments.evaluator import (
    compute_confusion_matrices,
    save_confusion_matrices,
)
from experiments.model_registry import MODEL_REGISTRY, build_model, compile_model
from experiments.reporting import save_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="Load cached official ImageNet weights during builder checks.",
    )
    parser.add_argument(
        "--plotting",
        action="store_true",
        help="Verify temporary confusion-matrix and history PNG output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.plotting:
        import matplotlib

        matplotlib.use("Agg")

    set_global_determinism(42)
    base_config = ExperimentConfig(model_key="custom_cnn")
    datasets = build_dataset_bundle(base_config, strict_validation=True)
    images, labels = next(iter(datasets.train))
    validation_images, validation_labels = next(iter(datasets.validation))
    test_images, test_labels = next(iter(datasets.test))
    augmented = build_shared_augmentation(base_config)(images[:1], training=True)

    results: dict[str, object] = {
        "dataset": {
            "train": len(datasets.train_entries),
            "validation": len(datasets.validation_entries),
            "test": len(datasets.test_entries),
            "classes": list(base_config.class_names),
            "batch_shape": images.shape.as_list(),
            "label_shape": labels.shape.as_list(),
            "validation_batch_shape": validation_images.shape.as_list(),
            "validation_label_shape": validation_labels.shape.as_list(),
            "test_batch_shape": test_images.shape.as_list(),
            "test_label_shape": test_labels.shape.as_list(),
            "augmentation_shape": augmented.shape.as_list(),
            "class_weights": datasets.class_weights,
        },
        "models": {},
    }

    for model_key in MODEL_REGISTRY:
        tf.keras.backend.clear_session()
        config = ExperimentConfig(model_key=model_key)
        weights = "imagenet" if args.imagenet else None
        bundle = build_model(model_key, config, weights=weights)
        compile_model(bundle.model, config.initial_learning_rate)
        output = bundle.model(images[:1], training=False).numpy()
        if output.shape != (1, len(config.class_names)):
            raise RuntimeError(
                f"{model_key} output shape mismatch: {output.shape}"
            )
        if not np.isfinite(output).all():
            raise RuntimeError(f"{model_key} produced invalid probabilities.")
        results["models"][model_key] = {
            "parameters": bundle.model.count_params(),
            "output_shape": list(output.shape),
            "probability_sum": float(output.sum()),
            "preprocessing": bundle.spec.preprocessing,
            "compiled_loss": str(bundle.model.loss),
            "optimizer": bundle.model.optimizer.__class__.__name__,
            "weights": "imagenet" if args.imagenet and model_key != "custom_cnn" else "random",
        }
        del bundle
        gc.collect()

    raw, normalized = compute_confusion_matrices(
        np.array([0, 1, 4, 5, 5]),
        np.array([0, 1, 5, 5, 4]),
        len(base_config.class_names),
    )
    results["confusion_matrix_preflight"] = {
        "raw_shape": list(raw.shape),
        "normalized_shape": list(normalized.shape),
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_config = ExperimentConfig(
            model_key="custom_cnn", results_root=temporary_root
        )
        first = create_experiment_directory(temporary_config)
        second = create_experiment_directory(temporary_config)
        if first == second or not first.is_dir() or not second.is_dir():
            raise RuntimeError("Unique experiment-directory creation failed.")
        results["experiment_directory_preflight"] = "passed"

        if args.plotting:
            plot_root = temporary_root / "plots"
            plot_root.mkdir()
            class_names = base_config.class_names
            save_confusion_matrices(
                raw, normalized, class_names, plot_root
            )
            save_history(
                {
                    "accuracy": [],
                    "val_accuracy": [],
                    "loss": [],
                    "val_loss": [],
                },
                plot_root,
            )
            expected_plots = (
                "confusion_matrix_raw.png",
                "confusion_matrix_normalized.png",
                "training_accuracy.png",
                "training_loss.png",
            )
            if not all((plot_root / name).is_file() for name in expected_plots):
                raise RuntimeError("Temporary plotting preflight failed.")
            results["plotting_preflight"] = "passed"

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
