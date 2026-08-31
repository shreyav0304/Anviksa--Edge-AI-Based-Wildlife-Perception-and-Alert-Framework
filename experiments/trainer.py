"""Shared staged trainer for registered comparative models."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import tensorflow as tf

from .config import (
    ExperimentConfig,
    create_experiment_directory,
    environment_metadata,
    set_global_determinism,
)
from .data_pipeline import build_dataset_bundle
from .evaluator import evaluate_model
from .model_registry import (
    build_model,
    compile_model,
    configure_fine_tuning,
)
from .reporting import save_history


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Record epoch durations and effective learning rates."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_durations_seconds: list[float] = []
        self.learning_rates: list[float] = []
        self._epoch_started = 0.0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._epoch_started = perf_counter()

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.epoch_durations_seconds.append(perf_counter() - self._epoch_started)
        self.learning_rates.append(
            float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        )


def _callbacks(
    config: ExperimentConfig,
    weights_path: Path,
    timing: EpochTimingCallback,
) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        timing,
    ]


def _merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list]:
    keys = {key for history in histories for key in history.history}
    return {
        key: [
            float(value)
            for history in histories
            for value in history.history.get(key, [])
        ]
        for key in sorted(keys)
    }


def run_experiment(config: ExperimentConfig) -> Path:
    """Train and evaluate one isolated experiment.

    This function performs full training and must only be called after explicit
    approval. Importing this module does not start training.
    """
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Matplotlib is required before training because every experiment "
            "must save training and confusion-matrix graphs."
        )

    set_global_determinism(config.seed)
    datasets = build_dataset_bundle(config, strict_validation=True)
    experiment_dir = create_experiment_directory(config)
    config.save(experiment_dir / "config.json")
    (experiment_dir / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2) + "\n", encoding="utf-8"
    )
    (experiment_dir / "class_weights.json").write_text(
        json.dumps(datasets.class_weights, indent=2) + "\n", encoding="utf-8"
    )
    (experiment_dir / "dataset_information.json").write_text(
        json.dumps(
            {
                "dataset_root": str(config.dataset_root.resolve()),
                "dataset_fingerprint": config.dataset_fingerprint,
                "class_order": list(config.class_names),
                "split_counts": {
                    "train": len(datasets.train_entries),
                    "validation": len(datasets.validation_entries),
                    "test": len(datasets.test_entries),
                },
                "manifest_root": str(config.metadata_root.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = build_model(config.model_key, config, weights="imagenet")
    model_metadata = asdict(bundle.spec)
    if config.model_key == "efficientnet_b0" and config.batch_size < 32:
        model_metadata["resource_adjustment"] = (
            "EfficientNetB0 used a reduced batch size because batch size 32 "
            "caused a confirmed TensorFlow CPU ResourceExhaustedError during "
            "the diagnostic forward pass."
        )
    (experiment_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2) + "\n", encoding="utf-8"
    )
    compile_model(bundle.model, config.initial_learning_rate)

    print("\nCONTROLLED EXPERIMENT CONFIGURATION")
    print("=" * 72)
    print("Model                 :", bundle.spec.display_name)
    print("Training method       :", bundle.spec.training_method)
    print("Dataset fingerprint   :", config.dataset_fingerprint)
    print("Batch size            :", config.batch_size)
    print("Maximum epochs        :", config.initial_epochs)
    print("Fine-tune max epochs  :", config.fine_tune_epochs)
    print("Optimizer             : Adam")
    print("Initial learning rate :", config.initial_learning_rate)
    print("Fine-tune learning rate:", config.fine_tune_learning_rate)
    print("Loss                  : categorical_crossentropy")
    print(
        "EarlyStopping         : monitor=val_loss, patience=",
        config.early_stopping_patience,
        ", restore_best_weights=True",
        sep="",
    )
    print(
        "ReduceLROnPlateau     : monitor=val_loss, factor=",
        config.reduce_lr_factor,
        ", patience=",
        config.reduce_lr_patience,
        sep="",
    )
    print("ModelCheckpoint       : monitor=val_accuracy, save_best_only=True")
    print("TensorFlow devices    :", tf.config.list_physical_devices())
    print("GPU devices           :", tf.config.list_physical_devices("GPU"))
    print("Backbone Stage 1      : fully frozen")
    print("Fine-tuning depth     :", bundle.spec.fine_tune_layers)
    print("BatchNormalization    : frozen during fine-tuning")
    print("=" * 72)

    stage1_weights = experiment_dir / "stage1_best.weights.h5"
    stage1_timing = EpochTimingCallback()
    training_started_at = datetime.now(timezone.utc)
    training_started_clock = perf_counter()
    stage1_history = bundle.model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=config.initial_epochs,
        class_weight=datasets.class_weights,
        callbacks=_callbacks(config, stage1_weights, stage1_timing),
    )
    bundle.model.load_weights(stage1_weights)
    stage1_best_accuracy = max(stage1_history.history["val_accuracy"])
    histories = [stage1_history]
    fine_tune_metadata = {
        "fine_tune_layers": 0,
        "trainable_backbone_layers": 0,
    }
    all_epoch_durations = list(stage1_timing.epoch_durations_seconds)
    all_learning_rates = list(stage1_timing.learning_rates)
    stage1_duration_seconds = sum(stage1_timing.epoch_durations_seconds)
    stage1_epochs_completed = len(stage1_timing.epoch_durations_seconds)
    fine_tune_duration_seconds = 0.0
    fine_tune_epochs_completed = 0
    fine_best_accuracy = None

    if bundle.spec.transfer_learning:
        fine_tune_metadata = configure_fine_tuning(bundle)
        compile_model(bundle.model, config.fine_tune_learning_rate)
        fine_weights = experiment_dir / "fine_tuned_best.weights.h5"
        fine_timing = EpochTimingCallback()
        fine_history = bundle.model.fit(
            datasets.train,
            validation_data=datasets.validation,
            epochs=config.fine_tune_epochs,
            class_weight=datasets.class_weights,
            callbacks=_callbacks(config, fine_weights, fine_timing),
        )
        all_epoch_durations.extend(fine_timing.epoch_durations_seconds)
        all_learning_rates.extend(fine_timing.learning_rates)
        fine_tune_duration_seconds = sum(fine_timing.epoch_durations_seconds)
        fine_tune_epochs_completed = len(fine_timing.epoch_durations_seconds)
        histories.append(fine_history)
        fine_best_accuracy = max(fine_history.history["val_accuracy"])
        if fine_best_accuracy >= stage1_best_accuracy:
            bundle.model.load_weights(fine_weights)
            selected_stage = "fine_tuning"
        else:
            bundle.model.load_weights(stage1_weights)
            selected_stage = "stage_1"
    else:
        selected_stage = "from_scratch"

    training_duration_seconds = perf_counter() - training_started_clock
    training_ended_at = datetime.now(timezone.utc)

    (experiment_dir / "fine_tuning.json").write_text(
        json.dumps(
            {**fine_tune_metadata, "selected_stage": selected_stage}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    history = _merge_histories(*histories)
    history["learning_rate"] = all_learning_rates
    history["fine_tuning_stage"] = (
        [0] * stage1_epochs_completed + [1] * fine_tune_epochs_completed
    )
    save_history(history, experiment_dir)

    timing_record = {
        "training_started_at_utc": training_started_at.isoformat(),
        "training_ended_at_utc": training_ended_at.isoformat(),
        "training_duration_seconds": training_duration_seconds,
        "epochs_completed": len(all_epoch_durations),
        "average_epoch_duration_seconds": (
            sum(all_epoch_durations) / len(all_epoch_durations)
            if all_epoch_durations
            else 0.0
        ),
        "epoch_durations_seconds": all_epoch_durations,
        "stage1_duration_seconds": stage1_duration_seconds,
        "stage1_epochs_completed": stage1_epochs_completed,
        "stage1_best_validation_accuracy": stage1_best_accuracy,
        "fine_tune_duration_seconds": fine_tune_duration_seconds,
        "fine_tune_epochs_completed": fine_tune_epochs_completed,
        "fine_tune_best_validation_accuracy": fine_best_accuracy,
        "initial_learning_rate": config.initial_learning_rate,
        "final_learning_rate": (
            all_learning_rates[-1]
            if all_learning_rates
            else config.initial_learning_rate
        ),
    }
    (experiment_dir / "training_timing.json").write_text(
        json.dumps(timing_record, indent=2) + "\n", encoding="utf-8"
    )

    best_model_path = experiment_dir / "best_model.keras"
    bundle.model.save(best_model_path)
    evaluate_model(
        bundle.model,
        bundle.spec,
        datasets.test,
        config,
        experiment_dir,
        best_model_path,
    )
    return experiment_dir
