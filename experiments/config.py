"""Central reproducible configuration for comparative experiments."""

from __future__ import annotations

import json
import platform
import random
import sys
import uuid
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATASET_ROOT = PROJECT_ROOT / "clean_dataset"
CLEAN_METADATA_ROOT = PROJECT_ROOT / "dataset_integrity_reports" / "clean"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
EXPECTED_DATASET_FINGERPRINT = (
    "b103634bb6a4a01785cb7c7e3b176223325052da47228dba687edaf43a84c594"
)
CLASS_NAMES = (
    "Cow",
    "Deer",
    "Elephant",
    "Monkey",
    "Non_Venomous_Snake",
    "Venomous_Snake",
    "Wild_Boar",
)
EXPECTED_SPLIT_COUNTS = {"train": 3711, "validation": 545, "test": 545}


@dataclass
class ExperimentConfig:
    """Serializable settings shared by all model experiments."""

    model_key: str
    dataset_root: Path = CLEAN_DATASET_ROOT
    metadata_root: Path = CLEAN_METADATA_ROOT
    results_root: Path = DEFAULT_RESULTS_ROOT
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    initial_epochs: int = 30
    fine_tune_epochs: int = 25
    initial_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    class_names: tuple[str, ...] = CLASS_NAMES
    dataset_fingerprint: str = EXPECTED_DATASET_FINGERPRINT
    augmentation: dict[str, object] = field(
        default_factory=lambda: {
            "horizontal_flip": True,
            "rotation_factor": 0.055,
            "translation_height": 0.10,
            "translation_width": 0.10,
            "zoom_factor": 0.15,
            "contrast_factor": 0.10,
            "brightness_factor": 0.10,
        }
    )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root.resolve())
        payload["metadata_root"] = str(self.metadata_root.resolve())
        payload["results_root"] = str(self.results_root.resolve())
        payload["image_size"] = list(self.image_size)
        payload["class_names"] = list(self.class_names)
        return payload

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )


def set_global_determinism(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow reproducibly."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def environment_metadata() -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "numpy_version": np.__version__,
        "gpu_devices": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
    }


def create_experiment_directory(config: ExperimentConfig) -> Path:
    """Atomically create a unique directory that cannot overwrite a run."""
    diagnostic_path = os.environ.get("ANVIKSA_EXPERIMENT_DIR")
    if diagnostic_path:
        path = Path(diagnostic_path).resolve()
        expected_parent = (config.results_root / config.model_key).resolve()
        if path.parent != expected_parent or not path.is_dir():
            raise ValueError("Invalid pre-created diagnostic experiment directory.")
        if (path / "config.json").exists():
            raise FileExistsError("Diagnostic experiment directory was already used.")
        return path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    path = config.results_root / config.model_key / experiment_id
    path.mkdir(parents=True, exist_ok=False)
    return path
