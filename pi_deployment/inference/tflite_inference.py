"""Shared MobileNetV3 Large TFLite preprocessing and inference."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES_PATH = Path(__file__).resolve().with_name("class_names.json")
EXPECTED_INPUT_SHAPE = (1, 224, 224, 3)
EXPECTED_OUTPUT_SHAPE = (1, 7)


def _load_interpreter_class() -> Tuple[Any, str]:
    """Prefer lightweight runtimes while retaining a development fallback."""
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime.interpreter.Interpreter"
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter

            return Interpreter, "ai_edge_litert.interpreter.Interpreter"
        except ImportError:
            try:
                import tensorflow as tf

                return tf.lite.Interpreter, "tensorflow.lite.Interpreter"
            except ImportError as exc:
                raise RuntimeError(
                    "No TFLite interpreter is installed. Install tflite-runtime "
                    "or a compatible LiteRT package for this Raspberry Pi OS."
                ) from exc


def load_class_names(path: Path = CLASS_NAMES_PATH) -> Tuple[str, ...]:
    names = tuple(json.loads(path.read_text(encoding="utf-8")))
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError("class_names.json must contain seven unique classes.")
    return names


def preprocess_image(image: Any) -> np.ndarray:
    """Return RGB nearest-neighbor 224x224 Float32 pixels in [0, 255].

    Paths are opened on every call. PIL images and RGB NumPy arrays are also
    accepted, which is useful when a future camera adapter supplies a frame.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
            resized = rgb.resize((224, 224), resample=Image.Resampling.NEAREST)
            array = np.asarray(resized, dtype=np.float32)
    elif isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        resized = rgb.resize((224, 224), resample=Image.Resampling.NEAREST)
        array = np.asarray(resized, dtype=np.float32)
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("NumPy image input must be an RGB HxWx3 array.")
        rgb = Image.fromarray(np.asarray(np.clip(array, 0, 255), dtype=np.uint8), mode="RGB")
        resized = rgb.resize((224, 224), resample=Image.Resampling.NEAREST)
        array = np.asarray(resized, dtype=np.float32)

    batch = np.expand_dims(array, axis=0)
    if batch.shape != EXPECTED_INPUT_SHAPE or batch.dtype != np.float32:
        raise ValueError(f"Unexpected preprocessed tensor: {batch.shape}, {batch.dtype}")
    return np.ascontiguousarray(batch)


class TFLiteWildlifeClassifier:
    """A reusable interpreter; model loading and allocation happen once."""

    def __init__(self, model_path: Any, num_threads: int = 1) -> None:
        if num_threads not in (1, 2, 4):
            raise ValueError("num_threads must be 1, 2, or 4.")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.class_names = load_class_names()
        interpreter_class, self.interpreter_source = _load_interpreter_class()
        self.interpreter = interpreter_class(
            model_path=str(self.model_path), num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self._validate_interface()

    def _validate_interface(self) -> None:
        input_shape = tuple(int(value) for value in self.input_detail["shape"])
        output_shape = tuple(int(value) for value in self.output_detail["shape"])
        if input_shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(f"Expected input {EXPECTED_INPUT_SHAPE}, received {input_shape}.")
        if output_shape != EXPECTED_OUTPUT_SHAPE:
            raise ValueError(f"Expected output {EXPECTED_OUTPUT_SHAPE}, received {output_shape}.")
        if self.input_detail["dtype"] != np.float32:
            raise TypeError(f"Expected Float32 input, received {self.input_detail['dtype']}.")
        if self.output_detail["dtype"] != np.float32:
            raise TypeError(f"Expected Float32 output, received {self.output_detail['dtype']}.")

    def invoke_preprocessed(self, batch: np.ndarray) -> Tuple[np.ndarray, float]:
        if batch.shape != EXPECTED_INPUT_SHAPE or batch.dtype != np.float32:
            raise ValueError("Input must come from preprocess_image().")
        started = perf_counter()
        self.interpreter.set_tensor(self.input_detail["index"], batch)
        self.interpreter.invoke()
        probabilities = self.interpreter.get_tensor(self.output_detail["index"])[0].copy()
        elapsed_ms = (perf_counter() - started) * 1000.0
        if probabilities.shape != (7,) or not np.isfinite(probabilities).all():
            raise ValueError("Interpreter returned invalid probabilities.")
        if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-4):
            raise ValueError("Interpreter output does not sum approximately to one.")
        return probabilities, elapsed_ms

    def predict(self, image: Any) -> dict:
        batch = preprocess_image(image)
        probabilities, inference_time_ms = self.invoke_preprocessed(batch)
        class_index = int(np.argmax(probabilities))
        return {
            "predicted_class": self.class_names[class_index],
            "class_index": class_index,
            "confidence": float(probabilities[class_index]),
            "probabilities": [float(value) for value in probabilities],
            "inference_time_ms": inference_time_ms,
        }
