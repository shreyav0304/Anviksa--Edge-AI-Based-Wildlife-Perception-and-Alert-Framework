"""Shared MobileNetV3 Large TFLite preprocessing and inference."""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES_PATH = Path(__file__).resolve().with_name("class_names.json")
EXPECTED_CLASSES = ('Big_Cat', 'Cat', 'Chicken', 'Cow', 'Deer', 'Dog', 'Elephant', 'Goat', 'Monkey', 'Mouse', 'Non_Venomous_Snake', 'Pig', 'Sheep', 'Venomous_Snake', 'Wild_Boar')
EXPECTED_INPUT_SHAPE = (1, 224, 224, 3)
EXPECTED_OUTPUT_SHAPE = (1, len(EXPECTED_CLASSES))


def _load_interpreter_class() -> Tuple[Any, str]:
    """Prefer lightweight runtimes while retaining a development fallback."""
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime.interpreter.Interpreter"
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter

            if sys.platform == "win32":
                # LiteRT 2.3.0's default Windows XNNPACK delegate exits natively on this model.
                # Built-in kernels are verified separately; keep Linux/Pi runtime defaults.
                from ai_edge_litert.interpreter import OpResolverType
                return partial(Interpreter, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES), "ai_edge_litert.interpreter.Interpreter (Windows built-in kernels)"
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
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or tuple(names) != EXPECTED_CLASSES:
        raise ValueError("class_names.json must match the exact deployed class order.")
    return tuple(names)


def preprocess_image(image: Any) -> np.ndarray:
    """Return RGB OpenCV bilinear 224x224 Float32 pixels in [0, 255].

    Paths are opened on every call. PIL images and RGB NumPy arrays are also
    accepted, which is useful when a future camera adapter supplies a frame.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to match the 15-class training preprocessing") from exc
    if isinstance(image, (str, Path)):
        # Match the frozen training decoder, including EXIF handling and OpenCV resize.
        decoded = cv2.imdecode(np.frombuffer(Path(image).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Cannot decode image: {image}")
        array = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Image input must be a nonempty RGB HxWx3 array.")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError("RGB pixels must contain finite real numbers.")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("RGB pixels must remain in [0, 255].")
    array = cv2.resize(np.asarray(array, dtype=np.uint8), (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)

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
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Classifier requires exactly one input and one output tensor.")
        self.input_detail, self.output_detail = inputs[0], outputs[0]
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
        if not np.isfinite(batch).all() or np.any(batch < 0) or np.any(batch > 255):
            raise ValueError("Preprocessed pixels must be finite and in [0, 255].")
        started = perf_counter()
        self.interpreter.set_tensor(self.input_detail["index"], batch)
        self.interpreter.invoke()
        probabilities = self.interpreter.get_tensor(self.output_detail["index"])[0].copy()
        elapsed_ms = (perf_counter() - started) * 1000.0
        if probabilities.shape != (len(EXPECTED_CLASSES),) or not np.isfinite(probabilities).all():
            raise ValueError("Interpreter returned invalid probabilities.")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Interpreter probabilities must lie in [0, 1].")
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
