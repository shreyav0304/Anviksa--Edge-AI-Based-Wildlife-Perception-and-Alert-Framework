"""Reusable offline inference for the Anviksa AI wildlife classifier."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


ImageInput: TypeAlias = str | Path | Image.Image | np.ndarray


class WildlifeInferenceEngine:
    """Load the trained model once and classify independent image inputs.

    NumPy inputs are interpreted as RGB for three-channel arrays and RGBA for
    four-channel arrays. Supported shapes are ``(H, W)``, ``(H, W, 1)``,
    ``(H, W, 3)``, and ``(H, W, 4)``.
    """

    DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parent
        / "models"
        / "anviksa_mobilenetv2_best.keras"
    )
    DEFAULT_CLASS_NAMES_PATH = (
        Path(__file__).resolve().parent / "models" / "class_names.json"
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
        class_names_path: str | Path | None = None,
    ) -> None:
        """Load and validate the local model and its ordered class mapping."""
        self.model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        self.class_names_path = Path(
            class_names_path or self.DEFAULT_CLASS_NAMES_PATH
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model file not found: {self.model_path}"
            )
        if not self.class_names_path.is_file():
            raise FileNotFoundError(
                f"Class names file not found: {self.class_names_path}"
            )

        self.class_names = self._load_class_names()

        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load trained model '{self.model_path}': {exc}"
            ) from exc

        self.input_size = self._validate_model_contract()

    def _load_class_names(self) -> tuple[str, ...]:
        try:
            with self.class_names_path.open("r", encoding="utf-8") as file:
                class_names = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Unable to read class names from "
                f"'{self.class_names_path}': {exc}"
            ) from exc

        if not isinstance(class_names, list) or not class_names:
            raise ValueError("Class names must be a non-empty JSON list.")
        if not all(isinstance(name, str) and name for name in class_names):
            raise ValueError("Every class name must be a non-empty string.")
        if len(set(class_names)) != len(class_names):
            raise ValueError("Class names must be unique.")

        return tuple(class_names)

    def _validate_model_contract(self) -> tuple[int, int]:
        input_shape = self.model.input_shape
        output_shape = self.model.output_shape

        if isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError(
                f"Expected one image input with four dimensions; received "
                f"{input_shape}."
            )
        if input_shape[-1] != 3:
            raise ValueError(
                f"Expected a three-channel RGB model input; received "
                f"{input_shape}."
            )

        height, width = input_shape[1], input_shape[2]
        if height is None or width is None:
            raise ValueError(
                f"Model must have a fixed image resolution; received "
                f"{input_shape}."
            )

        if isinstance(output_shape, list) or len(output_shape) != 2:
            raise ValueError(
                f"Expected one two-dimensional classification output; "
                f"received {output_shape}."
            )
        if output_shape[-1] != len(self.class_names):
            raise ValueError(
                "Model output/class-count mismatch: model produces "
                f"{output_shape[-1]} values but class mapping contains "
                f"{len(self.class_names)} names."
            )

        return int(width), int(height)

    @staticmethod
    def _array_to_image(array: np.ndarray) -> Image.Image:
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
            if array.shape[2] == 1:
                array = array[:, :, 0]
                mode = "L"
            else:
                mode = "RGB" if array.shape[2] == 3 else "RGBA"
        else:
            raise ValueError(
                "Incorrect NumPy image shape. Expected (H, W), (H, W, 1), "
                f"(H, W, 3), or (H, W, 4); received {array.shape}."
            )

        if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("NumPy image input cannot be empty.")

        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("NumPy image contains NaN or infinite values.")
            minimum = float(array.min())
            maximum = float(array.max())
            if 0.0 <= minimum and maximum <= 1.0:
                array = array * 255.0
            elif minimum < 0.0 or maximum > 255.0:
                raise ValueError(
                    "Floating-point NumPy images must use values in [0, 1] "
                    "or [0, 255]."
                )
            array = np.rint(array).astype(np.uint8)
        elif np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.uint8) * 255
        elif np.issubdtype(array.dtype, np.integer):
            minimum = int(array.min())
            maximum = int(array.max())
            if minimum < 0 or maximum > 255:
                raise ValueError(
                    "Integer NumPy images must use values in [0, 255]."
                )
            array = array.astype(np.uint8)
        else:
            raise TypeError(
                f"Unsupported NumPy image dtype: {array.dtype}."
            )

        return Image.fromarray(array, mode=mode)

    def _load_image(self, image_input: ImageInput) -> Image.Image:
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            try:
                with Image.open(image_path) as image:
                    image.load()
                    return image.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ValueError(
                    f"Unsupported or corrupt image '{image_path}': {exc}"
                ) from exc

        if isinstance(image_input, Image.Image):
            try:
                image_input.load()
                return image_input.convert("RGB")
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unable to read PIL image: {exc}") from exc

        if isinstance(image_input, np.ndarray):
            return self._array_to_image(image_input).convert("RGB")

        raise TypeError(
            "Unsupported image input type. Expected a file path, PIL image, "
            f"or NumPy array; received {type(image_input).__name__}."
        )

    def _preprocess(self, image_input: ImageInput) -> np.ndarray:
        image = self._load_image(image_input)
        image = image.resize(self.input_size, resample=Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
        return np.expand_dims(image_array, axis=0)

    def predict(self, image_input: ImageInput) -> dict[str, Any]:
        """Classify an image and return probabilities and timing information."""
        processing_started = perf_counter()
        model_input = self._preprocess(image_input)

        inference_started = perf_counter()
        try:
            raw_output = self.model.predict(model_input, verbose=0)
        except Exception as exc:
            raise RuntimeError(f"Model prediction failed: {exc}") from exc
        inference_time_ms = (perf_counter() - inference_started) * 1000.0

        probabilities = np.asarray(raw_output)
        if probabilities.shape != (1, len(self.class_names)):
            raise RuntimeError(
                "Unexpected prediction output shape: expected "
                f"(1, {len(self.class_names)}), received "
                f"{probabilities.shape}."
            )
        probabilities = probabilities[0]
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model prediction contains invalid values.")

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        probability_map = {
            class_name: float(probability)
            for class_name, probability in zip(
                self.class_names, probabilities, strict=True
            )
        }
        processing_time_ms = (perf_counter() - processing_started) * 1000.0

        return {
            "predicted_class": self.class_names[predicted_index],
            "confidence": confidence,
            "probabilities": probability_map,
            "inference_time_ms": inference_time_ms,
            "processing_time_ms": processing_time_ms,
        }
