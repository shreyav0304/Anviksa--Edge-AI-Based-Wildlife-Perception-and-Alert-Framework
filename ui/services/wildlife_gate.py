"""Reusable Float32 TFLite Wildlife Gate for RGB video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import hashlib
from .opencv import require_opencv
import numpy as np

from pi_deployment.inference.tflite_inference import _load_interpreter_class


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_MODEL_PATH = (
    PROJECT_ROOT / "wildlife_gate" / "results" / "gate_v1"
    / "deployment" / "wildlife_gate_v1_float32.tflite"
)
EXPECTED_GATE_SHA256 = "0ac3c6c0dfca692dc6c6d421f60479082e3338936d09af09301bb8c6cf61d4ca"
INPUT_SHAPE = (1, 224, 224, 3)
OUTPUT_SHAPE = (1, 1)
LOWER_THRESHOLD = 0.20
UPPER_THRESHOLD = 0.65


@dataclass(frozen=True)
class WildlifeGateResult:
    wildlife_probability: float
    non_wildlife_probability: float
    decision: str
    inference_time_ms: float


def raw_gate_decision(probability: float) -> str:
    """Map one Wildlife probability to the fixed prototype gate state."""
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Wildlife probability must be finite and in [0, 1]")
    if probability < LOWER_THRESHOLD:
        return "NON_WILDLIFE"
    if probability < UPPER_THRESHOLD:
        return "UNCERTAIN"
    return "WILDLIFE"


class WildlifeGateService:
    """Load and allocate one interpreter, then reuse it for every selected frame."""

    def __init__(self, model_path: Path = GATE_MODEL_PATH, num_threads: int = 1) -> None:
        if num_threads not in (1, 2, 4):
            raise ValueError("num_threads must be 1, 2, or 4")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if hashlib.sha256(self.model_path.read_bytes()).hexdigest() != EXPECTED_GATE_SHA256:
            raise ValueError("Wildlife Gate model hash mismatch; restore the validated deployment file")
        interpreter_class, self.interpreter_source = _load_interpreter_class()
        self.interpreter = interpreter_class(model_path=str(self.model_path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Wildlife Gate requires exactly one input and one output tensor")
        self.input_detail, self.output_detail = inputs[0], outputs[0]
        input_shape = tuple(int(value) for value in self.input_detail["shape"])
        output_shape = tuple(int(value) for value in self.output_detail["shape"])
        if input_shape != INPUT_SHAPE or self.input_detail["dtype"] != np.float32:
            raise ValueError(f"Unexpected gate input: {input_shape}, {self.input_detail['dtype']}")
        if output_shape != OUTPUT_SHAPE or self.output_detail["dtype"] != np.float32:
            raise ValueError(f"Unexpected gate output: {output_shape}, {self.output_detail['dtype']}")

    def predict(self, frame_rgb: np.ndarray) -> WildlifeGateResult:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("Gate input must be a nonempty RGB HxWx3 NumPy frame")
        if not np.issubdtype(frame.dtype, np.number) or np.iscomplexobj(frame) or not np.isfinite(frame).all():
            raise ValueError("Gate input must contain finite numeric RGB pixels")
        if frame.min() < 0 or frame.max() > 255:
            raise ValueError("Gate RGB pixels must remain in [0, 255]")
        cv2 = require_opencv()
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        batch = np.ascontiguousarray(resized[np.newaxis, ...], dtype=np.float32)
        started = perf_counter()
        self.interpreter.set_tensor(self.input_detail["index"], batch)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])
        elapsed_ms = (perf_counter() - started) * 1000.0
        if output.shape != OUTPUT_SHAPE:
            raise ValueError(f"Unexpected gate output shape: {output.shape}")
        probability = float(output[0, 0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Gate returned an invalid Wildlife probability")
        return WildlifeGateResult(
            wildlife_probability=probability,
            non_wildlife_probability=1.0 - probability,
            decision=raw_gate_decision(probability),
            inference_time_ms=elapsed_ms,
        )
