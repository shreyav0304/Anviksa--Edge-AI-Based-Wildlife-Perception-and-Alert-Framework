"""UI adapter for the validated MobileNetV3 Large Float32 TFLite backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pi_deployment.inference.tflite_inference import TFLiteWildlifeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "pi_deployment" / "models" / "mobilenet_v3_large_float32.tflite"
EXPECTED_MODEL_SHA256 = "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7"
CATEGORIES = {
    "Cow": "Domestic Animal",
    "Deer": "Wild Mammal",
    "Elephant": "Wild Mammal",
    "Monkey": "Wild Mammal",
    "Non_Venomous_Snake": "Snake / Reptile",
    "Venomous_Snake": "Snake / Reptile",
    "Wild_Boar": "Wild Mammal",
}


def friendly_name(name: str) -> str:
    return name.replace("_", " ").replace("Non Venomous", "Non-Venomous")


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.60:
        return "MODERATE"
    return "LOW"


def analysis_message(predicted_class: str, uncertain: bool) -> str:
    if uncertain:
        return "Low-confidence classification — manual verification recommended"
    if predicted_class == "Venomous_Snake":
        return "Potential High-Risk Wildlife"
    if predicted_class == "Non_Venomous_Snake":
        return "Wildlife Detected — Exercise Caution"
    if predicted_class == "Cow":
        return "Domestic Animal Detected"
    return "Wildlife Detected"


@dataclass(frozen=True)
class ImageAnalysisResult:
    predicted_class: str
    confidence: float
    top2_class: str
    top2_probability: float
    margin: float
    confidence_level: str
    status: str
    category: str
    analysis: str
    inference_time_ms: float
    probabilities: tuple[tuple[str, float], ...]

    @property
    def is_snake(self) -> bool:
        return self.predicted_class in {"Venomous_Snake", "Non_Venomous_Snake"}


class ImageInferenceService:
    """Hash-verified, lazily initialized, reusable TFLite interpreter."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._classifier: TFLiteWildlifeClassifier | None = None

    def verify_model(self) -> str:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Validated Float32 model not found: {self.model_path}")
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"Float32 model hash mismatch: expected {EXPECTED_MODEL_SHA256}, received {actual}")
        return actual

    def _get_classifier(self) -> TFLiteWildlifeClassifier:
        if self._classifier is None:
            self.verify_model()
            self._classifier = TFLiteWildlifeClassifier(self.model_path, num_threads=1)
        return self._classifier

    def analyze(self, image_path: Path) -> ImageAnalysisResult:
        classifier = self._get_classifier()
        raw = classifier.predict(image_path)
        probabilities = tuple(zip(classifier.class_names, raw["probabilities"], strict=True))
        ranked = sorted(probabilities, key=lambda item: item[1], reverse=True)
        top1_class, confidence = ranked[0]
        top2_class, top2_probability = ranked[1]
        margin = confidence - top2_probability
        uncertain = confidence < 0.60 or margin < 0.15
        return ImageAnalysisResult(
            predicted_class=top1_class,
            confidence=confidence,
            top2_class=top2_class,
            top2_probability=top2_probability,
            margin=margin,
            confidence_level=confidence_level(confidence),
            status="UNCERTAIN" if uncertain else "CONFIDENT",
            category=CATEGORIES[top1_class],
            analysis=analysis_message(top1_class, uncertain),
            inference_time_ms=float(raw["inference_time_ms"]),
            probabilities=probabilities,
        )
