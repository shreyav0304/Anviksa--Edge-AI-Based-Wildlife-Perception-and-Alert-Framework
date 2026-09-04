"""Thin adapters between the UI and validated project backends."""

from .image_inference import ImageAnalysisResult, ImageInferenceService
from .comparison_data import ModelComparisonRecord, load_comparison_records
from .confusion_data import ConfusionMatrixRecord, load_confusion_records
from .video_inference import LiveAnalysis, TemporalAnalyzer, VideoInferenceWorker
from .detection_history import DetectionEvent, DetectionHistoryService

__all__ = ["ImageAnalysisResult", "ImageInferenceService", "ModelComparisonRecord", "load_comparison_records", "ConfusionMatrixRecord", "load_confusion_records", "LiveAnalysis", "TemporalAnalyzer", "VideoInferenceWorker", "DetectionEvent", "DetectionHistoryService"]
