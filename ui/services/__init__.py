"""Lazy backend adapters; data loading needs no camera or plotting dependency."""
from importlib import import_module
_MODULES = {
    "ImageAnalysisResult": "image_inference", "ImageInferenceService": "image_inference",
    "ModelComparisonRecord": "comparison_data", "load_comparison_records": "comparison_data",
    "ConfusionMatrixRecord": "confusion_data", "load_confusion_records": "confusion_data",
    "LiveAnalysis": "video_inference", "TemporalAnalyzer": "video_inference",
    "VideoInferenceWorker": "video_inference", "DetectionEvent": "detection_history",
    "DetectionHistoryService": "detection_history",
}
__all__ = list(_MODULES)

def __getattr__(name):
    if name in _MODULES:
        value = getattr(import_module("." + _MODULES[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
