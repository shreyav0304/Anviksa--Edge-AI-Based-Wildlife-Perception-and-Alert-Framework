"""Page classes are imported only when requested."""
from importlib import import_module
_MODULES = {
    "HomePage": "home", "ImageAnalysisPage": "image_analysis",
    "VideoAnalysisPage": "video_analysis", "ModelComparisonPage": "model_comparison",
    "ConfusionMatrixPage": "confusion_matrix", "DetectionHistoryPage": "detection_history",
    "ProjectInfoPage": "project_info", "PlaceholderPage": "placeholder",
}
__all__ = list(_MODULES)

def __getattr__(name):
    if name in _MODULES:
        value = getattr(import_module("." + _MODULES[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
