"""Application pages."""

from .home import HomePage
from .image_analysis import ImageAnalysisPage
from .model_comparison import ModelComparisonPage
from .confusion_matrix import ConfusionMatrixPage
from .video_analysis import VideoAnalysisPage
from .detection_history import DetectionHistoryPage
from .placeholder import PlaceholderPage

__all__ = ["HomePage", "ImageAnalysisPage", "VideoAnalysisPage", "ModelComparisonPage", "ConfusionMatrixPage", "DetectionHistoryPage", "PlaceholderPage"]
