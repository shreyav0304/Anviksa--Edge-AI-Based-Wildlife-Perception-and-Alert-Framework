"""Load the optional camera/video dependency at the point of use."""
from importlib import import_module

def require_opencv():
    try:
        return import_module("cv2")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "OpenCV is missing or cannot load. Activate the app environment and "
            "install its dependencies: python -m pip install -r requirements.txt. "
            "See docs/raspberry_pi_setup.md for the supported Pi profile."
        ) from exc
