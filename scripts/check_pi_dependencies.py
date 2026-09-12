"""Import-only dependency checks: no training, inference, or hardware access."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile


def probe(label: str, code: str, *, optional: bool = False) -> bool:
    """Isolate native imports and report missing/broken modules without crashing."""
    with tempfile.TemporaryDirectory(prefix="anviksa-deps-") as cache:
        env = dict(os.environ, MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE="1")
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code], env=env,
                capture_output=True, text=True, errors="replace", timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            available, detail = False, str(exc)
        else:
            available = result.returncode == 0
            detail = result.stdout.strip() if available else (
                result.stderr.strip() or f"process exit {result.returncode}"
            )
    status = "AVAILABLE" if available else "NOT YET REQUIRED" if optional else "NOT AVAILABLE"
    print(f"{label}: {status}")
    if detail:
        print(f"  {detail}")
    if optional:
        print("  Optional future integration; no device was opened or tested.")
    return available


def main() -> int:
    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform: {platform.system()} {platform.machine()}")
    ok = sys.version_info >= (3, 10)
    print(f"Source Python >=3.10: {'AVAILABLE' if ok else 'NOT AVAILABLE'}")
    if not (platform.system() == "Linux" and platform.machine() == "aarch64"
            and sys.version_info[:2] == (3, 11)):
        print("NOTE: Outside documented Bookworm ARM64 / CPython 3.11 profile.")

    checks = (
        ("NumPy", "import numpy; print(numpy.__version__)"),
        ("OpenCV", "import cv2; print(cv2.__version__); assert callable(cv2.VideoCapture)"),
        ("Pillow / ImageTk", "import PIL; from PIL import Image, ImageTk; "
         "print(PIL.__version__); assert Image.Resampling.NEAREST == 0"),
        ("Matplotlib / TkAgg", "import matplotlib; from matplotlib.figure import Figure; "
         "from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg; "
         "print(matplotlib.__version__)"),
        ("Tkinter", "import tkinter; from tkinter import ttk; print(tkinter.TkVersion)"),
    )
    for label, code in checks:
        ok = probe(label, code) and ok

    interpreter_available = False
    for label, code in (
        ("tflite-runtime", "from tflite_runtime.interpreter import Interpreter; assert callable(Interpreter)"),
        ("LiteRT fallback", "from ai_edge_litert.interpreter import Interpreter; assert callable(Interpreter)"),
        ("TensorFlow development fallback", "import tensorflow as tf; assert callable(tf.lite.Interpreter)"),
    ):
        if probe(label, code):
            interpreter_available = True
            print(f"TFLite interpreter: AVAILABLE via {label}")
            break
    ok = interpreter_available and ok
    if not interpreter_available:
        print("TFLite interpreter: NOT AVAILABLE (install a compatible lightweight runtime)")

    for label, module in (("Pi Camera", "picamera2"), ("GPIO", "gpiozero"),
                          ("GPIO lgpio backend", "lgpio"), ("mmWave UART", "serial"),
                          ("mmWave I2C", "smbus")):
        probe(label, f"import {module}", optional=True)

    print("Display session, model execution, codecs and connected hardware: NOT TESTED")
    print(f"DEPENDENCY SMOKE TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
