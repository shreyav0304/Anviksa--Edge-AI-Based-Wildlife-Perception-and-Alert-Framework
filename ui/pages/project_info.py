"""Project and deployment information available without inference or hardware."""
from __future__ import annotations
import json
import platform
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from .base import ScrollablePage
from ..components import SectionCard

ROOT = Path(__file__).resolve().parents[2]

class ProjectInfoPage(ScrollablePage):
    MIN_CONTENT_WIDTH = 360
    def __init__(self, parent: tk.Misc, title: str, history_root: Path) -> None:
        super().__init__(parent)
        self.title = title
        self.history_root = history_root
        self.on_show()

    def on_show(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        ttk.Label(self.content, text=self.title, style="PageTitle.TLabel").pack(anchor="w", pady=(0, 16))
        if self.title == "System Deployment":
            self._section("This device", f"OS: {platform.system()} {platform.release()}\nArchitecture: {platform.machine()}\nPython: {sys.version.split()[0]}\nDetection history: {self.history_root}")
            try:
                manifest = json.loads((ROOT / "pi_deployment/model_manifest.json").read_text(encoding="utf-8"))
                lines = [f"Classifier: {manifest['architecture']}", "Format: TensorFlow Lite; offline single-frame classification"]
                for model in manifest["models"]:
                    path = ROOT / "pi_deployment" / model["package_path"]
                    lines.append(f"{model['representation']}: {'File available' if path.is_file() else 'File missing'} ({model['file_size_bytes'] / 1048576:.2f} MiB)")
                lines.append("File availability does not verify model integrity. Run: python -B pi_deployment/verify_package.py")
                self._section("Deployment models", "\n".join(lines))
            except (OSError, ValueError, KeyError) as exc:
                self._section("Deployment models", f"Manifest unavailable: {exc}")
            self._section("Integration status", "Image and video inference and USB webcam controls are implemented. Camera access depends on device configuration.\nPhysical Raspberry Pi benchmarking: pending.\nmmWave trigger, GPIO LED/buzzer, and solar field integration: planned; not hardware-verified.")
            self._section("Local launch", "From the repository root in the Raspberry Pi desktop terminal:\nsource .venv/bin/activate\npython run_ui.py\n\nSetup guide: docs/raspberry_pi_setup.md")
        else:
            self._section("Anviksa AI", "Edge AI-Based Wildlife Perception and Alert Framework. Offline wildlife classification near agricultural land, forest borders, and residential areas.")
            self._section("Perception pipeline", "Motion/presence trigger (planned) -> camera frame -> RGB preprocessing -> MobileNetV3 Large -> class and confidence -> application threat metadata -> local history and snapshot storage.\nThe classifier labels a whole frame; it does not locate animals with bounding boxes.")
            self._section("Seven classes", "Cow, Deer, Elephant, Monkey, Non-Venomous Snake, Venomous Snake, Wild Boar.")
            self._section("Research and deployment", "Custom CNN, MobileNetV2, and MobileNetV3 Large were evaluated on the same frozen 545-image test split. EfficientNetB0 training was incomplete and has no test matrix.\nMobileNetV3 Large: 95.41% test accuracy and 96.87% Macro F1. Float32 and Float16 TFLite models are validated; Raspberry Pi performance measurements remain pending.")
            self._section("Project documentation", "README.md\ndocs/raspberry_pi_setup.md\npi_deployment/README_PI.md")

    def _section(self, title: str, text: str) -> None:
        card = SectionCard(self.content, title)
        card.pack(fill="x", pady=(0, 12))
        label = ttk.Label(card, text=text, style="Body.TLabel", justify="left", wraplength=600)
        label.pack(fill="x", anchor="w")
        card.bind("<Configure>", lambda event: label.configure(wraplength=max(180, event.width - 40)))
