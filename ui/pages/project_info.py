"""Project and deployment information available without inference or hardware."""
from __future__ import annotations
import json
import platform
import sys
import os
import subprocess
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
        from ..services.evaluation import load_active_evaluation, format_model_score
        from pi_deployment.inference.tflite_inference import load_class_names
        measured = load_active_evaluation()
        classes = load_class_names()
        self._section("Active classifier", f"MobileNetV3 Large; {len(classes)} output classes; Float32 TFLite. "
                      f"Full frozen test: {measured['correct']}/{measured['total']} correct ({measured['accuracy']:.2%}); Macro F1 {measured['macro_f1']:.2%}. "
                      f"Mean model score: {format_model_score(measured['mean_confidence'])}. Scores are uncalibrated; test accuracy is not field accuracy.")
        self._section(f"{len(classes)} trained output classes", ", ".join(name.replace("_", " ") for name in classes))
        self._dataset()
        if self.title == "System Deployment":
            self._section("This device", f"OS: {platform.system()} {platform.release()}\nArchitecture: {platform.machine()}\nPython: {sys.version.split()[0]}\nDetection history: {self.history_root}")
            try:
                manifest = json.loads((ROOT / "pi_deployment/model_manifest.json").read_text(encoding="utf-8"))
                lines = [f"Classifier: {manifest['architecture']}", "Format: TensorFlow Lite; offline single-frame classification"]
                for model in manifest["models"]:
                    path = ROOT / "pi_deployment" / model["package_path"]
                    lines.append(f"{model['representation']}: {'File available' if path.is_file() else 'File missing'} ({model['file_size_bytes'] / 1048576:.2f} MiB)")
                lines.append("File availability does not verify model integrity. Run: python -B pi_deployment/verify_package.py and python -B scripts/check_dashboard.py")
                self._section("Deployment models", "\n".join(lines))
            except (OSError, ValueError, KeyError) as exc:
                self._section("Deployment models", f"Manifest unavailable: {exc}")
            self._section("Integration status", "Image and video inference and USB webcam controls are implemented. Camera access depends on device configuration.\nPhysical Raspberry Pi benchmarking: pending.\nmmWave trigger, GPIO LED/buzzer, and solar field integration: planned; not hardware-verified.")
            self._section("Local launch", "From the repository root in the Raspberry Pi desktop terminal:\nsource .venv/bin/activate\npython run_ui.py\n\nSetup guide: docs/raspberry_pi_setup.md")
        else:
            self._section("Anviksa AI", "Edge AI-Based Wildlife Perception and Alert Framework. Offline wildlife classification near agricultural land, forest borders, and residential areas.")
            self._section("Perception pipeline", "Motion/presence trigger (planned) -> camera frame -> RGB preprocessing -> MobileNetV3 Large -> class and confidence -> application threat metadata -> local history and snapshot storage.\nThe classifier labels a whole frame; it does not locate animals with bounding boxes.")
            self._section("Research and deployment", "The active 15-class model was trained in run mobilenetv3large_15class_gpu_20261011_123242. Both exports were evaluated on all 670 original hash-verified test images. "
                          "Historical CNN, MobileNetV2, and MobileNetV3 baselines used seven classes and 545 test images; their metrics describe a different evaluation. "
                          "The original wildlife gate is bypassed because its coverage has not been validated for the expanded classes. Raspberry Pi hardware benchmarking remains pending.")
            self._section("Project documentation", "README.md\ndocs/raspberry_pi_setup.md\npi_deployment/README_PI.md")

    def _dataset(self):
        inventory = json.loads((ROOT / "ui/dataset_inventory.json").read_text(encoding="utf-8"))
        folder = ROOT / inventory["project_path"]
        card = SectionCard(self.content, "Dataset inventory")
        card.pack(fill="x", pady=(0, 12))
        totals = {split: sum(counts.values()) for split, counts in inventory["splits"].items()}
        text = f"Project folder: {folder}\n" + ("Local dataset available." if folder.is_dir() else "Images are not installed on this device; the recorded inventory is bundled with the app.")
        text += f"\nRecorded source counts: train {totals['train']}; validation {totals['validation']}; test {totals['test']}. Copy verification: {inventory['copy_sha256_verified_files']} files matched SHA-256. Training-run exclusions are separate from these source counts."
        label = ttk.Label(card, text=text, style="Body.TLabel", wraplength=600)
        label.pack(fill="x")
        card.bind("<Configure>", lambda event: label.configure(wraplength=max(180,event.width-40)))
        table = ttk.Treeview(card, columns=("class", "train", "validation", "test"), show="headings", height=15)
        for column in ("class", "train", "validation", "test"):
            table.heading(column,text=column.title())
            table.column(column,width=180 if column == "class" else 80,minwidth=60,stretch=True)
        for name in inventory["class_names"]:
            table.insert("", "end", values=(name, *(inventory["splits"][split][name] for split in ("train","validation","test"))))
        table.pack(fill="x",pady=8)
        scroll=ttk.Scrollbar(card,orient="horizontal",command=table.xview)
        table.configure(xscrollcommand=scroll.set)
        scroll.pack(fill="x")
        if folder.is_dir():
            ttk.Button(card,text="Open dataset folder",command=lambda:self._open_folder(folder)).pack(anchor="w",pady=(8,0))

    @staticmethod
    def _open_folder(folder):
        if os.name == "nt":
            os.startfile(str(folder))
        else:
            subprocess.Popen(["xdg-open",str(folder)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def _section(self, title: str, text: str) -> None:
        card = SectionCard(self.content, title)
        card.pack(fill="x", pady=(0, 12))
        label = ttk.Label(card, text=text, style="Body.TLabel", justify="left", wraplength=600)
        label.pack(fill="x", anchor="w")
        card.bind("<Configure>", lambda event: label.configure(wraplength=max(180, event.width - 40)))
