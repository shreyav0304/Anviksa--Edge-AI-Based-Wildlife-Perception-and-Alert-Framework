"""Single-image classification page using the validated Float32 TFLite model."""

from __future__ import annotations

import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, UnidentifiedImageError

from ..components import SectionCard
from ..services import ImageAnalysisResult, ImageInferenceService
from ..services.detection_history import DetectionHistoryService
from ..services.image_inference import friendly_name
from .base import ScrollablePage


SUPPORTED_FORMATS = (("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png"), ("Bitmap", "*.bmp"), ("WebP", "*.webp"))


class ImageAnalysisPage(ScrollablePage):
    def __init__(self, parent: tk.Misc, history: DetectionHistoryService) -> None:
        super().__init__(parent)
        self.selected_path: Path | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._service = ImageInferenceService()
        self.history = history
        self._last_result: ImageAnalysisResult | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anviksa-tflite")
        self._running = False
        self._build()

    def _build(self) -> None:
        ttk.Label(self.content, text="WHOLE-IMAGE CLASSIFICATION", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Image Analysis", style="PageTitle.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(self.content, text="Analyze one local image with the validated MobileNetV3 Large Float32 TFLite classifier.", style="Muted.TLabel").pack(anchor="w")

        toolbar = ttk.Frame(self.content, style="Page.TFrame"); toolbar.pack(fill="x", pady=(16, 12))
        self.select_button = ttk.Button(toolbar, text="Select Image", command=self.select_image, style="Primary.TButton")
        self.select_button.pack(side="left")
        self.analyze_button = ttk.Button(toolbar, text="Analyze Image", command=self.analyze_image, state="disabled")
        self.analyze_button.pack(side="left", padx=8)
        self.clear_button = ttk.Button(toolbar, text="Clear", command=self.clear)
        self.clear_button.pack(side="left")
        self.save_button = ttk.Button(toolbar, text="Save Detection", command=self.save_detection, state="disabled")
        self.save_button.pack(side="left", padx=(8, 0))
        self.activity = ttk.Label(toolbar, text="Select an image to begin", style="Muted.TLabel")
        self.activity.pack(side="left", padx=16)

        body = ttk.Frame(self.content, style="Page.TFrame"); body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3, uniform="body"); body.columnconfigure(1, weight=2, uniform="body")
        preview_card = SectionCard(body, "Image preview"); preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.preview = tk.Label(preview_card, text="No image selected", bg="#e9efec", fg="#718078", font=("Segoe UI", 12), width=60, height=20, compound="center")
        self.preview.pack(fill="both", expand=True)
        self.filename = ttk.Label(preview_card, text="", style="MetricDetail.TLabel", wraplength=650)
        self.filename.pack(anchor="w", pady=(9, 0))

        result_card = SectionCard(body, "Analysis result"); result_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.result_vars = {name: tk.StringVar(value="—") for name in ("Prediction", "Confidence", "Category", "Status", "Top-2 Prediction", "Top-2 Probability", "Prediction Margin", "Inference Time")}
        for label, variable in self.result_vars.items():
            line = ttk.Frame(result_card, style="Card.TFrame"); line.pack(fill="x", pady=4)
            ttk.Label(line, text=label.upper(), style="MetricLabel.TLabel").pack(anchor="w")
            ttk.Label(line, textvariable=variable, style="ResultValue.TLabel", wraplength=400).pack(anchor="w", pady=(1, 0))
        self.analysis_var = tk.StringVar(value="Awaiting analysis")
        ttk.Separator(result_card).pack(fill="x", pady=10)
        ttk.Label(result_card, text="ANALYSIS", style="MetricLabel.TLabel").pack(anchor="w")
        ttk.Label(result_card, textvariable=self.analysis_var, style="BodyStrong.TLabel", wraplength=410).pack(anchor="w", pady=(3, 0))
        self.snake_warning = ttk.Label(result_card, text="Do not approach or handle an unidentified snake. AI classification is not a substitute for expert species identification.", style="Warning.TLabel", wraplength=410)

        probability_card = SectionCard(self.content, "All-class probabilities"); probability_card.pack(fill="x", pady=(12, 0))
        self.probability_widgets: dict[str, tuple[ttk.Progressbar, tk.StringVar]] = {}
        for class_name in self._class_names():
            row = ttk.Frame(probability_card, style="Card.TFrame"); row.pack(fill="x", pady=3)
            ttk.Label(row, text=friendly_name(class_name), style="Body.TLabel", width=23).pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, value=0); bar.pack(side="left", fill="x", expand=True, padx=8)
            value = tk.StringVar(value="0.00%")
            ttk.Label(row, textvariable=value, style="BodyStrong.TLabel", width=9, anchor="e").pack(side="right")
            self.probability_widgets[class_name] = (bar, value)

        footer = ttk.Frame(self.content, style="Card.TFrame", padding=14); footer.pack(fill="x", pady=(12, 0))
        ttk.Label(footer, text="MODEL", style="MetricLabel.TLabel").pack(side="left")
        ttk.Label(footer, text="MobileNetV3 Large — Float32 TFLite", style="BodyStrong.TLabel").pack(side="left", padx=10)
        ttk.Label(footer, text="The current classifier always selects one of seven trained classes. Predictions on images without a supported animal should be treated as exploratory.", style="Notice.TLabel", wraplength=650).pack(side="right")

    @staticmethod
    def _class_names() -> tuple[str, ...]:
        from pi_deployment.inference.tflite_inference import load_class_names
        return load_class_names()

    def select_image(self) -> None:
        filename = filedialog.askopenfilename(parent=self, title="Select wildlife image", filetypes=SUPPORTED_FORMATS)
        if not filename:
            return
        path = Path(filename)
        try:
            with Image.open(path) as image:
                image.load()
                preview = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            messagebox.showerror("Unable to open image", f"The selected file is unsupported, unreadable, or invalid.\n\n{exc}", parent=self)
            return
        preview.thumbnail((700, 430), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(preview)
        self.preview.configure(image=self._preview_photo, text="", width=1, height=1)
        self.filename.configure(text=str(path))
        self.selected_path = path
        self.analyze_button.configure(state="normal")
        self.activity.configure(text="Image ready for analysis")
        self._reset_results()

    def analyze_image(self) -> None:
        if self.selected_path is None or self._running:
            return
        self._running = True
        self.select_button.configure(state="disabled"); self.analyze_button.configure(state="disabled")
        self.activity.configure(text="Analyzing…")
        future = self._executor.submit(self._service.analyze, self.selected_path)
        self.after(40, self._poll_analysis, future)

    def _poll_analysis(self, future: Future[ImageAnalysisResult]) -> None:
        """Poll on Tk's main thread; worker threads never touch widgets."""
        if future.done():
            self._finish_analysis(future)
        else:
            self.after(40, self._poll_analysis, future)

    def _finish_analysis(self, future: Future[ImageAnalysisResult]) -> None:
        self._running = False
        self.select_button.configure(state="normal")
        self.analyze_button.configure(state="normal" if self.selected_path else "disabled")
        try:
            result = future.result()
        except Exception as exc:
            self.activity.configure(text="Analysis failed")
            messagebox.showerror("Image analysis failed", f"The classifier could not analyze this image.\n\n{exc}", parent=self)
            return
        self._show_result(result)
        self.activity.configure(text="Analysis complete")

    def _show_result(self, result: ImageAnalysisResult) -> None:
        self._last_result = result
        values = {
            "Prediction": friendly_name(result.predicted_class),
            "Confidence": f"{result.confidence * 100:.2f}% ({result.confidence_level})",
            "Category": result.category,
            "Status": result.status,
            "Top-2 Prediction": friendly_name(result.top2_class),
            "Top-2 Probability": f"{result.top2_probability * 100:.2f}%",
            "Prediction Margin": f"{result.margin * 100:.2f} percentage points",
            "Inference Time": f"{result.inference_time_ms:.2f} ms",
        }
        for key, value in values.items(): self.result_vars[key].set(value)
        self.analysis_var.set(result.analysis)
        for class_name, probability in result.probabilities:
            bar, text = self.probability_widgets[class_name]
            bar.configure(value=probability * 100); text.set(f"{probability * 100:.2f}%")
        if result.is_snake: self.snake_warning.pack(fill="x", pady=(12, 0))
        else: self.snake_warning.pack_forget()
        self.save_button.configure(state="normal")

    def _reset_results(self) -> None:
        self._last_result = None
        self.save_button.configure(state="disabled")
        for variable in self.result_vars.values(): variable.set("—")
        self.analysis_var.set("Awaiting analysis")
        self.snake_warning.pack_forget()
        for bar, text in self.probability_widgets.values(): bar.configure(value=0); text.set("0.00%")

    def clear(self) -> None:
        if self._running:
            return
        self.selected_path = None; self._preview_photo = None
        self.preview.configure(image="", text="No image selected", width=60, height=20)
        self.filename.configure(text="")
        self.analyze_button.configure(state="disabled")
        self.activity.configure(text="Select an image to begin")
        self._reset_results()

    def save_detection(self) -> None:
        result = self._last_result
        if result is None or self.selected_path is None:
            return
        try:
            event = self.history.log_image(
                self.selected_path,
                predicted_class=result.predicted_class,
                display_class=friendly_name(result.predicted_class),
                category=result.category,
                confidence=result.confidence,
                top2_class=result.top2_class,
                top2_probability=result.top2_probability,
                margin=result.margin,
                confidence_status=result.confidence_level,
                uncertain=result.status == "UNCERTAIN",
                inference_time_ms=result.inference_time_ms,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Save detection failed", str(exc), parent=self)
            return
        self.save_button.configure(state="disabled")
        self.activity.configure(text=f"Detection saved: {event.event_id}")

    def close(self) -> None:
        """Release the page worker without interrupting an active Lite call."""
        self._executor.shutdown(wait=False, cancel_futures=True)
