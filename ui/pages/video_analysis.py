"""Recorded-video and webcam whole-frame classification page."""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ..components import SectionCard
from ..services.image_inference import friendly_name
from ..services.image_inference import CATEGORIES
from ..services.detection_history import DetectionHistoryService
from ..services.video_inference import VideoInferenceWorker, VideoSourceInfo, VideoUpdate, inspect_video
from .base import ScrollablePage


VIDEO_FORMATS = (("Video files", "*.mp4 *.avi *.mov *.mkv"), ("MP4", "*.mp4"), ("AVI", "*.avi"), ("MOV", "*.mov"), ("Matroska", "*.mkv"))


class VideoAnalysisPage(ScrollablePage):
    def __init__(self, parent: tk.Misc, history: DetectionHistoryService) -> None:
        super().__init__(parent)
        self.source_mode = tk.StringVar(value="recorded")
        self.camera_index = tk.StringVar(value="0")
        self.frame_skip = tk.StringVar(value="1")
        self.smoothing_window = tk.StringVar(value="5")
        self.stable_frames = tk.StringVar(value="3")
        self.video_path: Path | None = None
        self.source_info: VideoSourceInfo | None = None
        self.history = history
        self.worker: VideoInferenceWorker | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._poll_id: str | None = None
        self._last_frame = None
        self._build()

    def _build(self) -> None:
        ttk.Label(self.content, text="WHOLE-FRAME CLASSIFICATION", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Video / Webcam Analysis", style="PageTitle.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(self.content, text="Local recorded-video and laptop/USB webcam inference with MobileNetV3 Large Float32 TFLite.", style="Muted.TLabel").pack(anchor="w")
        controls = ttk.Frame(self.content, style="Card.TFrame", padding=14); controls.pack(fill="x", pady=(14, 10))
        ttk.Label(controls, text="SOURCE", style="MetricLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Recorded Video", variable=self.source_mode, value="recorded", command=self._mode_changed).grid(row=0, column=1, padx=6)
        ttk.Radiobutton(controls, text="Webcam", variable=self.source_mode, value="webcam", command=self._mode_changed).grid(row=0, column=2, padx=6)
        self.select_button = ttk.Button(controls, text="Select Video", command=self.select_video, style="Primary.TButton"); self.select_button.grid(row=0, column=3, padx=(16, 5))
        ttk.Label(controls, text="Camera", style="MetricLabel.TLabel").grid(row=0, column=4, padx=(16, 4))
        self.camera_box = ttk.Combobox(controls, textvariable=self.camera_index, values=("0", "1", "2"), state="disabled", width=4); self.camera_box.grid(row=0, column=5)
        ttk.Label(controls, text="Frame Skip", style="MetricLabel.TLabel").grid(row=0, column=6, padx=(16, 4))
        ttk.Combobox(controls, textvariable=self.frame_skip, values=("1", "2", "3", "5"), state="readonly", width=4).grid(row=0, column=7)
        ttk.Label(controls, text="Smooth", style="MetricLabel.TLabel").grid(row=0, column=8, padx=(16, 4))
        ttk.Combobox(controls, textvariable=self.smoothing_window, values=("1", "3", "5", "7"), state="readonly", width=4).grid(row=0, column=9)
        ttk.Label(controls, text="Stable", style="MetricLabel.TLabel").grid(row=0, column=10, padx=(16, 4))
        ttk.Combobox(controls, textvariable=self.stable_frames, values=("2", "3", "5"), state="readonly", width=4).grid(row=0, column=11)
        actions = ttk.Frame(self.content, style="Page.TFrame"); actions.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(actions, text="Start", command=self.start, style="Primary.TButton"); self.start_button.pack(side="left")
        self.pause_button = ttk.Button(actions, text="Pause", command=self.pause, state="disabled"); self.pause_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state="disabled"); self.stop_button.pack(side="left")
        self.snapshot_button = ttk.Button(actions, text="Save Current Frame", command=self.save_current_frame, state="disabled"); self.snapshot_button.pack(side="left", padx=(8, 0))
        self.session_status = tk.StringVar(value="Select a recorded video or choose Webcam")
        ttk.Label(actions, textvariable=self.session_status, style="Muted.TLabel").pack(side="left", padx=14)

        body = ttk.Frame(self.content, style="Page.TFrame"); body.pack(fill="x"); body.columnconfigure(0, weight=3, uniform="video"); body.columnconfigure(1, weight=2, uniform="video")
        preview_card = SectionCard(body, "Live frame"); preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.preview = tk.Label(preview_card, text="Video preview", bg="#101915", fg="#b8c8c0", font=("Segoe UI", 12), width=60, height=22)
        self.preview.pack(fill="both", expand=True)
        self.source_details = tk.StringVar(value="No source selected")
        ttk.Label(preview_card, textvariable=self.source_details, style="MetricDetail.TLabel", wraplength=680).pack(anchor="w", pady=(8, 0))
        ttk.Label(preview_card, text="Classification Mode: Whole Frame — no bounding boxes", style="Notice.TLabel").pack(anchor="w", pady=(4, 0))
        result_card = SectionCard(body, "Current analysis"); result_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        names = ("Prediction", "Confidence", "Top-2", "Margin", "Status", "Stable Count", "Inference", "Processed FPS", "Analysis")
        self.result_vars = {name: tk.StringVar(value="—") for name in names}
        for name in names:
            row = ttk.Frame(result_card, style="Card.TFrame"); row.pack(fill="x", pady=3)
            ttk.Label(row, text=name.upper(), style="MetricLabel.TLabel").pack(anchor="w")
            ttk.Label(row, textvariable=self.result_vars[name], style="ResultValue.TLabel", wraplength=400).pack(anchor="w")
        self.snake_warning = ttk.Label(result_card, text="Do not approach or handle an unidentified snake. AI classification is not a substitute for expert identification.", style="Warning.TLabel", wraplength=420)

        stats = SectionCard(self.content, "Session statistics"); stats.pack(fill="x", pady=(12, 0))
        self.stat_vars = {name: tk.StringVar(value="0") for name in ("Frames Captured", "Frames Processed", "Source FPS", "Processed FPS", "Average Inference", "Stable Events")}
        for name, variable in self.stat_vars.items():
            cell = ttk.Frame(stats, style="Card.TFrame"); cell.pack(side="left", fill="x", expand=True, padx=6)
            ttk.Label(cell, text=name.upper(), style="MetricLabel.TLabel").pack(anchor="w")
            ttk.Label(cell, textvariable=variable, style="BodyStrong.TLabel").pack(anchor="w", pady=(3, 0))
        limitation = SectionCard(self.content, "Model limitation"); limitation.pack(fill="x", pady=(12, 0))
        ttk.Label(limitation, text="Current model has no Background/Unknown class. Empty-scene webcam predictions should be treated as exploratory.", style="Warning.TLabel", wraplength=1050).pack(fill="x")
        ttk.Label(limitation, text="Temporary stable events are held in memory only. Detection History persistence and runtime snapshots are deferred to Phase 6.", style="Notice.TLabel", wraplength=1050).pack(anchor="w", pady=(8, 0))

    def _mode_changed(self) -> None:
        self.stop()
        webcam = self.source_mode.get() == "webcam"
        self.camera_box.configure(state="readonly" if webcam else "disabled")
        self.select_button.configure(state="disabled" if webcam else "normal")
        self.session_status.set("Webcam ready" if webcam else "Select a recorded video")
        self.source_details.set(f"Camera index {self.camera_index.get()}" if webcam else "No video selected")

    def select_video(self) -> None:
        filename = filedialog.askopenfilename(parent=self, title="Select recorded video", filetypes=VIDEO_FORMATS)
        if not filename: return
        path = Path(filename)
        try: info = inspect_video(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Unable to open video", str(exc), parent=self); return
        self.video_path = path; self.source_info = info
        duration = f"{info.duration_seconds:.1f}s" if info.duration_seconds is not None else "N/A"
        fps = f"{info.fps:.2f}" if info.fps > 0 else "N/A"
        self.source_details.set(f"{path.name}  •  {info.width}×{info.height}  •  {fps} FPS  •  Duration {duration}")
        self.session_status.set("Video ready")

    def start(self) -> None:
        if self.worker and self.worker.running:
            self.worker.resume(); self.session_status.set("Running"); self.pause_button.configure(text="Pause"); return
        source = int(self.camera_index.get()) if self.source_mode.get() == "webcam" else self.video_path
        if source is None:
            messagebox.showinfo("Select a video", "Choose a recorded video before starting.", parent=self); return
        self.worker = VideoInferenceWorker(source, int(self.frame_skip.get()), int(self.smoothing_window.get()), int(self.stable_frames.get()))
        self.worker.start(); self.session_status.set("Starting…")
        self.start_button.configure(state="disabled"); self.pause_button.configure(state="normal", text="Pause"); self.stop_button.configure(state="normal")
        self._schedule_poll()

    def pause(self) -> None:
        if not self.worker or not self.worker.running: return
        if self.worker.pause_event.is_set():
            self.worker.resume(); self.pause_button.configure(text="Pause"); self.session_status.set("Running")
        else:
            self.worker.pause(); self.pause_button.configure(text="Resume"); self.session_status.set("Paused")

    def stop(self, preserve_status: bool = False) -> None:
        if self.worker: self.worker.stop()
        self.worker = None
        if self._poll_id:
            try: self.after_cancel(self._poll_id)
            except tk.TclError: pass
            self._poll_id = None
        self.start_button.configure(state="normal"); self.pause_button.configure(state="disabled", text="Pause"); self.stop_button.configure(state="disabled")
        if not preserve_status: self.session_status.set("Stopped")

    def _schedule_poll(self) -> None:
        if self._poll_id is None: self._poll_id = self.after(30, self._poll)

    def _poll(self) -> None:
        self._poll_id = None
        worker = self.worker
        if worker is None: return
        latest = None
        while True:
            try: latest = worker.updates.get_nowait()
            except queue.Empty: break
        if latest is not None: self._apply_update(latest)
        if self.worker is worker and worker.running: self._schedule_poll()

    def _apply_update(self, update: VideoUpdate) -> None:
        if update.frame_rgb is not None: self._show_frame(update.frame_rgb)
        self.stat_vars["Frames Captured"].set(str(update.frames_captured)); self.stat_vars["Frames Processed"].set(str(update.frames_processed))
        self.stat_vars["Source FPS"].set(f"{update.source_fps:.2f}"); self.stat_vars["Processed FPS"].set(f"{update.processed_fps:.2f}")
        self.stat_vars["Average Inference"].set(f"{update.average_inference_ms:.2f} ms"); self.stat_vars["Stable Events"].set(str(update.stable_events))
        if update.analysis is not None:
            self._show_analysis(update.analysis)
            self._persist_eligible_event(update)
        if update.kind in {"complete", "error"}:
            self.session_status.set(update.message)
            if update.kind == "error": messagebox.showerror("Video source error", update.message, parent=self)
            self.stop(preserve_status=True)
        elif self.session_status.get() == "Starting…": self.session_status.set("Running")

    def _show_frame(self, frame_rgb) -> None:
        self._last_frame = frame_rgb
        image = Image.fromarray(frame_rgb); image.thumbnail((720, 440), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self._preview_photo, text="", width=1, height=1)
        self.snapshot_button.configure(state="normal")

    def _show_analysis(self, result) -> None:
        analysis = "Potential High-Risk Wildlife" if result.predicted_class == "Venomous_Snake" and result.status != "UNCERTAIN" else "Low-confidence classification — manual verification recommended" if result.status == "UNCERTAIN" else "Wildlife Detected"
        values = {"Prediction": friendly_name(result.predicted_class), "Confidence": f"{result.confidence*100:.2f}% ({result.confidence_level})", "Top-2": f"{friendly_name(result.top2_class)} — {result.top2_probability*100:.2f}%", "Margin": f"{result.margin*100:.2f} percentage points", "Status": result.status, "Stable Count": str(result.stable_count), "Inference": f"{result.inference_time_ms:.2f} ms", "Processed FPS": self.stat_vars["Processed FPS"].get(), "Analysis": analysis}
        for key, value in values.items(): self.result_vars[key].set(value)
        if result.predicted_class in {"Venomous_Snake", "Non_Venomous_Snake"}: self.snake_warning.pack(fill="x", pady=(10, 0))
        else: self.snake_warning.pack_forget()

    def _source_identity(self) -> tuple[str, str]:
        if self.source_mode.get() == "webcam":
            return "WEBCAM", f"Camera {self.camera_index.get()}"
        return "VIDEO", self.video_path.name if self.video_path else "Recorded Video"

    def _persist_eligible_event(self, update: VideoUpdate) -> None:
        result = update.analysis
        if result is None or self._last_frame is None:
            return
        source_type, source_name = self._source_identity()
        video_time = f"{update.frames_captured/update.source_fps:.6f}" if source_type == "VIDEO" and update.source_fps > 0 else ""
        try:
            event = self.history.log_stable_frame(
                Image.fromarray(self._last_frame), source_type=source_type, source_name=source_name,
                predicted_class=result.predicted_class, display_class=friendly_name(result.predicted_class),
                category=CATEGORIES[result.predicted_class], confidence=result.confidence,
                top2_class=result.top2_class, top2_probability=result.top2_probability,
                margin=result.margin, confidence_status=result.confidence_level,
                uncertain=result.status == "UNCERTAIN", stable=result.stable,
                stable_count=result.stable_count, inference_time_ms=result.inference_time_ms,
                frame_number=str(update.frames_captured), video_time_seconds=video_time,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.session_status.set(f"History logging warning: {exc}")
            return
        if event is not None:
            self.session_status.set(f"Stable event saved: {event.display_class}")

    def save_current_frame(self) -> None:
        if self._last_frame is None:
            messagebox.showinfo("No current frame", "No valid current frame is available.", parent=self)
            return
        source_type, source_name = self._source_identity()
        try:
            relative = self.history.save_snapshot(Image.fromarray(self._last_frame), source_type, f"manual_{source_name}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Snapshot failed", str(exc), parent=self)
            return
        self.session_status.set(f"Runtime snapshot saved: {relative}")

    def on_hide(self) -> None:
        self.stop()

    def close(self) -> None:
        self.stop()
