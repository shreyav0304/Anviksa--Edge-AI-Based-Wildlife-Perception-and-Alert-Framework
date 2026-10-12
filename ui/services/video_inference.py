"""Responsive recorded-video/webcam inference using the validated TFLite backend."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .opencv import require_opencv
import numpy as np

from pi_deployment.inference.tflite_inference import TFLiteWildlifeClassifier
from .image_inference import ImageInferenceService, MODEL_PATH, confidence_level
from .wildlife_gate import WildlifeGateResult, WildlifeGateService


MAX_UNCERTAIN_FRAMES = 3


@dataclass(frozen=True)
class VideoSourceInfo:
    label: str
    fps: float
    width: int
    height: int
    frame_count: int | None
    duration_seconds: float | None


@dataclass(frozen=True)
class LiveAnalysis:
    predicted_class: str
    confidence: float
    top2_class: str
    top2_probability: float
    margin: float
    confidence_level: str
    status: str
    stable_count: int
    stable: bool
    inference_time_ms: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class VideoUpdate:
    kind: str
    frame_rgb: np.ndarray | None = None
    analysis: LiveAnalysis | None = None
    gate: WildlifeGateResult | None = None
    gate_timed_out: bool = False
    episode_key: str = ""
    frames_captured: int = 0
    frames_processed: int = 0
    source_fps: float = 0.0
    processed_fps: float = 0.0
    average_inference_ms: float = 0.0
    stable_events: int = 0
    message: str = ""


class TemporalAnalyzer:
    """Pure probability smoothing and stable-result state logic."""

    def __init__(self, class_names: tuple[str, ...], window: int = 5, stable_frames: int = 3) -> None:
        if window < 1 or stable_frames < 1:
            raise ValueError("Smoothing window and stable frames must be positive")
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("At least two unique classes are required")
        self.class_names = class_names
        self.window = window
        self.stable_frames = stable_frames
        self.history: deque[np.ndarray] = deque(maxlen=window)
        self.last_class: str | None = None
        self.stable_count = 0
        self.stable_events = 0
        self._episode_reported = False

    def reset(self) -> None:
        """Discard previous species evidence and restart the stable-frame run."""
        self.history.clear()
        self.last_class = None
        self.stable_count = 0
        self._episode_reported = False

    def update(self, probabilities: np.ndarray, inference_time_ms: float) -> LiveAnalysis:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (len(self.class_names),) or not np.isfinite(values).all():
            raise ValueError("Expected one finite probability per class")
        if np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(), 1.0, atol=1e-4):
            raise ValueError("Probabilities must lie in [0, 1] and sum to one")
        if not np.isfinite(inference_time_ms) or inference_time_ms < 0:
            raise ValueError("Inference latency must be finite and nonnegative")
        self.history.append(values.copy())
        smoothed = np.mean(np.stack(self.history), axis=0)
        order = np.argsort(-smoothed, kind="stable")
        top1, top2 = int(order[0]), int(order[1])
        predicted = self.class_names[top1]
        confidence = float(smoothed[top1])
        top2_probability = float(smoothed[top2])
        margin = confidence - top2_probability
        if predicted == self.last_class:
            self.stable_count += 1
        else:
            self.last_class = predicted
            self.stable_count = 1
            self._episode_reported = False
        stable = self.stable_count >= self.stable_frames
        if stable and confidence >= .80 and not self._episode_reported:
            self.stable_events += 1
            self._episode_reported = True
        uncertain = confidence < .60 or margin < .15
        return LiveAnalysis(
            predicted_class=predicted, confidence=confidence,
            top2_class=self.class_names[top2], top2_probability=top2_probability,
            margin=margin, confidence_level=confidence_level(confidence),
            status="UNCERTAIN" if uncertain else "STABLE" if stable else "TRACKING",
            stable_count=self.stable_count, stable=stable,
            inference_time_ms=float(inference_time_ms),
            probabilities=tuple(float(value) for value in smoothed),
        )


class GateTemporalState:
    """Count selected uncertain frames while holding species state for at most three."""

    def __init__(self, max_uncertain_frames: int = MAX_UNCERTAIN_FRAMES) -> None:
        if max_uncertain_frames < 1:
            raise ValueError("Maximum uncertain frames must be positive")
        self.max_uncertain_frames = max_uncertain_frames
        self.uncertain_count = 0

    def observe(self, decision: str, analyzer: TemporalAnalyzer) -> tuple[bool, bool]:
        """Return (timed_out, ended_sighting); never classify an uncertain frame."""
        if decision == "WILDLIFE":
            self.uncertain_count = 0
            return False, False
        if decision == "NON_WILDLIFE":
            self.uncertain_count = 0
            ended_sighting = analyzer.last_class is not None
            analyzer.reset()
            return False, ended_sighting
        if decision != "UNCERTAIN":
            raise ValueError(f"Unknown Wildlife Gate decision: {decision}")
        self.uncertain_count += 1
        if self.uncertain_count < self.max_uncertain_frames:
            return False, False
        self.uncertain_count = 0
        ended_sighting = analyzer.last_class is not None
        analyzer.reset()
        return True, ended_sighting


def inspect_video(path: Path) -> VideoSourceInfo:
    """Read container metadata and release the source immediately."""
    cv2 = require_opencv()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("Video has invalid frame dimensions")
        duration = count / fps if count > 0 and fps > 0 else None
        return VideoSourceInfo(path.name, fps, width, height, count if count > 0 else None, duration)
    finally:
        capture.release()


class VideoInferenceWorker:
    """One capture/inference worker; consumers poll ``updates`` from Tk."""

    def __init__(self, source: str | Path | int, frame_skip: int = 1, smoothing_window: int = 5, stable_frames: int = 3) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (frame_skip, smoothing_window, stable_frames)):
            raise ValueError("Frame skip, smoothing window, and stable frames must be positive integers")
        self.source = Path(source) if isinstance(source, str) else source
        self.frame_skip = frame_skip
        self.smoothing_window = smoothing_window
        self.stable_frames = stable_frames
        self.updates: queue.Queue[VideoUpdate] = queue.Queue(maxsize=3)
        # Preview frames may be replaced; eligible history events must not be.
        self.event_updates: queue.Queue[VideoUpdate] = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._capture: Any = None

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def start(self) -> None:
        if self.running:
            self.pause_event.clear()
            return
        self.stop_event.clear(); self.pause_event.clear()
        self.thread = threading.Thread(target=self._run, name="anviksa-video", daemon=True)
        self.thread.start()

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def stop(self) -> None:
        self.stop_event.set(); self.pause_event.clear()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)
            # Only the capture thread may release its native VideoCapture.
            # Releasing concurrently with read() can crash OpenCV.

    def _publish(self, update: VideoUpdate) -> None:
        while True:
            try:
                self.updates.put_nowait(update); return
            except queue.Full:
                try: self.updates.get_nowait()
                except queue.Empty: continue

    def _publish_event(self, update: VideoUpdate) -> bool:
        while not self.stop_event.is_set():
            try:
                self.event_updates.put(update, timeout=.05)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        capture = None
        try:
            cv2 = require_opencv()
            capture = cv2.VideoCapture(str(self.source) if isinstance(self.source, Path) else self.source)
            self._capture = capture
            if not capture.isOpened():
                self._publish(VideoUpdate("error", message="Camera or video source could not be opened.")); return
            ImageInferenceService(MODEL_PATH).verify_model()
            gate_service = WildlifeGateService(num_threads=1)
            classifier = TFLiteWildlifeClassifier(MODEL_PATH, num_threads=1)
            analyzer = TemporalAnalyzer(classifier.class_names, self.smoothing_window, self.stable_frames)
            gate_temporal = GateTemporalState()
            frames_captured = frames_processed = 0; latency_total = 0.0
            current_gate: WildlifeGateResult | None = None
            current_gate_timed_out = False
            episode_key = uuid.uuid4().hex
            emitted_classes: set[str] = set()
            source_fps = float(capture.get(cv2.CAP_PROP_FPS)); started = time.perf_counter()
            recorded = isinstance(self.source, Path)
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    time.sleep(.03); continue
                frame_started = time.perf_counter()
                ok, frame_bgr = capture.read()
                if not ok:
                    kind = "complete" if recorded else "error"
                    message = "Video Complete" if recorded else "Camera disconnected or frame read failed."
                    self._publish(VideoUpdate(kind, frames_captured=frames_captured, frames_processed=frames_processed, source_fps=source_fps, processed_fps=frames_processed/max(time.perf_counter()-started,1e-6), average_inference_ms=latency_total/max(frames_processed,1), stable_events=analyzer.stable_events, message=message)); return
                if self.stop_event.is_set():
                    break
                frames_captured += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                analysis = None
                gate = current_gate
                gate_timed_out = current_gate_timed_out
                if frames_captured % self.frame_skip == 0:
                    gate = gate_service.predict(frame_rgb)
                    gate_timed_out, ended_sighting = gate_temporal.observe(gate.decision, analyzer)
                    if ended_sighting:
                        episode_key = uuid.uuid4().hex
                        emitted_classes.clear()
                    if gate.decision == "WILDLIFE":
                        raw = classifier.predict(frame_rgb)
                        raw_probabilities = np.asarray(raw["probabilities"], dtype=np.float64)
                        analysis = analyzer.update(raw_probabilities, float(raw["inference_time_ms"]))
                        inference_ms = gate.inference_time_ms + analysis.inference_time_ms
                    else:
                        inference_ms = gate.inference_time_ms
                    if self.stop_event.is_set():
                        break
                    current_gate = gate
                    current_gate_timed_out = gate_timed_out
                    frames_processed += 1; latency_total += inference_ms
                elapsed = max(time.perf_counter() - started, 1e-6)
                update = VideoUpdate("frame", frame_rgb=frame_rgb, analysis=analysis, gate=gate, gate_timed_out=gate_timed_out, episode_key=episode_key, frames_captured=frames_captured, frames_processed=frames_processed, source_fps=source_fps, processed_fps=frames_processed/elapsed, average_inference_ms=latency_total/max(frames_processed,1), stable_events=analyzer.stable_events)
                if (analysis is not None and analysis.stable and analysis.status != "UNCERTAIN"
                        and analysis.confidence >= .80 and analysis.predicted_class not in emitted_classes):
                    if self._publish_event(update):
                        emitted_classes.add(analysis.predicted_class)
                self._publish(update)
                if recorded and source_fps > 0:
                    time.sleep(max(0.0, 1.0/source_fps - (time.perf_counter()-frame_started)))
        except Exception as exc:
            self._publish(VideoUpdate("error", message=f"Video inference failed: {exc}"))
        finally:
            if capture is not None:
                capture.release()
            self._capture = None
