"""Thread-safe local detection history and runtime snapshot persistence."""

from __future__ import annotations

import csv
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_ROOT = PROJECT_ROOT / "results" / "detection_history"
EVENT_COOLDOWN_SECONDS = 10.0


@dataclass(frozen=True)
class DetectionEvent:
    event_id: str
    timestamp: str
    source_type: str
    source_name: str
    predicted_class: str
    display_class: str
    category: str
    confidence: float
    top2_class: str
    top2_probability: float
    margin: float
    confidence_status: str
    uncertain: bool
    stable: bool
    stable_count: int
    threat_level: str
    inference_time_ms: float
    snapshot_path: str
    notes: str
    frame_number: str = ""
    video_time_seconds: str = ""


EVENT_FIELDS = tuple(field.name for field in fields(DetectionEvent))


def threat_level(predicted_class: str) -> str:
    """Application metadata, never a neural-network output."""
    if predicted_class == "Venomous_Snake":
        return "HIGH"
    if predicted_class == "Non_Venomous_Snake":
        return "CAUTION"
    return "MONITOR"


def new_event_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    return f"{current:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return token[:60] or "runtime"


class DetectionHistoryService:
    """Single append-oriented history service shared by every inference page."""

    def __init__(self, root: Path = DEFAULT_HISTORY_ROOT, cooldown_seconds: float = EVENT_COOLDOWN_SECONDS) -> None:
        self.root = Path(root)
        self.csv_path = self.root / "detection_history.csv"
        self.snapshot_dir = self.root / "snapshots"
        self.cooldown_seconds = float(cooldown_seconds)
        self._lock = threading.RLock()
        self._last_event_time: dict[tuple[str, str, str], float] = {}
        self.malformed_rows = 0

    def _ensure_directories(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _append(self, event: DetectionEvent) -> DetectionEvent:
        with self._lock:
            self._ensure_directories()
            needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(asdict(event))
                handle.flush()
        return event

    def save_snapshot(self, image: Image.Image, source_type: str, label: str = "frame", event_id: str | None = None) -> str:
        """Save a runtime JPEG; never writes to a dataset directory."""
        identifier = event_id or new_event_id()
        filename = f"{identifier}_{_safe_token(source_type.lower())}_{_safe_token(label)}.jpg"
        with self._lock:
            self._ensure_directories()
            destination = self.snapshot_dir / filename
            if destination.exists():
                raise FileExistsError(destination)
            image.convert("RGB").save(destination, format="JPEG", quality=92)
        return destination.relative_to(self.root).as_posix()

    def copy_image_snapshot(self, source: Path, event_id: str, label: str) -> str:
        suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".jpg"
        filename = f"{event_id}_image_{_safe_token(label)}{suffix}"
        with self._lock:
            self._ensure_directories()
            destination = self.snapshot_dir / filename
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(source, destination)
        return destination.relative_to(self.root).as_posix()

    def create_event(self, *, source_type: str, source_name: str, predicted_class: str, display_class: str, category: str, confidence: float, top2_class: str, top2_probability: float, margin: float, confidence_status: str, uncertain: bool, stable: bool, stable_count: int, inference_time_ms: float, snapshot_path: str, notes: str, frame_number: str = "", video_time_seconds: str = "", event_id: str | None = None) -> DetectionEvent:
        now = datetime.now().astimezone()
        return DetectionEvent(
            event_id=event_id or new_event_id(now), timestamp=now.isoformat(timespec="seconds"),
            source_type=source_type.upper(), source_name=source_name,
            predicted_class=predicted_class, display_class=display_class, category=category,
            confidence=float(confidence), top2_class=top2_class,
            top2_probability=float(top2_probability), margin=float(margin),
            confidence_status=confidence_status, uncertain=bool(uncertain), stable=bool(stable),
            stable_count=int(stable_count), threat_level=threat_level(predicted_class),
            inference_time_ms=float(inference_time_ms), snapshot_path=snapshot_path, notes=notes,
            frame_number=frame_number, video_time_seconds=video_time_seconds,
        )

    def log_image(self, image_path: Path, **event_values) -> DetectionEvent:
        identifier = new_event_id()
        snapshot = self.copy_image_snapshot(image_path, identifier, event_values["predicted_class"])
        event = self.create_event(source_type="IMAGE", source_name=str(image_path), snapshot_path=snapshot, event_id=identifier, stable=False, stable_count=0, notes="Explicitly saved image analysis", **event_values)
        return self._append(event)

    def log_stable_frame(self, image: Image.Image, *, source_type: str, source_name: str, now_monotonic: float | None = None, **event_values) -> DetectionEvent | None:
        confidence = float(event_values["confidence"])
        if not event_values["stable"] or event_values["uncertain"] or confidence < .80:
            return None
        key = (source_type.upper(), source_name, event_values["predicted_class"])
        clock = time.monotonic() if now_monotonic is None else now_monotonic
        with self._lock:
            previous = self._last_event_time.get(key)
            if previous is not None and clock - previous < self.cooldown_seconds:
                return None
            identifier = new_event_id()
            snapshot = self.save_snapshot(image, source_type, event_values["predicted_class"], identifier)
            event = self.create_event(source_type=source_type, source_name=source_name, snapshot_path=snapshot, event_id=identifier, notes=f"Automatic stable event; {self.cooldown_seconds:g}-second duplicate cooldown", **event_values)
            self._append(event)
            self._last_event_time[key] = clock
            return event

    def load_events(self) -> list[DetectionEvent]:
        if not self.csv_path.is_file():
            self.malformed_rows = 0
            return []
        events: list[DetectionEvent] = []
        malformed = 0
        with self._lock, self.csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    events.append(DetectionEvent(
                        **{key: row[key] for key in EVENT_FIELDS if key not in {"confidence", "top2_probability", "margin", "uncertain", "stable", "stable_count", "inference_time_ms"}},
                        confidence=float(row["confidence"]), top2_probability=float(row["top2_probability"]),
                        margin=float(row["margin"]), uncertain=row["uncertain"].lower() == "true",
                        stable=row["stable"].lower() == "true", stable_count=int(row["stable_count"]),
                        inference_time_ms=float(row["inference_time_ms"]),
                    ))
                except (KeyError, TypeError, ValueError):
                    malformed += 1
        self.malformed_rows = malformed
        return events

    def resolve_snapshot(self, relative_path: str) -> Path | None:
        if not relative_path:
            return None
        candidate = (self.root / relative_path).resolve()
        try: candidate.relative_to(self.snapshot_dir.resolve())
        except ValueError: return None
        return candidate if candidate.is_file() else None

    @staticmethod
    def export_csv(events: Iterable[DetectionEvent], destination: Path) -> None:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
            writer.writeheader(); writer.writerows(asdict(event) for event in events)
