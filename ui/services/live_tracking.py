"""Camera motion proposals + bounded centroid tracking; never metric positioning."""
from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque

import numpy as np

from .live_hardware import HardwareService, load_config
from .opencv import require_opencv


class CentroidTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes, shape, now):
        height, width = shape[:2]
        points = [((x + w / 2) / width, (y + h / 2) / height) for x, y, w, h in boxes]
        self.tracks = {key: t for key, t in self.tracks.items()
                       if now - t["seen"] <= self.cfg["expiry_seconds"]}
        pairs = []
        for key, track in self.tracks.items():
            for index, point in enumerate(points):
                distance = math.dist(track["center"], point)
                if distance <= self.cfg["max_match_distance"]:
                    pairs.append((distance, key, index))
        used_ids, used_boxes = set(), set()
        for _, key, index in sorted(pairs):
            if key in used_ids or index in used_boxes:
                continue
            track = self.tracks[key]
            track.update(center=points[index], box=boxes[index], seen=now, hits=track["hits"] + 1)
            track["trail"].append(points[index])
            used_ids.add(key)
            used_boxes.add(index)
        for index, point in enumerate(points):
            if index in used_boxes or len(self.tracks) >= self.cfg["max_objects"]:
                continue
            self.tracks[self.next_id] = {"id": self.next_id, "center": point,
                "box": boxes[index], "seen": now, "hits": 1,
                "trail": deque([point], maxlen=self.cfg["trail_length"])}
            self.next_id += 1
        return [dict(t, trail=list(t["trail"]), observed=t["seen"] == now)
                for t in self.tracks.values()]


class LiveTrackingService:
    def __init__(self, cfg=None):
        self.cfg = copy.deepcopy(cfg or load_config())
        self.hardware = HardwareService(self.cfg)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_tracks = []
        self.frame_time = 0.0
        self.camera_health = "stopped"
        self.camera_error = ""
        self.model_health = "disabled" if not self.cfg["classification"]["enabled"] else "starting"
        self.results = {}
        self.confirmed = set()
        self.threads = []
        self.capture = None

    @property
    def running(self):
        return any(t.is_alive() for t in self.threads) or any(t.is_alive() for t in self.hardware.threads)

    def start(self):
        if self.threads:
            raise RuntimeError("Create a new session after stopping")
        self.hardware.start()
        targets = [self._camera]
        if self.cfg["classification"]["enabled"] and not self.cfg["simulation"]:
            targets.append(self._classifier)
        if self.cfg["simulation"]:
            self.model_health = "SIMULATED camera; classification disabled"
        for target in targets:
            t = threading.Thread(target=target, daemon=True, name=target.__name__)
            self.threads.append(t)
            t.start()

    def confirm_risk(self, track_id):
        """Explicit operator review, not automatic volume/presence confirmation."""
        data = self.snapshot()
        for track in data["tracks"]:
            if track["id"] == track_id and track.get("risk_candidate"):
                with self.lock:
                    self.confirmed.add(track_id)
                return True
        return False

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            tracks = copy.deepcopy(self.latest_tracks)
            results = copy.deepcopy(self.results)
            confirmed = set(self.confirmed)
            age = now - self.frame_time
            health, error, model = self.camera_health, self.camera_error, self.model_health
        if age > self.cfg["camera"]["stale_seconds"] or health != "connected":
            frame, tracks = None, []
            if health == "connected":
                health = "stale"
        for track in tracks:
            result = results.get(track["id"])
            if result and now - result["updated"] < self.cfg["classification"]["stale_seconds"]:
                track["classification"] = result
                track["risk_candidate"] = (result["stable"]
                    and result["name"] in self.cfg["classification"]["risk_classes"])
            else:
                track["classification"] = None
                track["risk_candidate"] = False
            track["confirmed_risk"] = (track["id"] in confirmed and track["risk_candidate"]
                                       and track["observed"])
            track["color"] = ("#ff5252" if track["confirmed_risk"] else
                "#ffc857" if track["hits"] < 3 or not track["observed"] or
                (track["classification"] and (not track["classification"]["stable"] or track["risk_candidate"]))
                else "#54e69d")
        return {"frame": frame, "tracks": tracks, "camera_health": health,
                "camera_error": error, "frame_age": age, "model_health": model,
                "sensors": self.hardware.snapshot(), "simulation": self.cfg["simulation"]}

    def _publish(self, frame, tracks):
        with self.lock:
            self.latest_frame = frame
            self.latest_tracks = tracks
            self.frame_time = time.monotonic()
            self.camera_health, self.camera_error = "connected", ""
            active = {t["id"] for t in tracks}
            self.results = {key: value for key, value in self.results.items() if key in active}
            self.confirmed.intersection_update(active)

    def _camera(self):
        try:
            cv2 = require_opencv()
        except Exception as exc:
            with self.lock:
                self.camera_health, self.camera_error = "unavailable", str(exc)
            return
        cfg = self.cfg
        tracker = CentroidTracker(cfg["tracking"])
        start = time.monotonic()
        while not self.stop_event.is_set():
            cap = None
            try:
                foreground = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24,
                                                              detectShadows=True)
                warmup = 0
                if not cfg["simulation"]:
                    source = cfg["camera"]["source"]
                    cap = cv2.VideoCapture(source)
                    self.capture = cap
                    if not cap.isOpened():
                        raise RuntimeError("USB camera cannot open; check source/permissions/other pages")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
                    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps"])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while not self.stop_event.is_set():
                    tick = time.monotonic()
                    if cfg["simulation"]:
                        frame = np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), dtype=np.uint8)
                        h, w = frame.shape[:2]
                        boxes = []
                        for phase, y in ((0, 0.3), (2.2, 0.7)):
                            x = int((0.45 + 0.3 * math.sin((tick - start) * 0.3 + phase)) * w)
                            box = (x, int(y * h), max(10, w // 12), max(10, h // 10))
                            boxes.append(box)
                            bx, by, bw, bh = box
                            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 200, 110), -1)
                        cv2.putText(frame, "SIMULATION - NOT LIVE HARDWARE", (12, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
                    else:
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError("USB camera disconnected/frame read failed")
                        # Bound processing cost even if the device ignores requested resolution.
                        frame = cv2.resize(frame, (cfg["camera"]["width"], cfg["camera"]["height"]))
                        mask = foreground.apply(frame)
                        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                        mask = cv2.dilate(mask, None, iterations=2)
                        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                        area = frame.shape[0] * frame.shape[1]
                        boxes = [cv2.boundingRect(c) for c in contours
                                 if cfg["tracking"]["min_area_fraction"] * area <= cv2.contourArea(c) < area * 0.75]
                        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
                        warmup += 1
                        if warmup <= cfg["tracking"]["warmup_frames"]:
                            boxes = []
                    tracks = tracker.update(boxes[:cfg["tracking"]["max_objects"]], frame.shape, tick)
                    self._publish(frame, tracks)
                    self.stop_event.wait(max(0, 1 / cfg["camera"]["fps"] - (time.monotonic() - tick)))
            except Exception as exc:
                with self.lock:
                    self.camera_health, self.camera_error = "disconnected", str(exc)
                    self.latest_frame, self.latest_tracks = None, []
                    self.results.clear()
                    self.confirmed.clear()
                tracker.tracks.clear()
            finally:
                if cap is not None:
                    cap.release()
                self.capture = None
            self.stop_event.wait(2)

    def _classifier(self):
        try:
            from pi_deployment.inference.tflite_inference import TFLiteWildlifeClassifier
            from .image_inference import ImageInferenceService, MODEL_PATH
            ImageInferenceService().verify_model()
            classifier = TFLiteWildlifeClassifier(MODEL_PATH, num_threads=1)
            if not set(self.cfg["classification"]["risk_classes"]).issubset(classifier.class_names):
                raise ValueError("Risk classes do not match deployed class names")
            with self.lock:
                self.model_health = "connected; motion-crop predictions are exploratory"
        except Exception as exc:
            with self.lock:
                self.model_health = f"unavailable: {exc}"
            return
        cv2 = require_opencv()
        while not self.stop_event.is_set():
            data = self.snapshot()
            tracks = [t for t in data["tracks"] if t["observed"]]
            if data["frame"] is not None and tracks:
                # One ROI per interval; oldest result first. No growing inference queue.
                track = min(tracks, key=lambda t: (t.get("classification") or {}).get("updated", 0))
                x, y, w, h = track["box"]
                crop = data["frame"][y:y + h, x:x + w]
                try:
                    prediction = classifier.predict(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    now = time.monotonic()
                    with self.lock:
                        old = self.results.get(track["id"], {})
                        name, score = prediction["predicted_class"], prediction["confidence"]
                        hits = old.get("hits", 0) + 1 if (old.get("name") == name and
                            now - old.get("updated", 0) < self.cfg["classification"]["stale_seconds"]) else 1
                        if score < self.cfg["classification"]["score_threshold"]:
                            hits = 0
                        if (old.get("name") != name or
                                hits < self.cfg["classification"]["stable_results"]):
                            self.confirmed.discard(track["id"])
                        if track["id"] in {t["id"] for t in self.latest_tracks}:
                            self.results[track["id"]] = {"name": name, "confidence": score,
                                "hits": hits, "stable": hits >= self.cfg["classification"]["stable_results"],
                                "updated": now}
                except Exception as exc:
                    with self.lock:
                        self.model_health = f"inference error: {exc}"
            self.stop_event.wait(self.cfg["classification"]["interval_seconds"])

    def stop(self):
        self.stop_event.set()
        # GPIO/sensors stop first even when a camera driver/inference is slow.
        errors = []
        try:
            self.hardware.stop()
        except Exception as exc:
            errors.append(str(exc))
        for thread in self.threads:
            thread.join(timeout=5)
        if any(t.is_alive() for t in self.threads):
            errors.append("Camera/model still stopping. Do not reopen camera until worker exits.")
        if errors:
            raise RuntimeError("; ".join(errors))
