"""Additive Live Wildlife Tracking page in the existing Tkinter dashboard."""
from __future__ import annotations

import copy
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

from PIL import Image, ImageTk

from .base import ScrollablePage
from ..services.live_hardware import load_config
from ..services.live_tracking import LiveTrackingService


class LiveTrackingPage(ScrollablePage):
    MIN_CONTENT_WIDTH = 720

    def __init__(self, parent, history):
        super().__init__(parent)
        self.service = None
        self.poll_id = None
        self.photo = None
        self.positions = {}
        self.cfg = load_config()
        self.simulation = tk.BooleanVar(value=self.cfg["simulation"])
        self.threshold = tk.StringVar(value=str(self.cfg["microphone"]["threshold"]))
        self.status = tk.StringVar(value="Stopped. Choose hardware or clearly labeled simulation, then Start.")
        self.telemetry = tk.StringVar(value="Sensors: Unknown")
        self.track_text = tk.StringVar(value="No tracks")
        self.review_status = tk.StringVar(value="")
        self.confirm_id = tk.StringVar()
        ttk.Label(self.content, text="Live Wildlife Tracking", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Camera motion regions + centroid IDs. Relative camera view; no metric positioning.",
                  style="Muted.TLabel", wraplength=900).pack(anchor="w", pady=(4, 10))
        controls = ttk.Frame(self.content)
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.pack(side="left")
        ttk.Button(controls, text="Stop", command=self.stop).pack(side="left", padx=6)
        self.sim_check = ttk.Checkbutton(controls, text="SIMULATION", variable=self.simulation)
        self.sim_check.pack(side="left", padx=8)
        ttk.Label(controls, text="RMS threshold").pack(side="left")
        self.threshold_entry = ttk.Entry(controls, textvariable=self.threshold, width=8)
        self.threshold_entry.pack(side="left", padx=6)
        self.sim_press = ttk.Button(controls, text="Simulate button")
        self.sim_press.pack(side="left")
        self.sim_press.bind("<ButtonPress-1>", lambda _e: self._button(True))
        self.sim_press.bind("<ButtonRelease-1>", lambda _e: self._button(False))
        self.sim_press.bind("<Leave>", lambda _e: self._button(False))
        ttk.Label(self.content, textvariable=self.status, style="Muted.TLabel",
                  wraplength=950).pack(anchor="w", pady=8)
        body = ttk.Frame(self.content)
        body.pack(fill="x")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        self.preview = tk.Canvas(body, width=350, height=270, bg="#081713", highlightthickness=0)
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.map = tk.Canvas(body, width=350, height=270, bg="#061512", highlightthickness=0)
        self.map.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ttk.Label(self.content, text="Green: established motion track | Yellow: tentative / uncertain / awaiting risk review | Red: operator-confirmed risk",
                  style="Muted.TLabel", wraplength=950).pack(anchor="w", pady=(8, 4))
        ttk.Label(self.content, textvariable=self.telemetry, style="Body.TLabel",
                  wraplength=950, justify="left").pack(fill="x", pady=8)
        ttk.Label(self.content, textvariable=self.track_text, style="Body.TLabel",
                  wraplength=950, justify="left").pack(fill="x", pady=8)
        review = ttk.Frame(self.content)
        review.pack(fill="x")
        ttk.Label(review, text="Reviewed dangerous track ID:").pack(side="left")
        ttk.Entry(review, textvariable=self.confirm_id, width=6).pack(side="left", padx=6)
        ttk.Button(review, text="Confirm reviewed risk", command=self.confirm).pack(side="left")
        ttk.Label(self.content, textvariable=self.review_status, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Motion is not animal detection. Shadows, people, foliage, occlusion and crossings can cause false tracks or ID switches. "
                  "Stationary objects can disappear. Model scores are uncalibrated; crop predictions are exploratory. "
                  "Radar corroborates scene activity only. Distance, bearing and association with a camera track remain Unknown. "
                  "Physical alerts use microphone/radar/button rules only; classification does not activate the buzzer.",
                  style="Warning.TLabel", wraplength=950).pack(anchor="w", pady=12)
        self._draw_map([], False)

    def _button(self, pressed):
        if self.service and self.service.cfg["simulation"]:
            self.service.hardware.sim_button = pressed

    def start(self):
        if self.service and self.service.running:
            self.status.set("Existing session is running or stopping; stop it before restarting.")
            return
        try:
            cfg = copy.deepcopy(self.cfg)
            cfg["simulation"] = self.simulation.get()
            value = float(self.threshold.get())
            if not 0 < value <= 1:
                raise ValueError("RMS threshold must be greater than 0 and at most 1")
            cfg["microphone"]["threshold"] = value
            self.service = LiveTrackingService(cfg)
            self.positions.clear()
            self.service.start()
            self.start_button.configure(state="disabled")
            self.sim_check.configure(state="disabled")
            self.threshold_entry.configure(state="disabled")
            self._poll()
        except Exception as exc:
            self.status.set(f"Cannot start: {exc}")

    def confirm(self):
        try:
            confirmed = self.service and self.service.confirm_risk(int(self.confirm_id.get()))
            self.review_status.set("Operator review recorded." if confirmed else
                            "Requires a live track with a stable configured risk-class prediction.")
        except ValueError:
            self.review_status.set("Enter a numeric track ID.")

    def _poll(self):
        self.poll_id = None
        if self.service is None:
            return
        data = self.service.snapshot()
        sensors = data["sensors"]
        mic, radar, gpio = (sensors.get(key, {}) for key in ("mic", "radar", "gpio"))
        def bit(value):
            return "Unknown" if value is None else str(int(value))
        rms = mic.get("rms")
        distance = radar.get("distance")
        distance_text = "Unknown" if distance is None else f"{distance:.3f} m (aggregate radar target only)"
        self.telemetry.set(
            f"MIC: {bit(mic.get('active'))} | RMS: {rms if rms is not None else 'Unknown'} | "
            f"mmWave presence: {bit(radar.get('active'))} | Radar distance: {distance_text}\n"
            f"LED17: {bit(gpio.get('led1'))} | LED27: {bit(gpio.get('led2'))} | "
            f"LED22: {bit(gpio.get('led3'))} | Buzzer: {bit(gpio.get('buzzer'))} | Button: {bit(gpio.get('button'))}\n"
            f"Connections: camera={data['camera_health']}, mic={'live' if mic.get('live') else 'unavailable'}, "
            f"radar={'live' if radar.get('live') else 'unavailable'}, GPIO={gpio.get('health', 'starting')}\n"
            f"Audio: {mic.get('device', 'Unknown')} {mic.get('note', '')}\n"
            f"Model: {data['model_health']}")
        errors = [str(item["error"]) for item in (mic, radar, gpio) if item.get("error")]
        if data["camera_error"]:
            errors.append(data["camera_error"])
        mode = "SIMULATION - NO PHYSICAL HARDWARE" if data["simulation"] else "LIVE HARDWARE"
        self.status.set(f"{mode} | Tracked regions: {len(data['tracks'])} | " +
                        ("; ".join(errors) if errors else "Running"))
        lines = []
        for track in data["tracks"]:
            result = track["classification"]
            label = "Human (person detector)" if track.get("is_human", False) else (
                "Animal: Unknown" if result is None else (
                f"{result['name']} {result['confidence']:.1%} model score"
                + (" | RISK REVIEW REQUIRED" if track["risk_candidate"] else "")))
            lines.append(f"ID {track['id']} | {label} | " +
                         ("operator-confirmed risk" if track["confirmed_risk"] else "unverified") +
                         " | distance/bearing: Unknown")
        self.track_text.set("\n".join(lines) or "No tracked motion regions")
        self.preview.delete("all")
        if data["frame"] is not None:
            image = Image.fromarray(data["frame"][:, :, ::-1])
            image.thumbnail((max(100, self.preview.winfo_width()), 270))
            self.photo = ImageTk.PhotoImage(image)
            self.preview.create_image(0, 0, image=self.photo, anchor="nw")
            sx, sy = image.width / data["frame"].shape[1], image.height / data["frame"].shape[0]
            for track in data["tracks"]:
                x, y, w, h = track["box"]
                self.preview.create_rectangle(x*sx, y*sy, (x+w)*sx, (y+h)*sy, outline=track["color"])
                self.preview.create_text(x*sx+3, y*sy+3, text=f"ID {track['id']}", fill=track["color"], anchor="nw")
        else:
            self.preview.create_text(12, 25, text="Camera unavailable / starting", anchor="nw", fill="#ffc857")
            self.photo = None
        self._draw_map(data["tracks"], data["simulation"], data["camera_health"] == "connected",
                       time.time() - data["frame_age"])
        self.poll_id = self.after(50, self._poll)

    def _draw_map(self, tracks, simulation, live=False, updated_at=None):
        canvas = self.map
        canvas.delete("all")
        width, height = max(300, canvas.winfo_width()), 270
        left, right, top, bottom = 24, width-24, 40, height-35
        cx = (left + right)/2
        for fraction in (0.25, 0.5, 0.75, 1):
            r = (bottom-top)*fraction
            canvas.create_oval(cx-r, bottom-r, cx+r, bottom+r, outline="#21453a")
        canvas.create_rectangle(left, top, right, bottom, outline="#37735a")
        canvas.create_line(cx, top, cx, bottom, fill="#21453a")
        canvas.create_line(left, (top+bottom)/2, right, (top+bottom)/2, fill="#21453a")
        canvas.create_polygon(cx-6, bottom+5, cx+6, bottom+5, cx, bottom-6, fill="#cce9db")
        canvas.create_text(width/2, 14, text="RELATIVE CAMERA VIEW" + (" - SIMULATED" if simulation else ""),
                           fill="#d7eee3", font=("TkDefaultFont", 10, "bold"))
        canvas.create_text(width/2, height-12, text="X: left 0-100% right | Y: top 0-100% bottom",
                           fill="#9cb9ac", font=("TkDefaultFont", 8))
        active = {t["id"] for t in tracks}
        self.positions = {key: p for key, p in self.positions.items() if key in active}
        for track in tracks:
            tx, ty = track["center"]
            old = self.positions.get(track["id"], (tx, ty))
            point = (old[0]+0.35*(tx-old[0]), old[1]+0.35*(ty-old[1]))
            self.positions[track["id"]] = point
            trail = track["trail"]
            if len(trail) > 1:
                coordinates = [v for x, y in trail for v in (left+x*(right-left), top+y*(bottom-top))]
                canvas.create_line(*coordinates, fill="#366f57")
            x, y = left+point[0]*(right-left), top+point[1]*(bottom-top)
            canvas.create_oval(x-4, y-4, x+4, y+4, fill=track["color"], outline="")
            canvas.create_text(x+7, y-5, text=str(track["id"]), fill=track["color"], anchor="w")
        stamp = (datetime.fromtimestamp(updated_at or time.time()).astimezone().strftime("%H:%M:%S %Z")
                 if live else "No live update")
        canvas.create_text(left+4, top+4, text=f"Update: {stamp}\nRings: visual guide, not metres",
                           anchor="nw", fill="#9cb9ac", font=("TkDefaultFont", 8))

    def stop(self):
        if self.poll_id is not None:
            self.after_cancel(self.poll_id)
            self.poll_id = None
        if self.service is not None:
            try:
                self.service.stop()
            except Exception as exc:
                self.status.set(f"Shutdown: {exc}")
                raise
        self.positions.clear()
        self.preview.delete("all")
        self.photo = None
        self._draw_map([], self.simulation.get())
        self.telemetry.set("Stopped. Outputs OFF; sensor readings unavailable.")
        self.track_text.set("No live tracks")
        self.status.set("Stopped; outputs off and resources released.")
        self.start_button.configure(state="normal")
        self.sim_check.configure(state="normal")
        self.threshold_entry.configure(state="normal")

    def on_hide(self):
        self.stop()

    def close(self):
        self.stop()
