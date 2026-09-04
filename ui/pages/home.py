"""Artifact-backed Phase 1 home dashboard."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..components import MetricCard, SectionCard, StatusPill
from ..data import DashboardData
from .base import ScrollablePage


class HomePage(ScrollablePage):
    def __init__(self, parent: tk.Misc, data: DashboardData) -> None:
        super().__init__(parent)
        self.data = data
        self._build()

    @staticmethod
    def _pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def _build(self) -> None:
        ttk.Label(self.content, text="VERIFIED PROJECT OVERVIEW", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Home Dashboard", style="PageTitle.TLabel").pack(anchor="w", pady=(4, 5))
        ttk.Label(self.content, text="Current validated classifier, deployment readiness, and research status.", style="Muted.TLabel").pack(anchor="w", pady=(0, 20))

        metrics = ttk.Frame(self.content, style="Page.TFrame")
        metrics.pack(fill="x")
        cards = [
            ("System Status", "Offline Edge AI", "Offline operation supported"),
            ("Selected Model", self.data.selected_model, "Current best completed classifier"),
            ("Deployment Format", self.data.deployment_format, "Validated Float32 and Float16"),
            ("Number of Classes", str(len(self.data.classes)), "Whole-frame classification"),
            ("Best Test Accuracy", self._pct(self.data.accuracy), "Controlled frozen test set"),
            ("Macro F1", self._pct(self.data.macro_f1), "Seven-class macro average"),
            ("Venomous Snake Recall", self._pct(self.data.venomous_recall), "Validated test recall"),
            ("Snake Macro Recall", self._pct(self.data.snake_macro_recall), "Venomous + non-venomous"),
        ]
        for column in range(4):
            metrics.columnconfigure(column, weight=1, uniform="metric")
        for index, card in enumerate(cards):
            MetricCard(metrics, *card).grid(row=index // 4, column=index % 4, sticky="nsew", padx=5, pady=5)

        row = ttk.Frame(self.content, style="Page.TFrame")
        row.pack(fill="x", pady=(18, 0))
        row.columnconfigure(0, weight=1, uniform="panels"); row.columnconfigure(1, weight=1, uniform="panels")
        self._model_status(row).grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._classes(row).grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self._pipeline().pack(fill="x", pady=(12, 0))
        self._development_status().pack(fill="x", pady=(12, 0))
        self._edge_status().pack(fill="x", pady=(12, 0))
        self._limitations().pack(fill="x", pady=(12, 0))

    def _model_status(self, parent: tk.Misc) -> ttk.Frame:
        card = SectionCard(parent, "Model status")
        statuses = [("Custom CNN", "COMPLETE", "complete"), ("MobileNetV2", "COMPLETE", "complete"), ("MobileNetV3 Large", "SELECTED", "selected"), ("EfficientNetB0", "INCOMPLETE — resource-limited experiment", "incomplete")]
        for name, status, kind in statuses:
            line = ttk.Frame(card, style="Card.TFrame"); line.pack(fill="x", pady=5)
            ttk.Label(line, text=name, style="BodyStrong.TLabel").pack(side="left")
            StatusPill(line, status, kind).pack(side="right")
        return card

    def _classes(self, parent: tk.Misc) -> ttk.Frame:
        card = SectionCard(parent, "Current classification classes")
        for index, name in enumerate(self.data.classes, 1):
            label = name.replace("_", "-")
            ttk.Label(card, text=f"{index:02d}   {label}", style="Body.TLabel").pack(anchor="w", pady=3)
        return card

    def _pipeline(self) -> ttk.Frame:
        card = SectionCard(self.content, "Project pipeline")
        steps = ["Camera / Image / Video", "OpenCV Preprocessing", "MobileNetV3 Large", "Wildlife Classification", "Confidence / Analysis", "Alert Logic", "Logging"]
        flow = ttk.Frame(card, style="Card.TFrame"); flow.pack(fill="x")
        for index, step in enumerate(steps):
            ttk.Label(flow, text=step, style="Pipeline.TLabel").pack(side="left", expand=True, padx=2)
            if index < len(steps) - 1:
                ttk.Label(flow, text="→", style="Arrow.TLabel").pack(side="left")
        ttk.Label(card, text="Current capability: whole-frame classification only — no bounding-box localization.", style="Notice.TLabel").pack(anchor="w", pady=(14, 0))
        return card

    def _development_status(self) -> ttk.Frame:
        card = SectionCard(self.content, "Development status")
        columns = [
            ("COMPLETED", "complete", ["Clean image dataset preparation", "Multi-model comparison", "MobileNetV3 selection", "TFLite Float32 validation", "TFLite Float16 validation", "Windows inference validation", "Metrics and confusion matrices"]),
            ("IN PROGRESS / PENDING", "incomplete", ["Additional camouflaged snake images", "Snake video dataset preparation", "Video-derived training dataset", "Additional wildlife video data", "Raspberry Pi benchmarking", "Camera hardware integration", "mmWave integration", "LED / buzzer integration"]),
            ("FUTURE RESEARCH", "planned", ["Wildlife audio classification", "Multimodal wildlife perception", "Vision + audio + sensor fusion"]),
        ]
        body = ttk.Frame(card, style="Card.TFrame"); body.pack(fill="x")
        for index, (title, kind, items) in enumerate(columns):
            body.columnconfigure(index, weight=1, uniform="status")
            frame = ttk.Frame(body, style="Card.TFrame"); frame.grid(row=0, column=index, sticky="nsew", padx=8)
            StatusPill(frame, title, kind).pack(anchor="w", pady=(0, 8))
            for item in items:
                ttk.Label(frame, text=f"•  {item}", style="Small.TLabel", wraplength=270).pack(anchor="w", pady=2)
        return card

    def _edge_status(self) -> ttk.Frame:
        card = SectionCard(self.content, "Edge AI / offline status")
        items = [("OFFLINE OPERATION", "SUPPORTED"), ("CURRENT DEPLOYMENT CANDIDATE", self.data.deployment_candidates), ("RASPBERRY PI BENCHMARK", "PENDING HARDWARE")]
        for label, value in items:
            line = ttk.Frame(card, style="Card.TFrame"); line.pack(fill="x", pady=4)
            ttk.Label(line, text=label, style="MetricLabel.TLabel").pack(side="left")
            ttk.Label(line, text=value, style="BodyStrong.TLabel").pack(side="right")
        return card

    def _limitations(self) -> ttk.Frame:
        card = SectionCard(self.content, "Safety and model limitations")
        warning = "AI predictions are intended for wildlife monitoring and project research. Snake classification should not be used as the sole basis for approaching or handling an unidentified snake."
        ttk.Label(card, text=warning, style="Warning.TLabel", wraplength=1050).pack(fill="x")
        details = "Seven classes • No Background/Unknown class • No bounding boxes • Real-world field performance requires additional validation"
        ttk.Label(card, text=details, style="Muted.TLabel", wraplength=1050).pack(anchor="w", pady=(10, 0))
        return card
