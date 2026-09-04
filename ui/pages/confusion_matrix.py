"""Interactive viewer for existing validated confusion matrices."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ..components import SectionCard
from ..services.confusion_data import CLASS_NAMES, ConfusionMatrixRecord
from ..services.image_inference import friendly_name
from .base import ScrollablePage


class ConfusionMatrixPage(ScrollablePage):
    def __init__(self, parent: tk.Misc, records: dict[str, ConfusionMatrixRecord], navigate: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.records = records
        self.navigate = navigate
        self.model_var = tk.StringVar(value="MobileNetV3 Large")
        self.view_var = tk.StringVar(value="raw")
        self.metric_vars: dict[str, tk.StringVar] = {}
        self.summary_vars: dict[str, tk.StringVar] = {}
        self._build()

    @staticmethod
    def _pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def _build(self) -> None:
        ttk.Label(self.content, text="VALIDATED TEST-SET ANALYSIS", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Confusion Matrix Analysis", style="PageTitle.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(self.content, text="Rows are actual classes; columns are predicted classes.", style="Muted.TLabel").pack(anchor="w")

        controls = ttk.Frame(self.content, style="Card.TFrame", padding=14); controls.pack(fill="x", pady=(14, 10))
        ttk.Label(controls, text="MODEL", style="MetricLabel.TLabel").pack(side="left")
        selector = ttk.Combobox(controls, textvariable=self.model_var, values=tuple(self.records), state="readonly", width=24)
        selector.pack(side="left", padx=(8, 22)); selector.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        ttk.Label(controls, text="VIEW", style="MetricLabel.TLabel").pack(side="left")
        ttk.Radiobutton(controls, text="Raw Counts", variable=self.view_var, value="raw", command=self.refresh).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(controls, text="Normalized %", variable=self.view_var, value="normalized", command=self.refresh).pack(side="left", padx=4)
        ttk.Label(controls, text="EfficientNetB0: Confusion Matrix — N/A (controlled experiment incomplete)", style="Incomplete.Status.TLabel").pack(side="right")

        matrix_card = SectionCard(self.content, "Confusion matrix"); matrix_card.pack(fill="x")
        self.figure = Figure(figsize=(9.6, 6.2), dpi=100, facecolor="#ffffff")
        self.axis = self.figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(self.figure, master=matrix_card)
        self.chart.get_tk_widget().configure(height=570, highlightthickness=0)
        self.chart.get_tk_widget().pack(fill="x", expand=True)

        panels = ttk.Frame(self.content, style="Page.TFrame"); panels.pack(fill="x", pady=(12, 0))
        panels.columnconfigure(0, weight=1, uniform="details"); panels.columnconfigure(1, weight=1, uniform="details")
        self._metrics_panel(panels).grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._interpretation_panel(panels).grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        note = SectionCard(self.content, "What the matrix means"); note.pack(fill="x", pady=(12, 0))
        ttk.Label(note, text="Rows represent actual classes and columns represent predicted classes. Values on the diagonal are correct classifications, while off-diagonal values represent misclassifications. In normalized view, each row shows the percentage distribution of predictions for one actual class.", style="Body.TLabel", wraplength=1050).pack(anchor="w")
        ttk.Button(note, text="Back to Model Comparison", command=lambda: self.navigate("comparison")).pack(anchor="w", pady=(10, 0))
        self.refresh()

    def _metrics_panel(self, parent: tk.Misc) -> ttk.Frame:
        card = SectionCard(parent, "Validated model metrics")
        for key, label in (("accuracy", "Accuracy"), ("macro_precision", "Macro Precision"), ("macro_recall", "Macro Recall"), ("macro_f1", "Macro F1"), ("venomous_recall", "Venomous Snake Recall"), ("snake_macro_recall", "Snake Macro Recall")):
            value = tk.StringVar(value="—"); self.metric_vars[key] = value
            row = ttk.Frame(card, style="Card.TFrame"); row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, style="Body.TLabel").pack(side="left")
            ttk.Label(row, textvariable=value, style="BodyStrong.TLabel").pack(side="right")
        return card

    def _interpretation_panel(self, parent: tk.Misc) -> ttk.Frame:
        card = SectionCard(parent, "Matrix summary and interpretation")
        for key, label in (("total", "Total predictions"), ("correct", "Correct predictions"), ("incorrect", "Incorrect predictions"), ("matrix_accuracy", "Accuracy from matrix"), ("largest", "Largest confusion"), ("nv_to_v", "Non-Venomous → Venomous"), ("v_to_nv", "Venomous → Non-Venomous")):
            value = tk.StringVar(value="—"); self.summary_vars[key] = value
            row = ttk.Frame(card, style="Card.TFrame"); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, style="Small.TLabel").pack(side="left")
            ttk.Label(row, textvariable=value, style="BodyStrong.TLabel", wraplength=330).pack(side="right")
        self.interpretation_var = tk.StringVar(value="")
        ttk.Separator(card).pack(fill="x", pady=8)
        ttk.Label(card, textvariable=self.interpretation_var, style="Notice.TLabel", wraplength=500).pack(anchor="w")
        return card

    def refresh(self) -> None:
        record = self.records[self.model_var.get()]
        normalized = self.view_var.get() == "normalized"
        matrix = record.normalized if normalized else record.raw
        self._render(record.model, matrix, normalized)
        for key, variable in self.metric_vars.items(): variable.set(self._pct(record.metrics[key]))
        source, target, count = record.largest_confusion
        values = {
            "total": str(record.total), "correct": str(record.correct), "incorrect": str(record.incorrect),
            "matrix_accuracy": self._pct(record.matrix_accuracy),
            "largest": f"{friendly_name(source)} → {friendly_name(target)} ({count})",
            "nv_to_v": str(record.nonvenomous_to_venomous),
            "v_to_nv": str(record.venomous_to_nonvenomous),
        }
        for key, value in values.items(): self.summary_vars[key].set(value)
        self.interpretation_var.set(record.interpretation)

    def _render(self, model: str, matrix: np.ndarray, normalized: bool) -> None:
        self.figure.clear()
        self.axis = self.figure.add_subplot(111)
        image = self.axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        self.figure.colorbar(image, ax=self.axis, fraction=.046, pad=.04)
        labels = [friendly_name(name).replace(" ", "\n") if "Snake" in friendly_name(name) else friendly_name(name) for name in CLASS_NAMES]
        self.axis.set(xticks=np.arange(7), yticks=np.arange(7), xticklabels=labels, yticklabels=labels, xlabel="Predicted Class", ylabel="Actual Class", title=f"{model} — {'Row-Normalized (%)' if normalized else 'Raw Counts'}")
        self.axis.tick_params(axis="x", rotation=35, labelsize=8); self.axis.tick_params(axis="y", labelsize=8)
        for label in self.axis.get_xticklabels(): label.set_ha("right")
        threshold = float(matrix.max()) / 2
        for row in range(7):
            for column in range(7):
                text = f"{matrix[row, column] * 100:.1f}%" if normalized else str(int(matrix[row, column]))
                self.axis.text(column, row, text, ha="center", va="center", fontsize=8, color="white" if matrix[row, column] > threshold else "#20332b")
        self.figure.tight_layout(pad=1.4)
        self.chart.draw_idle()
