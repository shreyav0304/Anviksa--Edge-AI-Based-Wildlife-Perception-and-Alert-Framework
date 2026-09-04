"""Interactive visualization of existing validated model-comparison data."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ..components import SectionCard, StatusPill
from ..services.comparison_data import EXPECTED_FINGERPRINT, ModelComparisonRecord
from .base import ScrollablePage


class ModelComparisonPage(ScrollablePage):
    GRAPH_KEYS = (
        ("core", "Core Metrics"),
        ("mse", "MSE"),
        ("confidence", "Confidence"),
        ("snake", "Snake Performance"),
    )

    def __init__(self, parent: tk.Misc, records: tuple[ModelComparisonRecord, ...], navigate: callable) -> None:
        super().__init__(parent)
        self.records = records
        self.completed = tuple(record for record in records if record.status == "COMPLETED")
        self.navigate = navigate
        self._graph_buttons: dict[str, ttk.Button] = {}
        self.current_graph = ""
        self._build()

    @staticmethod
    def _percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    @staticmethod
    def _number(value: float | None, digits: int = 6) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    def _build(self) -> None:
        ttk.Label(self.content, text="VALIDATED EXPERIMENT RESULTS", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Model Comparison", style="PageTitle.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(self.content, text="Completed models evaluated on the same frozen 545-image test set.", style="Muted.TLabel").pack(anchor="w")

        status_row = ttk.Frame(self.content, style="Page.TFrame"); status_row.pack(fill="x", pady=(14, 10))
        for record in self.records:
            card = ttk.Frame(status_row, style="Metric.TFrame", padding=(13, 10)); card.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Label(card, text=record.model, style="BodyStrong.TLabel").pack(anchor="w")
            kind = "selected" if record.model == "MobileNetV3 Large" else "incomplete" if record.status == "INCOMPLETE" else "complete"
            text = "SELECTED" if kind == "selected" else record.status
            StatusPill(card, text, kind).pack(anchor="w", pady=(6, 0))

        winner = SectionCard(self.content, "Best validated classification model"); winner.pack(fill="x")
        left = ttk.Frame(winner, style="Card.TFrame"); left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="MobileNetV3 Large", style="Winner.TLabel").pack(anchor="w")
        ttk.Label(left, text="Highest test accuracy and Macro F1 • Lowest probability MSE • Highest venomous-snake and snake macro recall", style="Body.TLabel", wraplength=700).pack(anchor="w", pady=(5, 0))
        right = ttk.Frame(winner, style="Card.TFrame"); right.pack(side="right", padx=(18, 0))
        selected = next(record for record in self.records if record.model == "MobileNetV3 Large")
        ttk.Label(right, text=f"Accuracy  {self._percent(selected.accuracy)}     Macro F1  {self._percent(selected.macro_f1)}", style="ResultValue.TLabel").pack(anchor="e")
        ttk.Label(right, text=f"Venomous recall  {self._percent(selected.venomous_recall)}     Snake macro recall  {self._percent(selected.snake_macro_recall)}", style="Small.TLabel").pack(anchor="e", pady=(5, 0))

        graph_card = SectionCard(self.content, "Interactive comparison"); graph_card.pack(fill="x", pady=(12, 0))
        controls = ttk.Frame(graph_card, style="Card.TFrame"); controls.pack(fill="x", pady=(0, 8))
        for key, label in self.GRAPH_KEYS:
            button = ttk.Button(controls, text=label, command=lambda graph=key: self.show_graph(graph), style="Graph.TButton")
            button.pack(side="left", padx=(0, 6)); self._graph_buttons[key] = button
        ttk.Label(controls, text="EfficientNetB0 — INCOMPLETE / N/A", style="Incomplete.Status.TLabel").pack(side="right")
        self.figure = Figure(figsize=(10, 4.2), dpi=100, facecolor="#ffffff")
        self.axis = self.figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(self.figure, master=graph_card)
        self.chart.get_tk_widget().configure(height=410, highlightthickness=0)
        self.chart.get_tk_widget().pack(fill="x", expand=True)
        self.graph_note = ttk.Label(graph_card, text="", style="Notice.TLabel", wraplength=1000)
        self.graph_note.pack(anchor="w", pady=(8, 0))
        self.show_graph("core")

        self._table().pack(fill="x", pady=(12, 0))
        self._model_information().pack(fill="x", pady=(12, 0))
        self._edge_and_methodology().pack(fill="x", pady=(12, 0))

    def show_graph(self, key: str) -> None:
        if key not in dict(self.GRAPH_KEYS):
            raise KeyError(key)
        self.current_graph = key
        for name, button in self._graph_buttons.items(): button.configure(style="Active.Graph.TButton" if name == key else "Graph.TButton")
        self.axis.clear()
        renderers = {"core": self._core_graph, "mse": self._mse_graph, "confidence": self._confidence_graph, "snake": self._snake_graph}
        renderers[key]()
        self.axis.spines[["top", "right"]].set_visible(False)
        self.axis.grid(axis="y", alpha=.18, zorder=0)
        self.figure.tight_layout(pad=1.5)
        self.chart.draw_idle()

    def _grouped(self, fields: tuple[str, ...], labels: tuple[str, ...], percentage: bool, title: str) -> None:
        names = [record.model for record in self.completed]
        x = np.arange(len(names)); width = .72 / len(fields); colors = ("#287454", "#3f75a2", "#d68a36")
        for index, (field, label) in enumerate(zip(fields, labels, strict=True)):
            values = [getattr(record, field) * (100 if percentage else 1) for record in self.completed]
            bars = self.axis.bar(x + (index - (len(fields)-1)/2) * width, values, width, label=label, color=colors[index % len(colors)], zorder=3)
            self.axis.bar_label(bars, fmt="%.2f%%" if percentage else "%.6f", padding=3, fontsize=8)
        self.axis.set_xticks(x, names)
        self.axis.set_title(title, fontweight="bold", color="#173a2e")
        if percentage: self.axis.set_ylim(0, 108); self.axis.set_ylabel("Score (%)")
        self.axis.legend(ncols=min(len(fields), 4), loc="upper center", fontsize=8)

    def _core_graph(self) -> None:
        self._grouped(("accuracy", "macro_precision", "macro_recall", "macro_f1"), ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"), True, "Core Classification Metrics")
        self.graph_note.configure(text="Primary comparison uses ordinary test accuracy and macro-averaged precision, recall, and F1.")

    def _mse_graph(self) -> None:
        self._grouped(("mse",), ("Probability MSE — lower is better",), False, "Classification Probability MSE")
        self.axis.set_ylabel("Mean Squared Error")
        self.graph_note.configure(text="MSE measures the average squared difference between the one-hot ground-truth vector and the predicted seven-class probability vector: 1/(N × C) over N=545 images and C=7 classes.")

    def _confidence_graph(self) -> None:
        self._grouped(("mean_confidence", "median_confidence"), ("Mean Top-1 Confidence", "Median Top-1 Confidence"), True, "Prediction Confidence")
        self.graph_note.configure(text="Confidence measures the model's Top-1 predicted probability and is not the same as classification accuracy. MobileNetV3 Large's stored median is 0.9999572038650513, displayed as 100.00%.")

    def _snake_graph(self) -> None:
        self._grouped(("venomous_recall", "snake_macro_recall", "snake_macro_f1"), ("Venomous Snake Recall", "Snake Macro Recall", "Snake Macro F1"), True, "Snake-Class Performance")
        self.graph_note.configure(text="Snake Macro metrics average the two trained snake classes: Non-Venomous Snake and Venomous Snake.")

    def _table(self) -> ttk.Frame:
        card = SectionCard(self.content, "Detailed comparison table")
        columns = ("Model", "Status", "Accuracy", "Precision", "Recall", "F1", "MSE", "Mean Conf.", "Median Conf.", "Venomous Recall", "Snake Recall")
        tree = ttk.Treeview(card, columns=columns, show="headings", height=4)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=120 if column not in {"Model", "Status"} else 145, anchor="center")
        for record in self.records:
            tree.insert("", "end", values=(record.model, record.status, self._percent(record.accuracy), self._percent(record.macro_precision), self._percent(record.macro_recall), self._percent(record.macro_f1), self._number(record.mse), self._percent(record.mean_confidence), self._percent(record.median_confidence), self._percent(record.venomous_recall), self._percent(record.snake_macro_recall)))
        scroll = ttk.Scrollbar(card, orient="horizontal", command=tree.xview); tree.configure(xscrollcommand=scroll.set)
        tree.pack(fill="x"); scroll.pack(fill="x")
        return card

    def _model_information(self) -> ttk.Frame:
        card = SectionCard(self.content, "Model information")
        info = (
            ("Custom CNN", "Convolutional Neural Network trained from scratch"),
            ("MobileNetV2", "Lightweight CNN with depthwise separable convolution and inverted residual blocks • ImageNet transfer learning + fine-tuning"),
            ("MobileNetV3 Large", "Optimized lightweight CNN using inverted residuals, depthwise convolution, squeeze-and-excitation and hard-swish • ImageNet transfer learning + fine-tuning"),
            ("EfficientNetB0", "Efficient CNN using MBConv, depthwise separable convolution, squeeze-and-excitation and compound scaling • INCOMPLETE due to development-system resource limitations"),
        )
        for model, description in info:
            line = ttk.Frame(card, style="Card.TFrame"); line.pack(fill="x", pady=4)
            ttk.Label(line, text=model, style="BodyStrong.TLabel", width=22).pack(side="left", anchor="n")
            ttk.Label(line, text=description, style="Body.TLabel", wraplength=850).pack(side="left", fill="x", expand=True)
        return card

    def _edge_and_methodology(self) -> ttk.Frame:
        card = SectionCard(self.content, "Edge AI and methodology")
        ttk.Label(card, text="WINDOWS CPU / KERAS BENCHMARK", style="MetricLabel.TLabel").pack(anchor="w")
        benchmark = "   •   ".join(
            f"{record.model}: {record.model_size_mib:.2f} MiB / {record.inference_time_ms:.2f} ms"
            for record in self.completed
            if record.model_size_mib is not None and record.inference_time_ms is not None
        )
        ttk.Label(card, text=benchmark, style="Body.TLabel", wraplength=1050).pack(anchor="w", pady=(4, 10))
        ttk.Label(card, text="EDGE DEPLOYMENT: MobileNetV3 Large has validated Float32 and Float16 TFLite representations.   RASPBERRY PI PERFORMANCE: Pending hardware benchmarking.", style="BodyStrong.TLabel", wraplength=1050).pack(anchor="w")
        ttk.Label(card, text=f"Completed models used the same frozen 545-image test set, seven-class order, and dataset fingerprint: {EXPECTED_FINGERPRINT}", style="Notice.TLabel", wraplength=1050).pack(anchor="w", pady=(10, 8))
        ttk.Button(card, text="View Confusion Matrices (Phase 4)", command=lambda: self.navigate("confusion")).pack(anchor="w")
        return card
