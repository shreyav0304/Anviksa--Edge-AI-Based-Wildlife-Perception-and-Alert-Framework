"""Interactive visualization of existing validated model-comparison data."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ..services.evaluation import load_active_evaluation, format_model_score
from ..components import SectionCard, StatusPill
from ..services.comparison_data import EXPECTED_FINGERPRINT, ModelComparisonRecord
from .base import ScrollablePage


class ModelComparisonPage(ScrollablePage):
    MIN_CONTENT_WIDTH = 1120
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
        return "N/A" if value is None else format_model_score(value)

    @staticmethod
    def _number(value: float | None, digits: int = 6) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    def _build(self) -> None:
        ttk.Label(self.content, text="VALIDATED EXPERIMENT RESULTS", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Model Comparison", style="PageTitle.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(self.content, text="Historical baseline comparison only: seven classes on the same frozen 545-image test set.", style="Muted.TLabel").pack(anchor="w")

        current = load_active_evaluation()
        active = SectionCard(self.content, "Active deployment - 15 classes / 670 test images")
        active.pack(fill="x", pady=(14, 10))
        ttk.Label(active, text=f"Float32: {current['correct']}/{current['total']} correct ({current['accuracy']:.2%}); Macro F1 {current['macro_f1']:.2%}. "
                  f"Mean model score {format_model_score(current['mean_confidence'])}; median {format_model_score(current['median_confidence'])}.",
                  style="BodyStrong.TLabel", wraplength=1050).pack(anchor="w")
        ttk.Label(active, text=f"Uncalibrated scores. Mean score on errors: {format_model_score(current['error_mean_confidence'])}; "
                  f"errors with score >=99%: {current['errors_above_99']}; saturated scores: {current['saturated_scores']}/{current['total']}. "
                  f"15-bin expected calibration error: {current['ece']:.4f} (descriptive test statistic, not a calibration fit). "
                  "The historical charts below use a different 7-class / 545-image test set; they do not rank the active model against those baselines.",
                  style="Body.TLabel", wraplength=1050).pack(anchor="w", pady=(8, 0))
        status_row = ttk.Frame(self.content, style="Page.TFrame"); status_row.pack(fill="x", pady=(14, 10))
        for record in self.records:
            card = ttk.Frame(status_row, style="Metric.TFrame", padding=(13, 10)); card.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Label(card, text=record.model, style="BodyStrong.TLabel").pack(anchor="w")
            kind = "selected" if record.model == "MobileNetV3 Large" else "incomplete" if record.status == "INCOMPLETE" else "complete"
            text = "HISTORICAL WINNER" if kind == "selected" else record.status
            StatusPill(card, text, kind).pack(anchor="w", pady=(6, 0))

        winner = SectionCard(self.content, "Best historical seven-class baseline"); winner.pack(fill="x")
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
            bar_labels = [format_model_score(value/100) for value in values] if percentage and "confidence" in field else None
            self.axis.bar_label(bars, labels=bar_labels, fmt="%.2f%%" if percentage else "%.6f", padding=3, fontsize=8)
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
        self._grouped(("mean_confidence", "median_confidence"), ("Mean Top-1 Confidence", "Median Top-1 Confidence"), True, "Historical Uncalibrated Model Scores")
        self.graph_note.configure(text="Historical raw Top-1 model scores, recomputed from stored probabilities. Near-one scores retain extra precision; exact 1.0 is labeled saturated. These scores are uncalibrated and do not measure classification accuracy.")

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
        ttk.Button(card, text="View Confusion Matrices", command=lambda: self.navigate("confusion")).pack(anchor="w")
        return card
