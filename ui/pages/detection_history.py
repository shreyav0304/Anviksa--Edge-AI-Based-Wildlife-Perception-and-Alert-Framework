"""Local detection history viewer, filters, details, snapshots, and export."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ..components import MetricCard, SectionCard
from ..services.detection_history import DetectionEvent, DetectionHistoryService
from .base import ScrollablePage


class DetectionHistoryPage(ScrollablePage):
    def __init__(self, parent: tk.Misc, service: DetectionHistoryService) -> None:
        super().__init__(parent)
        self.service = service
        self.events: list[DetectionEvent] = []
        self.filtered_events: list[DetectionEvent] = []
        self.source_filter = tk.StringVar(value="All")
        self.class_filter = tk.StringVar(value="All")
        self.confidence_filter = tk.StringVar(value="All")
        self.search_text = tk.StringVar(value="")
        self.empty_var = tk.StringVar(value="")
        self.detail_vars: dict[str, tk.StringVar] = {}
        self.summary_vars: dict[str, tk.StringVar] = {}
        self._build()
        self.refresh_history()

    def _build(self) -> None:
        ttk.Label(self.content, text="LOCAL RUNTIME RECORDS", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(self.content, text="Detection History", style="PageTitle.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(self.content, text="Runtime detections and snapshots are separate from all training datasets.", style="Muted.TLabel").pack(anchor="w")

        summaries = ttk.Frame(self.content, style="Page.TFrame"); summaries.pack(fill="x", pady=(14, 8))
        labels = ("Total Events", "Image Events", "Video Events", "Webcam Events", "Snake Events", "Venomous Snake", "Non-Venomous Snake", "High-Threat Events")
        for column in range(4): summaries.columnconfigure(column, weight=1, uniform="summary")
        for index, label in enumerate(labels):
            variable = tk.StringVar(value="0"); self.summary_vars[label] = variable
            card = ttk.Frame(summaries, style="Metric.TFrame", padding=(12, 9)); card.grid(row=index//4, column=index%4, sticky="nsew", padx=4, pady=4)
            ttk.Label(card, text=label.upper(), style="MetricLabel.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=variable, style="MetricValue.TLabel").pack(anchor="w")

        filters = ttk.Frame(self.content, style="Card.TFrame", padding=14); filters.pack(fill="x", pady=(4, 10))
        for label, variable, values in (("Source", self.source_filter, ("All", "IMAGE", "VIDEO", "WEBCAM")), ("Class", self.class_filter, ("All", "Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake", "Venomous_Snake", "Wild_Boar")), ("Confidence", self.confidence_filter, ("All", "HIGH", "MODERATE", "LOW"))):
            ttk.Label(filters, text=label.upper(), style="MetricLabel.TLabel").pack(side="left", padx=(8, 4))
            box = ttk.Combobox(filters, textvariable=variable, values=values, state="readonly", width=18); box.pack(side="left"); box.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())
        ttk.Label(filters, text="SEARCH", style="MetricLabel.TLabel").pack(side="left", padx=(12, 4))
        search = ttk.Entry(filters, textvariable=self.search_text, width=20); search.pack(side="left"); search.bind("<Return>", lambda _e: self.apply_filters())
        ttk.Button(filters, text="Search", command=self.apply_filters).pack(side="left", padx=5)
        ttk.Button(filters, text="Clear Filters", command=self.clear_filters).pack(side="left")

        table_card = SectionCard(self.content, "History table")
        table_card.pack(fill="x")
        columns = ("timestamp", "source", "prediction", "confidence", "status", "threat", "snapshot")
        self.table = ttk.Treeview(table_card, columns=columns, show="headings", height=9)
        headings = ("Timestamp", "Source", "Prediction", "Confidence", "Status", "Threat", "Snapshot")
        widths = (190, 90, 170, 100, 100, 90, 100)
        for column, heading, width in zip(columns, headings, widths, strict=True):
            self.table.heading(column, text=heading); self.table.column(column, width=width, anchor="center")
        self.table.pack(fill="x"); self.table.bind("<<TreeviewSelect>>", self._selection_changed)
        ttk.Label(table_card, textvariable=self.empty_var, style="Notice.TLabel").pack(anchor="w", pady=(7, 0))

        details = SectionCard(self.content, "Selected event details"); details.pack(fill="x", pady=(12, 0))
        grid = ttk.Frame(details, style="Card.TFrame"); grid.pack(fill="x")
        fields = ("Event ID", "Timestamp", "Source", "Prediction", "Confidence", "Top-2", "Margin", "Threat", "Stable", "Inference", "Snapshot", "Notes")
        for index, label in enumerate(fields):
            variable = tk.StringVar(value="—"); self.detail_vars[label] = variable
            cell = ttk.Frame(grid, style="Card.TFrame"); cell.grid(row=index//3, column=index%3, sticky="nsew", padx=8, pady=5)
            grid.columnconfigure(index%3, weight=1, uniform="detail")
            ttk.Label(cell, text=label.upper(), style="MetricLabel.TLabel").pack(anchor="w")
            ttk.Label(cell, textvariable=variable, style="BodyStrong.TLabel", wraplength=320).pack(anchor="w")
        actions = ttk.Frame(details, style="Card.TFrame"); actions.pack(fill="x", pady=(10, 0))
        self.view_button = ttk.Button(actions, text="View Snapshot", command=self.view_snapshot, state="disabled"); self.view_button.pack(side="left")
        ttk.Button(actions, text="Refresh History", command=self.refresh_history).pack(side="left", padx=6)
        ttk.Button(actions, text="Export Filtered CSV", command=self.export_filtered).pack(side="left")

        limitation = SectionCard(self.content, "Important limitation"); limitation.pack(fill="x", pady=(12, 0))
        ttk.Label(limitation, text="The current seven-class model has no Background/No Animal/Unknown class. Empty or unsupported scenes can produce false wildlife events. History counts are runtime detections—not accuracy, precision, recall, or F1.", style="Warning.TLabel", wraplength=1050).pack(fill="x")
        ttk.Label(limitation, text="Threat level is application metadata, not a neural-network output. No history deletion controls are provided in this phase.", style="Notice.TLabel").pack(anchor="w", pady=(8, 0))

    def refresh_history(self) -> None:
        self.events = self.service.load_events()
        self._update_summaries()
        self.apply_filters()

    def _update_summaries(self) -> None:
        counts = {
            "Total Events": len(self.events),
            "Image Events": sum(e.source_type == "IMAGE" for e in self.events),
            "Video Events": sum(e.source_type == "VIDEO" for e in self.events),
            "Webcam Events": sum(e.source_type == "WEBCAM" for e in self.events),
            "Snake Events": sum(e.predicted_class in {"Venomous_Snake", "Non_Venomous_Snake"} for e in self.events),
            "Venomous Snake": sum(e.predicted_class == "Venomous_Snake" for e in self.events),
            "Non-Venomous Snake": sum(e.predicted_class == "Non_Venomous_Snake" for e in self.events),
            "High-Threat Events": sum(e.threat_level == "HIGH" for e in self.events),
        }
        for label, count in counts.items(): self.summary_vars[label].set(str(count))

    def apply_filters(self) -> None:
        query = self.search_text.get().strip().casefold()
        self.filtered_events = [event for event in self.events if (self.source_filter.get() == "All" or event.source_type == self.source_filter.get()) and (self.class_filter.get() == "All" or event.predicted_class == self.class_filter.get()) and (self.confidence_filter.get() == "All" or event.confidence_status == self.confidence_filter.get()) and (not query or query in " ".join((event.event_id, event.source_name, event.display_class, event.notes)).casefold())]
        self.table.delete(*self.table.get_children())
        for index, event in enumerate(reversed(self.filtered_events)):
            original_index = len(self.filtered_events) - 1 - index
            self.table.insert("", "end", iid=str(original_index), values=(event.timestamp, event.source_type, event.display_class, f"{event.confidence*100:.2f}%", event.confidence_status, event.threat_level, "Available" if self.service.resolve_snapshot(event.snapshot_path) else "Unavailable"))
        message = "No detection history recorded yet." if not self.events else "No events match the current filters." if not self.filtered_events else f"Showing {len(self.filtered_events)} event(s)."
        if self.service.malformed_rows: message += f" Skipped {self.service.malformed_rows} malformed row(s); the master file was not changed."
        self.empty_var.set(message); self._clear_details()

    def clear_filters(self) -> None:
        self.source_filter.set("All"); self.class_filter.set("All"); self.confidence_filter.set("All"); self.search_text.set(""); self.apply_filters()

    def _selection_changed(self, _event=None) -> None:
        selection = self.table.selection()
        if not selection: return
        event = self.filtered_events[int(selection[0])]
        values = {"Event ID": event.event_id, "Timestamp": event.timestamp, "Source": f"{event.source_type}: {event.source_name}", "Prediction": event.display_class, "Confidence": f"{event.confidence*100:.2f}% ({event.confidence_status})", "Top-2": f"{event.top2_class} — {event.top2_probability*100:.2f}%", "Margin": f"{event.margin*100:.2f} percentage points", "Threat": event.threat_level, "Stable": f"{event.stable} (count {event.stable_count})", "Inference": f"{event.inference_time_ms:.2f} ms", "Snapshot": event.snapshot_path or "Unavailable", "Notes": event.notes}
        for key, value in values.items(): self.detail_vars[key].set(value)
        self.view_button.configure(state="normal" if self.service.resolve_snapshot(event.snapshot_path) else "disabled")

    def _clear_details(self) -> None:
        for variable in self.detail_vars.values(): variable.set("—")
        self.view_button.configure(state="disabled")

    def _selected_event(self) -> DetectionEvent | None:
        selection = self.table.selection()
        return self.filtered_events[int(selection[0])] if selection else None

    def view_snapshot(self) -> None:
        event = self._selected_event()
        path = self.service.resolve_snapshot(event.snapshot_path) if event else None
        if path is None:
            messagebox.showinfo("Snapshot unavailable", "Snapshot unavailable", parent=self); return
        try:
            with Image.open(path) as opened:
                opened.load(); image = opened.convert("RGB"); image.thumbnail((850, 620), Image.Resampling.LANCZOS)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Snapshot unavailable", str(exc), parent=self); return
        window = tk.Toplevel(self); window.title("Detection Snapshot"); window.configure(bg="#101915")
        photo = ImageTk.PhotoImage(image); label = tk.Label(window, image=photo, bg="#101915"); label.image = photo; label.pack(padx=12, pady=12)
        ttk.Label(window, text=f"{event.timestamp} • {event.display_class}", style="BodyStrong.TLabel").pack(pady=(0, 12))

    def export_filtered(self) -> None:
        if not self.filtered_events:
            messagebox.showinfo("Nothing to export", "No filtered events are available.", parent=self); return
        filename = filedialog.asksaveasfilename(parent=self, title="Export filtered detection history", defaultextension=".csv", filetypes=(("CSV", "*.csv"),))
        if not filename: return
        try: self.service.export_csv(self.filtered_events, Path(filename))
        except OSError as exc: messagebox.showerror("Export failed", str(exc), parent=self); return
        messagebox.showinfo("Export complete", f"Exported {len(self.filtered_events)} event(s).", parent=self)

    def on_show(self) -> None:
        self.refresh_history()
