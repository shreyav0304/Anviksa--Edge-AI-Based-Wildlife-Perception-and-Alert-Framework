"""Small reusable ttk dashboard components."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SectionCard(ttk.Frame):
    def __init__(self, parent: tk.Misc, title: str, **kwargs) -> None:
        super().__init__(parent, style="Card.TFrame", padding=18, **kwargs)
        ttk.Label(self, text=title.upper(), style="Section.TLabel").pack(anchor="w")
        ttk.Separator(self).pack(fill="x", pady=(9, 13))


class MetricCard(ttk.Frame):
    def __init__(self, parent: tk.Misc, label: str, value: str, detail: str = "") -> None:
        super().__init__(parent, style="Metric.TFrame", padding=(16, 14))
        labels = []
        title = ttk.Label(self, text=label.upper(), style="TileMetricLabel.TLabel")
        title.pack(anchor="w")
        labels.append(title)
        number = ttk.Label(self, text=value, style="TileMetricValue.TLabel")
        number.pack(anchor="w", pady=(6, 2))
        labels.append(number)
        if detail:
            description = ttk.Label(self, text=detail, style="TileMetricDetail.TLabel")
            description.pack(anchor="w")
            labels.append(description)
        self.bind("<Configure>", lambda event: [item.configure(wraplength=max(100,event.width-32)) for item in labels])


class StatusPill(ttk.Label):
    STYLES = {
        "complete": "Complete.Status.TLabel",
        "selected": "Selected.Status.TLabel",
        "incomplete": "Incomplete.Status.TLabel",
        "planned": "Planned.Status.TLabel",
    }

    def __init__(self, parent: tk.Misc, text: str, kind: str) -> None:
        super().__init__(parent, text=f"  {text}  ", style=self.STYLES[kind])
