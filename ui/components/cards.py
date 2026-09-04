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
        ttk.Label(self, text=label.upper(), style="MetricLabel.TLabel").pack(anchor="w")
        ttk.Label(self, text=value, style="MetricValue.TLabel").pack(anchor="w", pady=(6, 2))
        if detail:
            ttk.Label(self, text=detail, style="MetricDetail.TLabel", wraplength=220).pack(anchor="w")


class StatusPill(ttk.Label):
    STYLES = {
        "complete": "Complete.Status.TLabel",
        "selected": "Selected.Status.TLabel",
        "incomplete": "Incomplete.Status.TLabel",
        "planned": "Planned.Status.TLabel",
    }

    def __init__(self, parent: tk.Misc, text: str, kind: str) -> None:
        super().__init__(parent, text=f"  {text}  ", style=self.STYLES[kind])
