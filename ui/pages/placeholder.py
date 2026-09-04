"""Safe Phase 1 placeholders for future modules."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class PlaceholderPage(ttk.Frame):
    def __init__(self, parent: tk.Misc, title: str, description: str, phase: str = "PLANNED") -> None:
        super().__init__(parent, style="Page.TFrame", padding=36)
        panel = ttk.Frame(self, style="Card.TFrame", padding=34)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text=phase, style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(panel, text=title, style="PageTitle.TLabel").pack(anchor="w", pady=(8, 12))
        ttk.Label(panel, text=description, style="Body.TLabel", wraplength=720).pack(anchor="w")
        ttk.Separator(panel).pack(fill="x", pady=24)
        ttk.Label(
            panel,
            text="This module is intentionally inactive in UI Phase 1. No model execution, hardware access, or dataset processing occurs here.",
            style="Muted.TLabel",
            wraplength=720,
        ).pack(anchor="w")
