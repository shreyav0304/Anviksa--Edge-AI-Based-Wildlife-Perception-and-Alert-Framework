"""Scrollable pages with Linux wheel support and access to wide content."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

class ScrollablePage(ttk.Frame):
    MIN_CONTENT_WIDTH = 1060
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, style="Page.TFrame")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg="#f2f5f3", highlightthickness=0)
        vertical = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.content = ttk.Frame(self.canvas, style="Page.TFrame", padding=(20, 18, 20, 24))
        self._window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.content.bind("<Configure>", self._resize_content)
        self.canvas.bind("<Configure>", self._resize_content)
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _resize_content(self, _event=None) -> None:
        self.canvas.itemconfigure(self._window, width=max(self.canvas.winfo_width(), self.MIN_CONTENT_WIDTH))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        if not self.winfo_ismapped():
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = getattr(widget, "master", None)
        if widget is not self:
            return
        delta = -1 if getattr(event, "num", None) == 4 else 1 if getattr(event, "num", None) == 5 else int(-event.delta / 120)
        self.canvas.yview_scroll(delta, "units")
