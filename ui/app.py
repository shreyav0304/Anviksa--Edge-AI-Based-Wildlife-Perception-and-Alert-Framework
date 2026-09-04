"""Main offline Tkinter application shell for Anvīkṣa AI."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from .data import load_dashboard_data
from .pages import ConfusionMatrixPage, DetectionHistoryPage, HomePage, ImageAnalysisPage, ModelComparisonPage, PlaceholderPage, VideoAnalysisPage
from .services import DetectionHistoryService, load_comparison_records, load_confusion_records


NAVIGATION = (
    ("home", "⌂  Home / Dashboard", "Home / Dashboard"),
    ("image", "▧  Image Analysis", "Image Analysis"),
    ("video", "▶  Video / Webcam", "Video / Webcam"),
    ("comparison", "≋  Model Comparison", "Model Comparison"),
    ("confusion", "▦  Confusion Matrix", "Confusion Matrix"),
    ("history", "◷  Detection History", "Detection History"),
    ("system", "⚙  System / Deployment", "System / Deployment"),
    ("about", "ⓘ  About Project", "About Project"),
)


class AnviksaApp(tk.Tk):
    """Modular application shell; backend execution is intentionally absent."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ANVĪKṢA AI")
        self.geometry("1280x820")
        self.minsize(1024, 680)
        self.configure(bg="#132a22")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_styles()
        self._pages: dict[str, ttk.Frame] = {}
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._current_page: str | None = None
        self.history_service = DetectionHistoryService()
        self._build_shell()
        self.show_page("home")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Page.TFrame", background="#f2f5f3")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Metric.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Header.TFrame", background="#173a2e")
        style.configure("Sidebar.TFrame", background="#102b22")
        style.configure("Brand.TLabel", background="#173a2e", foreground="#ffffff", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background="#173a2e", foreground="#c9ddd5", font=("Segoe UI", 10))
        style.configure("SidebarTitle.TLabel", background="#102b22", foreground="#6fbf92", font=("Segoe UI", 9, "bold"))
        style.configure("Footer.TLabel", background="#102b22", foreground="#8fac9f", font=("Segoe UI", 8))
        style.configure("Nav.TButton", background="#102b22", foreground="#dce9e3", font=("Segoe UI", 10), anchor="w", padding=(18, 12), borderwidth=0)
        style.map("Nav.TButton", background=[("active", "#1b4939")])
        style.configure("Active.Nav.TButton", background="#287454", foreground="#ffffff", font=("Segoe UI", 10, "bold"), anchor="w", padding=(18, 12), borderwidth=0)
        style.map("Active.Nav.TButton", background=[("active", "#287454")])
        style.configure("Eyebrow.TLabel", background="#f2f5f3", foreground="#287454", font=("Segoe UI", 9, "bold"))
        style.configure("PageTitle.TLabel", background="#f2f5f3", foreground="#14251e", font=("Segoe UI", 25, "bold"))
        style.configure("Section.TLabel", background="#ffffff", foreground="#2b6c52", font=("Segoe UI", 9, "bold"))
        style.configure("MetricLabel.TLabel", background="#ffffff", foreground="#718078", font=("Segoe UI", 8, "bold"))
        style.configure("MetricValue.TLabel", background="#ffffff", foreground="#153a2d", font=("Segoe UI", 16, "bold"))
        style.configure("MetricDetail.TLabel", background="#ffffff", foreground="#7a8780", font=("Segoe UI", 8))
        style.configure("Body.TLabel", background="#ffffff", foreground="#34443d", font=("Segoe UI", 10))
        style.configure("BodyStrong.TLabel", background="#ffffff", foreground="#203c31", font=("Segoe UI", 10, "bold"))
        style.configure("ResultValue.TLabel", background="#ffffff", foreground="#153a2d", font=("Segoe UI", 11, "bold"))
        style.configure("Winner.TLabel", background="#ffffff", foreground="#17623f", font=("Segoe UI", 20, "bold"))
        style.configure("Small.TLabel", background="#ffffff", foreground="#4c5c54", font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background="#f2f5f3", foreground="#68766f", font=("Segoe UI", 9))
        style.configure("Pipeline.TLabel", background="#e8f2ed", foreground="#173a2e", font=("Segoe UI", 9, "bold"), padding=(8, 10))
        style.configure("Arrow.TLabel", background="#ffffff", foreground="#4a9b73", font=("Segoe UI", 14, "bold"))
        style.configure("Notice.TLabel", background="#ffffff", foreground="#53675d", font=("Segoe UI", 9, "italic"))
        style.configure("Warning.TLabel", background="#fff7e6", foreground="#714e16", font=("Segoe UI", 9, "bold"), padding=12)
        for name, bg, fg in (("Complete", "#dff3e8", "#226847"), ("Selected", "#287454", "#ffffff"), ("Incomplete", "#f9e3df", "#983f32"), ("Planned", "#e8eaf6", "#4d5687")):
            style.configure(f"{name}.Status.TLabel", background=bg, foreground=fg, font=("Segoe UI", 8, "bold"), padding=(4, 3))
        style.configure("Primary.TButton", background="#287454", foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=(15, 8))
        style.map("Primary.TButton", background=[("active", "#1f6044"), ("disabled", "#93afa2")])
        style.configure("Graph.TButton", background="#e7efeb", foreground="#315045", padding=(12, 7))
        style.configure("Active.Graph.TButton", background="#287454", foreground="#ffffff", padding=(12, 7))

    def _build_shell(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame", padding=(24, 15))
        header.pack(fill="x")
        ttk.Label(header, text="ANVĪKṢA AI", style="Brand.TLabel").pack(side="left")
        ttk.Label(header, text="Edge AI-Based Wildlife Perception and Alert Framework", style="Subtitle.TLabel").pack(side="left", padx=22)
        ttk.Label(header, text="OFFLINE  •  PHASE 6", style="Subtitle.TLabel").pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=235, padding=(12, 20))
        sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="NAVIGATION", style="SidebarTitle.TLabel").pack(anchor="w", padx=12, pady=(0, 10))
        for key, label, _title in NAVIGATION:
            button = ttk.Button(sidebar, text=label, style="Nav.TButton", command=lambda page=key: self.show_page(page))
            button.pack(fill="x", pady=2); self._nav_buttons[key] = button
        ttk.Label(sidebar, text="Classification system\nNo network connection required", style="Footer.TLabel").pack(side="bottom", anchor="w", padx=12)

        self.page_host = ttk.Frame(body, style="Page.TFrame")
        self.page_host.pack(side="left", fill="both", expand=True)
        self._create_pages()

    def _create_pages(self) -> None:
        try:
            data = load_dashboard_data()
        except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Validated artifact unavailable", str(exc), parent=self)
            raise
        self._pages["home"] = HomePage(self.page_host, data)
        self._pages["image"] = ImageAnalysisPage(self.page_host, self.history_service)
        self._pages["video"] = VideoAnalysisPage(self.page_host, self.history_service)
        self._pages["comparison"] = ModelComparisonPage(self.page_host, load_comparison_records(), self.show_page)
        self._pages["confusion"] = ConfusionMatrixPage(self.page_host, load_confusion_records(), self.show_page)
        self._pages["history"] = DetectionHistoryPage(self.page_host, self.history_service)
        placeholders = {
            "system": ("System / Deployment", "Raspberry Pi, camera, mmWave, LED, and buzzer integration remain pending hardware work."),
            "about": ("About Project", "Anvīkṣa AI is an offline Edge AI wildlife classification and alert research framework."),
        }
        for key, (title, description) in placeholders.items():
            self._pages[key] = PlaceholderPage(self.page_host, title, description)
        for page in self._pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_page(self, key: str) -> None:
        if key not in self._pages:
            raise KeyError(f"Unknown page: {key}")
        if self._current_page and self._current_page != key:
            previous = self._pages[self._current_page]
            if hasattr(previous, "on_hide"):
                previous.on_hide()
        self._pages[key].tkraise()
        self._current_page = key
        if hasattr(self._pages[key], "on_show"):
            self._pages[key].on_show()
        for name, button in self._nav_buttons.items():
            button.configure(style="Active.Nav.TButton" if name == key else "Nav.TButton")

    def close(self) -> None:
        for page in self._pages.values():
            if hasattr(page, "close"):
                page.close()
        self.destroy()


def main() -> int:
    app = AnviksaApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
