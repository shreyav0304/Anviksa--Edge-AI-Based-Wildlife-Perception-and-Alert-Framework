"""Offline dashboard with independent, lazily constructed pages."""
from __future__ import annotations

import logging
import tkinter as tk
from tkinter import font as tkfont, ttk
from importlib import import_module

LOG = logging.getLogger(__name__)
NAV_ITEMS = (
    ("home", "Overview"), ("image", "Image Analysis"), ("video", "Video / Webcam"),
    ("comparison", "Model Comparison"), ("confusion", "Confusion Matrix"),
    ("history", "Detection History"), ("system", "System Deployment"), ("about", "About Project"),
)

class AnviksaApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Anv\u012bk\u1e63a AI")
        self.geometry(f"{min(1180, self.winfo_screenwidth())}x{min(720, max(320, self.winfo_screenheight() - 60))}+0+0")
        self.minsize(min(480, self.winfo_screenwidth()), min(320, self.winfo_screenheight()))
        self.protocol("WM_DELETE_WINDOW", self.close)
        from ui.services.detection_history import DetectionHistoryService
        self.history_service = DetectionHistoryService()
        self._pages = {}
        self._nav_buttons = {}
        self._current_page = None
        self._failed_pages = set()
        self._closing = False
        self.status_var = tk.StringVar(value="Offline wildlife dashboard")
        self._configure_styles()
        self._build_layout()
        self.show_page("home")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        style.configure(".", font=(family, 10))
        for name, color in (("Root.TFrame", "#f2f5f3"), ("Page.TFrame", "#f2f5f3"),
                            ("Sidebar.TFrame", "#153a2f"), ("Card.TFrame", "#ffffff"),
                            ("Metric.TFrame", "#e9efec")):
            style.configure(name, background=color)
        for name in ("Body", "BodyStrong", "Small", "Section", "ResultValue", "Winner",
                     "MetricLabel", "MetricValue", "MetricDetail", "Notice", "Warning", "Muted", "Eyebrow", "PageTitle"):
            size = 22 if name == "PageTitle" else 18 if name in {"MetricValue", "Winner"} else 12 if name == "ResultValue" else 9 if name in {"Small", "MetricLabel", "MetricDetail", "Eyebrow"} else 10
            bold = name in {"PageTitle", "BodyStrong", "Section", "ResultValue", "Winner", "MetricValue"}
            style.configure(name + ".TLabel", foreground="#173a2e", background="#ffffff",
                            font=(family, size, "bold" if bold else "normal"))
        style.configure("PageBody.TLabel", background="#f2f5f3", foreground="#53665c")
        for name in ("Eyebrow", "PageTitle", "Muted"):
            style.configure(name + ".TLabel", background="#f2f5f3")
        for name in ("MetricLabel", "MetricValue", "MetricDetail"):
            style.configure("Tile" + name + ".TLabel", background="#e9efec", foreground="#173a2e", font=(family, 18 if name == "MetricValue" else 9, "bold" if name == "MetricValue" else "normal"))
        style.configure("Pipeline.TLabel", background="#ffffff", foreground="#173a2e")
        style.configure("Arrow.TLabel", background="#ffffff", foreground="#53665c")
        style.configure("Warning.TLabel", foreground="#9d3b25")
        style.configure("Muted.TLabel", foreground="#53665c")
        style.configure("Title.TLabel", background="#153a2f", foreground="#ffffff", font=(family, 16, "bold"))
        style.configure("Subtitle.TLabel", background="#153a2f", foreground="#d7e8e1", font=(family, 9))
        for name in ("Nav.TButton", "Active.Nav.TButton", "Graph.TButton", "Active.Graph.TButton"):
            style.configure(name, padding=(8, 4), font=(family, 10, "bold" if name.startswith("Active") else "normal"))
        style.map("Active.Nav.TButton", background=[("!disabled", "#cce2d7")])
        style.configure("Primary.TButton", padding=(8, 5), font=(family, 10, "bold"))
        for name, color in (("Complete", "#287454"), ("Selected", "#153a2f"),
                            ("Incomplete", "#9d3b25"), ("Planned", "#53665c")):
            style.configure(name + ".Status.TLabel", foreground=color, background="#ffffff")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame")
        root.pack(fill="both", expand=True)
        sidebar = ttk.Frame(root, style="Sidebar.TFrame", width=210)
        sidebar.pack(side="left", fill="y")
        self.sidebar = sidebar
        sidebar.pack_propagate(False)
        header = ttk.Frame(sidebar, style="Sidebar.TFrame")
        header.pack(fill="x", padx=14, pady=(8, 6))
        ttk.Label(header, text="Anv\u012bk\u1e63a AI", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Edge AI Wildlife Perception and Alert Framework", style="Subtitle.TLabel", wraplength=175).pack(anchor="w", pady=(3, 0))
        nav_host = ttk.Frame(sidebar, style="Sidebar.TFrame")
        nav_host.pack(fill="both", expand=True)
        self.nav_canvas = tk.Canvas(nav_host, bg="#153a2f", highlightthickness=0, width=190)
        scroll = ttk.Scrollbar(nav_host, orient="vertical", command=self.nav_canvas.yview)
        scroll.pack(side="right", fill="y")
        self.nav_canvas.pack(side="left", fill="both", expand=True)
        self.nav_canvas.configure(yscrollcommand=scroll.set)
        self.nav_frame = ttk.Frame(self.nav_canvas, style="Sidebar.TFrame")
        window = self.nav_canvas.create_window((0, 0), window=self.nav_frame, anchor="nw")
        self.nav_frame.bind("<Configure>", lambda _event: self.nav_canvas.configure(scrollregion=self.nav_canvas.bbox("all")))
        self.nav_canvas.bind("<Configure>", lambda event: self.nav_canvas.itemconfigure(window, width=event.width))
        for key, label in NAV_ITEMS:
            button = ttk.Button(self.nav_frame, text=label, style="Nav.TButton", command=lambda page=key: self.show_page(page))
            button.pack(fill="x", padx=8, pady=3)
            self._nav_buttons[key] = button
        self.bind("<Button-4>", self._scroll_navigation, add="+")
        self.bind("<Button-5>", self._scroll_navigation, add="+")
        self.bind("<MouseWheel>", self._scroll_navigation, add="+")
        right = ttk.Frame(root, style="Page.TFrame")
        right.pack(side="left", fill="both", expand=True)
        self.section_var = tk.StringVar(value=dict(NAV_ITEMS)["home"])
        toolbar = ttk.Frame(right, style="Page.TFrame", padding=(8, 6))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Section", style="PageBody.TLabel").pack(side="left", padx=(0, 8))
        self.section_picker = ttk.Combobox(
            toolbar, textvariable=self.section_var, values=tuple(dict(NAV_ITEMS).values()),
            state="readonly", width=24,
        )
        self.section_picker.pack(side="left", fill="x", expand=True)
        self.section_picker.bind("<<ComboboxSelected>>", self._select_section)
        self.bind("<Configure>", self._adapt_navigation, add="+")
        for index, (key, _label) in enumerate(NAV_ITEMS, 1):
            self.bind(f"<Alt-Key-{index}>", lambda _event, page=key: self.show_page(page))
        self.page_host = ttk.Frame(right, style="Page.TFrame")
        self.page_host.pack(fill="both", expand=True)
        status = ttk.Label(right, textvariable=self.status_var, style="PageBody.TLabel", padding=(8, 4), wraplength=700)
        status.pack(fill="x")
        right.bind("<Configure>", lambda event: status.configure(wraplength=max(160, event.width - 20)))

    def _select_section(self, _event=None) -> None:
        selected = self.section_var.get()
        for key, label in NAV_ITEMS:
            if label == selected:
                self.show_page(key)
                break

    def _adapt_navigation(self, event) -> None:
        if event.widget is not self:
            return
        # Leave room for content on Pi touchscreens; the section picker is always available.
        if event.width < 900:
            self.sidebar.pack_forget()
        elif not self.sidebar.winfo_manager():
            self.sidebar.pack(side="left", fill="y", before=self.sidebar.master.winfo_children()[1])

    def _reveal_navigation(self, key) -> None:
        self.nav_canvas.update_idletasks()
        button = self._nav_buttons[key]
        height = max(1, self.nav_frame.winfo_height())
        top = button.winfo_y()
        bottom = top + button.winfo_height()
        visible_top = self.nav_canvas.canvasy(0)
        visible_bottom = visible_top + self.nav_canvas.winfo_height()
        if top < visible_top:
            self.nav_canvas.yview_moveto(top / height)
        elif bottom > visible_bottom:
            self.nav_canvas.yview_moveto((bottom - self.nav_canvas.winfo_height()) / height)

    def _scroll_navigation(self, event) -> None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget not in (self.nav_frame, self.nav_canvas):
            widget = getattr(widget, "master", None)
        if widget is None:
            return
        delta = -1 if getattr(event, "num", None) == 4 else 1 if getattr(event, "num", None) == 5 else (-1 if event.delta > 0 else 1 if event.delta < 0 else 0)
        self.nav_canvas.yview_scroll(delta, "units")

    def _make_page(self, key):
        if key == "home":
            from ui.data import load_dashboard_data
            from ui.pages.home import HomePage
            return HomePage(self.page_host, load_dashboard_data())
        if key in ("system", "about"):
            from ui.pages.project_info import ProjectInfoPage
            return ProjectInfoPage(self.page_host, dict(NAV_ITEMS)[key], self.history_service.root)
        modules = {
            "image": ("image_analysis", "ImageAnalysisPage"),
            "video": ("video_analysis", "VideoAnalysisPage"),
            "history": ("detection_history", "DetectionHistoryPage"),
            "comparison": ("model_comparison", "ModelComparisonPage"),
            "confusion": ("confusion_matrix", "ConfusionMatrixPage"),
        }
        module, name = modules[key]
        page_class = getattr(import_module("ui.pages." + module), name)
        if key == "comparison":
            from ui.services.comparison_data import load_comparison_records
            return page_class(self.page_host, load_comparison_records(), self.show_page)
        if key == "confusion":
            from ui.services.confusion_data import load_confusion_records
            return page_class(self.page_host, load_confusion_records(), self.show_page)
        return page_class(self.page_host, self.history_service)

    def _unavailable_page(self, key, exc):
        from ui.pages.base import ScrollablePage
        page = ScrollablePage(self.page_host)
        ttk.Label(page.content, text=dict(NAV_ITEMS)[key], style="PageTitle.TLabel").pack(anchor="w", pady=(0, 16))
        ttk.Label(page.content, text=f"This section could not load: {exc}", style="Warning.TLabel", wraplength=800).pack(anchor="w")
        ttk.Label(page.content, text="Other sections remain available. Check the app environment and required artifacts.\nSetup: docs/raspberry_pi_setup.md\nDiagnostics: python -B scripts/check_pi_dependencies.py", style="Body.TLabel", wraplength=800).pack(anchor="w", pady=12)
        ttk.Button(page.content, text="Retry this section", command=lambda: self.show_page(key, retry=True)).pack(anchor="w")
        return page

    def show_page(self, key: str, retry: bool = False) -> None:
        if key not in dict(NAV_ITEMS):
            raise KeyError(f"Unknown page: {key}")
        if self._current_page == key and not retry:
            return
        previous = self._pages.get(self._current_page)
        if previous is not None and hasattr(previous, "on_hide"):
            try:
                previous.on_hide()
            except Exception:
                LOG.exception("Unable to stop previous page")
                self.status_var.set("Unable to stop previous page; see terminal diagnostics.")
                return
        if retry and key in self._pages:
            self._pages.pop(key).destroy()
            self._failed_pages.discard(key)
        if key not in self._pages:
            # Failed constructors can leave widgets behind; clear only their new children.
            before = set(self.page_host.winfo_children())
            try:
                page = self._make_page(key)
            except Exception as exc:
                LOG.exception("Unable to load %s", key)
                for child in set(self.page_host.winfo_children()) - before:
                    if hasattr(child, "close"):
                        child.close()
                    child.destroy()
                page = self._unavailable_page(key, exc)
                self._failed_pages.add(key)
            self._pages[key] = page
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._current_page = key
        self.section_var.set(dict(NAV_ITEMS)[key])
        self._reveal_navigation(key)
        page = self._pages[key]
        page.tkraise()
        self.status_var.set("Section unavailable; see details and retry." if key in self._failed_pages else dict(NAV_ITEMS)[key])
        if hasattr(page, "on_show"):
            try:
                page.on_show()
            except Exception as exc:
                LOG.exception("Unable to refresh %s", key)
                self.status_var.set(f"Refresh failed: {exc}")
        for name, button in self._nav_buttons.items():
            button.configure(style="Active.Nav.TButton" if name == key else "Nav.TButton")

    def report_callback_exception(self, exception, value, traceback):
        LOG.error("UI callback failed", exc_info=(exception, value, traceback))
        self.status_var.set(f"Action failed: {value}. See terminal diagnostics.")

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        for page in self._pages.values():
            if hasattr(page, "close"):
                try:
                    page.close()
                except Exception:
                    LOG.exception("Unable to close page")
        self.destroy()

def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        app = AnviksaApp()
    except tk.TclError as exc:
        LOG.error("Cannot open the desktop interface: %s. Launch from an active graphical desktop; plain SSH has no display.", exc)
        return 1
    app.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
