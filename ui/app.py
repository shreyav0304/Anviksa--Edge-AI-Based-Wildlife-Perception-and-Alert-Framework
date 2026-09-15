from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk

from ui.data import load_dashboard_data
from ui.pages.home import HomePage
from ui.pages.image_analysis import ImageAnalysisPage
from ui.pages.video_analysis import VideoAnalysisPage
from ui.pages.model_comparison import ModelComparisonPage
from ui.pages.confusion_matrix import ConfusionMatrixPage
from ui.pages.detection_history import DetectionHistoryPage
from ui.pages.placeholder import PlaceholderPage
from ui.services.detection_history import DetectionHistoryService


class AnviksaApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("ANVĪKṢA AI")
        self.geometry("1180x720")
        self.minsize(1000, 650)

        self.protocol("WM_DELETE_WINDOW", self.close)

        self.history_service = DetectionHistoryService()

        self._pages: dict[str, ttk.Frame] = {}
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._current_page: str | None = None

        self._configure_styles()
        self._build_layout()
        self._create_pages()

        self.show_page("home")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Root.TFrame",
            background="#ffffff",
        )

        style.configure(
            "Sidebar.TFrame",
            background="#153a2f",
        )

        style.configure(
            "Page.TFrame",
            background="#ffffff",
        )

        style.configure(
            "Title.TLabel",
            background="#153a2f",
            foreground="#ffffff",
            font=("Segoe UI", 16, "bold"),
        )

        style.configure(
            "Subtitle.TLabel",
            background="#153a2f",
            foreground="#d7e8e1",
            font=("Segoe UI", 9),
        )

        style.configure(
            "Nav.TButton",
            font=("Segoe UI", 10),
            padding=(12, 8),
        )

        style.configure(
            "Active.Nav.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
        )

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame")
        root.pack(fill="both", expand=True)

        sidebar = ttk.Frame(
            root,
            style="Sidebar.TFrame",
            width=210,
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        header = ttk.Frame(
            sidebar,
            style="Sidebar.TFrame",
        )
        header.pack(
            fill="x",
            padx=14,
            pady=(18, 15),
        )

        ttk.Label(
            header,
            text="ANVĪKṢA AI",
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Edge AI Wildlife Perception and Alert Framework",
            style="Subtitle.TLabel",
            wraplength=175,
        ).pack(
            anchor="w",
            pady=(3, 0),
        )

        nav_items = [
            ("home", "Overview"),
            ("image", "Image Analysis"),
            ("video", "Video / Webcam"),
            ("comparison", "Model Comparison"),
            ("confusion", "Confusion Matrix"),
            ("history", "Detection History"),
            ("system", "System / Deployment"),
            ("about", "About Project"),
        ]

        for key, label in nav_items:
            button = ttk.Button(
                sidebar,
                text=label,
                style="Nav.TButton",
                command=lambda page_key=key: self.show_page(page_key),
            )
            button.pack(
                fill="x",
                padx=12,
                pady=3,
            )

            self._nav_buttons[key] = button

        self.page_host = ttk.Frame(
            root,
            style="Page.TFrame",
        )
        self.page_host.pack(
            side="left",
            fill="both",
            expand=True,
        )

    def _create_pages(self) -> None:
        # ============================================================
        # HISTORICAL DASHBOARD DATA
        # ============================================================
        try:
            data = load_dashboard_data()

        except (
            FileNotFoundError,
            OSError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:

            messagebox.showwarning(
                "Historical dashboard unavailable",
                (
                    f"{exc}\n\n"
                    "The historical V1 dashboard artifact is not available "
                    "on this Raspberry Pi.\n\n"
                    "Edge AI inference and deployment features will "
                    "continue to operate normally."
                ),
                parent=self,
            )

            data = None

        # ============================================================
        # HOME PAGE
        # ============================================================
        if data is not None:
            self._pages["home"] = HomePage(
                self.page_host,
                data,
            )

        else:
            home = ttk.Frame(
                self.page_host,
                style="Page.TFrame",
            )

            content = ttk.Frame(
                home,
                style="Page.TFrame",
            )
            content.pack(
                fill="both",
                expand=True,
                padx=40,
                pady=40,
            )

            ttk.Label(
                content,
                text="ANVĪKṢA AI",
                font=("Segoe UI", 24, "bold"),
                background="#ffffff",
            ).pack(
                pady=(80, 12),
            )

            ttk.Label(
                content,
                text="Edge AI-Based Wildlife Perception and Alert Framework",
                font=("Segoe UI", 13),
                background="#ffffff",
            ).pack(
                pady=(0, 30),
            )

            ttk.Label(
                content,
                text=(
                    "Historical MobileNetV3 V1 dashboard metrics are not "
                    "available in this Raspberry Pi repository.\n\n"
                    "The validated Edge AI deployment model remains available.\n"
                    "Image Analysis, Video/Webcam inference and deployment "
                    "features can still be used normally."
                ),
                justify="center",
                background="#ffffff",
                font=("Segoe UI", 11),
            ).pack(
                pady=10,
            )

            self._pages["home"] = home

        # ============================================================
        # IMAGE ANALYSIS
        # ============================================================
        self._pages["image"] = ImageAnalysisPage(
            self.page_host,
            self.history_service,
        )

        # ============================================================
        # VIDEO / WEBCAM
        # ============================================================
        self._pages["video"] = VideoAnalysisPage(
            self.page_host,
            self.history_service,
        )

        # ============================================================
        # MODEL COMPARISON
        # ============================================================
        comparison_records = self._safe_load_comparison_records()

        if comparison_records:
            self._pages["comparison"] = ModelComparisonPage(
                self.page_host,
                comparison_records,
                self.show_page,
            )

        else:
            comparison_page = ttk.Frame(
                self.page_host,
                style="Page.TFrame",
            )

            ttk.Label(
                comparison_page,
                text="Model Comparison",
                font=("Segoe UI", 20, "bold"),
                background="#ffffff",
            ).pack(
                pady=(80, 20),
            )

            ttk.Label(
                comparison_page,
                text=(
                    "Historical model-comparison artifacts are not available "
                    "on this Raspberry Pi.\n\n"
                    "This does not affect Edge AI inference or deployment."
                ),
                justify="center",
                background="#ffffff",
                font=("Segoe UI", 11),
            ).pack()

            self._pages["comparison"] = comparison_page

        # ============================================================
        # CONFUSION MATRIX
        # ============================================================
        confusion_records = self._safe_load_confusion_records()

        if confusion_records:
            self._pages["confusion"] = ConfusionMatrixPage(
                self.page_host,
                confusion_records,
                self.show_page,
            )

        else:
            confusion_page = ttk.Frame(
                self.page_host,
                style="Page.TFrame",
            )

            ttk.Label(
                confusion_page,
                text="Confusion Matrix",
                font=("Segoe UI", 20, "bold"),
                background="#ffffff",
            ).pack(
                pady=(80, 20),
            )

            ttk.Label(
                confusion_page,
                text=(
                    "Historical confusion-matrix artifacts are not available "
                    "on this Raspberry Pi.\n\n"
                    "The validated deployment model remains available."
                ),
                justify="center",
                background="#ffffff",
                font=("Segoe UI", 11),
            ).pack()

            self._pages["confusion"] = confusion_page

        # ============================================================
        # DETECTION HISTORY
        # ============================================================
        self._pages["history"] = DetectionHistoryPage(
            self.page_host,
            self.history_service,
        )

        # ============================================================
        # PLACEHOLDER PAGES
        # ============================================================
        placeholders = {
            "system": (
                "System / Deployment",
                (
                    "Raspberry Pi, camera, mmWave, LED, buzzer, and hardware "
                    "integration status can be validated here during deployment."
                ),
            ),
            "about": (
                "About Project",
                (
                    "Anvīkṣa AI is an offline Edge AI wildlife perception, "
                    "classification, and alert framework."
                ),
            ),
        }

        for key, (title, description) in placeholders.items():
            self._pages[key] = PlaceholderPage(
                self.page_host,
                title,
                description,
            )

        # ============================================================
        # PLACE ALL PAGES IN SAME CONTAINER
        # ============================================================
        for page in self._pages.values():
            page.place(
                relx=0,
                rely=0,
                relwidth=1,
                relheight=1,
            )

    def _safe_load_comparison_records(self):
        try:
            from ui.services.comparison_data import (
                load_comparison_records,
            )

            records = load_comparison_records()

            if records is None:
                return []

            return records

        except Exception as exc:
            print(
                "[WARNING] Model comparison data unavailable:",
                exc,
            )

            return []

    def _safe_load_confusion_records(self):
        try:
            from ui.services.confusion_data import (
                load_confusion_records,
            )

            records = load_confusion_records()

            if records is None:
                return []

            return records

        except Exception as exc:
            print(
                "[WARNING] Confusion matrix data unavailable:",
                exc,
            )

            return []

    def show_page(self, key: str) -> None:
        if key not in self._pages:
            raise KeyError(
                f"Unknown page: {key}"
            )

        if self._current_page == key:
            return

        previous = self._pages.get(
            self._current_page
        )

        if previous is not None and hasattr(
            previous,
            "on_hide",
        ):
            try:
                previous.on_hide()
            except Exception:
                pass

        self._current_page = key

        page = self._pages[key]
        page.tkraise()

        if hasattr(
            page,
            "on_show",
        ):
            try:
                page.on_show()
            except Exception:
                pass

        for name, button in self._nav_buttons.items():
            if name == key:
                button.configure(
                    style="Active.Nav.TButton",
                )
            else:
                button.configure(
                    style="Nav.TButton",
                )

    def close(self) -> None:
        for page in self._pages.values():
            if hasattr(
                page,
                "close",
            ):
                try:
                    page.close()
                except Exception:
                    pass

        self.destroy()


def main() -> None:
    app = AnviksaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
