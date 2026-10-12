"""Opt-in desktop checks: ANVIKSA_GUI_TESTS=1 python -m unittest discover -s tests."""
from __future__ import annotations
import os
import unittest
from unittest.mock import patch
from concurrent.futures import Future
from pathlib import Path

@unittest.skipUnless(os.environ.get("ANVIKSA_GUI_TESTS") == "1", "requires an active Tk desktop")
class DashboardTests(unittest.TestCase):
    def setUp(self):
        from ui.app import AnviksaApp
        self.app = AnviksaApp()
        self.app.geometry("800x480+10000+10000")
        self.app.update()
        self.addCleanup(self.app.close)

    def test_every_page_and_matrix_column_at_pi_resolutions(self):
        self.assertEqual(len(self.app._nav_buttons), 8)
        for size in ("800x480", "480x320"):
            self.app.geometry(size + "+10000+10000")
            self.app.update()
            for key in self.app._nav_buttons:
                self.app.show_page(key)
                self.app.update()
                self.assertNotIn(key, self.app._failed_pages)
                self.assertTrue(self.app._pages[key].winfo_ismapped())
        page = self.app._pages["confusion"]
        self.assertEqual(len(page.all_figure.axes), 4)
        page.view_var.set("normalized")
        page.refresh()
        self.app.update()
        self.assertEqual(len(page.all_figure.axes), 4)

    def test_missing_page_dependency_does_not_block_other_sections(self):
        original = self.app._make_page
        def unavailable(key):
            if key == "comparison":
                raise ImportError("plotting dependency unavailable")
            return original(key)
        with self.assertLogs("ui.app", level="ERROR"), patch.object(self.app, "_make_page", side_effect=unavailable):
            self.app.show_page("comparison")
        self.assertIn("comparison", self.app._failed_pages)
        self.app.show_page("system")
        self.app.update()
        self.assertNotIn("system", self.app._failed_pages)
        self.app.show_page("comparison", retry=True)
        self.app.update()
        self.assertNotIn("comparison", self.app._failed_pages)

    def test_failed_reanalysis_clears_previous_saveable_prediction(self):
        self.app.show_page("image")
        page = self.app._pages["image"]
        page.selected_path = Path("test.jpg")
        page._last_result = object()
        failed = Future()
        failed.set_exception(RuntimeError("inference failed"))
        with patch("ui.pages.image_analysis.messagebox.showerror"):
            page._finish_analysis(failed)
        self.assertIsNone(page._last_result)
        self.assertEqual(str(page.save_button.cget("state")), "disabled")

    def test_repeated_page_retry_keeps_one_wheel_binding(self):
        self.app.show_page("about")
        binding = self.app.bind("<Button-4>")
        for _ in range(3):
            self.app.show_page("about", retry=True)
            self.app.update()
        self.assertEqual(self.app.bind("<Button-4>"), binding)

    def test_export_cannot_overwrite_master_history(self):
        self.app.show_page("history")
        page = self.app._pages["history"]
        page.filtered_events = [object()]
        with patch("ui.pages.detection_history.filedialog.asksaveasfilename", return_value=str(page.service.csv_path)), \
             patch("ui.pages.detection_history.messagebox.showerror") as dialog, \
             patch.object(page.service, "export_csv") as export:
            page.export_filtered()
        dialog.assert_called_once()
        export.assert_not_called()

if __name__ == "__main__":
    unittest.main()
