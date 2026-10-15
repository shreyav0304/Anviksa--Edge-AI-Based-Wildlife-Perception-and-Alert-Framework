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
        self.assertEqual(len(page.axis.images), 1)
        page.view_var.set("normalized")
        page.refresh()
        self.app.update()
        self.assertEqual(len(page.axis.images), 1)

    def test_section_picker_reaches_all_pages_and_sidebar_recovers(self):
        from ui.app import NAV_ITEMS
        for size in ("480x320", "800x480", "1180x720", "800x480"):
            self.app.geometry(size + "+10000+10000")
            self.app.update()
            self.assertEqual(bool(self.app.sidebar.winfo_manager()), int(size.split("x")[0]) >= 900)
            picker = self.app.section_picker
            self.assertTrue(picker.winfo_ismapped())
            self.assertGreater(picker.winfo_width(), 150)
            self.assertEqual(tuple(picker.cget("values")), tuple(label for _, label in NAV_ITEMS))
            for key, label in NAV_ITEMS:
                self.app.section_var.set(label)
                picker.event_generate("<<ComboboxSelected>>")
                self.app.update()
                self.assertEqual(self.app._current_page, key)
                self.assertNotIn(key, self.app._failed_pages)
                page = self.app._pages[key]
                self.assertGreater(page.canvas.winfo_height(), 100)
                self.assertGreater(page.canvas.winfo_width(), 200)
        self.app.geometry("1180x480+10000+10000")
        self.app.update()
        self.app.show_page("about")
        self.app.update()
        button = self.app._nav_buttons["about"]
        top = self.app.nav_canvas.canvasy(0)
        self.assertGreaterEqual(button.winfo_y(), top - 1)
        self.assertLessEqual(button.winfo_y() + button.winfo_height(), top + self.app.nav_canvas.winfo_height() + 1)

    def test_wide_pages_can_scroll_horizontally_with_shift_wheel(self):
        from types import SimpleNamespace
        self.app.show_page("comparison")
        self.app.update()
        page = self.app._pages["comparison"]
        self.assertTrue(page.scroll_hint.winfo_ismapped())
        before = page.canvas.xview()[0]
        event = SimpleNamespace(x_root=page.canvas.winfo_rootx() + 20,
                                y_root=page.canvas.winfo_rooty() + 20,
                                num=5, delta=0, state=1)
        page._on_mousewheel(event)
        self.assertGreater(page.canvas.xview()[0], before)

    def test_current_classes_and_confusion_matrix_default_are_consistent(self):
        from ui.services.confusion_data import ACTIVE_MODEL
        self.app.show_page("image")
        self.assertEqual(len(self.app._pages["image"].probability_widgets), 15)
        self.app.show_page("history")
        self.assertIn("Big_Cat", self.app._pages["history"].class_box.cget("values"))
        self.app.show_page("confusion")
        page = self.app._pages["confusion"]
        self.assertEqual(page.model_var.get(), ACTIVE_MODEL)
        self.assertEqual(len(page.axis.get_xticklabels()), 15)
        self.assertEqual(page.summary_vars["total"].get(), "670")
        self.assertEqual(page.summary_vars["correct"].get(), "620")
        self.assertEqual(page.summary_vars["matrix_accuracy"].get(), "92.54%")

    def test_single_matrix_fits_small_screens_and_cells_match_counts(self):
        from types import SimpleNamespace
        self.app.show_page("confusion")
        page=self.app._pages["confusion"]
        for size in ("480x320","800x480","1180x720"):
            self.app.geometry(size+"+10000+10000")
            self.app.update()
            page._fit_layout()
            self.app.update()
            self.assertLessEqual(page.chart.get_tk_widget().winfo_width(),page.canvas.winfo_width())
            self.assertLessEqual(page.chart.get_tk_widget().winfo_height(),page.canvas.winfo_height())
            self.assertFalse(page.scroll_hint.winfo_ismapped())
            self.assertFalse(hasattr(page,"all_chart"))
            self.assertEqual(len(page.class_table.get_children()),15)
            self.assertEqual(len(page.axis.images),1)
        record=page.records[page.model_var.get()]
        self.assertEqual(len(page.axis.texts),int((record.raw>0).sum()))
        page._hover_cell(SimpleNamespace(inaxes=page.axis,xdata=13,ydata=10))
        self.assertIn("16/128",page.cell_var.get())
        page.class_table.selection_set("0")
        page._class_selected()
        self.assertIn("None on this test split",page.cell_var.get())
        page.close()

    def test_three_historical_models_have_separate_matching_matrices(self):
        import numpy as np
        from ui.pages.confusion_matrix import HISTORICAL_MODELS
        from ui.services.confusion_data import ACTIVE_MODEL
        self.app.show_page("confusion")
        page=self.app._pages["confusion"]
        page.show_historical_models()
        self.app.update()
        page._fit_layout()
        self.app.update()
        self.assertEqual(tuple(page.historical_charts),HISTORICAL_MODELS)
        self.assertFalse(page.matrix_card.winfo_ismapped())
        self.assertTrue(page.historical_host.winfo_ismapped())
        for name,entry in page.historical_charts.items():
            self.assertTrue(entry["chart"].get_tk_widget().winfo_ismapped())
            self.assertLessEqual(entry["chart"].get_tk_widget().winfo_width(),page.canvas.winfo_width())
            self.assertEqual(len(entry["axis"].images),1)
            np.testing.assert_array_equal(entry["axis"].images[0].get_array(),page.records[name].raw)
            self.assertIn(f"{page.records[name].correct}/545",entry["axis"].get_title())
        page.view_var.set("normalized")
        page.refresh();page._fit_layout();self.app.update()
        for name,entry in page.historical_charts.items():
            np.testing.assert_allclose(entry["axis"].images[0].get_array(),page.records[name].normalized)
        page.show_active_model();self.app.update()
        self.assertEqual(page.model_var.get(),ACTIVE_MODEL)
        self.assertTrue(page.matrix_card.winfo_ismapped())
        self.assertFalse(page.historical_host.winfo_ismapped())

    def test_matrix_defaults_to_active_model_independent_of_record_order(self):
        from ui.services.confusion_data import load_confusion_records, ACTIVE_MODEL
        from ui.pages.confusion_matrix import ConfusionMatrixPage
        records = dict(reversed(list(load_confusion_records().items())))
        page = ConfusionMatrixPage(self.app.page_host, records, self.app.show_page)
        self.addCleanup(page.destroy)
        self.assertEqual(page.model_var.get(), ACTIVE_MODEL)
        self.assertIn("ACTIVE DEPLOYMENT EVALUATION", page.selection_var.get())
        page.model_var.set("Custom CNN")
        page.refresh()
        self.assertIn("HISTORICAL COMPARISON ONLY", page.selection_var.get())
        page.on_show()
        self.assertEqual(page.model_var.get(), ACTIVE_MODEL)
        self.assertEqual(len(page.axis.get_xticklabels()), 15)
        self.assertIn("ACTIVE DEPLOYMENT EVALUATION", page.selection_var.get())
        self.app.update()

    def test_confidence_chart_keeps_near_one_precision(self):
        self.app.show_page("comparison")
        page = self.app._pages["comparison"]
        page.show_graph("confidence")
        self.app.update()
        labels = [text.get_text() for text in page.axis.texts]
        self.assertTrue(any("99.99572" in text for text in labels))
        self.assertFalse(any(text == "100.00%" for text in labels))

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
