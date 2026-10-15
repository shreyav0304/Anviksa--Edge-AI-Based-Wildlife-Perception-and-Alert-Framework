"""Regression checks for published data, inference contracts, capture and history."""
from __future__ import annotations
import csv
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from pi_deployment.inference import tflite_inference as backend
from ui.data import load_dashboard_data
from ui.services.comparison_data import load_comparison_records
from ui.services.confusion_data import load_confusion_records
from ui.services.detection_history import DetectionHistoryService, EVENT_FIELDS
from ui.services.video_inference import TemporalAnalyzer, GateTemporalState, VideoInferenceWorker
from ui.services.wildlife_gate import WildlifeGateService, WildlifeGateResult, raw_gate_decision

class PublishedDataTests(unittest.TestCase):
    def test_published_model_results_agree_with_matrices(self):
        records = load_comparison_records()
        matrices = load_confusion_records()
        self.assertEqual(len(records), 4)
        self.assertEqual(len(matrices), 4)
        for record in records:
            if record.status == "COMPLETED":
                matrix = matrices[record.model]
                self.assertEqual(matrix.total, 545)
                self.assertAlmostEqual(matrix.matrix_accuracy, record.accuracy, places=9)
                np.testing.assert_allclose(matrix.normalized.sum(axis=1), 1)
            else:
                self.assertEqual(record.model, "EfficientNetB0")
                self.assertIsNone(record.accuracy)
                self.assertNotIn(record.model, matrices)
        self.assertEqual(load_dashboard_data().selected_model, "MobileNetV3 Large (15 classes)")

class ActiveEvaluationTests(unittest.TestCase):
    def test_current_export_has_fifteen_classes_and_recomputed_full_test_results(self):
        from ui.services.evaluation import load_active_evaluation
        from ui.services.confusion_data import ACTIVE_MODEL
        active = load_active_evaluation()
        matrix = load_confusion_records()[ACTIVE_MODEL]
        self.assertEqual(active["classes"], backend.load_class_names())
        self.assertEqual(len(active["classes"]), 15)
        self.assertEqual(active["total"], 670)
        self.assertEqual(active["correct"], 620)
        self.assertEqual(matrix.raw.shape, (15, 15))
        np.testing.assert_array_equal(matrix.raw, active["raw"])
        self.assertAlmostEqual(load_dashboard_data().accuracy, 620/670)
        self.assertEqual(sum(c["support"] for c in active["per_class"]), 670)
        self.assertLess(active["accuracy_interval"][0], active["accuracy"])
        self.assertGreater(active["accuracy_interval"][1], active["accuracy"])

    def test_metrics_use_ground_truth_and_keep_orientation(self):
        from ui.services.evaluation import prediction_metrics
        probabilities = np.array([[.8,.1,.1],[.2,.7,.1],[.1,.8,.1],[.1,.6,.3]])
        measured = prediction_metrics(np.array([0,0,1,2]), probabilities, ("A","B","C"))
        np.testing.assert_array_equal(measured["raw"], [[1,1,0],[0,1,0],[0,1,0]])
        self.assertEqual(measured["accuracy"], .5)
        self.assertEqual(measured["per_class"][0]["recall"], .5)
        self.assertAlmostEqual(measured["per_class"][1]["precision"], 1/3)
        self.assertAlmostEqual(measured["macro_f1"], (2/3+.5)/3)
        with self.assertRaises(ValueError):
            prediction_metrics(np.array([0,0,1,2]), probabilities[:, :2], ("A","B","C"))
        with self.assertRaises(ValueError):
            prediction_metrics(np.array([0,0,1,5]), probabilities, ("A","B","C"))

    def test_scores_do_not_round_near_one_into_perfect_certainty(self):
        from ui.services.evaluation import format_model_score
        self.assertEqual(format_model_score(.9999572038650513), "99.99572%")
        self.assertIn("saturated", format_model_score(1))
        self.assertNotIn("100", format_model_score(.99999999999))
        for invalid in (float("nan"), float("inf"), -1, 1.01):
            with self.assertRaises(ValueError): format_model_score(invalid)

    def test_bilinear_preprocessing_matches_frozen_opencv_pipeline(self):
        import cv2
        pixels = np.arange(11*13*3, dtype=np.uint8).reshape(11,13,3)
        expected = cv2.resize(pixels, (224,224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        np.testing.assert_array_equal(backend.preprocess_image(pixels)[0], expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.png"
            path.write_bytes(cv2.imencode(".png", cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))[1].tobytes())
            np.testing.assert_array_equal(backend.preprocess_image(path)[0], expected)

class InferenceContractTests(unittest.TestCase):
    def test_pixel_preprocessing_retains_model_contract(self):
        pixels = np.full((16, 32, 3), 127, dtype=np.uint8)
        batch = backend.preprocess_image(pixels)
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(batch.flags.c_contiguous)
        self.assertTrue(np.all(batch == 127))
        np.testing.assert_array_equal(batch, backend.preprocess_image(Image.fromarray(pixels)))

    def test_invalid_pixels_fail_without_silent_clipping(self):
        for pixels in (np.empty((0, 3, 3)), np.full((2, 2, 3), np.nan),
                       np.full((2, 2, 3), 256), np.full((2, 2, 3), -1),
                       np.full((2, 2, 3), 1j)):
            with self.subTest(shape=pixels.shape), self.assertRaises(ValueError):
                backend.preprocess_image(pixels)

    def test_reordered_class_labels_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.json"
            path.write_text(json.dumps(list(reversed(backend.EXPECTED_CLASSES))))
            with self.assertRaises(ValueError):
                backend.load_class_names(path)

    def test_invalid_model_probabilities_fail(self):
        classifier = backend.TFLiteWildlifeClassifier.__new__(backend.TFLiteWildlifeClassifier)
        classifier.input_detail = {"index": 0}
        classifier.output_detail = {"index": 1}
        classifier.interpreter = Mock()
        classifier.interpreter.get_tensor.return_value = np.array([[-.1, 1.1, 0, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            classifier.invoke_preprocessed(np.zeros((1,224,224,3), dtype=np.float32))

    def test_corrupt_gate_model_fails_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.tflite"
            path.write_bytes(b"corrupt")
            with patch("ui.services.wildlife_gate._load_interpreter_class") as loader:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    WildlifeGateService(path)
                loader.assert_not_called()

class TemporalTests(unittest.TestCase):
    def test_late_high_confidence_is_counted_once(self):
        analyzer = TemporalAnalyzer(("A", "B"), window=1, stable_frames=2)
        analyzer.update(np.array([.7, .3]), 1)
        analyzer.update(np.array([.7, .3]), 1)
        self.assertEqual(analyzer.stable_events, 0)
        result = analyzer.update(np.array([.9, .1]), 1)
        self.assertTrue(result.stable)
        self.assertEqual(analyzer.stable_events, 1)
        analyzer.update(np.array([.95, .05]), 1)
        self.assertEqual(analyzer.stable_events, 1)

    def test_gate_timeout_resets_species_evidence(self):
        analyzer = TemporalAnalyzer(("A", "B"), window=1, stable_frames=1)
        analyzer.update(np.array([.9, .1]), 1)
        gate = GateTemporalState()
        self.assertEqual(gate.observe("UNCERTAIN", analyzer), (False, False))
        self.assertEqual(gate.observe("UNCERTAIN", analyzer), (False, False))
        self.assertEqual(gate.observe("UNCERTAIN", analyzer), (True, True))
        self.assertIsNone(analyzer.last_class)
        self.assertEqual(len(analyzer.history), 0)

    def test_bad_probability_vectors_are_rejected(self):
        analyzer = TemporalAnalyzer(("A", "B"))
        for vector in ([-.1, 1.1], [.2, .2], [np.nan, .5]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                analyzer.update(np.asarray(vector), 1)
        self.assertEqual(len(analyzer.history), 0)

    def test_gate_threshold_boundaries(self):
        self.assertEqual(raw_gate_decision(.19), "NON_WILDLIFE")
        self.assertEqual(raw_gate_decision(.2), "UNCERTAIN")
        self.assertEqual(raw_gate_decision(.65), "WILDLIFE")
        with self.assertRaises(ValueError):
            raw_gate_decision(float("nan"))

class CaptureTests(unittest.TestCase):
    def test_bad_frame_skip_rejected_before_thread_start(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                VideoInferenceWorker(0, frame_skip=value)

    def test_capture_constructor_failure_is_reported(self):
        worker = VideoInferenceWorker(0)
        cv = Mock()
        cv.VideoCapture.side_effect = RuntimeError("device failure")
        with patch("ui.services.video_inference.require_opencv", return_value=cv):
            worker._run()
        result = worker.updates.get_nowait()
        self.assertEqual(result.kind, "error")
        self.assertIn("device failure", result.message)

    def test_stop_never_releases_capture_from_caller_thread(self):
        worker = VideoInferenceWorker(0)
        worker._capture = Mock()
        worker.stop()
        worker._capture.release.assert_not_called()
        self.assertTrue(worker.stop_event.is_set())

    def test_eligible_event_survives_replaced_preview_frames(self):
        worker = VideoInferenceWorker(Path("fake.mp4"), stable_frames=1)
        capture = Mock()
        frames = [np.full((2,2,3), value, dtype=np.uint8) for value in (10,20,30,40,50)]
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        capture.get.return_value = 0
        cv = Mock()
        cv.VideoCapture.return_value = capture
        cv.cvtColor.side_effect = lambda frame, _code: frame
        classifier = Mock(class_names=backend.EXPECTED_CLASSES)
        classifier.predict.return_value = {"probabilities": [.9, .1] + [0] * (len(backend.EXPECTED_CLASSES)-2), "inference_time_ms": 1}
        gate = Mock()
        gate.predict.return_value = WildlifeGateResult(.9, .1, "WILDLIFE", 1)
        with patch("ui.services.video_inference.require_opencv", return_value=cv), \
             patch("ui.services.video_inference.ImageInferenceService") as verifier, \
             patch("ui.services.video_inference.TFLiteWildlifeClassifier", return_value=classifier), \
             patch("ui.services.video_inference.WildlifeGateService", return_value=gate):
            worker._run()
        gate.predict.assert_not_called()
        self.assertEqual(worker.event_updates.qsize(), 1)
        event = worker.event_updates.get_nowait()
        self.assertEqual(event.frames_captured, 1)
        self.assertTrue(np.all(event.frame_rgb == 10))
        updates = [worker.updates.get_nowait() for _ in range(worker.updates.qsize())]
        self.assertEqual(updates[-1].kind, "complete")
        self.assertFalse(any(update.frames_captured == 1 for update in updates))
        capture.release.assert_called_once()

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = DetectionHistoryService(Path(self.directory.name))
        self.values = dict(predicted_class="Cow", display_class="Cow", category="Domestic Animal",
                           confidence=.9, top2_class="Deer", top2_probability=.1, margin=.8,
                           confidence_status="HIGH", uncertain=False, stable=True, stable_count=3,
                           inference_time_ms=1)

    def test_stable_events_cooldown_and_distinct_sightings(self):
        image = Image.new("RGB", (4,4))
        first = self.service.log_stable_frame(image, source_type="WEBCAM", source_name="Camera 0",
                                              now_monotonic=100, episode_key="first", **self.values)
        self.assertIsNotNone(first)
        self.assertIsNone(self.service.log_stable_frame(image, source_type="WEBCAM", source_name="Camera 0",
                                                        now_monotonic=101, episode_key="first", **self.values))
        self.assertIsNotNone(self.service.log_stable_frame(image, source_type="WEBCAM", source_name="Camera 0",
                                                           now_monotonic=101, episode_key="second", **self.values))
        self.assertEqual(len(self.service.load_events()), 2)
        self.assertIsNotNone(self.service.resolve_snapshot(first.snapshot_path))

    def test_truncated_nan_and_bad_boolean_history_rows_are_skipped(self):
        event = self.service.create_event(source_type="VIDEO", source_name="test", snapshot_path="",
                                         notes="", **self.values)
        self.service._append(event)
        with self.service.csv_path.open("a", newline="", encoding="utf-8") as handle:
            handle.write("truncated,row\n")
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
            from dataclasses import asdict
            invalid = asdict(event)
            invalid["confidence"] = "nan"
            writer.writerow(invalid)
            invalid["confidence"] = .9
            invalid["stable"] = "maybe"
            writer.writerow(invalid)
        self.assertEqual(len(self.service.load_events()), 1)
        self.assertEqual(self.service.malformed_rows, 3)

    def test_snapshot_resolution_rejects_external_paths(self):
        outside = Path(self.directory.name) / "external.jpg"
        outside.write_bytes(b"file")
        self.assertIsNone(self.service.resolve_snapshot("external.jpg"))
        self.assertIsNone(self.service.resolve_snapshot("../external.jpg"))

    def test_old_history_without_video_fields_can_be_read(self):
        event = self.service.create_event(source_type="VIDEO", source_name="test", snapshot_path="",
                                         notes="", **self.values)
        from dataclasses import asdict
        columns = EVENT_FIELDS[:-2]
        with self.service.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(asdict(event))
        loaded = self.service.load_events()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].frame_number, "")
        self.service._append(event)
        self.assertEqual(len(self.service.load_events()), 2)

class DiagnosticTests(unittest.TestCase):
    def test_missing_benchmark_image_folder_has_actionable_error(self):
        from pi_deployment.benchmark import benchmark_tflite
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark_tflite, "PACKAGE_ROOT", Path(directory)):
            with self.assertRaisesRegex(FileNotFoundError, "--image PATH"):
                benchmark_tflite.choose_image(None)

    def test_benchmark_comparison_rejects_different_inputs_or_protocols(self):
        from copy import deepcopy
        from pi_deployment.benchmark.compare_pi_results import validate_compatible_runs
        reference = {
            "benchmark_image": "image.jpg", "benchmark_image_sha256": "a"*64, "batch_size": 1,
            "system": dict(raspberry_pi_model="Pi 4", os="Linux", architecture="aarch64",
                           python_version="3.11", interpreter_source="tflite_runtime"),
            "model_only": dict(warmup_iterations=20, timed_iterations=200),
            "end_to_end": dict(timed_iterations=50),
        }
        other = deepcopy(reference)
        selected = {("float32",1): (Path("first"), reference), ("float16",1): (Path("second"), other)}
        validate_compatible_runs(selected)
        other["benchmark_image_sha256"] = "b"*64
        with self.assertRaisesRegex(ValueError, "images differ"):
            validate_compatible_runs(selected)
        other["benchmark_image_sha256"] = "a"*64
        other["end_to_end"]["timed_iterations"] = 10
        with self.assertRaisesRegex(ValueError, "settings differ"):
            validate_compatible_runs(selected)

    def test_corrupt_package_json_produces_failed_report(self):
        import contextlib, io
        from pi_deployment import verify_package
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"inference").mkdir()
            (root/"inference/class_names.json").write_text("{bad")
            (root/"model_manifest.json").write_text("[]")
            output = io.StringIO()
            with patch.object(verify_package, "ROOT", root), contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit):
                    verify_package.main()
            report = json.loads(output.getvalue())
            self.assertFalse(report["pass"])
            self.assertTrue(any("readable JSON" in item["check"] for item in report["checks"]))

    def test_nonfinite_comparison_values_are_rejected(self):
        from ui.services.comparison_data import _optional_float
        for value in ("nan", "inf", "-0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _optional_float(value)

if __name__ == "__main__":
    unittest.main()
