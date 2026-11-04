"""Protocol, stale-data, safe alert, tracking and reconnect-isolation checks."""
import copy
import time
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

from ui.services.live_hardware import (ButtonPulse, C4001Parser, HardwareService,
                                      alert_outputs, first_channel_rms, load_config)
from ui.services.live_tracking import CentroidTracker, LiveTrackingService


class ProtocolTests(unittest.TestCase):
    def test_partial_concatenated_reserved_spaces_and_recovery(self):
        parser = C4001Parser()
        self.assertEqual(parser.feed(b"garbage$DFHP", 1), [])
        self.assertEqual(parser.feed(b"D,1, , , *$DFHPD,0,,,*", 1.1),
                         [{"presence": True, "distance": None, "speed": None},
                          {"presence": False, "distance": None, "speed": None}])
        self.assertEqual(parser.feed(b"$DFHPD,2,,,*$DFHPD,1,,*", 2), [])
        self.assertEqual(parser.feed(b"$broken$DFHPD,1,,,*", 2.1)[0]["presence"], True)
        self.assertGreater(parser.invalid, 0)

    def test_timeout_noise_and_oversize_do_not_make_live_data(self):
        parser = C4001Parser()
        parser.feed(b"$DFHPD,", 1)
        self.assertEqual(parser.feed(b"1,,,*", 3), [])
        parser.feed(b"$" + b"x"*1000, 4)
        self.assertLessEqual(len(parser.buffer), 256)
        self.assertEqual(parser.feed(b"$DFHPD,1,x,,*", 5), [])
        self.assertEqual(parser.feed(b"$DFHPD,0,,,*", 6)[0]["presence"], False)

    def test_distance_requires_opt_in_and_valid_one_target_frame(self):
        frame = b"$DFDMD,1,1,1.817,0.129,15304, , *"
        self.assertEqual(C4001Parser().feed(frame), [])
        parser = C4001Parser(True)
        result = parser.feed(frame)[0]
        self.assertEqual(result["distance"], 1.817)
        self.assertEqual(result["speed"], 0.129)
        for bad in (b"$DFDMD,2,1,1,1,1,,*", b"$DFDMD,1,1,nan,0,1,,*",
                    b"$DFDMD,1,1,26,0,1,,*", b"$DFDMD,0,1,1,1,1,,*"):
            self.assertEqual(parser.feed(bad), [])
        self.assertIsNone(parser.feed(b"$DFDMD,0, , , , , , *")[0]["distance"])


class HardwareTests(unittest.TestCase):
    def test_radar_reconnects_after_read_error(self):
        service = HardwareService(load_config())
        attempts = []
        class Port:
            def __init__(self, number):
                self.number = number
                self.closed = False
            def read(self, _size):
                if self.number == 1:
                    raise OSError("Disconnected")
                service.stop_event.set()
                return b"$DFHPD,1,,,*"
            def close(self):
                self.closed = True
        def connect(*_args, **_kwargs):
            port = Port(len(attempts) + 1)
            attempts.append(port)
            return port
        with patch.dict("sys.modules", {"serial": types.SimpleNamespace(Serial=connect)}):
            worker = threading.Thread(target=service._radar)
            worker.start()
            worker.join(timeout=5)
            if worker.is_alive():
                service.stop_event.set()
                worker.join(timeout=1)
                self.fail("Radar reconnect did not finish")
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(port.closed for port in attempts))
        self.assertTrue(service.snapshot()["radar"]["active"])

    def test_first_channel_and_threshold(self):
        samples = np.zeros((4800, 2), dtype="<i4")
        samples[:, 1] = 2**30
        self.assertEqual(first_channel_rms(samples.tobytes()), 0)
        samples[:, 0] = int(0.04 * 2**31)
        self.assertAlmostEqual(first_channel_rms(samples.tobytes()), 0.04, places=7)

    def test_button_debounce_held_duration_and_repress(self):
        pulse = ButtonPulse()
        self.assertEqual(pulse.update(True, 1, .04, .5), (False, False))
        self.assertEqual(pulse.update(False, 1.01, .04, .5), (False, False))
        pulse.update(True, 1.02, .04, .5)
        self.assertEqual(pulse.update(True, 1.07, .04, .5), (True, True))
        self.assertEqual(pulse.update(True, 1.56, .04, .5), (True, True))
        self.assertEqual(pulse.update(True, 1.58, .04, .5), (True, False))
        pulse.update(False, 2, .04, .5)
        self.assertEqual(pulse.update(False, 2.05, .04, .5), (False, False))
        pulse.update(True, 3, .04, .5)
        self.assertEqual(pulse.update(True, 3.05, .04, .5), (True, True))

    def test_alert_truth_table_and_no_sensor_buzzer(self):
        rules = load_config()["alerts"]
        for mic in (False, True):
            for radar in (False, True):
                result = alert_outputs(mic, radar, False, False, rules)
                self.assertEqual(result["led1"], mic or radar)
                self.assertEqual(result["led2"], mic and radar)
                self.assertFalse(result["buzzer"])
        result = alert_outputs(False, False, True, True, rules)
        self.assertTrue(result["led3"])
        self.assertTrue(result["buzzer"])

    def test_stale_data_invalidates_without_breaking_other_sensor(self):
        service = HardwareService(load_config())
        service.state.set("mic", health="connected", active=True, rms=.1)
        service.state.set("radar", health="connected", active=True, distance=2, speed=1)
        with service.state.lock:
            service.state.values["radar"]["updated"] = 0
        data = service.snapshot()
        self.assertTrue(data["mic"]["live"])
        self.assertIsNone(data["radar"]["active"])
        self.assertIsNone(data["radar"]["distance"])
        service.state.set("radar", health="connected", active=False, distance=None)
        self.assertTrue(service.snapshot()["radar"]["live"])

    def test_audio_enumeration_uses_unique_name_not_fixed_index(self):
        service = HardwareService(load_config())
        with patch("ui.services.live_hardware.subprocess.run") as run:
            run.return_value.stdout = "card 4: i2s [INMP441], device 0: capture"
            self.assertEqual(service._audio_device()[0], "plughw:4,0")
            run.return_value.stdout = "card 3: Camera [USB Camera], device 0: capture"
            device, note = service._audio_device()
            self.assertEqual(device, "plughw:3,0")
            self.assertIn("verify", note)


class TrackingTests(unittest.TestCase):
    def test_two_tracks_stable_under_small_motion_and_expire(self):
        tracker = CentroidTracker(load_config()["tracking"])
        first = tracker.update([(10, 10, 20, 20), (200, 100, 20, 20)], (480, 640), 1)
        second = tracker.update([(15, 10, 20, 20), (205, 100, 20, 20)], (480, 640), 1.1)
        self.assertEqual([t["id"] for t in first], [t["id"] for t in second])
        self.assertNotEqual(first[0]["center"], second[0]["center"])
        self.assertTrue(all(0 <= v <= 1 for t in second for v in t["center"]))
        self.assertEqual(tracker.update([], (480, 640), 3), [])

    def test_camera_stale_clears_map_and_risk_requires_operator(self):
        service = LiveTrackingService(load_config())
        now = time.monotonic()
        track = {"id": 1, "center": (.5, .5), "box": (0, 0, 20, 20),
                 "hits": 5, "trail": [], "seen": now, "observed": True}
        service._publish(np.zeros((40, 40, 3), np.uint8), [track])
        service.results[1] = {"name": "Venomous_Snake", "confidence": .99,
                             "stable": True, "updated": now}
        self.assertFalse(service.snapshot()["tracks"][0]["confirmed_risk"])
        self.assertTrue(service.confirm_risk(1))
        self.assertTrue(service.snapshot()["tracks"][0]["confirmed_risk"])
        service.frame_time = 0
        self.assertEqual(service.snapshot()["tracks"], [])
        self.assertFalse(service.confirm_risk(1))

    def test_radar_disconnect_does_not_stop_camera(self):
        cfg = copy.deepcopy(load_config())
        cfg["simulation"] = True
        service = LiveTrackingService(cfg)
        try:
            service.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and service.snapshot()["frame"] is None:
                time.sleep(.05)
            self.assertIsNotNone(service.snapshot()["frame"])
            service.hardware.state.set("radar", health="disconnected", active=None, error="unplugged")
            # Immediate snapshot while camera thread remains independent.
            self.assertIsNotNone(service.snapshot()["frame"])
            self.assertTrue(service.running)
        finally:
            service.stop()
        self.assertFalse(service.running)
        self.assertFalse(service.hardware.snapshot()["gpio"]["buzzer"])


if __name__ == "__main__":
    unittest.main()
