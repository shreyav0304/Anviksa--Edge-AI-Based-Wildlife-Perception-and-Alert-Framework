"""Observe real hardware or run a labeled, bounded simulation integration check."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.services.live_hardware import CONFIG_PATH, load_config
from ui.services.live_tracking import LiveTrackingService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--simulate", action="store_true", help="No real hardware accessed")
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.simulate:
        cfg["simulation"] = True
    service = LiveTrackingService(cfg)
    signal.signal(signal.SIGINT, lambda *_: service.stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: service.stop_event.set())
    seen = {"mic": set(), "radar": set(), "led1": set(), "led2": set(),
            "led3": set(), "buzzer": set(), "button": set()}
    frames = 0
    ids = set()
    moved = set()
    last_positions = {}
    last_log = 0
    start = time.monotonic()
    print("SIMULATION ONLY - NO HARDWARE VERIFIED" if cfg["simulation"] else
          "PHYSICAL OBSERVATION - clap, enter/leave radar range, press/release button")
    try:
        service.start()
        while time.monotonic() - start < args.seconds and not service.stop_event.is_set():
            elapsed = time.monotonic() - start
            if cfg["simulation"]:
                service.hardware.sim_button = 1 < elapsed < 2
            data = service.snapshot()
            sensors = data["sensors"]
            for key in ("mic", "radar"):
                item = sensors.get(key, {})
                if item.get("live"):
                    seen[key].add(int(item["active"]))
            gpio = sensors.get("gpio", {})
            if gpio.get("health") == "connected":
                for key in ("led1", "led2", "led3", "buzzer", "button"):
                    seen[key].add(int(gpio[key]))
            if data["frame"] is not None:
                frames += 1
            for track in data["tracks"]:
                key = track["id"]
                ids.add(key)
                if key in last_positions and last_positions[key] != track["center"]:
                    moved.add(key)
                last_positions[key] = track["center"]
            if elapsed - last_log >= 1:
                print(json.dumps({"mode": "SIMULATED" if cfg["simulation"] else "PHYSICAL",
                    "camera": data["camera_health"], "tracked_regions": len(data["tracks"]),
                    "sensors": sensors, "model": data["model_health"]}), flush=True)
                last_log = elapsed
            time.sleep(0.03)
    finally:
        service.stop()
    report = {"simulation": cfg["simulation"], "frame_observations": frames,
              "observed_transitions": {key: sorted(value) for key, value in seen.items()},
              "track_ids": sorted(ids), "moving_ids": sorted(moved),
              "workers_stopped": not service.running,
              "gpio_after_stop": service.hardware.snapshot().get("gpio"),
              "physical_led_light_and_beep_require_human_confirmation": True}
    print(json.dumps(report, indent=2))
    if cfg["simulation"]:
        return 0 if (frames > 0 and len(moved) >= 2 and not service.running
                     and all(value == {0, 1} for value in seen.values())) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
