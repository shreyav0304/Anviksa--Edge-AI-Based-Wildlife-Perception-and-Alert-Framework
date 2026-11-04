"""Concurrent Pi sensors. Importable without GPIO/serial on development machines."""
from __future__ import annotations

import copy
import json
import math
import re
import selectors
import subprocess
import threading
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/live_tracking.json"


def load_config(path=CONFIG_PATH):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    pins = list(cfg["gpio"].values())
    if (len(set(pins)) != 5 or any(type(p) is not int or not 2 <= p <= 27 for p in pins)
            or set(pins) & {14, 15, 18, 19, 20, 21}):
        raise ValueError("GPIOs must be unique BCM pins, excluding UART14/15 and I2S18-21")
    for section, field in (("microphone", "threshold"), ("camera", "fps"),
                           ("camera", "width"), ("camera", "height"),
                           ("microphone", "stale_seconds"), ("radar", "stale_seconds"),
                           ("radar", "partial_timeout"), ("camera", "stale_seconds"),
                           ("tracking", "expiry_seconds"), ("tracking", "max_objects"),
                           ("tracking", "max_match_distance"), ("tracking", "trail_length"),
                           ("classification", "interval_seconds"), ("classification", "stale_seconds"),
                           ("classification", "stable_results"), ("alerts", "debounce_seconds")):
        value = cfg[section][field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{field} must be finite and positive")
    if not 0 < cfg["alerts"]["beep_seconds"] <= 1:
        raise ValueError("Buzzer pulse must be greater than 0 and at most 1 second")
    if not 0 <= cfg["classification"]["score_threshold"] <= 1:
        raise ValueError("Invalid classification threshold")
    for name in ("led1", "led2", "led3"):
        if cfg["alerts"][name] not in {"activity_or", "activity_and", "button", "off"}:
            raise ValueError("Unsupported LED rule")
    if cfg["alerts"]["buzzer"] not in {"button", "off"}:
        raise ValueError("Buzzer supports button/off; classification alerts require explicit review")
    return cfg


class C4001Parser:
    """ASCII '$...*' framing; '*' is a terminator, not an invented NMEA checksum.

    DFHPD reserved fields are empty/spaces. DFDMD is opt-in for verified 25m
    firmware only. It is one aggregate target, never an association to a camera ID.
    """
    def __init__(self, allow_distance=False, partial_timeout=1):
        self.buffer = b""
        self.updated = 0.0
        self.partial_timeout = partial_timeout
        self.allow_distance = allow_distance
        self.invalid = 0

    def feed(self, chunk, now=None):
        now = time.monotonic() if now is None else now
        if self.buffer and now - self.updated > self.partial_timeout:
            self.buffer = b""
        if chunk:
            self.updated = now
        self.buffer += chunk
        packets = []
        while self.buffer:
            start = self.buffer.find(b"$")
            if start < 0:
                self.buffer = b""
                break
            self.buffer = self.buffer[start:]
            end = self.buffer.find(b"*")
            nested = self.buffer.find(b"$", 1)
            if nested >= 0 and (end < 0 or nested < end):
                self.invalid += 1
                self.buffer = self.buffer[nested:]
                continue
            if end < 0:
                if len(self.buffer) > 256:
                    self.buffer = b""
                    self.invalid += 1
                break
            raw, self.buffer = self.buffer[1:end], self.buffer[end + 1:]
            try:
                fields = [v.strip() for v in raw.decode("ascii").split(",")]
                if fields[0] == "DFHPD":
                    if len(fields) != 5 or fields[1] not in ("0", "1") or any(fields[2:]):
                        raise ValueError("Invalid DFHPD fields")
                    packets.append({"presence": fields[1] == "1", "distance": None, "speed": None})
                elif fields[0] == "DFDMD" and self.allow_distance:
                    if len(fields) != 8 or fields[1] not in ("0", "1") or any(fields[6:]):
                        raise ValueError("Invalid DFDMD fields")
                    distance = speed = None
                    if fields[1] == "1":
                        if fields[2] != "1":
                            raise ValueError("Only one radar target supported")
                        distance, speed, energy = map(float, fields[3:6])
                        if not (math.isfinite(distance) and 0 <= distance <= 25
                                and math.isfinite(speed) and 0 <= speed <= 10
                                and math.isfinite(energy) and energy >= 0):
                            raise ValueError("Invalid measurement")
                    elif any(fields[2:]):
                        raise ValueError("No-target measurements must be empty")
                    packets.append({"presence": fields[1] == "1", "distance": distance, "speed": speed})
            except (ValueError, UnicodeError):
                self.invalid += 1
        return packets


def first_channel_rms(data):
    import numpy as np
    samples = np.frombuffer(data, dtype="<i4").reshape(-1, 2)
    first = samples[:, 0].astype(np.float64) / 2147483648.0
    return float(np.sqrt(np.mean(first * first)))


class LatestState:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}

    def set(self, key, **values):
        with self.lock:
            self.values[key] = dict(values, updated=time.monotonic())

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.values)


class ButtonPulse:
    """Debounced rising press; held button cannot extend or retrigger a pulse."""
    def __init__(self):
        self.raw = self.pressed = False
        self.changed = self.until = 0.0

    def update(self, raw, now, debounce, duration):
        if raw != self.raw:
            self.raw, self.changed = raw, now
        if now - self.changed >= debounce and raw != self.pressed:
            self.pressed = raw
            if raw:
                self.until = now + duration
        return self.pressed, now < self.until


def alert_outputs(mic, radar, button, beep, rules):
    conditions = {"activity_or": mic or radar, "activity_and": mic and radar,
                  "button": button, "off": False}
    result = {name: bool(conditions[rules[name]]) for name in ("led1", "led2", "led3")}
    result.update(button=button, buzzer=bool(beep and rules["buzzer"] == "button"))
    return result


class HardwareService:
    def __init__(self, cfg, state=None):
        self.cfg = cfg
        self.state = state or LatestState()
        self.stop_event = threading.Event()
        self.threads = []
        self.process = None
        self.port = None
        self.sim_button = False

    def start(self):
        targets = (self._simulation,) if self.cfg["simulation"] else (self._mic, self._radar)
        for target in (*targets, self._gpio):
            t = threading.Thread(target=target, daemon=True, name=target.__name__)
            self.threads.append(t)
            t.start()

    def snapshot(self):
        data = self.state.snapshot()
        now = time.monotonic()
        for key, section in (("mic", "microphone"), ("radar", "radar")):
            item = data.setdefault(key, {})
            item["live"] = (item.get("health") == "connected"
                            and now - item.get("updated", 0) < self.cfg[section]["stale_seconds"])
            if not item["live"]:
                item["active"] = None
                if key == "radar":
                    item.update(distance=None, speed=None)
                else:
                    item["rms"] = None
        return data

    def _audio_device(self):
        cfg = self.cfg["microphone"]
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=3)
        matches = []
        for line in result.stdout.splitlines():
            match = re.search(r"card (\d+):.*?device (\d+):", line)
            if match and re.search(cfg["card_hint"], line, re.I):
                matches.append(f"plughw:{match[1]},{match[2]}")
        if len(set(matches)) == 1:
            return matches[0], "ALSA card matched configured name hint"
        return cfg["device"], "Explicit ALSA device; verify identity with arecord -l"

    def _mic(self):
        while not self.stop_event.is_set():
            proc = None
            try:
                device, note = self._audio_device()
                proc = subprocess.Popen(["arecord", "-q", "-D", device, "-t", "raw",
                                         "-f", "S32_LE", "-r", "48000", "-c", "2"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                self.process = proc
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    pending = b""
                    last = time.monotonic()
                    while not self.stop_event.is_set():
                        if not selector.select(0.2):
                            if time.monotonic() - last > 3:
                                raise RuntimeError("No PCM data for 3 seconds")
                            continue
                        chunk = proc.stdout.read(38400)
                        if not chunk:
                            raise RuntimeError("arecord ended; check ALSA device/permissions/format")
                        last = time.monotonic()
                        pending += chunk
                        while len(pending) >= 38400:
                            rms = first_channel_rms(pending[:38400])
                            pending = pending[38400:]
                            self.state.set("mic", health="connected", rms=rms,
                                           active=rms >= self.cfg["microphone"]["threshold"],
                                           device=device, note=note)
            except Exception as exc:
                self.state.set("mic", health="disconnected", active=None, error=str(exc))
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    proc.stdout.close()
                self.process = None
            self.stop_event.wait(2)

    def _radar(self):
        while not self.stop_event.is_set():
            port = None
            try:
                import serial
                cfg = self.cfg["radar"]
                parser = C4001Parser(cfg["allow_dfdmd_25m"], cfg["partial_timeout"])
                port = serial.Serial(cfg["device"], cfg["baud"], timeout=0.2)
                self.port = port
                last = time.monotonic()
                while not self.stop_event.is_set():
                    for packet in parser.feed(port.read(256)):
                        last = time.monotonic()
                        self.state.set("radar", health="connected", active=packet["presence"],
                                       distance=packet["distance"], speed=packet["speed"],
                                       malformed=parser.invalid)
                    if time.monotonic() - last > cfg["stale_seconds"]:
                        raise RuntimeError("No valid C4001 report; check UART mode and output cycle")
            except Exception as exc:
                self.state.set("radar", health="disconnected", active=None, error=str(exc))
            finally:
                if port is not None:
                    port.close()
                self.port = None
            self.stop_event.wait(2)

    def _simulation(self):
        start = time.monotonic()
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - start
            rms = 0.06 if int(elapsed) % 6 >= 3 else 0.005
            self.state.set("mic", health="connected", active=rms >= self.cfg["microphone"]["threshold"],
                           rms=rms, device="SIMULATED", note="No real hardware")
            self.state.set("radar", health="connected", active=int(elapsed) % 8 >= 4,
                           distance=None, speed=None)
            self.stop_event.wait(0.1)

    def _gpio(self):
        gpio = None
        failure = None
        cfg = self.cfg
        pins = cfg["gpio"]
        outputs = [pins[k] for k in ("led1", "led2", "led3", "buzzer")]
        try:
            if not cfg["simulation"]:
                import RPi.GPIO as gpio
                gpio.setmode(gpio.BCM)
                gpio.setup(outputs, gpio.OUT, initial=gpio.LOW)
                gpio.setup(pins["button"], gpio.IN, pull_up_down=gpio.PUD_UP)
            pulse = ButtonPulse()
            while not self.stop_event.is_set():
                data = self.snapshot()
                raw = self.sim_button if cfg["simulation"] else gpio.input(pins["button"]) == gpio.LOW
                button, beep = pulse.update(raw, time.monotonic(), cfg["alerts"]["debounce_seconds"],
                                            cfg["alerts"]["beep_seconds"])
                status = alert_outputs(bool(data["mic"]["active"]), bool(data["radar"]["active"]),
                                       button, beep, cfg["alerts"])
                if gpio is not None:
                    for name in ("led1", "led2", "led3", "buzzer"):
                        gpio.output(pins[name], status[name])
                self.state.set("gpio", health="connected", **status)
                self.stop_event.wait(0.01)
        except Exception as exc:
            failure = str(exc)
        finally:
            if gpio is not None:
                try:
                    gpio.output(outputs, gpio.LOW)
                except Exception as exc:
                    failure = f"Output shutdown failed: {exc}"
                try:
                    gpio.cleanup(outputs + [pins["button"]])
                except Exception as exc:
                    failure = f"GPIO cleanup failed: {exc}"
            self.state.set("gpio", health="disconnected" if failure else "stopped",
                           led1=None if failure else False, led2=None if failure else False,
                           led3=None if failure else False, button=None if failure else False,
                           buzzer=None if failure else False,
                           error=failure)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)
        if any(t.is_alive() for t in self.threads):
            raise RuntimeError("Sensor worker still stopping; do not start another GPIO session")
