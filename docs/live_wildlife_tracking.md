# Live Wildlife Tracking: integrated Anviksa application

This extends the existing Tkinter dashboard launched by run_ui.py. It does not
replace Video / Webcam, Image Analysis, history, or model-comparison pages.
Live Wildlife Tracking is a ninth section (Alt+9); the existing Alt+1..8 bindings
remain in their original order. Stop or navigate away to release its devices.
Do not run two Anviksa processes or separate hardware scripts simultaneously.

## Architecture and scope

- ui/app.py adds one lazy page registration.
- ui/pages/live_tracking.py adds preview, relative map, telemetry and controls.
- ui/services/live_hardware.py owns independent ALSA, UART and GPIO workers.
- ui/services/live_tracking.py owns camera motion segmentation, centroid tracks
  and a separate classifier worker. Only the latest frame/results are retained.
- config/live_tracking.json holds resolution/FPS, pins, thresholds, timeouts,
  motion parameters, risk classes and alert rules.
- requirements_hardware.txt adds serial and Pi GPIO dependencies.
- scripts/test_live_hardware.py observes physical operation or checks simulation.
- tests/test_live_tracking.py verifies parser, alerts, stale states and tracking.

Existing image/video inference code and model files are untouched. The live page
reuses the hash-verified deployed Float32 model and exact class mapping:
pi_deployment/models/anviksa_15class_float32.tflite and
pi_deployment/inference/class_names.json. The loader validates all 15 names;
Non_Venomous_Snake is index 10 and Venomous_Snake is index 13 in that deployed
mapping. Runtime rules compare validated names, not guessed indices.
There is no retraining.

The classifier produces one class vector, not boxes. MOG2 background subtraction
provides independent motion-region proposals; centroid matching tracks them.
One proposal crop per classification interval is processed, oldest result first,
with one interpreter thread. Motion crops differ from the training distribution:
scores are exploratory, uncalibrated and can label people/background as animals.
The model has no Background/Unknown output. Missing results display Unknown.

The map uses normalized camera-frame X/Y coordinates, not world coordinates:
left/top are 0%, right/bottom are 100%. Rings are decorative relative guides.
The sensor icon is a reference symbol, not calibrated geometry. Camera Y does
not measure distance. Radar has no per-track identity association.
Distance and bearing stay Unknown for camera tracks. Radar presence only
corroborates scene activity; it does not confirm species or individual tracks.

Green = established motion track; yellow = tentative/lost briefly, uncertain
classification or risk candidate awaiting review. A high-score stable configured
risk-class prediction is yellow until an operator reviews the camera and enters
that track ID in Confirm reviewed risk. Only that explicit review permits red.
Red is an operator flag, not proof of identification. No classification result
activates a physical danger alarm in this initial integration.

Centroid IDs can switch at crossings/occlusions. Shadows, weather, foliage and
camera motion can create false regions. A fixed camera and stationary background
are required; warmup defaults to 20 frames. Stationary objects fade from motion
segmentation and tracks expire after one second without a matched region.

## Wiring: BCM numbering

Keep your tested power wiring, LED resistors, common ground and buzzer driver.
Power down before rewiring. Pi GPIO signals must remain 3.3 V compatible.

| Function | BCM GPIO | Physical header pin |
|---|---:|---:|
| LED1 | 17 | 11 |
| LED2 | 27 | 13 |
| LED3 | 22 | 15 |
| Active-high buzzer driver control | 23 | 16 |
| Button to GND, internal pull-up | 24 | 18 |
| INMP441 SCK | 18 | 12 |
| INMP441 WS | 19 | 35 |
| INMP441 SD | 20 | 38 |
| Radar TX -> Pi RX | 15 | 10 |
| Radar RX <- Pi TX | 14 | 8 |
| Webcam | USB | USB |

INMP441 uses 3.3 V supply. Its L/R selection must match the working first-channel
capture (normally L/R to GND for left). Pin validation rejects duplicate GPIOs
and UART14/15 or I2S18..21 reuse by LEDs/buzzer/button. Existing I2S overlay and
working buzzer circuit must be retained.

## Raspberry Pi installation

Run the following in the Terminal on the Raspberry Pi desktop, not Windows.
Open it with Ctrl+Alt+T. The word bash above a command block is not a command.

Copy this updated existing repository to the Pi first. In Terminal enter cd,
a space, then drag its project folder into Terminal, and press Enter. This
avoids guessing the folder's installation path.

Use the existing supported profile: Raspberry Pi OS Bookworm 64-bit Desktop,
system Python 3.11. See raspberry_pi_setup.md for runtime/version limitations.
Do not reinstall or downgrade an already working environment unnecessarily.

For a fresh environment in this repository folder:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-tk python3-dev build-essential alsa-utils v4l-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements_hardware.txt
```

For an existing working environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements_hardware.txt
```

Check access and devices:

```bash
groups
lsusb
v4l2-ctl --list-devices
arecord -l
arecord -L
ls -l /dev/serial0 /dev/gpiomem /dev/video*
python -c "import cv2, numpy, serial, RPi.GPIO; print('Hardware dependencies loaded')"
```

If the current login lacks the relevant access groups:

```bash
sudo usermod -aG audio,video,dialout,gpio "$USER"
```

Log out and back in after a group change. Do not run the dashboard with sudo.

UART setup:

```bash
sudo raspi-config
```

Interface Options -> Serial Port: serial login shell No; serial hardware Yes.
Reboot if changed. Preserve the working 9600 baud setting. No UART configuration,
reset, saveConfig, or working-mode commands are sent by this application.

## ALSA capture and C4001 protocol

Capture is exactly S32_LE, 48000 Hz, 2 channels, 100 ms blocks; the first
signed little-endian channel is divided by 2^31 before RMS calculation.
Threshold defaults to 0.035 and is editable on the page before Start or in JSON.
The code looks for a unique ALSA card name matching microphone.card_hint.
If found, its current card/device numbers override the configured device.
Otherwise it uses explicit plughw:3,0 and displays a warning to verify identity.
Do not assume card 3 is still the INMP441: USB webcams can add capture cards.
Change card_hint to the actual ALSA card name if needed; ambiguous matches never
choose an arbitrary card.

Standalone capture check (stop the application first):

```bash
arecord -D plughw:3,0 -t wav -f S32_LE -r 48000 -c 2 -d 3 /tmp/anviksa-mic-check.wav
```

The parser uses '$' start and '*' end framing; it handles split/concatenated
packets, whitespace in reserved fields, bad values, nested starts, bounded
buffers and partial-frame expiry. Malformed packets do not refresh freshness.
DFHPD has status 0/1 and three reserved fields. The observed
$DFHPD,1,,,* and space-filled reserved fields are accepted.

Protocol source: [DFRobot C4001 UART documentation, active reporting sections
3.1/3.2](https://dfimg.dfrobot.com/5ea64bf6cf1d8c7738ad2881/wiki/3b88a4ecd7d7f18918a0fa8ba1f970c6.pdf).
The referenced document covers the 25m variant. Its baud descriptions differ
between sections; this app deliberately retains your observed 9600 setting.

Presence-only DFHPD contains no distance. By default distance remains Unknown.
Only after verifying a matching 25m device/firmware and observing actual DFDMD
reports in its distance mode may you set radar.allow_dfdmd_25m=true.
DFDMD supports at most one aggregate target. Its target number, distance in m,
speed in m/s, energy, reserved fields and valid ranges are checked. These values
are never attached to camera IDs. A DFHPD report or no-target report clears
previous distance. No angle or coordinates are inferred. The application never
changes sensor mode automatically.

## Launch

```bash
python -B run_ui.py
```

Choose Live Wildlife Tracking using the Section picker or Alt+9. Leave
SIMULATION unchecked for physical hardware and press Start. The webcam preview,
relative map, RMS, presence, track count, classification, GPIO state and
connection health appear together. Model failure leaves camera/sensors operating.
Switching away stops the live session so Video / Webcam can acquire the camera.
Plain SSH cannot show Tk windows: use a Pi desktop monitor or remote desktop.

Hardware-free check (Windows or Pi existing app environment):

```bash
python -B scripts/test_live_hardware.py --simulate --seconds 13
```

Or check SIMULATION on the live page, then Start. All camera regions, sensor
states and button alerts are explicitly simulated; no GPIO/UART/ALSA is opened.
Hold Simulate button to check one beep pulse per press. Classification is
disabled in simulation so synthetic shapes are never presented as animal evidence.

Console observation on the real Pi (run separately from the dashboard):

```bash
python -B scripts/test_live_hardware.py --seconds 60
```

This reports observed transitions and cleanup, not an automatic physical pass.
The dashboard provides the frame display; the console command is suitable for
headless observation. A person must verify LED light and audible buzzer output.

## Physical acceptance checklist

1. Start live mode: camera shows current frames. Wave in view after warmup;
   regions/dots appear and move, with reasonably stable IDs in separated paths.
2. Quiet room: MIC 0; clap near INMP441: MIC 1 and RMS crosses the threshold.
   Adjust threshold for ambient noise rather than treating volume as species.
3. Enter and leave radar coverage: presence changes 0/1, allowing the radar's
   configured internal hold/disappearance delay.
4. With either microphone or radar active: GPIO17 ON. With both active:
   GPIO27 ON. With neither: both OFF.
5. Press button: GPIO22 ON while held; GPIO23 beeps once for 0.5 seconds.
   Hold for two seconds: no repeated or extended beep. Release/repress: one
   new pulse. Confirm physically, not just from software status indicators.
6. Multiple moving separated objects: distinct IDs. Crossings/occlusions are
   limitations of lightweight centroid matching; they are not radar targets.
7. Stop the mic capture source through a controlled software disconnect or
   disable its device: camera/radar continue; MIC becomes Unknown and its
   contribution to activity LEDs clears within the configured stale timeout.
   Restore device: capture retries. Do not remove GPIO wiring while powered.
8. For radar failure testing, use a controlled serial device disconnect or
   disable UART access; camera/mic continue, radar becomes Unknown, and prior
   distance clears. Restore access and observe reconnection. Adjust stale_seconds
   to exceed the configured periodic report interval. Query-only mode has no
   periodic heartbeat and requires sensor configuration outside this app.
9. Unplug/reconnect USB webcam: map clears, camera health reports unavailable,
   sensors/button continue, camera retries. If its device number changes,
   update camera.source (a stable /dev/v4l/by-id/... path is supported).
10. Stop, navigate away, close window, or Ctrl+C in the console observer:
    outputs OFF, capture released, serial closed, arecord terminated. Restart
    without device-busy errors. SIGTERM is handled by the console observer;
    normal GUI closure and SIGINT/SIGTERM use the page lifecycle. Forced kill/power loss cannot
    guarantee software cleanup; hardware driver defaults remain important.

Alert rules are named values in JSON: activity_or, activity_and, button, off.
Defaults exactly implement the required three LEDs. Buzzer accepts button/off
only and defaults to 0.5 seconds. Classification risk classes and review display
are configurable; adding future species-based physical actions requires explicit
reviewed rules rather than reusing mic volume or scene-level radar presence.

## Troubleshooting

| Symptom | Action |
|---|---|
| OpenCV missing | Activate existing environment; install its requirements.txt. Do not mix pip OpenCV variants with apt OpenCV in this venv. |
| No GUI / display error | Run from the Pi graphical desktop; use the console observer over SSH. |
| ALSA device busy | Stop other test scripts/recorders and other application instances. |
| RMS zero | Verify ALSA identity, working I2S overlay, 3.3 V supply, first channel and L/R selection. |
| Radar unavailable | Verify 9600, serial console disabled, UART enabled, common ground, crossed TX/RX, UART mode and periodic DFHPD reports. |
| GPIO unavailable | Check Pi 4 hardware, RPi.GPIO import, gpio group and pin ownership. No fake GPIO fallback is used in live mode. |
| Buzzer silent | Check the previously tested active-high driver and supply; software state alone does not prove sound. |
| Motion everywhere | Fix the camera, wait warmup, increase min_area_fraction, reduce moving foliage/shadows in view. |
| Tracks disappear at rest | Expected: localization is motion segmentation, not a trained object detector. |
| Wrong confident animal label | Model lacks Background; crop/scene predictions are exploratory. Do not confirm risk solely from score. |
| Sensor stopped / stale | Unknown is intentional, not a negative detection; other components continue independently. |
| Still stopping warning | Do not start another session. Wait for worker exit; investigate a blocked camera driver. GPIO stop is attempted first. |

## Development validation

```bash
python -B -m unittest discover -s tests -v
ANVIKSA_GUI_TESTS=1 python -B -m unittest discover -s tests -v
```

GUI tests need an active desktop. Physical Pi performance, acoustic threshold,
device reconnection and animal field behavior remain unverified until the
acceptance checklist is performed on the Pi.

Development-machine results (2026-11-04): all 51 unit and opt-in dashboard
tests passed, including simulated preview/map and navigation cleanup. After
final cleanup/status changes, all 14 affected hardware/tracking and dashboard
checks passed again. The 13-second simulation observed both states for the mic,
radar, three LEDs, button and buzzer; two persistent IDs moved and all workers
stopped with outputs OFF. The existing hash-verified 15-class Float32 model
loaded and completed a synthetic-input invocation using the existing Windows
LiteRT backend. This checks loading/interface only, not animal recognition
accuracy, Raspberry Pi speed, or physical hardware operation.
