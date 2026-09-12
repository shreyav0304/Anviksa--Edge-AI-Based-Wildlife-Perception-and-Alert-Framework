# ANVIKSA Raspberry Pi software setup

This prepares the complete existing Tkinter application for Raspberry Pi 4.
It does not integrate hardware or optimize the UI. The deployment model remains
**MobileNetV3 Large V1 Float32 TFLite**. V2/V3 experiments are not deployment models.

## Supported installation profile and version strategy

Use **Raspberry Pi OS Bookworm 64-bit with Desktop and its system CPython 3.11**
for the root requirements file. This is a conservative compatibility profile,
not a claim of physical Pi validation. The source uses Python 3.10 features
(union annotations and `zip(strict=True)`); 3.11 is selected because the published
`tflite-runtime` 2.14 wheel includes CPython 3.11 / ARM64 / glibc 2.34+.
Bookworm supplies a compatible system Python and glibc. This choice is independent
of the Windows development environment. See the [published runtime wheels](https://pypi.org/project/tflite-runtime/2.14.0/).

The NumPy `<2` constraint protects the older runtime's NumPy 1.x binary ABI;
[NumPy documents this ABI incompatibility](https://numpy.org/doc/stable/user/troubleshooting-importerror.html).
OpenCV is bounded below 4.12 to avoid its newer NumPy 2 requirement on Python 3.11.
Pillow's minimum supplies `Image.Resampling`. Matplotlib supplies Figure/TkAgg.
Other versions are resolved by pip for Python 3.11 rather than copied from Windows.
This is a bounded runtime dependency set, not a fully locked environment.

`opencv-python-headless` supplies the existing `cv2.VideoCapture`, resizing and
color conversion operations. The UI renders through Pillow/Tkinter, with no
OpenCV HighGUI calls, so this package supports the graphical application without
Qt/OpenGL dependencies. Its [ARM64 wheels and packaging guidance](https://pypi.org/project/opencv-python-headless/4.11.0.86/)
support this choice. Install only one pip OpenCV distribution; do not also install
`opencv-python`, contrib variants or a competing apt `python3-opencv` in this environment.

Do not apply this requirements file unchanged to a different Python/OS profile.
Raspberry Pi OS Trixie uses Python 3.13, as noted in the
[Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/ai.html).
The old runtime has no CPython 3.13 wheel, and NumPy 1.x is unsuitable there.
For a future Trixie deployment, use a separately reviewed NumPy/OpenCV dependency
set with an official compatible `ai-edge-litert` wheel. Published
[LiteRT wheels](https://pypi.org/project/ai-edge-litert/) include ARM64/CPython 3.13;
wheel existence does not establish model compatibility. Validate that profile
before adopting it. Do not downgrade the OS system Python or silently build native
packages from source. ARM32, alternative interpreters and other Python versions
are outside the documented installation profile.

The unchanged backend tries `tflite_runtime.interpreter.Interpreter`, then
`ai_edge_litert.interpreter.Interpreter`, then `tensorflow.lite.Interpreter`.
**Full TensorFlow is not required on the Pi.** LiteRT is an alternative, not an
additional required package. A broken native runtime may raise an exception other
than ImportError and prevent the existing fallback; fix that installation rather
than assuming fallback always works.

## Static dependency audit

All 66 existing Python source files were recursively parsed, including nested
imports, before adding the smoke test. The runtime path is `run_ui.py` -> `ui/`
-> `pi_deployment/inference/tflite_inference.py`. `src/snake_feature_reference.py`
uses standard-library modules. No camera/GPIO/serial/I2C Python imports currently
exist in the application. Local project modules are not pip packages.

| Area / import | Classification and provision |
| --- | --- |
| Home, navigation, widgets: `tkinter`, `ttk` | Standard-library interface with apt `python3-tk` |
| Image Analysis and Detection History: `PIL.Image`, `ImageTk` | pip `Pillow`, plus Tk system support |
| Video/Webcam: `cv2`, `numpy`, Pillow | pip `opencv-python-headless`, `numpy`, `Pillow` |
| Model Comparison / Confusion Matrix: `numpy`, `matplotlib.figure`, `matplotlib.backends.backend_tkagg` | pip `numpy`, `matplotlib`, plus Tk system support |
| TFLite preprocessing / benchmark: `numpy`, `PIL`, `tflite_runtime` | pip `numpy`, `Pillow`, `tflite-runtime` |
| `ai_edge_litert`, `tensorflow` in interpreter loader | Explicit alternative / development fallback, excluded from default requirements |
| CSV/JSON persistence, package verifier, benchmark result comparison | Standard library; no extra dependency |
| `os`, `sys`, `csv`, `json`, `hashlib`, `pathlib`, `datetime`, `threading`, `queue`, `time`, `statistics`, `argparse`, `collections`, `subprocess`, `shutil`, `dataclasses`, `typing`, `concurrent.futures`, etc. | Standard library; excluded from pip requirements |
| Picamera2/libcamera, GPIO Zero/lgpio | Optional future hardware support through apt |
| `serial` / `smbus` | Conditional future UART/I2C support; not currently required |

**DEVELOPMENT/TRAINING ONLY (outside the Pi application runtime):** TensorFlow
and scikit-learn support historical training, conversion and evaluation scripts
under `experiments/`, `train*.py`, `convert_validate_*.py`, and related utilities.
`model_comparison_dashboard.py` is an offline artifact generator: it imports
scikit-learn and can invoke TensorFlow to regenerate prediction caches. It is
not the Tkinter Model Comparison page. Likewise, root `inference_engine.py`
and `external_image_test.py` use full TensorFlow for legacy/development Keras
inference; they are not called by `run_ui.py`. Do not run these utilities as part
of Pi setup. Existing dataset/video auditing uses NumPy/Pillow/OpenCV and standard
library utilities; that does not justify adding training dependencies. A separate
`requirements-dev.txt` can be prepared later; none is created here.

## Raspberry Pi OS System Dependencies

Required for this profile (run on the Pi, not Windows):

```bash
sudo apt install -y git python3-pip python3-venv python3-tk
```

Use the Desktop image and an active graphical login for Tkinter/TkAgg. A plain
headless SSH session is insufficient to launch the UI. The selected headless
OpenCV wheel does not require extra Qt/OpenGL apt packages; camera drivers and
codec support remain platform-dependent and must be tested on the actual Pi.
Do not blindly add `libgl1` or a second OpenCV installation.

Optional future camera support:

```bash
sudo apt install -y python3-picamera2
```

The [official Picamera2 installation guidance](https://github.com/raspberrypi/picamera2/blob/main/README.md)
recommends apt, which supplies compatible libcamera bindings/dependencies.
Use `--system-site-packages` with the OS Python when a future camera environment
must see these bindings. Recheck NumPy/native-package compatibility in that
environment; do not blindly pip-install Picamera2 over system packages.
The backend already accepts PIL images and RGB NumPy arrays, providing an
appropriate future camera-adapter boundary. Current video handling uses
OpenCV VideoCapture; no Picamera2 adapter exists yet. Installation alone does
not make a CSI Pi Camera work through the current webcam control. Modern
libcamera-based camera support does not need the legacy camera toggle.

Optional future LED/buzzer support:

```bash
sudo apt install -y python3-gpiozero python3-lgpio
```

[GPIO Zero](https://gpiozero.readthedocs.io/en/stable/installing.html) is appropriate
for simple LED and buzzer objects, with an lgpio pin backend. No pin assignments,
outputs or devices are initialized in this phase. A future venv using these apt
bindings needs `--system-site-packages` and the OS Python.

mmWave remains a planned motion/presence trigger; the sensor and protocol are
not specified. Do not install a proprietary library or both interfaces by default.
If the chosen sensor uses UART, `sudo apt install python3-serial` supplies
`serial` (PySerial). If it uses I2C, `sudo apt install python3-smbus i2c-tools`
supplies `smbus` and diagnostic tools. These apt bindings also require system
site-package visibility in a venv. A digital presence output might need only GPIO.
Use `sudo raspi-config` to enable I2C or the serial hardware only once the
interface is selected; for UART disable the serial login console while retaining
the serial port. Check device permissions (`dialout`, `i2c`, `gpio`, `video` as
applicable), logging out/in after any group change. Do not run ANVIKSA as root.

## Installation sequence

1. Flash Raspberry Pi OS **Bookworm 64-bit with Desktop** using Raspberry Pi Imager
   or the official archived image. Verify the release selection explicitly.
2. Boot, configure networking/user access, and update:
   ```bash
   sudo apt update
   sudo apt full-upgrade -y
   sudo reboot
   ```
3. Confirm `uname -m` reports `aarch64`, `python3 --version` reports 3.11, and
   `/etc/os-release` identifies Bookworm. Configure only interfaces actually
   needed as described above; hardware is not needed for dependency checks.
4. Install the required apt packages from the system-dependencies section.
   Camera/GPIO/mmWave apt packages are optional future work.
5. Clone the complete repository (replace the placeholder with your repository URL):
   ```bash
   git clone <ANVIKSA_REPOSITORY_URL> anviksa_ai
   cd anviksa_ai
   ```
   Retain `ui/`, `src/`, `pi_deployment/`, and the existing `results/` artifacts.
   Copy any locally retained artifacts absent from the clone without regenerating
   them. Home needs the V1 metrics and deployment manifest. Model Comparison needs
   `results/model_comparison_dashboard/model_comparison_metrics.csv`, and both
   comparison pages need `metrics.json`, `config.json`,
   `confusion_matrix_raw.csv`, and `confusion_matrix_normalized.csv` in these runs:
   ```text
   results/custom_cnn/20260830T142449Z_c23a5d39/
   results/mobilenet_v2/20260830T161743Z_552c91eb/
   results/mobilenet_v3_large/20260830T171150Z_94069968/
   ```
   Keep existing detection-history CSV and image files if prior history is wanted.
   Runtime logging needs a writable project location. Training datasets/Keras
   models are not required for these pages or TFLite inference.
6. Create an isolated application virtual environment:
   ```bash
   python3 -m venv .venv
   ```
   For future apt camera/GPIO integration, create a separate environment with
   `python3 -m venv --system-site-packages .venv-hardware` and reassess its native
   dependencies before adopting it. The default app environment stays isolated.
7. Activate:
   ```bash
   source .venv/bin/activate
   ```
8. Install Python dependencies, requiring wheels to avoid unexpected native builds:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install --only-binary=:all: -r requirements.txt
   python -m pip check
   ```
   If no compatible wheel is found, check OS/architecture/Python against the
   documented profile. Do not install full TensorFlow as an automatic workaround.
   The historical `pi_deployment/requirements_pi.txt` is unchanged; use the root
   file for the full application.
9. Verify both model files with the existing read-only verifier:
   ```bash
   python -B pi_deployment/verify_package.py
   ```
   It must report `"pass": true`. Expected SHA-256 values:
   ```text
   pi_deployment/models/mobilenet_v3_large_float32.tflite
   086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7
   pi_deployment/models/mobilenet_v3_large_float16.tflite
   5b9e7763e605c7e5f8081dcf12c2ff6afd1cb1d7cd0f6f3cda909e07d1e7c4e5
   ```
10. The same verifier checks `pi_deployment/inference/class_names.json` and the
    manifest against the exact order: `Cow`, `Deer`, `Elephant`, `Monkey`,
    `Non_Venomous_Snake`, `Venomous_Snake`, `Wild_Boar`. Do not replace this with a
    V2/V3 class file or reorder labels.
11. Run the dependency smoke test:
    ```bash
    python -B scripts/check_pi_dependencies.py
    ```
    Exit 0 means required imports and an interpreter class were available; exit 1
    means at least one required check failed. Missing alternative interpreters
    are non-fatal if a supported interpreter is available. Optional modules show
    AVAILABLE or NOT YET REQUIRED and never affect exit status. Native imports
    run in subprocesses with timeouts; no model or hardware device is opened.
    This does not verify model allocation, inference, GUI display access, video
    codecs or hardware functionality. Those need later on-device testing.
12. In the Pi desktop terminal, with the venv active, launch:
    ```bash
    python run_ui.py
    ```
    The dependency set covers Home, Image Analysis, Video/Webcam, Model Comparison,
    Confusion Matrix and Detection History. Pi performance and hardware operation
    remain unverified. Do not run training, conversion or dashboard regeneration
    during setup.

## Preparation validation

The dependency inventory contains only imports used by the application, with
standard-library, system, alternate-runtime and future-hardware imports accounted
for above. No inference implementation, UI, historical requirements, models,
datasets, video data or experiment results are changed by this preparation.
No packages are installed on the Windows development PC. Successful installation
and physical Pi testing remain deployment steps, not claimed results here.
