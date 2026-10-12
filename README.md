# Anvīkṣa - Edge AI-Based Wildlife Perception and Alert Framework

Anvīkṣa AI is an offline wildlife classification project for agricultural land, forest borders, and residential areas. It combines a validated MobileNetV3 Large classifier with a Tkinter dashboard, image/video analysis, Wildlife Gate pre-screening, and local detection history.

**Software status:** dashboard and inference implemented; model validation and development-machine tests pass. **Physical Raspberry Pi validation and field hardware integration remain pending.**

## Dashboard

Launch the desktop application with `python run_ui.py`. Its sidebar provides:

| Section | Features |
| --- | --- |
| Overview | Selected model, validated metrics, classes, and project status |
| Image Analysis | Local image classification, class probabilities, confidence, and explicitly saved detections |
| Video / Webcam | Recorded videos or USB webcam frames, Wildlife Gate pre-screening, probability smoothing, and stable sightings |
| Model Comparison | Accuracy, precision, recall, F1, probability MSE, confidence, and snake-class performance |
| Confusion Matrix | Separate columns for all models, raw counts or normalized percentages, and selected-model details |
| Detection History | Local events, filters, snapshots, summaries, and CSV export |
| System Deployment | Device information, model-file availability, integration status, and launch instructions |
| About Project | Purpose, perception pipeline, classes, research results, and documentation |

Navigation and page scrollbars keep these sections accessible on small displays. Desktop regression checks cover **800x480** and **480x320**. Pages load independently; a failed section shows its error and a Retry button while other sections remain available.

Comparison pages read committed summaries and frozen prediction caches. Private historical training runs and dashboard regeneration are not required. EfficientNetB0 shows **N/A** because its controlled experiment was incomplete.

## Raspberry Pi quick start

The documented installation profile is **Raspberry Pi OS Bookworm 64-bit with Desktop and Python 3.11**. See [the full setup guide](docs/raspberry_pi_setup.md) for compatibility details and troubleshooting. The root requirements file targets this profile; do not apply it unchanged to other OS/Python combinations.

From a terminal in the Pi desktop:

```bash
sudo apt install -y git git-lfs python3-pip python3-venv python3-tk

# Skip large LFS source videos; they are not needed to run the dashboard.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/shreyav0304/Anviksa--Edge-AI-Based-Wildlife-Perception-and-Alert-Framework.git
cd Anviksa--Edge-AI-Based-Wildlife-Perception-and-Alert-Framework

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip check

python -B pi_deployment/verify_package.py
python -B scripts/check_dashboard.py
python -B scripts/check_pi_dependencies.py
python run_ui.py
```

Use an active graphical desktop session; a plain SSH session has no display for Tkinter. The app uses headless OpenCV for capture and processing, and Pillow/Tkinter for display. Full TensorFlow is not required on the Pi.

To update an existing clean checkout:

```bash
git pull --ff-only origin main
source .venv/bin/activate
python -B scripts/check_dashboard.py
python run_ui.py
```

For a USB webcam preview without model inference:

```bash
python pi_deployment/webcam_test.py --camera 0
```

Try another camera index if needed. Camera drivers, access permissions, codecs, and Pi performance require on-device checks.

## Perception pipeline and limits

```text
Local image / recorded video / USB webcam
                    |
          RGB frame preprocessing
                    |
     Wildlife Gate (video/webcam only)
                    |
     MobileNetV3 Large classification
                    |
       Class, confidence, threat metadata
                    |
     Dashboard + local history/snapshots
```

Wildlife Gate rejects non-wildlife frames and pauses species classification when uncertain. Confident, stable video/webcam sightings are saved automatically; image detections are saved explicitly. Eligible events use a separate queue so dropped preview frames do not discard detection history.

The seven-class model classifies a **whole frame**; it does not produce bounding boxes or identify individual animals. It has no Background/Unknown class. Gate decisions, confidence thresholds, and threat labels are application logic, not guarantees of species identification. Snake predictions require careful human verification.

The proposed field hardware adds an mmWave presence trigger, Raspberry Pi camera, GPIO LED/buzzer alerts, rechargeable battery, charge controller, and solar panel. These integrations remain planned. The mmWave sensor would trigger capture; camera images provide the classification input.

## Dataset and class order

The original frozen wildlife training dataset and local integrity manifests are excluded from Git. The repository also contains supplemental snake-image/video material and related audit tools; these are separate from the original controlled experiment.

The validated classifier's output order is fixed:

1. Cow
2. Deer
3. Elephant
4. Monkey
5. Non_Venomous_Snake
6. Venomous_Snake
7. Wild_Boar

Completed baseline models were evaluated on the same **545-image frozen test split**. Training and dataset utilities require their corresponding local inputs and are not part of dashboard startup.

## Model comparison

| Model | Test accuracy | Macro F1 | Status |
| --- | ---: | ---: | --- |
| Custom CNN | 50.46% | 50.73% | Completed baseline |
| MobileNetV2 | 92.48% | 95.20% | Completed transfer-learning baseline |
| **MobileNetV3 Large** | **95.41%** | **96.87%** | **Selected deployment architecture** |
| EfficientNetB0 | N/A | N/A | Incomplete; excluded from ranking |

MobileNetV3 Large achieved **92.91% venomous-snake recall** and **92.16% snake macro recall**. Its frozen-test confusion matrix contains **520 correct predictions out of 545**.

See [comparison results](results/comparison/) and [dashboard artifacts](results/model_comparison_dashboard/) for tables, plots, and prediction caches. Additional V2/V3/video-training experiments in the repository do not replace the validated **MobileNetV3 Large V1 Float32 TFLite** model used by the app.

## TensorFlow Lite deployment

| Representation | Model size | Test accuracy | Frozen-test prediction agreement |
| --- | ---: | ---: | --- |
| Float32 TFLite | 12.32 MiB | 95.41% | 545/545 with Keras |
| Float16 TFLite | 6.25 MiB | 95.41% | 545/545 with Keras and Float32 |

Float16 reduces storage by approximately **49.31%**. The current dashboard uses Float32; choosing the preferred representation on the Pi awaits physical benchmarking.

The species classifier expects RGB images resized to **224 x 224** with nearest-neighbor interpolation, Float32 dtype, and pixels in **[0, 255]**. Normalization is embedded in the model: do not divide pixels by 255 or apply MobileNetV2 preprocessing. Wildlife Gate has its own preprocessing implementation.

Deployed files are in [pi_deployment/models/](pi_deployment/models/); conversion metrics and matrices are in [results/deployment/](results/deployment/). Model integrity, exact class order, input tensors, and output probabilities are checked by the runtime and diagnostics.

Development-machine timing measurements are **not Raspberry Pi performance measurements**. For separate model-only and end-to-end benchmarks using 1, 2, and 4 threads, follow [the benchmark guide](pi_deployment/README_PI.md). Compare all six configurations using the same image, device, and measurement protocol.

## Detection history

Events and snapshots are stored locally in `results/detection_history/`, which is excluded from Git and separate from training data. A new Pi starts with an empty history.

The viewer supports filtering, snapshot inspection, and export to a separate CSV. It protects the master history from export overwrite and reports skipped malformed rows. Older history files without optional video frame/time fields remain readable and appendable.

History counts represent runtime detections, not accuracy, precision, recall, or F1.

## Health checks and regression tests

From the activated app environment:

```bash
python -B pi_deployment/verify_package.py
python -B scripts/check_dashboard.py
python -B scripts/check_pi_dependencies.py
python -B -m unittest discover -s tests -v
```

The package/dashboard checks read artifacts without running models or opening hardware. The dependency check probes imports. Regression tests use temporary files and simulated camera behavior.

Include desktop tests from an active graphical login:

```bash
ANVIKSA_GUI_TESTS=1 python -B -m unittest discover -s tests -v
```

The [12 October 2026 checkup report](docs/checkup_report.md) records:

- 90 Python sources and 64 tracked JSON artifacts validated.
- 27 regression checks passed on Windows, including five desktop checks.
- 22 non-GUI checks passed under Linux.
- Float32, Float16, and Wildlife Gate model hashes verified.
- Real Linux CPU inference and a temporary six-frame video pipeline passed.

These checks do not establish physical Pi installation, camera reliability, thermal behavior, memory use, or throughput.

## Project structure

```text
run_ui.py                           Desktop dashboard entry point
ui/app.py                           Navigation, styles, page loading, and error reporting
ui/pages/                           Dashboard sections
ui/services/                        Data, inference, video, gate, and history services
scripts/check_dashboard.py          Published-artifact and model-integrity checks
scripts/check_pi_dependencies.py    Isolated dependency probes
tests/                              Runtime and optional desktop regression checks
requirements.txt                    Supported Raspberry Pi app dependencies
pi_deployment/                      TFLite models, inference, benchmarks, and webcam test
wildlife_gate/                      Video/webcam pre-screening model and research scripts
experiments/                        Controlled training and evaluation utilities
results/comparison/                 Baseline comparison tables and plots
results/model_comparison_dashboard/ Dashboard summaries and prediction caches
results/deployment/                 TFLite validation metrics and matrices
snake_video_dataset/                Video collection, review, and dataset utilities
docs/                               Setup, validation, and reference documentation
inference_engine.py                 Legacy MobileNetV2 Keras inference component
train.py                            Controlled experiment entry point
convert_validate_float*.py          Reproducible TFLite conversion and validation
```

## Remaining work

Physical Raspberry Pi installation, display/camera testing, and benchmarking remain pending. mmWave triggers, GPIO alerts, and solar-powered field deployment remain planned. This repository contains validated models and implemented desktop software; it is not yet a fully validated field-hardware prototype.
