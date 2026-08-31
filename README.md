# Anvīkṣa — Edge AI-Based Wildlife Perception and Alert Framework

Anvīkṣa AI is a final-year AI/ML and Edge AI project for offline wildlife perception near agricultural land, forest borders, and residential areas. It focuses on classifying snakes—including venomous and non-venomous snakes—and other wildlife from camera images while keeping inference suitable for an edge device.

## Problem statement

Wildlife entering human-used areas can threaten people, crops, livestock, and the animals themselves. Many field locations have unreliable connectivity, so Anvīkṣa is designed around local sensing, classification, alerting, and event storage instead of a cloud dependency.

## Proposed system architecture

```text
mmWave radar (motion/presence trigger)
                ↓
Camera image or video frame
                ↓
Image preprocessing
                ↓
MobileNetV3 Large wildlife classifier
                ↓
Class + confidence → threat/decision logic
                ↓
LED/buzzer alert + local logging/image storage
```

The mmWave radar is intended only to detect motion or presence and trigger the camera. Wildlife classification is performed by the camera and AI model. The final field concept targets a Raspberry Pi 4, Raspberry Pi Camera, LED, buzzer, rechargeable battery, charge controller, and solar panel.

## Dataset and classes

The private working dataset is intentionally excluded from this public repository. The fixed model output order is:

1. Cow
2. Deer
3. Elephant
4. Monkey
5. Non_Venomous_Snake
6. Venomous_Snake
7. Wild_Boar

The controlled experiments used frozen train, validation, and test manifests. The final test split contains 545 images. Dataset preparation and integrity utilities remain available in the source, but image data and local integrity reports are not published.

## Preprocessing and offline inference

Training and validated TFLite evaluation decode images as RGB, resize to `224 × 224` with nearest-neighbor interpolation, cast to Float32, and retain the external pixel range near `[0,255]`. MobileNetV3 normalization is embedded in the model; callers must not divide pixels by 255 or apply MobileNetV2 preprocessing.

The Raspberry Pi package implements this exact shared input contract. OpenCV is planned as the application-layer interface for camera frames and outdoor image/video processing; camera-stream integration has not yet been completed. The model itself runs completely offline through TensorFlow Lite.

## Model comparison

All completed models were evaluated on the same frozen test split.

| Model | Test accuracy | Macro F1 | Status |
|---|---:|---:|---|
| Custom CNN | 50.46% | 50.73% | Completed baseline |
| MobileNetV2 | 92.48% | 95.20% | Completed transfer-learning baseline |
| **MobileNetV3 Large** | **95.41%** | **96.87%** | **Selected model** |
| EfficientNetB0 | — | — | Attempted; excluded from ranking because training could not complete with available computational resources |

The selected MobileNetV3 Large model achieved:

- Test accuracy: **95.41%**
- Macro F1: **96.87%**
- Venomous Snake recall: **92.91%**
- Snake Macro Recall: **92.16%**

Detailed comparison tables and plots are available in [`results/comparison`](results/comparison/). EfficientNetB0 has no fabricated or partial test score in the ranking.

## TensorFlow Lite deployment

Two byte-verified deployment representations are included:

| Representation | Size | Test accuracy | Prediction agreement |
|---|---:|---:|---|
| Float32 TFLite | 12.32 MiB | 95.41% | 545/545 with Keras |
| Float16 TFLite | 6.25 MiB | 95.41% | 545/545 with both Keras and Float32 |

Float16 preserves the complete test-set classification result while reducing storage by approximately 49.31% relative to Float32. Validation artifacts are under [`results/deployment`](results/deployment/), and the deployable models are under [`pi_deployment/models`](pi_deployment/models/).

Windows CPU timing measurements in the validation artifacts are development-machine measurements. They are **not Raspberry Pi performance results** and should not be used to claim Pi throughput.

## Raspberry Pi benchmark package

[`pi_deployment`](pi_deployment/) contains:

- Both validated TFLite models and their SHA-256 manifest
- Reusable single-image TFLite inference
- Separate model-only and end-to-end benchmarking
- 1, 2, and 4-thread configurations
- Memory, temperature, system-information, and stability recording
- Six-configuration result comparison
- Package-integrity verification

See [`README_PI.md`](pi_deployment/README_PI.md) for setup and benchmark commands. No Raspberry Pi benchmark has been performed yet, so the final choice between Float32 and Float16 on the target device remains pending.

## Project structure

```text
experiments/                 Reproducible model registry, data pipeline, training, and evaluation
pi_deployment/               Raspberry Pi inference and benchmark package
results/comparison/          Valid model comparison tables and plots
results/deployment/          Float32/Float16 validation metrics and confusion matrices
inference_engine.py          Existing desktop inference component
dataset_integrity.py         Dataset integrity and leakage audit tooling
train.py                     Controlled experiment entry point
convert_validate_float*.py   Reproducible TFLite conversion and numerical validation
```

## Current status

- Dataset preparation and integrity checking: complete locally
- Custom CNN, MobileNetV2, and MobileNetV3 Large comparison: complete
- MobileNetV3 Large selection: complete
- Float32 and Float16 TFLite numerical validation: complete
- Offline Raspberry Pi inference/benchmark package: prepared
- Physical Raspberry Pi benchmarking: **pending**
- OpenCV live camera pipeline: **pending**
- mmWave trigger integration: **pending**
- GPIO LED/buzzer integration: **pending**
- Solar-powered field integration: **concept/planned**

The repository currently represents a validated AI model and deployment-preparation stage, not a completed field-hardware prototype.

## Quick package verification

```bash
cd pi_deployment
python verify_package.py
```

For Raspberry Pi environment setup and benchmarking, follow [`pi_deployment/README_PI.md`](pi_deployment/README_PI.md).
