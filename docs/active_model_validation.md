# Active 15-class deployment validation - 15 October 2026

The supplied run is `mobilenetv3large_15class_gpu_20261011_123242`. The application now uses its Float32 export and the exact ordered 15-class metadata. The earlier seven-class deployment models and historical baseline artifacts are retained as historical files.

## Full export evaluation

All 670 original test image SHA-256 hashes were verified. Original and reviewed test manifest rows matched exactly. The stored Keras confusion matrix was independently rebuilt from its labeled prediction cache. Both exports were then run on every original test image through the application preprocessing and interpreter adapter under Linux CPU TensorFlow Lite (TensorFlow 2.21.0). This involved inference only: no training, calibration fitting, threshold tuning, test-set edits, or checkpoint reselection.

| Model | Correct / tested | Accuracy | Macro F1 | Mean raw model score |
|---|---:|---:|---:|---:|
| Keras checkpoint (saved evaluation) | 621/670 | 92.6866% | 91.8177% | 93.579768% |
| Float32 TFLite (active) | 620/670 | 92.5373% | 91.7682% | 93.581476% |
| Float16 TFLite | 621/670 | 92.6866% | 91.8177% | 93.570036% |

The earlier export check covered only 45 images. Full-test Float32 accuracy is 620/670, so the app reports 92.54%, rather than borrowing Keras accuracy (92.69%) or the old model score (95.41%).

Active accuracy 95% Wilson interval: 90.2959% to 94.2937%. This is sampling uncertainty on this split, not a field-performance guarantee.

## Active Float32 per-class results

| Class | Correct / support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Big_Cat | 6/6 | 85.7143% | 100.0000% | 92.3077% |
| Cat | 12/12 | 92.3077% | 100.0000% | 96.0000% |
| Chicken | 7/7 | 87.5000% | 100.0000% | 93.3333% |
| Cow | 39/39 | 100.0000% | 100.0000% | 100.0000% |
| Deer | 70/72 | 98.5915% | 97.2222% | 97.9021% |
| Dog | 9/12 | 100.0000% | 75.0000% | 85.7143% |
| Elephant | 79/79 | 98.7500% | 100.0000% | 99.3711% |
| Goat | 45/49 | 88.2353% | 91.8367% | 90.0000% |
| Monkey | 28/28 | 100.0000% | 100.0000% | 100.0000% |
| Mouse | 7/7 | 100.0000% | 100.0000% | 100.0000% |
| Non_Venomous_Snake | 112/128 | 90.3226% | 87.5000% | 88.8889% |
| Pig | 13/15 | 76.4706% | 86.6667% | 81.2500% |
| Sheep | 11/17 | 68.7500% | 64.7059% | 66.6667% |
| Venomous_Snake | 127/141 | 88.8112% | 90.0709% | 89.4366% |
| Wild_Boar | 55/58 | 96.4912% | 94.8276% | 95.6522% |

Matrix rows are actual labels and columns are predicted labels. All raw cells and normalized rows are recomputed from argmax predictions. The UI defaults to the active 15-by-15 matrix and labels its 670-image evaluation. Historical 7-by-7 matrices remain available with their separate 545-image scope. Per-class perfect rows are retained when measured, with their sample support; no values are lowered or capped to make results look more plausible.

## Confidence semantics

Active raw mean score: 93.581476%; median: 99.658701%; mean score on errors: 73.570575%. Errors with score at least 99%: 0. Exact floating-point scores of 1.0: 26/670. Fifteen equal-width-bin ECE: 0.02835556. These are descriptive statistics, not calibrated probabilities of correctness.

Scores close to one keep six decimal places; exact one is displayed as `1.0 (saturated score)`. Underlying outputs are preserved. A high-score result is not labeled certain. Calibration would require a separate validation/calibration split and another untouched evaluation; no calibration was fitted on the test set.

## Runtime and dataset changes

- Path decoding uses OpenCV imdecode, BGR-to-RGB conversion, INTER_LINEAR resize to 224 by 224, Float32 pixels in [0,255], and embedded model rescaling. RGB frame inputs use the same resize. No external division by 255 is applied.
- The older Wildlife Gate is bypassed for this classifier because its coverage has not been validated for the expanded domestic-animal classes. The UI explicitly reports this. No Background/Unknown output exists.
- The dataset is copied to `datasets/Anviksa_15Class_Dataset`: 4,692 train, 667 validation, 670 test files, all 6,029 copies SHA-256 verified. These are original source counts; reviewed training exclusions remain documented in the training run. Raw images are excluded from Git. The app bundles the class/split inventory for Pi installations without local dataset images.
- Dataset inventory, current class names, accuracy, and raw-score semantics are shown in the project information pages. Historical detection events are preserved.

## Remaining platform verification

Full export inference was measured on the development Linux CPU runtime. Physical Raspberry Pi OS display, runtime allocation, camera reliability, and performance still need on-device verification. Runtime or delegate differences can affect probabilities and borderline predictions. This report is not a Raspberry Pi benchmark.

Reproduce with `python -B scripts/evaluate_active_model.py --source-run PATH_TO_THE_SUPPLIED_RUN` in an environment with an interpreter and the original frozen test files. The output includes checksum-linked prediction caches and all measured values in `results/deployment/mobilenet_v3_large_15class/evaluation.json`.

## Windows runtime verification

All 670 test images were also hash-verified and evaluated under Windows LiteRT 2.3.0 using built-in CPU kernels: 620/670 correct (92.5373%), with top-1 predictions matching the Linux Float32 reference on all 670 samples. The default Windows XNNPACK delegate exited natively on this machine, so the adapter disables default delegates for Windows LiteRT. Linux and Pi interpreter defaults are retained. Results are recorded in `windows_runtime_check.json` and a separate Windows prediction cache.

## Final application checks

All 35 regression tests passed in the Windows project environment, including all eight sections at Pi-sized windows, 15 output probability rows, active 15-by-15 confusion matrix defaults, score formatting, and OpenCV preprocessing. Real local image analysis and a temporary six-frame video completed successfully; the video produced one eligible stable event in the worker queue without writing synthetic events to detection history. Deployment integrity and metric-consistency checks passed.

## Independent confusion-matrix check and compact UI

A separate scikit-learn calculation using the Windows full-test prediction cache matched every one of the 225 active matrix cells. Cached labels also matched the original test manifest in class order and image order: 670 images, 620 correct, 50 errors. This checks arithmetic and label alignment, not expert relabeling of the source images. Details are in `independent_matrix_check.json`.

The UI now shows one matrix at a time, sizes it to the viewport with a 720-pixel width cap, uses numbered class IDs with a class/support key, and omits zero-value text. Cell hover and class selection show exact errors. The miniature multi-model strip was removed; historical matrices remain in the evaluation selector.

## Three-model historical view

The Show all 3 historical models action displays Custom CNN (275/545 correct), MobileNetV2 (504/545), and historical MobileNetV3 Large (520/545) as three separate responsive charts stacked vertically. Every plotted raw and normalized cell is checked against its corresponding record. The active 15-class export remains a separate evaluation. Twelve desktop UI checks passed.
