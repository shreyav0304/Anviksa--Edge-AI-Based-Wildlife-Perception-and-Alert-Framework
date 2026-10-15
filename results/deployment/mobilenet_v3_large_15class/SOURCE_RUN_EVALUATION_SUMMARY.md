# Anviksa final test evaluation

Selected checkpoint: `anviksa_15class_best.keras`. All 670 original test SHA-256 hashes verified. Class order and built-in MobileNetV3 rescaling verified against frozen training code. OpenCV BGR-to-RGB, bilinear 224x224, float32 0..255; inference mode without augmentation.

Accuracy: 92.6866% (621/670 correct). Unweighted sparse categorical cross-entropy: 0.225272.

| Average | Precision | Recall | F1 |
|---|---:|---:|---:|
| macro | 91.5171% | 92.5693% | 91.8177% |
| weighted | 92.7743% | 92.6866% | 92.6646% |

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Big_Cat | 85.71% | 100.00% | 92.31% | 6 |
| Cat | 92.31% | 100.00% | 96.00% | 12 |
| Chicken | 87.50% | 100.00% | 93.33% | 7 |
| Cow | 100.00% | 100.00% | 100.00% | 39 |
| Deer | 98.59% | 97.22% | 97.90% | 72 |
| Dog | 100.00% | 75.00% | 85.71% | 12 |
| Elephant | 98.75% | 100.00% | 99.37% | 79 |
| Goat | 88.24% | 91.84% | 90.00% | 49 |
| Monkey | 100.00% | 100.00% | 100.00% | 28 |
| Mouse | 100.00% | 100.00% | 100.00% | 7 |
| Non_Venomous_Snake | 91.06% | 87.50% | 89.24% | 128 |
| Pig | 76.47% | 86.67% | 81.25% | 15 |
| Sheep | 68.75% | 64.71% | 66.67% | 17 |
| Venomous_Snake | 88.89% | 90.78% | 89.82% | 141 |
| Wild_Boar | 96.49% | 94.83% | 95.65% | 58 |

Venomous snakes: 128/141 correct; 10 predicted non-venomous, 3 predicted other classes (Cat, Deer, Goat: one each). Non-venomous snakes: 112/128 correct; 16 predicted venomous.

Both TFLite exports matched Keras top-1 on all 45 sampled images (three per true class). Maximum absolute probability differences: float32 0.00432247; float16 0.02086541. This sample agreement does not establish full-test TFLite accuracy.

Sheep recall is the lowest at 64.71% (11/17). Several classes have small test supports, so their estimates are based on few images.

No retraining, threshold tuning, model reselection, test modifications, or deployment occurred. Previous 7-class model and Raspberry Pi deployment files were unchanged. Reports, predictions, confusion matrix and logs are in this run folder.

Workflow completion is verified by final_evaluation.json; PowerShell rendered TensorFlow informational stderr as NativeCommandError. No Python traceback or evaluation error occurred.
