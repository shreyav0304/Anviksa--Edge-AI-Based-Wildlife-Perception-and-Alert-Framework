# Anviksa AI Raspberry Pi 4 Benchmark Package

## 1. Purpose

This package measures the real Raspberry Pi 4 efficiency of two already validated MobileNetV3 Large models: Float32 TFLite and Float16 TFLite. It does not train, convert, or modify either model and does not select a winner before measurements exist.

## 2. Raspberry Pi prerequisites

- Raspberry Pi 4 Model B
- 64-bit Raspberry Pi OS is recommended
- Python 3 with `venv` and `pip`
- At least one representative wildlife image in `test_images/`
- Adequate cooling and a stable power supply for repeatable measurements

No camera, GPIO, mmWave sensor, or network connection is required during inference after dependencies have been installed.

## 3. Environment setup

From the copied `pi_deployment` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. Dependency installation

```bash
python -m pip install -r requirements_pi.txt
```

The scripts prefer the lightweight `tflite-runtime` interpreter. Wheel availability depends on Raspberry Pi OS, CPU architecture, and Python version. If the listed runtime has no compatible wheel, install the official compatible LiteRT runtime for that environment; the inference module also supports `ai_edge_litert`. Full TensorFlow is only a final development fallback and should not be installed blindly.

## 5. Verify model integrity

Run this immediately after copying the package:

```bash
python verify_package.py
```

Both model hashes, sizes, required scripts, manifest, and exact seven-class order must pass before benchmarking.

## 6. Add a benchmark image

Place a small representative image set in `test_images/`. The benchmark uses the first supported image unless `--image /path/to/image.jpg` is supplied. Speed benchmarking does not require the full frozen 545-image test set.

Use the same image for all six configurations. Do not compare runs made with different images or materially different system conditions.

## 7. Benchmark Float32

```bash
python benchmark/benchmark_tflite.py --model float32 --threads 1
python benchmark/benchmark_tflite.py --model float32 --threads 2
python benchmark/benchmark_tflite.py --model float32 --threads 4
```

## 8. Benchmark Float16

```bash
python benchmark/benchmark_tflite.py --model float16 --threads 1
python benchmark/benchmark_tflite.py --model float16 --threads 2
python benchmark/benchmark_tflite.py --model float16 --threads 4
```

Each command creates a uniquely named JSON file in `pi_results/`; previous runs are never overwritten. The model-only test uses batch size one, 20 warm-up iterations, and 200 timed iterations. End-to-end testing defaults to 50 iterations and can be changed with `--end-to-end-iterations`, but the same count must be used for all compared configurations.

## 9. Compare all six results

After all six configurations finish:

```bash
python benchmark/compare_pi_results.py
```

The newest run for each representation/thread combination is written to unique comparison CSV and JSON files. The comparison deliberately does not declare a winner automatically.

## 10. Interpreting latency and FPS

Lower latency is better. FPS is approximately `1000 / average latency in milliseconds`, so higher is better. Compare average, median, P95, maximum, and standard deviation: a low average with poor P95 or high variation may be unsuitable for a stable live system. Do not assume four threads will be best.

## 11. Model-only versus end-to-end latency

Model-only latency measures only tensor assignment, interpreter invocation, and output retrieval using an already prepared tensor. End-to-end latency additionally includes reading the image file, RGB conversion, nearest-neighbor resizing, Float32 conversion, batch creation, and inference. Camera capture is not included.

## 12. Temperature interpretation

The script reads `/sys/class/thermal/thermal_zone0/temp` before and after each run. `NOT AVAILABLE` is non-fatal. Allow the Pi to return to a similar starting temperature before each configuration. Rising temperature or throttling can distort comparisons; repeat suspicious runs after cooling.

## 13. Memory interpretation

The script reads current RSS (`VmRSS`) and peak RSS (`VmHWM`) from `/proc/self/status` without an extra monitoring dependency. Compare memory before load, after load, and peak during/after inference. These are process-memory measurements, not total system power consumption.

## 14. Returning results for deployment selection

Send back the six original benchmark JSON files plus the generated comparison CSV and JSON. Record the Pi model, cooling method, power supply, OS/Python versions, and whether any thermal-throttling warning occurred. Final selection will consider classification equivalence, model-only and end-to-end latency, FPS, memory, storage, thermal behavior, and stability. No Raspberry Pi performance result exists until these scripts run on the physical device.

