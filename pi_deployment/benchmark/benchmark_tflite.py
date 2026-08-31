"""Benchmark either validated TFLite representation on a Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from inference.tflite_inference import TFLiteWildlifeClassifier, preprocess_image


MODELS = {
    "float32": PACKAGE_ROOT / "models/mobilenet_v3_large_float32.tflite",
    "float16": PACKAGE_ROOT / "models/mobilenet_v3_large_float16.tflite",
}
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


def memory_mib() -> dict:
    values = {"rss_mib": None, "peak_rss_mib": None}
    status = read_text(Path("/proc/self/status"))
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                values["rss_mib"] = float(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                values["peak_rss_mib"] = float(line.split()[1]) / 1024.0
    return values


def cpu_temperature_c():
    raw = read_text(Path("/sys/class/thermal/thermal_zone0/temp"))
    if raw is None:
        return "NOT AVAILABLE"
    try:
        value = float(raw)
        return value / 1000.0 if value > 1000 else value
    except ValueError:
        return "NOT AVAILABLE"


def available_ram_mib():
    meminfo = read_text(Path("/proc/meminfo"))
    if not meminfo:
        return "NOT AVAILABLE"
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    return "NOT AVAILABLE"


def system_information(interpreter_source: str) -> dict:
    return {
        "raspberry_pi_model": read_text(Path("/proc/device-tree/model")) or "NOT AVAILABLE",
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "interpreter_source": interpreter_source,
        "cpu_count": os.cpu_count(),
        "available_ram_mib": available_ram_mib(),
        "kernel_version": platform.release(),
    }


def summarize_ms(timings: list[float]) -> dict:
    ordered = np.asarray(timings, dtype=np.float64)
    average = statistics.fmean(timings)
    return {
        "average_latency_ms": average,
        "median_latency_ms": statistics.median(timings),
        "minimum_latency_ms": min(timings),
        "maximum_latency_ms": max(timings),
        "standard_deviation_ms": statistics.pstdev(timings),
        "p95_latency_ms": float(np.percentile(ordered, 95)),
        "approximate_fps": 1000.0 / average if average > 0 else 0.0,
    }


def choose_image(requested):
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(
        path for path in (PACKAGE_ROOT / "test_images").iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(
            "Place at least one benchmark image in pi_deployment/test_images/ "
            "or pass --image PATH."
        )
    return candidates[0]


def reserve_output(output_dir: Path, representation: str, threads: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{representation}_{threads}thread_{stamp}_{uuid.uuid4().hex[:8]}.json"
    if path.exists():
        raise FileExistsError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--image", help="Image path; otherwise the first test_images file is used.")
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "pi_results"))
    parser.add_argument("--end-to-end-iterations", type=int, default=50)
    args = parser.parse_args()
    if args.end_to_end_iterations < 1:
        parser.error("--end-to-end-iterations must be positive")

    model_path = MODELS[args.model]
    image_path = choose_image(args.image)
    rss_before = memory_mib()
    temperature_before = cpu_temperature_c()
    classifier = TFLiteWildlifeClassifier(model_path, num_threads=args.threads)
    rss_after_load = memory_mib()
    batch = preprocess_image(image_path)

    for _ in range(20):
        classifier.invoke_preprocessed(batch)
    model_timings = []
    predictions = []
    for _ in range(200):
        probabilities, latency = classifier.invoke_preprocessed(batch)
        model_timings.append(latency)
        predictions.append(int(np.argmax(probabilities)))

    # Each end-to-end iteration starts from the path so file decoding, RGB
    # conversion, resizing, allocation, and inference are all included.
    end_to_end_timings = []
    for _ in range(args.end_to_end_iterations):
        started = perf_counter()
        classifier.predict(image_path)
        end_to_end_timings.append((perf_counter() - started) * 1000.0)

    rss_during = memory_mib()
    temperature_after = cpu_temperature_c()
    end_average = statistics.fmean(end_to_end_timings)
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "representation": args.model,
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "thread_count": args.threads,
        "batch_size": 1,
        "benchmark_image": str(image_path),
        "model_only": {
            "warmup_iterations": 20,
            "timed_iterations": 200,
            **summarize_ms(model_timings),
        },
        "end_to_end": {
            "timed_iterations": args.end_to_end_iterations,
            "average_latency_ms": end_average,
            "median_latency_ms": statistics.median(end_to_end_timings),
            "approximate_fps": 1000.0 / end_average if end_average > 0 else 0.0,
            "includes": "image file load, RGB conversion, nearest-neighbor resize, Float32 conversion, batch dimension, and inference",
        },
        "memory": {
            "before_model_load": rss_before,
            "after_model_load": rss_after_load,
            "during_or_after_inference": rss_during,
        },
        "temperature_c": {
            "before": temperature_before,
            "after": temperature_after,
            "change": (
                temperature_after - temperature_before
                if isinstance(temperature_before, float) and isinstance(temperature_after, float)
                else "NOT AVAILABLE"
            ),
        },
        "stability": {
            "top1_predictions_consistent": len(set(predictions)) == 1,
            "unique_top1_indices": sorted(set(predictions)),
            "completed_all_iterations": len(model_timings) == 200,
        },
        "system": system_information(classifier.interpreter_source),
        "selection_note": "No representation is selected by this script.",
    }
    output = reserve_output(Path(args.output_dir), args.model, args.threads)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(output), **result}, indent=2))


if __name__ == "__main__":
    main()

