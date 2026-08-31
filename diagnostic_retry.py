"""One-shot diagnostic wrapper for the approved EfficientNetB0 retry."""

from __future__ import annotations

import csv
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from experiments.config import ExperimentConfig, environment_metadata
from experiments.data_pipeline import build_dataset_bundle
from experiments.model_registry import build_model, compile_model


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def memory_status() -> dict[str, int]:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {"total_physical_bytes": status.ullTotalPhys,
            "available_physical_bytes": status.ullAvailPhys,
            "memory_load_percent": status.dwMemoryLoad}


def process_working_set(process_handle: int) -> int:
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process_handle, ctypes.byref(counters), counters.cb
    )
    return int(counters.WorkingSetSize) if ok else -1


def pump(stream, destinations) -> None:
    for line in iter(stream.readline, ""):
        for destination in destinations:
            destination.write(line)
            destination.flush()
    stream.close()


def main() -> int:
    experiment_dir = Path(sys.argv[1]).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=False)
    error_path = experiment_dir / "error_traceback.log"
    try:
        config = ExperimentConfig(model_key="efficientnet_b0")
        datasets = build_dataset_bundle(config, strict_validation=True)
        bundle = build_model("efficientnet_b0", config, weights="imagenet")
        compile_model(bundle.model, config.initial_learning_rate)
        train_x, train_y = next(iter(datasets.train))
        val_x, val_y = next(iter(datasets.validation))
        train_out = bundle.model(train_x, training=False).numpy()
        val_out = bundle.model(val_x, training=False).numpy()
        train_loss = float(tf.reduce_mean(tf.keras.losses.categorical_crossentropy(train_y, train_out)))
        val_loss = float(tf.reduce_mean(tf.keras.losses.categorical_crossentropy(val_y, val_out)))
        disk = shutil.disk_usage(experiment_dir.anchor)
        diagnostic = {
            **environment_metadata(), "platform_detail": platform.platform(),
            **memory_status(), "disk_free_bytes": disk.free,
            "physical_devices": [str(d) for d in tf.config.list_physical_devices()],
            "execution": "CPU-only" if not tf.config.list_physical_devices("GPU") else "GPU available",
            "dataset_fingerprint": config.dataset_fingerprint,
            "manifest_counts": {"train": len(datasets.train_entries), "validation": len(datasets.validation_entries), "test": len(datasets.test_entries)},
            "parameters": bundle.model.count_params(), "batch_size": config.batch_size,
            "input_shape": list(train_x.shape), "output_shape": list(train_out.shape),
            "preprocessing": bundle.spec.preprocessing, "imagenet_weights_loaded": True,
            "train_output_finite": bool(np.isfinite(train_out).all()),
            "validation_output_finite": bool(np.isfinite(val_out).all()),
            "train_probability_sums_valid": bool(np.allclose(train_out.sum(axis=1), 1.0, atol=1e-4)),
            "validation_probability_sums_valid": bool(np.allclose(val_out.sum(axis=1), 1.0, atol=1e-4)),
            "train_batch_loss": train_loss, "validation_batch_loss": val_loss,
            "weights_updated_during_check": False,
        }
        (experiment_dir / "system_resources_before_training.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")

        env = os.environ.copy()
        env["ANVIKSA_EXPERIMENT_DIR"] = str(experiment_dir)
        command = [sys.executable, "-u", "train.py", "--model", "efficientnet_b0"]
        with (experiment_dir / "console_output.log").open("w", encoding="utf-8", buffering=1) as console, error_path.open("w", encoding="utf-8", buffering=1) as errors, (experiment_dir / "resource_usage.csv").open("w", newline="", encoding="utf-8", buffering=1) as resource_file:
            process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            threads = [threading.Thread(target=pump, args=(process.stdout, (console,)), daemon=True), threading.Thread(target=pump, args=(process.stderr, (console, errors)), daemon=True)]
            [thread.start() for thread in threads]
            writer = csv.DictWriter(resource_file, fieldnames=("timestamp_utc", "process_working_set_bytes", "system_available_ram_bytes", "system_memory_load_percent", "disk_free_bytes"))
            writer.writeheader()
            while process.poll() is None:
                mem = memory_status()
                writer.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "process_working_set_bytes": process_working_set(process._handle), "system_available_ram_bytes": mem["available_physical_bytes"], "system_memory_load_percent": mem["memory_load_percent"], "disk_free_bytes": shutil.disk_usage(experiment_dir.anchor).free})
                resource_file.flush()
                time.sleep(10)
            [thread.join() for thread in threads]
            exit_code = process.returncode
        (experiment_dir / "retry_status.json").write_text(json.dumps({"exit_code": exit_code, "completed": exit_code == 0, "ended_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
        return exit_code
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
