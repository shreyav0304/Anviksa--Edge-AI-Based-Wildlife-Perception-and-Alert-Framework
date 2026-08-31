"""Run one prevalidated experiment with persistent logs and resource samples."""

from __future__ import annotations

import csv
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path


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


def memory_status():
    value = MEMORYSTATUSEX(); value.dwLength = ctypes.sizeof(value)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return value


def process_memory(handle):
    value = PROCESS_MEMORY_COUNTERS(); value.cb = ctypes.sizeof(value)
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb)
    return value


def pump(stream, files):
    for line in iter(stream.readline, ""):
        for file in files:
            file.write(line); file.flush()


def main() -> int:
    directory = Path(sys.argv[1]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env["ANVIKSA_EXPERIMENT_DIR"] = str(directory)
    command = [sys.executable, "-u", "train.py", "--model", "efficientnet_b0", "--batch-size", "16"]
    with (directory / "console_output.log").open("w", encoding="utf-8", buffering=1) as console, (directory / "error_traceback.log").open("w", encoding="utf-8", buffering=1) as errors, (directory / "resource_usage.csv").open("w", newline="", encoding="utf-8", buffering=1) as resources:
        process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        threads = [threading.Thread(target=pump, args=(process.stdout, (console,)), daemon=True), threading.Thread(target=pump, args=(process.stderr, (console, errors)), daemon=True)]
        [thread.start() for thread in threads]
        fields = ("timestamp_utc", "working_set_bytes", "peak_working_set_bytes", "available_ram_bytes", "memory_load_percent", "disk_free_bytes")
        writer = csv.DictWriter(resources, fieldnames=fields); writer.writeheader()
        while process.poll() is None:
            mem, proc = memory_status(), process_memory(process._handle)
            writer.writerow(dict(zip(fields, (datetime.now(timezone.utc).isoformat(), proc.WorkingSetSize, proc.PeakWorkingSetSize, mem.ullAvailPhys, mem.dwMemoryLoad, shutil.disk_usage(directory.anchor).free))))
            resources.flush(); time.sleep(10)
        [thread.join() for thread in threads]
    (directory / "resource_adjustment.json").write_text(json.dumps({"selected_batch_size": 16, "batch_16_diagnostic": "PASS", "batch_8_diagnostic": "NOT REQUIRED", "reason": "Batch size 32 caused a confirmed TensorFlow CPU ResourceExhaustedError during the diagnostic forward pass.", "exit_code": process.returncode}, indent=2) + "\n", encoding="utf-8")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
