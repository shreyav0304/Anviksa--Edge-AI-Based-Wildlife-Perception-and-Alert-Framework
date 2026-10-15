"""Verify the immutable models and required Raspberry Pi package files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_CLASSES = ['Big_Cat', 'Cat', 'Chicken', 'Cow', 'Deer', 'Dog', 'Elephant', 'Goat', 'Monkey', 'Mouse', 'Non_Venomous_Snake', 'Pig', 'Sheep', 'Venomous_Snake', 'Wild_Boar']
EXPECTED_MODELS = {'models/anviksa_15class_float32.tflite': ('36494a90660ac550e1e6db753861a8024d14eefd2c23e229d955f70673b6585a', 11976156), 'models/anviksa_15class_float16.tflite': ('46109f6b5702fd4927d94fd6a5688dce172f48f9ce05e20a735af093e722bf6c', 6066664)}
REQUIRED_FILES = [
    "benchmark/benchmark_tflite.py",
    "benchmark/compare_pi_results.py",
    "inference/tflite_inference.py",
    "inference/class_names.json",
    "requirements_pi.txt",
    "model_manifest.json",
    "README_PI.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks = []
    def read_object(relative, expected_type):
        path = ROOT / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, expected_type):
                raise ValueError(f"Expected {expected_type.__name__}")
            return value
        except (OSError, ValueError) as exc:
            checks.append({"check": f"readable JSON: {relative}", "pass": False, "error": str(exc)})
            return expected_type()
    for relative in REQUIRED_FILES:
        exists = (ROOT / relative).is_file()
        checks.append({"check": f"required file: {relative}", "pass": exists})

    class_path = ROOT / "inference/class_names.json"
    classes = read_object("inference/class_names.json", list)
    checks.append({"check": "exact deployed 15-class order", "pass": classes == EXPECTED_CLASSES})

    manifest_path = ROOT / "model_manifest.json"
    manifest = read_object("model_manifest.json", dict)
    checks.append({"check": "manifest class order", "pass": manifest.get("class_order") == EXPECTED_CLASSES})
    model_entries = manifest.get("models", [])
    if not isinstance(model_entries, list) or any(not isinstance(item, dict) for item in model_entries):
        checks.append({"check": "manifest model entries", "pass": False})
        model_entries = []
    manifest_models = {item.get("package_path"): item for item in model_entries if isinstance(item.get("package_path"), str)}
    for relative, (expected_hash, expected_size) in EXPECTED_MODELS.items():
        path = ROOT / relative
        try:
            actual_hash = sha256(path) if path.is_file() else None
            actual_size = path.stat().st_size if path.is_file() else None
        except OSError as exc:
            checks.append({"check": f"readable model: {relative}", "pass": False, "error": str(exc)})
            actual_hash = actual_size = None
        item = manifest_models.get(relative, {})
        checks.extend([
            {"check": f"model exists: {relative}", "pass": path.is_file()},
            {"check": f"model SHA-256: {relative}", "pass": actual_hash == expected_hash, "actual": actual_hash},
            {"check": f"model size: {relative}", "pass": actual_size == expected_size, "actual": actual_size},
            {"check": f"manifest hash/size: {relative}", "pass": item.get("sha256") == expected_hash and item.get("file_size_bytes") == expected_size},
        ])

    passed = all(item["pass"] for item in checks)
    print(json.dumps({"package": str(ROOT), "pass": passed, "checks": checks}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

