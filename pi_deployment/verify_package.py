"""Verify the immutable models and required Raspberry Pi package files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_CLASSES = [
    "Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake",
    "Venomous_Snake", "Wild_Boar",
]
EXPECTED_MODELS = {
    "models/mobilenet_v3_large_float32.tflite": (
        "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7",
        12920548,
    ),
    "models/mobilenet_v3_large_float16.tflite": (
        "5b9e7763e605c7e5f8081dcf12c2ff6afd1cb1d7cd0f6f3cda909e07d1e7c4e5",
        6549452,
    ),
}
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
    for relative in REQUIRED_FILES:
        exists = (ROOT / relative).is_file()
        checks.append({"check": f"required file: {relative}", "pass": exists})

    class_path = ROOT / "inference/class_names.json"
    classes = json.loads(class_path.read_text(encoding="utf-8")) if class_path.is_file() else []
    checks.append({"check": "exact seven-class order", "pass": classes == EXPECTED_CLASSES})

    manifest_path = ROOT / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    checks.append({"check": "manifest class order", "pass": manifest.get("class_order") == EXPECTED_CLASSES})
    manifest_models = {item.get("package_path"): item for item in manifest.get("models", [])}
    for relative, (expected_hash, expected_size) in EXPECTED_MODELS.items():
        path = ROOT / relative
        actual_hash = sha256(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
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

