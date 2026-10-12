"""Read-only health checks for a complete dashboard clone; no model execution."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> int:
    checks = []
    def check(label, function):
        try:
            detail = function()
        except Exception as exc:
            checks.append({"check": label, "pass": False, "detail": str(exc)})
        else:
            checks.append({"check": label, "pass": True, "detail": detail})

    def comparison():
        from ui.services.comparison_data import load_comparison_records
        return [record.model + ": " + record.status for record in load_comparison_records()]
    def matrices():
        from ui.services.confusion_data import load_confusion_records
        return {name: {"total": record.total, "correct": record.correct}
                for name, record in load_confusion_records().items()}
    def overview():
        from ui.data import load_dashboard_data
        return load_dashboard_data().selected_model
    def model_files():
        from pi_deployment.verify_package import EXPECTED_MODELS
        paths = dict(EXPECTED_MODELS)
        paths["../wildlife_gate/results/gate_v1/deployment/wildlife_gate_v1_float32.tflite"] = (
            "0ac3c6c0dfca692dc6c6d421f60479082e3338936d09af09301bb8c6cf61d4ca", None
        )
        for relative, (expected, size) in paths.items():
            path = ROOT / "pi_deployment" / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected or (size is not None and path.stat().st_size != size):
                raise ValueError(f"Model integrity mismatch: {path}")
        return "Float32, Float16, and Wildlife Gate hashes verified"

    check("Model comparison", comparison)
    check("Confusion matrices", matrices)
    check("Overview", overview)
    check("Deployment models", model_files)
    passed = all(item["pass"] for item in checks)
    print(json.dumps({"pass": passed, "checks": checks,
                      "not_tested": ["GUI display", "runtime interpreter", "inference", "camera", "GPIO", "Pi performance"]}, indent=2))
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
