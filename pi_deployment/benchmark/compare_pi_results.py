"""Compare completed Raspberry Pi Float32/Float16 benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {(representation, threads) for representation in ("float32", "float16") for threads in (1, 2, 4)}


def load_latest(results_dir: Path) -> dict:
    found = {}
    for path in results_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = (payload["representation"], int(payload["thread_count"]))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if key in EXPECTED and (key not in found or path.stat().st_mtime > found[key][0].stat().st_mtime):
            found[key] = (path, payload)
    missing = EXPECTED - set(found)
    if missing:
        missing_text = ", ".join(f"{name}/{threads}-thread" for name, threads in sorted(missing))
        raise RuntimeError(f"Six configurations are required. Missing: {missing_text}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(PACKAGE_ROOT / "pi_results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()
    selected = load_latest(results_dir)
    rows = []
    for key in sorted(EXPECTED):
        path, item = selected[key]
        memory = item["memory"]
        rows.append({
            "representation": item["representation"],
            "thread_count": item["thread_count"],
            "model_size_bytes": item["model_size_bytes"],
            "average_latency_ms": item["model_only"]["average_latency_ms"],
            "median_latency_ms": item["model_only"]["median_latency_ms"],
            "p95_latency_ms": item["model_only"]["p95_latency_ms"],
            "model_only_fps": item["model_only"]["approximate_fps"],
            "end_to_end_average_latency_ms": item["end_to_end"]["average_latency_ms"],
            "end_to_end_fps": item["end_to_end"]["approximate_fps"],
            "rss_before_load_mib": memory["before_model_load"]["rss_mib"],
            "rss_after_load_mib": memory["after_model_load"]["rss_mib"],
            "peak_rss_mib": memory["during_or_after_inference"]["peak_rss_mib"],
            "temperature_change_c": item["temperature_c"]["change"],
            "stable": item["stability"]["top1_predictions_consistent"] and item["stability"]["completed_all_iterations"],
            "source_result": str(path),
        })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"pi_comparison_{stamp}_{uuid.uuid4().hex[:8]}"
    csv_path = results_dir / f"{stem}.csv"
    json_path = results_dir / f"{stem}.json"
    with csv_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configurations": rows,
        "selection": "NOT AUTOMATICALLY DETERMINED",
        "selection_rule": [
            "classification equivalence", "model-only latency", "end-to-end latency",
            "FPS", "memory", "storage", "thermal behavior", "stability",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), **payload}, indent=2))


if __name__ == "__main__":
    main()

