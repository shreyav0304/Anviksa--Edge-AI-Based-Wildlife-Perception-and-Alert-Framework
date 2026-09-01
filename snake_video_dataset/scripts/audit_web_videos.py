"""Validate and non-destructively audit web videos for duplicate content.

Representative frames are decoded in memory only. No frames are written.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata/source_manifest.csv"
VIDEO_ROOT = ROOT / "source_videos/web_collected"
REPORTS = ROOT / "reports"
DUPLICATE_REPORT = REPORTS / "web_video_duplicate_report.csv"
MANUAL_REPORT = REPORTS / "manual_review_required.csv"
SUMMARY = REPORTS / "web_duplicate_summary.md"
COLLECTION = REPORTS / "video_collection_report.md"
PAIR_FIELDS = (
    "video_a", "video_b", "sha256_a", "sha256_b", "duration_a",
    "duration_b", "duplicate_type", "similarity_score", "decision",
    "canonical_video", "excluded_video", "source_group_id", "reason",
    "manual_review_required",
)
MANUAL_FIELDS = (
    "source_id", "video_title", "source_url", "assigned_class", "reason",
    "review_status", "notes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return sum(int(value) << index for index, value in enumerate(bits.flat))


def phash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    median = float(np.median(low.flat[1:]))
    bits = low > median
    return sum(int(value) << index for index, value in enumerate(bits.flat))


def distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    d = (a[0] ^ b[0]).bit_count() + (a[1] ^ b[1]).bit_count()
    return 1.0 - d / 128.0


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"valid": False, "reason": "CORRUPT: VideoCapture failed", "fingerprints": []}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0 or duration <= 0:
        capture.release()
        return {"valid": False, "reason": "CORRUPT: invalid video metadata", "fingerprints": []}
    sample_count = min(120, max(3, int(math.ceil(duration / 2.0)) + 1))
    times = np.linspace(0.0, max(0.0, duration - 1.0 / fps), sample_count)
    fingerprints = []
    failed = 0
    for timestamp in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            failed += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fingerprints.append((dhash(gray), phash(gray)))
    capture.release()
    valid = bool(fingerprints) and failed <= max(2, sample_count // 4)
    return {
        "valid": valid,
        "reason": "" if valid else "CORRUPT: sampled frames could not be decoded",
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration,
        "fingerprints": fingerprints,
        "sample_failures": failed,
    }


def compare_sequences(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> tuple[float, float]:
    count = min(len(a), len(b), 32)
    if count == 0:
        return 0.0, 0.0
    ia = np.linspace(0, len(a) - 1, count).round().astype(int)
    ib = np.linspace(0, len(b) - 1, count).round().astype(int)
    aligned = float(np.mean([distance(a[x], b[y]) for x, y in zip(ia, ib, strict=True)]))
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    best_run = 0
    for offset in range(-len(short) + 1, len(long)):
        run = 0
        for index, value in enumerate(short):
            other = index + offset
            if 0 <= other < len(long) and distance(value, long[other]) >= 0.84:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
    return aligned, best_run / max(1, len(short))


def choose_canonical(a: dict, b: dict) -> tuple[dict, dict]:
    def rank(row: dict) -> tuple:
        info = row["_info"]
        return (
            row["label_verification_status"] == "VERIFIED",
            row["license_verification_status"] == "VERIFIED",
            info.get("width", 0) * info.get("height", 0),
            info.get("duration", 0),
            row["source_id"],
        )
    return (a, b) if rank(a) >= rank(b) else (b, a)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        manifest_fields = list(rows[0])

    downloaded = []
    manual = []
    for row in rows:
        if row["download_status"] != "DOWNLOADED":
            if row["label_verification_status"] != "VERIFIED" or row["license_verification_status"] != "VERIFIED":
                manual.append({
                    "source_id": row["source_id"], "video_title": row["video_title"],
                    "source_url": row["source_url"], "assigned_class": row["assigned_class"],
                    "reason": row["exclusion_reason"] or "VERIFICATION_REQUIRED",
                    "review_status": "PENDING", "notes": row["notes"],
                })
            continue
        path = VIDEO_ROOT / row["assigned_class"] / row["local_filename"]
        if not path.is_file() or path.stat().st_size <= 0:
            row["download_status"] = "CORRUPT"
            row["eligible_for_dataset"] = "NO"
            row["exclusion_reason"] = "CORRUPT"
            continue
        row["sha256"] = sha256(path)
        row["file_size_bytes"] = str(path.stat().st_size)
        info = inspect_video(path)
        row["_info"] = info
        if not info["valid"]:
            row["download_status"] = "CORRUPT"
            row["eligible_for_dataset"] = "NO"
            row["exclusion_reason"] = "CORRUPT"
            manual.append({
                "source_id": row["source_id"], "video_title": row["video_title"],
                "source_url": row["source_url"], "assigned_class": row["assigned_class"],
                "reason": info["reason"], "review_status": "PENDING", "notes": "",
            })
            continue
        row["duration_seconds"] = f"{info['duration']:.6f}"
        row["resolution"] = f"{info['width']}x{info['height']}"
        row["fps"] = f"{info['fps']:.6f}"
        row["frame_count"] = str(info["frames"])
        row["eligible_for_dataset"] = "YES"
        downloaded.append(row)

    parent = {row["source_id"]: row["source_id"] for row in downloaded}
    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pair_rows = []
    for a, b in itertools.combinations(downloaded, 2):
        exact = a["sha256"] == b["sha256"]
        aligned, partial = compare_sequences(a["_info"]["fingerprints"], b["_info"]["fingerprints"])
        duration_ratio = min(a["_info"]["duration"], b["_info"]["duration"]) / max(a["_info"]["duration"], b["_info"]["duration"])
        if exact:
            category, review = "EXACT_DUPLICATE", "NO"
        elif aligned >= 0.92 and duration_ratio >= 0.80:
            category, review = "HIGH_CONFIDENCE_NEAR_DUPLICATE", "NO"
        elif partial >= 0.50 and min(len(a["_info"]["fingerprints"]), len(b["_info"]["fingerprints"])) >= 5:
            category, review = "POSSIBLE_PARTIAL_DUPLICATE", "YES"
        elif aligned >= 0.78:
            category, review = "MANUAL_REVIEW_REQUIRED", "YES"
        else:
            category, review = "UNIQUE", "NO"
        if category in {"EXACT_DUPLICATE", "HIGH_CONFIDENCE_NEAR_DUPLICATE", "POSSIBLE_PARTIAL_DUPLICATE"}:
            union(a["source_id"], b["source_id"])
        canonical = excluded = ""
        decision = "KEEP_BOTH"
        if category in {"EXACT_DUPLICATE", "HIGH_CONFIDENCE_NEAR_DUPLICATE"}:
            keep, drop = choose_canonical(a, b)
            canonical, excluded = keep["local_filename"], drop["local_filename"]
            drop["eligible_for_dataset"] = "NO"
            drop["exclusion_reason"] = category
            decision = "KEEP_CANONICAL_NON_DESTRUCTIVE"
        elif category in {"POSSIBLE_PARTIAL_DUPLICATE", "MANUAL_REVIEW_REQUIRED"}:
            decision = "MANUAL_REVIEW"
            manual.append({
                "source_id": f"{a['source_id']}|{b['source_id']}",
                "video_title": f"{a['video_title']} | {b['video_title']}",
                "source_url": f"{a['source_url']} | {b['source_url']}",
                "assigned_class": f"{a['assigned_class']}|{b['assigned_class']}",
                "reason": category, "review_status": "PENDING",
                "notes": f"aligned={aligned:.6f}; partial={partial:.6f}",
            })
        pair_rows.append({
            "video_a": a["local_filename"], "video_b": b["local_filename"],
            "sha256_a": a["sha256"], "sha256_b": b["sha256"],
            "duration_a": a["duration_seconds"], "duration_b": b["duration_seconds"],
            "duplicate_type": category, "similarity_score": f"{max(aligned, partial):.6f}",
            "decision": decision, "canonical_video": canonical, "excluded_video": excluded,
            "source_group_id": "", "reason": f"aligned={aligned:.6f}; partial={partial:.6f}",
            "manual_review_required": review,
        })

    groups = {}
    for row in downloaded:
        root = find(row["source_id"])
        if root not in groups:
            groups[root] = f"snake_source_group_{len(groups) + 1:03d}"
        row["source_group_id"] = groups[root]
    filename_group = {row["local_filename"]: row["source_group_id"] for row in downloaded}
    for pair in pair_rows:
        if pair["duplicate_type"] != "UNIQUE":
            pair["source_group_id"] = filename_group[pair["video_a"]]

    for row in rows:
        row.pop("_info", None)
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(rows)
    write_csv(DUPLICATE_REPORT, PAIR_FIELDS, pair_rows)
    write_csv(MANUAL_REPORT, MANUAL_FIELDS, manual)

    dup_counts = Counter(pair["duplicate_type"] for pair in pair_rows)
    verified = [row for row in rows if row["download_status"] == "DOWNLOADED" and row["label_verification_status"] == "VERIFIED"]
    eligible = [row for row in verified if row["eligible_for_dataset"] == "YES"]
    by_class = defaultdict(list)
    for row in verified:
        by_class[row["assigned_class"]].append(row)
    summary = [
        "# Web Video Duplicate Summary", "",
        "Representative frames were decoded only in memory at approximately one sample every two seconds; no frames were extracted or saved.", "",
        f"- Downloaded videos audited: {len(downloaded)}",
        f"- Exact duplicate pairs: {dup_counts['EXACT_DUPLICATE']}",
        f"- High-confidence near-duplicate pairs: {dup_counts['HIGH_CONFIDENCE_NEAR_DUPLICATE']}",
        f"- Possible partial-duplicate pairs: {dup_counts['POSSIBLE_PARTIAL_DUPLICATE']}",
        f"- Manual-review duplicate pairs: {dup_counts['MANUAL_REVIEW_REQUIRED']}",
        f"- Unique pair comparisons: {dup_counts['UNIQUE']}",
        f"- Unique dataset-eligible videos: {len(eligible)}", "",
        "Source files were not deleted. Videos with related footage share a `source_group_id`.",
    ]
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")

    species = {name: Counter(row["species_common_name"] for row in items) for name, items in by_class.items()}
    duration = {name: sum(float(row["duration_seconds"]) for row in items) for name, items in by_class.items()}
    skipped = [row for row in rows if row["download_status"].startswith("SKIPPED")]
    corrupt = [row for row in rows if row["download_status"] == "CORRUPT"]
    report = [
        "# Anviksa Snake Web Video Collection Report", "",
        "## Collection totals", "",
        f"1. Candidate sources found: {len(rows)}",
        f"2. Videos downloaded: {len(verified)}",
        f"3. Videos rejected/skipped: {len(skipped)}",
        f"4. Verified venomous videos: {len(by_class['Venomous_Snake'])}",
        f"5. Verified non-venomous videos: {len(by_class['Non_Venomous_Snake'])}",
        f"6. Unverified videos: {sum(row['assigned_class'] == 'Unverified' for row in rows)}",
        "7. Species distribution: " + "; ".join(f"{key}: {dict(value)}" for key, value in species.items()),
        "8. Platform/source distribution: Wikimedia Commons only",
        f"9. Total video duration: {sum(duration.values()):.3f} seconds",
        "10. Duration per class: " + "; ".join(f"{key}: {value:.3f}s" for key, value in duration.items()),
        f"11. Exact duplicate pairs: {dup_counts['EXACT_DUPLICATE']}",
        f"12. High-confidence near-duplicate pairs: {dup_counts['HIGH_CONFIDENCE_NEAR_DUPLICATE']}",
        f"13. Possible partial duplicates: {dup_counts['POSSIBLE_PARTIAL_DUPLICATE']}",
        f"14. Corrupt videos: {len(corrupt)}",
        "15. License summary: " + str(dict(Counter(row["license"] for row in verified))),
        "16. Label-verification summary: " + str(dict(Counter(row["label_verification_status"] for row in rows))),
        f"17. Manual-review count: {len(manual)}",
        f"18. Unique eligible videos: {len(eligible)}",
        "19. Species diversity: " + "; ".join(f"{key}: {len(value)} species" for key, value in species.items()),
        "20. Limitations: small initial web collection; Indian priority species are underrepresented; user videos and a global duplicate audit are still pending.", "",
        "## Stop condition", "",
        "No frame files were extracted. No train/validation/test split was created. No model training was performed.",
        "Wait for user-collected videos, then run a new global web-vs-web, user-vs-user, and web-vs-user duplicate audit before frame extraction.",
    ]
    COLLECTION.write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
