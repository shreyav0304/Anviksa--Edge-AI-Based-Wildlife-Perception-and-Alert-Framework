"""Audit web and user snake videos without extracting or deleting frames."""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "source_videos"
WEB_MANIFEST = ROOT / "metadata/source_manifest.csv"
GLOBAL_MANIFEST = ROOT / "metadata/global_video_manifest.csv"
REPORTS = ROOT / "reports"
DUPLICATES = REPORTS / "global_video_duplicate_report.csv"
MANUAL = REPORTS / "global_manual_review_required.csv"
SUMMARY = REPORTS / "global_duplicate_summary.md"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".ogv"}
USER_EVIDENCE = {
    "3D-Methods-Deinagkistrodon-acutus_STV1747-0001-HR.mp4": {
        "common": "Hundred-pace viper", "scientific": "Deinagkistrodon acutus",
        "source": "https://movie.biologists.com/video/10.1242/jeb.250347/video-1",
    },
    "Deinagkistrodon-acutus_STV1747-0001_2A-label-HR.mp4": {
        "common": "Hundred-pace viper", "scientific": "Deinagkistrodon acutus",
        "source": "https://movie.biologists.com/video/10.1242/jeb.250347/video-1",
    },
    "Aspidelaps-lubricus_STV1808-0007-3A-label-HR.mp4": {
        "common": "Cape coral snake", "scientific": "Aspidelaps lubricus",
        "source": "https://movie.biologists.com/video/10.1242/jeb.250347/video-4",
    },
    "Bitis-nasicornis_STV1757-0027-3A-label-HR.mp4": {
        "common": "Rhinoceros viper", "scientific": "Bitis nasicornis",
        "source": "https://doi.org/10.26180/29624891.v2",
    },
    "Crotalus-atrox_STV1637-0021-5A-label-HR.mp4": {
        "common": "Western diamondback rattlesnake", "scientific": "Crotalus atrox",
        "source": "https://doi.org/10.26180/29624891.v2",
    },
    "Echis-ocellatus_STV1545-0028-3A-label-HR.mp4": {
        "common": "West African carpet viper", "scientific": "Echis ocellatus",
        "source": "https://doi.org/10.26180/29624891.v2",
    },
    "Macrovipera-lebetina_STV1608-0010-1A-label-HR.mp4": {
        "common": "Blunt-nosed viper", "scientific": "Macrovipera lebetina",
        "source": "https://doi.org/10.26180/29624891.v2",
    },
    "Macrovipera-lebetina_STV1608-0010-7A-label-HR.mp4": {
        "common": "Blunt-nosed viper", "scientific": "Macrovipera lebetina",
        "source": "https://doi.org/10.26180/29624891.v2",
    },
    "Toxicodryas-pulverulenta_STV1810-0001-2A-label-HR.mp4": {
        "common": "Fischer's cat snake", "scientific": "Toxicodryas pulverulenta",
        "source": "https://doi.org/10.26180/29624891.v2",
    },
    "Vipera-ammodytes_STV1447-0133-3A-label-HR.mp4": {
        "common": "Nose-horned viper", "scientific": "Vipera ammodytes",
        "source": "https://doi.org/10.26180/29624891.v2",
    },
}
MANIFEST_FIELDS = (
    "video_id", "source_type", "relative_path", "filename", "file_extension",
    "file_size_bytes", "sha256", "duration_seconds", "fps", "frame_count",
    "width", "height", "opencv_readable", "current_class",
    "verification_status", "species_common_name", "species_scientific_name",
    "species_verification_source", "venom_status_verification_source",
    "license_verification_status", "eligible_for_dataset", "exclusion_reason",
    "source_group_id", "sampled_frames", "sample_decode_failures", "notes",
)
PAIR_FIELDS = (
    "video_a", "video_b", "source_type_a", "source_type_b", "sha256_a",
    "sha256_b", "duration_a", "duration_b", "duplicate_type",
    "similarity_score", "decision", "canonical_video", "excluded_video",
    "source_group_id", "reason", "manual_review_required",
)
MANUAL_FIELDS = (
    "video_id", "source_type", "relative_path", "filename", "current_class",
    "reason", "verification_evidence", "recommended_action", "review_status",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprints(gray: np.ndarray) -> tuple[int, int]:
    dh = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dbits = dh[:, 1:] > dh[:, :-1]
    dvalue = sum(int(bit) << i for i, bit in enumerate(dbits.flat))
    ph = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(ph)[:8, :8]
    pbits = low > float(np.median(low.flat[1:]))
    pvalue = sum(int(bit) << i for i, bit in enumerate(pbits.flat))
    return dvalue, pvalue


def frame_similarity(a: tuple[int, int], b: tuple[int, int]) -> float:
    bits = (a[0] ^ b[0]).bit_count() + (a[1] ^ b[1]).bit_count()
    return 1.0 - bits / 128.0


def inspect(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"readable": False, "reason": "UNREADABLE", "fingerprints": []}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = count / fps if fps > 0 and count > 0 else 0.0
    if min(fps, count, width, height, duration) <= 0:
        capture.release()
        return {"readable": False, "reason": "CORRUPT", "fingerprints": []}
    sample_count = min(180, max(3, int(math.ceil(duration)) + 1))
    times = np.linspace(0.0, max(0.0, duration - 1.0 / fps), sample_count)
    values = []
    failures = 0
    for timestamp in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            failures += 1
            continue
        values.append(fingerprints(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    capture.release()
    readable = bool(values) and failures <= max(2, sample_count // 4)
    return {
        "readable": readable, "reason": "" if readable else "UNREADABLE",
        "fps": fps, "frame_count": count, "width": width, "height": height,
        "duration": duration, "fingerprints": values, "failures": failures,
        "samples": sample_count,
    }


def compare(a: dict, b: dict) -> tuple[float, float, float]:
    left, right = a["_fingerprints"], b["_fingerprints"]
    count = min(len(left), len(right), 48)
    if not count:
        return 0.0, 0.0, 0.0
    li = np.linspace(0, len(left) - 1, count).round().astype(int)
    ri = np.linspace(0, len(right) - 1, count).round().astype(int)
    aligned = float(np.mean([frame_similarity(left[x], right[y]) for x, y in zip(li, ri, strict=True)]))
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    best_run = 0
    best_matches = 0
    for offset in range(-len(short) + 1, len(long)):
        run = matches = 0
        for index, value in enumerate(short):
            other = index + offset
            matched = 0 <= other < len(long) and frame_similarity(value, long[other]) >= 0.84
            if matched:
                matches += 1
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        best_matches = max(best_matches, matches)
    return aligned, best_run / len(short), best_matches / len(short)


def canonical(a: dict, b: dict) -> tuple[dict, dict]:
    def rank(row: dict) -> tuple:
        return (
            row["verification_status"] == "VERIFIED",
            row["license_verification_status"] == "VERIFIED",
            row["source_type"] == "web_collected",
            int(row["width"]) * int(row["height"]),
            float(row["duration_seconds"]),
            row["video_id"],
        )
    return (a, b) if rank(a) >= rank(b) else (b, a)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with WEB_MANIFEST.open(newline="", encoding="utf-8") as file:
        web_source = {row["local_filename"]: row for row in csv.DictReader(file) if row["local_filename"]}
    paths = sorted(
        (path for path in VIDEO_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.relative_to(VIDEO_ROOT).as_posix().casefold(),
    )
    rows = []
    web_number = user_number = 0
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        parts = path.relative_to(VIDEO_ROOT).parts
        source_type = parts[0]
        current_class = parts[1] if len(parts) > 2 else "Unverified"
        if source_type == "web_collected":
            web_number += 1
            video_id = web_source.get(path.name, {}).get("source_id") or f"WEB_GLOBAL_{web_number:03d}"
            source = web_source.get(path.name, {})
            verification = source.get("label_verification_status", "UNVERIFIED")
            license_status = source.get("license_verification_status", "UNVERIFIED")
        else:
            user_number += 1
            video_id = f"USER_{user_number:03d}"
            source = {}
            verification = "UNVERIFIED"
            license_status = "UNVERIFIED"
            evidence = USER_EVIDENCE.get(path.name)
            if evidence:
                verification = "VERIFIED"
                current_class = "Venomous_Snake"
                license_status = "REVIEW"
        info = inspect(path) if path.stat().st_size > 0 else {"readable": False, "reason": "CORRUPT", "fingerprints": []}
        readable = info["readable"]
        eligibility = "YES" if readable and verification == "VERIFIED" and license_status == "VERIFIED" else "REVIEW"
        exclusion = ""
        if not readable:
            eligibility, exclusion = "NO", info["reason"]
        elif verification != "VERIFIED":
            eligibility, exclusion = "NO", "LABEL_NOT_VERIFIED"
        elif license_status != "VERIFIED":
            exclusion = "LICENSE_NOT_VERIFIED"
        note = ""
        if source_type == "user_collected" and "STV" in path.name:
            note = "Filename resembles scientific-video identifiers; provenance and reuse rights require confirmation."
        if source_type == "user_collected" and evidence:
            note = "Exact scientific-video ID matched a Journal of Experimental Biology supplement; reuse rights for this local copy still require confirmation."
        row = {
            "video_id": video_id, "source_type": source_type,
            "relative_path": relative, "filename": path.name,
            "file_extension": path.suffix.lower(), "file_size_bytes": str(path.stat().st_size),
            "sha256": digest(path), "duration_seconds": f"{info.get('duration', 0):.6f}",
            "fps": f"{info.get('fps', 0):.6f}", "frame_count": str(info.get("frame_count", 0)),
            "width": str(info.get("width", 0)), "height": str(info.get("height", 0)),
            "opencv_readable": "YES" if readable else "NO", "current_class": current_class,
            "verification_status": verification,
            "species_common_name": source.get("species_common_name", "") or (evidence["common"] if source_type == "user_collected" and evidence else ""),
            "species_scientific_name": source.get("species_scientific_name", "") or (evidence["scientific"] if source_type == "user_collected" and evidence else ""),
            "species_verification_source": source.get("species_verification_source", "") or (evidence["source"] if source_type == "user_collected" and evidence else ""),
            "venom_status_verification_source": source.get("venom_status_verification_source", "") or (evidence["source"] if source_type == "user_collected" and evidence else ""),
            "license_verification_status": license_status,
            "eligible_for_dataset": eligibility, "exclusion_reason": exclusion,
            "source_group_id": "", "sampled_frames": str(len(info["fingerprints"])),
            "sample_decode_failures": str(info.get("failures", 0)), "notes": note,
            "_fingerprints": info["fingerprints"],
        }
        rows.append(row)

    parent = {row["video_id"]: row["video_id"] for row in rows}
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
    for a, b in itertools.combinations(rows, 2):
        pair_kind = f"{a['source_type']} vs {b['source_type']}"
        exact = a["sha256"] == b["sha256"]
        aligned, run, coverage = compare(a, b)
        duration_ratio = min(float(a["duration_seconds"]), float(b["duration_seconds"])) / max(float(a["duration_seconds"]), float(b["duration_seconds"]), 1e-9)
        filename_score = SequenceMatcher(None, a["filename"].casefold(), b["filename"].casefold()).ratio()
        metadata_candidate = duration_ratio >= 0.97 and (
            a["width"] == b["width"] or a["frame_count"] == b["frame_count"] or filename_score >= 0.75
        )
        if exact:
            duplicate_type, review = "EXACT_DUPLICATE", "NO"
        elif aligned >= 0.91 and duration_ratio >= 0.80 and coverage >= 0.65:
            duplicate_type, review = "HIGH_CONFIDENCE_NEAR_DUPLICATE", "NO"
        elif run >= 0.40 and min(len(a["_fingerprints"]), len(b["_fingerprints"])) >= 5:
            duplicate_type, review = "POSSIBLE_PARTIAL_DUPLICATE", "YES"
        elif aligned >= 0.80 or (metadata_candidate and aligned >= 0.72):
            duplicate_type, review = "MANUAL_REVIEW_REQUIRED", "YES"
        else:
            duplicate_type, review = "UNIQUE", "NO"
        if duplicate_type in {"EXACT_DUPLICATE", "HIGH_CONFIDENCE_NEAR_DUPLICATE", "POSSIBLE_PARTIAL_DUPLICATE"}:
            union(a["video_id"], b["video_id"])
        keep_name = drop_name = ""
        decision = "KEEP_BOTH"
        if duplicate_type in {"EXACT_DUPLICATE", "HIGH_CONFIDENCE_NEAR_DUPLICATE"}:
            keep, drop = canonical(a, b)
            keep_name, drop_name = keep["relative_path"], drop["relative_path"]
            drop["eligible_for_dataset"] = "NO"
            drop["exclusion_reason"] = "EXACT_DUPLICATE" if exact else "NEAR_DUPLICATE"
            decision = "KEEP_CANONICAL_NON_DESTRUCTIVE"
        elif duplicate_type in {"POSSIBLE_PARTIAL_DUPLICATE", "MANUAL_REVIEW_REQUIRED"}:
            if a["eligible_for_dataset"] != "NO":
                a["eligible_for_dataset"] = "REVIEW"
            if b["eligible_for_dataset"] != "NO":
                b["eligible_for_dataset"] = "REVIEW"
            decision = "MANUAL_REVIEW"
        pair_rows.append({
            "video_a": a["relative_path"], "video_b": b["relative_path"],
            "source_type_a": a["source_type"], "source_type_b": b["source_type"],
            "sha256_a": a["sha256"], "sha256_b": b["sha256"],
            "duration_a": a["duration_seconds"], "duration_b": b["duration_seconds"],
            "duplicate_type": duplicate_type, "similarity_score": f"{max(aligned, run, coverage):.6f}",
            "decision": decision, "canonical_video": keep_name, "excluded_video": drop_name,
            "source_group_id": "", "reason": (
                f"pair={pair_kind}; aligned={aligned:.6f}; consecutive_run={run:.6f}; "
                f"coverage={coverage:.6f}; duration_ratio={duration_ratio:.6f}; "
                f"filename_similarity={filename_score:.6f}; metadata_candidate={metadata_candidate}"
            ), "manual_review_required": review,
        })

    group_names = {}
    for row in rows:
        root = find(row["video_id"])
        if root not in group_names:
            group_names[root] = f"snake_source_group_{len(group_names) + 1:03d}"
        row["source_group_id"] = group_names[root]
    by_path = {row["relative_path"]: row["source_group_id"] for row in rows}
    for pair in pair_rows:
        if pair["duplicate_type"] != "UNIQUE":
            pair["source_group_id"] = by_path[pair["video_a"]]

    manual_rows = []
    for row in rows:
        reasons = []
        if row["opencv_readable"] != "YES": reasons.append(row["exclusion_reason"])
        if row["verification_status"] != "VERIFIED": reasons.append("LABEL_NOT_VERIFIED")
        if row["license_verification_status"] != "VERIFIED": reasons.append("LICENSE_NOT_VERIFIED")
        if reasons:
            evidence = row["notes"] or "No reliable species/source evidence recorded."
            manual_rows.append({
                "video_id": row["video_id"], "source_type": row["source_type"],
                "relative_path": row["relative_path"], "filename": row["filename"],
                "current_class": row["current_class"], "reason": "|".join(dict.fromkeys(reasons)),
                "verification_evidence": evidence,
                "recommended_action": "Confirm provenance, species identity, venom status, and reuse rights; do not guess from appearance.",
                "review_status": "PENDING",
            })
    for pair in pair_rows:
        if pair["manual_review_required"] == "YES":
            manual_rows.append({
                "video_id": "PAIR", "source_type": f"{pair['source_type_a']}|{pair['source_type_b']}",
                "relative_path": f"{pair['video_a']} | {pair['video_b']}",
                "filename": "", "current_class": "", "reason": pair["duplicate_type"],
                "verification_evidence": pair["reason"],
                "recommended_action": "Review suspected related footage and retain the shared source_group_id.",
                "review_status": "PENDING",
            })

    for row in rows:
        row.pop("_fingerprints", None)
    write_csv(GLOBAL_MANIFEST, MANIFEST_FIELDS, rows)
    write_csv(DUPLICATES, PAIR_FIELDS, pair_rows)
    write_csv(MANUAL, MANUAL_FIELDS, manual_rows)

    pair_counts = Counter((pair["source_type_a"], pair["source_type_b"], pair["duplicate_type"]) for pair in pair_rows)
    duplicate_counts = Counter(pair["duplicate_type"] for pair in pair_rows)
    readable = sum(row["opencv_readable"] == "YES" for row in rows)
    web = [row for row in rows if row["source_type"] == "web_collected"]
    user = [row for row in rows if row["source_type"] == "user_collected"]
    summary = [
        "# Global Snake Video Duplicate Summary", "",
        "Representative frames were decoded in memory at approximately one-second intervals. No frames were written.", "",
        f"- Total web videos: {len(web)}", f"- Total user videos: {len(user)}",
        f"- Total audited: {len(rows)}", f"- Readable: {readable}",
        f"- Corrupt/unreadable: {len(rows) - readable}", "",
        "## Web vs web", "",
        f"- Exact: {pair_counts[('web_collected', 'web_collected', 'EXACT_DUPLICATE')]}",
        f"- High-confidence near: {pair_counts[('web_collected', 'web_collected', 'HIGH_CONFIDENCE_NEAR_DUPLICATE')]}",
        f"- Possible partial: {pair_counts[('web_collected', 'web_collected', 'POSSIBLE_PARTIAL_DUPLICATE')]}", "",
        "## User vs user", "",
        f"- Exact: {pair_counts[('user_collected', 'user_collected', 'EXACT_DUPLICATE')]}",
        f"- High-confidence near: {pair_counts[('user_collected', 'user_collected', 'HIGH_CONFIDENCE_NEAR_DUPLICATE')]}",
        f"- Possible partial: {pair_counts[('user_collected', 'user_collected', 'POSSIBLE_PARTIAL_DUPLICATE')]}", "",
        "## Web vs user", "",
        f"- Exact: {pair_counts[('web_collected', 'user_collected', 'EXACT_DUPLICATE')]}",
        f"- High-confidence near: {pair_counts[('web_collected', 'user_collected', 'HIGH_CONFIDENCE_NEAR_DUPLICATE')]}",
        f"- Possible partial: {pair_counts[('web_collected', 'user_collected', 'POSSIBLE_PARTIAL_DUPLICATE')]}", "",
        f"- All high-confidence near-duplicate pairs: {duplicate_counts['HIGH_CONFIDENCE_NEAR_DUPLICATE']}",
        f"- All possible partial-duplicate pairs: {duplicate_counts['POSSIBLE_PARTIAL_DUPLICATE']}",
        f"- Unique source groups: {len(set(row['source_group_id'] for row in rows))}", "",
        "## Mandatory future leakage rule", "",
        "All frames derived from a single `source_group_id` must later remain in exactly one of train, validation, or test. Related or reposted videos must never be separated across splits.", "",
        "No splitting, frame extraction, dataset merge, or training was performed.",
    ]
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
