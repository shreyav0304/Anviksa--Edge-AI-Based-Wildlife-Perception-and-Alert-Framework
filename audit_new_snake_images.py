#!/usr/bin/env python3
"""Read-only audit of the proposed Indian snake image dataset.

The only writes performed are CSV/Markdown audit artifacts below
results/new_snake_image_audit. Source datasets, manifests, models, UI, and
video assets are opened read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
EXPECTED = {
    "Non-Venomous": {
        "Banded Racer", "Checkered Keelback", "Common Rat Snake",
        "Common Sand Boa", "Common Trinket", "Green Tree Vine",
        "Indian Rock Python",
    },
    "Venomous": {
        "Common Krait", "King Cobra", "Monocled Cobra", "Russell's Viper",
        "Saw-scaled Viper", "Spectacled Cobra",
    },
}
PARENT_MODEL = {
    "Non-Venomous": "Non_Venomous_Snake",
    "Venomous": "Venomous_Snake",
}
EXPECTED_CLEAN_FINGERPRINT = "b103634bb6a4a01785cb7c7e3b176223325052da47228dba687edaf43a84c594"
EXPECTED_COMBINED_FINGERPRINT = "502af418ea92ba2a40190208c809180d4ba73ec2597c124bcba5cd6f11a28641"
EXPECTED_TFLITE = {
    "pi_deployment/models/mobilenet_v3_large_float32.tflite": "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7",
    "pi_deployment/models/mobilenet_v3_large_float16.tflite": "5b9e7763e605c7e5f8081dcf12c2ff6afd1cb1d7cd0f6f3cda909e07d1e7c4e5",
}


def dct_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    factor = math.pi / (2.0 * size)
    for row in range(size):
        scale = math.sqrt(1.0 / size) if row == 0 else math.sqrt(2.0 / size)
        for column in range(size):
            matrix[row, column] = scale * math.cos((2 * column + 1) * row * factor)
    return matrix


DCT_32 = dct_matrix(32)


def bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:0{math.ceil(bits.size / 4)}x}"


def perceptual_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    transformed = DCT_32 @ pixels @ DCT_32.T
    low = transformed[:8, :8]
    threshold = float(np.median(low.reshape(-1)[1:]))
    return bits_to_hex(low > threshold)


def difference_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a

    def named_groups(self, prefix: str) -> dict[str, str]:
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        result: dict[str, str] = {}
        index = 0
        for members in sorted(groups.values(), key=lambda x: sorted(x)[0].lower()):
            if len(members) < 2:
                continue
            index += 1
            for member in members:
                result[member] = f"{prefix}{index:04d}"
        return result


def confidence(pd: int, dd: int) -> str | None:
    if pd <= 2 and dd <= 2:
        return "HIGH_CONFIDENCE_NEAR_DUPLICATE"
    if pd <= 6 and dd <= 8:
        return "POSSIBLE_NEAR_DUPLICATE"
    if pd <= 10 and dd <= 4:
        return "MANUAL_REVIEW"
    return None


def protected_state(root: Path) -> dict:
    state: dict[str, object] = {}
    clean_fp = root / "dataset_integrity_reports/clean/dataset_fingerprint.sha256"
    combined_fp = root / "dataset_integrity_reports/original/dataset_fingerprint.sha256"
    state["clean_dataset_fingerprint"] = clean_fp.read_text(encoding="utf-8").strip() if clean_fp.exists() else "MISSING"
    state["combined_dataset_fingerprint"] = combined_fp.read_text(encoding="utf-8").strip() if combined_fp.exists() else "MISSING"
    state["clean_fingerprint_matches_reference"] = state["clean_dataset_fingerprint"] == EXPECTED_CLEAN_FINGERPRINT
    state["combined_fingerprint_matches_reference"] = state["combined_dataset_fingerprint"] == EXPECTED_COMBINED_FINGERPRINT
    for relative, expected in EXPECTED_TFLITE.items():
        path = root / relative
        actual = sha256_file(path) if path.exists() else "MISSING"
        state[relative] = {"sha256": actual, "matches_reference": actual == expected}
    protected_roots = [
        "clean_dataset", "combined_dataset", "dataset_integrity_reports", "models",
        "experiments", "pi_deployment", "results/comparison",
        "results/model_comparison_dashboard", "ui", "snake_video_dataset",
    ]
    snapshots = {}
    for relative in protected_roots:
        base = root / relative
        entries = []
        if base.exists():
            for path in sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.as_posix().lower()):
                stat = path.stat()
                entries.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
        snapshots[relative] = entries
    state["protected_file_metadata"] = snapshots
    return state


def load_existing_manifest(root: Path) -> list[dict]:
    result = []
    manifest_dir = root / "dataset_integrity_reports/clean"
    for split in ("train", "validation", "test"):
        path = manifest_dir / f"{split}_manifest.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["split"] = split.upper()
                result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "new_training_data/snake_images/Indian-Snakes-Dataset-master"
    output = root / "results/new_snake_image_audit"
    if not source.is_dir():
        raise SystemExit(f"Dataset not found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    before = protected_state(root)

    all_paths = sorted(source.rglob("*"), key=lambda p: p.as_posix().lower())
    all_files = [p for p in all_paths if p.is_file()]
    candidate_paths = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]
    non_images = [p for p in all_files if p.suffix.lower() not in IMAGE_EXTENSIONS]
    records: list[dict] = []
    integrity: list[dict] = []
    hidden_files = []
    for path in all_files:
        rel = path.relative_to(source).as_posix()
        if any(part.startswith(".") for part in path.relative_to(source).parts):
            hidden_files.append(rel)

    for number, path in enumerate(candidate_paths, 1):
        rel = path.relative_to(source).as_posix()
        parts = path.relative_to(source).parts
        parent = parts[0] if len(parts) >= 1 else ""
        species = parts[1] if len(parts) >= 2 else ""
        item = {
            "relative_path": rel, "filename": path.name,
            "extension": path.suffix.lower(), "parent_label": parent,
            "species_label": species, "intended_model_class": PARENT_MODEL.get(parent, ""),
            "width": "", "height": "", "channels_or_mode": "",
            "file_size_bytes": path.stat().st_size, "sha256": "",
            "perceptual_hash": "", "dhash": "", "readable": False,
            "exact_duplicate_group": "", "near_duplicate_group": "",
            "source_group_id": "", "existing_dataset_match": False,
            "existing_split_match": "", "label_conflict": False,
            "provenance_status": "NOT VERIFIED", "audit_status": "",
            "review_reason": "", "image_format": "", "aspect_ratio": "",
            "integrity_error": "", "truncated_warning": False,
        }
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    item.update({
                        "width": width, "height": height, "channels_or_mode": image.mode,
                        "image_format": image.format or "UNKNOWN",
                        "aspect_ratio": round(width / height, 6) if height else "",
                        "perceptual_hash": perceptual_hash(image),
                        "dhash": difference_hash(image),
                    })
                item["truncated_warning"] = any("truncat" in str(w.message).lower() for w in caught)
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid dimensions {width}x{height}")
            item["sha256"] = sha256_file(path)
            item["readable"] = True
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            item["integrity_error"] = f"{type(exc).__name__}: {exc}"
        reasons = []
        if not item["readable"]:
            item["audit_status"] = "CORRUPT"
            reasons.append(item["integrity_error"] or "unreadable")
        elif item["file_size_bytes"] == 0:
            item["audit_status"] = "CORRUPT"
            reasons.append("zero-byte image")
        elif int(item["width"]) < 100 or int(item["height"]) < 100 or item["file_size_bytes"] < 5000:
            item["audit_status"] = "QUALITY_REVIEW"
            reasons.append("unusually tiny dimension (<100 px) or file size (<5 KB)")
        elif parent not in EXPECTED or species not in EXPECTED.get(parent, set()) or len(parts) != 3:
            item["audit_status"] = "LABEL_REVIEW"
            reasons.append("unexpected directory-derived label or nesting")
        else:
            item["audit_status"] = "PROVENANCE_REVIEW"
            reasons.append("source provenance and training reuse rights not verified")
        item["review_reason"] = "; ".join(reasons)
        records.append(item)
        integrity.append({k: item[k] for k in [
            "relative_path", "readable", "integrity_error", "truncated_warning", "width",
            "height", "aspect_ratio", "channels_or_mode", "image_format", "file_size_bytes",
            "audit_status", "review_reason",
        ]})
        if number % 250 == 0:
            print(f"Decoded {number}/{len(candidate_paths)} new images", flush=True)

    readable = [r for r in records if r["readable"]]
    by_path = {r["relative_path"]: r for r in records}
    exact_buckets: dict[str, list[dict]] = defaultdict(list)
    for row in readable:
        exact_buckets[row["sha256"]].append(row)
    exact_rows, exact_dsu = [], DSU()
    exact_index = 0
    for sha, members in sorted(exact_buckets.items()):
        if len(members) < 2:
            continue
        exact_index += 1
        group = f"EXACT{exact_index:04d}"
        base = members[0]
        for member in members:
            member["exact_duplicate_group"] = group
            member["audit_status"] = "DUPLICATE_REVIEW"
            member["review_reason"] = "exact duplicate within new dataset"
            exact_dsu.union(base["relative_path"], member["relative_path"])
            exact_rows.append({
                "duplicate_group_id": group, "sha256": sha,
                "relative_path": member["relative_path"], "parent_label": member["parent_label"],
                "species_label": member["species_label"],
                "cross_species": len({x["species_label"] for x in members}) > 1,
                "cross_parent_class": len({x["parent_label"] for x in members}) > 1,
            })

    near_rows, near_dsu = [], DSU()
    for i, first in enumerate(readable):
        for second in readable[i + 1:]:
            if first["sha256"] == second["sha256"]:
                continue
            pd = hash_distance(first["perceptual_hash"], second["perceptual_hash"])
            dd = hash_distance(first["dhash"], second["dhash"])
            level = confidence(pd, dd)
            if not level:
                continue
            near_dsu.union(first["relative_path"], second["relative_path"])
            near_rows.append({
                "first_path": first["relative_path"], "second_path": second["relative_path"],
                "first_parent": first["parent_label"], "second_parent": second["parent_label"],
                "first_species": first["species_label"], "second_species": second["species_label"],
                "phash_distance": pd, "dhash_distance": dd, "classification": level,
                "cross_species": first["species_label"] != second["species_label"],
                "cross_parent_class": first["parent_label"] != second["parent_label"],
            })
    near_groups = near_dsu.named_groups("NEAR")
    for rel, group in near_groups.items():
        by_path[rel]["near_duplicate_group"] = group
        by_path[rel]["source_group_id"] = group
        if by_path[rel]["audit_status"] == "PROVENANCE_REVIEW":
            by_path[rel]["audit_status"] = "DUPLICATE_REVIEW"
            by_path[rel]["review_reason"] = "perceptual near-duplicate candidate within new dataset"
    near_rows.sort(key=lambda r: (r["classification"] != "HIGH_CONFIDENCE_NEAR_DUPLICATE", r["phash_distance"] + r["dhash_distance"], r["first_path"], r["second_path"]))

    existing = load_existing_manifest(root)
    existing_sha: dict[str, list[dict]] = defaultdict(list)
    for row in existing:
        existing_sha[row["sha256"]].append(row)
    existing_rows = []
    for number, new in enumerate(readable, 1):
        exact_matches = existing_sha.get(new["sha256"], [])
        for old in exact_matches:
            leakage = old["split"] == "TEST"
            existing_rows.append({
                "new_relative_path": new["relative_path"], "new_parent": new["parent_label"],
                "new_species": new["species_label"], "match_type": "EXACT",
                "classification": "EXACT_DUPLICATE", "existing_relative_path": old["relative_path"],
                "existing_class": old["class_name"], "existing_split": old["split"],
                "phash_distance": 0, "dhash_distance": 0, "leakage_risk": leakage,
            })
            new["existing_dataset_match"] = True
            new["existing_split_match"] = old["split"]
            new["audit_status"] = "LEAKAGE_RISK" if leakage else "DUPLICATE_REVIEW"
            new["review_reason"] = f"exact match to existing {old['split']} image"
        for old in existing:
            if old in exact_matches:
                continue
            pd = hash_distance(new["perceptual_hash"], old["phash"])
            dd = hash_distance(new["dhash"], old["dhash"])
            level = confidence(pd, dd)
            if not level:
                continue
            leakage = old["split"] == "TEST"
            existing_rows.append({
                "new_relative_path": new["relative_path"], "new_parent": new["parent_label"],
                "new_species": new["species_label"], "match_type": "NEAR",
                "classification": level, "existing_relative_path": old["relative_path"],
                "existing_class": old["class_name"], "existing_split": old["split"],
                "phash_distance": pd, "dhash_distance": dd, "leakage_risk": leakage,
            })
            new["existing_dataset_match"] = True
            if not new["existing_split_match"]:
                new["existing_split_match"] = old["split"]
            if leakage:
                new["audit_status"] = "LEAKAGE_RISK"
                new["review_reason"] = f"near-duplicate candidate matching existing TEST image ({level})"
            elif new["audit_status"] == "PROVENANCE_REVIEW":
                new["audit_status"] = "DUPLICATE_REVIEW"
                new["review_reason"] = f"near-duplicate candidate matching existing {old['split']} image"
        if number % 250 == 0:
            print(f"Compared {number}/{len(readable)} new images to clean manifest", flush=True)

    conflict_rows = []
    for row in exact_rows:
        if row["cross_species"]:
            severity = "CRITICAL" if row["cross_parent_class"] else "REVIEW"
            conflict_rows.append({"severity": severity, "conflict_type": "EXACT", **row})
    for row in near_rows:
        if row["cross_species"]:
            severity = "HIGH" if row["cross_parent_class"] and row["classification"] == "HIGH_CONFIDENCE_NEAR_DUPLICATE" else "REVIEW"
            conflict_rows.append({"severity": severity, "conflict_type": "NEAR", **row})
    for conflict in conflict_rows:
        paths = [conflict.get("relative_path"), conflict.get("first_path"), conflict.get("second_path")]
        for rel in filter(None, paths):
            by_path[rel]["label_conflict"] = True
            if conflict["severity"] in {"CRITICAL", "HIGH"}:
                by_path[rel]["audit_status"] = "LABEL_REVIEW"
                by_path[rel]["review_reason"] = f"{conflict['severity']} cross-label duplicate conflict"

    species_counts = Counter(r["species_label"] for r in records)
    parent_counts = Counter(r["parent_label"] for r in records)
    species_rows = []
    for parent in sorted(EXPECTED):
        for species in sorted(EXPECTED[parent]):
            species_rows.append({"parent_label": parent, "species_label": species, "image_count": species_counts.get(species, 0), "expected": True, "biological_verification": "NOT VERIFIED"})
    actual_species = {(r["parent_label"], r["species_label"]) for r in records}
    for parent, species in sorted(actual_species):
        if species not in EXPECTED.get(parent, set()):
            species_rows.append({"parent_label": parent, "species_label": species, "image_count": species_counts[species], "expected": False, "biological_verification": "NOT VERIFIED"})
    parent_rows = [{"parent_label": p, "intended_model_class": PARENT_MODEL.get(p, ""), "image_count": parent_counts[p]} for p in sorted(parent_counts)]

    widths = [int(r["width"]) for r in readable]
    heights = [int(r["height"]) for r in readable]
    resolutions = Counter(f"{r['width']}x{r['height']}" for r in readable)
    dimension_summary = [
        {"metric": "minimum_width", "value": min(widths) if widths else ""},
        {"metric": "maximum_width", "value": max(widths) if widths else ""},
        {"metric": "median_width", "value": statistics.median(widths) if widths else ""},
        {"metric": "mean_width", "value": round(statistics.mean(widths), 3) if widths else ""},
        {"metric": "minimum_height", "value": min(heights) if heights else ""},
        {"metric": "maximum_height", "value": max(heights) if heights else ""},
        {"metric": "median_height", "value": statistics.median(heights) if heights else ""},
        {"metric": "mean_height", "value": round(statistics.mean(heights), 3) if heights else ""},
        {"metric": "portrait_count", "value": sum(r["height"] > r["width"] for r in readable)},
        {"metric": "landscape_count", "value": sum(r["width"] > r["height"] for r in readable)},
        {"metric": "square_count", "value": sum(r["width"] == r["height"] for r in readable)},
    ]
    for resolution, count in resolutions.most_common(20):
        dimension_summary.append({"metric": f"common_resolution_{resolution}", "value": count})

    nonven = parent_counts.get("Non-Venomous", 0)
    ven = parent_counts.get("Venomous", 0)
    total_labeled = nonven + ven
    positive_species = [v for v in species_counts.values() if v > 0]
    balance_rows = [
        {"metric": "Non-Venomous images", "value": nonven},
        {"metric": "Venomous images", "value": ven},
        {"metric": "Non-Venomous percentage", "value": round(100 * nonven / total_labeled, 3) if total_labeled else ""},
        {"metric": "Venomous percentage", "value": round(100 * ven / total_labeled, 3) if total_labeled else ""},
        {"metric": "largest species", "value": species_counts.most_common(1)[0][0] if species_counts else ""},
        {"metric": "largest species count", "value": max(positive_species) if positive_species else ""},
        {"metric": "smallest species", "value": min(species_counts, key=species_counts.get) if species_counts else ""},
        {"metric": "smallest species count", "value": min(positive_species) if positive_species else ""},
        {"metric": "max/min species ratio", "value": round(max(positive_species) / min(positive_species), 4) if positive_species else ""},
        {"metric": "parent-class imbalance ratio", "value": round(max(nonven, ven) / min(nonven, ven), 4) if min(nonven, ven) else ""},
    ]

    provenance_candidates = [p for p in all_files if p.name.lower().startswith(("readme", "license", "citation", "metadata")) or p.suffix.lower() in {".csv", ".json"}]
    provenance_text = [
        "# Provenance Report", "", "- Dataset/repository directory name: `Indian-Snakes-Dataset-master`",
        f"- Local provenance files found: {len(provenance_candidates)}",
    ]
    provenance_text += [f"  - `{p.relative_to(source).as_posix()}`" for p in provenance_candidates]
    provenance_text += ["", "- SOURCE PROVENANCE: NOT VERIFIED", "- TRAINING REUSE RIGHTS: REVIEW REQUIRED", "", "No authoritative local license or biological ground-truth metadata was established by this audit. Directory labels are metadata only; biologically verified labels are NOT VERIFIED."]
    (output / "provenance_report.md").write_text("\n".join(provenance_text) + "\n", encoding="utf-8")

    manifest_fields = [
        "relative_path", "filename", "extension", "parent_label", "species_label",
        "intended_model_class", "width", "height", "channels_or_mode", "file_size_bytes",
        "sha256", "perceptual_hash", "dhash", "readable", "exact_duplicate_group",
        "near_duplicate_group", "source_group_id", "existing_dataset_match",
        "existing_split_match", "label_conflict", "provenance_status", "audit_status",
        "review_reason", "image_format", "aspect_ratio", "integrity_error", "truncated_warning",
    ]
    csv_write(output / "new_snake_image_manifest.csv", records, manifest_fields)
    csv_write(output / "species_counts.csv", species_rows, ["parent_label", "species_label", "image_count", "expected", "biological_verification"])
    csv_write(output / "parent_class_counts.csv", parent_rows, ["parent_label", "intended_model_class", "image_count"])
    csv_write(output / "image_integrity_report.csv", integrity, list(integrity[0]) if integrity else ["relative_path"])
    csv_write(output / "exact_duplicate_report.csv", exact_rows, ["duplicate_group_id", "sha256", "relative_path", "parent_label", "species_label", "cross_species", "cross_parent_class"])
    csv_write(output / "near_duplicate_report.csv", near_rows, ["first_path", "second_path", "first_parent", "second_parent", "first_species", "second_species", "phash_distance", "dhash_distance", "classification", "cross_species", "cross_parent_class"])
    csv_write(output / "existing_dataset_duplicate_report.csv", existing_rows, ["new_relative_path", "new_parent", "new_species", "match_type", "classification", "existing_relative_path", "existing_class", "existing_split", "phash_distance", "dhash_distance", "leakage_risk"])
    conflict_fields = ["severity", "conflict_type", "duplicate_group_id", "sha256", "relative_path", "parent_label", "species_label", "first_path", "second_path", "first_parent", "second_parent", "first_species", "second_species", "phash_distance", "dhash_distance", "classification", "cross_species", "cross_parent_class"]
    csv_write(output / "label_conflict_report.csv", conflict_rows, conflict_fields)
    csv_write(output / "image_dimension_summary.csv", dimension_summary, ["metric", "value"])
    csv_write(output / "dataset_balance_report.csv", balance_rows, ["metric", "value"])
    manual = [r for r in records if r["audit_status"] != "ELIGIBLE_CANDIDATE"]
    csv_write(output / "manual_review_required.csv", manual, manifest_fields)

    unexpected_folders = []
    expected_dirs = {source / parent for parent in EXPECTED} | {source / parent / species for parent, species_set in EXPECTED.items() for species in species_set}
    for p in all_paths:
        if p.is_dir() and p not in expected_dirs:
            unexpected_folders.append(p.relative_to(source).as_posix())
    missing_species = [f"{p}/{s}" for p, ss in EXPECTED.items() for s in ss if not (source / p / s).is_dir()]
    unexpected_species = [f"{p}/{s}" for p, s in actual_species if s not in EXPECTED.get(p, set())]
    file_sizes = [p.stat().st_size for p in all_files]
    after = protected_state(root)
    protected_unchanged = before == after
    state_for_file = {"before": before, "after": after, "unchanged": protected_unchanged}
    (output / "protected_path_integrity.json").write_text(json.dumps(state_for_file, indent=2), encoding="utf-8")

    exact_groups = len({r["duplicate_group_id"] for r in exact_rows})
    new_near_pairs = len(near_rows)
    existing_exact = sum(r["match_type"] == "EXACT" for r in existing_rows)
    existing_near = sum(r["match_type"] == "NEAR" for r in existing_rows)
    test_leaks = len({r["new_relative_path"] for r in existing_rows if r["leakage_risk"]})
    quality = sum(r["audit_status"] == "QUALITY_REVIEW" for r in records)
    eligible = sum(r["audit_status"] == "ELIGIBLE_CANDIDATE" for r in records)
    cross_parent_conflicts = sum(bool(r.get("cross_parent_class")) for r in conflict_rows)
    summary = [
        "# New Snake Image Dataset Audit", "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}", "",
        "## Filesystem inventory", "",
        f"- Total files: {len(all_files)}", f"- Total candidate image files: {len(candidate_paths)}",
        f"- Non-image files: {len(non_images)} ({', '.join(p.relative_to(source).as_posix() for p in non_images) or 'none'})",
        f"- Parent categories found: {len({r['parent_label'] for r in records})}",
        f"- Species folders represented: {len(species_counts)}",
        f"- Extensions: {dict(sorted(Counter(p.suffix.lower() for p in all_files).items()))}",
        f"- File size bytes (min/median/mean/max): {min(file_sizes) if file_sizes else 0}/{statistics.median(file_sizes) if file_sizes else 0}/{round(statistics.mean(file_sizes), 2) if file_sizes else 0}/{max(file_sizes) if file_sizes else 0}",
        f"- Zero-byte files: {sum(p.stat().st_size == 0 for p in all_files)}",
        f"- Hidden files: {hidden_files or 'none'}", f"- Unexpected folders: {unexpected_folders or 'none'}",
        "", "## Label validation", "",
        f"- Missing expected species folders: {missing_species or 'none'}",
        f"- Unexpected species folders: {unexpected_species or 'none'}",
        "- Directory labels: audited against the approved mapping.",
        "- Biological verification: NOT VERIFIED.", "",
        "## Required final report", "",
        f"1. NEW IMAGES FOUND: {len(candidate_paths)}",
        f"2. READABLE IMAGES: {len(readable)}",
        f"3. CORRUPT/UNREADABLE: {len(records) - len(readable)}",
        f"4. VENOMOUS DIRECTORY-LABELED IMAGES: {ven}",
        f"5. NON-VENOMOUS DIRECTORY-LABELED IMAGES: {nonven}",
        f"6. SPECIES COUNT: {len(species_counts)}",
        f"7. EXACT DUPLICATES WITHIN NEW DATA: {exact_groups} groups / {len(exact_rows)} file entries",
        f"8. NEAR-DUPLICATE CANDIDATES WITHIN NEW DATA: {new_near_pairs} pairs",
        f"9. EXACT DUPLICATES AGAINST EXISTING DATA: {existing_exact} match records",
        f"10. NEAR-DUPLICATE CANDIDATES AGAINST EXISTING DATA: {existing_near} match records",
        f"11. FROZEN-TEST LEAKAGE RISKS: {test_leaks} new images",
        f"12. CROSS-CLASS LABEL CONFLICTS: {cross_parent_conflicts}",
        f"13. QUALITY-REVIEW IMAGES: {quality}",
        "14. PROVENANCE STATUS: NOT VERIFIED",
        "15. TRAINING REUSE RIGHTS STATUS: REVIEW REQUIRED",
        f"16. ELIGIBLE CANDIDATES: {eligible}",
        f"17. MANUAL-REVIEW REQUIRED: {len(manual)}",
        f"18. DATASET BALANCE: Venomous {ven} ({round(100*ven/total_labeled, 2) if total_labeled else 0}%), Non-Venomous {nonven} ({round(100*nonven/total_labeled, 2) if total_labeled else 0}%); parent ratio {round(max(ven, nonven)/min(ven, nonven), 3) if min(ven, nonven) else 'N/A'}",
        f"19. EXISTING DATASET MODIFIED: {'NO' if protected_unchanged else 'YES'}",
        f"20. EXISTING TEST SET MODIFIED: {'NO' if protected_unchanged else 'YES'}",
        f"21. MODELS MODIFIED: {'NO' if protected_unchanged else 'YES'}",
        f"22. TFLITE MODELS MODIFIED: {'NO' if protected_unchanged else 'YES'}",
        f"23. UI MODIFIED: {'NO' if protected_unchanged else 'YES'}",
        f"24. VIDEO DATA MODIFIED: {'NO' if protected_unchanged else 'YES'}",
        "25. TRAINING PERFORMED: NO", "",
        f"- Clean dataset fingerprint matches reference: {before['clean_fingerprint_matches_reference']}",
        f"- Combined dataset fingerprint matches reference: {before['combined_fingerprint_matches_reference']}",
        f"- Protected paths unchanged during audit: {protected_unchanged}", "",
        "NEW SNAKE IMAGE AUDIT: COMPLETE", "",
        "SAFE FOR CONTROLLED INTEGRATION: NEEDS MANUAL REVIEW",
    ]
    (output / "audit_summary.md").write_text("\n".join(map(str, summary)) + "\n", encoding="utf-8")
    print("Audit complete")
    print(f"Readable: {len(readable)}/{len(candidate_paths)}; within-new exact groups: {exact_groups}; within-new near pairs: {new_near_pairs}")
    print(f"Existing exact records: {existing_exact}; existing near records: {existing_near}; test-risk images: {test_leaks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
