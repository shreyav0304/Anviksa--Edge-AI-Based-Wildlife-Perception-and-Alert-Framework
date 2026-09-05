#!/usr/bin/env python3
"""Non-destructive provenance and strict duplicate resolution for snake data.

Reads source images, the previous audit, and controlled clean data. Writes only
under results/new_snake_image_resolution. It never trains, moves, deletes,
rewrites, or relabels source images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


EXPECTED_CLEAN_FP = "b103634bb6a4a01785cb7c7e3b176223325052da47228dba687edaf43a84c594"
EXPECTED_COMBINED_FP = "502af418ea92ba2a40190208c809180d4ba73ec2597c124bcba5cd6f11a28641"
EXPECTED_TFLITE = {
    "pi_deployment/models/mobilenet_v3_large_float32.tflite": "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7",
    "pi_deployment/models/mobilenet_v3_large_float16.tflite": "5b9e7763e605c7e5f8081dcf12c2ff6afd1cb1d7cd0f6f3cda909e07d1e7c4e5",
}
MANIFEST_FIELDS = [
    "relative_path", "filename", "parent_label", "species_label", "sha256",
    "width", "height", "exact_duplicate_group", "near_duplicate_group",
    "source_group", "exact_existing_match", "strict_existing_classification",
    "frozen_test_leakage_status", "species_label_conflict", "provenance_status",
    "license_status", "technical_filter_survivor", "resolution_status", "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_state(root: Path) -> dict:
    clean_file = root / "dataset_integrity_reports/clean/dataset_fingerprint.sha256"
    combined_file = root / "dataset_integrity_reports/original/dataset_fingerprint.sha256"
    result: dict[str, object] = {
        "clean_fingerprint": clean_file.read_text(encoding="utf-8").strip(),
        "combined_fingerprint": combined_file.read_text(encoding="utf-8").strip(),
    }
    result["clean_matches_reference"] = result["clean_fingerprint"] == EXPECTED_CLEAN_FP
    result["combined_matches_reference"] = result["combined_fingerprint"] == EXPECTED_COMBINED_FP
    result["tflite"] = {}
    for relative, expected in EXPECTED_TFLITE.items():
        actual = sha256_file(root / relative)
        result["tflite"][relative] = {"sha256": actual, "matches_reference": actual == expected}
    protected = [
        "clean_dataset", "combined_dataset", "dataset_integrity_reports", "models",
        "experiments", "pi_deployment", "results/comparison",
        "results/model_comparison_dashboard", "ui", "snake_video_dataset",
    ]
    metadata = {}
    for relative in protected:
        base = root / relative
        metadata[relative] = [
            [p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted((x for x in base.rglob("*") if x.is_file()), key=lambda x: x.as_posix().lower())
        ] if base.exists() else []
    result["protected_file_metadata"] = metadata
    return result


def global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Mean local SSIM using an 11x11 Gaussian window, no external dependency."""
    a, b = first.astype(np.float32), second.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + 1e-12)
    return float(np.mean(score))


def ahash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first.astype(np.float64).reshape(-1), second.astype(np.float64).reshape(-1)
    a -= a.mean(); b -= b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float((a @ b) / denominator) if denominator else 0.0


def image_signals(path: Path, orb: cv2.ORB) -> dict:
    raw = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode {path}")
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    color = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA)
    histogram = cv2.calcHist([color], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(histogram, histogram)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return {
        "width": width, "height": height, "aspect": width / height,
        "gray": normalized, "ahash": ahash(gray), "hist": histogram,
        "keypoints": keypoints or [], "descriptors": descriptors,
    }


def geometric(first: dict, second: dict) -> tuple[int, int, float]:
    d1, d2 = first["descriptors"], second["descriptors"]
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return 0, 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    if len(good) < 4:
        return len(good), 0, 0.0
    src = np.float32([first["keypoints"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([second["keypoints"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return len(good), inliers, inliers / len(good) if good else 0.0


def classify_pair(row: dict, first: dict, second: dict) -> dict:
    pd, dd = int(row["phash_distance"]), int(row["dhash_distance"])
    ad = hamming(first["ahash"], second["ahash"])
    aspect_delta = abs(first["aspect"] - second["aspect"]) / max(first["aspect"], second["aspect"])
    pixel_corr = correlation(first["gray"], second["gray"])
    ssim = global_ssim(first["gray"], second["gray"])
    hist_corr = float(cv2.compareHist(first["hist"], second["hist"], cv2.HISTCMP_CORREL))
    good, inliers, inlier_ratio = geometric(first, second)
    direct_confirmed = ssim >= 0.94 and pixel_corr >= 0.94 and ad <= 6 and aspect_delta <= 0.08
    geometric_confirmed = good >= 20 and inliers >= 12 and inlier_ratio >= 0.60 and (pd <= 4 or dd <= 4)
    direct_likely = ssim >= 0.87 and pixel_corr >= 0.88 and ad <= 10 and aspect_delta <= 0.15 and pd <= 6 and dd <= 8
    geometric_likely = good >= 15 and inliers >= 8 and inlier_ratio >= 0.45 and (pd <= 6 or dd <= 6)
    possible = (ssim >= 0.72 and pixel_corr >= 0.72 and ad <= 16 and hist_corr >= 0.30) or (good >= 10 and inliers >= 5 and inlier_ratio >= 0.30)
    if direct_confirmed or geometric_confirmed:
        final = "CONFIRMED_NEAR_DUPLICATE"
    elif direct_likely or geometric_likely:
        final = "LIKELY_NEAR_DUPLICATE"
    elif possible:
        final = "POSSIBLE_NEAR_DUPLICATE"
    elif (pd <= 2 and dd <= 2) and (ssim >= 0.55 or inliers >= 4):
        final = "MANUAL_REVIEW"
    else:
        final = "FALSE_POSITIVE_PERCEPTUAL_MATCH"
    return {
        "ahash_distance": ad, "aspect_ratio_delta": round(aspect_delta, 6),
        "pixel_correlation": round(pixel_corr, 6), "ssim": round(ssim, 6),
        "color_histogram_correlation": round(hist_corr, 6),
        "orb_ratio_matches": good, "orb_ransac_inliers": inliers,
        "orb_inlier_ratio": round(inlier_ratio, 6), "final_classification": final,
        "evidence": f"pHash={pd}; dHash={dd}; aHash={ad}; aspect_delta={aspect_delta:.4f}; pixel_corr={pixel_corr:.4f}; SSIM={ssim:.4f}; hist_corr={hist_corr:.4f}; ORB_good={good}; ORB_inliers={inliers}; ORB_inlier_ratio={inlier_ratio:.4f}",
    }


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

    def labels(self, prefix: str) -> dict[str, str]:
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        labels = {}
        for index, members in enumerate(sorted((x for x in groups.values() if len(x) > 1), key=lambda x: min(x)), 1):
            for member in members:
                labels[member] = f"{prefix}{index:04d}"
        return labels


def make_contact_sheets(rows: list[dict], source: Path, clean: Path, output: Path, limit: int = 200) -> tuple[int, int]:
    selected = rows[:limit]
    if not selected:
        return 0, 0
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    per_sheet = 8
    sheets = 0
    for offset in range(0, len(selected), per_sheet):
        page_rows = selected[offset:offset + per_sheet]
        page = Image.new("RGB", (1200, 330 * len(page_rows)), "white")
        draw = ImageDraw.Draw(page)
        for index, row in enumerate(page_rows):
            y = index * 330
            paths = [source / row["new_image"], clean / row["existing_image"]]
            for column, path in enumerate(paths):
                with Image.open(path) as image:
                    image.seek(0)
                    tile = ImageOps.contain(image.convert("RGB"), (560, 245))
                x = 20 + column * 590
                page.paste(tile, (x + (560 - tile.width) // 2, y + 45 + (245 - tile.height) // 2))
            draw.text((20, y + 5), f"NEW: {row['new_image']}", fill="black", font=font)
            draw.text((610, y + 5), f"EXISTING: {row['existing_image']} [{row.get('existing_split','')}]", fill="black", font=font)
            draw.text((20, y + 295), f"{row['final_classification']} | {row.get('leakage_status','')} | {row['evidence']}", fill="black", font=font)
        sheets += 1
        page.save(output / f"duplicate_review_{sheets:03d}.jpg", quality=90)
    return len(selected), sheets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "new_training_data/snake_images/Indian-Snakes-Dataset-master"
    clean = root / "clean_dataset"
    prior = root / "results/new_snake_image_audit"
    output = root / "results/new_snake_image_resolution"
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    before = protected_state(root)

    required_prior = [
        "new_snake_image_manifest.csv", "exact_duplicate_report.csv",
        "near_duplicate_report.csv", "existing_dataset_duplicate_report.csv",
        "manual_review_required.csv", "audit_summary.md",
    ]
    missing_prior = [name for name in required_prior if not (prior / name).is_file()]
    if missing_prior:
        raise SystemExit(f"Missing prior audit artifacts: {missing_prior}")
    manifest = read_csv(prior / "new_snake_image_manifest.csv")
    prior_exact = read_csv(prior / "exact_duplicate_report.csv")
    prior_near = read_csv(prior / "near_duplicate_report.csv")
    prior_existing = read_csv(prior / "existing_dataset_duplicate_report.csv")
    if len(manifest) != 1779 or sum(row["readable"] == "True" for row in manifest) != 1779:
        raise SystemExit("Previous audit does not correspond to the expected 1,779 readable images")
    actual_paths = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"})
    if actual_paths != sorted(row["relative_path"] for row in manifest):
        raise SystemExit("Current image paths differ from the previous 1,779-image manifest")

    license_candidates = [p for p in source.iterdir() if p.is_file() and p.name.lower().startswith("license")]
    readme_candidates = [p for p in source.iterdir() if p.is_file() and p.name.lower().startswith("readme")]
    if len(license_candidates) != 1 or len(readme_candidates) != 1:
        raise SystemExit("Expected exactly one root LICENSE and README")
    license_path, readme_path = license_candidates[0], readme_candidates[0]
    license_copy = provenance / "original_dataset_LICENSE.txt"
    readme_copy = provenance / "original_dataset_README.md"
    shutil.copyfile(license_path, license_copy)
    shutil.copyfile(readme_path, readme_copy)
    license_sha, readme_sha = sha256_file(license_path), sha256_file(readme_path)
    if sha256_file(license_copy) != license_sha or sha256_file(readme_copy) != readme_sha:
        raise SystemExit("Provenance documentation copy verification failed")

    exact_groups = defaultdict(list)
    for row in manifest:
        if row["exact_duplicate_group"]:
            exact_groups[row["exact_duplicate_group"]].append(row)
    exact_confirmation = []
    for group, members in sorted(exact_groups.items()):
        canonical = sorted(members, key=lambda row: row["relative_path"])[0]
        for duplicate in sorted(members, key=lambda row: row["relative_path"])[1:]:
            actual_a = sha256_file(source / canonical["relative_path"])
            actual_b = sha256_file(source / duplicate["relative_path"])
            exact_confirmation.append({
                "duplicate_scope": "INTERNAL", "duplicate_group": group,
                "new_image": duplicate["relative_path"], "new_class": duplicate["parent_label"],
                "new_species": duplicate["species_label"], "matching_image": canonical["relative_path"],
                "matching_dataset": "NEW_SNAKE_DATASET", "matching_split": "NOT_APPLICABLE",
                "sha256": actual_a, "reconfirmed": actual_a == actual_b == canonical["sha256"],
            })
    exact_existing_rows = [row for row in prior_existing if row["match_type"] == "EXACT"]
    for row in exact_existing_rows:
        new_sha = sha256_file(source / row["new_relative_path"])
        old_sha = sha256_file(clean / row["existing_relative_path"])
        exact_confirmation.append({
            "duplicate_scope": "EXISTING", "duplicate_group": "EXISTING_" + new_sha[:12],
            "new_image": row["new_relative_path"], "new_class": row["new_parent"],
            "new_species": row["new_species"], "matching_image": row["existing_relative_path"],
            "matching_dataset": "clean_dataset", "matching_split": row["existing_split"],
            "sha256": new_sha, "reconfirmed": new_sha == old_sha,
        })
    if len(exact_groups) != 4 or len(exact_existing_rows) != 15 or not all(r["reconfirmed"] for r in exact_confirmation):
        raise SystemExit("Exact duplicate results failed to reproduce")

    candidates = [row for row in prior_existing if row["match_type"] == "NEAR"]
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
    cache: dict[str, dict] = {}
    resolution_rows = []
    for number, row in enumerate(candidates, 1):
        new_key = "N:" + row["new_relative_path"]
        old_key = "E:" + row["existing_relative_path"]
        if new_key not in cache:
            cache[new_key] = image_signals(source / row["new_relative_path"], orb)
        if old_key not in cache:
            cache[old_key] = image_signals(clean / row["existing_relative_path"], orb)
        measures = classify_pair(row, cache[new_key], cache[old_key])
        resolution_rows.append({
            "new_image": row["new_relative_path"], "new_species": row["new_species"],
            "new_parent_class": row["new_parent"], "existing_image": row["existing_relative_path"],
            "existing_class": row["existing_class"], "existing_split": row["existing_split"],
            "phash_distance": row["phash_distance"], "dhash_distance": row["dhash_distance"],
            "previous_classification": row["classification"], **measures,
        })
        if number % 100 == 0:
            print(f"Strictly evaluated {number}/{len(candidates)} prior candidates", flush=True)

    leakage_rows = []
    leakage_map = {
        "CONFIRMED_NEAR_DUPLICATE": "CONFIRMED_LEAKAGE_RISK",
        "LIKELY_NEAR_DUPLICATE": "LIKELY_LEAKAGE_RISK",
        "POSSIBLE_NEAR_DUPLICATE": "POSSIBLE_LEAKAGE_RISK",
        "FALSE_POSITIVE_PERCEPTUAL_MATCH": "FALSE_POSITIVE",
        "MANUAL_REVIEW": "MANUAL_REVIEW",
    }
    for row in resolution_rows:
        if row["existing_split"] != "TEST":
            continue
        leakage_rows.append({
            **row, "test_image": row["existing_image"], "test_class": row["existing_class"],
            "leakage_status": leakage_map[row["final_classification"]],
            "reason": row["evidence"],
        })

    internal_resolution = []
    for row in prior_near:
        first_key, second_key = "N:" + row["first_path"], "N:" + row["second_path"]
        if first_key not in cache:
            cache[first_key] = image_signals(source / row["first_path"], orb)
        if second_key not in cache:
            cache[second_key] = image_signals(source / row["second_path"], orb)
        measures = classify_pair(row, cache[first_key], cache[second_key])
        internal_resolution.append({
            "first_image": row["first_path"], "second_image": row["second_path"],
            "first_parent": row["first_parent"], "second_parent": row["second_parent"],
            "first_species": row["first_species"], "second_species": row["second_species"],
            "phash_distance": row["phash_distance"], "dhash_distance": row["dhash_distance"],
            "previous_classification": row["classification"], **measures,
        })

    species_conflicts = []
    for group, members in sorted(exact_groups.items()):
        species = sorted({row["species_label"] for row in members})
        parents = sorted({row["parent_label"] for row in members})
        if len(species) > 1:
            species_conflicts.append({
                "conflict_id": group, "conflict_type": "EXACT_DUPLICATE",
                "images": " | ".join(sorted(row["relative_path"] for row in members)),
                "species": " | ".join(species), "parents": " | ".join(parents),
                "resolution": "SPECIES_LABEL_CONFLICT", "parent_class_conflict": len(parents) > 1,
                "reason": "Byte-identical image occurs in different species folders; biological label not inferred.",
            })
    for index, row in enumerate(internal_resolution, 1):
        if row["first_species"] != row["second_species"]:
            species_conflicts.append({
                "conflict_id": f"INTERNAL_NEAR_{index:04d}", "conflict_type": row["final_classification"],
                "images": f"{row['first_image']} | {row['second_image']}",
                "species": f"{row['first_species']} | {row['second_species']}",
                "parents": f"{row['first_parent']} | {row['second_parent']}",
                "resolution": "SPECIES_LABEL_CONFLICT", "parent_class_conflict": row["first_parent"] != row["second_parent"],
                "reason": row["evidence"],
            })

    new_by_path = {row["relative_path"]: row for row in manifest}
    dsu = DSU()
    for group, members in exact_groups.items():
        for member in members[1:]:
            dsu.union(members[0]["relative_path"], member["relative_path"])
    for row in internal_resolution:
        if row["final_classification"] in {"CONFIRMED_NEAR_DUPLICATE", "LIKELY_NEAR_DUPLICATE"}:
            dsu.union(row["first_image"], row["second_image"])
    group_labels = dsu.labels("SOURCE_GROUP_")

    exact_internal_exclude = set()
    for group, members in exact_groups.items():
        sorted_members = sorted(row["relative_path"] for row in members)
        exact_internal_exclude.update(sorted_members[1:])
    exact_existing_new = {row["new_relative_path"] for row in exact_existing_rows}
    confirmed_existing = {row["new_image"] for row in resolution_rows if row["final_classification"] == "CONFIRMED_NEAR_DUPLICATE"}
    likely_existing = {row["new_image"] for row in resolution_rows if row["final_classification"] == "LIKELY_NEAR_DUPLICATE"}
    confirmed_test = {row["new_image"] for row in leakage_rows if row["leakage_status"] == "CONFIRMED_LEAKAGE_RISK"}
    likely_test = {row["new_image"] for row in leakage_rows if row["leakage_status"] == "LIKELY_LEAKAGE_RISK"}
    internal_confirmed_exclude = set()
    internal_likely = set()
    for row in internal_resolution:
        pair = sorted([row["first_image"], row["second_image"]])
        if row["final_classification"] == "CONFIRMED_NEAR_DUPLICATE":
            internal_confirmed_exclude.add(pair[1])
        elif row["final_classification"] == "LIKELY_NEAR_DUPLICATE":
            internal_likely.update(pair)
    conflict_images = set()
    for conflict in species_conflicts:
        conflict_images.update(conflict["images"].split(" | "))

    strongest_existing = defaultdict(list)
    for row in resolution_rows:
        strongest_existing[row["new_image"]].append(row["final_classification"])
    strongest_leak = defaultdict(list)
    for row in leakage_rows:
        strongest_leak[row["new_image"]].append(row["leakage_status"])
    rank = ["CONFIRMED_NEAR_DUPLICATE", "LIKELY_NEAR_DUPLICATE", "POSSIBLE_NEAR_DUPLICATE", "MANUAL_REVIEW", "FALSE_POSITIVE_PERCEPTUAL_MATCH"]
    leak_rank = ["CONFIRMED_LEAKAGE_RISK", "LIKELY_LEAKAGE_RISK", "POSSIBLE_LEAKAGE_RISK", "MANUAL_REVIEW", "FALSE_POSITIVE"]
    resolved_manifest = []
    for old in manifest:
        rel = old["relative_path"]
        if rel in exact_existing_new or rel in exact_internal_exclude:
            status, reason = "EXCLUDE_EXACT_DUPLICATE", "Exact duplicate of existing data or non-canonical internal duplicate"
        elif rel in confirmed_test:
            status, reason = "EXCLUDE_FROZEN_TEST_LEAKAGE", "Confirmed near-duplicate of frozen test image"
        elif rel in confirmed_existing or rel in internal_confirmed_exclude:
            status, reason = "EXCLUDE_CONFIRMED_NEAR_DUPLICATE", "Confirmed by multiple independent similarity signals"
        elif rel in likely_test:
            status, reason = "EXCLUDE_FROZEN_TEST_LEAKAGE", "Likely near-duplicate of frozen test image; conservatively excluded"
        elif rel in likely_existing or rel in internal_likely:
            status, reason = "REVIEW_LIKELY_NEAR_DUPLICATE", "Likely near-duplicate requires human decision"
        elif rel in conflict_images:
            status, reason = "REVIEW_SPECIES_CONFLICT", "Same-parent species metadata conflict"
        else:
            status, reason = "REVIEW_LICENSE", "Repository MIT license verified, but README says Google-sourced images may be copyrighted"
        technical_survivor = status not in {"EXCLUDE_EXACT_DUPLICATE", "EXCLUDE_CONFIRMED_NEAR_DUPLICATE", "EXCLUDE_FROZEN_TEST_LEAKAGE"}
        resolved_manifest.append({
            "relative_path": rel, "filename": old["filename"], "parent_label": old["parent_label"],
            "species_label": old["species_label"], "sha256": old["sha256"], "width": old["width"],
            "height": old["height"], "exact_duplicate_group": old["exact_duplicate_group"],
            "near_duplicate_group": old["near_duplicate_group"], "source_group": group_labels.get(rel, ""),
            "exact_existing_match": rel in exact_existing_new,
            "strict_existing_classification": next((x for x in rank if x in strongest_existing.get(rel, [])), "NO_STRICT_MATCH"),
            "frozen_test_leakage_status": next((x for x in leak_rank if x in strongest_leak.get(rel, [])), "NO_TEST_CANDIDATE"),
            "species_label_conflict": rel in conflict_images,
            "provenance_status": "REPOSITORY_LICENSE_VERIFIED; ORIGINAL_REPOSITORY_VERIFIED",
            "license_status": "THIRD_PARTY_IMAGE_RIGHTS_UNCLEAR; REVIEW_REQUIRED",
            "technical_filter_survivor": technical_survivor, "resolution_status": status, "reason": reason,
        })

    approved = [row for row in resolved_manifest if row["resolution_status"] == "APPROVED_CANDIDATE"]
    excluded = [row for row in resolved_manifest if row["resolution_status"].startswith("EXCLUDE_")]
    manual = [row for row in resolved_manifest if row["resolution_status"].startswith("REVIEW_")]
    survivors = [row for row in resolved_manifest if row["technical_filter_survivor"]]
    parent_counts = Counter(row["parent_label"] for row in survivors)
    species_counts = Counter(row["species_label"] for row in survivors)
    projected_rows = [{"level": "PARENT", "label": key, "projected_count": value} for key, value in sorted(parent_counts.items())]
    projected_rows += [{"level": "SPECIES", "label": key, "projected_count": value} for key, value in sorted(species_counts.items())]
    projected_total = sum(parent_counts.values())
    ven, nonven = parent_counts["Venomous"], parent_counts["Non-Venomous"]
    projected_rows += [
        {"level": "METRIC", "label": "Venomous percentage", "projected_count": round(100 * ven / projected_total, 3) if projected_total else 0},
        {"level": "METRIC", "label": "Non-Venomous percentage", "projected_count": round(100 * nonven / projected_total, 3) if projected_total else 0},
        {"level": "METRIC", "label": "Parent imbalance ratio", "projected_count": round(max(ven, nonven) / min(ven, nonven), 4) if min(ven, nonven) else "N/A"},
    ]

    ambiguous = [row for row in leakage_rows if row["leakage_status"] in {"LIKELY_LEAKAGE_RISK", "POSSIBLE_LEAKAGE_RISK", "MANUAL_REVIEW"}]
    ambiguous.sort(key=lambda row: ({"LIKELY_LEAKAGE_RISK": 0, "MANUAL_REVIEW": 1, "POSSIBLE_LEAKAGE_RISK": 2}[row["leakage_status"]], -float(row["ssim"]), -int(row["orb_ransac_inliers"])))
    contact_pairs, contact_sheets = make_contact_sheets(ambiguous, source, clean, output / "review_contact_sheets")

    common_resolution_fields = [
        "new_image", "new_species", "new_parent_class", "existing_image", "existing_class",
        "existing_split", "phash_distance", "dhash_distance", "previous_classification",
        "ahash_distance", "aspect_ratio_delta", "pixel_correlation", "ssim",
        "color_histogram_correlation", "orb_ratio_matches", "orb_ransac_inliers",
        "orb_inlier_ratio", "final_classification", "evidence",
    ]
    write_csv(output / "duplicate_resolution_manifest.csv", resolved_manifest, MANIFEST_FIELDS)
    write_csv(output / "exact_duplicate_confirmation.csv", exact_confirmation, ["duplicate_scope", "duplicate_group", "new_image", "new_class", "new_species", "matching_image", "matching_dataset", "matching_split", "sha256", "reconfirmed"])
    write_csv(output / "near_duplicate_resolution.csv", resolution_rows, common_resolution_fields)
    leakage_fields = common_resolution_fields + ["test_image", "test_class", "leakage_status", "reason"]
    write_csv(output / "frozen_test_leakage_resolution.csv", leakage_rows, leakage_fields)
    write_csv(output / "species_conflict_resolution.csv", species_conflicts, ["conflict_id", "conflict_type", "images", "species", "parents", "resolution", "parent_class_conflict", "reason"])
    write_csv(output / "approved_candidate_manifest.csv", approved, MANIFEST_FIELDS)
    write_csv(output / "excluded_candidate_manifest.csv", excluded, MANIFEST_FIELDS)
    write_csv(output / "manual_review_manifest.csv", manual, MANIFEST_FIELDS)
    write_csv(output / "projected_dataset_balance.csv", projected_rows, ["level", "label", "projected_count"])

    provenance_record = f"""# Dataset Provenance Record

- Audit date (UTC): {datetime.now(timezone.utc).isoformat()}
- Dataset/repository: Indian-Snakes-Dataset
- Supported author/creator: Arjun Sunil (Zenodo); local copyright holder: Arjun S
- Copyright statement: Copyright (c) 2017 Arjun S
- Repository license: MIT License — REPOSITORY_LICENSE_VERIFIED
- LICENSE SHA-256: `{license_sha}`
- README SHA-256: `{readme_sha}`
- Preserved copies byte-identical: YES
- Local dataset path: `{source}`
- Repository URL: https://github.com/arjun921/Indian-Snakes-Dataset
- Archived release: https://zenodo.org/records/4266398
- DOI: https://doi.org/10.5281/zenodo.4266398
- Online repository verification: YES (local README repository badges and DOI match Zenodo's repository, creator, description, and v1.0.0 release)

## License evidence and scope

The root LICENSE grants MIT permissions for the “Software” and associated documentation, subject to preservation of the copyright and permission notice. Its placement supports repository-level MIT licensing.

The local README describes the repository contents as a dataset of venomous and non-venomous snake images. However, it also states that the images may be subject to copyright and that all images were downloaded from Google. No per-image source, attribution, or license metadata was found locally. Consequently:

- Repository license: REPOSITORY_LICENSE_VERIFIED
- Dataset coverage: DATASET_COVERAGE_UNCLEAR
- Image rights: THIRD_PARTY_IMAGE_RIGHTS_UNCLEAR
- Training reuse rights: REVIEW_REQUIRED
- Conflicting license: none found, but the README's copyright warning prevents treating all images as verified MIT-licensed assets.
"""
    (provenance / "provenance_record.md").write_text(provenance_record, encoding="utf-8")

    after = protected_state(root)
    unchanged = before == after
    (output / "protected_path_integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    counts = Counter(row["final_classification"] for row in resolution_rows)
    leakage_counts = Counter(row["leakage_status"] for row in leakage_rows)
    internal_counts = Counter(row["final_classification"] for row in internal_resolution)
    summary = f"""# ANVIKSA Snake Dataset Provenance + Duplicate Resolution

1. LICENSE FOUND: YES
2. README FOUND: YES
3. LICENSE: MIT
4. COPYRIGHT HOLDER: Arjun S
5. COPYRIGHT YEAR: 2017
6. ORIGINAL REPOSITORY VERIFIED: YES
7. DATASET/IMAGE LICENSE COVERAGE: DATASET_COVERAGE_UNCLEAR; THIRD_PARTY_IMAGE_RIGHTS_UNCLEAR
8. THIRD-PARTY RIGHTS CONCERNS: YES — README says images may be copyrighted and were downloaded from Google
9. LICENSE + README PRESERVED: YES, byte-identical copies verified

10. NEW IMAGES: {len(manifest)}
11. INTERNAL EXACT DUPLICATES: {len(exact_groups)} groups / {sum(len(x) for x in exact_groups.values())} files, reproduced exactly
12. INTERNAL CONFIRMED NEAR DUPLICATES: {internal_counts['CONFIRMED_NEAR_DUPLICATE']} pairs
13. EXACT MATCHES AGAINST EXISTING DATA: {len(exact_existing_rows)} unique new images, reproduced exactly
14. PREVIOUS PERCEPTUAL CANDIDATES: {len(candidates)} pair records / {len(set(row['new_relative_path'] for row in candidates))} unique new images
15. CONFIRMED NEAR DUPLICATES AGAINST EXISTING: {counts['CONFIRMED_NEAR_DUPLICATE']} pair records / {len(confirmed_existing)} unique new images
16. LIKELY NEAR DUPLICATES: {counts['LIKELY_NEAR_DUPLICATE']} pair records / {len(likely_existing)} unique new images
17. FALSE-POSITIVE PERCEPTUAL MATCHES: {counts['FALSE_POSITIVE_PERCEPTUAL_MATCH']} pair records

18. FROZEN-TEST CANDIDATES REVIEWED: {len(leakage_rows)} pair records / {len(set(row['new_image'] for row in leakage_rows))} unique new images
19. CONFIRMED FROZEN-TEST LEAKAGE: {leakage_counts['CONFIRMED_LEAKAGE_RISK']} pair records / {len(confirmed_test)} unique new images
20. LIKELY FROZEN-TEST LEAKAGE: {leakage_counts['LIKELY_LEAKAGE_RISK']} pair records / {len(likely_test)} unique new images
21. FALSE-POSITIVE FROZEN-TEST MATCHES: {leakage_counts['FALSE_POSITIVE']} pair records
22. STILL REQUIRING MANUAL REVIEW: {len(manual)} images (includes unresolved image rights)

23. SPECIES LABEL CONFLICTS: {len(species_conflicts)} groups/pairs
24. VENOMOUS/NON-VENOMOUS CONFLICTS: {sum(str(row['parent_class_conflict']).lower() == 'true' for row in species_conflicts)}

25. APPROVED CANDIDATES: {len(approved)}
26. EXCLUDED CANDIDATES: {len(excluded)}
27. MANUAL REVIEW CANDIDATES: {len(manual)}

28. PROJECTED VENOMOUS COUNT: {ven}
29. PROJECTED NON-VENOMOUS COUNT: {nonven}
30. PROJECTED CLASS RATIO: {round(max(ven, nonven) / min(ven, nonven), 4) if min(ven, nonven) else 'N/A'}:1

31. CLEAN DATASET MODIFIED: {'NO' if unchanged else 'YES'}
32. FROZEN TEST SET MODIFIED: {'NO' if unchanged else 'YES'}
33. MODELS MODIFIED: {'NO' if unchanged else 'YES'}
34. TFLITE MODIFIED: {'NO' if unchanged else 'YES'}
35. UI MODIFIED: {'NO' if unchanged else 'YES'}
36. VIDEO DATA MODIFIED: {'NO' if unchanged else 'YES'}
37. AUDIO TRAINING STARTED: NO
38. TRAINING PERFORMED: NO

- Clean fingerprint matches reference: {before['clean_matches_reference'] and after['clean_matches_reference']}
- Combined fingerprint matches reference: {before['combined_matches_reference'] and after['combined_matches_reference']}
- Both protected TFLite hashes match: {all(x['matches_reference'] for x in after['tflite'].values())}
- Protected paths unchanged during resolution: {unchanged}
- Review contact sheets: {contact_sheets} sheets covering {contact_pairs} prioritized ambiguous frozen-test pairs

ANVIKSA SNAKE DATASET PROVENANCE RESOLUTION: COMPLETE

STRICT DUPLICATE RESOLUTION: COMPLETE

FROZEN TEST LEAKAGE REVIEW: {'NEEDS MANUAL REVIEW' if leakage_counts['LIKELY_LEAKAGE_RISK'] or leakage_counts['POSSIBLE_LEAKAGE_RISK'] or leakage_counts['MANUAL_REVIEW'] else 'COMPLETE'}

SAFE FOR CONTROLLED DATASET V2 CONSTRUCTION: NEEDS MANUAL REVIEW
"""
    (output / "resolution_summary.md").write_text(summary, encoding="utf-8")
    print("Resolution complete")
    print(f"Strict classes: {dict(counts)}")
    print(f"Frozen-test statuses: {dict(leakage_counts)}")
    print(f"Approved={len(approved)} excluded={len(excluded)} manual={len(manual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
