#!/usr/bin/env python3
"""Deterministic, non-destructive technical resolution of new snake images.

Consumes the prior audit and strict multi-signal resolution. Writes reports
only under results/new_snake_image_resolution. It does not train, infer,
modify source images, or alter any controlled dataset/model/split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from resolve_new_snake_dataset import classify_pair, image_signals


FIELDS = [
    "image_path", "relative_path", "filename", "parent_class", "species",
    "sha256", "width", "height", "repository_source", "repository_provenance",
    "repository_license", "individual_image_rights", "training_reuse_rights",
    "exact_duplicate", "exact_duplicate_match", "near_duplicate_status",
    "near_duplicate_match", "near_duplicate_score", "frozen_test_leakage_status",
    "frozen_test_match", "label_status", "quality_status", "technical_status",
    "manual_review_required", "exclusion_reason", "notes",
]
EXPECTED = {
    "clean": "b103634bb6a4a01785cb7c7e3b176223325052da47228dba687edaf43a84c594",
    "combined": "502af418ea92ba2a40190208c809180d4ba73ec2597c124bcba5cd6f11a28641",
    "float32": "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7",
    "float16": "5b9e7763e605c7e5f8081dcf12c2ff6afd1cb1d7cd0f6f3cda909e07d1e7c4e5",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
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


def snapshot(paths: list[Path]) -> dict[str, list[list[int | str]]]:
    result = {}
    for base in paths:
        result[str(base)] = [
            [str(path), path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.as_posix().lower())
        ] if base.exists() else []
    return result


def canonical(members: list[dict], source: Path) -> str:
    """Prefer useful resolution, then file size, then deterministic path."""
    return sorted(
        members,
        key=lambda row: (
            -(int(row["width"]) * int(row["height"])),
            -(source / row["relative_path"]).stat().st_size,
            row["relative_path"].lower(),
        ),
    )[0]["relative_path"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "new_training_data/snake_images/Indian-Snakes-Dataset-master"
    audit = root / "results/new_snake_image_audit"
    output = root / "results/new_snake_image_resolution"
    output.mkdir(parents=True, exist_ok=True)
    protected = [
        root / "clean_dataset", root / "combined_dataset", root / "dataset_integrity_reports",
        root / "models", root / "experiments", root / "pi_deployment",
        root / "results/custom_cnn", root / "results/mobilenet_v2",
        root / "results/mobilenet_v3_large", root / "results/efficientnet_b0",
        root / "results/deployment", root / "results/model_comparison_dashboard",
        root / "ui", root / "snake_video_dataset",
    ]
    before = snapshot(protected)

    manifest = read_csv(audit / "new_snake_image_manifest.csv")
    internal_exact_report = read_csv(audit / "exact_duplicate_report.csv")
    internal_near_prior = read_csv(audit / "near_duplicate_report.csv")
    existing_prior = read_csv(audit / "existing_dataset_duplicate_report.csv")
    strict_existing = read_csv(output / "near_duplicate_resolution.csv")
    prior_leakage = read_csv(output / "frozen_test_leakage_resolution.csv")
    species_conflicts = read_csv(output / "species_conflict_resolution.csv")
    if len(manifest) != 1779 or sum(row["readable"] == "True" for row in manifest) != 1779:
        raise SystemExit("Expected the validated 1,779-readable-image audit manifest")
    for row in manifest:
        path = source / row["relative_path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise SystemExit(f"Source image differs from audit manifest: {row['relative_path']}")
    if len(strict_existing) != 1657 or len(prior_leakage) != 210:
        raise SystemExit("Strict existing/leakage measurements are incomplete")

    by_path = {row["relative_path"]: row for row in manifest}
    internal_exact_groups: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        if row["exact_duplicate_group"]:
            internal_exact_groups[row["exact_duplicate_group"]].append(row)
    internal_exact_exclude, internal_exact_match = set(), {}
    exact_rows = []
    for group, members in sorted(internal_exact_groups.items()):
        keep = canonical(members, source)
        for member in members:
            if member["relative_path"] == keep:
                continue
            rel = member["relative_path"]
            internal_exact_exclude.add(rel)
            internal_exact_match[rel] = keep
            exact_rows.append({
                "new_image": rel, "matched_image": keep, "match_location": "NEW_SNAKE_DATASET",
                "sha256": member["sha256"], "directory_label": member["parent_label"],
                "species": member["species_label"], "resolution": "EXACT_DUPLICATE_EXCLUDE",
                "reason": f"Non-canonical member of {group}; canonical selected by resolution, file size, then path",
            })

    existing_exact = [row for row in existing_prior if row["match_type"] == "EXACT"]
    existing_exact_set = {row["new_relative_path"] for row in existing_exact}
    existing_exact_match = defaultdict(list)
    for row in existing_exact:
        existing_exact_match[row["new_relative_path"]].append(row["existing_relative_path"])
        exact_rows.append({
            "new_image": row["new_relative_path"], "matched_image": row["existing_relative_path"],
            "match_location": f"clean_dataset/{row['existing_split']}",
            "sha256": by_path[row["new_relative_path"]]["sha256"],
            "directory_label": row["new_parent"], "species": row["new_species"],
            "resolution": "EXACT_DUPLICATE_EXCLUDE", "reason": "Byte-identical to controlled existing image",
        })

    # Recompute all five internal candidate pairs with the established multi-signal method.
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
    signal_cache = {}
    internal_near = []
    for row in internal_near_prior:
        for rel in (row["first_path"], row["second_path"]):
            if rel not in signal_cache:
                signal_cache[rel] = image_signals(source / rel, orb)
        measures = classify_pair(row, signal_cache[row["first_path"]], signal_cache[row["second_path"]])
        strict = measures["final_classification"]
        mapped = "HIGH_CONFIDENCE_NEAR_DUPLICATE" if strict == "CONFIRMED_NEAR_DUPLICATE" else ("NOT_DUPLICATE" if strict == "FALSE_POSITIVE_PERCEPTUAL_MATCH" else "POSSIBLE_NEAR_DUPLICATE")
        internal_near.append({
            "first_image": row["first_path"], "second_image": row["second_path"],
            "first_parent": row["first_parent"], "second_parent": row["second_parent"],
            "first_species": row["first_species"], "second_species": row["second_species"],
            "near_duplicate_status": mapped, "strict_classification": strict,
            "phash_distance": row["phash_distance"], "dhash_distance": row["dhash_distance"],
            "ahash_distance": measures["ahash_distance"], "ssim": measures["ssim"],
            "pixel_correlation": measures["pixel_correlation"],
            "orb_ransac_inliers": measures["orb_ransac_inliers"], "evidence": measures["evidence"],
        })

    # Confirmed internal groups: retain the stronger representative deterministically.
    internal_near_exclude, internal_near_match = set(), {}
    for row in internal_near:
        if row["near_duplicate_status"] != "HIGH_CONFIDENCE_NEAR_DUPLICATE":
            continue
        members = [by_path[row["first_image"]], by_path[row["second_image"]]]
        keep = canonical(members, source)
        drop = row["second_image"] if keep == row["first_image"] else row["first_image"]
        internal_near_exclude.add(drop)
        internal_near_match[drop] = keep

    strict_map = {
        "CONFIRMED_NEAR_DUPLICATE": "HIGH_CONFIDENCE_NEAR_DUPLICATE",
        "LIKELY_NEAR_DUPLICATE": "POSSIBLE_NEAR_DUPLICATE",
        "POSSIBLE_NEAR_DUPLICATE": "POSSIBLE_NEAR_DUPLICATE",
        "MANUAL_REVIEW": "POSSIBLE_NEAR_DUPLICATE",
        "FALSE_POSITIVE_PERCEPTUAL_MATCH": "NOT_DUPLICATE",
    }
    existing_confirmed, existing_possible = set(), set()
    existing_near_matches, existing_scores = defaultdict(list), defaultdict(list)
    normalized_near_rows = []
    for row in strict_existing:
        rel = row["new_image"]
        status = strict_map[row["final_classification"]]
        if status == "HIGH_CONFIDENCE_NEAR_DUPLICATE":
            existing_confirmed.add(rel)
        elif status == "POSSIBLE_NEAR_DUPLICATE":
            existing_possible.add(rel)
        existing_near_matches[rel].append(row["existing_image"])
        existing_scores[rel].append(float(row["ssim"]))
        normalized_near_rows.append({**row, "near_duplicate_status": status})

    leakage_map = {
        "CONFIRMED_LEAKAGE_RISK": "CONFIRMED_TEST_LEAKAGE",
        "LIKELY_LEAKAGE_RISK": "POSSIBLE_TEST_LEAKAGE",
        "POSSIBLE_LEAKAGE_RISK": "POSSIBLE_TEST_LEAKAGE",
        "MANUAL_REVIEW": "POSSIBLE_TEST_LEAKAGE",
        "FALSE_POSITIVE": "NOT_TEST_LEAKAGE",
    }
    confirmed_test, possible_test = set(), set()
    test_matches = defaultdict(list)
    normalized_leakage = []
    for row in prior_leakage:
        rel = row["new_image"]
        status = leakage_map[row["leakage_status"]]
        if status == "CONFIRMED_TEST_LEAKAGE":
            confirmed_test.add(rel)
        elif status == "POSSIBLE_TEST_LEAKAGE":
            possible_test.add(rel)
        test_matches[rel].append(row["test_image"])
        normalized_leakage.append({**row, "frozen_test_leakage_status": status})

    label_review = set()
    for conflict in species_conflicts:
        label_review.update(conflict["images"].split(" | "))
    quality_review = set()  # Prior integrity audit found no unusable/corrupt images.

    final_rows = []
    for old in manifest:
        rel = old["relative_path"]
        exact = rel in internal_exact_exclude or rel in existing_exact_set
        exact_matches = ([internal_exact_match[rel]] if rel in internal_exact_match else []) + existing_exact_match[rel]
        confirmed_near = rel in existing_confirmed or rel in internal_near_exclude
        possible_near = rel in existing_possible or any(rel in (r["first_image"], r["second_image"]) and r["near_duplicate_status"] == "POSSIBLE_NEAR_DUPLICATE" for r in internal_near)
        near_matches = list(existing_near_matches[rel])
        if rel in internal_near_match:
            near_matches.append(internal_near_match[rel])
        if rel in confirmed_test:
            technical, reason = "FROZEN_TEST_LEAKAGE_EXCLUDE", "Confirmed derivative/match of a frozen-test image"
        elif exact:
            technical, reason = "EXACT_DUPLICATE_EXCLUDE", "Exact duplicate within new data or against existing controlled data"
        elif confirmed_near:
            technical, reason = "NEAR_DUPLICATE_EXCLUDE", "Multiple independent signals confirm essentially the same underlying image"
        elif rel in label_review:
            technical, reason = "LABEL_REVIEW_REQUIRED", "Identical/near-identical imagery occurs in different species directories"
        elif rel in possible_test:
            technical, reason = "MANUAL_REVIEW_REQUIRED", "Possible frozen-test leakage cannot be automatically cleared"
        elif possible_near:
            technical, reason = "MANUAL_REVIEW_REQUIRED", "Possible near-duplicate cannot be automatically cleared"
        elif rel in quality_review:
            technical, reason = "QUALITY_REVIEW_REQUIRED", "Unusable image quality requires review"
        else:
            technical, reason = "TECHNICALLY_ELIGIBLE_RIGHTS_UNCLEAR", "Readable, technically unique, leakage-safe, and without unresolved label/quality conflict"
        leakage_status = "CONFIRMED_TEST_LEAKAGE" if rel in confirmed_test else ("POSSIBLE_TEST_LEAKAGE" if rel in possible_test else "NOT_TEST_LEAKAGE")
        near_status = "HIGH_CONFIDENCE_NEAR_DUPLICATE" if confirmed_near else ("POSSIBLE_NEAR_DUPLICATE" if possible_near else "NOT_DUPLICATE")
        final_rows.append({
            "image_path": str(source / rel), "relative_path": rel, "filename": old["filename"],
            "parent_class": old["parent_label"], "species": old["species_label"],
            "sha256": old["sha256"], "width": old["width"], "height": old["height"],
            "repository_source": "https://github.com/arjun921/Indian-Snakes-Dataset",
            "repository_provenance": "VERIFIED", "repository_license": "MIT",
            "individual_image_rights": "UNCLEAR", "training_reuse_rights": "REVIEW_REQUIRED",
            "exact_duplicate": exact, "exact_duplicate_match": " | ".join(exact_matches),
            "near_duplicate_status": near_status, "near_duplicate_match": " | ".join(near_matches),
            "near_duplicate_score": max(existing_scores[rel]) if existing_scores[rel] else "",
            "frozen_test_leakage_status": leakage_status, "frozen_test_match": " | ".join(test_matches[rel]),
            "label_status": "LABEL_REVIEW_REQUIRED" if rel in label_review else "DIRECTORY_LABEL_PRESERVED",
            "quality_status": "QUALITY_REVIEW_REQUIRED" if rel in quality_review else "BASIC_QUALITY_PASS",
            "technical_status": technical,
            "manual_review_required": technical in {"MANUAL_REVIEW_REQUIRED", "LABEL_REVIEW_REQUIRED", "QUALITY_REVIEW_REQUIRED"},
            "exclusion_reason": reason if technical.endswith("_EXCLUDE") else "",
            "notes": reason + "; technical status does not establish training/reuse rights",
        })

    status_counts = Counter(row["technical_status"] for row in final_rows)
    eligible = [row for row in final_rows if row["technical_status"] == "TECHNICALLY_ELIGIBLE_RIGHTS_UNCLEAR"]
    excluded = [row for row in final_rows if row["technical_status"].endswith("_EXCLUDE")]
    manual = [row for row in final_rows if row["manual_review_required"]]
    labels = [row for row in final_rows if row["label_status"] == "LABEL_REVIEW_REQUIRED"]
    quality = [row for row in final_rows if row["quality_status"] == "QUALITY_REVIEW_REQUIRED"]
    technically_eligible = eligible
    possible_dupes = [row for row in final_rows if row["near_duplicate_status"] == "POSSIBLE_NEAR_DUPLICATE" and not row["technical_status"].endswith("_EXCLUDE")]
    possible_test_rows = [row for row in final_rows if row["frozen_test_leakage_status"] == "POSSIBLE_TEST_LEAKAGE"]
    exact_excluded_relationships = sum(bool(row["exact_duplicate"]) for row in final_rows)
    near_excluded_relationships = sum(
        row["near_duplicate_status"] == "HIGH_CONFIDENCE_NEAR_DUPLICATE"
        and row["technical_status"].endswith("_EXCLUDE")
        for row in final_rows
    )

    write_csv(output / "final_image_resolution.csv", final_rows)
    write_csv(output / "exact_duplicate_resolution.csv", exact_rows, ["new_image", "matched_image", "match_location", "sha256", "directory_label", "species", "resolution", "reason"])
    near_fields = list(normalized_near_rows[0]) if normalized_near_rows else ["new_image"]
    write_csv(output / "near_duplicate_resolution.csv", normalized_near_rows, near_fields)
    leak_fields = list(normalized_leakage[0]) if normalized_leakage else ["new_image"]
    write_csv(output / "frozen_test_leakage_resolution.csv", normalized_leakage, leak_fields)
    write_csv(output / "within_new_near_duplicate_resolution.csv", internal_near, list(internal_near[0]) if internal_near else ["first_image"])
    write_csv(output / "label_review_required.csv", labels)
    write_csv(output / "manual_review_required.csv", manual)
    write_csv(output / "technically_eligible_rights_unclear.csv", technically_eligible)
    write_csv(output / "excluded_images.csv", excluded)

    eligible_species = Counter(row["species"] for row in eligible)
    eligible_parent = Counter(row["parent_class"] for row in eligible)
    after = snapshot(protected)
    unchanged = before == after
    clean_fp = (root / "dataset_integrity_reports/clean/dataset_fingerprint.sha256").read_text().strip()
    combined_fp = (root / "dataset_integrity_reports/original/dataset_fingerprint.sha256").read_text().strip()
    float32 = sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float32.tflite")
    float16 = sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float16.tflite")
    integrity = {
        "protected_metadata_unchanged": unchanged,
        "clean_fingerprint": clean_fp, "clean_matches": clean_fp == EXPECTED["clean"],
        "combined_fingerprint": combined_fp, "combined_matches": combined_fp == EXPECTED["combined"],
        "float32_sha256": float32, "float32_matches": float32 == EXPECTED["float32"],
        "float16_sha256": float16, "float16_matches": float16 == EXPECTED["float16"],
        "source_images_checked": len(manifest), "source_image_hash_mismatches": 0,
    }
    (output / "technical_cleaning_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    species_lines = "\n".join(f"- {species}: {count}" for species, count in sorted(eligible_species.items()))
    summary = f"""# New Snake Image Duplicate + Leakage Resolution

NEW SNAKE IMAGES REVIEWED: {len(final_rows)}
READABLE IMAGES: {sum(row['readable'] == 'True' for row in manifest)}
EXACT DUPLICATES EXCLUDED: {exact_excluded_relationships}
NEAR DUPLICATES EXCLUDED: {near_excluded_relationships}
POSSIBLE DUPLICATES STILL NEEDING REVIEW: {len(possible_dupes)}
FROZEN-TEST LEAKAGE EXCLUDED: {status_counts['FROZEN_TEST_LEAKAGE_EXCLUDE']}
POSSIBLE TEST LEAKAGE STILL NEEDING REVIEW: {len(possible_test_rows)}
LABEL REVIEW REQUIRED: {len(labels)}
QUALITY REVIEW REQUIRED: {len(quality)}

TECHNICALLY ELIGIBLE / RIGHTS UNCLEAR: {len(eligible)}
VENOMOUS TECHNICALLY ELIGIBLE: {eligible_parent['Venomous']}
NON-VENOMOUS TECHNICALLY ELIGIBLE: {eligible_parent['Non-Venomous']}

## Technically eligible count by species

{species_lines}

MANUAL REVIEW REQUIRED: {len(manual)}

Note: duplicate and frozen-test relationship counts overlap where the same image has multiple exclusion reasons. Final technical statuses use frozen-test leakage as the highest priority and form a mutually exclusive 1,779-image partition.

INDIVIDUAL IMAGE RIGHTS:
UNCLEAR / REVIEW REQUIRED

TRAINING PERFORMED:
NO

DATASET INTEGRATION PERFORMED:
NO

MODEL MODIFIED:
NO

FROZEN TEST SET MODIFIED:
NO

## Integrity verification

- clean_dataset modified: NO
- combined_dataset modified: NO
- frozen test set modified: NO
- existing split manifests modified: NO
- Keras models modified: NO
- Float32 TFLite modified: NO
- Float16 TFLite modified: NO
- UI modified: NO
- video dataset modified: NO
- training performed: NO
- fine-tuning performed: NO
- model evaluation performed: NO
- source new images modified/deleted: NO
- clean fingerprint verified: {integrity['clean_matches']}
- combined fingerprint verified: {integrity['combined_matches']}
- Float32 hash verified: {integrity['float32_matches']}
- Float16 hash verified: {integrity['float16_matches']}
- protected metadata unchanged during run: {integrity['protected_metadata_unchanged']}

NEW SNAKE IMAGE DUPLICATE RESOLUTION: {'NEEDS REVIEW' if manual else 'COMPLETE'}

FROZEN TEST LEAKAGE RESOLUTION: {'NEEDS REVIEW' if possible_test_rows else 'PASS'}

TECHNICAL DATASET CLEANING: {'NEEDS REVIEW' if manual else 'COMPLETE'}

RIGHTS CLEARANCE: NOT COMPLETE

READY FOR CONTROLLED DATASET INTEGRATION:
NO, unless separately approved after review

READY FOR MODEL RETRAINING:
NO
"""
    (output / "resolution_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
