#!/usr/bin/env python3
"""Conservatively auto-resolve ANVIKSA's unresolved snake-image comparisons.

This script is non-destructive: it reads images and prior evidence, then writes
manifests only under results/new_snake_image_auto_resolution. It never trains,
integrates, relabels, deletes, moves, or rewrites source/protected data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from resolve_new_snake_dataset import image_signals, geometric, global_ssim, correlation, hamming


FIELDS = [
    "review_id", "new_image", "suspected_match", "review_type",
    "frozen_test_involved", "sha256_match", "phash_distance", "dhash_distance",
    "ahash_distance", "ssim_score", "pixel_similarity_score",
    "dimension_similarity", "edge_similarity", "color_histogram_correlation",
    "orb_ratio_matches", "orb_ransac_inliers", "orb_inlier_ratio",
    "other_evidence", "final_auto_decision", "decision_confidence", "decision_reason",
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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict:
    return {
        str(base): [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in sorted((x for x in base.rglob("*") if x.is_file()), key=lambda x: x.as_posix().lower())] if base.exists() else []
        for base in paths
    }


def tolerant_edge_similarity(first: np.ndarray, second: np.ndarray) -> float:
    edge_a = cv2.Canny(first, 70, 150) > 0
    edge_b = cv2.Canny(second, 70, 150) > 0
    if not edge_a.any() or not edge_b.any():
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    dilated_a = cv2.dilate(edge_a.astype(np.uint8), kernel) > 0
    dilated_b = cv2.dilate(edge_b.astype(np.uint8), kernel) > 0
    recall_a = float(np.logical_and(edge_a, dilated_b).sum() / edge_a.sum())
    recall_b = float(np.logical_and(edge_b, dilated_a).sum() / edge_b.sum())
    return 2 * recall_a * recall_b / (recall_a + recall_b) if recall_a + recall_b else 0.0


def classify(metrics: dict, frozen: bool) -> tuple[str, str, str]:
    exact = metrics["sha256_match"]
    pd, dd, ad = metrics["phash_distance"], metrics["dhash_distance"], metrics["ahash_distance"]
    ssim, pixel, edge = metrics["ssim_score"], metrics["pixel_similarity_score"], metrics["edge_similarity"]
    hist = metrics["color_histogram_correlation"]
    good, inliers, ratio = metrics["orb_ratio_matches"], metrics["orb_ransac_inliers"], metrics["orb_inlier_ratio"]

    direct_same = (
        ssim >= 0.965 and pixel >= 0.970 and ad <= 6
        and (pd <= 4 or dd <= 4) and (edge >= 0.55 or inliers >= 6)
    )
    geometric_same = (
        good >= 15 and inliers >= 10 and ratio >= 0.52
        and (pd <= 6 or dd <= 6) and (ssim >= 0.62 or pixel >= 0.72 or edge >= 0.48)
    )
    crop_or_watermark_same = (
        good >= 22 and inliers >= 14 and ratio >= 0.45
        and hist >= 0.60 and (pixel >= 0.55 or edge >= 0.40)
    )
    clearly_different = (
        ssim <= 0.28 and abs(pixel) <= 0.28 and edge <= 0.18
        and inliers <= 2 and hist <= 0.35 and ad >= 16
        and pd >= 4 and dd >= 4
    )
    if exact or direct_same or geometric_same or crop_or_watermark_same:
        decision = "AUTO_FROZEN_TEST_LEAKAGE" if frozen else "AUTO_DUPLICATE"
        triggers = []
        if exact: triggers.append("SHA-256 exact match")
        if direct_same: triggers.append("very high normalized SSIM/pixel agreement plus hash and edge/feature support")
        if geometric_same: triggers.append("strong geometrically consistent local-feature agreement plus supporting signals")
        if crop_or_watermark_same: triggers.append("strong crop/watermark-compatible feature and color/structure evidence")
        return decision, "HIGH", "; ".join(triggers) + "; same underlying photograph/derivative indicated by multiple independent signals"
    if clearly_different:
        return "AUTO_NOT_DUPLICATE", "HIGH", "Low structure, pixel, edge, color, and feature agreement with materially different hashes; clearly different underlying image"
    borderline = sum([
        ssim >= 0.80, pixel >= 0.82, edge >= 0.40, hist >= 0.65,
        inliers >= 6, pd <= 6, dd <= 6, ad <= 10,
    ])
    confidence = "MEDIUM" if borderline >= 4 else "LOW"
    return "AMBIGUOUS_REVIEW_REQUIRED", confidence, "Signals do not meet conservative HIGH-confidence same-image or clearly-different rules"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    manual_root = root / "results/new_snake_image_manual_review"
    resolution_root = root / "results/new_snake_image_resolution"
    output = root / "results/new_snake_image_auto_resolution"
    output.mkdir(parents=True, exist_ok=True)
    source = root / "new_training_data/snake_images/Indian-Snakes-Dataset-master"
    clean = root / "clean_dataset"
    protected = [
        root / "clean_dataset", root / "combined_dataset", root / "dataset_integrity_reports",
        root / "models", root / "experiments", root / "pi_deployment",
        root / "results/custom_cnn", root / "results/mobilenet_v2",
        root / "results/mobilenet_v3_large", root / "results/efficientnet_b0",
        root / "results/deployment", root / "results/model_comparison_dashboard",
        root / "ui", root / "snake_video_dataset",
    ]
    before = snapshot(protected)
    decisions = read_csv(manual_root / "manual_decisions.csv")
    final_resolution = read_csv(resolution_root / "final_image_resolution.csv")
    eligible_original = read_csv(resolution_root / "technically_eligible_rights_unclear.csv")
    if len(decisions) != 324 or len(eligible_original) != 135:
        raise SystemExit("Expected 324 unresolved cases and 135 original technically eligible images")
    if any(row["human_decision"] for row in decisions):
        raise SystemExit("Human decisions already exist; automatic phase refuses to overwrite or reinterpret them")
    final_by_path = {row["image_path"]: row for row in final_resolution}
    source_before = {
        row["image_path"]: (Path(row["image_path"]).stat().st_size, Path(row["image_path"]).stat().st_mtime_ns)
        for row in final_resolution
    }
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    cache = {}
    rows = []
    for number, item in enumerate(decisions, 1):
        new_path = Path(item["image_path"])
        if item["review_type"] == "LABEL_REVIEW":
            rows.append({
                "review_id": item["review_id"], "new_image": str(new_path),
                "suspected_match": item["suspected_match_path"], "review_type": item["review_type"],
                "frozen_test_involved": item["frozen_test_involved"], "sha256_match": "",
                "phash_distance": "", "dhash_distance": "", "ahash_distance": "",
                "ssim_score": "", "pixel_similarity_score": "", "dimension_similarity": "",
                "edge_similarity": "", "color_histogram_correlation": "",
                "orb_ratio_matches": "", "orb_ransac_inliers": "", "orb_inlier_ratio": "",
                "other_evidence": "Same-parent cross-species metadata conflict; biological appearance was not evaluated",
                "final_auto_decision": "LABEL_REVIEW_REQUIRED", "decision_confidence": "",
                "decision_reason": "Folder/source metadata does not resolve which species label is correct",
            })
            continue
        match_path = clean / item["suspected_match_path"]
        if not new_path.is_file() or not match_path.is_file():
            raise SystemExit(f"Comparison path missing for {item['review_id']}")
        for path in (new_path, match_path):
            key = str(path)
            if key not in cache:
                cache[key] = image_signals(path, orb)
        first, second = cache[str(new_path)], cache[str(match_path)]
        pd = int(next(part.split("=")[1] for part in item["similarity_summary"].split("; ") if part.startswith("pHash=")))
        dd = int(next(part.split("=")[1] for part in item["similarity_summary"].split("; ") if part.startswith("dHash=")))
        ad = hamming(first["ahash"], second["ahash"])
        ssim = global_ssim(first["gray"], second["gray"])
        pixel = correlation(first["gray"], second["gray"])
        edge = tolerant_edge_similarity(first["gray"], second["gray"])
        aspect_similarity = min(first["aspect"], second["aspect"]) / max(first["aspect"], second["aspect"])
        area_a, area_b = first["width"] * first["height"], second["width"] * second["height"]
        area_similarity = min(area_a, area_b) / max(area_a, area_b)
        dimension_similarity = math.sqrt(aspect_similarity * area_similarity)
        hist = float(cv2.compareHist(first["hist"], second["hist"], cv2.HISTCMP_CORREL))
        good, inliers, ratio = geometric(first, second)
        metrics = {
            "sha256_match": sha256_file(new_path) == sha256_file(match_path),
            "phash_distance": pd, "dhash_distance": dd, "ahash_distance": ad,
            "ssim_score": ssim, "pixel_similarity_score": pixel,
            "dimension_similarity": dimension_similarity, "edge_similarity": edge,
            "color_histogram_correlation": hist, "orb_ratio_matches": good,
            "orb_ransac_inliers": inliers, "orb_inlier_ratio": ratio,
        }
        frozen = item["frozen_test_involved"] == "YES"
        decision, confidence, reason = classify(metrics, frozen)
        rows.append({
            "review_id": item["review_id"], "new_image": str(new_path),
            "suspected_match": str(match_path), "review_type": item["review_type"],
            "frozen_test_involved": item["frozen_test_involved"],
            "sha256_match": metrics["sha256_match"], "phash_distance": pd,
            "dhash_distance": dd, "ahash_distance": ad,
            "ssim_score": round(ssim, 6), "pixel_similarity_score": round(pixel, 6),
            "dimension_similarity": round(dimension_similarity, 6), "edge_similarity": round(edge, 6),
            "color_histogram_correlation": round(hist, 6), "orb_ratio_matches": good,
            "orb_ransac_inliers": inliers, "orb_inlier_ratio": round(ratio, 6),
            "other_evidence": f"aspect_similarity={aspect_similarity:.6f}; area_similarity={area_similarity:.6f}; normalized 128x128 comparison; tolerant Canny-edge F1; ORB+RANSAC geometry",
            "final_auto_decision": decision, "decision_confidence": confidence,
            "decision_reason": reason,
        })
        if number % 50 == 0:
            print(f"Resolved {number}/{len(decisions)} cases", flush=True)

    counts = Counter(row["final_auto_decision"] for row in rows)
    if counts["LABEL_REVIEW_REQUIRED"] != 2 or len(rows) != 324:
        raise SystemExit("Automatic resolution partition is incomplete")
    write_csv(output / "auto_resolution.csv", rows, FIELDS)
    write_csv(output / "auto_duplicate_cases.csv", [r for r in rows if r["final_auto_decision"] == "AUTO_DUPLICATE"], FIELDS)
    write_csv(output / "auto_frozen_test_leakage.csv", [r for r in rows if r["final_auto_decision"] == "AUTO_FROZEN_TEST_LEAKAGE"], FIELDS)
    write_csv(output / "auto_not_duplicate.csv", [r for r in rows if r["final_auto_decision"] == "AUTO_NOT_DUPLICATE"], FIELDS)
    write_csv(output / "ambiguous_review_required.csv", [r for r in rows if r["final_auto_decision"] == "AMBIGUOUS_REVIEW_REQUIRED"], FIELDS)
    write_csv(output / "label_review_remaining.csv", [r for r in rows if r["final_auto_decision"] == "LABEL_REVIEW_REQUIRED"], FIELDS)

    auto_not_paths = {row["new_image"] for row in rows if row["final_auto_decision"] == "AUTO_NOT_DUPLICATE"}
    projected = []
    for row in eligible_original:
        projected.append({**row, "auto_resolution_source": "ORIGINAL_TECHNICALLY_ELIGIBLE", "auto_decision": "NOT_APPLICABLE", "individual_image_rights": "UNCLEAR", "training_reuse_rights": "REVIEW_REQUIRED"})
    for path in sorted(auto_not_paths):
        old = final_by_path[path]
        projected.append({**old, "auto_resolution_source": "NEWLY_TECHNICALLY_CLEARED", "auto_decision": "AUTO_NOT_DUPLICATE", "individual_image_rights": "UNCLEAR", "training_reuse_rights": "REVIEW_REQUIRED"})
    projected_fields = list(dict.fromkeys((list(projected[0]) if projected else []) + ["auto_resolution_source", "auto_decision", "individual_image_rights", "training_reuse_rights"]))
    write_csv(output / "projected_technically_eligible_rights_unclear.csv", projected, projected_fields)

    venomous = sum(row.get("parent_class") == "Venomous" for row in projected)
    nonvenomous = sum(row.get("parent_class") == "Non-Venomous" for row in projected)
    after = snapshot(protected)
    source_after = {
        row["image_path"]: (Path(row["image_path"]).stat().st_size, Path(row["image_path"]).stat().st_mtime_ns)
        for row in final_resolution
    }
    source_hash_mismatches = sum(
        sha256_file(Path(row["image_path"])) != row["sha256"]
        for row in final_resolution
    )
    clean_fp = (root / "dataset_integrity_reports/clean/dataset_fingerprint.sha256").read_text().strip()
    combined_fp = (root / "dataset_integrity_reports/original/dataset_fingerprint.sha256").read_text().strip()
    integrity = {
        "protected_metadata_unchanged": before == after,
        "clean_fingerprint_matches": clean_fp == EXPECTED["clean"],
        "combined_fingerprint_matches": combined_fp == EXPECTED["combined"],
        "float32_tflite_matches": sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float32.tflite") == EXPECTED["float32"],
        "float16_tflite_matches": sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float16.tflite") == EXPECTED["float16"],
        "source_images_checked": len(final_resolution),
        "source_image_hash_mismatches": source_hash_mismatches,
        "source_image_metadata_unchanged": source_before == source_after,
    }
    (output / "integrity_verification.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    status = "COMPLETE" if counts["AMBIGUOUS_REVIEW_REQUIRED"] == 0 else "PARTIAL"
    frozen_status = "COMPLETE" if not any(
        r["frozen_test_involved"] == "YES"
        and r["final_auto_decision"] not in {"AUTO_FROZEN_TEST_LEAKAGE", "AUTO_NOT_DUPLICATE"}
        for r in rows
    ) else "PARTIAL"
    summary = f"""# ANVIKSA Automatic Unresolved Snake Image Resolution

TOTAL UNRESOLVED REVIEWED: {len(rows)}

AUTO_DUPLICATE: {counts['AUTO_DUPLICATE']}

AUTO_FROZEN_TEST_LEAKAGE: {counts['AUTO_FROZEN_TEST_LEAKAGE']}

AUTO_NOT_DUPLICATE: {counts['AUTO_NOT_DUPLICATE']}

AMBIGUOUS_REVIEW_REQUIRED: {counts['AMBIGUOUS_REVIEW_REQUIRED']}

LABEL_REVIEW_REQUIRED: {counts['LABEL_REVIEW_REQUIRED']}

ORIGINAL TECHNICALLY ELIGIBLE: {len(eligible_original)}

NEWLY TECHNICALLY CLEARED: {len(auto_not_paths)}

PROJECTED TECHNICALLY ELIGIBLE TOTAL: {len(projected)}

PROJECTED VENOMOUS: {venomous}

PROJECTED NON-VENOMOUS: {nonvenomous}

## Integrity

- source images modified: NO
- clean_dataset modified: NO
- combined_dataset modified: NO
- frozen test modified: NO
- models modified: NO
- TFLite modified: NO
- UI modified: NO
- video data modified: NO
- training performed: NO
- dataset integration performed: NO
- protected metadata unchanged: {integrity['protected_metadata_unchanged']}
- source image metadata unchanged: {integrity['source_image_metadata_unchanged']}
- source image SHA-256 mismatches: {integrity['source_image_hash_mismatches']}
- clean fingerprint verified: {integrity['clean_fingerprint_matches']}
- combined fingerprint verified: {integrity['combined_fingerprint_matches']}
- Float32 TFLite verified: {integrity['float32_tflite_matches']}
- Float16 TFLite verified: {integrity['float16_tflite_matches']}

AUTOMATIC DUPLICATE RESOLUTION: {status}

FROZEN-TEST LEAKAGE AUTO-RESOLUTION: {frozen_status}

AMBIGUOUS CASES REMAINING: {counts['AMBIGUOUS_REVIEW_REQUIRED'] + counts['LABEL_REVIEW_REQUIRED']}

PROJECTED TECHNICALLY ELIGIBLE POOL: {len(projected)}

RIGHTS CLEARANCE: STILL NOT COMPLETE

READY FOR DATASET INTEGRATION: NO

READY FOR MODEL RETRAINING: NO
"""
    (output / "auto_resolution_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
