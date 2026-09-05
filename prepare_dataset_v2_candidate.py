#!/usr/bin/env python3
"""Build and validate the experimental ANVIKSA Dataset V2 candidate.

Copies the controlled clean dataset and exactly the 135 authoritative new
snake candidates. Never writes to sources, models, historical results, UI,
video, or audio assets and never performs training or inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


CLASSES = ["Cow", "Deer", "Elephant", "Monkey", "Non_Venomous_Snake", "Venomous_Snake", "Wild_Boar"]
EXPECTED = {
    "clean": "b103634bb6a4a01785cb7c7e3b176223325052da47228dba687edaf43a84c594",
    "combined": "502af418ea92ba2a40190208c809180d4ba73ec2597c124bcba5cd6f11a28641",
    "float32": "086b94af27c4669710a0afb54a1ce0efdb1d37b01d4dcba8c15c5d7de32c5ac7",
    "float16": "5b9e7763e605c7e5f8081dcf12c2ff6afd1cb1d7cd0f6f3cda909e07d1e7c4e5",
}
MANIFEST_FIELDS = [
    "image_id", "destination_path", "split", "class_name", "species",
    "source_type", "source_path", "original_filename", "is_original_dataset_image",
    "is_new_candidate_image", "source_group_id", "duplicate_status",
    "frozen_test_leakage_status", "rights_status", "sha256",
]


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


def hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def protected_snapshot(paths: list[Path]) -> dict:
    return {
        str(base): [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in sorted((x for x in base.rglob("*") if x.is_file()), key=lambda x: x.as_posix().lower())] if base.exists() else []
        for base in paths
    }


class DSU:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def normalized_source_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"\b(?:small|medium|large|thumb|thumbnail|cropped|edited|resized|compressed)\b", "", stem)
    stem = re.sub(r"\b\d{2,5}x\d{2,5}\b", "", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def build_source_groups(candidates: list[dict], audit_by_path: dict[str, dict]) -> dict[str, str]:
    paths = [row["relative_path"] for row in candidates]
    dsu = DSU(paths)
    for index, first in enumerate(candidates):
        first_audit = audit_by_path[first["relative_path"]]
        for second in candidates[index + 1:]:
            if first["species"] != second["species"]:
                continue
            second_audit = audit_by_path[second["relative_path"]]
            same_named_source = normalized_source_name(first["filename"]) == normalized_source_name(second["filename"]) and len(normalized_source_name(first["filename"])) >= 5
            perceptually_related = hamming(first_audit["perceptual_hash"], second_audit["perceptual_hash"]) <= 4 and hamming(first_audit["dhash"], second_audit["dhash"]) <= 4
            prior_group = bool(first_audit["near_duplicate_group"] and first_audit["near_duplicate_group"] == second_audit["near_duplicate_group"])
            if same_named_source or perceptually_related or prior_group:
                dsu.union(first["relative_path"], second["relative_path"])
    roots = sorted({dsu.find(path) for path in paths})
    root_ids = {root: f"NEW_SOURCE_GROUP_{index:04d}" for index, root in enumerate(roots, 1)}
    return {path: root_ids[dsu.find(path)] for path in paths}


def choose_validation_groups(groups: dict[str, list[dict]], total: int) -> set[str]:
    if total <= 1 or len(groups) <= 1:
        return set()
    target = max(1, round(total * 0.20))
    ordered = sorted(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())
    possibilities: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        weight = len(groups[group])
        additions = {}
        for current, selection in possibilities.items():
            proposed = current + weight
            if proposed < total and proposed not in possibilities and proposed not in additions:
                additions[proposed] = selection + (group,)
        possibilities.update(additions)
    best = min((count for count in possibilities if 0 < count < total), key=lambda count: (abs(count - target), count > target, count))
    return set(possibilities[best])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    clean = root / "clean_dataset"
    source = root / "new_training_data/snake_images/Indian-Snakes-Dataset-master"
    destination = root / "dataset_v2_candidate"
    metadata = destination / "metadata"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing Dataset V2 candidate: {destination}")
    protected = [
        clean, root / "combined_dataset", root / "dataset_integrity_reports", root / "models",
        root / "experiments", root / "pi_deployment", root / "results/custom_cnn",
        root / "results/mobilenet_v2", root / "results/mobilenet_v3_large",
        root / "results/efficientnet_b0", root / "results/deployment",
        root / "results/model_comparison_dashboard", root / "ui", root / "snake_video_dataset",
    ]
    before = protected_snapshot(protected)
    clean_fp = (root / "dataset_integrity_reports/clean/dataset_fingerprint.sha256").read_text().strip()
    combined_fp = (root / "dataset_integrity_reports/original/dataset_fingerprint.sha256").read_text().strip()
    if clean_fp != EXPECTED["clean"] or combined_fp != EXPECTED["combined"]:
        raise SystemExit("Base dataset fingerprint verification failed before construction")

    candidate_path = root / "results/new_snake_image_auto_resolution/projected_technically_eligible_rights_unclear.csv"
    candidates = read_csv(candidate_path)
    if len(candidates) != 135 or Counter(row["parent_class"] for row in candidates) != Counter({"Non-Venomous": 82, "Venomous": 53}):
        raise SystemExit("Authoritative 135-image candidate manifest count mismatch")
    if any(row["individual_image_rights"] != "UNCLEAR" or row["training_reuse_rights"] != "REVIEW_REQUIRED" for row in candidates):
        raise SystemExit("Candidate rights status mismatch")
    if any(row["technical_status"] != "TECHNICALLY_ELIGIBLE_RIGHTS_UNCLEAR" for row in candidates):
        raise SystemExit("Non-eligible image found in authoritative candidate list")
    audit_rows = read_csv(root / "results/new_snake_image_audit/new_snake_image_manifest.csv")
    audit_by_path = {row["relative_path"]: row for row in audit_rows}
    for row in candidates:
        path = Path(row["image_path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise SystemExit(f"Candidate source mismatch: {row['relative_path']}")

    source_groups = build_source_groups(candidates, audit_by_path)
    groups_by_species: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        groups_by_species[row["species"]][source_groups[row["relative_path"]]].append(row)
    validation_groups = set()
    for species, groups in sorted(groups_by_species.items()):
        validation_groups |= choose_validation_groups(groups, sum(len(rows) for rows in groups.values()))
    new_splits = {row["relative_path"]: ("validation" if source_groups[row["relative_path"]] in validation_groups else "train") for row in candidates}

    for split in ("train", "validation", "test"):
        for class_name in CLASSES:
            (destination / split / class_name).mkdir(parents=True, exist_ok=False)
    metadata.mkdir(parents=True, exist_ok=False)

    historical_manifests = {}
    v2_rows = []
    image_index = 0
    for split in ("train", "validation", "test"):
        manifest_path = root / f"dataset_integrity_reports/clean/{split}_manifest.csv"
        rows = read_csv(manifest_path)
        historical_manifests[split] = rows
        if split == "test":
            shutil.copyfile(manifest_path, metadata / "historical_frozen_test_manifest.csv")
        for row in rows:
            source_path = clean / row["relative_path"]
            destination_path = destination / row["relative_path"]
            if sha256_file(source_path) != row["sha256"]:
                raise SystemExit(f"Base source hash mismatch: {row['relative_path']}")
            shutil.copy2(source_path, destination_path)
            image_index += 1
            v2_rows.append({
                "image_id": f"V2_{image_index:06d}", "destination_path": row["relative_path"],
                "split": split, "class_name": row["class_name"], "species": "",
                "source_type": "BASE_CLEAN_DATASET", "source_path": str(source_path),
                "original_filename": source_path.name, "is_original_dataset_image": True,
                "is_new_candidate_image": False, "source_group_id": "BASE_" + row["sha256"][:16],
                "duplicate_status": "CLEAN_BASELINE", "frozen_test_leakage_status": "HISTORICAL_FROZEN_TEST" if split == "test" else "NOT_APPLICABLE",
                "rights_status": "CONTROLLED_EXISTING_DATASET", "sha256": row["sha256"],
            })
        print(f"Copied historical {split}: {len(rows)}", flush=True)

    split_rows = []
    used_destinations = {row["destination_path"].lower() for row in v2_rows}
    for row in sorted(candidates, key=lambda item: item["relative_path"].lower()):
        split = new_splits[row["relative_path"]]
        class_name = "Venomous_Snake" if row["parent_class"] == "Venomous" else "Non_Venomous_Snake"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", row["filename"])
        filename = f"new_{row['sha256'][:12]}_{safe_name}"
        relative = f"{split}/{class_name}/{filename}"
        if relative.lower() in used_destinations:
            filename = f"new_{row['sha256'][:20]}_{safe_name}"
            relative = f"{split}/{class_name}/{filename}"
        if relative.lower() in used_destinations:
            raise SystemExit(f"Unresolved deterministic destination collision: {relative}")
        used_destinations.add(relative.lower())
        src = Path(row["image_path"])
        shutil.copy2(src, destination / relative)
        image_index += 1
        group = source_groups[row["relative_path"]]
        v2_rows.append({
            "image_id": f"V2_{image_index:06d}", "destination_path": relative,
            "split": split, "class_name": class_name, "species": row["species"],
            "source_type": "NEW_TECHNICALLY_CLEARED_SNAKE_IMAGE", "source_path": str(src),
            "original_filename": row["filename"], "is_original_dataset_image": False,
            "is_new_candidate_image": True, "source_group_id": group,
            "duplicate_status": "TECHNICALLY_UNIQUE", "frozen_test_leakage_status": "NOT_TEST_LEAKAGE",
            "rights_status": "REVIEW_REQUIRED", "sha256": row["sha256"],
        })
        split_rows.append({
            "source_image": str(src), "species": row["species"], "parent_class": row["parent_class"],
            "source_group": group, "assigned_split": split,
            "reason_for_split_assignment": "Deterministic species-stratified group-aware approximately 80/20 allocation; related groups remain intact",
            "sha256": row["sha256"], "rights_status": "REVIEW_REQUIRED",
        })
    write_csv(metadata / "dataset_v2_manifest.csv", v2_rows, MANIFEST_FIELDS)
    write_csv(metadata / "new_snake_image_split.csv", split_rows, ["source_image", "species", "parent_class", "source_group", "assigned_split", "reason_for_split_assignment", "sha256", "rights_status"])

    # Full copied-file integrity and readability validation.
    missing, corrupt, hash_mismatches = [], [], []
    for row in v2_rows:
        path = destination / row["destination_path"]
        if not path.is_file():
            missing.append(row["destination_path"]); continue
        if sha256_file(path) != row["sha256"]:
            hash_mismatches.append(row["destination_path"])
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            corrupt.append({"path": row["destination_path"], "error": f"{type(exc).__name__}: {exc}"})

    # Exact cross-split duplicate audit.
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for row in v2_rows:
        by_sha[row["sha256"]].append(row)
    exact_cross_split = []
    duplicate_destinations = len(v2_rows) - len({row["destination_path"].lower() for row in v2_rows})
    for sha, rows in by_sha.items():
        splits = {row["split"] for row in rows}
        if len(splits) > 1:
            exact_cross_split.append({"sha256": sha, "splits": sorted(splits), "paths": sorted(row["destination_path"] for row in rows)})

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in v2_rows:
        if row["is_new_candidate_image"] is True:
            group_splits[row["source_group_id"]].add(row["split"])
    group_leakage = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}

    # Recheck every new hash/perceptual hash against the frozen test manifest.
    test_rows = historical_manifests["test"]
    test_hashes = {row["sha256"] for row in test_rows}
    new_test_exact = [row["relative_path"] for row in candidates if row["sha256"] in test_hashes]
    new_test_near = []
    for candidate in candidates:
        audit = audit_by_path[candidate["relative_path"]]
        for test in test_rows:
            pd, dd = hamming(audit["perceptual_hash"], test["phash"]), hamming(audit["dhash"], test["dhash"])
            if (pd <= 2 and dd <= 2) or (pd <= 6 and dd <= 8) or (pd <= 10 and dd <= 4):
                new_test_near.append({"new": candidate["relative_path"], "test": test["relative_path"], "phash_distance": pd, "dhash_distance": dd})

    copied_test_rows = [row for row in v2_rows if row["split"] == "test"]
    test_count_match = len(copied_test_rows) == len(test_rows) == 545
    test_order_match = [row["destination_path"] for row in copied_test_rows] == [row["relative_path"] for row in test_rows]
    test_hash_match = all((destination / row["relative_path"]).is_file() and sha256_file(destination / row["relative_path"]) == row["sha256"] for row in test_rows)
    test_manifest_copy_match = sha256_file(metadata / "historical_frozen_test_manifest.csv") == sha256_file(root / "dataset_integrity_reports/clean/test_manifest.csv")

    counts = Counter((row["split"], row["class_name"], bool(row["is_new_candidate_image"])) for row in v2_rows)
    count_rows = []
    for split in ("train", "validation", "test"):
        for class_name in CLASSES:
            original = counts[(split, class_name, False)]
            new = counts[(split, class_name, True)]
            count_rows.append({"split": split, "class_name": class_name, "original_count": original, "new_count": new, "final_v2_count": original + new})
    for split in ("train", "validation", "test"):
        original = sum(row["original_count"] for row in count_rows if row["split"] == split)
        new = sum(row["new_count"] for row in count_rows if row["split"] == split)
        count_rows.append({"split": split, "class_name": "ALL_CLASSES", "original_count": original, "new_count": new, "final_v2_count": original + new})
    write_csv(metadata / "dataset_v2_counts.csv", count_rows, ["split", "class_name", "original_count", "new_count", "final_v2_count"])

    canonical_lines = []
    for row in sorted(v2_rows, key=lambda item: item["destination_path"]):
        canonical_lines.append(json.dumps({"destination_path": row["destination_path"], "split": row["split"], "class_name": row["class_name"], "sha256": row["sha256"]}, sort_keys=True, separators=(",", ":")))
    fingerprint = hashlib.sha256(("\n".join(canonical_lines) + "\n").encode()).hexdigest()
    split_totals = Counter(row["split"] for row in v2_rows)
    fingerprint_record = {
        "fingerprint": fingerprint,
        "algorithm": "SHA-256 of UTF-8 canonical JSON lines sorted by destination_path; each line contains destination_path, split, class_name, and file SHA-256",
        "image_count": len(v2_rows), "train_count": split_totals["train"],
        "validation_count": split_totals["validation"], "test_count": split_totals["test"],
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_fingerprint": clean_fp,
        "new_candidate_manifest_source": str(candidate_path),
    }
    (metadata / "dataset_v2_fingerprint.json").write_text(json.dumps(fingerprint_record, indent=2), encoding="utf-8")
    information = {
        "dataset_name": "ANVIKSA Dataset V2 Candidate", "status": "EXPERIMENTAL",
        "training_authorization": "NOT YET APPROVED", "rights_status": "REVIEW REQUIRED",
        "base_dataset": "clean_dataset", "base_fingerprint": clean_fp,
        "new_technically_cleared_snake_images": 135, "new_venomous": 53,
        "new_non_venomous": 82, "frozen_test": "UNCHANGED",
        "new_images_allowed_splits": ["train", "validation"],
    }
    (metadata / "dataset_v2_information.json").write_text(json.dumps(information, indent=2), encoding="utf-8")
    baseline = {
        "model": "MobileNetV3 Large", "source": "existing historical baseline; not recalculated",
        "test_accuracy_percent": 95.412844, "macro_precision_percent": 96.886727,
        "macro_recall_percent": 96.869714, "macro_f1_percent": 96.872452,
        "venomous_snake_recall_percent": 92.907801, "non_venomous_snake_recall_percent": 91.406250,
        "snake_macro_recall_percent": 92.157026, "snake_macro_f1_percent": 92.371820,
        "test_images": 545,
    }
    (metadata / "baseline_model_metrics.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    future = {
        "future_model": "MobileNetV3 Large", "execution_status": "DOCUMENTED_ONLY_NOT_EXECUTED",
        "starting_point_decision": "DEFERRED_TO_NEXT_PHASE",
        "options_not_selected": ["retrain from ImageNet", "continue from existing Keras checkpoint", "fine-tune existing model"],
        "required_test_set": "same historical frozen 545-image test set",
        "required_comparison": "compare against historical MobileNetV3 Large baseline",
    }
    (metadata / "future_experiment_design.json").write_text(json.dumps(future, indent=2), encoding="utf-8")

    readme = """# ANVIKSA Dataset V2 Candidate

This experimental dataset candidate was created to support a future, separately approved MobileNetV3 Large experiment with additional technically cleaned snake imagery.

1. The original `clean_dataset` and `combined_dataset` remain preserved and unchanged.
2. Exactly 135 new snake images passed the technical cleaning manifests: 53 venomous and 82 non-venomous.
3. The 319 automatic duplicate/leakage exclusions were not included.
4. The five unresolved similarity/label cases were not included.
5. The historical 545-image frozen test set was copied byte-for-byte and remains unchanged.
6. New images occur only in train and validation; none occurs in test.
7. New-image assignment is deterministic, approximately 80/20, species-stratified, and source-group-aware. Group integrity takes priority over exact percentages.
8. Species names are metadata; model targets remain the original seven classes.
9. Dataset V2 has not been used for training, fine-tuning, evaluation, or model conversion.
10. Individual image reuse/training rights remain unresolved because repository-level MIT licensing does not establish rights for every Google-sourced image.
11. Status: **EXPERIMENTAL / RIGHTS REVIEW REQUIRED**.
12. Future training requires separate explicit approval. The starting-point strategy is intentionally deferred.

See `metadata/dataset_v2_manifest.csv`, `metadata/new_snake_image_split.csv`, `metadata/dataset_v2_information.json`, and `metadata/dataset_v2_fingerprint.json` for reproducible details.
"""
    (destination / "README_DATASET_V2.md").write_text(readme, encoding="utf-8")

    new_train = [row for row in v2_rows if row["is_new_candidate_image"] is True and row["split"] == "train"]
    new_val = [row for row in v2_rows if row["is_new_candidate_image"] is True and row["split"] == "validation"]
    new_test = [row for row in v2_rows if row["is_new_candidate_image"] is True and row["split"] == "test"]
    after = protected_snapshot(protected)
    validation = {
        "original_images_copied": sum(row["is_original_dataset_image"] is True for row in v2_rows),
        "new_images_copied": sum(row["is_new_candidate_image"] is True for row in v2_rows),
        "missing_manifest_files": missing, "corrupt_images": corrupt,
        "hash_mismatches": hash_mismatches, "duplicate_destination_files": duplicate_destinations,
        "exact_cross_split_duplicate_groups": exact_cross_split,
        "new_source_group_split_leakage": group_leakage,
        "new_exact_frozen_test_matches": new_test_exact,
        "new_perceptual_frozen_test_candidates": new_test_near,
        "historical_test_count": len(test_rows), "dataset_v2_test_count": len(copied_test_rows),
        "test_count_match": test_count_match, "test_file_hash_match": test_hash_match,
        "test_manifest_order_match": test_order_match, "test_manifest_copy_match": test_manifest_copy_match,
        "protected_metadata_unchanged": before == after,
        "clean_fingerprint_matches": (root / "dataset_integrity_reports/clean/dataset_fingerprint.sha256").read_text().strip() == EXPECTED["clean"],
        "combined_fingerprint_matches": (root / "dataset_integrity_reports/original/dataset_fingerprint.sha256").read_text().strip() == EXPECTED["combined"],
        "float32_tflite_matches": sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float32.tflite") == EXPECTED["float32"],
        "float16_tflite_matches": sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float16.tflite") == EXPECTED["float16"],
    }
    (metadata / "dataset_v2_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    failed = any([missing, corrupt, hash_mismatches, duplicate_destinations, exact_cross_split, group_leakage, new_test_exact, new_test_near, new_test, not test_count_match, not test_hash_match, not test_order_match, not test_manifest_copy_match, before != after])
    if failed:
        raise SystemExit("Dataset V2 validation failed; inspect metadata/dataset_v2_validation.json")
    print(json.dumps({
        "dataset_v2": str(destination), "original": 4801, "new": 135,
        "total": len(v2_rows), "train": split_totals["train"],
        "validation": split_totals["validation"], "test": split_totals["test"],
        "new_train": len(new_train), "new_validation": len(new_val), "new_test": len(new_test),
        "new_venomous_train": sum(row["class_name"] == "Venomous_Snake" for row in new_train),
        "new_venomous_validation": sum(row["class_name"] == "Venomous_Snake" for row in new_val),
        "new_nonvenomous_train": sum(row["class_name"] == "Non_Venomous_Snake" for row in new_train),
        "new_nonvenomous_validation": sum(row["class_name"] == "Non_Venomous_Snake" for row in new_val),
        "fingerprint": fingerprint, "status": "COMPLETE",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
