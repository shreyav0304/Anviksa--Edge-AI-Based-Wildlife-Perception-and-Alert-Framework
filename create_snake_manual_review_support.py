#!/usr/bin/env python3
"""Create offline manual-review support for unresolved snake images.

Reads existing audit/resolution evidence and source images. Writes only under
results/new_snake_image_manual_review. It never changes source or protected
assets and never makes human decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DECISION_FIELDS = [
    "review_id", "image_path", "species", "parent_class", "review_type",
    "suspected_match_path", "suspected_match_split", "frozen_test_involved",
    "similarity_summary", "contact_sheet_path", "human_decision", "human_notes",
    "final_recommended_status",
]
ALLOWED_DECISIONS = [
    "KEEP_TECHNICALLY_ELIGIBLE", "EXCLUDE_DUPLICATE",
    "EXCLUDE_FROZEN_TEST_LEAKAGE", "KEEP_LABEL_REVIEW",
    "EXCLUDE_LABEL_CONFLICT", "KEEP_MANUAL_REVIEW", "UNRESOLVED",
]
EXPECTED_HASHES = {
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
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_snapshot(paths: list[Path]) -> dict:
    return {
        str(base): [
            [str(path), path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.as_posix().lower())
        ] if base.exists() else []
        for base in paths
    }


def wrap(draw: ImageDraw.ImageDraw, text: str, width: int = 88) -> list[str]:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=True) or [""])
    return lines


def first_frame(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.seek(0)
        return ImageOps.exif_transpose(image).convert("RGB")


def contact_sheet(item: dict, destination: Path) -> None:
    width, height = 1400, 860
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width - 1, height - 1), outline="#333333", width=2)
    title = f"{item['review_id']} | {item['review_type']}"
    draw.text((24, 18), title, fill="#111111", font=font)
    draw.text((24, 42), f"NEW: {item['relative_path']}", fill="#111111", font=font)
    draw.text((24, 64), f"PARENT CLASS: {item['parent_class']}    SPECIES FOLDER A: {item['species']}", fill="#111111", font=font)
    if item.get("match_species"):
        draw.text((24, 86), f"SPECIES FOLDER B: {item['match_species']}    SAME PARENT: {item.get('same_parent', '')}", fill="#8a2d00", font=font)
    match_heading = "FROZEN TEST IMAGE" if item["suspected_match_split"] == "TEST" else "SUSPECTED MATCH"
    panels = [("NEW IMAGE", Path(item["image_path"]), "#005a9c"), (match_heading, Path(item["absolute_match_path"]), "#a40000" if item["suspected_match_split"] == "TEST" else "#005a9c")]
    if item.get("secondary_absolute_match_path"):
        panels.append(("FROZEN TEST IMAGE", Path(item["secondary_absolute_match_path"]), "#a40000"))
    panel_width = 650 if len(panels) == 2 else 432
    panel_gap = 50 if len(panels) == 2 else 26
    for panel_index, (heading, path, color) in enumerate(panels):
        x = 24 + panel_index * (panel_width + panel_gap)
        draw.text((x + panel_width // 2 - 45, 124), heading, fill=color, font=font)
        image = first_frame(path)
        tile = ImageOps.contain(image, (panel_width, 500), Image.Resampling.LANCZOS)
        paste_x = x + (panel_width - tile.width) // 2
        paste_y = 150 + (500 - tile.height) // 2
        canvas.paste(tile, (paste_x, paste_y))
        draw.rectangle((x, 150, x + panel_width, 650), outline="#777777", width=1)
    draw.text((724, 658), f"MATCH: {item['suspected_match_path']} [{item['suspected_match_split']}]", fill="#111111", font=font)
    if item.get("secondary_match_path"):
        draw.text((24, 676), f"ADDITIONAL FROZEN TEST MATCH: {item['secondary_match_path']} [TEST]", fill="#a40000", font=font)
    evidence = [
        f"SIMILARITY: {item['similarity_summary']}",
        f"FROZEN-TEST INVOLVEMENT: {item['frozen_test_involved']}",
        f"CURRENT REASON: {item['current_reason']}",
        "HUMAN OPTIONS: SAME IMAGE/DERIVATIVE | CLEARLY DIFFERENT | POSSIBLE TEST LEAKAGE | LABEL CONFLICT | UNCERTAIN—KEEP EXCLUDED",
    ]
    y = 708 if item.get("secondary_match_path") else 690
    for entry in evidence:
        for line in wrap(draw, entry, 145):
            draw.text((24, y), line, fill="#111111", font=font)
            y += 18
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "new_training_data/snake_images/Indian-Snakes-Dataset-master"
    clean = root / "clean_dataset"
    resolution = root / "results/new_snake_image_resolution"
    output = root / "results/new_snake_image_manual_review"
    sheets = output / "contact_sheets"
    output.mkdir(parents=True, exist_ok=True)
    protected = [
        root / "clean_dataset", root / "combined_dataset", root / "dataset_integrity_reports",
        root / "models", root / "experiments", root / "pi_deployment",
        root / "results/custom_cnn", root / "results/mobilenet_v2",
        root / "results/mobilenet_v3_large", root / "results/efficientnet_b0",
        root / "results/deployment", root / "results/model_comparison_dashboard",
        root / "ui", root / "snake_video_dataset",
    ]
    before = metadata_snapshot(protected)

    final = read_csv(resolution / "final_image_resolution.csv")
    unresolved = [row for row in final if row["technical_status"] in {"MANUAL_REVIEW_REQUIRED", "LABEL_REVIEW_REQUIRED"}]
    if len(unresolved) != 324:
        raise SystemExit(f"Expected 324 unresolved images, found {len(unresolved)}")
    manual_count = sum(row["technical_status"] == "MANUAL_REVIEW_REQUIRED" for row in unresolved)
    label_final_count = sum(row["technical_status"] == "LABEL_REVIEW_REQUIRED" for row in unresolved)
    if (manual_count, label_final_count) != (322, 2):
        raise SystemExit(f"Expected 322 manual + 2 label cases, found {manual_count} + {label_final_count}")

    near = read_csv(resolution / "near_duplicate_resolution.csv")
    leakage = read_csv(resolution / "frozen_test_leakage_resolution.csv")
    conflicts = read_csv(resolution / "species_conflict_resolution.csv")
    near_by_new = defaultdict(list)
    for row in near:
        near_by_new[row["new_image"]].append(row)
    leakage_by_new = defaultdict(list)
    for row in leakage:
        leakage_by_new[row["new_image"]].append(row)
    conflict_by_image = defaultdict(list)
    for row in conflicts:
        images = row["images"].split(" | ")
        for image_path in images:
            conflict_by_image[image_path].append((row, images))

    items = []
    for index, row in enumerate(sorted(unresolved, key=lambda x: x["relative_path"].lower()), 1):
        rel = row["relative_path"]
        review_id = f"REVIEW_{index:04d}"
        if row["technical_status"] == "LABEL_REVIEW_REQUIRED":
            review_type, folder = "LABEL_REVIEW", "label_review"
            conflict, images = conflict_by_image[rel][0]
            match_rel = next(path for path in images if path != rel)
            match_path = source / match_rel
            match_split = "NEW_DATASET"
            summary = f"{conflict['conflict_type']}; {conflict['reason']}"
            match_species = match_rel.split("/")[1]
            same_parent = "YES" if rel.split("/")[0] == match_rel.split("/")[0] else "NO"
            secondary_match_rel = ""
            secondary_match_path = ""
            if row["frozen_test_leakage_status"] == "POSSIBLE_TEST_LEAKAGE":
                test_candidates = leakage_by_new[rel]
                test_candidates.sort(key=lambda x: (x.get("leakage_status") != "LIKELY_LEAKAGE_RISK", -float(x.get("ssim") or 0), -int(x.get("orb_ransac_inliers") or 0)))
                secondary_match_rel = test_candidates[0]["test_image"]
                secondary_match_path = str(clean / secondary_match_rel)
        elif row["frozen_test_leakage_status"] == "POSSIBLE_TEST_LEAKAGE":
            review_type, folder = "FROZEN_TEST_REVIEW", "frozen_test_review"
            candidates = leakage_by_new[rel]
            candidates.sort(key=lambda x: (x.get("leakage_status") != "LIKELY_LEAKAGE_RISK", -float(x.get("ssim") or 0), -int(x.get("orb_ransac_inliers") or 0)))
            evidence = candidates[0]
            match_rel = evidence["test_image"]
            match_path = clean / match_rel
            match_split = "TEST"
            summary = evidence["evidence"]
            match_species, same_parent = "", ""
            secondary_match_rel, secondary_match_path = "", ""
        else:
            review_type, folder = "DUPLICATE_REVIEW", "duplicate_review"
            candidates = near_by_new[rel]
            candidates.sort(key=lambda x: (x.get("final_classification") != "LIKELY_NEAR_DUPLICATE", -float(x.get("ssim") or 0), -int(x.get("orb_ransac_inliers") or 0)))
            if not candidates:
                raise SystemExit(f"No comparison evidence for manual case: {rel}")
            evidence = candidates[0]
            match_rel = evidence["existing_image"]
            match_path = clean / match_rel
            match_split = evidence["existing_split"]
            summary = evidence["evidence"]
            match_species, same_parent = "", ""
            secondary_match_rel, secondary_match_path = "", ""
        relative_sheet = Path("contact_sheets") / folder / f"review_{index:04d}_{folder}.png"
        item = {
            "review_id": review_id, "relative_path": rel,
            "image_path": str(source / rel), "species": row["species"],
            "parent_class": row["parent_class"], "review_type": review_type,
            "suspected_match_path": match_rel, "absolute_match_path": str(match_path),
            "suspected_match_split": match_split,
            "frozen_test_involved": "YES" if row["frozen_test_leakage_status"] == "POSSIBLE_TEST_LEAKAGE" else "NO",
            "similarity_summary": summary, "contact_sheet_path": relative_sheet.as_posix(),
            "current_reason": row["notes"], "match_species": match_species,
            "same_parent": same_parent,
            "secondary_match_path": secondary_match_rel,
            "secondary_absolute_match_path": secondary_match_path,
        }
        contact_sheet(item, output / relative_sheet)
        items.append(item)
        if index % 50 == 0:
            print(f"Created {index}/{len(unresolved)} contact sheets", flush=True)

    decisions = [{
        "review_id": item["review_id"], "image_path": item["image_path"],
        "species": item["species"], "parent_class": item["parent_class"],
        "review_type": item["review_type"], "suspected_match_path": item["suspected_match_path"],
        "suspected_match_split": item["suspected_match_split"],
        "frozen_test_involved": item["frozen_test_involved"],
        "similarity_summary": item["similarity_summary"],
        "contact_sheet_path": item["contact_sheet_path"],
        "human_decision": "", "human_notes": "", "final_recommended_status": "",
    } for item in items]
    write_csv(output / "manual_decisions.csv", decisions, DECISION_FIELDS)

    guide = f"""# ANVIKSA Snake Image Manual Review Guide

This review resolves technical duplicate, leakage, and metadata ambiguity only. It does **not** establish copyright or training reuse rights.

## Allowed values for `human_decision`

Use exactly one of these values:

- `KEEP_TECHNICALLY_ELIGIBLE`: Images clearly show different underlying photographs or scenes.
- `EXCLUDE_DUPLICATE`: Same photograph or an obvious resized, cropped, recompressed, watermarked, or minimally edited derivative.
- `EXCLUDE_FROZEN_TEST_LEAKAGE`: Same underlying photograph or derivative as the displayed frozen-test image.
- `KEEP_LABEL_REVIEW`: Technically unique, but species metadata remains unresolved.
- `EXCLUDE_LABEL_CONFLICT`: Conflicting species metadata makes the example unsuitable.
- `KEEP_MANUAL_REVIEW`: The case remains ambiguous and should stay outside integration.
- `UNRESOLVED`: The reviewer cannot confidently decide.

Leave `human_notes` free-form. Fill `final_recommended_status` only after review; it is intentionally blank initially.

## Review method

1. Compare the underlying scene, snake pose, background geometry, cropping, and distinctive objects.
2. Treat resized, compressed, color-adjusted, watermarked, or cropped forms of the same source photograph as duplicates.
3. For frozen-test sheets, use `EXCLUDE_FROZEN_TEST_LEAKAGE` if the new image derives from the displayed test photograph.
4. Clearly different scenes, poses, backgrounds, camera positions, or animals are not duplicates merely because they show the same species.
5. For label sheets, review the conflicting directory metadata. Do not infer species or venom status from appearance.
6. When confidence is insufficient, select `KEEP_MANUAL_REVIEW` or `UNRESOLVED`; do not clear the image automatically.

## Biological-label warning

Do not use head shape, pupil shape, color, body thickness, or other simplistic visual rules to determine venom status. Directory labels remain metadata, not independently verified biological ground truth.

## Rights status

All images retain:

- `repository_provenance = VERIFIED`
- `repository_license = MIT`
- `individual_image_rights = UNCLEAR`
- `training_reuse_rights = REVIEW_REQUIRED`

Human duplicate review does not resolve copyright.
"""
    (output / "REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")

    cards = []
    for item in items:
        cards.append(f"""<article><h2>{html.escape(item['review_id'])}</h2><a href="{html.escape(item['contact_sheet_path'])}"><img loading="lazy" src="{html.escape(item['contact_sheet_path'])}" alt="{html.escape(item['review_id'])}"></a><p><strong>{html.escape(item['review_type'])}</strong> · Frozen test: {item['frozen_test_involved']}</p><p>{html.escape(item['current_reason'])}</p></article>""")
    index_html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ANVIKSA Snake Manual Review</title><style>body{font-family:system-ui,sans-serif;margin:2rem;background:#f5f5f5;color:#222}header{max-width:1100px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem;max-width:1600px;margin:2rem auto}article{background:white;padding:1rem;border:1px solid #ccc;border-radius:8px}img{width:100%;height:auto;border:1px solid #ddd}h2{font-size:1rem}p{font-size:.9rem}</style></head><body><header><h1>ANVIKSA Snake Manual Review</h1><p>324 unresolved technical cases. Decisions must be recorded in manual_decisions.csv. Rights remain unclear.</p></header><main class="grid">""" + "\n".join(cards) + "</main></body></html>"
    (output / "review_index.html").write_text(index_html, encoding="utf-8")

    counts = Counter(item["review_type"] for item in items)
    frozen_related = sum(item["frozen_test_involved"] == "YES" for item in items)
    summary = f"""# ANVIKSA New Snake Image Manual-Review Support

TOTAL UNRESOLVED: {len(items)}

MANUAL REVIEW CASES: {manual_count}

LABEL REVIEW CASES: {label_final_count}

FROZEN-TEST-RELATED REVIEW CASES: {frozen_related}

DUPLICATE-RELATED REVIEW CASES: {counts['DUPLICATE_REVIEW']}

OTHER REVIEW CASES: {counts['OTHER_MANUAL_REVIEW']}

Note: frozen-test involvement overlaps sheet categories; one label-review case is also frozen-test-related. Contact-sheet categories themselves are mutually exclusive.

CONTACT SHEETS CREATED: {len(items)}

DECISION ROWS CREATED: {len(decisions)}

HTML REVIEW INDEX: CREATED

Rights status remains `UNCLEAR / REVIEW_REQUIRED`. No human decision was pre-filled.
"""
    (output / "manual_review_summary.md").write_text(summary, encoding="utf-8")

    after = metadata_snapshot(protected)
    clean_fp = (root / "dataset_integrity_reports/clean/dataset_fingerprint.sha256").read_text().strip()
    combined_fp = (root / "dataset_integrity_reports/original/dataset_fingerprint.sha256").read_text().strip()
    integrity = {
        "protected_metadata_unchanged": before == after,
        "clean_fingerprint_matches": clean_fp == EXPECTED_HASHES["clean"],
        "combined_fingerprint_matches": combined_fp == EXPECTED_HASHES["combined"],
        "float32_tflite_matches": sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float32.tflite") == EXPECTED_HASHES["float32"],
        "float16_tflite_matches": sha256_file(root / "pi_deployment/models/mobilenet_v3_large_float16.tflite") == EXPECTED_HASHES["float16"],
        "contact_sheets": len(items), "decision_rows": len(decisions),
        "blank_human_decisions": all(not row["human_decision"] for row in decisions),
        "blank_human_notes": all(not row["human_notes"] for row in decisions),
        "blank_final_recommended_status": all(not row["final_recommended_status"] for row in decisions),
    }
    (output / "integrity_verification.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
