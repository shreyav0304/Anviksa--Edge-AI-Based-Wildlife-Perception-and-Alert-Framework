"""Audit and safely version the Anviksa AI image dataset.

The tool never mutates its source dataset. It can produce read-only audit
reports or construct a new dataset by copying approved source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


CLASS_NAMES = (
    "Cow",
    "Deer",
    "Elephant",
    "Monkey",
    "Non_Venomous_Snake",
    "Venomous_Snake",
    "Wild_Boar",
)
SPLITS = ("train", "validation", "test")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2}


@dataclass(frozen=True)
class ImageRecord:
    """Reproducible metadata for one valid source image."""

    relative_path: str
    split: str
    class_name: str
    sha256: str
    width: int
    height: int
    image_format: str
    file_size: int
    phash: str
    dhash: str


@dataclass(frozen=True)
class NearDuplicatePair:
    """A conservative perceptual-hash duplicate candidate."""

    first: str
    second: str
    first_split: str
    second_split: str
    first_class: str
    second_class: str
    phash_distance: int
    dhash_distance: int
    confidence: str

    @property
    def cross_split(self) -> bool:
        return self.first_split != self.second_split


def sha256_file(path: Path) -> str:
    """Calculate a file hash without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(math.pi * (positions + 0.5) * frequencies / size)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


DCT_32 = _dct_matrix(32)


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    width = math.ceil(bits.size / 4)
    return f"{value:0{width}x}"


def perceptual_hash(image: Image.Image) -> str:
    """Return a 64-bit pHash robust to modest encoding and size changes."""
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    transformed = DCT_32 @ pixels @ DCT_32.T
    low_frequency = transformed[:8, :8]
    threshold = float(np.median(low_frequency.reshape(-1)[1:]))
    return _bits_to_hex(low_frequency > threshold)


def difference_hash(image: Image.Image) -> str:
    """Return a 64-bit horizontal difference hash."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return _bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def inspect_dataset(dataset_root: Path) -> dict:
    """Decode and inventory a dataset without modifying it."""
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    records: list[ImageRecord] = []
    corrupt_images: list[dict[str, str]] = []
    unsupported_files: list[str] = []
    missing_classes: list[dict[str, str]] = []
    empty_classes: list[dict[str, str]] = []
    unexpected_classes: list[dict[str, str]] = []

    for split in SPLITS:
        split_path = dataset_root / split
        if not split_path.is_dir():
            for class_name in CLASS_NAMES:
                missing_classes.append({"split": split, "class": class_name})
            continue

        actual_directories = {
            path.name for path in split_path.iterdir() if path.is_dir()
        }
        for class_name in sorted(actual_directories - set(CLASS_NAMES)):
            unexpected_classes.append({"split": split, "class": class_name})

        for class_name in CLASS_NAMES:
            class_path = split_path / class_name
            if not class_path.is_dir():
                missing_classes.append({"split": split, "class": class_name})
                continue

            class_files = sorted(
                (path for path in class_path.rglob("*") if path.is_file()),
                key=lambda path: path.as_posix().lower(),
            )
            supported_count = 0
            for path in class_files:
                relative_path = path.relative_to(dataset_root).as_posix()
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    unsupported_files.append(relative_path)
                    continue

                try:
                    with Image.open(path) as image:
                        image.load()
                        width, height = image.size
                        image_format = image.format or "UNKNOWN"
                        phash = perceptual_hash(image)
                        dhash = difference_hash(image)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    corrupt_images.append(
                        {"path": relative_path, "error": str(exc)}
                    )
                    continue

                records.append(
                    ImageRecord(
                        relative_path=relative_path,
                        split=split,
                        class_name=class_name,
                        sha256=sha256_file(path),
                        width=width,
                        height=height,
                        image_format=image_format,
                        file_size=path.stat().st_size,
                        phash=phash,
                        dhash=dhash,
                    )
                )
                supported_count += 1

            if supported_count == 0:
                empty_classes.append({"split": split, "class": class_name})

    records.sort(key=lambda record: record.relative_path)
    return {
        "dataset_root": str(dataset_root),
        "records": records,
        "corrupt_images": corrupt_images,
        "unsupported_files": sorted(unsupported_files),
        "missing_classes": missing_classes,
        "empty_classes": empty_classes,
        "unexpected_classes": unexpected_classes,
    }


def exact_duplicate_groups(records: Sequence[ImageRecord]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record.relative_path)
    return [sorted(paths) for paths in groups.values() if len(paths) > 1]


def find_near_duplicates(
    records: Sequence[ImageRecord],
) -> list[NearDuplicatePair]:
    """Find conservative candidates using two independent 64-bit hashes.

    Exact duplicates are excluded. High-confidence candidates require very
    small distances in both hashes. Wider matches are review-only candidates.
    """
    candidates: list[NearDuplicatePair] = []
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            if first.sha256 == second.sha256:
                continue

            phash_distance = hash_distance(first.phash, second.phash)
            dhash_distance = hash_distance(first.dhash, second.dhash)

            if phash_distance <= 2 and dhash_distance <= 2:
                confidence = "high"
            elif (
                phash_distance <= 6
                and dhash_distance <= 8
            ) or (
                phash_distance <= 10
                and dhash_distance <= 4
            ):
                confidence = "review"
            else:
                continue

            candidates.append(
                NearDuplicatePair(
                    first=first.relative_path,
                    second=second.relative_path,
                    first_split=first.split,
                    second_split=second.split,
                    first_class=first.class_name,
                    second_class=second.class_name,
                    phash_distance=phash_distance,
                    dhash_distance=dhash_distance,
                    confidence=confidence,
                )
            )

    return sorted(
        candidates,
        key=lambda pair: (
            pair.confidence != "high",
            not pair.cross_split,
            pair.phash_distance + pair.dhash_distance,
            pair.first,
            pair.second,
        ),
    )


def distribution(records: Iterable[ImageRecord]) -> dict[str, dict[str, int]]:
    counts = {
        class_name: {split: 0 for split in SPLITS} for class_name in CLASS_NAMES
    }
    for record in records:
        counts[record.class_name][record.split] += 1
    for class_name in CLASS_NAMES:
        counts[class_name]["total"] = sum(
            counts[class_name][split] for split in SPLITS
        )
    return counts


def dataset_fingerprint(records: Sequence[ImageRecord]) -> str:
    """Hash class order and canonical manifest content."""
    digest = hashlib.sha256()
    digest.update(json.dumps(CLASS_NAMES).encode("utf-8"))
    for record in sorted(records, key=lambda item: item.relative_path):
        canonical = {
            "relative_path": record.relative_path,
            "split": record.split,
            "class_name": record.class_name,
            "sha256": record.sha256,
            "width": record.width,
            "height": record.height,
        }
        digest.update(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _split_duplicate_groups(
    groups: Sequence[Sequence[str]], record_lookup: dict[str, ImageRecord]
) -> tuple[list[list[str]], list[list[str]]]:
    cross_split: list[list[str]] = []
    within_split: list[list[str]] = []
    for group in groups:
        splits = {record_lookup[path].split for path in group}
        if len(splits) > 1:
            cross_split.append(list(group))
        else:
            within_split.append(list(group))
    return cross_split, within_split


def build_report(audit: dict) -> dict:
    records: list[ImageRecord] = audit["records"]
    lookup = {record.relative_path: record for record in records}
    exact_groups = exact_duplicate_groups(records)
    cross_exact, within_exact = _split_duplicate_groups(exact_groups, lookup)
    near_pairs = find_near_duplicates(records)
    counts = distribution(records)
    class_totals = [counts[name]["total"] for name in CLASS_NAMES]
    imbalance_ratio = max(class_totals) / min(class_totals)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": audit["dataset_root"],
        "dataset_fingerprint": dataset_fingerprint(records),
        "class_order": list(CLASS_NAMES),
        "total_images": len(records),
        "split_counts": {
            split: sum(record.split == split for record in records)
            for split in SPLITS
        },
        "class_distribution": counts,
        "imbalance_ratio": imbalance_ratio,
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_group_count": len(exact_groups),
        "cross_split_exact_duplicate_groups": cross_exact,
        "cross_split_exact_duplicate_group_count": len(cross_exact),
        "within_split_exact_duplicate_groups": within_exact,
        "within_split_exact_duplicate_group_count": len(within_exact),
        "near_duplicate_candidates": [
            {**asdict(pair), "cross_split": pair.cross_split}
            for pair in near_pairs
        ],
        "near_duplicate_candidate_count": len(near_pairs),
        "cross_split_near_duplicate_candidate_count": sum(
            pair.cross_split for pair in near_pairs
        ),
        "high_confidence_near_duplicate_count": sum(
            pair.confidence == "high" for pair in near_pairs
        ),
        "corrupt_images": audit["corrupt_images"],
        "unsupported_files": audit["unsupported_files"],
        "missing_classes": audit["missing_classes"],
        "empty_classes": audit["empty_classes"],
        "unexpected_classes": audit["unexpected_classes"],
    }


def write_manifests(records: Sequence[ImageRecord], output_dir: Path) -> None:
    fields = tuple(asdict(records[0]).keys()) if records else tuple(
        ImageRecord.__dataclass_fields__.keys()
    )
    for split in SPLITS:
        split_records = [record for record in records if record.split == split]
        path = output_dir / f"{split}_manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(record) for record in split_records)

    with (output_dir / "complete_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for record in records:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    with (output_dir / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(list(CLASS_NAMES), file, indent=2)
        file.write("\n")


def write_distribution_csv(report: dict, output_dir: Path) -> None:
    path = output_dir / "class_distribution.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        fields = ("class", "train", "validation", "test", "total")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for class_name in CLASS_NAMES:
            counts = report["class_distribution"][class_name]
            writer.writerow({"class": class_name, **counts})


def write_markdown_report(report: dict, output_dir: Path) -> None:
    lines = [
        "# Dataset Integrity Report",
        "",
        f"- Dataset: `{report['dataset_root']}`",
        f"- Fingerprint: `{report['dataset_fingerprint']}`",
        f"- Total valid images: {report['total_images']}",
        f"- Exact duplicate groups: {report['exact_duplicate_group_count']}",
        "- Cross-split exact duplicate groups: "
        f"{report['cross_split_exact_duplicate_group_count']}",
        "- Near-duplicate candidates: "
        f"{report['near_duplicate_candidate_count']}",
        "- Cross-split near-duplicate candidates: "
        f"{report['cross_split_near_duplicate_candidate_count']}",
        f"- Corrupt images: {len(report['corrupt_images'])}",
        f"- Imbalance ratio: {report['imbalance_ratio']:.4f}",
        "",
        "## Class distribution",
        "",
        "| Class | Train | Validation | Test | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in CLASS_NAMES:
        counts = report["class_distribution"][class_name]
        lines.append(
            f"| {class_name} | {counts['train']} | {counts['validation']} | "
            f"{counts['test']} | {counts['total']} |"
        )

    lines.extend(["", "## Exact duplicate groups", ""])
    if report["exact_duplicate_groups"]:
        for index, group in enumerate(report["exact_duplicate_groups"], 1):
            lines.append(f"### Group {index}")
            lines.extend(f"- `{path}`" for path in group)
            lines.append("")
    else:
        lines.append("None.")

    lines.extend(["", "## Near-duplicate candidates", ""])
    if report["near_duplicate_candidates"]:
        for index, pair in enumerate(report["near_duplicate_candidates"], 1):
            lines.extend(
                [
                    f"### Candidate {index} ({pair['confidence']})",
                    f"- `{pair['first']}`",
                    f"- `{pair['second']}`",
                    f"- pHash distance: {pair['phash_distance']}",
                    f"- dHash distance: {pair['dhash_distance']}",
                    f"- Cross split: {pair['cross_split']}",
                    "",
                ]
            )
    else:
        lines.append("None under the configured conservative thresholds.")

    (output_dir / "integrity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_audit(dataset_root: Path, output_dir: Path) -> tuple[dict, dict]:
    if output_dir.exists():
        raise FileExistsError(f"Audit output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    audit = inspect_dataset(dataset_root)
    report = build_report(audit)
    write_manifests(audit["records"], output_dir)
    write_distribution_csv(report, output_dir)
    with (output_dir / "integrity_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    write_markdown_report(report, output_dir)
    (output_dir / "dataset_fingerprint.sha256").write_text(
        report["dataset_fingerprint"] + "\n", encoding="utf-8"
    )
    return audit, report


def choose_duplicate_exclusions(
    records: Sequence[ImageRecord],
    confirmed_near_pairs: Sequence[dict],
) -> tuple[set[str], list[dict]]:
    """Resolve exact and explicitly confirmed groups into one split."""
    lookup = {record.relative_path: record for record in records}
    parent = {path: path for path in lookup}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for group in exact_duplicate_groups(records):
        class_names = {lookup[path].class_name for path in group}
        if len(class_names) != 1:
            raise ValueError(
                "Exact duplicate content has conflicting class labels and "
                f"requires manual review: {group}"
            )
        for path in group[1:]:
            union(group[0], path)

    for pair in confirmed_near_pairs:
        first = pair.get("first")
        second = pair.get("second")
        if first not in lookup or second not in lookup:
            raise ValueError(
                "Confirmed near-duplicate decision references an unknown "
                f"path: {first!r}, {second!r}"
            )
        if lookup[first].class_name != lookup[second].class_name:
            raise ValueError(
                "Confirmed near-duplicate pair crosses class labels and "
                f"requires separate review: {first}, {second}"
            )
        union(first, second)

    components: dict[str, list[str]] = defaultdict(list)
    for path in lookup:
        components[find(path)].append(path)

    exclusions: set[str] = set()
    resolutions: list[dict] = []
    for group in components.values():
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda path: (
                SPLIT_PRIORITY[lookup[path].split],
                path.lower(),
            ),
        )
        retained = ordered[0]
        excluded = ordered[1:]
        exclusions.update(excluded)
        resolutions.append(
            {
                "group": sorted(group),
                "retained": retained,
                "excluded_from_clean_copy": excluded,
                "reason": (
                    "Exact SHA-256 or explicitly confirmed perceptual "
                    "duplicate; evaluation-split priority"
                ),
            }
        )

    return exclusions, resolutions


def build_clean_dataset(
    source_root: Path,
    destination_root: Path,
    build_output: Path,
    confirmed_near_path: Path,
) -> None:
    """Copy a clean exact-deduplicated dataset without touching the source."""
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    build_output = build_output.resolve()

    if destination_root.exists():
        raise FileExistsError(
            f"Clean dataset destination already exists: {destination_root}"
        )
    if build_output.exists():
        raise FileExistsError(f"Build output already exists: {build_output}")
    if destination_root == source_root:
        raise ValueError("Source and clean dataset paths must be different.")

    audit = inspect_dataset(source_root)
    if audit["corrupt_images"] or audit["unsupported_files"]:
        raise ValueError(
            "Source contains corrupt or unsupported files. Review the audit "
            "before constructing a clean dataset."
        )
    if audit["missing_classes"] or audit["empty_classes"]:
        raise ValueError(
            "Source has missing or empty classes. Review the audit first."
        )

    records: list[ImageRecord] = audit["records"]
    try:
        confirmed_payload = json.loads(
            confirmed_near_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read confirmed near-duplicate decisions: {exc}"
        ) from exc
    confirmed_pairs = confirmed_payload.get("confirmed_pairs")
    if not isinstance(confirmed_pairs, list):
        raise ValueError(
            "Confirmed near-duplicate file must contain a confirmed_pairs list."
        )

    exclusions, resolutions = choose_duplicate_exclusions(
        records, confirmed_pairs
    )
    build_output.mkdir(parents=True)

    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (destination_root / split / class_name).mkdir(parents=True)

    copied = 0
    for record in records:
        if record.relative_path in exclusions:
            continue
        source = source_root / Path(record.relative_path)
        destination = destination_root / Path(record.relative_path)
        shutil.copy2(source, destination)
        copied += 1

    build_record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "source_fingerprint": dataset_fingerprint(records),
        "source_images": len(records),
        "copied_images": copied,
        "excluded_duplicate_images": len(exclusions),
        "duplicate_resolutions": resolutions,
        "confirmed_near_duplicate_pairs": len(confirmed_pairs),
        "near_duplicates_automatically_resolved": sum(
            pair.get("resolution_mode") == "automatic_high_confidence"
            for pair in confirmed_pairs
        ),
        "near_duplicates_manually_confirmed": sum(
            pair.get("resolution_mode") == "manual_visual_confirmation"
            for pair in confirmed_pairs
        ),
        "confirmed_near_duplicate_decisions": confirmed_pairs,
        "policy": (
            "One file retained per connected exact or explicitly confirmed "
            "near-duplicate group. Test is preferred, then validation, then "
            "train. Unconfirmed candidates are not excluded."
        ),
    }
    with (build_output / "clean_build_record.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(build_record, file, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit a dataset")
    audit_parser.add_argument("--dataset", required=True, type=Path)
    audit_parser.add_argument("--output", required=True, type=Path)

    build_parser = subparsers.add_parser(
        "build-clean", help="Create a non-destructive exact-deduplicated copy"
    )
    build_parser.add_argument("--source", required=True, type=Path)
    build_parser.add_argument("--destination", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument(
        "--confirmed-near", required=True, type=Path
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "audit":
            _, report = save_audit(args.dataset, args.output)
            print(json.dumps(report, indent=2))
        else:
            build_clean_dataset(
                args.source,
                args.destination,
                args.output,
                args.confirmed_near,
            )
            print(f"Clean dataset created: {args.destination.resolve()}")
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
