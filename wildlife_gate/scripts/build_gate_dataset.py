from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2


SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WILDLIFE_SOURCE = PROJECT_ROOT / "clean_dataset"
NEGATIVE_SOURCE = PROJECT_ROOT / "wildlife_gate" / "raw_non_wildlife"
OUTPUT_ROOT = PROJECT_ROOT / "wildlife_gate" / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# We deliberately do NOT use clean_dataset/test.
# That test split belongs to the validated 7-class V1 experiment.
WILDLIFE_SPLITS = ("train", "validation")

NEGATIVE_CATEGORIES = (
    "human",
    "empty_scene",
    "indoor_objects",
    "outdoor_background",
)

OUTPUT_SPLITS = ("train", "validation", "test")

# Negative images were collected in short webcam sessions.
# Keep consecutive captures together so nearly identical frames
# cannot be scattered randomly across train/validation/test.
NEGATIVE_GROUP_SIZE = 5

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def readable(path: Path) -> bool:
    return cv2.imread(str(path)) is not None


def reset_output() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in OUTPUT_SPLITS:
        for label in ("Non_Wildlife", "Wildlife"):
            (OUTPUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)


def copy_image(source: Path, destination_dir: Path, prefix: str) -> Path:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
    destination = destination_dir / f"{prefix}_{digest}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return destination


def build_negative_groups() -> list[list[Path]]:
    groups: list[list[Path]] = []

    for category in NEGATIVE_CATEGORIES:
        files = [
            p for p in image_files(NEGATIVE_SOURCE / category)
            if readable(p)
        ]

        # Files are timestamp-named, so sorted order preserves capture order.
        for start in range(0, len(files), NEGATIVE_GROUP_SIZE):
            group = files[start:start + NEGATIVE_GROUP_SIZE]
            if group:
                groups.append(group)

    return groups


def assign_groups(
    groups: list[list[Path]],
) -> dict[str, list[Path]]:
    rng = random.Random(SEED)
    groups = list(groups)
    rng.shuffle(groups)

    total_images = sum(len(group) for group in groups)

    targets = {
        "train": total_images * TRAIN_RATIO,
        "validation": total_images * VALIDATION_RATIO,
        "test": total_images * TEST_RATIO,
    }

    assigned = {split: [] for split in OUTPUT_SPLITS}

    # Greedy group assignment by current deficit.
    for group in groups:
        deficits = {
            split: targets[split] - len(assigned[split])
            for split in OUTPUT_SPLITS
        }

        split = max(deficits, key=deficits.get)
        assigned[split].extend(group)

    return assigned


def collect_wildlife_pool() -> list[Path]:
    pool: list[Path] = []

    for split in WILDLIFE_SPLITS:
        split_dir = WILDLIFE_SOURCE / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Wildlife source split not found: {split_dir}"
            )

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            pool.extend(
                p for p in image_files(class_dir)
                if readable(p)
            )

    return pool


def main() -> None:
    print("=" * 68)
    print("ANVIKSA AI - WILDLIFE GATE DATASET BUILDER")
    print("=" * 68)

    reset_output()

    negative_groups = build_negative_groups()

    if not negative_groups:
        raise RuntimeError("No readable NON_WILDLIFE images were found.")

    negative_assignment = assign_groups(negative_groups)

    wildlife_pool = collect_wildlife_pool()

    if not wildlife_pool:
        raise RuntimeError("No readable wildlife images were found.")

    rng = random.Random(SEED)
    rng.shuffle(wildlife_pool)

    # Balance Wildlife against Non_Wildlife independently in each split.
    required_wildlife = sum(
        len(images) for images in negative_assignment.values()
    )

    if len(wildlife_pool) < required_wildlife:
        raise RuntimeError(
            f"Need {required_wildlife} wildlife images but only "
            f"{len(wildlife_pool)} are available."
        )

    cursor = 0
    manifest = {
        "seed": SEED,
        "negative_group_size": NEGATIVE_GROUP_SIZE,
        "wildlife_source_splits": list(WILDLIFE_SPLITS),
        "splits": {},
    }

    for split in OUTPUT_SPLITS:
        negatives = negative_assignment[split]
        wildlife_count = len(negatives)

        wildlife = wildlife_pool[cursor:cursor + wildlife_count]
        cursor += wildlife_count

        for index, source in enumerate(negatives, start=1):
            copy_image(
                source,
                OUTPUT_ROOT / split / "Non_Wildlife",
                f"negative_{index:04d}",
            )

        for index, source in enumerate(wildlife, start=1):
            copy_image(
                source,
                OUTPUT_ROOT / split / "Wildlife",
                f"wildlife_{index:04d}",
            )

        manifest["splits"][split] = {
            "Non_Wildlife": len(negatives),
            "Wildlife": len(wildlife),
            "total": len(negatives) + len(wildlife),
        }

    manifest_path = OUTPUT_ROOT / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print()
    for split in OUTPUT_SPLITS:
        info = manifest["splits"][split]
        print(
            f"{split.upper():10s} | "
            f"Non_Wildlife={info['Non_Wildlife']:3d} | "
            f"Wildlife={info['Wildlife']:3d} | "
            f"Total={info['total']:3d}"
        )

    print()
    print(f"Dataset : {OUTPUT_ROOT}")
    print(f"Manifest: {manifest_path}")
    print()
    print("[IMPORTANT]")
    print("clean_dataset/test was NOT used.")
    print("Original clean_dataset was NOT modified.")
    print("=" * 68)


if __name__ == "__main__":
    main()