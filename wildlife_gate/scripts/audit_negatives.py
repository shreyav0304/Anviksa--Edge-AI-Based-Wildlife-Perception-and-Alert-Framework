from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "wildlife_gate" / "raw_non_wildlife"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lower = stricter duplicate detection.
# This is only an audit; nothing is deleted.
SIMILARITY_THRESHOLD = 4.0


def fingerprint(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    return small.astype(np.float32)


def mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    print("=" * 65)
    print("ANVIKSA AI - NON-WILDLIFE NEAR-DUPLICATE AUDIT")
    print("=" * 65)

    total_images = 0
    total_pairs = 0

    for category_dir in sorted(RAW_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue

        paths = sorted(
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        fingerprints = []

        for path in paths:
            fp = fingerprint(path)

            if fp is None:
                print(f"[WARNING] Could not read: {path}")
                continue

            fingerprints.append((path, fp))

        near_duplicates = []

        # Compare consecutive captures first because webcam near-duplicates
        # are most likely to occur next to each other in timestamp order.
        for index in range(1, len(fingerprints)):
            previous_path, previous_fp = fingerprints[index - 1]
            current_path, current_fp = fingerprints[index]

            difference = mean_difference(previous_fp, current_fp)

            if difference < SIMILARITY_THRESHOLD:
                near_duplicates.append(
                    (previous_path.name, current_path.name, difference)
                )

        print()
        print(f"Category        : {category_dir.name}")
        print(f"Readable images : {len(fingerprints)}")
        print(f"Near-dup pairs  : {len(near_duplicates)}")

        for first, second, difference in near_duplicates[:10]:
            print(
                f"  POSSIBLE DUPLICATE | difference={difference:.3f}"
            )
            print(f"    {first}")
            print(f"    {second}")

        if len(near_duplicates) > 10:
            print(
                f"  ... {len(near_duplicates) - 10} additional pair(s)"
            )

        total_images += len(fingerprints)
        total_pairs += len(near_duplicates)

    print()
    print("=" * 65)
    print(f"TOTAL IMAGES              : {total_images}")
    print(f"POSSIBLE NEAR-DUP PAIRS   : {total_pairs}")
    print("=" * 65)
    print("AUDIT ONLY - no images were deleted.")


if __name__ == "__main__":
    main()