import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# combined_dataset is outside anviksa_ai
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "combined_dataset")

SPLITS = ["train", "validation", "test"]

print("=" * 70)
print("ANVIKSA AI - 7 CLASS DATASET CHECK")
print("=" * 70)

for split in SPLITS:
    split_dir = os.path.join(DATASET_DIR, split)

    print(f"\n{split.upper()}")
    print("-" * 70)

    if not os.path.exists(split_dir):
        print(f"ERROR: Folder not found: {split_dir}")
        continue

    total = 0

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)

        if os.path.isdir(class_dir):
            image_count = len([
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ])

            total += image_count
            print(f"{class_name:<25} : {image_count}")

    print("-" * 70)
    print(f"TOTAL {split.upper()} IMAGES: {total}")

print("\n" + "=" * 70)
print("DATASET CHECK COMPLETED")
print("=" * 70)