import os

# ============================================================
# ANVIKSA AI - SNAKE TRAINING CLEANUP
# Removes validation images that were COPIED from train
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# combined_dataset is outside the anviksa_ai folder
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "combined_dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")

SNAKE_CLASSES = [
    "Venomous_Snake",
    "Non_Venomous_Snake"
]

print("=" * 60)
print("ANVIKSA AI - SNAKE TRAINING CLEANUP")
print("=" * 60)

for class_name in SNAKE_CLASSES:

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)

    print(f"\nProcessing: {class_name}")
    print("-" * 60)

    if not os.path.exists(train_class_dir):
        print(f"ERROR: Training folder not found:")
        print(train_class_dir)
        continue

    if not os.path.exists(val_class_dir):
        print(f"ERROR: Validation folder not found:")
        print(val_class_dir)
        continue

    train_files = set(os.listdir(train_class_dir))
    val_files = set(os.listdir(val_class_dir))

    common_files = train_files.intersection(val_files)

    print(f"Training images before cleanup : {len(train_files)}")
    print(f"Validation images              : {len(val_files)}")
    print(f"Duplicate images found         : {len(common_files)}")

    removed = 0

    for filename in common_files:
        train_file = os.path.join(train_class_dir, filename)

        try:
            os.remove(train_file)
            removed += 1
        except Exception as e:
            print(f"Could not remove {filename}: {e}")

    remaining = len(os.listdir(train_class_dir))

    print(f"Removed from training          : {removed}")
    print(f"Training images after cleanup  : {remaining}")

print("\n" + "=" * 60)
print("SNAKE TRAINING CLEANUP COMPLETED")
print("=" * 60)