import os

# ============================================================
# ANVIKSA AI - DATASET VERIFICATION
# Compares source datasets with combined_dataset
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

COMBINED_DIR = os.path.join(PARENT_DIR, "combined_dataset")

ANIMAL_SOURCE = os.path.join(
    PARENT_DIR,
    "archive (1)",
    "Dataset"
)

SNAKE_SOURCE = os.path.join(
    PARENT_DIR,
    "archive (2)",
    "Snake Images"
)

# Combined dataset class -> source dataset class
CLASS_MAPPING = {
    "Cow": ("animal", "Cow"),
    "Deer": ("animal", "Deer"),
    "Elephant": ("animal", "Elephant"),
    "Monkey": ("animal", "Monkey"),
    "Wild_Boar": ("animal", "Wild Boar"),
    "Venomous_Snake": ("snake", "Venomous"),
    "Non_Venomous_Snake": ("snake", "Non Venomous"),
}

VALID_EXTENSIONS = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
)

def count_files(folder):
    if not os.path.exists(folder):
        return 0

    return len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(VALID_EXTENSIONS)
    ])


print("=" * 85)
print("ANVIKSA AI - COMPLETE DATASET VERIFICATION")
print("=" * 85)

splits = ["train", "validation", "test"]

for combined_class, source_info in CLASS_MAPPING.items():

    source_type, source_class = source_info

    print(f"\nCLASS: {combined_class}")
    print("-" * 85)

    for split in splits:

        combined_folder = os.path.join(
            COMBINED_DIR,
            split,
            combined_class
        )

        combined_count = count_files(combined_folder)

        if source_type == "animal":

            source_folder = os.path.join(
                ANIMAL_SOURCE,
                split,
                source_class
            )

            source_count = count_files(source_folder)

        else:
            # Snake source has only train and test
            if split == "validation":
                source_count = "Created from train"
            else:
                source_folder = os.path.join(
                    SNAKE_SOURCE,
                    split,
                    source_class
                )

                source_count = count_files(source_folder)

        print(f"{split.upper():12} | "
              f"Source: {str(source_count):<20} | "
              f"Combined: {combined_count}")

print("\n" + "=" * 85)
print("VERIFICATION COMPLETED")
print("=" * 85)