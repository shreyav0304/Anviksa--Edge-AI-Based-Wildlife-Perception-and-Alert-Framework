import os
import shutil
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(r"C:\Users\S Yashveer\Downloads\animal datasets")

ANIMAL_DATASET = BASE_DIR / "archive (1)" / "Dataset"
SNAKE_DATASET = BASE_DIR / "archive (2)" / "Snake Images"

OUTPUT_DATASET = BASE_DIR / "combined_dataset"


# ==========================================
# CLASSES TO KEEP
# ==========================================

ANIMAL_CLASSES = {
    "Wild Boar": "Wild_Boar",
    "Monkey": "Monkey",
    "Elephant": "Elephant",
    "Deer": "Deer",
    "Cow": "Cow"
}

SNAKE_CLASSES = {
    "Venomous": "Venomous_Snake",
    "Non Venomous": "Non_Venomous_Snake"
}


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

splits = ["train", "validation", "test"]

for split in splits:
    for class_name in list(ANIMAL_CLASSES.values()) + list(SNAKE_CLASSES.values()):
        folder = OUTPUT_DATASET / split / class_name
        folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# COPY ANIMAL DATASET
# ==========================================

print("\nCopying animal images...")

for split in splits:

    for old_name, new_name in ANIMAL_CLASSES.items():

        source = ANIMAL_DATASET / split / old_name
        destination = OUTPUT_DATASET / split / new_name

        if source.exists():

            images = list(source.glob("*"))

            for image in images:
                if image.is_file():
                    shutil.copy2(image, destination / image.name)

            print(f"{split} | {old_name}: {len(images)} images copied")

        else:
            print(f"WARNING: Folder not found -> {source}")


# ==========================================
# COPY SNAKE DATASET
# ==========================================

print("\nCopying snake images...")

for split in splits:

    for old_name, new_name in SNAKE_CLASSES.items():

        source = SNAKE_DATASET / split / old_name
        destination = OUTPUT_DATASET / split / new_name

        if source.exists():

            images = list(source.glob("*"))

            for image in images:
                if image.is_file():
                    shutil.copy2(image, destination / image.name)

            print(f"{split} | {old_name}: {len(images)} images copied")

        else:
            print(f"WARNING: Folder not found -> {source}")


print("\n===================================")
print("DATASET PREPARATION COMPLETE!")
print("===================================")
print(f"\nCombined dataset saved at:\n{OUTPUT_DATASET}")