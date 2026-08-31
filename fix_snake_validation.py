import shutil
import random
from pathlib import Path

# Path to the combined dataset
BASE_DIR = Path(r"C:\Users\S Yashveer\Downloads\animal datasets")
DATASET_DIR = BASE_DIR / "combined_dataset"

# Make results reproducible
random.seed(42)

# Snake classes that need validation images
SNAKE_CLASSES = [
    "Venomous_Snake",
    "Non_Venomous_Snake"
]

VALIDATION_PERCENT = 0.15  # 15%


for class_name in SNAKE_CLASSES:

    train_folder = DATASET_DIR / "train" / class_name
    validation_folder = DATASET_DIR / "validation" / class_name

    validation_folder.mkdir(parents=True, exist_ok=True)

    # Get all image files
    images = [
        f for f in train_folder.iterdir()
        if f.is_file() and f.suffix.lower() in
        [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    # Shuffle images
    random.shuffle(images)

    # Select 15% for validation
    num_validation = int(len(images) * VALIDATION_PERCENT)
    selected_images = images[:num_validation]

    print(f"\n{class_name}")
    print(f"Total training images before split: {len(images)}")
    print(f"Copying {num_validation} images to validation...")

    # COPY images, don't move them
    for image in selected_images:
        destination = validation_folder / image.name
        shutil.copy2(image, destination)

    print("Done!")


print("\n===================================")
print("SNAKE VALIDATION SET CREATED!")
print("===================================")