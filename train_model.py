import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error
)

# ============================================================
# ANVIKSA AI
# 7-CLASS WILDLIFE CLASSIFICATION
# MobileNetV2 Transfer Learning + Fine-Tuning
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PARENT_DIR, "combined_dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Maximum epochs.
# EarlyStopping can finish training before these values.
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 25

NUM_CLASSES = 7

CLASS_NAMES = [
    "Cow",
    "Deer",
    "Elephant",
    "Monkey",
    "Non_Venomous_Snake",
    "Venomous_Snake",
    "Wild_Boar"
]

# ============================================================
# GPU CHECK
# ============================================================

print("\n" + "=" * 75)
print("ANVIKSA AI - MOBILENETV2 TRAINING")
print("=" * 75)

print("\nTensorFlow Version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPU Devices:", gpus)

if gpus:
    print("\nGPU DETECTED - TRAINING WILL USE GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

else:
    print("\nWARNING: GPU NOT DETECTED")

# ============================================================
# VERIFY DATASET
# ============================================================

for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"Dataset folder not found:\n{directory}"
        )

# ============================================================
# IMAGE GENERATORS
# ============================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.15,

    shear_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    classes=CLASS_NAMES,

    class_mode="categorical",

    shuffle=True,

    seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(

    VAL_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    classes=CLASS_NAMES,

    class_mode="categorical",

    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    classes=CLASS_NAMES,

    class_mode="categorical",

    shuffle=False
)

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print("\nClass mapping:")

for name, index in train_generator.class_indices.items():

    print(index, "->", name)

print("\nTraining Images   :", train_generator.samples)
print("Validation Images :", val_generator.samples)
print("Testing Images    :", test_generator.samples)

# ============================================================
# CLASS WEIGHTS
# ============================================================

labels = train_generator.classes

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(labels),

    y=labels
)

class_weights = {

    index: float(weight)

    for index, weight in enumerate(weights)
}

print("\nClass Weights:")

for index, weight in class_weights.items():

    print(
        f"{CLASS_NAMES[index]:25s} : {weight:.4f}"
    )

# ============================================================
# BUILD MOBILENETV2
# ============================================================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3)
)

# Freeze MobileNetV2 first
base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(
    0.4
)(x)

output = Dense(

    NUM_CLASSES,

    activation="softmax"
)(x)

model = Model(

    inputs=base_model.input,

    outputs=output
)

# ============================================================
# STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

# ============================================================
# MODEL FILE PATHS
# ============================================================

STAGE1_PATH = os.path.join(
    MODELS_DIR,
    "anviksa_stage1_best.keras"
)

FINETUNE_PATH = os.path.join(
    MODELS_DIR,
    "anviksa_finetuned_best.keras"
)

FINAL_PATH = os.path.join(
    MODELS_DIR,
    "anviksa_mobilenetv2_best.keras"
)

# ============================================================
# STAGE 1 CALLBACKS
# ============================================================

stage1_callbacks = [

    ModelCheckpoint(

        STAGE1_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    EarlyStopping(

        monitor="val_loss",

        patience=6,

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1
    )
]

# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n" + "=" * 75)
print("STAGE 1 - TRANSFER LEARNING")
print("=" * 75)

history1 = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=INITIAL_EPOCHS,

    callbacks=stage1_callbacks,

    class_weight=class_weights
)

# ============================================================
# STAGE 2 FINE-TUNING
# ============================================================

print("\n" + "=" * 75)
print("STAGE 2 - FINE-TUNING")
print("=" * 75)

# Unfreeze MobileNetV2
base_model.trainable = True

# Keep most pretrained layers frozen.
# Fine-tune final 40 layers.
fine_tune_at = len(base_model.layers) - 40

for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False

# BatchNormalization layers remain frozen
for layer in base_model.layers[fine_tune_at:]:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False

print(
    "\nFine-tuning final MobileNetV2 layers..."
)

# Very low LR for fine-tuning
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

fine_callbacks = [

    ModelCheckpoint(

        FINETUNE_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    EarlyStopping(

        monitor="val_loss",

        patience=6,

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-7,

        verbose=1
    )
]

history2 = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=fine_callbacks,

    class_weight=class_weights
)

# ============================================================
# COMPARE STAGE 1 VS FINE-TUNED MODEL
# ============================================================

print("\n" + "=" * 75)
print("SELECTING BEST VALIDATION MODEL")
print("=" * 75)

stage1_model = load_model(
    STAGE1_PATH
)

fine_model = load_model(
    FINETUNE_PATH
)

val_generator.reset()

stage1_loss, stage1_accuracy = stage1_model.evaluate(

    val_generator,

    verbose=0
)

val_generator.reset()

fine_loss, fine_accuracy = fine_model.evaluate(

    val_generator,

    verbose=0
)

print(
    f"\nStage 1 Validation Accuracy: "
    f"{stage1_accuracy * 100:.2f}%"
)

print(
    f"Fine-Tuned Validation Accuracy: "
    f"{fine_accuracy * 100:.2f}%"
)

if fine_accuracy >= stage1_accuracy:

    final_model = fine_model

    selected_model = "Fine-Tuned MobileNetV2"

else:

    final_model = stage1_model

    selected_model = "Stage-1 MobileNetV2"

print("\nSelected Model:", selected_model)

final_model.save(
    FINAL_PATH
)

# ============================================================
# TEST DATASET EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TESTING")
print("=" * 75)

test_generator.reset()

test_loss, test_accuracy = final_model.evaluate(

    test_generator,

    verbose=1
)

# ============================================================
# PREDICTIONS
# ============================================================

test_generator.reset()

predictions = final_model.predict(

    test_generator,

    verbose=1
)

predicted_classes = np.argmax(

    predictions,

    axis=1
)

true_classes = test_generator.classes

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(

    true_classes,

    predicted_classes
)

precision = precision_score(

    true_classes,

    predicted_classes,

    average="weighted",

    zero_division=0
)

recall = recall_score(

    true_classes,

    predicted_classes,

    average="weighted",

    zero_division=0
)

weighted_f1 = f1_score(

    true_classes,

    predicted_classes,

    average="weighted",

    zero_division=0
)

macro_f1 = f1_score(

    true_classes,

    predicted_classes,

    average="macro",

    zero_division=0
)

# Probability MSE
true_one_hot = tf.keras.utils.to_categorical(

    true_classes,

    NUM_CLASSES
)

mse = mean_squared_error(

    true_one_hot,

    predictions
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("ANVIKSA AI FINAL TEST RESULTS")
print("=" * 75)

print(
    f"\nTest Loss              : {test_loss:.4f}"
)

print(
    f"Test Accuracy          : {accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision     : {precision:.4f}"
)

print(
    f"Weighted Recall        : {recall:.4f}"
)

print(
    f"Weighted F1 Score      : {weighted_f1:.4f}"
)

print(
    f"Macro F1 Score         : {macro_f1:.4f}"
)

print(
    f"Mean Squared Error     : {mse:.6f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    true_classes,

    predicted_classes,

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0
)

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(report)

REPORT_PATH = os.path.join(

    MODELS_DIR,

    "classification_report.txt"
)

with open(

    REPORT_PATH,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "ANVIKSA AI CLASSIFICATION REPORT\n\n"
    )

    file.write(report)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "selected_model":
        selected_model,

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(accuracy),

    "weighted_precision":
        float(precision),

    "weighted_recall":
        float(recall),

    "weighted_f1":
        float(weighted_f1),

    "macro_f1":
        float(macro_f1),

    "probability_mse":
        float(mse)
}

METRICS_PATH = os.path.join(

    MODELS_DIR,

    "test_metrics.json"
)

with open(

    METRICS_PATH,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        metrics,

        file,

        indent=4
    )

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    true_classes,

    predicted_classes
)

CM_PATH = os.path.join(

    MODELS_DIR,

    "confusion_matrix.png"
)

fig, ax = plt.subplots(

    figsize=(11, 9)
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=CLASS_NAMES
)

display.plot(

    ax=ax,

    xticks_rotation=45,

    values_format="d"
)

plt.title(
    "Anviksa AI - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(

    CM_PATH,

    dpi=300,

    bbox_inches="tight"
)

plt.close()

# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

training_accuracy = (

    history1.history["accuracy"] +

    history2.history["accuracy"]
)

validation_accuracy = (

    history1.history["val_accuracy"] +

    history2.history["val_accuracy"]
)

training_loss = (

    history1.history["loss"] +

    history2.history["loss"]
)

validation_loss = (

    history1.history["val_loss"] +

    history2.history["val_loss"]
)

fine_tuning_start = len(
    history1.history["accuracy"]
)

# ============================================================
# ACCURACY GRAPH
# ============================================================

ACCURACY_GRAPH = os.path.join(

    MODELS_DIR,

    "training_accuracy.png"
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(

    training_accuracy,

    label="Training Accuracy"
)

plt.plot(

    validation_accuracy,

    label="Validation Accuracy"
)

plt.axvline(

    x=fine_tuning_start - 1,

    linestyle="--",

    label="Fine-Tuning Started"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Anviksa AI - Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    ACCURACY_GRAPH,

    dpi=300
)

plt.close()

# ============================================================
# LOSS GRAPH
# ============================================================

LOSS_GRAPH = os.path.join(

    MODELS_DIR,

    "training_loss.png"
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(

    training_loss,

    label="Training Loss"
)

plt.plot(

    validation_loss,

    label="Validation Loss"
)

plt.axvline(

    x=fine_tuning_start - 1,

    linestyle="--",

    label="Fine-Tuning Started"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Anviksa AI - Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    LOSS_GRAPH,

    dpi=300
)

plt.close()

# ============================================================
# SAVE CLASS NAMES
# ============================================================

CLASS_FILE = os.path.join(

    MODELS_DIR,

    "class_names.json"
)

with open(

    CLASS_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        CLASS_NAMES,

        file,

        indent=4
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("ANVIKSA AI TRAINING COMPLETED")
print("=" * 75)

print("\nBest Model:")
print(FINAL_PATH)

print("\nTest Accuracy:")
print(
    f"{accuracy * 100:.2f}%"
)

print("\nWeighted F1 Score:")
print(
    f"{weighted_f1:.4f}"
)

print("\nFiles generated inside:")
print(MODELS_DIR)

print("\n1. anviksa_mobilenetv2_best.keras")
print("2. anviksa_stage1_best.keras")
print("3. anviksa_finetuned_best.keras")
print("4. classification_report.txt")
print("5. test_metrics.json")
print("6. confusion_matrix.png")
print("7. training_accuracy.png")
print("8. training_loss.png")
print("9. class_names.json")