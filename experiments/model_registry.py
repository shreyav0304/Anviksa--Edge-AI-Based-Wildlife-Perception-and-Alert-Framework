"""Model registry and architecture-specific preprocessing policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from .config import ExperimentConfig
from .data_pipeline import build_shared_augmentation


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    algorithm: str
    techniques: tuple[str, ...]
    training_method: str
    purpose: str
    transfer_learning: bool
    preprocessing: str
    fine_tune_layers: int


@dataclass
class ModelBundle:
    spec: ModelSpec
    model: tf.keras.Model
    backbone: tf.keras.Model | None


CUSTOM_CNN_SPEC = ModelSpec(
    key="custom_cnn",
    display_name="Custom CNN",
    algorithm="Convolutional Neural Network",
    techniques=(
        "Standard convolution",
        "Batch normalization",
        "Pooling",
        "Dropout",
    ),
    training_method="Trained from scratch",
    purpose="Baseline",
    transfer_learning=False,
    preprocessing="Rescaling from [0, 255] to [0, 1]",
    fine_tune_layers=0,
)

MOBILENET_V2_SPEC = ModelSpec(
    key="mobilenet_v2",
    display_name="MobileNetV2",
    algorithm="Lightweight Convolutional Neural Network",
    techniques=(
        "Depthwise separable convolution",
        "Inverted residuals",
        "Linear bottlenecks",
    ),
    training_method="ImageNet transfer learning + fine-tuning",
    purpose="Lightweight Edge AI baseline",
    transfer_learning=True,
    preprocessing="Rescaling from [0, 255] to [-1, 1]",
    fine_tune_layers=40,
)

MOBILENET_V3_SPEC = ModelSpec(
    key="mobilenet_v3_large",
    display_name="MobileNetV3 Large",
    algorithm="Optimized Lightweight Convolutional Neural Network",
    techniques=(
        "Depthwise convolution",
        "Inverted residuals",
        "Squeeze-and-Excitation",
        "Hard-swish",
    ),
    training_method="ImageNet transfer learning + fine-tuning",
    purpose="Modern mobile/Edge AI comparison",
    transfer_learning=True,
    preprocessing="Keras MobileNetV3 internal Rescaling to [-1, 1]",
    fine_tune_layers=45,
)

EFFICIENTNET_B0_SPEC = ModelSpec(
    key="efficientnet_b0",
    display_name="EfficientNetB0",
    algorithm="Efficient Convolutional Neural Network",
    techniques=("MBConv blocks", "Compound scaling"),
    training_method="ImageNet transfer learning + fine-tuning",
    purpose="Accuracy-efficiency comparison",
    transfer_learning=True,
    preprocessing="Keras EfficientNet internal Rescaling/Normalization",
    fine_tune_layers=50,
)


def _classification_head(
    features: tf.Tensor, num_classes: int, dropout_rate: float = 0.4
) -> tf.Tensor:
    features = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)
    features = tf.keras.layers.Dense(256, activation="relu", name="head_dense")(
        features
    )
    features = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(
        features
    )
    return tf.keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(features)


def _build_custom_cnn(
    config: ExperimentConfig, _: str | None
) -> ModelBundle:
    inputs = tf.keras.Input((*config.image_size, 3), name="rgb_image")
    x = build_shared_augmentation(config)(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="custom_rescaling")(x)

    for index, (filters, dropout) in enumerate(
        ((32, 0.10), (64, 0.15), (128, 0.20), (192, 0.25)), start=1
    ):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            name=f"block_{index}_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"block_{index}_bn")(x)
        x = tf.keras.layers.ReLU(name=f"block_{index}_relu")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"block_{index}_pool")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"block_{index}_dropout")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        len(config.class_names),
        activation="softmax",
        name="class_probabilities",
    )(x)
    model = tf.keras.Model(inputs, outputs, name="anviksa_custom_cnn")
    return ModelBundle(CUSTOM_CNN_SPEC, model, None)


def _build_mobilenet_v2(
    config: ExperimentConfig, weights: str | None
) -> ModelBundle:
    inputs = tf.keras.Input((*config.image_size, 3), name="rgb_image")
    x = build_shared_augmentation(config)(inputs)
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5, offset=-1.0, name="mobilenet_v2_preprocessing"
    )(x)
    backbone = tf.keras.applications.MobileNetV2(
        weights=weights,
        include_top=False,
        input_shape=(*config.image_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    outputs = _classification_head(x, len(config.class_names), 0.4)
    model = tf.keras.Model(inputs, outputs, name="anviksa_mobilenet_v2")
    return ModelBundle(MOBILENET_V2_SPEC, model, backbone)


def _build_mobilenet_v3_large(
    config: ExperimentConfig, weights: str | None
) -> ModelBundle:
    inputs = tf.keras.Input((*config.image_size, 3), name="rgb_image")
    x = build_shared_augmentation(config)(inputs)
    backbone = tf.keras.applications.MobileNetV3Large(
        weights=weights,
        include_top=False,
        include_preprocessing=True,
        input_shape=(*config.image_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    outputs = _classification_head(x, len(config.class_names), 0.35)
    model = tf.keras.Model(inputs, outputs, name="anviksa_mobilenet_v3_large")
    return ModelBundle(MOBILENET_V3_SPEC, model, backbone)


def _build_efficientnet_b0(
    config: ExperimentConfig, weights: str | None
) -> ModelBundle:
    inputs = tf.keras.Input((*config.image_size, 3), name="rgb_image")
    x = build_shared_augmentation(config)(inputs)
    backbone = tf.keras.applications.EfficientNetB0(
        weights=weights,
        include_top=False,
        input_shape=(*config.image_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    outputs = _classification_head(x, len(config.class_names), 0.35)
    model = tf.keras.Model(inputs, outputs, name="anviksa_efficientnet_b0")
    return ModelBundle(EFFICIENTNET_B0_SPEC, model, backbone)


Builder = Callable[[ExperimentConfig, str | None], ModelBundle]
MODEL_REGISTRY: dict[str, tuple[ModelSpec, Builder]] = {
    "custom_cnn": (CUSTOM_CNN_SPEC, _build_custom_cnn),
    "mobilenet_v2": (MOBILENET_V2_SPEC, _build_mobilenet_v2),
    "mobilenet_v3_large": (MOBILENET_V3_SPEC, _build_mobilenet_v3_large),
    "efficientnet_b0": (EFFICIENTNET_B0_SPEC, _build_efficientnet_b0),
}


def build_model(
    model_key: str,
    config: ExperimentConfig,
    weights: str | None = "imagenet",
) -> ModelBundle:
    """Build a registered architecture; production defaults to ImageNet."""
    try:
        _, builder = MODEL_REGISTRY[model_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {model_key!r}. Available: {sorted(MODEL_REGISTRY)}"
        ) from exc
    effective_weights = None if model_key == "custom_cnn" else weights
    return builder(config, effective_weights)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def configure_fine_tuning(bundle: ModelBundle) -> dict[str, int]:
    """Apply architecture-specific upper-layer fine-tuning safely."""
    if bundle.backbone is None:
        return {"fine_tune_layers": 0, "trainable_backbone_layers": 0}

    backbone = bundle.backbone
    backbone.trainable = True
    fine_tune_layers = min(bundle.spec.fine_tune_layers, len(backbone.layers))
    boundary = len(backbone.layers) - fine_tune_layers

    for layer in backbone.layers[:boundary]:
        layer.trainable = False
    for layer in backbone.layers[boundary:]:
        layer.trainable = not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )

    return {
        "fine_tune_layers": fine_tune_layers,
        "trainable_backbone_layers": sum(
            layer.trainable for layer in backbone.layers
        ),
    }
