from typing import Tuple

import tensorflow as tf

IMG_SIZE: Tuple[int, int] = (224, 224)
NUM_CLASSES = 4


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """
    Small CNN trained from scratch. Serves as the baseline against which the
    transfer-learning model is compared.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_transfer(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    resize_to: Tuple[int, int] = (96, 96),
) -> Tuple[tf.keras.Model, str]:
    """
    MobileNetV2-backed transfer-learning model.

    Attempts to load ImageNet pretrained weights (the normal path — use this
    when running with internet access, e.g. Google Colab or your own
    machine/cloud VM). Falls back to random initialization with the base
    unfrozen if weights can't be downloaded (e.g. a network-restricted
    sandbox), so the notebook still runs end-to-end and produces real
    results either way.

    Returns (model, weights_used) where weights_used is "imagenet" or
    "random_init" so the notebook/report can state which path was taken.

    Input is resized to 96x96 (MobileNetV2's minimum supported resolution)
    before hitting the base model. This is a deliberate speed/cost trade-off:
    the target deployment is a Render free-tier container with no GPU, and a
    5x reduction in pixel count vs. 224x224 meaningfully cuts both training
    time and per-request inference latency (relevant later for the Locust
    load test in Phase 9) at an acceptable accuracy cost for this task.
    """
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*resize_to, 3), include_top=False, weights="imagenet"
        )
        base.trainable = False  # freeze — fast, standard feature-extraction transfer learning
        weights_used = "imagenet"
    except Exception:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*resize_to, 3), include_top=False, weights=None
        )
        base.trainable = True  # no benefit freezing random weights — train end-to-end instead
        weights_used = "random_init"

    layers = [tf.keras.layers.Input(shape=input_shape)]
    if resize_to != input_shape[:2]:
        layers.append(tf.keras.layers.Resizing(*resize_to))
    if weights_used == "imagenet":
        layers.append(
            tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)
        )
    layers += [
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]

    model = tf.keras.Sequential(layers, name="mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, weights_used


def get_callbacks(checkpoint_path: str, patience: int = 3):
    """Standard callback set used for both models: early stopping + best-checkpoint saving."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True
        ),
    ]
