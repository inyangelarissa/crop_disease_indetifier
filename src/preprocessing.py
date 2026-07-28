import shutil
import pathlib
from typing import List, Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE: Tuple[int, int] = (224, 224)
CLASSES: List[str] = [
    "Cercospora_Gray_Leaf_Spot",
    "Common_Rust",
    "Healthy",
    "Northern_Leaf_Blight",
]

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.15),
    ],
    name="data_augmentation",
)


def split_raw_dataset(
    raw_dir: pathlib.Path,
    train_dir: pathlib.Path,
    val_dir: pathlib.Path,
    test_dir: pathlib.Path,
    classes: List[str] = CLASSES,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Stratified split of raw class-labeled images into train/val/test directory
    trees (image_dataset_from_directory-compatible layout).

    Used both for the initial dataset build and for incorporating newly
    uploaded bulk images during a retraining trigger.

    Returns a per-class summary dict of how many files landed in each split.
    """
    for d in (train_dir, val_dir, test_dir):
        for c in classes:
            (d / c).mkdir(parents=True, exist_ok=True)

    summary = {}
    holdout = val_size + test_size
    for c in classes:
        files = sorted((raw_dir / c).glob("*"))
        if not files:
            summary[c] = {"train": 0, "val": 0, "test": 0}
            continue

        train_files, temp_files = train_test_split(
            files, test_size=holdout, random_state=seed
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=test_size / holdout, random_state=seed
        )

        for f in train_files:
            shutil.copy(f, train_dir / c / f.name)
        for f in val_files:
            shutil.copy(f, val_dir / c / f.name)
        for f in test_files:
            shutil.copy(f, test_dir / c / f.name)

        summary[c] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    return summary


def make_dataset(
    directory: pathlib.Path,
    shuffle: bool,
    batch_size: int = 32,
    image_size: Tuple[int, int] = IMG_SIZE,
) -> tf.data.Dataset:
    """
    Build a normalized ([0,1]) tf.data.Dataset from a directory of class
    subfolders. Used for train/val/test datasets alike; augmentation is
    applied separately (see `augment_dataset`) only to the training split.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=42,
    )
    return ds.map(lambda x, y: (x / 255.0, y))


def augment_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Apply the training-time augmentation pipeline to a normalized dataset."""
    return ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def preprocess_single_image(image_path: str, image_size: Tuple[int, int] = IMG_SIZE):
    """
    Load + resize + normalize a single image for inference.
    Used by src/prediction.py and the API's /predict endpoint.
    Returns a (1, H, W, 3) float32 array in [0,1], ready for model.predict().
    """
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return tf.expand_dims(arr, axis=0)


def compute_class_weights(raw_dir: pathlib.Path, classes: List[str] = CLASSES) -> dict:
    """
    Inverse-frequency class weights, to counter the imbalance identified in
    EDA (Cercospora_Gray_Leaf_Spot has roughly half the samples of the other
    classes). Pass directly to model.fit(..., class_weight=...).
    """
    counts = {c: len(list((raw_dir / c).glob("*"))) for c in classes}
    total = sum(counts.values())
    n_classes = len(classes)
    return {
        i: total / (n_classes * counts[c]) if counts[c] > 0 else 1.0
        for i, c in enumerate(classes)
    }
