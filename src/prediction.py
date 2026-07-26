"""
prediction.py
Maize Leaf Disease Classifier — Andiza ML Extension

Inference utilities. Used by:
- notebook/maize_leaf_disease.ipynb (Phase 4 evaluation, single-prediction demo)
- the FastAPI /predict endpoint (Phase 6)
"""

import pathlib
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from preprocessing import CLASSES, IMG_SIZE, preprocess_single_image

DEFAULT_MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "best_model.keras"

_model_cache: Dict[str, tf.keras.Model] = {}


def load_model(model_path: Union[str, pathlib.Path] = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Load (and cache) the trained model. Cached by path string so the API doesn't
    reload from disk on every request, but a retrain that writes a new file at
    the same path will need `clear_model_cache()` called first.
    """
    key = str(model_path)
    if key not in _model_cache:
        _model_cache[key] = tf.keras.models.load_model(model_path)
    return _model_cache[key]


def clear_model_cache() -> None:
    """Call after retraining swaps in a new model file at the same path."""
    _model_cache.clear()


def predict_single(
    image_path: Union[str, pathlib.Path],
    model: tf.keras.Model = None,
) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Predict the disease class for a single leaf image.

    Returns:
        {
            "predicted_class": "Healthy",
            "confidence": 0.94,
            "class_probabilities": {"Healthy": 0.94, "Common_Rust": 0.03, ...}
        }
    """
    if model is None:
        model = load_model()

    x = preprocess_single_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "class_probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
    }


def predict_batch(
    image_paths: List[Union[str, pathlib.Path]],
    model: tf.keras.Model = None,
) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
    """Predict for a list of image paths. Used by bulk-upload flows in the UI/API."""
    if model is None:
        model = load_model()
    return [predict_single(p, model=model) for p in image_paths]


def evaluate_on_test_set(
    test_dir: Union[str, pathlib.Path],
    model: tf.keras.Model = None,
    batch_size: int = 32,
):
    """
    Run the model over every image in a class-labeled test directory and
    return (y_true, y_pred, y_pred_probs) as numpy arrays.

    Used by the evaluation notebook (confusion matrix, classification_report,
    ROC-AUC) and by the API's production-evaluation endpoint (Phase 6), so
    "how good is the model right now" can be recomputed on demand against
    whatever test set is currently on disk — including after a retrain.
    """
    import preprocessing as pp

    if model is None:
        model = load_model()

    test_ds = pp.make_dataset(pathlib.Path(test_dir), shuffle=False, batch_size=batch_size)

    y_true, y_pred_probs = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_pred_probs.append(probs)
        y_true.append(y_batch.numpy())

    y_true = np.concatenate(y_true).argmax(axis=1)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = y_pred_probs.argmax(axis=1)

    return y_true, y_pred, y_pred_probs
