"""
retrain.py
Maize Leaf Disease Classifier — Andiza ML Extension

The retraining pipeline: takes newly uploaded, class-labeled images, folds
them into the dataset, retrains (warm-started from the current production
model), evaluates the candidate against the held-out test set, and only
promotes it to production if it's actually better than what's currently
deployed.

Used by:
- CLI: `python src/retrain.py --check-trigger` / `python src/retrain.py --run`
- the API's POST /retrain endpoint (Phase 6), which calls run_retraining()
  directly after a bulk upload lands in data/incoming/
"""

import argparse
import json
import pathlib
import shutil
import time
from typing import Dict

import tensorflow as tf

import model as model_lib
import prediction as pred
import preprocessing as pp

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"  # where bulk-uploaded images land before ingestion
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
STATE_PATH = MODELS_DIR / "retrain_state.json"
METRICS_PATH = MODELS_DIR / "test_eval_summary.json"

DEFAULT_TRIGGER_THRESHOLD = 100  # new images since last retrain before auto-trigger fires


# ---------------------------------------------------------------------------
# 1. Ingest newly uploaded images
# ---------------------------------------------------------------------------
def ingest_incoming_images(
    incoming_dir: pathlib.Path = INCOMING_DIR,
    raw_dir: pathlib.Path = RAW_DIR,
) -> Dict[str, int]:
    """
    Move class-labeled images from data/incoming/<class>/*.jpg into data/raw/<class>/,
    where <class> must be one of preprocessing.CLASSES (this is the same "bulk
    upload" folder structure users are asked to zip/upload in the UI).

    Returns a dict of {class_name: n_images_ingested}.
    """
    ingested = {}
    for c in pp.CLASSES:
        src = incoming_dir / c
        if not src.exists():
            ingested[c] = 0
            continue
        dest = raw_dir / c
        dest.mkdir(parents=True, exist_ok=True)
        files = list(src.glob("*"))
        for f in files:
            shutil.move(str(f), str(dest / f.name))
        ingested[c] = len(files)
    return ingested


# ---------------------------------------------------------------------------
# 2. Trigger check
# ---------------------------------------------------------------------------
def count_raw_images(raw_dir: pathlib.Path = RAW_DIR) -> int:
    return sum(len(list((raw_dir / c).glob("*"))) for c in pp.CLASSES)


def load_state() -> Dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    # First time this has ever run: persist the current count as the baseline,
    # so future calls measure "new images since THIS point" instead of
    # silently recomputing a fresh baseline (and always reporting 0 new) every
    # single time the state file happens to be missing.
    state = {"last_retrain_image_count": count_raw_images(), "last_retrain_time": None}
    save_state(state)
    return state


def save_state(state: Dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def should_trigger_retrain(threshold: int = DEFAULT_TRIGGER_THRESHOLD) -> Dict:
    """
    Retraining trigger condition: fire when enough new images have accumulated
    in data/raw since the last retrain. This is the same check the API calls
    after every bulk upload (POST /upload-retrain-data) to decide whether to
    recommend/auto-fire a retrain, and what the UI's retrain button reflects.
    """
    state = load_state()
    current_count = count_raw_images()
    new_images = current_count - state["last_retrain_image_count"]
    return {
        "should_retrain": new_images >= threshold,
        "new_images_since_last_retrain": new_images,
        "threshold": threshold,
        "current_total_images": current_count,
    }


# ---------------------------------------------------------------------------
# 3. Retrain + evaluate + promote-if-better
# ---------------------------------------------------------------------------
def run_retraining(epochs: int = 3, resplit_data: bool = True) -> Dict:
    """
    Full retraining cycle:
      1. Re-split data/raw (now including any newly ingested images) into
         train/val/test.
      2. Warm-start from the current models/best_model.keras (if present)
         rather than training from scratch — faster convergence, and the
         model doesn't forget what it already learned.
      3. Fine-tune for a few epochs on the updated training set.
      4. Evaluate the candidate on the (freshly re-split) test set.
      5. Promote the candidate to models/best_model.keras ONLY if its test
         accuracy beats the currently recorded production accuracy. Otherwise
         the candidate is discarded and production is left untouched — a
         retrain is never allowed to silently make things worse.

    Returns a summary dict (also written to models/retrain_log.json) with the
    before/after metrics and the promotion decision, so the API can return it
    directly and the UI can render "retrain succeeded, accuracy improved from
    X to Y" or "retrain did not improve on production; kept existing model".
    """
    t0 = time.time()
    log = {"started_at": time.time(), "epochs": epochs}

    if resplit_data:
        for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
            if d.exists():
                shutil.rmtree(d)
        split_summary = pp.split_raw_dataset(RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR)
        log["data_split"] = split_summary

    train_ds = pp.make_dataset(TRAIN_DIR, shuffle=True, batch_size=32)
    val_ds = pp.make_dataset(VAL_DIR, shuffle=False, batch_size=32)
    class_weights = pp.compute_class_weights(RAW_DIR, pp.CLASSES)

    prod_model_path = MODELS_DIR / "best_model.keras"
    if prod_model_path.exists():
        candidate = tf.keras.models.load_model(prod_model_path)
        log["warm_started_from"] = "best_model.keras"
    else:
        candidate = model_lib.build_baseline_cnn()
        log["warm_started_from"] = None

    history = candidate.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        verbose=2,
    )
    log["final_val_accuracy"] = float(history.history["val_accuracy"][-1])
    log["best_val_accuracy_this_run"] = float(max(history.history["val_accuracy"]))

    candidate_path = MODELS_DIR / "candidate_model.keras"
    candidate.save(candidate_path)

    y_true, y_pred, y_pred_probs = pred.evaluate_on_test_set(TEST_DIR, model=candidate)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    import numpy as np

    candidate_test_acc = float(accuracy_score(y_true, y_pred))
    log["candidate_test_accuracy"] = candidate_test_acc

    y_true_onehot = np.eye(len(pp.CLASSES))[y_true]
    per_class_auc = {
        c: float(roc_auc_score(y_true_onehot[:, i], y_pred_probs[:, i]))
        for i, c in enumerate(pp.CLASSES)
    }
    macro_auc = float(roc_auc_score(y_true_onehot, y_pred_probs, average="macro", multi_class="ovr"))
    full_metrics = {
        "classification_report_text": classification_report(y_true, y_pred, target_names=pp.CLASSES, digits=3),
        "classification_report_dict": classification_report(y_true, y_pred, target_names=pp.CLASSES, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_auc": per_class_auc,
        "macro_auc": macro_auc,
        "test_accuracy": candidate_test_acc,
        "classes": pp.CLASSES,
    }

    previous_test_acc = None
    if METRICS_PATH.exists():
        previous_test_acc = json.loads(METRICS_PATH.read_text()).get("test_accuracy")
    log["previous_test_accuracy"] = previous_test_acc

    promoted = previous_test_acc is None or candidate_test_acc > previous_test_acc
    log["promoted"] = promoted

    if promoted:
        shutil.copy(candidate_path, prod_model_path)
        full_metrics["retrained_at"] = time.time()
        METRICS_PATH.write_text(json.dumps(full_metrics, indent=2))
        pred.clear_model_cache()  # so the API's next /predict call reloads the new weights

    state = load_state()
    state["last_retrain_image_count"] = count_raw_images()
    state["last_retrain_time"] = time.time()
    save_state(state)

    log["elapsed_seconds"] = time.time() - t0
    (MODELS_DIR / "retrain_log.json").write_text(json.dumps(log, indent=2))
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining pipeline for the maize disease classifier")
    parser.add_argument("--check-trigger", action="store_true", help="Check if enough new data has accumulated to warrant a retrain")
    parser.add_argument("--ingest", action="store_true", help="Move data/incoming/<class>/* into data/raw/<class>/")
    parser.add_argument("--run", action="store_true", help="Run the full retraining cycle")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=DEFAULT_TRIGGER_THRESHOLD)
    args = parser.parse_args()

    if args.ingest:
        print(json.dumps(ingest_incoming_images(), indent=2))
    if args.check_trigger:
        print(json.dumps(should_trigger_retrain(args.threshold), indent=2))
    if args.run:
        print(json.dumps(run_retraining(epochs=args.epochs), indent=2))
