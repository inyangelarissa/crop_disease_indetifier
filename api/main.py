import json
import pathlib
import shutil
import sys
import time
import uuid
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import prediction as pred          # noqa: E402
import preprocessing as pp         # noqa: E402
import retrain as retrain_module   # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
INCOMING_DIR = DATA_DIR / "incoming"
TMP_UPLOAD_DIR = PROJECT_ROOT / "tmp_uploads"
TMP_UPLOAD_DIR.mkdir(exist_ok=True)

APP_START_TIME = time.time()

app = FastAPI(
    title="Maize Leaf Disease Classifier API",
    description="Predict, monitor, and retrain a maize leaf disease classifier.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health / uptime / model info — powers the UI's "model up-time" panel
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    uptime_seconds = time.time() - APP_START_TIME
    model_loaded = (MODELS_DIR / "best_model.keras").exists()
    return {
        "status": "ok" if model_loaded else "degraded",
        "uptime_seconds": round(uptime_seconds, 1),
        "model_file_present": model_loaded,
        "server_time": time.time(),
    }


@app.get("/model-info")
def model_info():
    """Current production model's metrics, as last recorded by a retrain/evaluation run."""
    metrics_path = MODELS_DIR / "test_eval_summary.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation metrics recorded yet.")
    metrics = json.loads(metrics_path.read_text())
    return {
        "classes": pp.CLASSES,
        "test_accuracy": metrics.get("test_accuracy"),
        "macro_auc": metrics.get("macro_auc"),
        "classification_report": metrics.get("classification_report_dict"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "retrained_at": metrics.get("retrained_at"),
    }


# ---------------------------------------------------------------------------
# Prediction — single image
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    tmp_path = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = pred.predict_single(tmp_path)
        return JSONResponse(result)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Bulk upload for retraining
# ---------------------------------------------------------------------------
@app.post("/upload-retrain-data")
async def upload_retrain_data(class_name: str, files: List[UploadFile] = File(...)):
    """
    Bulk-upload new labeled images for retraining. `class_name` must be one of
    preprocessing.CLASSES. Files land in data/incoming/<class_name>/, ready for
    the next /retrain call to ingest.
    """
    if class_name not in pp.CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"class_name must be one of {pp.CLASSES}, got '{class_name}'",
        )

    dest_dir = INCOMING_DIR / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        dest_path = dest_dir / f"{uuid.uuid4().hex}_{file.filename}"
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(dest_path.name)

    trigger_status = retrain_module.should_trigger_retrain()

    return {
        "class_name": class_name,
        "files_saved": len(saved),
        "trigger_status": trigger_status,
    }


# ---------------------------------------------------------------------------
# Retraining trigger + execution
# ---------------------------------------------------------------------------
@app.get("/retrain/trigger-status")
def retrain_trigger_status():
    """Used by the UI to decide whether to show 'retrain recommended'."""
    return retrain_module.should_trigger_retrain()


@app.post("/retrain")
def run_retrain(epochs: int = 3):
    """
    Manually fire a retraining cycle (also what an automatic trigger would
    call). Ingests anything staged in data/incoming/, re-splits, warm-starts
    from the current production model, trains, evaluates, and promotes only
    if the candidate is actually better.
    """
    ingested = retrain_module.ingest_incoming_images()
    result = retrain_module.run_retraining(epochs=epochs)
    result["ingested"] = ingested
    return result


# ---------------------------------------------------------------------------
# Visualization data for the UI dashboard (Phase 7)
# ---------------------------------------------------------------------------
@app.get("/visualizations/class-distribution")
def class_distribution():
    counts = {c: len(list((DATA_DIR / "raw" / c).glob("*"))) for c in pp.CLASSES}
    return counts


@app.get("/visualizations/training-history")
def training_history():
    """Most recent retrain's history, for a UI chart of accuracy/loss over time."""
    log_path = MODELS_DIR / "retrain_log.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No retraining has been run yet.")
    return json.loads(log_path.read_text())


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = PROJECT_ROOT / "ui" / "index.html"

    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="UI file not found"
        )

    return html_path.read_text(encoding="utf-8")
