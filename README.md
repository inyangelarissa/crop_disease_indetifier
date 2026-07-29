# Maize Leaf Disease Classifier

**Machine Learning Pipeline Summative** 

**Name** INYANGE Larissa 

An end-to-end machine learning pipeline that classifies maize (corn) leaf photos into
one of four classes — `Healthy`, `Common_Rust`, `Northern_Leaf_Blight`,
`Cercospora_Gray_Leaf_Spot`. This extends my earlier tabular crop-yield-prediction project into image data, covering the full cycle: data acquisition, preprocessing, model training, evaluation, a retraining pipeline with an automatic trigger, a FastAPI service, a monitoring/diagnosis dashboard, cloud deployment, and load testing.

- **Video demo:** _add your YouTube link here_
- **Live app:** https://maize-disease-indetifier.streamlit.app/
- **Flood/load test results:** see `LOCUST_RESULTS.md` The summary is below

## Project structure

crop_disease_indetifier/
├── README.md                  
├── STREAMLIT_DEPLOYMENT.md     
├── DEPLOYMENT.md              
├── LOCUST_RESULTS.md            
├── locustfile.py               
├── streamlit_app.py              
├── .streamlit/
│   └── config.toml               
├── Dockerfile                    
├── .dockerignore
├── render.yaml
├── requirements.txt
│
├── notebook/
│   └── maize_leaf_disease.ipynb   
│
├── src/
│   ├── preprocessing.py       
│   ├── model.py               
│   ├── prediction.py           
│   └── retrain.py              
│
├── api/
│   └── main.py                 
│
├── ui/
│   └── index.html               
│
├── data/
│   ├── raw/                    
│   ├── train/ val/ test/       
│   ├── incoming/                
│   └── *.png                    
│
└── models/
    ├── best_model.keras         
    ├── test_eval_summary.json   
    └── retrain_state.json      

## Setup

### 1. Clone and install

```bash
git clone https://github.com/inyangelarissa/crop_disease_indetifier.git
cd crop_disease_indetifier
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
```

### 2. Get the raw data into `data/raw/`

You need `data/raw/Healthy/`, `data/raw/Common_Rust/`, `data/raw/Northern_Leaf_Blight/`,
`data/raw/Cercospora_Gray_Leaf_Spot/`, each full of leaf images (sourced from
PlantVillage). Either extract the provided raw-data zip into `data/raw/`, or
re-download via the sparse git checkout documented in the notebook's Section 1.

### 3. Run the notebook once (optional — a trained model is already included)

```bash
jupyter notebook notebook/maize_leaf_disease.ipynb
```
Run all cells top to bottom to reproduce data acquisition, preprocessing, training,
and evaluation from scratch, and regenerate `data/train/`, `data/val/`, `data/test/`.
`models/best_model.keras` is already trained and committed, so this step is for
reproducibility/grading, not a prerequisite for running the API.

### 4. Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
```
Check it's alive: `curl http://localhost:8000/health` → `{"status":"ok",...}`
Interactive API docs: http://localhost:8000/docs
The dashboard is served at the same address: http://localhost:8000/

### 5. Or just use the live deployment

No local setup needed — the dashboard and API are both live at
**https://maize-leaf-disease-api-5aqm.onrender.com/**

## How the pieces run together

```
 notebook/*.ipynb  ──trains──▶  models/best_model.keras
                                        │
 src/preprocessing.py                  │  loaded by
 src/model.py         ──imported by──▶ │
 src/prediction.py                     ▼
 src/retrain.py       ──imported by──▶ api/main.py  ──serves──▶  ui/index.html at "/"
                                        │
                                        ▼
                              Dockerfile / render.yaml
                                        │
                                        ▼
                    https://maize-leaf-disease-api-5aqm.onrender.com/
```

`src/*.py` are plain importable modules with no dependency on the API or notebook —
both import from them instead of duplicating logic. `api/main.py` is the only thing
that talks HTTP: it wraps `src/prediction.py` (predict, evaluate) and `src/retrain.py`
(ingest, trigger, retrain), and also serves `ui/index.html` directly at `/`, so the
dashboard and the API live at one single URL with no separate hosting or manual
API-address configuration needed.

## Model evaluation

Full detail (confusion matrix, per-class precision/recall/F1, ROC-AUC, misclassified
sample review) is in the notebook. Summary on the held-out test set:

| Metric | Value |
|---|---|
| Test accuracy | 92.9% (baseline), improved to 94-95%+ after retraining on additional data |
| Macro precision / recall / F1 | 0.901 / 0.900 / 0.900 |
| Macro ROC-AUC | 0.986 |
| Weakest class | `Cercospora_Gray_Leaf_Spot` — visually confusable with `Northern_Leaf_Blight` |
| Strongest classes | `Healthy`, `Common_Rust` |

Two models were compared: a baseline CNN trained from scratch, and a MobileNetV2
transfer-learning model. The baseline CNN was selected for production after the
MobileNetV2 run showed a BatchNorm cold-start instability when trained without
ImageNet pretrained weights — documented and diagnosed in the notebook rather than
hidden, since it's a genuine, explainable finding about transfer learning without
pretrained weights.

## Load testing (flood simulation)

`locustfile.py` simulates concurrent users hitting `/predict`, `/health`, and
`/model-info`. Full methodology and raw CSVs are in `LOCUST_RESULTS.md`; summary:

| Scenario | Concurrent users | Median | p95 | Throughput | Failures |
|---|---|---|---|---|---|
| 1 worker, 10 users | 10 | 150ms | 380ms | 4.93 req/s | 0% |
| 2 workers, 10 users | 10 | 150ms | 340ms | 5.02 req/s | 0% |
| 1 worker, 20 users | 20 | 400ms | 820ms | 8.12 req/s | 0% |

Zero failures across all scenarios — the API queues gracefully under load rather than
erroring out.

To re-run against the live deployment:
```bash
locust -f locustfile.py --host https://maize-leaf-disease-api-5aqm.onrender.com \
    --users 10 --spawn-rate 2 --run-time 60s --headless --csv locust_results/render_run1
```

## Retraining

The retraining pipeline (`src/retrain.py`) ingests newly uploaded, class-labeled
images, re-splits the dataset, warm-starts from the current production model, trains
for a few epochs, evaluates the candidate against the held-out test set, and only
promotes it to production if it's actually better — a retrain can never silently make
production worse.

```bash
# Check whether enough new data has accumulated to warrant a retrain
python src/retrain.py --check-trigger

# Move newly uploaded images from data/incoming/<class>/ into data/raw/<class>/
python src/retrain.py --ingest

# Run a full retrain cycle
python src/retrain.py --run --epochs 3
```

Or via the API/dashboard: bulk-upload through `POST /upload-retrain-data` (or the
dashboard's "Upload & Retrain" panel), then trigger `POST /retrain`.

### A real constraint I ran into: Render free tier and memory

Render's free tier gives 512MB of RAM total, shared between the already-running API
(which keeps a model loaded to serve predictions) and a new training job. Training at
full batch size (32) reliably crashed the service with an out-of-memory kill. I fixed
this by:
- Reducing the training batch size (32 → 8)
- Explicitly releasing the cached prediction model's memory before training starts
  (`tf.keras.backend.clear_session()` — dropping a Python reference to a Keras model
  alone doesn't release TensorFlow's internal graph memory, which was the real gap)

This is documented here rather than smoothed over because it's a genuine production
constraint worth understanding: a model that trains fine on a full-sized machine can
still need real tuning to fit a constrained hosting tier, and "it works on my machine"
isn't the same as "it works within the resource limits of where it's deployed."

## Deployment

Live at **https://maize-leaf-disease-api-5aqm.onrender.com/**, deployed via Docker on
Render 
(see `DEPLOYMENT.md` for the full steps, plus two limitations worth knowing:
free-tier storage which makes a live retrain's output doesn't survive a restart
without a persistent disk  and free-tier cold starts add 30-60s of latency after
idle periods).

