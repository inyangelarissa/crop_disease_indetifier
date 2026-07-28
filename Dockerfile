# Maize Leaf Disease Classifier — Andiza ML Extension
# Serves the FastAPI app (api/main.py) for deployment on Render.

FROM python:3.12-slim

WORKDIR /app

# System deps needed by Pillow/matplotlib at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/

# Trained model + current production metrics — bundled into the image so a
# fresh deploy serves real predictions immediately, without needing a
# retrain first.
COPY models/best_model.keras ./models/best_model.keras
COPY models/test_eval_summary.json ./models/test_eval_summary.json
COPY models/retrain_state.json ./models/retrain_state.json

# Raw dataset — needed so /retrain has real data to re-split and train on.
# NOTE: on Render's free tier the filesystem is ephemeral (wiped on every
# redeploy/restart), so anything written here at runtime — newly uploaded
# retrain images, a newly promoted model — does NOT persist across restarts
# unless a paid persistent disk is attached. See README.md "Production
# storage caveat" for details and mitigation options.
COPY data/raw/ ./data/raw/

RUN mkdir -p data/train data/val data/test data/incoming tmp_uploads

EXPOSE 8000

# Render injects $PORT at runtime; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
