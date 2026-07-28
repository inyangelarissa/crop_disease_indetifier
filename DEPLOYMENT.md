# Deploying to Render

This covers Task 3 of the assignment: deploying the Phase 1-6 pipeline to a cloud
platform and evaluating the model in production.

## Prerequisites

- This repo pushed to GitHub (with `Dockerfile`, `render.yaml`, `requirements.txt`,
  `src/`, `api/`, `models/best_model.keras`, `models/test_eval_summary.json`,
  `models/retrain_state.json`, and `data/raw/` all present — see the "what gets
  deployed" note below).
- A free Render account: https://render.com

## Option A — Blueprint deploy (one click, uses render.yaml)

1. Push this repo to GitHub.
2. In the Render dashboard: **New → Blueprint**.
3. Select this repo. Render reads `render.yaml` automatically and shows the
   `maize-leaf-disease-api` service it's about to create.
4. Click **Apply**. Render builds the Docker image and deploys it.
5. First build takes several minutes (TensorFlow is a large dependency). Watch
   the build logs in the Render dashboard.
6. Once live, Render gives you a URL like `https://maize-leaf-disease-api.onrender.com`.

## Option B — Manual web service (no Blueprint)

1. Push this repo to GitHub.
2. In the Render dashboard: **New → Web Service**.
3. Connect the repo. Set **Environment: Docker** (Render auto-detects the
   `Dockerfile` at the repo root).
4. Plan: **Free** (or a paid plan — see the storage caveat below).
5. Health check path: `/health`.
6. Click **Create Web Service**.

## Verifying the deployment

Once live, confirm it's actually working (don't just trust a green checkmark):

```bash
curl https://<your-app>.onrender.com/health
curl https://<your-app>.onrender.com/model-info
```

`/health` should return `{"status":"ok","model_file_present":true,...}`. If
`model_file_present` is `false`, the model file didn't make it into the image —
double check `models/best_model.keras` is committed to git (it's a binary file;
some `.gitignore` templates exclude `*.keras` by accident) and not excluded by
your own `.gitignore`.

Then point `ui/index.html`'s API URL field at the Render URL instead of
`http://localhost:8000` — the dashboard should populate live.

## Evaluating the model in production

`GET /model-info` on the deployed URL returns the same evaluation package Phase 4
produced locally (test accuracy, macro ROC-AUC, full per-class classification
report, confusion matrix) — this is "the evaluation process of the model in
production" the assignment asks for: it's not a separate step, it's the same
metrics pipeline, just reachable over the deployed URL instead of a notebook cell.
To re-run evaluation against fresh data in production, `POST /retrain` re-splits,
retrains, and re-evaluates, then reports the new numbers in its response.

## Production storage caveat (read before relying on retraining in prod)

**Render's free tier has an ephemeral filesystem.** Anything written to disk at
runtime — images uploaded via `/upload-retrain-data`, a model newly promoted by
`/retrain` — is lost the next time the service restarts, redeploys, or spins down
from inactivity. The model bundled into the Docker image at build time (Phase 3's
`best_model.keras`) always survives; anything produced by a live retrain does not,
on the free tier.

This is a real, worth-documenting limitation of the free-tier deployment, not a bug
in the pipeline. Two ways to fix it for a genuine production deployment:

1. **Attach a Render persistent disk** (paid plans) and mount it at `/app/data`
   and `/app/models`, so retrain output survives restarts.
2. **Push promoted models to external storage** (e.g. S3, or even committing back
   to a `models/` branch via the GitHub API) at the end of `run_retraining()` in
   `src/retrain.py`, and load from there at startup instead of relying on local disk.

For this assignment, the free tier is sufficient to demonstrate the full pipeline
end-to-end (predict, upload, trigger, retrain, evaluate) — just don't expect a
retrain to "stick" across a Render restart without one of the above.

## Cold starts (relevant to Phase 9's Locust results)

Render's free tier spins the service down after ~15 minutes of inactivity. The
first request after that takes 30-60+ seconds while it spins back up, and
loading TensorFlow + the model adds a few more seconds on top. When reading the
Locust load-test results in Phase 9, treat the first request(s) after idle time
as cold-start outliers, distinct from steady-state latency under load.
