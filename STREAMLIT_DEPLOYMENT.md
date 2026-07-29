# Deploying to Streamlit Community Cloud

This replaces the earlier Render/FastAPI deployment (see `DEPLOYMENT.md` for that
history) — the switch was made specifically to get more headroom for retraining:
Streamlit Community Cloud gives roughly 1GB RAM per app, vs. Render's free-tier
512MB, which we repeatedly hit during live retraining.

## Why this is architecturally simpler (and why it fixes the memory problem)

The old setup had **two things needing memory in the same 512MB container at
once**: the FastAPI server (with a model loaded to serve `/predict`) and a
separate training job triggered by `/retrain`. `streamlit_app.py` is a single
process — there's no separate API server keeping its own model loaded while a
retrain job runs. Combined with more available RAM, this removes the exact
failure mode we spent a long time debugging on Render.

## Prerequisites

- This repo pushed to GitHub, including:
  - `streamlit_app.py`
  - `requirements.txt` (now Streamlit + TensorFlow, no FastAPI/uvicorn)
  - `.streamlit/config.toml` (theme)
  - `src/`, `models/best_model.keras`, `models/test_eval_summary.json`,
    `models/retrain_state.json`
  - `data/raw/` — **this needs to be committed now**, unlike the Render setup.
    Streamlit Community Cloud clones your repo directly (no Docker build step
    to selectively COPY files), so anything the app reads at runtime —
    including the raw dataset used for retraining — needs to actually be in
    the repo.
- A free account at https://streamlit.io/cloud (sign in with GitHub)

## Steps

1. **Commit `data/raw/`** if it's currently gitignored:
   ```bash
   git add -f data/raw/
   git commit -m "Include raw dataset for Streamlit Cloud (no Docker build step to copy it in)"
   ```
   Check your repo size isn't unreasonable afterward (`data/raw/` is ~58MB — fine
   for GitHub, well under any size limits).

2. Push everything:
   ```bash
   git add .
   git commit -m "Switch deployment to Streamlit"
   git push origin main
   ```

3. Go to https://share.streamlit.io → **New app**.

4. Select this repo, branch `main`, and set **Main file path** to:
   ```
   streamlit_app.py
   ```

5. Click **Deploy**. First build takes a few minutes (TensorFlow is the slow
   part, same as it was on Render).

6. Once live, you'll get a URL like:
   ```
   https://<your-app-name>.streamlit.app
   ```

## Verifying the deployment

Open the URL. You should see three tabs: **Overview**, **Diagnose**, **Upload &
Retrain**. Specifically check:

- **Overview tab** — class distribution chart and per-class F1 table populate
  (confirms `data/raw/` and `models/test_eval_summary.json` made it into the
  deployment correctly)
- **Diagnose tab** — upload a known-healthy leaf photo, confirm it says
  "Healthy" (this is the exact bug we chased for many turns earlier — worth
  re-confirming here since it's a fresh deployment)
- **Upload & Retrain tab** — upload a small batch, then click "Retrain now"
  with epochs set to 1 first. Watch for a success or "not promoted" message,
  not a crash. This is the step that repeatedly failed on Render — if it
  works here, that's confirmation the platform switch solved the actual
  problem, not just the batch-size/session-clearing code changes on their own.

## Important limitation, same spirit as Render's — Streamlit Cloud storage is also not permanent

Streamlit Community Cloud apps can restart (redeploys, inactivity, resource
limits), and the underlying filesystem is not guaranteed to persist writes
across restarts either. The same caveat from `DEPLOYMENT.md` applies here:
treat live retraining as something to demonstrate in a single session (upload
→ retrain, back to back, before any restart), not something guaranteed to
survive indefinitely. For a genuinely persistent production setup, either
platform would need external storage (e.g. S3) wired into `src/retrain.py`'s
promotion step instead of writing to local disk.

## What happened to the old Render/FastAPI files?

`Dockerfile`, `render.yaml`, `.dockerignore`, and `api/main.py` are still in
the repo — they're not deleted, since they represent real, working, tested
code (and satisfy the assignment's "API creation with Python" requirement on
their own merits). They're just no longer the *deployed* path. If you want to
demonstrate the API separately (e.g. via `/docs` locally), it still works:
```bash
uvicorn api.main:app --reload --port 8000
```
