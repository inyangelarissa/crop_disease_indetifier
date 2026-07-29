import json
import pathlib
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import prediction as pred          # noqa: E402
import preprocessing as pp         # noqa: E402
import retrain as retrain_module   # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
INCOMING_DIR = DATA_DIR / "incoming"

CLASS_LABELS = {
    "Healthy": "Healthy",
    "Common_Rust": "Common Rust",
    "Northern_Leaf_Blight": "Northern Leaf Blight",
    "Cercospora_Gray_Leaf_Spot": "Cercospora Gray Leaf Spot",
}
CLASS_COLORS = {
    "Healthy": "#6ba97c",
    "Common_Rust": "#d4a017",
    "Northern_Leaf_Blight": "#b45535",
    "Cercospora_Gray_Leaf_Spot": "#d97a56",
}

st.set_page_config(
    page_title="Maize Scout — Field Diagnostics",
    page_icon="🌽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cache the loaded model across Streamlit reruns (Streamlit reruns the whole
# script on every interaction by default — without this, every click would
# reload the model from disk).
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return pred.load_model()


if "app_started_at" not in st.session_state:
    st.session_state.app_started_at = time.time()


def invalidate_model_cache():
    """Call after a retrain promotes a new model, so the next prediction uses it."""
    pred.clear_model_cache()
    get_model.clear()


# ---------------------------------------------------------------------------
# Sidebar — model health / uptime (rubric requirement: "model up-time")
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌽 Maize Scout")
    st.caption("Field Diagnostics Center")

    model_path = MODELS_DIR / "best_model.keras"
    model_present = model_path.exists()
    session_uptime = time.time() - st.session_state.app_started_at

    st.metric("Session uptime", f"{session_uptime:.0f}s")
    st.write("**Model file:**", "✅ present" if model_present else "❌ missing")

    metrics_path = MODELS_DIR / "test_eval_summary.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.metric("Test accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%")
        if metrics.get("macro_auc"):
            st.metric("Macro ROC-AUC", f"{metrics['macro_auc']:.3f}")
        
    else:
        st.warning("No evaluation metrics recorded yet.")

    st.divider()
    st.caption(
        "The model's own production metrics are the more meaningful health signal here."
    )

tab_overview, tab_diagnose, tab_retrain = st.tabs(
    ["Overview", " Diagnose", "Upload & Retrain"]
)

# ---------------------------------------------------------------------------
# Overview — visualizations (rubric requirement: "data visualizations")
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Class distribution")
    counts = {c: len(list((DATA_DIR / "raw" / c).glob("*"))) for c in pp.CLASSES}
    if sum(counts.values()) > 0:
        df = pd.DataFrame(
            {"Class": [CLASS_LABELS[c] for c in counts], "Images": list(counts.values())}
        )
        st.bar_chart(df.set_index("Class"))
    else:
        st.info("No images found in data/raw/ yet.")

    st.subheader("Per-class performance (test set)")
    if metrics_path.exists() and metrics.get("classification_report_dict"):
        report = metrics["classification_report_dict"]
        rows = []
        for c in pp.CLASSES:
            if c in report:
                rows.append(
                    {
                        "Class": CLASS_LABELS[c],
                        "Precision": report[c]["precision"],
                        "Recall": report[c]["recall"],
                        "F1": report[c]["f1-score"],
                    }
                )
        st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)
    else:
        st.info("No evaluation report available yet — run a retrain to generate one.")

    st.subheader("Confusion matrix (test set)")
    if metrics_path.exists() and metrics.get("confusion_matrix"):
        cm = metrics["confusion_matrix"]
        short = [CLASS_LABELS[c].replace(" ", "\n") for c in pp.CLASSES]
        cm_df = pd.DataFrame(cm, index=short, columns=short)
        st.dataframe(cm_df.style.background_gradient(cmap="YlOrBr"), use_container_width=True)
    else:
        st.info("No confusion matrix available yet.")

# ---------------------------------------------------------------------------
# Diagnose — single prediction (rubric requirement: "model prediction")
# ---------------------------------------------------------------------------
with tab_diagnose:
    st.subheader("Scan a leaf")
    uploaded = st.file_uploader("Upload a maize leaf photo", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, caption="Uploaded leaf", use_container_width=True)

        tmp_path = PROJECT_ROOT / "tmp_uploads" / uploaded.name
        tmp_path.parent.mkdir(exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        try:
            with st.spinner("Scanning…"):
                model = get_model()
                result = pred.predict_single(tmp_path, model=model)

            with col2:
                predicted = result["predicted_class"]
                confidence = result["confidence"]
                is_healthy = predicted == "Healthy"

                st.markdown(f"### {CLASS_LABELS.get(predicted, predicted)}")
                st.markdown(f"**{confidence*100:.1f}% confidence**")
                if is_healthy:
                    st.success("✅ No disease detected")
                else:
                    st.error("⚠️ Disease detected")

                st.write("**Full breakdown:**")
                probs_df = pd.DataFrame(
                    {
                        "Class": [CLASS_LABELS[c] for c in pp.CLASSES],
                        "Probability": [result["class_probabilities"][c] for c in pp.CLASSES],
                    }
                ).set_index("Class")
                st.bar_chart(probs_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Upload & Retrain (rubric requirements: "upload data" + "trigger retraining")
# ---------------------------------------------------------------------------
with tab_retrain:
    st.subheader("Bulk upload new labeled images")
    col1, col2 = st.columns([1, 2])
    with col1:
        upload_class = st.selectbox(
            "Disease class", pp.CLASSES, format_func=lambda c: CLASS_LABELS[c]
        )
    with col2:
        bulk_files = st.file_uploader(
            "Select multiple leaf photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="bulk_uploader",
        )

    if st.button("Upload batch", disabled=not bulk_files):
        dest_dir = INCOMING_DIR / upload_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in bulk_files:
            with open(dest_dir / f.name, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(bulk_files)} image(s) to data/incoming/{upload_class}/")
        st.rerun()

    st.divider()
    st.subheader("Retraining trigger")

    trigger_status = retrain_module.should_trigger_retrain()
    progress = min(1.0, trigger_status["new_images_since_last_retrain"] / trigger_status["threshold"])
    st.progress(progress)
    st.write(
        f"**{trigger_status['new_images_since_last_retrain']} / {trigger_status['threshold']}** "
        f"new images since last retrain"
        + (" — retrain recommended" if trigger_status["should_retrain"] else "")
    )

    epochs = st.slider("Epochs for this retrain", 1, 5, 1)
    if st.button("Retrain now", type="primary"):
        log_area = st.empty()
        log_area.info("Ingesting uploaded images…")
        try:
            ingested = retrain_module.ingest_incoming_images()
            log_area.info(f"Ingested: {ingested}\n\nRetraining (this can take a minute)…")

            with st.spinner("Retraining — re-splitting data, warm-starting, training, evaluating…"):
                result = retrain_module.run_retraining(epochs=epochs)

            invalidate_model_cache()

            if result["promoted"]:
                prev = result.get("previous_test_accuracy")
                st.success(
                    f"✅ Retrain complete in {result['elapsed_seconds']:.0f}s — **promoted to production**. "
                    f"Test accuracy: "
                    f"{f'{prev*100:.1f}%' if prev else 'n/a'} → "
                    f"{result['candidate_test_accuracy']*100:.1f}%"
                )
            else:
                st.warning(
                    f"Retrain complete in {result['elapsed_seconds']:.0f}s — candidate "
                    f"({result['candidate_test_accuracy']*100:.1f}%) did not beat production "
                    f"({result.get('previous_test_accuracy', 0)*100:.1f}%); production model kept."
                )
            st.json(result)
        except Exception as e:
            st.error(f"Retrain failed: {e}")

st.divider()
st.caption("Maize Scout — a maize leaf disease classifier · built by Inyange Larissa")
