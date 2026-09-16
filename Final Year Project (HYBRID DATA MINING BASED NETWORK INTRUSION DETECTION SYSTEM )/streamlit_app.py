"""
Streamlit web interface for the NIDS Hybrid System
Run with:  streamlit run streamlit_app.py
"""
import sys
import io
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from config.settings import (
    PREPROCESSING_CONFIG, FEATURE_ENGINEERING_CONFIG, XGBOOST_CONFIG,
    LIGHTGBM_CONFIG, RANDOM_FOREST_CONFIG, SVM_CONFIG,
    LOGISTIC_REGRESSION_CONFIG, ENSEMBLE_CONFIG, MODEL_ARTIFACTS_DIR,
    PIPELINE_ARTIFACTS_DIR, RESULTS_DIR_PATH, SIGNATURE_ENGINE_CONFIG, FUSION_CONFIG
)
from utils.data_utils import (
    load_nsl_kdd, convert_to_binary_classification, split_train_test,
    NSL_KDD_COLUMNS
)
from utils.file_handler import ModelFileHandler
from pipeline.training_pipeline import TrainingPipeline
from predict import NIDSInference
from evaluation.metrics import NIDSEvaluator

st.set_page_config(page_title="NIDS Hybrid System", layout="wide")

RUN_SUMMARY_PATH = RESULTS_DIR_PATH / "last_run_summary.joblib"
FEATURE_COLUMNS = [c for c in NSL_KDD_COLUMNS if c not in ("label", "difficulty")]
CATEGORICAL_OPTIONS = {
    "protocol_type": ["tcp", "udp", "icmp"],
    "flag": ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "RSTOS0", "OTH"],
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def artifacts_exist() -> bool:
    return (MODEL_ARTIFACTS_DIR / "hybrid_nids.joblib").exists() and \
           (PIPELINE_ARTIFACTS_DIR / "preprocessor.joblib").exists() and \
           (PIPELINE_ARTIFACTS_DIR / "feature_engineer.joblib").exists()


@st.cache_resource(show_spinner=False)
def load_inference_components(_cache_bust: float):
    return NIDSInference()


def run_full_inference(X: pd.DataFrame, threshold: float = None):
    engine = load_inference_components(st.session_state.get("artifacts_version", 0.0))
    proba = engine.predict_proba(X)
    pred = engine.predict(X, threshold=threshold)
    return pred, proba


def parse_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Accepts either raw headerless NSL-KDD txt/csv (41-43 cols) or a
    CSV that already has the NSL_KDD_COLUMNS header."""
    raw = uploaded_file.read()
    text_buffer = io.StringIO(raw.decode("utf-8"))
    first_line = text_buffer.readline()
    text_buffer.seek(0)

    looks_like_header = any(col in first_line for col in ("duration", "protocol_type", "service"))

    if looks_like_header:
        df = pd.read_csv(text_buffer)
    else:
        n_cols = first_line.count(",") + 1
        if n_cols == len(NSL_KDD_COLUMNS):
            df = pd.read_csv(text_buffer, header=None, names=NSL_KDD_COLUMNS)
        elif n_cols == len(NSL_KDD_COLUMNS) - 1:
            df = pd.read_csv(text_buffer, header=None, names=FEATURE_COLUMNS)
        else:
            raise ValueError(
                f"Unrecognized column count ({n_cols}). Expected {len(NSL_KDD_COLUMNS)} "
                f"(full NSL-KDD incl. label/difficulty) or {len(FEATURE_COLUMNS)} (features only)."
            )
    return df


def save_run_summary(summary: dict):
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(summary, RUN_SUMMARY_PATH)


def load_run_summary():
    if not RUN_SUMMARY_PATH.exists():
        return None
    import joblib
    return joblib.load(RUN_SUMMARY_PATH)


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.title("🛡️ NIDS Hybrid System")
page = st.sidebar.radio(
    "Navigate",
    ["Train Model", "Batch Predict", "Manual Prediction", "Metrics & Charts"],
)

if "artifacts_version" not in st.session_state:
    st.session_state["artifacts_version"] = 0.0

if not artifacts_exist() and page != "Train Model":
    st.sidebar.warning("No trained model found yet. Go to **Train Model** first.")


# --------------------------------------------------------------------------
# PAGE: Train Model
# --------------------------------------------------------------------------

if page == "Train Model":
    st.title("Train Model")
    st.write(
        "Upload the NSL-KDD training file (e.g. `KDDTrain+_20Percent.txt`). "
        "It will be split into an internal train/test set for evaluation."
    )

    train_file = st.file_uploader(
        "NSL-KDD training file (headerless .txt/.csv, 43 columns)", type=["txt", "csv"]
    )
    test_size = st.slider("Test split proportion", 0.1, 0.4, 0.2, 0.05)

    with st.expander("Model configuration (defaults from config/settings.py)"):
        st.json({
            "preprocessing": PREPROCESSING_CONFIG,
            "feature_engineering": FEATURE_ENGINEERING_CONFIG,
            "ensemble": ENSEMBLE_CONFIG,
            "signature_engine": SIGNATURE_ENGINE_CONFIG,
            "fusion": FUSION_CONFIG,
        })

    start = st.button("🚀 Train Model", type="primary", disabled=train_file is None)

    if start and train_file is not None:
        with st.spinner("Loading and preparing data..."):
            df = parse_uploaded_dataset(train_file)
            if "label" not in df.columns:
                st.error("Uploaded file has no 'label' column — cannot train without ground truth.")
                st.stop()
            df = convert_to_binary_classification(df)
            y = df["label"]
            drop_cols = [c for c in ["label", "difficulty"] if c in df.columns]
            X = df.drop(columns=drop_cols)

            X_train, X_val, y_train, y_val = split_train_test(
                X, y, test_size=test_size, random_state=42, stratify=True
            )

        st.success(f"Loaded {len(df)} rows — Train: {len(X_train)}, Validation: {len(X_val)}")

        model_configs = {
            "xgboost": XGBOOST_CONFIG,
            "lightgbm": LIGHTGBM_CONFIG,
            "random_forest": RANDOM_FOREST_CONFIG,
            "svm": SVM_CONFIG,
            "logistic_regression": LOGISTIC_REGRESSION_CONFIG,
        }

        progress_placeholder = st.empty()
        with st.spinner("Training hybrid pipeline (raw signatures + preprocessing → features → 5 models → ensemble → fusion)..."):
            training_pipeline = TrainingPipeline(
                preprocessing_config=PREPROCESSING_CONFIG,
                feature_config=FEATURE_ENGINEERING_CONFIG,
                model_configs=model_configs,
                ensemble_config=ENSEMBLE_CONFIG,
                signature_config=SIGNATURE_ENGINE_CONFIG,
                fusion_config=FUSION_CONFIG,
            )
            results = training_pipeline.train(X_train, y_train, X_val, y_val)

        progress_placeholder.success("Training complete!")

        st.subheader("Results by model")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.format("{:.4f}"))

        # Detailed evaluation + threshold optimization (mirrors main.py)
        y_pred_ensemble = training_pipeline.predict(X_val)
        y_pred_proba_ensemble = training_pipeline.predict_proba(X_val)

        evaluator = NIDSEvaluator()
        cm = evaluator.get_confusion_matrix(y_val.values, y_pred_ensemble)
        report = evaluator.get_classification_report(y_val.values, y_pred_ensemble)
        optimal_threshold = training_pipeline.classification_threshold
        optimal_value = float(results.get("ValidationThreshold", {}).get("f1", 0.0))
        fpr, tpr, _ = evaluator.get_roc_curve(y_val.values, y_pred_proba_ensemble)

        # Feature importance from the fitted feature engineer
        feature_importance = None
        try:
            X_train_pre = training_pipeline.preprocessor.transform(X_train)
            feature_importance = training_pipeline.feature_engineer.get_feature_importance(
                X_train_pre[training_pipeline.feature_engineer.selected_features], y_train
            )
        except Exception:
            pass

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.code(report)

        st.info(f"Optimal F1 threshold: **{optimal_threshold:.3f}** (F1 = {optimal_value:.4f})")

        # Save artifacts (same layout predict.py / NIDSInference expect)
        with st.spinner("Saving model artifacts..."):
            handler = ModelFileHandler()
            handler.save_model(training_pipeline.final_model, MODEL_ARTIFACTS_DIR, filename="hybrid_nids")
            handler.save_ensemble(training_pipeline.ensemble, MODEL_ARTIFACTS_DIR, name="ensemble_nids")
            for model_name, model in training_pipeline.models.items():
                handler.save_model(model.model, MODEL_ARTIFACTS_DIR, filename=f"model_{model_name.lower()}")
            handler.save_model(training_pipeline.preprocessor, PIPELINE_ARTIFACTS_DIR, filename="preprocessor")
            handler.save_model(training_pipeline.feature_engineer, PIPELINE_ARTIFACTS_DIR, filename="feature_engineer")
            (PIPELINE_ARTIFACTS_DIR / "threshold.json").write_text(
                json.dumps({"classification_threshold": training_pipeline.classification_threshold}, indent=2),
                encoding="utf-8"
            )

        summary = {
            "results_df": results_df,
            "confusion_matrix": cm,
            "classification_report": report,
            "roc_fpr": fpr,
            "roc_tpr": tpr,
            "optimal_threshold": optimal_threshold,
            "optimal_f1": optimal_value,
            "feature_importance": feature_importance,
            "n_train": len(X_train),
            "n_test": len(X_val),
        }
        save_run_summary(summary)

        st.session_state["artifacts_version"] = st.session_state["artifacts_version"] + 1.0
        load_inference_components.clear()

        st.success(
            "Artifacts saved. Head to **Batch Predict**, **Manual Prediction**, "
            "or **Metrics & Charts**."
        )


# --------------------------------------------------------------------------
# PAGE: Batch Predict
# --------------------------------------------------------------------------

elif page == "Batch Predict":
    st.title("Batch Predict")

    if not artifacts_exist():
        st.warning("No trained model found. Go to **Train Model** first.")
        st.stop()

    st.write(
        "Upload a file to classify. Accepts the raw NSL-KDD format "
        "(headerless, 41 or 43 columns) or a CSV with matching column headers."
    )

    pred_file = st.file_uploader("File to classify", type=["txt", "csv"], key="predict_upload")
    threshold = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.01)

    if pred_file is not None:
        df = parse_uploaded_dataset(pred_file)
        has_labels = "label" in df.columns

        y_true = None
        if has_labels:
            df_labeled = convert_to_binary_classification(df.copy())
            y_true = df_labeled["label"]

        drop_cols = [c for c in ["label", "difficulty"] if c in df.columns]
        X = df.drop(columns=drop_cols)

        with st.spinner(f"Running inference on {len(X)} rows..."):
            pred, proba = run_full_inference(X, threshold=threshold)

        out = X.copy()
        out["prediction"] = ["Attack" if p == 1 else "Normal" for p in pred]
        out["confidence"] = np.max(proba, axis=1)
        if y_true is not None:
            out["actual"] = ["Attack" if v == 1 else "Normal" for v in y_true]

        st.subheader("Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total samples", len(out))
        c2.metric("Predicted Attack", int((pred == 1).sum()))
        c3.metric("Predicted Normal", int((pred == 0).sum()))

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(
            [int((pred == 0).sum()), int((pred == 1).sum())],
            labels=["Normal", "Attack"], autopct="%1.1f%%",
            colors=["#4C72B0", "#C44E52"],
        )
        st.pyplot(fig)

        if y_true is not None:
            evaluator = NIDSEvaluator()
            metrics = evaluator.evaluate(y_true.values, pred, proba)
            st.subheader("Accuracy against provided labels")
            st.dataframe(pd.DataFrame([metrics]).style.format("{:.4f}"))

        st.subheader("Predictions")
        st.dataframe(out, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions as CSV",
            data=csv_bytes,
            file_name="nids_predictions.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# PAGE: Manual Prediction
# --------------------------------------------------------------------------

elif page == "Manual Prediction":
    st.title("Manual Prediction")

    if not artifacts_exist():
        st.warning("No trained model found. Go to **Train Model** first.")
        st.stop()

    st.write("Enter connection feature values for a single sample.")

    with st.form("manual_predict_form"):
        st.markdown("**Categorical features**")
        c1, c2 = st.columns(2)
        protocol_type = c1.selectbox("protocol_type", CATEGORICAL_OPTIONS["protocol_type"])
        flag = c2.selectbox("flag", CATEGORICAL_OPTIONS["flag"])
        service = st.text_input("service (e.g. http, ftp, smtp, private, domain_u)", value="http")

        st.markdown("**Key numeric features**")
        c1, c2, c3 = st.columns(3)
        duration = c1.number_input("duration", min_value=0, value=0)
        src_bytes = c2.number_input("src_bytes", min_value=0, value=0)
        dst_bytes = c3.number_input("dst_bytes", min_value=0, value=0)

        c1, c2, c3 = st.columns(3)
        count = c1.number_input("count", min_value=0, value=1)
        srv_count = c2.number_input("srv_count", min_value=0, value=1)
        logged_in = c3.selectbox("logged_in", [0, 1])

        c1, c2, c3 = st.columns(3)
        same_srv_rate = c1.slider("same_srv_rate", 0.0, 1.0, 1.0)
        diff_srv_rate = c2.slider("diff_srv_rate", 0.0, 1.0, 0.0)
        serror_rate = c3.slider("serror_rate", 0.0, 1.0, 0.0)

        with st.expander("Advanced: remaining features (defaults are usually fine)"):
            remaining = [c for c in FEATURE_COLUMNS if c not in [
                "protocol_type", "flag", "service", "duration", "src_bytes",
                "dst_bytes", "count", "srv_count", "logged_in",
                "same_srv_rate", "diff_srv_rate", "serror_rate",
            ]]
            advanced_values = {}
            cols = st.columns(4)
            for i, feat in enumerate(remaining):
                advanced_values[feat] = cols[i % 4].number_input(feat, value=0.0, key=f"adv_{feat}")

        threshold = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.01)
        submitted = st.form_submit_button("🔍 Predict", type="primary")

    if submitted:
        row = {
            "protocol_type": protocol_type, "flag": flag, "service": service,
            "duration": duration, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
            "count": count, "srv_count": srv_count, "logged_in": logged_in,
            "same_srv_rate": same_srv_rate, "diff_srv_rate": diff_srv_rate,
            "serror_rate": serror_rate,
            **advanced_values,
        }
        X = pd.DataFrame([row])[FEATURE_COLUMNS]

        pred, proba = run_full_inference(X, threshold=threshold)
        label = "🔴 Attack" if pred[0] == 1 else "🟢 Normal"
        confidence = float(np.max(proba[0]))

        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Prediction", label)
        c2.metric("Confidence", f"{confidence:.1%}")
        st.progress(float(proba[0][1]), text=f"Attack probability: {proba[0][1]:.1%}")


# --------------------------------------------------------------------------
# PAGE: Metrics & Charts
# --------------------------------------------------------------------------

elif page == "Metrics & Charts":
    st.title("Metrics & Charts")

    summary = load_run_summary()
    if summary is None:
        st.warning("No training run recorded yet. Train a model first.")
        st.stop()

    st.caption(f"From last training run — {summary['n_train']} train / {summary['n_test']} test samples")

    st.subheader("Model Comparison")
    st.dataframe(summary["results_df"].style.format("{:.4f}").highlight_max(axis=0, color="#d4f4dd"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix (Ensemble)")
        cm = summary["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.subheader("ROC Curve (Ensemble)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(summary["roc_fpr"], summary["roc_tpr"], label="Ensemble")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.code(summary["classification_report"])

    st.info(
        f"Optimal F1 threshold from last run: **{summary['optimal_threshold']:.3f}** "
        f"(F1 = {summary['optimal_f1']:.4f})"
    )

    if summary.get("feature_importance"):
        st.subheader("Feature Importance (f_classif)")
        fi = pd.Series(summary["feature_importance"]).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(fi))))
        ax.barh(fi.index, fi.values, color="#4C72B0")
        ax.set_xlabel("F-score")
        st.pyplot(fig)
