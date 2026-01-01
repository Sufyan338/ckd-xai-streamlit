import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from src.config import TARGET_BINARY, TARGET_STAGE, N_SPLITS
from src.data_io import load_csv
from src.preprocess import apply_outlier_rules, build_preprocessor, infer_columns, get_feature_names
from src.models import get_model_specs
from src.evaluate import evaluate_models_cv
from src.persistence import load_artifact, artifact_exists
from src.xai import TREE_MODELS, local_shap_reason, local_perturbation_reason

st.set_page_config(page_title="CKD XAI Framework", layout="wide")
st.title("CKD Smart Diagnostic Framework (ML/DL + XAI)")

# Cache pretrained models (best for Streamlit performance) [web:210]
@st.cache_resource
def load_pretrained():
    if not (artifact_exists("binary_pipeline.joblib") and artifact_exists("binary_label_encoder.joblib") and
            artifact_exists("stage_pipeline.joblib") and artifact_exists("stage_label_encoder.joblib")):
        return None

    bin_pipe = load_artifact("binary_pipeline.joblib")
    bin_le = load_artifact("binary_label_encoder.joblib")
    stg_pipe = load_artifact("stage_pipeline.joblib")
    stg_le = load_artifact("stage_label_encoder.joblib")
    return bin_pipe, bin_le, stg_pipe, stg_le


uploaded = st.file_uploader("Upload CKD dataset (CSV)", type=["csv"])
if not uploaded:
    st.info("Upload CSV to continue.")
    st.stop()

df = load_csv(uploaded)
missing = [c for c in [TARGET_BINARY, TARGET_STAGE] if c not in df.columns]
if missing:
    st.error(f"Missing required target column(s): {missing}")
    st.stop()

st.write("Dataset preview")
st.dataframe(df.head(), use_container_width=True)

# Sidebar
st.sidebar.header("Settings")
outlier_method = st.sidebar.selectbox("Outlier handling", ["None", "IQR", "Z-Score"])
use_smote = st.sidebar.checkbox("Use SMOTE (CV + training)", value=True)

# Prepare features once
df2 = apply_outlier_rules(df, method=outlier_method, exclude_cols=[TARGET_BINARY, TARGET_STAGE])
X_all = df2.drop(columns=[TARGET_BINARY, TARGET_STAGE], errors="ignore")
num_cols, cat_cols = infer_columns(X_all)

tabs = st.tabs(["Train & Compare", "Live Prediction (Reason)", "Explain (XAI)"])

# ---------------- Tab 1: Train & Compare ----------------
with tabs[0]:
    st.subheader("Train & Compare (5-fold CV)")

    task = st.selectbox("Task", ["Binary (ckd_pred)", "Multiclass (ckd_stage)"], key="train_task")
    target = TARGET_BINARY if task.startswith("Binary") else TARGET_STAGE
    task_mode = "binary" if target == TARGET_BINARY else "multiclass"

    y_raw = df2[target]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    n_classes = int(y.nunique())
    model_specs = get_model_specs(task=task_mode, use_hpo=True, n_classes=n_classes)

    available_models = [k for k, v in model_specs.items() if v.available]
    selected_models = st.multiselect("Models", list(model_specs.keys()), default=available_models[:3])

    st.caption("Target encoding map (original → encoded):")
    st.json({orig: int(enc) for orig, enc in zip(le.classes_, le.transform(le.classes_))})

    if st.button("Run 5-fold CV", type="primary"):
        def pre_fn(Xfit, dense_output=True):
            return build_preprocessor(Xfit, dense_output=dense_output)

        summary, fold_store = evaluate_models_cv(
            X_all, y,
            build_preprocessor_fn=pre_fn,
            model_specs=model_specs,
            selected_models=selected_models,
            use_smote=use_smote,
            n_splits=N_SPLITS
        )

        st.session_state["train_X"] = X_all
        st.session_state["train_y"] = y
        st.session_state["train_le"] = le
        st.session_state["train_target"] = target
        st.session_state["train_task_mode"] = task_mode
        st.session_state["model_specs"] = model_specs
        st.session_state["selected_models"] = selected_models

        st.success("CV complete")
        st.dataframe(summary, use_container_width=True)

# ---------------- Tab 2: Live Prediction + Reason ----------------
with tabs[1]:
    st.subheader("Live Prediction (with Reasons)")

    mode = st.radio(
        "Prediction mode",
        ["Use pretrained models (recommended)", "Train instantly from uploaded dataset"],
        horizontal=True
    )

    pretrained = load_pretrained()

    if mode.startswith("Use pretrained"):
        if pretrained is None:
            st.warning("Pretrained artifacts not found. Add joblib files in /models/ folder in GitHub repo.")
            st.stop()
        bin_pipe, bin_le, stg_pipe, stg_le = pretrained

        pipe_bin = bin_pipe
        pipe_stg = stg_pipe

        X_ref = X_all.copy()  # for fallback explanations
        model_name_for_reason_bin = "TreeOrOther"
        model_name_for_reason_stg = "TreeOrOther"

        # Try to detect base estimator name from pipeline
        def get_model_name(pipe):
            try:
                m = pipe.named_steps.get("model", None)
                if m is None:
                    return "Unknown"
                n = m.__class__.__name__.lower()
                if "xgb" in n or "xgboost" in n:
                    return "XGBoost"
                if "lgbm" in n or "lightgbm" in n:
                    return "LightGBM"
                if "catboost" in n:
                    return "CatBoost"
                return "Other"
            except Exception:
                return "Unknown"

        model_name_for_reason_bin = get_model_name(pipe_bin)
        model_name_for_reason_stg = get_model_name(pipe_stg)

    else:
        # train from dataset (demo)
        task_mode = st.selectbox("Train which model for prediction", ["binary", "multiclass"], key="live_train_task")
        target = TARGET_BINARY if task_mode == "binary" else TARGET_STAGE

        y_raw = df2[target]
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

        n_classes = int(y.nunique())
        specs = get_model_specs(task=("binary" if task_mode == "binary" else "multiclass"), use_hpo=True, n_classes=n_classes)
        avail = [k for k, v in specs.items() if v.available]

        chosen = st.selectbox("Model", avail, key="live_model_choice")
        spec = specs[chosen]

        # Training pipeline (SMOTE applied only at fit-time in imblearn Pipeline)
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        pre = build_preprocessor(X_all, dense_output=True)
        model = spec.builder()

        if use_smote:
            pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("model", model)])
        else:
            pipe = Pipeline([("pre", pre), ("model", model)])

        with st.spinner("Training selected model on full dataset..."):
            pipe.fit(X_all, y)

        # set both tasks to same pipe just for demo
        if task_mode == "binary":
            pipe_bin, bin_le = pipe, le
            pipe_stg, stg_le = None, None
            model_name_for_reason_bin = chosen
        else:
            pipe_stg, stg_le = pipe, le
            pipe_bin, bin_le = None, None
            model_name_for_reason_stg = chosen

        X_ref = X_all.copy()

    st.write("Enter one patient (features). Then app will predict AND explain which features influenced it.")
    # Forms submit pattern [web:193]
    with st.form("live_pred_form"):
        c1, c2, c3 = st.columns(3)
        row = {}

        for i, col in enumerate(num_cols):
            with [c1, c2, c3][i % 3]:
                default_val = float(X_all[col].median()) if X_all[col].notna().any() else 0.0
                row[col] = st.number_input(col, value=default_val)

        for j, col in enumerate(cat_cols):
            with [c1, c2, c3][(len(num_cols) + j) % 3]:
                options = sorted(X_all[col].dropna().astype(str).unique().tolist())
                if not options:
                    options = [""]
                row[col] = st.selectbox(col, options=options, index=0)

        submit = st.form_submit_button("Predict & Explain")

    if submit:
        row_df = pd.DataFrame([row])
        for col in cat_cols:
            row_df[col] = row_df[col].astype(str)

        # ---- Binary prediction ----
        if "pipe_bin" in locals() and pipe_bin is not None:
            pred_enc = int(pipe_bin.predict(row_df)[0])
            pred_label = bin_le.inverse_transform([pred_enc])[0]
            st.success(f"Binary (ckd_pred): {pred_label}")

            # reason
            base_model = getattr(pipe_bin, "named_steps", {}).get("model", None)
            pre = getattr(pipe
