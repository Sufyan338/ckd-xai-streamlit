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

# Cache pretrained models for performance on Streamlit reruns. [web:210]
@st.cache_resource
def load_pretrained_models():
    required = [
        "binary_pipeline.joblib",
        "binary_label_encoder.joblib",
        "stage_pipeline.joblib",
        "stage_label_encoder.joblib",
    ]
    if not all(artifact_exists(x) for x in required):
        return None

    bin_pipe = load_artifact("binary_pipeline.joblib")
    bin_le = load_artifact("binary_label_encoder.joblib")
    stg_pipe = load_artifact("stage_pipeline.joblib")
    stg_le = load_artifact("stage_label_encoder.joblib")
    return bin_pipe, bin_le, stg_pipe, stg_le


def pipeline_predict_and_explain(pipe: Pipeline, le: LabelEncoder, X_ref: pd.DataFrame, row_df: pd.DataFrame, top_k=12):
    """
    Predict + reason for one sample.
    - If tree model: use SHAP in transformed feature space.
    - Else: model-agnostic local perturbation importance.
    """
    pred_enc = int(pipe.predict(row_df)[0])
    pred_label = le.inverse_transform([pred_enc])[0]

    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(row_df)[0]
        except Exception:
            proba = None

    # Explanation
    base_model = None
    pre = None
    try:
        pre = pipe.named_steps.get("pre", None)
        base_model = pipe.named_steps.get("model", None)
    except Exception:
        pre, base_model = None, None

    model_name = "Other"
    if base_model is not None:
        n = base_model.__class__.__name__.lower()
        if "xgb" in n or "xgboost" in n:
            model_name = "XGBoost"
        elif "lgbm" in n or "lightgbm" in n:
            model_name = "LightGBM"
        elif "catboost" in n:
            model_name = "CatBoost"

    # Prefer SHAP only for tree models [web:74]
    if model_name in TREE_MODELS and pre is not None and base_model is not None:
        # Background and row in transformed feature space
        Xb = X_ref.sample(min(200, len(X_ref)), random_state=42)
        Xb_t = pre.transform(Xb)
        row_t = pre.transform(row_df)

        feat_names = get_feature_names(pre, fallback_input_cols=list(X_ref.columns))

        class_index = None
        if proba is not None and len(proba) > 1:
            class_index = int(np.argmax(proba))

        try:
            fig, df_reason = local_shap_reason(
                tree_model=base_model,
                X_background_trans=Xb_t,
                row_trans=row_t,
                feature_names=feat_names,
                class_index=class_index,
                top_k=top_k
            )
            method = "SHAP (tree)"
            return pred_label, proba, method, fig, df_reason
        except Exception:
            # fallback to perturbation
            pass

    fig, df_reason = local_perturbation_reason(pipe, X_ref, row_df, top_k=top_k)
    method = "Local perturbation (model-agnostic)"
    return pred_label, proba, method, fig, df_reason


# ---------------- Upload dataset ----------------
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

# Sidebar settings
st.sidebar.header("Settings")
outlier_method = st.sidebar.selectbox("Outlier handling", ["None", "IQR", "Z-Score"])
use_smote = st.sidebar.checkbox("Use SMOTE (CV + full-train)", value=True)

# Prepare features
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

        summary, _fold_store = evaluate_models_cv(
            X_all, y,
            build_preprocessor_fn=pre_fn,
            model_specs=model_specs,
            selected_models=selected_models,
            use_smote=use_smote,
            n_splits=N_SPLITS
        )

        st.success("CV complete")
        st.dataframe(summary, use_container_width=True)

# ---------------- Tab 2: Live Prediction (Reason) ----------------
with tabs[1]:
    st.subheader("Live Prediction (with Reasons)")

    mode = st.radio(
        "Prediction mode",
        ["Use pretrained models (recommended)", "Train instantly from uploaded dataset"],
        horizontal=True
    )

    # ---- Build one input row via form (submit once) [web:193]
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

        submit = st.form_submit_button("Predict & Explain")  # must be inside form [web:191]

    if submit:
        row_df = pd.DataFrame([row])
        for col in cat_cols:
            row_df[col] = row_df[col].astype(str)

        if mode.startswith("Use pretrained"):
            pretrained = load_pretrained_models()
            if pretrained is None:
                st.error("Pretrained artifacts not found. Please add them in /models/ folder in GitHub repo.")
                st.stop()

            bin_pipe, bin_le, stg_pipe, stg_le = pretrained

            # Binary
            st.markdown("### Binary result (ckd_pred)")
            pred_label, proba, method, fig, df_reason = pipeline_predict_and_explain(
                pipe=bin_pipe, le=bin_le, X_ref=X_all, row_df=row_df, top_k=12
            )
            st.success(f"Prediction: {pred_label}")
            if proba is not None:
                st.write("Probabilities:", proba)
            st.write("Reason method:", method)
            st.pyplot(fig, clear_figure=True)
            st.dataframe(df_reason, use_container_width=True)

            # Stage
            st.markdown("### Stage result (ckd_stage)")
            pred_label, proba, method, fig, df_reason = pipeline_predict_and_explain(
                pipe=stg_pipe, le=stg_le, X_ref=X_all, row_df=row_df, top_k=12
            )
            st.success(f"Prediction: {pred_label}")
            if proba is not None:
                st.write("Probabilities:", proba)
            st.write("Reason method:", method)
            st.pyplot(fig, clear_figure=True)
            st.dataframe(df_reason, use_container_width=True)

        else:
            # Train instantly from uploaded dataset (demo)
            task_pick = st.selectbox("Choose task to train for prediction", ["Binary (ckd_pred)", "Multiclass (ckd_stage)"], key="instant_task")
            target = TARGET_BINARY if task_pick.startswith("Binary") else TARGET_STAGE
            task_mode = "binary" if target == TARGET_BINARY else "multiclass"

            y_raw = df2[target]
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

            n_classes = int(y.nunique())
            specs = get_model_specs(task=task_mode, use_hpo=True, n_classes=n_classes)
            avail = [k for k, v in specs.items() if v.available]
            chosen = st.selectbox("Model", avail, key="instant_model")
            spec = specs[chosen]

            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import SMOTE

            pre = build_preprocessor(X_all, dense_output=True)
            model = spec.builder()

            if use_smote:
                pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("model", model)])
            else:
                pipe = Pipeline([("pre", pre), ("model", model)])

            with st.spinner("Training model on full dataset..."):
                pipe.fit(X_all, y)

            pred_label, proba, method, fig, df_reason = pipeline_predict_and_explain(
                pipe=pipe, le=le, X_ref=X_all, row_df=row_df, top_k=12
            )

            st.success(f"{target} Prediction: {pred_label}")
            if proba is not None:
                st.write("Probabilities:", proba)
            st.write("Reason method:", method)
            st.pyplot(fig, clear_figure=True)
            st.dataframe(df_reason, use_container_width=True)

# ---------------- Tab 3: Explain (XAI) ----------------
with tabs[2]:
    st.subheader("Explain (XAI)")

    st.write("This tab is kept minimal; the main 'why' is already shown in Live Prediction tab per patient.")
    st.write("Use Live Prediction to see per-sample feature impacts.")
