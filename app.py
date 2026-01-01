import streamlit as st
import pandas as pd

from src.config import TARGET_BINARY, TARGET_STAGE, N_SPLITS
from src.data_io import load_csv
from src.preprocess import apply_outlier_rules, build_preprocessor
from src.models import get_model_specs
from src.evaluate import evaluate_models_cv
from src.xai import global_permutation_importance, shap_summary_if_available

from sklearn.pipeline import Pipeline

st.set_page_config(page_title="CKD XAI Framework", layout="wide")
st.title("CKD Smart Diagnostic Framework (ML/DL + XAI)")

uploaded = st.file_uploader("Upload CKD dataset (CSV)", type=["csv"])
if not uploaded:
    st.stop()

df = load_csv(uploaded)

if TARGET_BINARY not in df.columns or TARGET_STAGE not in df.columns:
    st.error(f"Targets must exist: {TARGET_BINARY}, {TARGET_STAGE}")
    st.stop()

st.write("Preview")
st.dataframe(df.head(), use_container_width=True)

# Sidebar
st.sidebar.header("Settings")
task = st.sidebar.selectbox("Task", ["Binary", "Stage (0–5)"])
target = TARGET_BINARY if task == "Binary" else TARGET_STAGE

outlier_method = st.sidebar.selectbox("Outlier handling", ["None", "IQR", "Z-Score"])
use_smote = st.sidebar.checkbox("Use SMOTE (inside CV folds)", value=True)
use_hpo = st.sidebar.checkbox("Use stronger hyperparams (preset)", value=True)

task_mode = "binary" if task == "Binary" else "multiclass"
n_classes = int(df[TARGET_STAGE].nunique())

model_specs = get_model_specs(task=task_mode, use_hpo=use_hpo, n_classes=n_classes)

available_models = [k for k, v in model_specs.items() if v.available]
selected_models = st.sidebar.multiselect("Models", list(model_specs.keys()), default=available_models[:3])

tabs = st.tabs(["Train & Compare", "Explain (XAI)"])

with tabs[0]:
    st.subheader("Train & Compare (5-fold CV)")
    if st.button("Run 5-fold CV", type="primary"):
        df2 = apply_outlier_rules(df, method=outlier_method, exclude_cols=[TARGET_BINARY, TARGET_STAGE])
        X = df2.drop(columns=[TARGET_BINARY, TARGET_STAGE])
        y = df2[target]

        def pre_fn(Xfit, dense_output=True):
            return build_preprocessor(Xfit, dense_output=dense_output)

        with st.spinner("Running CV..."):
            summary, fold_store = evaluate_models_cv(
                X, y,
                build_preprocessor_fn=pre_fn,
                model_specs=model_specs,
                selected_models=selected_models,
                use_smote=use_smote,
                n_splits=N_SPLITS
            )

        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["summary"] = summary
        st.success("Done")
        st.dataframe(summary, use_container_width=True)

with tabs[1]:
    st.subheader("Explain (XAI)")

    if "X" not in st.session_state:
        st.warning("Run training first.")
        st.stop()

    X = st.session_state["X"]
    y = st.session_state["y"]

    model_name = st.selectbox("Model to explain", selected_models)
    spec = model_specs.get(model_name)

    if spec is None or not spec.available:
        st.error("Selected model is not available in this deployment.")
        st.stop()

    # Train one model on full data just for explanation
    pre = build_preprocessor(X, dense_output=True)
    model = spec.builder()
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)

    st.write("Permutation importance (safe, works for any sklearn-like pipeline):")
    fig = global_permutation_importance(pipe, X, y, top_k=15)
    st.pyplot(fig, clear_figure=True)

    if model_name in ["CatBoost", "LightGBM", "XGBoost"]:
        st.write("Optional SHAP summary (if SHAP works in your environment):")
        shap_fig, err = shap_summary_if_available(model, X)
        if shap_fig is not None:
            st.pyplot(shap_fig, clear_figure=True)
        else:
            st.info(f"SHAP not available/failed: {err}")
