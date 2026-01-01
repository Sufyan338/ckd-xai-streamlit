import streamlit as st
import pandas as pd

from src.config import TARGET_BINARY, TARGET_STAGE
from src.data_io import load_csv
from src.preprocess import build_preprocessor, apply_outlier_rules
from src.train import train_all_models
from src.evaluate import evaluate_models_cv
from src.xai import explain_tree_model_global

st.set_page_config(page_title="CKD XAI Framework", layout="wide")

st.title("A Smart Diagnostic Framework for Kidney Disease (XAI)")

# ---------- Sidebar ----------
st.sidebar.header("Settings")

task = st.sidebar.selectbox(
    "Task",
    ["Binary (ckd_pred)", "Multiclass (ckd_stage)"]
)

use_smote = st.sidebar.checkbox("Use SMOTE (train folds only)", value=True)
outlier_method = st.sidebar.selectbox("Outlier handling", ["None", "IQR", "Z-Score"])
do_hpo = st.sidebar.checkbox("Use hyperparameters (preset)", value=True)

st.sidebar.divider()
selected_models = st.sidebar.multiselect(
    "Models",
    ["CatBoost", "LightGBM", "XGBoost", "TabPFN", "TabNet"],
    default=["CatBoost", "LightGBM", "XGBoost"]
)

# ---------- Upload ----------
uploaded = st.file_uploader("Upload your dataset CSV", type=["csv"])

if uploaded is None:
    st.info("Upload CSV to continue. Required targets: ckd_pred and ckd_stage.")
    st.stop()

df = load_csv(uploaded)

st.subheader("Dataset preview")
st.write(df.head())

# choose target column
target_col = TARGET_BINARY if task.startswith("Binary") else TARGET_STAGE

if target_col not in df.columns:
    st.error(f"Target column missing: {target_col}")
    st.stop()

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Train & Compare", "Predict", "Explain (XAI)"])

with tab1:
    st.subheader("Train & Compare (5-fold CV)")

    if st.button("Run training + 5-fold CV", type="primary"):
        df2 = apply_outlier_rules(df, method=outlier_method, target_cols=[TARGET_BINARY, TARGET_STAGE])

        X = df2.drop(columns=[TARGET_BINARY, TARGET_STAGE], errors="ignore")
        y = df2[target_col]

        pre = build_preprocessor(X)

        with st.spinner("Training models..."):
            models = train_all_models(
                X, y,
                preprocessor=pre,
                models=selected_models,
                task=("binary" if target_col == TARGET_BINARY else "multiclass"),
                use_smote=use_smote,
                use_hpo=do_hpo
            )

        with st.spinner("Cross-validating..."):
            results = evaluate_models_cv(
                X, y,
                preprocessor=pre,
                fitted_models=models,
                task=("binary" if target_col == TARGET_BINARY else "multiclass"),
                use_smote=use_smote,
                n_splits=5
            )

        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["preprocessor"] = pre
        st.session_state["models"] = models
        st.session_state["cv_results"] = results

        st.success("Done.")

    if "cv_results" in st.session_state:
        st.subheader("CV results")
        st.dataframe(st.session_state["cv_results"], use_container_width=True)

with tab2:
    st.subheader("Predict")

    if "models" not in st.session_state:
        st.warning("Train models first (Train & Compare tab).")
        st.stop()

    models = st.session_state["models"]
    pre = st.session_state["preprocessor"]
    X_cols = st.session_state["X"].columns.tolist()

    st.write("Enter one patient record (single-row prediction).")
    input_data = {}
    cols = st.columns(3)
    for i, c in enumerate(X_cols):
        with cols[i % 3]:
            input_data[c] = st.text_input(c, value="")

    if st.button("Predict"):
        row = pd.DataFrame([input_data])

        # NOTE: For a real app you should cast types properly (float/int/category).
        # This starter keeps it simple; type-casting will be added after you share your dataset schema.
        row_t = pre.fit_transform(st.session_state["X"]).shape  # warm-up placeholder

        st.info("Starter UI ready. After you share column dtypes, type-casting + real predict will be enabled.")

with tab3:
    st.subheader("Explain (XAI)")

    if "models" not in st.session_state:
        st.warning("Train models first.")
        st.stop()

    if len(st.session_state["models"]) == 0:
        st.warning("No trained models found.")
        st.stop()

    model_name = st.selectbox("Pick a model to explain", list(st.session_state["models"].keys()))
    model = st.session_state["models"][model_name]

    st.write("Global explanation (top features).")

    if model_name in ["CatBoost", "LightGBM", "XGBoost"]:
        fig = explain_tree_model_global(
            model=model,
            X=st.session_state["X"],
            preprocessor=st.session_state["preprocessor"],
            max_background=200,
            max_explain=500
        )
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("For TabPFN/TabNet, this starter shows XAI later via permutation importance / native feature masks.")

