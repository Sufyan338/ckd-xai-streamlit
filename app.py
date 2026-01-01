# app.py
import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from src.config import TARGET_BINARY, TARGET_STAGE, N_SPLITS
from src.data_io import load_csv
from src.preprocess import apply_outlier_rules, build_preprocessor
from src.models import get_model_specs
from src.evaluate import evaluate_models_cv
from src.xai import global_permutation_importance, shap_summary_if_available

st.set_page_config(page_title="CKD XAI Framework", layout="wide")
st.title("CKD Smart Diagnostic Framework (ML/DL + XAI)")

uploaded = st.file_uploader("Upload CKD dataset (CSV)", type=["csv"])
if not uploaded:
    st.stop()

df = load_csv(uploaded)

# Validate targets
missing = [c for c in [TARGET_BINARY, TARGET_STAGE] if c not in df.columns]
if missing:
    st.error(f"Missing required target column(s): {missing}")
    st.stop()

st.write("Dataset preview")
st.dataframe(df.head(), use_container_width=True)

# Sidebar settings
st.sidebar.header("Settings")
task = st.sidebar.selectbox("Task", ["Binary", "Stage (0–5)"])
target = TARGET_BINARY if task == "Binary" else TARGET_STAGE

outlier_method = st.sidebar.selectbox("Outlier handling", ["None", "IQR", "Z-Score"])
use_smote = st.sidebar.checkbox("Use SMOTE (inside CV folds)", value=True)
use_hpo = st.sidebar.checkbox("Use stronger hyperparams (preset)", value=True)

# Prepare data
df2 = apply_outlier_rules(df, method=outlier_method, exclude_cols=[TARGET_BINARY, TARGET_STAGE])
X = df2.drop(columns=[TARGET_BINARY, TARGET_STAGE], errors="ignore")
y_raw = df2[target]

# Encode y to 0..K-1 to prevent XGBoost class errors (also handles strings like "Stage 3") [web:164]
le = LabelEncoder()
y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

# For display/debug
class_map = {orig: int(enc) for orig, enc in zip(le.classes_, le.transform(le.classes_))}
n_classes = int(y.nunique())
task_mode = "binary" if task == "Binary" else "multiclass"

st.sidebar.divider()
st.sidebar.write("Target encoding map:")
st.sidebar.json(class_map)
st.sidebar.write(f"Encoded classes: {n_classes} (min={int(y.min())}, max={int(y.max())})")

model_specs = get_model_specs(task=task_mode, use_hpo=use_hpo, n_classes=n_classes)
available_models = [k for k, v in model_specs.items() if v.available]

selected_models = st.sidebar.multiselect(
    "Models",
    list(model_specs.keys()),
    default=available_models[:3] if available_models else []
)

tabs = st.tabs(["Train & Compare", "Explain (XAI)"])

with tabs[0]:
    st.subheader("Train & Compare (5-fold CV)")

    if not selected_models:
        st.warning("Select at least one available model from sidebar.")
        st.stop()

    if st.button("Run 5-fold CV", type="primary"):
        def pre_fn(Xfit, dense_output=True):
            return build_preprocessor(Xfit, dense_output=dense_output)

        with st.spinner("Running cross-validation..."):
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
        st.session_state["label_encoder"] = le
        st.session_state["model_specs"] = model_specs
        st.session_state["selected_models"] = selected_models
        st.session_state["summary"] = summary
        st.session_state["fold_store"] = fold_store

        st.success("CV complete")
        st.dataframe(summary, use_container_width=True)

with tabs[1]:
    st.subheader("Explain (XAI)")

    if "X" not in st.session_state:
        st.warning("Run Train & Compare first.")
        st.stop()

    X_ = st.session_state["X"]
    y_ = st.session_state["y"]
    model_specs_ = st.session_state["model_specs"]
    selected_ = st.session_state["selected_models"]

    model_name = st.selectbox("Model to explain", selected_)
    spec = model_specs_.get(model_name)

    if spec is None or not spec.available:
        st.error("Selected model is not available in this deployment.")
        st.stop()

    # Train a single model on full dataset for explanation
    pre = build_preprocessor(X_, dense_output=True)
    model = spec.builder()

    pipe = Pipeline([("pre", pre), ("model", model)])
    with st.spinner("Fitting model for explanation..."):
        pipe.fit(X_, y_)

    st.write("Global feature importance (Permutation) — safe and works broadly:")
    fig = global_permutation_importance(pipe, X_, y_, top_k=15)
    st.pyplot(fig, clear_figure=True)

    if model_name in ["CatBoost", "LightGBM", "XGBoost"]:
        st.write("Optional SHAP summary (tree models):")
        shap_fig, err = shap_summary_if_available(model, X_)
        if shap_fig is not None:
            st.pyplot(shap_fig, clear_figure=True)
        else:
            st.info(f"SHAP failed/not available: {err}")
    else:
        st.info("SHAP TreeExplainer is not used for this model in this version.")
