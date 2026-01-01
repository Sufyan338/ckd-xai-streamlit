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


def fit_any_model_for_prediction(model_name, spec, X, y):
    """
    Returns a fitted predictor object.
    For most models: a sklearn Pipeline(preprocessor+model)
    For TabNet: returns (preprocessor, tabnet_model) because TabNet may not behave well inside sklearn Pipeline.
    """
    pre = build_preprocessor(X, dense_output=True)
    model = spec.builder()

    if model_name == "TabNet":
        X_np = pre.fit_transform(X)
        model.fit(X_np, y.values)
        return ("TABNET", pre, model)

    # TabPFN usually works with numpy too; pipeline works for most cases
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)
    return ("PIPE", pipe, None)


def predict_one(model_name, fitted_obj, row_df):
    kind = fitted_obj[0]

    if kind == "TABNET":
        _, pre, model = fitted_obj
        row_np = pre.transform(row_df)
        pred = model.predict(row_np)
        # pred may come as shape (1,) or (1,1)
        pred = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
        proba = None
        try:
            proba = model.predict_proba(row_np)[0]
        except Exception:
            pass
        return pred, proba

    # PIPE
    _, pipe, _ = fitted_obj
    pred = int(pipe.predict(row_df)[0])
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(row_df)[0]
        except Exception:
            proba = None
    return pred, proba


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

# Encode y to 0..K-1 (fixes XGBoost multiclass label issues)
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

# 3 tabs now
tabs = st.tabs(["Train & Compare", "Live Prediction", "Explain (XAI)"])

# ---------------- Tab 1: Train & Compare ----------------
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

# ---------------- Tab 2: Live Prediction ----------------
with tabs[1]:
    st.subheader("Live Prediction")

    st.write("Select a model, fill patient features, and click Predict. Forms submit once (no auto-rerun).")  # forms concept [web:193]

    if not selected_models:
        st.warning("Select at least one available model from sidebar.")
        st.stop()

    model_name_pred = st.selectbox("Model for live prediction", selected_models, key="live_pred_model")
    spec_pred = model_specs.get(model_name_pred)

    if spec_pred is None or not spec_pred.available:
        st.error("Selected model is not available in this deployment.")
        st.stop()

    # Build form inputs
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    with st.form("live_pred_form"):  # Every form needs a submit button [web:191]
        c1, c2, c3 = st.columns(3)
        input_row = {}

        for i, col in enumerate(num_cols):
            with [c1, c2, c3][i % 3]:
                default_val = float(X[col].median()) if X[col].notna().any() else 0.0
                input_row[col] = st.number_input(col, value=default_val)

        for j, col in enumerate(cat_cols):
            with [c1, c2, c3][(len(num_cols) + j) % 3]:
                options = sorted(X[col].dropna().astype(str).unique().tolist())
                if not options:
                    options = [""]
                input_row[col] = st.selectbox(col, options=options, index=0)

        submit_pred = st.form_submit_button("Predict")  # required inside form [web:191]

    if submit_pred:
        row_df = pd.DataFrame([input_row])

        # Keep categorical as strings
        for col in cat_cols:
            row_df[col] = row_df[col].astype(str)

        # Train-on-full-data (demo style) then predict one row
        with st.spinner("Training selected model on full dataset (for live prediction)..."):
            fitted_obj = fit_any_model_for_prediction(model_name_pred, spec_pred, X, y)

        pred_enc, proba = predict_one(model_name_pred, fitted_obj, row_df)

        # Decode label back to original
        pred_label = le.inverse_transform([pred_enc])[0]
        st.success(f"Prediction: {pred_label}")

        if proba is not None:
            st.write("Probabilities (encoded class order):")
            st.write(proba)

# ---------------- Tab 3: Explain (XAI) ----------------
with tabs[2]:
    st.subheader("Explain (XAI)")

    if not selected_models:
        st.warning("Select at least one available model from sidebar.")
        st.stop()

    model_name = st.selectbox("Model to explain", selected_models, key="xai_model")
    spec = model_specs.get(model_name)

    if spec is None or not spec.available:
        st.error("Selected model is not available in this deployment.")
        st.stop()

    # Fit one model on full dataset for explanation
    pre = build_preprocessor(X, dense_output=True)
    model = spec.builder()
    pipe = Pipeline([("pre", pre), ("model", model)])

    with st.spinner("Fitting model for explanation..."):
        pipe.fit(X, y)

    st.write("Global feature importance (Permutation):")
    fig = global_permutation_importance(pipe, X, y, top_k=15)
    st.pyplot(fig, clear_figure=True)

    if model_name in ["CatBoost", "LightGBM", "XGBoost"]:
        st.write("Optional SHAP summary (tree models):")
        shap_fig, err = shap_summary_if_available(model, X)
        if shap_fig is not None:
            st.pyplot(shap_fig, clear_figure=True)
        else:
            st.info(f"SHAP failed/not available: {err}")
    else:
        st.info("SHAP TreeExplainer is not used for this model in this version.")
