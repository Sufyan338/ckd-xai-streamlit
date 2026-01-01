import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
import requests

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from src.config import TARGET_BINARY, TARGET_STAGE, N_SPLITS
from src.data_io import load_csv
from src.preprocess import apply_outlier_rules, build_preprocessor, infer_columns, get_feature_names
from src.models import get_model_specs
from src.evaluate import evaluate_models_cv
from src.persistence import load_artifact, artifact_exists
from src.xai import TREE_MODELS, local_shap_reason, local_perturbation_reason


# ---------------- Page ----------------
st.set_page_config(page_title="CKD XAI Framework", layout="wide")
st.title("CKD Smart Diagnostic Framework (ML/DL + XAI)")


# ---------------- Option A: Download pretrained artifacts from GitHub Releases ----------------
# Repo: https://github.com/Sufyan338/ckd-xai-streamlit  [web:274]
GITHUB_OWNER = "Sufyan338"
GITHUB_REPO = "ckd-xai-streamlit"

RELEASE_ASSET_URLS = {
    "binary_pipeline.joblib": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/binary_pipeline.joblib",
    "binary_label_encoder.joblib": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/binary_label_encoder.joblib",
    "stage_pipeline.joblib": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/stage_pipeline.joblib",
    "stage_label_encoder.joblib": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/stage_label_encoder.joblib",
}

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dst: Path, timeout: int = 180):
    r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    with dst.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


@st.cache_resource
def ensure_pretrained_artifacts():
    """
    Download pretrained artifacts once per container and reuse them via Streamlit resource cache. [web:210][web:215]
    """
    for name, url in RELEASE_ASSET_URLS.items():
        dst = MODELS_DIR / name
        if (not dst.exists()) or (dst.stat().st_size < 1024):
            _download_file(url, dst)
    return True


# ---------------- Helpers ----------------
def render_figure(fig):
    """Render matplotlib or plotly figures safely in Streamlit."""
    if fig is None:
        st.info("No figure returned.")
        return

    # Plotly
    if hasattr(fig, "to_plotly_json"):
        st.plotly_chart(fig, use_container_width=True)
        return

    # Matplotlib
    try:
        import matplotlib.figure
        if isinstance(fig, matplotlib.figure.Figure):
            st.pyplot(fig, clear_figure=True)
            return
    except Exception:
        pass

    # Fallback
    st.write(fig)


def get_pipeline_parts(pipe):
    """Robustly extract preprocessor and model from sklearn/imbalance pipelines."""
    pre = None
    model = None

    named = getattr(pipe, "named_steps", {}) or {}

    for k in ("pre", "preprocessor", "prep", "transform", "transformer", "ct"):
        if k in named:
            pre = named[k]
            break

    for k in ("model", "clf", "classifier", "estimator"):
        if k in named:
            model = named[k]
            break

    if model is None and hasattr(pipe, "steps") and pipe.steps:
        model = pipe.steps[-1][1]

    return pre, model


def safe_inverse_transform(le, pred):
    """Return human-readable label from prediction using LabelEncoder when possible."""
    try:
        pred_enc = int(pred)
        return le.inverse_transform([pred_enc])[0]
    except Exception:
        return str(pred)


@st.cache_resource
def load_pretrained_models():
    """Load pretrained artifacts (download first)."""
    try:
        ensure_pretrained_artifacts()
    except Exception as e:
        st.error(f"Failed to download pretrained artifacts: {e}")
        return None

    required = [
        "binary_pipeline.joblib",
        "binary_label_encoder.joblib",
        "stage_pipeline.joblib",
        "stage_label_encoder.joblib",
    ]

    if not all(artifact_exists(x) for x in required):
        return None

    try:
        bin_pipe = load_artifact("binary_pipeline.joblib")
        bin_le = load_artifact("binary_label_encoder.joblib")
        stg_pipe = load_artifact("stage_pipeline.joblib")
        stg_le = load_artifact("stage_label_encoder.joblib")
        return bin_pipe, bin_le, stg_pipe, stg_le
    except Exception as e:
        st.error(str(e))
        return None


def pipeline_predict_and_explain(
    pipe: Pipeline,
    le: LabelEncoder,
    X_ref: pd.DataFrame,
    row_df: pd.DataFrame,
    top_k: int = 12,
):
    """Predict + reason for one sample."""
    raw_pred = pipe.predict(row_df)[0]
    pred_label = safe_inverse_transform(le, raw_pred)

    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(row_df)[0]
        except Exception:
            proba = None

    if X_ref is None or len(X_ref) == 0:
        return pred_label, proba, "No XAI (empty reference set)", None, pd.DataFrame()

    pre, base_model = get_pipeline_parts(pipe)

    model_name = base_model.__class__.__name__.lower() if base_model is not None else ""
    is_tree = (
        model_name in {m.lower() for m in TREE_MODELS}
        or any(x in model_name for x in ("xgb", "xgboost", "lgbm", "lightgbm", "catboost"))
    )

    if is_tree and pre is not None and base_model is not None:
        try:
            Xb = X_ref.sample(n=min(200, len(X_ref)), random_state=42)
            Xb_t = pre.transform(Xb)
            row_t = pre.transform(row_df)

            feat_names = get_feature_names(pre, fallback_input_cols=list(X_ref.columns))

            class_index = None
            if proba is not None:
                try:
                    if np.ndim(proba) == 1 and len(proba) > 1:
                        class_index = int(np.argmax(proba))
                except Exception:
                    class_index = None

            fig, df_reason = local_shap_reason(
                tree_model=base_model,
                X_background_trans=Xb_t,
                row_trans=row_t,
                feature_names=feat_names,
                class_index=class_index,
                top_k=top_k,
            )
            return pred_label, proba, "SHAP (tree)", fig, df_reason
        except Exception:
            pass

    fig, df_reason = local_perturbation_reason(pipe, X_ref, row_df, top_k=top_k)
    return pred_label, proba, "Local perturbation (model-agnostic)", fig, df_reason


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


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
outlier_method = st.sidebar.selectbox("Outlier handling", ["None", "IQR", "Z-Score"])
use_smote = st.sidebar.checkbox("Use SMOTE (CV + full-train)", value=True)


# ---------------- Prepare features ----------------
df2 = apply_outlier_rules(df, method=outlier_method, exclude_cols=[TARGET_BINARY, TARGET_STAGE])
X_all = df2.drop(columns=[TARGET_BINARY, TARGET_STAGE], errors="ignore")

if X_all.shape[1] == 0:
    st.error("No feature columns found after dropping target columns.")
    st.stop()

num_cols, cat_cols = infer_columns(X_all)


tabs = st.tabs(["Train & Compare", "Live Prediction (Reason)", "Explain (XAI)"])


# ---------------- Tab 1: Train & Compare ----------------
with tabs[0]:
    st.subheader("Train & Compare (CV)")

    task = st.selectbox("Task", ["Binary (ckd_pred)", "Multiclass (ckd_stage)"], key="train_task")
    target = TARGET_BINARY if task.startswith("Binary") else TARGET_STAGE
    task_mode = "binary" if target == TARGET_BINARY else "multiclass"

    y_raw = df2[target]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    n_classes = int(y.nunique())
    model_specs = get_model_specs(task=task_mode, use_hpo=True, n_classes=n_classes)

    available_models = [k for k, v in model_specs.items() if getattr(v, "available", False)]
    default_models = available_models[:3] if available_models else []

    selected_models = st.multiselect("Models", list(model_specs.keys()), default=default_models)

    st.caption("Target encoding map (original → encoded):")
    st.json({orig: int(enc) for orig, enc in zip(le.classes_, le.transform(le.classes_))})

    if st.button("Run CV", type="primary"):
        if not selected_models:
            st.error("Select at least one model to run CV.")
            st.stop()

        def pre_fn(Xfit, dense_output=True):
            return build_preprocessor(Xfit, dense_output=dense_output)

        summary, _fold_store = evaluate_models_cv(
            X_all,
            y,
            build_preprocessor_fn=pre_fn,
            model_specs=model_specs,
            selected_models=selected_models,
            use_smote=use_smote,
            n_splits=N_SPLITS,
        )

        st.success("CV complete")
        st.dataframe(summary, use_container_width=True)


# ---------------- Tab 2: Live Prediction ----------------
with tabs[1]:
    st.subheader("Live Prediction (with Reasons)")

    mode = st.radio(
        "Prediction mode",
        ["Use pretrained models (recommended)", "Train instantly from uploaded dataset"],
        horizontal=True,
    )

    instant_target = None
    instant_task_mode = None
    instant_model_name = None

    if mode.startswith("Train instantly"):
        st.markdown("#### Instant-training options")
        task_pick = st.selectbox(
            "Choose task to train for prediction",
            ["Binary (ckd_pred)", "Multiclass (ckd_stage)"],
            key="instant_task",
        )
        instant_target = TARGET_BINARY if task_pick.startswith("Binary") else TARGET_STAGE
        instant_task_mode = "binary" if instant_target == TARGET_BINARY else "multiclass"

        y_raw_i = df2[instant_target]
        le_i = LabelEncoder()
        y_i = pd.Series(le_i.fit_transform(y_raw_i.astype(str)), index=y_raw_i.index)
        n_classes_i = int(y_i.nunique())

        specs = get_model_specs(task=instant_task_mode, use_hpo=True, n_classes=n_classes_i)
        avail = [k for k, v in specs.items() if getattr(v, "available", False)]

        if not avail:
            st.error("No available models for instant training.")
            st.stop()

        instant_model_name = st.selectbox("Model", avail, key="instant_model")

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

        if mode.startswith("Use pretrained"):
            pretrained = load_pretrained_models()
            if pretrained is None:
                st.error(
                    "Pretrained artifacts not available.\n"
                    "1) Upload these 4 files to GitHub Releases assets:\n"
                    "   - binary_pipeline.joblib\n"
                    "   - binary_label_encoder.joblib\n"
                    "   - stage_pipeline.joblib\n"
                    "   - stage_label_encoder.joblib\n"
                    "2) Redeploy Streamlit app."
                )
                st.stop()

            bin_pipe, bin_le, stg_pipe, stg_le = pretrained

            st.markdown("### Binary result (ckd_pred)")
            pred_label, proba, method, fig, df_reason = pipeline_predict_and_explain(
                pipe=bin_pipe,
                le=bin_le,
                X_ref=X_all,
                row_df=row_df,
                top_k=12,
            )
            st.success(f"Prediction: {pred_label}")
            if proba is not None:
                st.write("Probabilities:", proba)
            st.write("Reason method:", method)
            render_figure(fig)
            st.dataframe(df_reason, use_container_width=True)

            st.markdown("### Stage result (ckd_stage)")
            pred_label, proba, method, fig, df_reason = pipeline_predict_and_explain(
                pipe=stg_pipe,
                le=stg_le,
                X_ref=X_all,
                row_df=row_df,
                top_k=12,
            )
            st.success(f"Prediction: {pred_label}")
            if proba is not None:
                st.write("Probabilities:", proba)
            st.write("Reason method:", method)
            render_figure(fig)
            st.dataframe(df_reason, use_container_width=True)

        else:
            if instant_target is None or instant_task_mode is None or instant_model_name is None:
                st.error("Instant-training options are not configured.")
                st.stop()

            y_raw = df2[instant_target]
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

            n_classes = int(y.nunique())
            specs = get_model_specs(task=instant_task_mode, use_hpo=True, n_classes=n_classes)
            spec = specs[instant_model_name]

            pre = build_preprocessor(X_all, dense_output=True)
            model = spec.builder()

            if use_smote:
                try:
                    from imblearn.pipeline import Pipeline as ImbPipeline
                    from imblearn.over_sampling import SMOTE

                    pipe = ImbPipeline(
                        [
                            ("pre", pre),
                            ("smote", SMOTE(random_state=42)),
                            ("model", model),
                        ]
                    )
                except Exception:
                    st.warning("imblearn not available; training without SMOTE.")
                    pipe = Pipeline([("pre", pre), ("model", model)])
            else:
                pipe = Pipeline([("pre", pre), ("model", model)])

            with st.spinner("Training model on full dataset..."):
                pipe.fit(X_all, y)

            pred_label, proba, method, fig, df_reason = pipeline_predict_and_explain(
                pipe=pipe,
                le=le,
                X_ref=X_all,
                row_df=row_df,
                top_k=12,
            )

            st.success(f"{instant_target} Prediction: {pred_label}")
            if proba is not None:
                st.write("Probabilities:", proba)
            st.write("Reason method:", method)
            render_figure(fig)
            st.dataframe(df_reason, use_container_width=True)


# ---------------- Tab 3: Explain (XAI) ----------------
with tabs[2]:
    st.subheader("Explain (XAI)")
    st.write("The main per-patient explanation is shown in the Live Prediction tab.")
    st.write("Use Live Prediction to see per-sample feature impacts.")
