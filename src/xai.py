import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TREE_MODELS = {"XGBoost", "LightGBM", "CatBoost"}


def _safe_predict_proba(pipe, row_df: pd.DataFrame):
    """Safely call predict_proba for one-row dataframe."""
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(row_df)
            if proba is None:
                return None
            return np.asarray(proba)
        except Exception:
            return None
    return None


def _reference_value(X_ref: pd.DataFrame, col: str):
    """Return a stable reference value (median for numeric, mode for categorical)."""
    if X_ref is None or len(X_ref) == 0 or col not in X_ref.columns:
        return 0.0

    s = X_ref[col]

    if pd.api.types.is_numeric_dtype(s):
        m = s.median()
        return float(m) if pd.notna(m) else 0.0

    s2 = s.dropna().astype(str)
    if len(s2) == 0:
        return ""
    try:
        return str(s2.mode().iloc[0])
    except Exception:
        return str(s2.iloc[0])


def local_perturbation_reason(pipe, X_ref: pd.DataFrame, row_df: pd.DataFrame, top_k: int = 10):
    """
    Model-agnostic local reason:
    Replace each feature with a reference value (median/mode) and measure probability change.
    """
    if X_ref is None or len(X_ref) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No reference data (X_ref is empty)", ha="center", va="center")
        ax.axis("off")
        return fig, pd.DataFrame(columns=["feature", "impact"])

    proba0 = _safe_predict_proba(pipe, row_df)

    # ---- Fallback to label-only sensitivity if proba not available
    if proba0 is None:
        try:
            base_pred = pipe.predict(row_df)[0]
        except Exception:
            base_pred = None

        rows = []
        for col in X_ref.columns:
            pert = row_df.copy()
            pert[col] = _reference_value(X_ref, col)

            try:
                pred = pipe.predict(pert)[0]
                impact = float(pred != base_pred)
            except Exception:
                impact = np.nan

            rows.append((col, impact))

        df = (
            pd.DataFrame(rows, columns=["feature", "impact"])
            .dropna(subset=["impact"])
            .sort_values("impact", ascending=False)
            .head(top_k)
        )

        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(df))))
        ax.barh(df["feature"][::-1], df["impact"][::-1])
        ax.set_title("Local reason (label-change sensitivity)")
        ax.set_xlabel("Impact (0/1)")
        fig.tight_layout()
        return fig, df

    # ---- Probability shift mode
    proba0 = np.asarray(proba0)[0]
    k = int(np.argmax(proba0))
    base = float(proba0[k])

    rows = []
    for col in X_ref.columns:
        pert = row_df.copy()
        pert[col] = _reference_value(X_ref, col)

        p1 = _safe_predict_proba(pipe, pert)
        if p1 is None:
            continue
        p1 = np.asarray(p1)[0]

        # Guard for inconsistent class dimensions
        if k >= len(p1):
            continue

        delta = float(p1[k] - base)
        rows.append((col, delta))

    df = (
        pd.DataFrame(rows, columns=["feature", "delta_prob"])
        .sort_values("delta_prob", key=lambda s: np.abs(s), ascending=False)
        .head(top_k)
    )

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(df))))
    ax.barh(df["feature"][::-1], df["delta_prob"][::-1])
    ax.set_title("Local reason (probability shift)")
    ax.set_xlabel("Δ probability (predicted class)")
    fig.tight_layout()
    return fig, df


def _extract_shap_row_values(shap_values, class_index=None):
    """Normalize different SHAP return formats to a 1D array for a single row."""
    # shap.Explanation
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # multiclass: list of arrays
    if isinstance(shap_values, list):
        if class_index is None:
            class_index = 0
        arr = np.asarray(shap_values[class_index])
    else:
        arr = np.asarray(shap_values)

    # Common shapes: (1, n_features) or (n_features,)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 1:
        return arr

    return arr.reshape(-1)


def local_shap_reason(tree_model, X_background_trans, row_trans, feature_names, class_index=None, top_k: int = 12):
    """
    Tree-only local reason (SHAP TreeExplainer).
    """
    try:
        import shap
    except Exception as e:
        raise ImportError("shap is required for local_shap_reason. Install with: pip install shap") from e

    explainer = shap.TreeExplainer(tree_model, data=X_background_trans)

    # SHAP version compatibility
    try:
        sv = explainer.shap_values(row_trans)
    except Exception:
        sv = explainer(row_trans)

    vals = _extract_shap_row_values(sv, class_index=class_index)

    # Align feature names length
    feature_names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(len(vals))]
    if len(feature_names) != len(vals):
        n = min(len(feature_names), len(vals))
        feature_names = feature_names[:n]
        vals = vals[:n]

    s = (
        pd.Series(vals, index=feature_names)
        .sort_values(key=lambda x: np.abs(x), ascending=False)
        .head(top_k)
    )

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(s))))
    ax.barh(s.index[::-1], s.values[::-1])
    ax.set_title("Local reason (SHAP)")
    ax.set_xlabel("SHAP value (impact)")
    fig.tight_layout()

    df = s.reset_index()
    df.columns = ["feature", "shap_value"]
    return fig, df
