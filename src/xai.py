import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TREE_MODELS = {"XGBoost", "LightGBM", "CatBoost"}


def _safe_predict_proba(pipe, row_df: pd.DataFrame):
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(row_df)
        except Exception:
            return None
    return None


def local_perturbation_reason(pipe, X_ref: pd.DataFrame, row_df: pd.DataFrame, top_k: int = 10):
    """
    Model-agnostic local reason:
    Replace each feature with a reference value (median/mode) and measure probability change.
    """
    proba0 = _safe_predict_proba(pipe, row_df)
    if proba0 is None:
        # fallback to label-only delta
        base_pred = int(pipe.predict(row_df)[0])
        rows = []
        for col in X_ref.columns:
            pert = row_df.copy()
            if pd.api.types.is_numeric_dtype(X_ref[col]):
                pert[col] = float(X_ref[col].median())
            else:
                pert[col] = str(X_ref[col].mode(dropna=True).iloc[0]) if X_ref[col].dropna().shape[0] else ""
            pred = int(pipe.predict(pert)[0])
            rows.append((col, float(pred != base_pred)))
        df = pd.DataFrame(rows, columns=["feature", "impact"]).sort_values("impact", ascending=False).head(top_k)
        fig, ax = plt.subplots()
        ax.barh(df["feature"][::-1], df["impact"][::-1])
        ax.set_title("Local reason (label-change sensitivity)")
        ax.set_xlabel("Impact (0/1)")
        return fig, df

    # choose predicted class index
    proba0 = proba0[0]
    k = int(np.argmax(proba0))
    base = float(proba0[k])

    rows = []
    for col in X_ref.columns:
        pert = row_df.copy()
        if pd.api.types.is_numeric_dtype(X_ref[col]):
            pert[col] = float(X_ref[col].median())
        else:
            pert[col] = str(X_ref[col].mode(dropna=True).iloc[0]) if X_ref[col].dropna().shape[0] else ""
        p1 = _safe_predict_proba(pipe, pert)
        if p1 is None:
            continue
        delta = float(p1[0][k] - base)
        rows.append((col, delta))

    df = pd.DataFrame(rows, columns=["feature", "delta_prob"]).sort_values(
        "delta_prob", key=lambda s: np.abs(s), ascending=False
    ).head(top_k)

    fig, ax = plt.subplots()
    ax.barh(df["feature"][::-1], df["delta_prob"][::-1])
    ax.set_title("Local reason (probability shift)")
    ax.set_xlabel("Δ probability (predicted class)")
    return fig, df


def local_shap_reason(tree_model, X_background_trans, row_trans, feature_names, class_index=None, top_k: int = 12):
    """
    Tree-only local reason (SHAP TreeExplainer).
    """
    import shap  # TreeExplainer for tree models [web:74]

    explainer = shap.TreeExplainer(tree_model, data=X_background_trans)  # [web:74]
    sv = explainer.shap_values(row_trans)

    if isinstance(sv, list):
        if class_index is None:
            class_index = 0
        vals = sv[class_index][0]
    else:
        vals = sv[0]

    s = pd.Series(vals, index=feature_names).sort_values(key=lambda x: np.abs(x), ascending=False).head(top_k)

    fig, ax = plt.subplots()
    ax.barh(s.index[::-1], s.values[::-1])
    ax.set_title("Local reason (SHAP)")
    ax.set_xlabel("SHAP value (impact)")
    return fig, s.reset_index().rename(columns={"index": "feature", 0: "shap_value"})
