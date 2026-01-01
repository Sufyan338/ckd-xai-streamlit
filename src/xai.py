import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

def global_permutation_importance(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, top_k: int = 15):
    r = permutation_importance(
        pipeline, X, y,
        n_repeats=10,
        random_state=42,
        scoring="f1_macro",
    )
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head(top_k)

    fig, ax = plt.subplots()
    imp.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title("Global Feature Importance (Permutation)")
    ax.set_xlabel("Mean importance")
    return fig

def shap_summary_if_available(model, X: pd.DataFrame):
    # Optional SHAP (tree models only). TreeExplainer exists for supported tree models. [web:74]
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model, data=X.sample(min(200, len(X)), random_state=42))
        shap_values = explainer.shap_values(X.sample(min(500, len(X)), random_state=7))

        plt.figure()
        shap.summary_plot(shap_values, X.sample(min(500, len(X)), random_state=7), show=False)
        return plt.gcf(), None
    except Exception as e:
        return None, str(e)
