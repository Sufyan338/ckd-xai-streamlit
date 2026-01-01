import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_tree_model_global(model, X: pd.DataFrame, preprocessor, max_background=200, max_explain=500):
    # Fit a simple preprocessing+model pipeline externally if you want the exact transformed space.
    # Starter: explain in original feature space for tree models that can handle it after preprocessing in pipeline later.
    Xb = X.sample(min(max_background, len(X)), random_state=42)
    Xe = X.sample(min(max_explain, len(X)), random_state=7)

    explainer = shap.TreeExplainer(model, data=Xb)  # tree-specific SHAP [web:74]
    shap_values = explainer.shap_values(Xe)

    plt.figure()
    shap.summary_plot(shap_values, Xe, show=False)
    return plt.gcf()
