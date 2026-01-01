import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    fitted_models: dict,
    task: str,
    use_smote: bool,
    n_splits: int = 5
) -> pd.DataFrame:
    # Starter CV: reports macro metrics; ROC-AUC + t-tests will be added after you approve structure.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    for name, base_model in fitted_models.items():
        fold_scores = []

        for tr_idx, te_idx in skf.split(X, y):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

            # NOTE: SMOTE should be applied only on transformed train fold (we’ll add with imblearn Pipeline).
            pipe = Pipeline(steps=[
                ("pre", preprocessor),
                ("model", base_model),
            ])

            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)

            fold_scores.append({
                "F1_macro": f1_score(yte, pred, average="macro"),
                "Precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
                "Recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
            })

        df_f = pd.DataFrame(fold_scores)
        rows.append({
            "Model": name,
            "F1_macro_mean": df_f["F1_macro"].mean(),
            "F1_macro_std": df_f["F1_macro"].std(ddof=1),
            "Precision_macro_mean": df_f["Precision_macro"].mean(),
            "Recall_macro_mean": df_f["Recall_macro"].mean(),
        })

    return pd.DataFrame(rows).sort_values("F1_macro_mean", ascending=False)
