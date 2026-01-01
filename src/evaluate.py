import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

def _make_pipeline(preprocessor, model, use_smote: bool):
    if use_smote:
        # SMOTE must be inside CV folds to avoid leakage.
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline(steps=[
            ("pre", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])
    return Pipeline(steps=[
        ("pre", preprocessor),
        ("model", model),
    ])

def evaluate_models_cv(X, y, build_preprocessor_fn, model_specs, selected_models, use_smote, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    summary_rows = []
    fold_store = {}  # for t-tests later (optional)

    for name in selected_models:
        spec = model_specs.get(name)
        if spec is None or not spec.available:
            summary_rows.append({"Model": name, "Status": f"Unavailable: {getattr(spec,'reason', 'not found')}"})
            continue

        fold_metrics = []

        for tr_idx, te_idx in skf.split(X, y):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

            pre = build_preprocessor_fn(Xtr, dense_output=True)  # dense helps SMOTE
            model = spec.builder()

            # clone when possible to avoid state leakage
            try:
                model = clone(model)
            except Exception:
                pass

            pipe = _make_pipeline(pre, model, use_smote=use_smote)

            # TabNet needs numpy arrays; handle separately
            if name == "TabNet":
                pipe.fit(Xtr, ytr)  # will work only if TabNet supports inside sklearn pipeline in your environment
                pred = pipe.predict(Xte)
            else:
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)

            fold_metrics.append({
                "accuracy": accuracy_score(yte, pred),
                "f1_macro": f1_score(yte, pred, average="macro"),
                "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
                "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
            })

        dfm = pd.DataFrame(fold_metrics)
        fold_store[name] = dfm

        summary_rows.append({
            "Model": name,
            "Status": "OK",
            "Accuracy (mean±std)": f"{dfm['accuracy'].mean():.4f} ± {dfm['accuracy'].std(ddof=1):.4f}",
            "F1-macro (mean±std)": f"{dfm['f1_macro'].mean():.4f} ± {dfm['f1_macro'].std(ddof=1):.4f}",
            "Precision-macro (mean±std)": f"{dfm['precision_macro'].mean():.4f} ± {dfm['precision_macro'].std(ddof=1):.4f}",
            "Recall-macro (mean±std)": f"{dfm['recall_macro'].mean():.4f} ± {dfm['recall_macro'].std(ddof=1):.4f}",
        })

    return pd.DataFrame(summary_rows), fold_store
