from typing import Dict
import numpy as np
import pandas as pd

def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    models: list[str],
    task: str,
    use_smote: bool,
    use_hpo: bool
) -> Dict[str, object]:
    trained = {}

    # IMPORTANT: This is a starter skeleton.
    # After you confirm library availability + hyperparams, we will plug real training here.

    for name in models:
        try:
            if name == "XGBoost":
                import xgboost as xgb
                clf = xgb.XGBClassifier(
                    n_estimators=300 if use_hpo else 100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    eval_metric="logloss"
                )
                trained[name] = clf

            elif name == "LightGBM":
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(
                    n_estimators=500 if use_hpo else 200,
                    learning_rate=0.05,
                    random_state=42
                )
                trained[name] = clf

            elif name == "CatBoost":
                from catboost import CatBoostClassifier
                clf = CatBoostClassifier(
                    iterations=800 if use_hpo else 300,
                    learning_rate=0.05,
                    depth=6,
                    verbose=False,
                    random_seed=42
                )
                trained[name] = clf

            elif name == "TabNet":
                from pytorch_tabnet.tab_model import TabNetClassifier
                clf = TabNetClassifier(verbose=0)
                trained[name] = clf

            elif name == "TabPFN":
                from tabpfn import TabPFNClassifier
                clf = TabPFNClassifier()
                trained[name] = clf

        except Exception:
            # If a library isn't installed, just skip in starter.
            continue

    return trained
