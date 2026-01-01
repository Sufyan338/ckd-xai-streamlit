# src/models.py
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[], Any]
    available: bool
    reason: Optional[str] = None


def _try_import(module_name: str):
    try:
        __import__(module_name)
        return True, None
    except Exception as e:
        return False, str(e)


def get_model_specs(task: str, use_hpo: bool, n_classes: int) -> Dict[str, ModelSpec]:
    """
    task: 'binary' or 'multiclass'
    n_classes: number of encoded classes in y (after LabelEncoder)
    """
    specs: Dict[str, ModelSpec] = {}

    # ---------------- XGBoost ----------------
    ok, err = _try_import("xgboost")
    if ok:
        import xgboost as xgb

        def build():
            # Keep this simple; let XGBoost infer multiclass from labels/objective defaults.
            # Critical: y must be encoded to 0..K-1 for multiclass.
            return xgb.XGBClassifier(
                n_estimators=600 if use_hpo else 200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                tree_method="hist",
                eval_metric="mlogloss" if task == "multiclass" else "logloss",
            )

        specs["XGBoost"] = ModelSpec("XGBoost", build, True)
    else:
        specs["XGBoost"] = ModelSpec("XGBoost", lambda: None, False, err)

    # ---------------- LightGBM ----------------
    ok, err = _try_import("lightgbm")
    if ok:
        import lightgbm as lgb

        def build():
            return lgb.LGBMClassifier(
                n_estimators=1200 if use_hpo else 400,
                learning_rate=0.03 if use_hpo else 0.05,
                random_state=42,
            )

        specs["LightGBM"] = ModelSpec("LightGBM", build, True)
    else:
        specs["LightGBM"] = ModelSpec("LightGBM", lambda: None, False, err)

    # ---------------- CatBoost ----------------
    ok, err = _try_import("catboost")
    if ok:
        from catboost import CatBoostClassifier

        def build():
            return CatBoostClassifier(
                iterations=1500 if use_hpo else 400,
                learning_rate=0.03 if use_hpo else 0.05,
                depth=6,
                loss_function="MultiClass" if task == "multiclass" else "Logloss",
                verbose=False,
                random_seed=42,
            )

        specs["CatBoost"] = ModelSpec("CatBoost", build, True)
    else:
        specs["CatBoost"] = ModelSpec("CatBoost", lambda: None, False, err)

    # ---------------- TabNet ----------------
    ok, err = _try_import("pytorch_tabnet")
    if ok:
        from pytorch_tabnet.tab_model import TabNetClassifier

        def build():
            # Minimal safe settings; can tune later
            return TabNetClassifier(verbose=0)

        specs["TabNet"] = ModelSpec("TabNet", build, True)
    else:
        specs["TabNet"] = ModelSpec("TabNet", lambda: None, False, err)

    # ---------------- TabPFN ----------------
    ok, err = _try_import("tabpfn")
    if ok:
        from tabpfn import TabPFNClassifier

        def build():
            return TabPFNClassifier()

        specs["TabPFN"] = ModelSpec("TabPFN", build, True)
    else:
        specs["TabPFN"] = ModelSpec("TabPFN", lambda: None, False, err)

    return specs
