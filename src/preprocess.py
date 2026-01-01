import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def infer_columns(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame, dense_output: bool = True) -> ColumnTransformer:
    num_cols, cat_cols = infer_columns(X)

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre


def apply_outlier_rules(df: pd.DataFrame, method: str, exclude_cols: list[str]) -> pd.DataFrame:
    if method is None or method == "None":
        return df.copy()

    df2 = df.copy()
    feature_cols = [c for c in df2.columns if c not in exclude_cols]
    num_cols = df2[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if method == "Z-Score":
        for c in num_cols:
            mu = df2[c].mean()
            sigma = df2[c].std(ddof=0)
            if not np.isfinite(sigma) or sigma == 0:
                continue
            z = (df2[c] - mu) / sigma
            lo, hi = mu - 3 * sigma, mu + 3 * sigma
            df2.loc[z < -3, c] = lo
            df2.loc[z > 3, c] = hi

    if method == "IQR":
        for c in num_cols:
            q1 = df2[c].quantile(0.25)
            q3 = df2[c].quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            df2[c] = df2[c].clip(lo, hi)

    return df2


def get_feature_names(preprocessor: ColumnTransformer, fallback_input_cols: list[str]) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return [str(c) for c in fallback_input_cols]
