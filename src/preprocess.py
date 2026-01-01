import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre

def apply_outlier_rules(df: pd.DataFrame, method: str, target_cols: list[str]) -> pd.DataFrame:
    if method == "None":
        return df.copy()

    # Starter: apply only to numeric feature columns (not targets)
    df2 = df.copy()
    feature_cols = [c for c in df2.columns if c not in target_cols]
    num_cols = df2[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if method == "Z-Score":
        for c in num_cols:
            mu = df2[c].mean()
            sigma = df2[c].std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                continue
            z = (df2[c] - mu) / sigma
            df2.loc[z > 3, c] = mu + 3 * sigma
            df2.loc[z < -3, c] = mu - 3 * sigma

    if method == "IQR":
        for c in num_cols:
            q1 = df2[c].quantile(0.25)
            q3 = df2[c].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                continue
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            df2[c] = df2[c].clip(lo, hi)

    return df2
