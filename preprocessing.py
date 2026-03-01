# preprocessing.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def to_minutes_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric coercion:
    - numeric stays numeric
    - datetime.time -> minutes
    - timedelta-like strings -> minutes
    - fallback: to_numeric
    """
    if pd.api.types.is_numeric_dtype(s):
        return s

    # datetime.time -> minutes
    if s.dropna().map(type).isin([dt.time]).any():
        def f(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, dt.time):
                return (x.hour * 3600 + x.minute * 60 + x.second) / 60.0
            return np.nan

        out = s.map(f)
        if out.isna().all():
            out = pd.to_numeric(s, errors="coerce")
        return out

    # timedelta from string
    td = pd.to_timedelta(s.astype(str), errors="coerce")
    out = td.dt.total_seconds() / 60.0
    if out.isna().all():
        out = pd.to_numeric(s, errors="coerce")
    return out


def ensure_patient_id(
    df: pd.DataFrame,
    patient_id_col: str,
    fallback_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Ensures a patient identifier column exists.
    If patient_id_col doesn't exist, try fallback columns (e.g., '차트번호').
    """
    fallback_cols = fallback_cols or ["차트번호", "chart_no", "chart_number", "id", "patientid"]
    if patient_id_col in df.columns:
        df[patient_id_col] = df[patient_id_col].astype(str)
        return df, patient_id_col

    for c in fallback_cols:
        if c in df.columns:
            df = df.copy()
            df["patient_id"] = df[c].astype(str)
            return df, "patient_id"

    raise ValueError(
        f"Missing patient id column '{patient_id_col}'. "
        f"Tried fallbacks: {fallback_cols}"
    )


@dataclass(frozen=True)
class FeatureSpec:
    features: List[str]
    cat_cols: List[str]


def default_feature_spec() -> FeatureSpec:
    """
    Default feature set used in the manuscript (edit here if needed).
    NOTE: Only columns present in the provided dataset will be used.
    """
    features = [
        "age", "BMI",
        "sexM1F2", "ASAclass", "URIsx",
        "rocuron_dose_kg", "Fentanly_total_dose_intraop",
        "ane_time", "op_time",
        "tube1LMA2", "ane_drug_K1M2KM3pofol45",
    ]
    cat_cols = [
        "sexM1F2", "ASAclass", "URIsx",
        "tube1LMA2", "ane_drug_K1M2KM3pofol45",
    ]
    return FeatureSpec(features=features, cat_cols=cat_cols)


def build_X_y_groups(
    df: pd.DataFrame,
    outcome_col: str,
    patient_id_col: str,
    feature_spec: Optional[FeatureSpec] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Returns:
      X: feature dataframe (only existing columns)
      y: binary outcome (int 0/1)
      groups: patient id array
      used_features: list of actually used feature columns
      used_cat_cols: list of actually used categorical columns
    """
    if outcome_col not in df.columns:
        raise ValueError(f"Missing outcome column: {outcome_col}")

    df, pid_col = ensure_patient_id(df, patient_id_col)
    groups = df[pid_col].astype(str).values

    # outcome coercion
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    if y.isna().any():
        raise ValueError(
            f"Outcome column '{outcome_col}' contains non-numeric or missing values "
            f"after coercion. Please ensure it is coded as 0/1."
        )
    y = y.astype(int).values

    feature_spec = feature_spec or default_feature_spec()
    used_features = [c for c in feature_spec.features if c in df.columns]
    if len(used_features) == 0:
        raise ValueError("No feature columns found in the dataset. Check column names.")

    X = df[used_features].copy()

    # categorical columns intersection
    used_cat_cols = [c for c in feature_spec.cat_cols if c in used_features]
    used_num_cols = [c for c in used_features if c not in used_cat_cols]

    # coerce numeric-like columns
    for c in used_num_cols:
        X[c] = to_minutes_series(X[c])

    # enforce category dtype for categorical
    for c in used_cat_cols:
        X[c] = X[c].astype("category")

    return X, y, groups, used_features, used_cat_cols


def build_preprocessor(
    X: pd.DataFrame,
    cat_cols: List[str],
    drop_first: bool = True,
) -> ColumnTransformer:
    """
    Shared preprocessing for fair model comparison:
    - numeric: median impute
    - categorical: most_frequent impute + onehot (drop first by default)
    """
    num_cols = [c for c in X.columns if c not in cat_cols]
    oh = OneHotEncoder(
        handle_unknown="ignore",
        drop="first" if drop_first else None,
        sparse_output=True,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", oh),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocess
