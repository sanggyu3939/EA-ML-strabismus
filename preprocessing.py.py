# preprocessing.py

import pandas as pd
import datetime as dt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def to_minutes_series(s: pd.Series) -> pd.Series:
    ...
    
def build_preprocessor(X):
    cat_cols = [...]
    num_cols = [...]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median"))
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))
            ]), cat_cols),
        ]
    )
    return preprocess
