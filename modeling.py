# modeling.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def build_models(preprocess, seed: int) -> Dict[str, Pipeline]:
    """
    Pipelines share the same preprocessing to ensure fair comparison across models.
    Hyperparameters should match what you report in the Supplementary hyperparameter table.
    """
    models: Dict[str, Pipeline] = {
        "Logistic": Pipeline(
            [
                ("prep", preprocess),
                ("model", LogisticRegression(max_iter=5000, random_state=seed)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("prep", preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=800,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", preprocess),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=600,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "CatBoost": Pipeline(
            [
                ("prep", preprocess),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=600,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="Logloss",
                        verbose=False,
                        random_seed=seed,
                    ),
                ),
            ]
        ),
    }
    return models


def get_oof_and_fold_scores(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Stratified Group K-Fold:
      - stratify by outcome
      - keep same patient_id within the same fold
    Produces pooled out-of-fold predicted probabilities.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups), start=1):
        model.fit(X.iloc[tr_idx], y[tr_idx])
        p = model.predict_proba(X.iloc[te_idx])[:, 1]
        oof[te_idx] = p

        fold_rows.append(
            {
                "fold": fold,
                "n_test": int(len(te_idx)),
                "prevalence_test": float(np.mean(y[te_idx])),
                "AUROC": float(roc_auc_score(y[te_idx], p)),
                "AUPRC": float(average_precision_score(y[te_idx], p)),
                "Brier": float(brier_score_loss(y[te_idx], p)),
            }
        )

    return oof, pd.DataFrame(fold_rows)


def summarize_oof(y: np.ndarray, oof_pred: np.ndarray, fold_df: pd.DataFrame) -> dict:
    def ms(x):
        x = np.asarray(x, dtype=float)
        return float(np.mean(x)), float(np.std(x, ddof=1))

    auroc_m, auroc_s = ms(fold_df["AUROC"])
    auprc_m, auprc_s = ms(fold_df["AUPRC"])
    brier_m, brier_s = ms(fold_df["Brier"])

    return {
        "N": int(len(y)),
        "Prevalence": float(np.mean(y)),
        "AUROC (OOF)": float(roc_auc_score(y, oof_pred)),
        "AUPRC (OOF)": float(average_precision_score(y, oof_pred)),
        "Brier (OOF)": float(brier_score_loss(y, oof_pred)),
        "AUROC_mean": auroc_m,
        "AUROC_sd": auroc_s,
        "AUPRC_mean": auprc_m,
        "AUPRC_sd": auprc_s,
        "Brier_mean": brier_m,
        "Brier_sd": brier_s,
    }


def evaluate_all_models(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """
    Returns:
      perf_df: model performance summary table
      oof_store: {model_name: oof_pred}
      fold_store: {model_name: fold_df}
    """
    oof_store: Dict[str, np.ndarray] = {}
    fold_store: Dict[str, pd.DataFrame] = {}
    rows = []

    for name, model in models.items():
        oof, fold_df = get_oof_and_fold_scores(
            model, X, y, groups, n_splits=n_splits, seed=seed
        )
        oof_store[name] = oof
        fold_store[name] = fold_df

        s = summarize_oof(y, oof, fold_df)
        rows.append(
            {
                "Model": name,
                "N": s["N"],
                "Prevalence": s["Prevalence"],
                "AUROC (mean±SD)": f'{s["AUROC_mean"]:.3f} ± {s["AUROC_sd"]:.3f}',
                "AUPRC (mean±SD)": f'{s["AUPRC_mean"]:.3f} ± {s["AUPRC_sd"]:.3f}',
                "Brier (mean±SD)": f'{s["Brier_mean"]:.3f} ± {s["Brier_sd"]:.3f}',
                "AUROC (OOF)": s["AUROC (OOF)"],
                "AUPRC (OOF)": s["AUPRC (OOF)"],
                "Brier (OOF)": s["Brier (OOF)"],
            }
        )

    perf_df = (
        pd.DataFrame(rows)
        .sort_values("AUROC (OOF)", ascending=False)
        .reset_index(drop=True)
    )
    return perf_df, oof_store, fold_store


def pick_best_model(perf_df: pd.DataFrame) -> str:
    """Select best model by AUROC (OOF)."""
    return str(perf_df.iloc[0]["Model"])


def fit_pipeline_on_full_data(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    model.fit(X, y)
    return model
