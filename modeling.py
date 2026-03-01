# ea_ml_pipeline.py
import argparse
import os
import datetime as dt
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

# -------------------------
# Utility
# -------------------------
def to_minutes_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    if s.dropna().map(type).isin([dt.time]).any():
        def f(x):
            if pd.isna(x): return np.nan
            if isinstance(x, dt.time):
                return (x.hour*3600 + x.minute*60 + x.second)/60.0
            return np.nan
        out = s.map(f)
        if out.isna().all():
            out = pd.to_numeric(s, errors="coerce")
        return out
    td = pd.to_timedelta(s.astype(str), errors="coerce")
    out = td.dt.total_seconds()/60.0
    if out.isna().all():
        out = pd.to_numeric(s, errors="coerce")
    return out


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


# -------------------------
# OOF evaluation
# -------------------------
def get_oof_and_fold_scores(model, X, y, groups, n_splits=5, seed=42):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups), start=1):
        model.fit(X.iloc[tr_idx], y[tr_idx])
        p = model.predict_proba(X.iloc[te_idx])[:, 1]
        oof[te_idx] = p

        fold_rows.append({
            "fold": fold,
            "n_test": int(len(te_idx)),
            "prevalence_test": float(np.mean(y[te_idx])),
            "AUROC": roc_auc_score(y[te_idx], p),
            "AUPRC": average_precision_score(y[te_idx], p),
            "Brier": brier_score_loss(y[te_idx], p),
        })
    return oof, pd.DataFrame(fold_rows)


def summarize_oof(y, oof_pred, fold_df):
    overall = {
        "AUROC_oof": roc_auc_score(y, oof_pred),
        "AUPRC_oof": average_precision_score(y, oof_pred),
        "Brier_oof": brier_score_loss(y, oof_pred),
        "N": int(len(y)),
        "Prevalence": float(np.mean(y)),
    }
    def ms(x): return float(np.mean(x)), float(np.std(x, ddof=1))
    auroc_m, auroc_s = ms(fold_df["AUROC"])
    auprc_m, auprc_s = ms(fold_df["AUPRC"])
    brier_m, brier_s = ms(fold_df["Brier"])
    overall.update({
        "AUROC_mean": auroc_m, "AUROC_sd": auroc_s,
        "AUPRC_mean": auprc_m, "AUPRC_sd": auprc_s,
        "Brier_mean": brier_m, "Brier_sd": brier_s,
    })
    return overall


# -------------------------
# Plotting
# -------------------------
def plot_roc_pr_calibration(y, oof_store, outdir):
    # ROC
    plt.figure()
    for name, p in oof_store.items():
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={roc_auc:.3f})")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves (out-of-fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Figure_ROC_OOF_all_models.png"), dpi=300)
    plt.close()

    # PR
    plt.figure()
    for name, p in oof_store.items():
        prec, rec, _ = precision_recall_curve(y, p)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{name} (AUPRC={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall curves (out-of-fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Figure_PR_OOF_all_models.png"), dpi=300)
    plt.close()

    # Calibration
    plt.figure()
    for name, p in oof_store.items():
        pt, pp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.plot(pp, pt, marker="o", label=name)
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("Predicted risk"); plt.ylabel("Observed risk")
    plt.title("Calibration plots (out-of-fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Figure_Calibration_OOF_all_models.png"), dpi=300)
    plt.close()


def youden_table(y, oof_store, outdir):
    rows = []
    for name, p in oof_store.items():
        fpr, tpr, thr = roc_curve(y, p)
        youden = tpr - fpr
        t_star = thr[np.argmax(youden)]

        y_hat = (p >= t_star).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()

        rows.append({
            "Model": name,
            "Threshold(Youden)": float(t_star),
            "AUROC": auc(fpr, tpr),
            "Sensitivity": tp/(tp+fn) if (tp+fn)>0 else np.nan,
            "Specificity": tn/(tn+fp) if (tn+fp)>0 else np.nan,
            "PPV": tp/(tp+fp) if (tp+fp)>0 else np.nan,
            "NPV": tn/(tn+fn) if (tn+fn)>0 else np.nan,
            "Accuracy": accuracy_score(y, y_hat),
            "F1": f1_score(y, y_hat),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
        })
    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    df.to_csv(os.path.join(outdir, "Table_Youden_metrics_all_models.csv"),
              index=False, encoding="utf-8-sig")
    return df


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to local data file (e.g., .xlsx). Not included in repo.")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)

    # IMPORTANT: set these to match your *final* outcome definition in the manuscript
    parser.add_argument("--outcome_col", default="EA", help="Binary outcome column name (behavior-defined EA)")
    parser.add_argument("--patient_id_col", default="patient_id", help="Patient identifier column name")

    args = parser.parse_args()
    outdir = ensure_outdir(args.outdir)

    df = pd.read_excel(args.data_path)

    # --- Example: define patient_id and outcome ---
    # You should modify these lines to match your dataset columns.
    if args.patient_id_col not in df.columns:
        raise ValueError(f"Missing patient_id column: {args.patient_id_col}")
    if args.outcome_col not in df.columns:
        raise ValueError(f"Missing outcome column: {args.outcome_col}")

    groups = df[args.patient_id_col].astype(str).values
    y = df[args.outcome_col].astype(int).values

    # --- Define features (edit to match your final feature set) ---
    my_features = [
        "age","BMI",
        "sexM1F2","ASAclass","URIsx",
        "rocuron_dose_kg","fentanly_total_dose_intraop","ane_time","op_time",
        "tube1LMA2","ane_drug_K1M2KM3pofol45",
    ]
    features = [c for c in my_features if c in df.columns]
    X = df[features].copy()

    # Convert numeric-like columns
    for c in X.columns:
        X[c] = to_minutes_series(X[c])

    # ---- Build models here (you already have models dict in notebook) ----
    # Replace this section with your finalized pipelines/models.
    # Example placeholders:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    cat_cols = [c for c in ["sexM1F2","ASAclass","URIsx","tube1LMA2","ane_drug_K1M2KM3pofol45"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    models = {
        "Logistic": Pipeline([("prep", preprocess),
                              ("model", LogisticRegression(max_iter=2000, random_state=args.seed))]),
        "RandomForest": Pipeline([("prep", preprocess),
                                  ("model", RandomForestClassifier(n_estimators=500, random_state=args.seed))]),
        "XGBoost": Pipeline([("prep", preprocess),
                             ("model", XGBClassifier(
                                 n_estimators=500, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 reg_lambda=1.0, eval_metric="logloss",
                                 random_state=args.seed, n_jobs=-1
                             ))]),
        "CatBoost": Pipeline([("prep", preprocess),
                              ("model", CatBoostClassifier(
                                  iterations=500, depth=6, learning_rate=0.05,
                                  loss_function="Logloss", verbose=False, random_seed=args.seed
                              ))]),
    }

    # ---- OOF evaluation ----
    oof_store = {}
    perf_rows = []
    for name, model in models.items():
        oof, fold_df = get_oof_and_fold_scores(model, X, y, groups,
                                              n_splits=args.n_splits, seed=args.seed)
        oof_store[name] = oof
        fold_df.to_csv(os.path.join(outdir, f"FoldScores_{name}.csv"), index=False, encoding="utf-8-sig")

        summ = summarize_oof(y, oof, fold_df)
        perf_rows.append({
            "Model": name,
            "N": summ["N"],
            "Prevalence": summ["Prevalence"],
            "AUROC (OOF)": summ["AUROC_oof"],
            "AUPRC (OOF)": summ["AUPRC_oof"],
            "Brier (OOF)": summ["Brier_oof"],
            "AUROC (mean±SD)": f'{summ["AUROC_mean"]:.3f} ± {summ["AUROC_sd"]:.3f}',
            "AUPRC (mean±SD)": f'{summ["AUPRC_mean"]:.3f} ± {summ["AUPRC_sd"]:.3f}',
            "Brier (mean±SD)": f'{summ["Brier_mean"]:.3f} ± {summ["Brier_sd"]:.3f}',
        })

    perf = pd.DataFrame(perf_rows).sort_values("AUROC (OOF)", ascending=False)
    perf.to_csv(os.path.join(outdir, "Table_OOF_performance.csv"), index=False, encoding="utf-8-sig")

    # ---- Plots + Youden ----
    plot_roc_pr_calibration(y, oof_store, outdir)
    youden_df = youden_table(y, oof_store, outdir)

    print("Saved outputs to:", outdir)
    print(perf)
    print(youden_df)


if __name__ == "__main__":
    main()
