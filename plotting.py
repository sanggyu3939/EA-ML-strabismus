# plotting.py
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import calibration_curve


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def plot_roc_curves(y: np.ndarray, oof_store: Dict[str, np.ndarray], outdir: str) -> str:
    outdir = ensure_outdir(outdir)
    plt.figure()
    for name, p in oof_store.items():
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves (out-of-fold)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "Figure_ROC_OOF_all_models.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_pr_curves(y: np.ndarray, oof_store: Dict[str, np.ndarray], outdir: str) -> str:
    outdir = ensure_outdir(outdir)
    plt.figure()
    for name, p in oof_store.items():
        prec, rec, _ = precision_recall_curve(y, p)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{name} (AUPRC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curves (out-of-fold)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "Figure_PR_OOF_all_models.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_calibration(y: np.ndarray, oof_store: Dict[str, np.ndarray], outdir: str, n_bins: int = 10) -> str:
    outdir = ensure_outdir(outdir)
    plt.figure()
    for name, p in oof_store.items():
        pt, pp = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        plt.plot(pp, pt, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed risk")
    plt.title("Calibration plots (out-of-fold)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "Figure_Calibration_OOF_all_models.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def youden_table(y: np.ndarray, oof_store: Dict[str, np.ndarray], outdir: str) -> pd.DataFrame:
    outdir = ensure_outdir(outdir)
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
            "AUROC": float(auc(fpr, tpr)),
            "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "PPV": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            "NPV": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
            "Accuracy": float(accuracy_score(y, y_hat)),
            "F1": float(f1_score(y, y_hat)),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        })

    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(outdir, "Table_Youden_metrics_all_models.csv"), index=False, encoding="utf-8-sig")
    return df


def try_shap_summary_plot(
    fitted_pipeline,
    X,
    outdir: str,
    max_display: int = 20,
    sample_size: int = 1000,
    seed: int = 42,
) -> str | None:
    """
    Optional: create SHAP summary plot for tree models (e.g., XGBoost/CatBoost/RandomForest).
    Requires `shap` installed. If SHAP fails, returns None.
    """
    try:
        import shap  # noqa: F401
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        return None

    outdir = ensure_outdir(outdir)

    try:
        prep = fitted_pipeline.named_steps["prep"]
        model = fitted_pipeline.named_steps["model"]
        X_enc = prep.transform(X)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()

        feat_names = prep.get_feature_names_out()

        rs = np.random.RandomState(seed)
        n = X_enc.shape[0]
        idx = rs.choice(n, size=min(sample_size, n), replace=False)
        X_sub = X_enc[idx]

        # TreeExplainer works for many tree models; use model_output="raw" for log-odds style
        explainer = shap.TreeExplainer(model, model_output="raw")
        sv = explainer.shap_values(X_sub)

        # normalize shape (class-1)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[-1] == 2:
            sv = sv[:, :, 1]

        plt.figure(figsize=(9, 7), dpi=300)
        shap.summary_plot(sv, X_sub, feature_names=feat_names, max_display=max_display, show=False)
        plt.tight_layout()
        path = os.path.join(outdir, "Figure_SHAP_summary.png")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception:
        return None
