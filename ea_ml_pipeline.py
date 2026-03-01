# ea_ml_pipeline.py
from __future__ import annotations

import argparse
import os
import pandas as pd

from preprocessing import (
    build_X_y_groups,
    build_preprocessor,
    default_feature_spec,
)
from modeling import (
    set_global_seed,
    build_models,
    evaluate_all_models,
    pick_best_model,
    fit_pipeline_on_full_data,
)
from plotting import (
    ensure_outdir,
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration,
    youden_table,
    try_shap_summary_plot,
)


def main():
    parser = argparse.ArgumentParser(description="EA ML pipeline (OOF CV, plots, Youden, optional SHAP).")
    parser.add_argument("--data_path", required=True, help="Path to local dataset file (.xlsx). Not included in repo.")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--outcome_col", default="EA", help="Binary outcome column (0/1).")
    parser.add_argument("--patient_id_col", default="patient_id", help="Patient identifier column name.")

    parser.add_argument("--do_shap", action="store_true", help="If set, generate SHAP summary plot for best model (if possible).")
    args = parser.parse_args()

    set_global_seed(args.seed)
    outdir = ensure_outdir(args.outdir)

    # Load
    df = pd.read_excel(args.data_path)

    # Build X/y/groups
    spec = default_feature_spec()
    X, y, groups, used_features, used_cat_cols = build_X_y_groups(
        df=df,
        outcome_col=args.outcome_col,
        patient_id_col=args.patient_id_col,
        feature_spec=spec,
    )

    # Preprocess + models
    preprocess = build_preprocessor(X, cat_cols=used_cat_cols, drop_first=True)
    models = build_models(preprocess, seed=args.seed)

    # Evaluate
    perf_df, oof_store, fold_store = evaluate_all_models(
        models=models,
        X=X,
        y=y,
        groups=groups,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    # Save tables
    perf_path = os.path.join(outdir, "Table_OOF_performance.csv")
    perf_df.to_csv(perf_path, index=False, encoding="utf-8-sig")

    for name, fold_df in fold_store.items():
        fold_df.to_csv(os.path.join(outdir, f"FoldScores_{name}.csv"), index=False, encoding="utf-8-sig")

    # Plots + Youden
    roc_path = plot_roc_curves(y, oof_store, outdir)
    pr_path = plot_pr_curves(y, oof_store, outdir)
    cal_path = plot_calibration(y, oof_store, outdir, n_bins=10)
    youden_df = youden_table(y, oof_store, outdir)

    # Best model + optional SHAP
    best_name = pick_best_model(perf_df)
    best_model = models[best_name]
    best_model_fitted = fit_pipeline_on_full_data(best_model, X, y)

    shap_path = None
    if args.do_shap:
        shap_path = try_shap_summary_plot(best_model_fitted, X, outdir=outdir, max_display=20, sample_size=1000, seed=args.seed)

    # Console summary
    print("\n=== Pipeline summary ===")
    print("Outcome:", args.outcome_col)
    print("Used features:", used_features)
    print("Used categorical columns:", used_cat_cols)
    print("\nSaved:")
    print(" -", perf_path)
    print(" -", os.path.join(outdir, "Table_Youden_metrics_all_models.csv"))
    print(" -", roc_path)
    print(" -", pr_path)
    print(" -", cal_path)
    if shap_path:
        print(" -", shap_path)
    print("\nBest model (by AUROC OOF):", best_name)
    print("\nPerformance table:")
    print(perf_df.to_string(index=False))
    print("\nYouden table:")
    print(youden_df.to_string(index=False))


if __name__ == "__main__":
    main()
