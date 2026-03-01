![GitHub Workflow Status](https://github.com/sanggyu3939/EA-ML-strabismus/actions/workflows/python-app.yml/badge.svg)


# Machine Learning–Based Prediction of Emergence Agitation After Pediatric Strabismus Surgery

This repository contains the reproducible analysis pipeline for the study:

> **Machine Learning–Based Prediction of Emergence Agitation After Pediatric Strabismus Surgery: Development and Explainable Validation of Predictive Models**

---

## Overview

We developed and internally validated machine learning models to predict emergence agitation (EA) in pediatric patients undergoing strabismus surgery under sevoflurane anesthesia.

All performance metrics are computed from **pooled out-of-fold predictions** generated through **stratified group 5-fold cross-validation**, with grouping performed at the patient level to prevent data leakage.

---

## Outcome

The analysis expects a binary outcome column named `EA` by default.

- `EA = 1` indicates emergence agitation (as defined in the manuscript).
- `EA = 0` indicates no emergence agitation.

You may override the outcome column name using:

`--outcome_col`

---

## Data Availability and Privacy

- Patient-level data are **not included** in this repository.
- The dataset will not be publicly shared due to privacy regulations and institutional policy.
- The code is structured to allow full reproducibility of model development and internal validation using a locally stored dataset.

---

## Requirements

- Python 3.12 or higher
- Required packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Expected Input Format

The pipeline reads an Excel file (`.xlsx`) that must include:

- A patient identifier column (default: `patient_id`)
- The binary outcome column (default: `EA`)
- Predictor variables specified in `preprocessing.py` (`default_feature_spec()`)

If a candidate predictor column is not present in the dataset, it is automatically excluded.

---

## How to Run

### Example

```bash
python ea_ml_pipeline.py \
  --data_path /path/to/your_local_data.xlsx \
  --outdir results \
  --outcome_col EA \
  --patient_id_col patient_id \
  --n_splits 5 \
  --seed 42
```

### Optional: SHAP Summary Plot

```bash
python ea_ml_pipeline.py \
  --data_path /path/to/your_local_data.xlsx \
  --do_shap
```

---

## Outputs

Saved to `--outdir` (default: `results/`):

- `Table_OOF_performance.csv`
- `FoldScores_<MODEL>.csv`
- `Table_Youden_metrics_all_models.csv`
- `Figure_ROC_OOF_all_models.png`
- `Figure_PR_OOF_all_models.png`
- `Figure_Calibration_OOF_all_models.png`
- `Figure_SHAP_summary.png` (optional)

---

## Reproducibility Statement

Random seeds are fixed across all models to enhance reproducibility.  
All reported performance metrics are derived exclusively from pooled out-of-fold predictions to prevent optimistic bias.

---

## Notes

This repository is provided for research transparency and reproducibility purposes only.  
It is **not intended for clinical deployment or real-time decision support**.

---

## Citation

If you use this code, please cite the associated manuscript (to be updated upon publication).
