# comparative-experiment

Comparative ML experiment for breast cancer classification.

This repository contains a reproducible benchmark comparing multiple
scikit-learn classifiers on the Breast Cancer Wisconsin dataset. The project
uses a shared stratified train/test split, hyperparameter tuning on the
training set only, and nested cross-validation to assess tuning stability.

## Contents

- `comparative_experiment.py` — experiment script, run from the repository root.
- `results/` — generated CSV summaries and charts from the experiment.

## Quick start

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install scikit-learn pandas numpy matplotlib
   ```

3. Run the experiment:

   ```bash
   python comparative_experiment.py
   ```

## Results summary

Logistic Regression achieved the highest held-out malignant F1 score
(approximately 0.964) while also having the shortest hyperparameter-search
runtime. Random Forest and Gradient Boosting were competitive on test F1 but
required substantially longer tuning times. The full experiment outputs are
stored in `results/`.

### Key metrics (held-out test set)

| Model | Test F1 (malignant) | Test ROC-AUC (malignant) | Search time (s) |
|---|---:|---:|---:|
| Logistic Regression | 0.964 | 0.996 | 0.40 |
| Random Forest | 0.940 | 0.994 | 8.17 |
| Gradient Boosting | 0.927 | 0.991 | 8.80 |
| Decision Tree | 0.897 | 0.916 | 0.46 |

### Files in `results/`

- `final_comparison.csv` — final comparison table with test metrics and nested-CV summary.
- `tuning_summary.csv` — best CV F1 and hyperparameter-search time for each model.
- `nested_cv_fold_scores.csv` — outer-fold F1 scores used to assess stability.
- `classification_reports.csv` — per-class precision, recall, F1-score, and support.
- `test_error_counts.csv` — clinically important error counts (missed malignant cases and benign cases flagged as malignant).
- `*.png` / `*.svg` plots — ROC curves, confusion matrices, nested CV stability, and search-time vs test F1 charts.

### Interpretation and limitations

False negatives are clinically important in this task: malignant lesions should
not be classified as benign. In this experiment, Logistic Regression provided a
strong overall trade-off between predictive quality and efficiency, while the
more complex models required much longer tuning. Results are descriptive and
should be interpreted with caution because they are based on a single dataset and
single held-out test split. Search times are hardware-dependent and should be
interpreted comparatively rather than absolutely.

## Notes

- The experiment uses a single stratified train/test split shared by all
  models.
- Hyperparameter tuning is performed on the training split only.
- `n_jobs=1` is intentional to reduce memory pressure on some systems.
- The held-out test set is used for final descriptive reporting; it should not be
  used repeatedly to tune a model.

## License

MIT
