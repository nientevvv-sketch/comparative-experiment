# comparative-experiment

Comparative ML experiment for breast cancer classification.

This repository contains a small benchmarking experiment that compares several
scikit-learn classifiers on the Breast Cancer Wisconsin dataset. The workflow
is intentionally simple and reproducible so you can reproduce the figures and
CSV summaries in the `results/` directory.

Contents
- comparative_experiment.py — experiment script (run from the repository root).
- results/ — outputs produced by running the script (CSV summaries, PNG/SVG
  plots). The directory is tracked in the repository and contains example
  outputs in results/.

Quick start

1. Create a virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

2. Install dependencies:

   pip install scikit-learn pandas numpy matplotlib

3. Run the experiment:

   python comparative_experiment.py

Outputs

When you run the script the following files will be written under `results/`:
- final_comparison.csv
- tuning_summary.csv
- nested_cv_fold_scores.csv
- classification_reports.csv
- test_error_counts.csv
- test_metrics_comparison.png (or .svg)
- nested_cv_stability.png (or .svg)
- test_roc_curves.png (or .svg)
- test_confusion_matrices.png (or .svg)
- search_time_vs_test_f1.png (or .svg)

(There is a placeholder `results/README.md` so the empty results folder is
tracked in the repository.)

Notes

- The experiment uses a single stratified train/test split shared by all
  models. Hyperparameter tuning is performed on the training split only.
- `n_jobs=1` is used intentionally to reduce memory pressure on some systems.
- Interpret the held-out test metrics descriptively; avoid repeatedly tuning
  models to chase the same test set.

License

MIT
