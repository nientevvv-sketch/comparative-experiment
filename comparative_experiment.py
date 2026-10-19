"""
Comparative ML experiment: Breast Cancer Wisconsin
=================================================

Run from the repository root:
    python comparative_experiment.py

Outputs are written to results/comparative_experiment/.

Methodology:
- One stratified train/test split, shared by all four models.
- Hyperparameter tuning is performed on the training split only.
- Repeated nested cross-validation estimates tuning/generalization stability
  using training data only.
- The held-out test set is used for final descriptive metrics and plots.
- Class 0 is malignant in sklearn's load_breast_cancer dataset.
- n_jobs=1 is intentional for lower memory use on Windows.

Dependencies:
    pip install scikit-learn pandas numpy matplotlib
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ------------------------------ Configuration ------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
INNER_SPLITS = 3
OUTER_SPLITS = 3
OUTER_REPEATS = 1
N_JOBS = 1
RESULTS_DIR = Path("results") / "comparative_experiment"

MALIGNANT_LABEL = 0
MALIGNANT_NAME = "malignant"


def build_models():
    """Return fresh, unfitted estimators."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_param_grids():
    return {
        "Logistic Regression": {
            "logisticregression__C": [0.1, 1.0, 10.0],
            "logisticregression__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 8],
            "min_samples_leaf": [1, 4],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 8],
            "min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    }


def malignant_f1_scorer():
    return make_scorer(f1_score, pos_label=MALIGNANT_LABEL, zero_division=0)


def tune_models(X_train, y_train, models):
    """Tune each model using training data only."""
    param_grids = build_param_grids()
    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scorer = malignant_f1_scorer()

    best_models = {}
    tuning_rows = []
    for name, estimator in models.items():
        print(f"Tuning {name} ...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grids[name],
            scoring=scorer,
            cv=cv,
            n_jobs=N_JOBS,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        best_models[name] = search.best_estimator_
        tuning_rows.append(
            {
                "Model": name,
                "Best CV F1 (malignant)": search.best_score_,
                "Search time (sec)": elapsed,
                "Best parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        print(f"  Best CV F1 (malignant): {search.best_score_:.4f}")
        print(f"  Best parameters: {search.best_params_}")
    return best_models, pd.DataFrame(tuning_rows) 
      
def nested_cv_scores(X_train, y_train, models):
    """
    Repeated nested CV. Each outer validation fold is untouched by that fold's
    inner GridSearchCV. This estimates the performance of the tuning procedure.
    """
    param_grids = build_param_grids()
    scorer = malignant_f1_scorer()
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []
    for name, estimator in models.items():
        print(f"Nested CV: {name} ...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grids[name],
            scoring=scorer,
            cv=inner_cv,
            n_jobs=N_JOBS,
            refit=True,
            error_score="raise",
        )
        result = cross_validate(
            search,
            X_train,
            y_train,
            cv=outer_cv,
            scoring=scorer,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        for fold_index, score in enumerate(result["test_score"], start=1):
            rows.append(
                {
                    "Model": name,
                    "Outer fold": fold_index,
                    "F1 (malignant)": float(score),
                }
            )
        print(
            f"  Nested CV F1: {np.mean(result['test_score']):.4f} "
            f"+/- {np.std(result['test_score'], ddof=1):.4f}"
        )

    return pd.DataFrame(rows)


def evaluate_test_set(X_test, y_test, fitted_models):
    """Evaluate fitted models on the held-out test set."""
    rows = []
    report_rows = []
    predictions = {}
    probability_scores = {}

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # Find the probability column for class 0 explicitly; do not assume
        # a column position if an estimator's class order changes.
        proba = model.predict_proba(X_test)
        class_index = list(model.classes_).index(MALIGNANT_LABEL)
        malignant_proba = proba[:, class_index]
        probability_scores[name] = malignant_proba

        y_malignant = (np.asarray(y_test) == MALIGNANT_LABEL).astype(int)
        auc = roc_auc_score(y_malignant, malignant_proba)

        rows.append(
            {
                "Model": name,
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Test Precision (malignant)": precision_score(
                    y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
                ),
                "Test Recall (malignant)": recall_score(
                    y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
                ),
                "Test F1 (malignant)": f1_score(
                    y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
                ),
                "Test ROC-AUC (malignant)": auc,
            }
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            output_dict=True,
            zero_division=0,
        )
        for label in ["malignant", "benign"]:
            report_rows.append(
                {
                    "Model": name,
                    "Class": label,
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1-score": report[label]["f1-score"],
                    "Support": report[label]["support"],
                }
            )

    return (
        pd.DataFrame(rows),
        predictions,
        probability_scores,
        pd.DataFrame(report_rows),
    )


def save_metric_chart(test_metrics, output_dir):
    metric_cols = [
        "Test Accuracy",
        "Test Precision (malignant)",
        "Test Recall (malignant)",
        "Test F1 (malignant)",
        "Test ROC-AUC (malignant)",
    ]
    plot_df = test_metrics.set_index("Model")[metric_cols]
    ax = plot_df.plot(kind="bar", figsize=(13, 6), ylim=(0, 1.05))
    ax.set_title("Held-out test metrics by model")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "test_metrics_comparison.png", dpi=180)
    plt.close()


def save_roc_chart(y_test, probability_scores, output_dir):
    y_malignant = (np.asarray(y_test) == MALIGNANT_LABEL).astype(int)
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, scores in probability_scores.items():
        fpr, tpr, _ = roc_curve(y_malignant, scores)
        auc = roc_auc_score(y_malignant, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title("ROC curves on held-out test set")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (malignant)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "test_roc_curves.png", dpi=180)
    plt.close(fig)


def save_confusion_matrices(y_test, predictions, output_dir):
    names = list(predictions.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(y_test, predictions[name], labels=[0, 1])
        image = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=["malignant", "benign"])
        ax.set_yticks([0, 1], labels=["malignant", "benign"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )
    fig.suptitle("Confusion matrices — held-out test set", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion_matrices.png", dpi=180)
    plt.close(fig) 
    


def save_speed_quality_chart(test_metrics, tuning_metrics, output_dir):
    """Compare hyperparameter-search time with test-set malignant F1."""

    # Search time is already included in test_metrics in main().
    # Avoid merging the same column twice.
    merged = test_metrics.copy()

    required_columns = [
        "Model",
        "Search time (sec)",
        "Test F1 (malignant)",
    ]

    missing = [col for col in required_columns if col not in merged.columns]
    if missing:
        raise ValueError(
            f"Missing columns for speed-quality chart: {missing}"
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in merged.iterrows():
        ax.scatter(
            row["Search time (sec)"],
            row["Test F1 (malignant)"],
            s=90,
        )
        ax.annotate(
            row["Model"],
            (row["Search time (sec)"], row["Test F1 (malignant)" ]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Hyperparameter-search time vs held-out test F1")
    ax.set_xlabel("Grid-search elapsed time (seconds; hardware-dependent)")
    ax.set_ylabel("Test F1 (malignant)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "search_time_vs_test_f1.png", dpi=180)
    plt.close(fig)




def save_error_chart(y_test, predictions, output_dir):
    """Show false negatives for malignant and false positives for malignant."""
    rows = []
    y_arr = np.asarray(y_test)
    for name, pred in predictions.items():
        cm = confusion_matrix(y_arr, pred, labels=[0, 1])
        # Labels are [malignant, benign]:
        # cm[0, 1] = malignant cases predicted benign (false negatives).
        # cm[1, 0] = benign cases predicted malignant (false positives).
        rows.append(
            {
                "Model": name,
                "Missed malignant cases": int(cm[0, 1]),
                "Benign flagged as malignant": int(cm[1, 0]),
            }
        )
    errors = pd.DataFrame(rows)
    errors.to_csv(output_dir / "test_error_counts.csv", index=False)

    ax = errors.set_index("Model").plot(kind="bar", figsize=(10, 6))
    ax.set_title("Clinically important classification errors on test set")
    ax.set_ylabel("Number of cases")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "test_error_analysis.png", dpi=180)
    plt.close() 
    


def save_stability_chart(nested_scores, output_dir):
    """Plot the distribution of outer-fold scores for each model."""
    model_order = (
        nested_scores.groupby("Model")["F1 (malignant)"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    score_groups = [
        nested_scores.loc[nested_scores["Model"] == name, "F1 (malignant)"]
        for name in model_order
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(score_groups, tick_labels=model_order, showmeans=True)
    ax.set_title("Nested cross-validation F1 stability")
    ax.set_ylabel("Outer-fold F1 (malignant)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "nested_cv_stability.png", dpi=180)
    plt.close(fig)



def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    print("Dataset: Breast Cancer Wisconsin (scikit-learn)")
    print(f"Samples: {len(X)}, features: {X.shape[1]}")
    print("Class mapping:", {0: "malignant", 1: "benign"})
    print("Class counts:", y.value_counts().sort_index().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train samples: {len(X_train)}, test samples: {len(X_test)}")

    # Nested CV uses only the training split. Use fresh unfitted estimators.
    nested_scores = nested_cv_scores(X_train, y_train, build_models())
    nested_scores.to_csv(RESULTS_DIR / "nested_cv_fold_scores.csv", index=False)

    # Final tuning also uses only the training split.
    fitted_models, tuning_metrics = tune_models(X_train, y_train, build_models())
    tuning_metrics.to_csv(RESULTS_DIR / "tuning_summary.csv", index=False)

    test_metrics, predictions, probability_scores, reports = evaluate_test_set(
        X_test, y_test, fitted_models
    )
    test_metrics = test_metrics.merge(
        nested_scores.groupby("Model")["F1 (malignant)"]
        .agg(["mean", "std"]) 
        .reset_index()
        .rename(
            columns={
                "mean": "Nested CV F1 mean (malignant)",
                "std": "Nested CV F1 std (malignant)",
            }
        ),
        on="Model",
        how="left",
    )
    test_metrics = test_metrics.merge(
        tuning_metrics[["Model", "Best CV F1 (malignant)", "Search time (sec)", "Best parameters"]],
        on="Model",
        how="left",
    )
    test_metrics = test_metrics.sort_values(
        "Nested CV F1 mean (malignant)", ascending=False
    )
    test_metrics.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)
    reports.to_csv(RESULTS_DIR / "classification_reports.csv", index=False)

    save_metric_chart(test_metrics, RESULTS_DIR)
    save_stability_chart(nested_scores, RESULTS_DIR)
    save_roc_chart(y_test, probability_scores, RESULTS_DIR)
    save_confusion_matrices(y_test, predictions, RESULTS_DIR)
    save_speed_quality_chart(test_metrics, tuning_metrics, RESULTS_DIR)
    save_error_chart(y_test, predictions, RESULTS_DIR)

    print("\n=== Final comparison (sorted by nested CV F1) ===")
    print(test_metrics.round(4).to_string(index=False))
    print(f"\nAll outputs saved to: {RESULTS_DIR.resolve()}")
    print("Note: search times depend on hardware and system load.")
    print(
        "Interpret test-set results descriptively; avoid repeatedly changing "
        "models based on this test set."
    )


if __name__ == "__main__":
    main()
