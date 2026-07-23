"""
model_utils.py
--------------
Utility functions for model evaluation, cross-validation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from typing import Optional, Dict, Any


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate a classifier on the test set and print a labeled confusion matrix.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Trained classifier with predict() and predict_proba() methods.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        True labels.
    model_name : str
        Display name for the model in printed output.

    Returns
    -------
    dict
        Dictionary with keys: 'Model', 'F1', 'AUC_PR', 'confusion_matrix'.
    """
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, pred)
    auc_pr = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    print(f"\n=== {model_name} ===")
    print(f"  F1:     {f1:.4f}")
    print(f"  AUC-PR: {auc_pr:.4f}")
    plot_confusion_matrix(cm, title=f"{model_name} — Confusion Matrix")

    return {
        "Model": model_name,
        "F1": f1,
        "AUC_PR": auc_pr,
        "confusion_matrix": cm,
    }


def run_stratified_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "average_precision",
) -> tuple:
    """Run StratifiedKFold cross-validation and return mean and std of scores.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Classifier to evaluate. Should be unfitted or will be cloned internally.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    n_splits : int, optional
        Number of CV folds (default 5).
    scoring : str, optional
        Sklearn scoring string (default "average_precision").

    Returns
    -------
    tuple of (float, float)
        (mean_score, std_score) across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=skf)
    return float(scores.mean()), float(scores.std())


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    labels: Optional[list] = None,
) -> None:
    """Plot a labeled seaborn heatmap for a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array (2x2 for binary classification).
    title : str, optional
        Title for the plot.
    save_path : str or None, optional
        If provided, the plot is saved to this file path instead of shown.
    labels : list or None, optional
        Class labels for axes. Defaults to ["Not Fraud", "Fraud"].
    """
    if labels is None:
        labels = ["Not Fraud", "Fraud"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Pred {l}" for l in labels],
        yticklabels=[f"True {l}" for l in labels],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
