"""
test_model_utils.py
--------------------
Unit tests for src/model_utils.py.
All tests use synthetic data and mock classifiers — no real data files required.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from src.model_utils import (
    evaluate_model,
    run_stratified_cv,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def _make_mock_classifier(self, n_samples=100, fraud_rate=0.2):
        """Create a tiny synthetic dataset and a fitted LogisticRegression."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, 5),
            columns=[f"f{i}" for i in range(5)]
        )
        y = pd.Series(
            np.random.choice([0, 1], size=n_samples, p=[1 - fraud_rate, fraud_rate])
        )
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X, y)
        return clf, X, y

    def test_evaluate_model_returns_dict(self):
        """evaluate_model returns a dict with F1 and AUC_PR keys."""
        clf, X, y = self._make_mock_classifier()

        with patch("matplotlib.pyplot.show"):  # suppress display in CI
            result = evaluate_model(clf, X, y, "TestModel")

        assert isinstance(result, dict), "Result should be a dict"
        assert "F1" in result, "Dict missing 'F1' key"
        assert "AUC_PR" in result, "Dict missing 'AUC_PR' key"

    def test_evaluate_model_f1_range(self):
        """F1 score is within [0, 1]."""
        clf, X, y = self._make_mock_classifier()

        with patch("matplotlib.pyplot.show"):
            result = evaluate_model(clf, X, y, "TestModel")

        assert 0.0 <= result["F1"] <= 1.0

    def test_evaluate_model_auc_pr_range(self):
        """AUC-PR is within [0, 1]."""
        clf, X, y = self._make_mock_classifier()

        with patch("matplotlib.pyplot.show"):
            result = evaluate_model(clf, X, y, "TestModel")

        assert 0.0 <= result["AUC_PR"] <= 1.0

    def test_evaluate_model_includes_model_name(self):
        """The returned dict includes the model name."""
        clf, X, y = self._make_mock_classifier()

        with patch("matplotlib.pyplot.show"):
            result = evaluate_model(clf, X, y, "MySpecialModel")

        assert result["Model"] == "MySpecialModel"

    def test_evaluate_model_confusion_matrix_shape(self):
        """Confusion matrix in the result dict has shape (2, 2) for binary classification."""
        clf, X, y = self._make_mock_classifier()

        with patch("matplotlib.pyplot.show"):
            result = evaluate_model(clf, X, y, "TestModel")

        assert result["confusion_matrix"].shape == (2, 2)


# ---------------------------------------------------------------------------
# run_stratified_cv
# ---------------------------------------------------------------------------

class TestRunStratifiedCV:
    def _make_cv_dataset(self, n_samples=200, n_features=10, fraud_rate=0.15):
        """Generate a synthetic classification dataset."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=5,
            n_redundant=2,
            weights=[1 - fraud_rate, fraud_rate],
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
        y_series = pd.Series(y)
        return X_df, y_series

    def test_run_stratified_cv_returns_floats(self):
        """run_stratified_cv returns (mean, std) both as floats."""
        X, y = self._make_cv_dataset()
        clf = DummyClassifier(strategy="stratified", random_state=42)

        mean, std = run_stratified_cv(clf, X, y, n_splits=3)

        assert isinstance(mean, float), f"Expected float, got {type(mean)}"
        assert isinstance(std, float), f"Expected float, got {type(std)}"

    def test_run_stratified_cv_mean_range(self):
        """CV mean score is within [0, 1] for average_precision scoring."""
        X, y = self._make_cv_dataset()
        clf = LogisticRegression(max_iter=200, random_state=42)

        mean, std = run_stratified_cv(clf, X, y, n_splits=3)

        assert 0.0 <= mean <= 1.0, f"Mean AUC-PR out of range: {mean}"

    def test_run_stratified_cv_std_non_negative(self):
        """CV std is always non-negative."""
        X, y = self._make_cv_dataset()
        clf = DummyClassifier(strategy="stratified", random_state=42)

        _, std = run_stratified_cv(clf, X, y, n_splits=3)

        assert std >= 0.0, f"Std should be >= 0, got {std}"

    def test_run_stratified_cv_custom_splits(self):
        """Function works with custom n_splits parameter."""
        X, y = self._make_cv_dataset()
        clf = DummyClassifier(strategy="most_frequent")

        mean, std = run_stratified_cv(clf, X, y, n_splits=4)

        assert isinstance(mean, float)
        assert isinstance(std, float)


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:
    def _sample_cm(self):
        return np.array([[80, 5], [10, 15]])

    def test_plot_confusion_matrix_saves_file(self, tmp_path):
        """When save_path is given, a file is created."""
        save_file = str(tmp_path / "cm.png")
        cm = self._sample_cm()

        plot_confusion_matrix(cm, title="Test CM", save_path=save_file)

        import os
        assert os.path.exists(save_file), "Confusion matrix PNG was not saved"

    def test_plot_confusion_matrix_no_error_without_save(self):
        """Function runs without error when save_path is None (just shows)."""
        cm = self._sample_cm()
        with patch("matplotlib.pyplot.show"):
            plot_confusion_matrix(cm, title="Test CM", save_path=None)
