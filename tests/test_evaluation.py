"""Tests for evaluation metrics and comparison utilities."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics
from src.evaluation.comparison import compare_models, compare_experiments, parse_model_name


# ---------------------------------------------------------------------------
#  compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for metric calculation."""

    def test_perfect_prediction(self):
        """Perfect prediction should give MSE=0, R2=1."""
        y = np.array([400, 500, 600, 700, 800], dtype=float)
        metrics = compute_metrics(y, y)
        assert metrics["mse"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_constant_offset(self):
        """Constant offset should give MAE = offset."""
        y_true = np.array([400, 500, 600, 700, 800], dtype=float)
        y_pred = y_true + 10.0
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(10.0)
        assert metrics["rmse"] == pytest.approx(10.0)

    def test_all_keys_present(self):
        """All 5 standard metrics should be in the output."""
        y = np.random.default_rng(42).normal(500, 50, 100)
        metrics = compute_metrics(y, y + np.random.default_rng(0).normal(0, 5, 100))
        for key in ["mse", "rmse", "mae", "mape", "r2"]:
            assert key in metrics

    def test_mape_with_near_zero(self):
        """MAPE should handle near-zero values by masking them."""
        y_true = np.array([0.5, 0.0, 400, 500], dtype=float)
        y_pred = np.array([0.6, 0.1, 410, 510], dtype=float)
        metrics = compute_metrics(y_true, y_pred)
        # Should not raise and should not be inf
        assert np.isfinite(metrics["mape"]) or np.isnan(metrics["mape"])


# ---------------------------------------------------------------------------
#  parse_model_name
# ---------------------------------------------------------------------------

class TestParseModelName:
    """Tests for experiment tag parsing from model names."""

    def test_standard_name(self):
        """'exp1_LSTM_h1' should parse correctly."""
        exp, model, horizon = parse_model_name("exp1_LSTM_h1")
        assert exp == "exp1"
        assert model == "LSTM"
        assert horizon == "h1"

    def test_cnn_lstm_name(self):
        """'exp2_CNN-LSTM_h24' should keep hyphen in model name."""
        exp, model, horizon = parse_model_name("exp2_CNN-LSTM_h24")
        assert exp == "exp2"
        assert model == "CNN-LSTM"
        assert horizon == "h24"

    def test_legacy_name(self):
        """'LSTM_h1' (no experiment prefix) should default to 'exp1'."""
        exp, model, horizon = parse_model_name("LSTM_h1")
        assert exp == "exp1"
        assert model == "LSTM"
        assert horizon == "h1"

    def test_exp3_name(self):
        """'exp3_HMM-LSTM_h1' should parse correctly."""
        exp, model, horizon = parse_model_name("exp3_HMM-LSTM_h1")
        assert exp == "exp3"
        assert model == "HMM-LSTM"
        assert horizon == "h1"


# ---------------------------------------------------------------------------
#  compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    """Tests for compare_models comparison table generation."""

    def test_comparison_table_structure(self, tmp_path):
        """compare_models should return a DataFrame with correct structure."""
        results = {
            "exp1_LSTM_h1": {"rmse": 10.0, "mae": 8.0, "mape": 2.0, "r2": 0.95},
            "exp1_CNN-LSTM_h1": {"rmse": 9.0, "mae": 7.0, "mape": 1.8, "r2": 0.96},
        }
        df = compare_models(results, tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert "rmse" in df.columns
        assert len(df) == 2

    def test_csv_saved(self, tmp_path):
        """compare_models should save comparison_table.csv."""
        results = {
            "LSTM": {"rmse": 10.0, "mae": 8.0, "mape": 2.0, "r2": 0.95},
        }
        compare_models(results, tmp_path)
        assert (tmp_path / "comparison_table.csv").exists()

    def test_summary_saved(self, tmp_path):
        """compare_models should save comparison_summary.txt."""
        results = {
            "LSTM": {"rmse": 10.0, "mae": 8.0, "mape": 2.0, "r2": 0.95},
            "CNN-LSTM": {"rmse": 9.0, "mae": 7.0, "mape": 1.8, "r2": 0.96},
        }
        compare_models(results, tmp_path)
        assert (tmp_path / "comparison_summary.txt").exists()


# ---------------------------------------------------------------------------
#  compare_experiments
# ---------------------------------------------------------------------------

class TestCompareExperiments:
    """Tests for cross-experiment comparison."""

    def test_cross_experiment_output(self, tmp_path):
        """compare_experiments should produce a DataFrame and save outputs."""
        all_results = {
            "exp1": {
                "exp1_LSTM_h1": {"rmse": 10.0, "mae": 8.0, "r2": 0.95},
                "exp1_CNN-LSTM_h1": {"rmse": 9.0, "mae": 7.0, "r2": 0.96},
            },
            "exp2": {
                "exp2_LSTM_h1": {"rmse": 11.0, "mae": 9.0, "r2": 0.93},
                "exp2_CNN-LSTM_h1": {"rmse": 10.5, "mae": 8.5, "r2": 0.94},
            },
        }
        df = compare_experiments(all_results, tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert (tmp_path / "cross_experiment_table.csv").exists()
        assert (tmp_path / "cross_experiment_summary.txt").exists()

    def test_empty_results(self, tmp_path):
        """compare_experiments should handle empty input gracefully."""
        df = compare_experiments({}, tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
