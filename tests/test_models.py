"""Tests for model forward passes and output shapes.

Tests LSTM, CNN-LSTM, and HMM-LSTM architectures. TFT is tested
separately because it requires pytorch-forecasting's TimeSeriesDataSet.
"""

import pytest
import torch

from src.models.lstm import LSTMForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.hmm_lstm import HMMLSTMForecaster


# ---------------------------------------------------------------------------
#  LSTM
# ---------------------------------------------------------------------------

class TestLSTMForecaster:
    """Tests for the baseline LSTM model."""

    def _make_config(self, n_features: int = 10, horizon: int = 12) -> dict:
        return {
            "data": {
                "feature_columns": [f"f{i}" for i in range(n_features - 1)],
                "samples_per_hour": 12,
                "forecast_horizon_hours": horizon // 12 if horizon >= 12 else 1,
            },
            "model": {
                "name": "LSTM",
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.1,
                "bidirectional": False,
            },
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
            },
        }

    def test_forward_pass(self):
        """Output shape should be (batch, horizon)."""
        config = self._make_config(n_features=10, horizon=12)
        model = LSTMForecaster(config, input_size=10, output_size=12)
        x = torch.randn(4, 24, 10)  # batch=4, lookback=24, features=10
        out = model(x)
        assert out.shape == (4, 12)

    def test_different_input_sizes(self):
        """Should work with both 10 features (exp1) and 19 features (exp3)."""
        for n_feat in [10, 19]:
            config = self._make_config(n_features=n_feat, horizon=12)
            model = LSTMForecaster(config, input_size=n_feat, output_size=12)
            x = torch.randn(2, 24, n_feat)
            out = model(x)
            assert out.shape == (2, 12)

    def test_single_step_horizon(self):
        """Should work with horizon=1 (e.g., 1h at 1h sampling)."""
        config = self._make_config(n_features=10, horizon=1)
        model = LSTMForecaster(config, input_size=10, output_size=1)
        x = torch.randn(4, 6, 10)
        out = model(x)
        assert out.shape == (4, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self):
        """Model should move to GPU correctly."""
        config = self._make_config()
        model = LSTMForecaster(config, input_size=10, output_size=12).cuda()
        x = torch.randn(2, 24, 10).cuda()
        out = model(x)
        assert out.is_cuda


# ---------------------------------------------------------------------------
#  CNN-LSTM
# ---------------------------------------------------------------------------

class TestCNNLSTMForecaster:
    """Tests for the CNN-LSTM hybrid model."""

    def _make_config(self, n_features: int = 10) -> dict:
        return {
            "data": {
                "feature_columns": [f"f{i}" for i in range(n_features - 1)],
                "samples_per_hour": 12,
                "forecast_horizon_hours": 1,
            },
            "model": {
                "name": "CNN-LSTM",
                "cnn_channels": [16, 32],
                "cnn_kernel_sizes": [7, 5],
                "cnn_pool_size": 2,
                "cnn_dropout": 0.1,
                "lstm_hidden_size": 32,
                "lstm_num_layers": 1,
                "lstm_dropout": 0.1,
                "fc_hidden_size": 16,
            },
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
            },
        }

    def test_forward_pass(self):
        """Output shape should be (batch, horizon)."""
        config = self._make_config(n_features=10)
        model = CNNLSTMForecaster(config, input_size=10, output_size=12)
        x = torch.randn(4, 24, 10)
        out = model(x)
        assert out.shape == (4, 12)

    def test_different_input_sizes(self):
        """Should work with both baseline (10) and enhanced (19) features."""
        for n_feat in [10, 19]:
            config = self._make_config(n_features=n_feat)
            model = CNNLSTMForecaster(config, input_size=n_feat, output_size=12)
            x = torch.randn(2, 24, n_feat)
            out = model(x)
            assert out.shape == (2, 12)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self):
        """Model should move to GPU correctly."""
        config = self._make_config()
        model = CNNLSTMForecaster(config, input_size=10, output_size=12).cuda()
        x = torch.randn(2, 24, 10).cuda()
        out = model(x)
        assert out.is_cuda


# ---------------------------------------------------------------------------
#  HMM-LSTM
# ---------------------------------------------------------------------------

class TestHMMLSTMForecaster:
    """Tests for the HMM-augmented LSTM model."""

    def _make_config(self, n_features: int = 10, n_hmm_states: int = 3) -> dict:
        return {
            "data": {
                "feature_columns": [f"f{i}" for i in range(n_features - 1)],
                "samples_per_hour": 12,
                "forecast_horizon_hours": 1,
            },
            "model": {
                "name": "HMM-LSTM",
                "hmm_n_states": n_hmm_states,
                "lstm_hidden_size": 32,
                "lstm_num_layers": 1,
                "lstm_dropout": 0.1,
                "fc_hidden_size": 16,
            },
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
            },
        }

    def test_forward_pass(self):
        """Output shape should be (batch, horizon) with augmented input."""
        n_features = 10
        n_hmm_states = 3
        config = self._make_config(n_features=n_features, n_hmm_states=n_hmm_states)
        total_input = n_features + n_hmm_states
        model = HMMLSTMForecaster(config, input_size=total_input, output_size=12)
        x = torch.randn(4, 24, total_input)
        out = model(x)
        assert out.shape == (4, 12)

    def test_different_input_sizes(self):
        """Should work with exp1 (10+3) and exp3 (19+3) feature sizes."""
        for n_feat in [10, 19]:
            config = self._make_config(n_features=n_feat, n_hmm_states=3)
            total = n_feat + 3
            model = HMMLSTMForecaster(config, input_size=total, output_size=12)
            x = torch.randn(2, 24, total)
            out = model(x)
            assert out.shape == (2, 12)
