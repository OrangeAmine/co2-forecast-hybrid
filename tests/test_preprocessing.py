"""Tests for src/data/preprocessing.py — scalers, splitting, sequences."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.preprocessing import (
    apply_scalers,
    chronological_split,
    create_sequences,
    fit_scalers,
    inverse_scale_target,
    load_and_parse_data,
)


# ---------------------------------------------------------------------------
#  chronological_split
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    """Tests for train/val/test splitting."""

    def test_default_ratios(self, sample_co2_df):
        """70/15/15 split should produce correct sizes."""
        train, val, test = chronological_split(sample_co2_df)
        n = len(sample_co2_df)
        assert len(train) == int(n * 0.70)
        assert len(val) == int(n * 0.15)
        assert len(test) == n - int(n * 0.70) - int(n * 0.15)

    def test_ratios_sum_check(self, sample_co2_df):
        """Ratios that don't sum to 1 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1"):
            chronological_split(sample_co2_df, 0.5, 0.2, 0.1)

    def test_no_overlap(self, sample_co2_df):
        """Splits should be non-overlapping and cover all rows."""
        train, val, test = chronological_split(sample_co2_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_co2_df)


# ---------------------------------------------------------------------------
#  fit_scalers
# ---------------------------------------------------------------------------

class TestFitScalers:
    """Tests for scaler fitting."""

    def test_standard_scaler(self, sample_co2_df, base_feature_columns):
        """StandardScaler should be returned for scaler_type='standard'."""
        feat_scaler, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "standard"
        )
        assert isinstance(feat_scaler, StandardScaler)
        assert isinstance(tgt_scaler, StandardScaler)

    def test_minmax_scaler(self, sample_co2_df, base_feature_columns):
        """MinMaxScaler should be returned for scaler_type='minmax'."""
        feat_scaler, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "minmax"
        )
        assert isinstance(feat_scaler, MinMaxScaler)
        assert isinstance(tgt_scaler, MinMaxScaler)

    def test_robust_scaler(self, sample_co2_df, base_feature_columns):
        """RobustScaler should be returned for scaler_type='robust'."""
        feat_scaler, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "robust"
        )
        assert isinstance(feat_scaler, RobustScaler)
        assert isinstance(tgt_scaler, RobustScaler)

    def test_unknown_scaler_raises(self, sample_co2_df, base_feature_columns):
        """Unknown scaler type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scaler_type"):
            fit_scalers(sample_co2_df, base_feature_columns, "CO2", "unknown")


# ---------------------------------------------------------------------------
#  apply_scalers
# ---------------------------------------------------------------------------

class TestApplyScalers:
    """Tests for scaler application."""

    def test_output_shape(self, sample_co2_df, base_feature_columns):
        """Scaled array should be (n_rows, n_features + 1)."""
        feat_scaler, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "standard"
        )
        scaled = apply_scalers(
            sample_co2_df, feat_scaler, tgt_scaler,
            base_feature_columns, "CO2",
        )
        assert scaled.shape == (len(sample_co2_df), len(base_feature_columns) + 1)


# ---------------------------------------------------------------------------
#  inverse_scale_target
# ---------------------------------------------------------------------------

class TestInverseScaleTarget:
    """Tests for inverse scaling round-trip."""

    def test_roundtrip_1d(self, sample_co2_df, base_feature_columns):
        """Scale then inverse should recover original values (1-D)."""
        _, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "standard"
        )
        original = sample_co2_df["CO2"].values
        scaled = tgt_scaler.transform(original.reshape(-1, 1)).ravel()
        recovered = inverse_scale_target(scaled, tgt_scaler)
        np.testing.assert_allclose(recovered.ravel(), original, rtol=1e-5)

    def test_roundtrip_2d(self, sample_co2_df, base_feature_columns):
        """Scale then inverse should work for (n, horizon) shape."""
        _, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "standard"
        )
        original = sample_co2_df["CO2"].values[:50].reshape(10, 5)
        scaled = np.zeros_like(original)
        for i in range(original.shape[1]):
            scaled[:, i] = tgt_scaler.transform(
                original[:, i].reshape(-1, 1)
            ).ravel()
        recovered = inverse_scale_target(scaled, tgt_scaler)
        np.testing.assert_allclose(recovered, original, rtol=1e-5)

    def test_roundtrip_robust(self, sample_co2_df, base_feature_columns):
        """Round-trip should also work with RobustScaler."""
        _, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "robust"
        )
        original = sample_co2_df["CO2"].values
        scaled = tgt_scaler.transform(original.reshape(-1, 1)).ravel()
        recovered = inverse_scale_target(scaled, tgt_scaler)
        np.testing.assert_allclose(recovered.ravel(), original, rtol=1e-5)


# ---------------------------------------------------------------------------
#  create_sequences
# ---------------------------------------------------------------------------

class TestCreateSequences:
    """Tests for sliding window sequence creation."""

    def test_shapes_5min(self, sample_co2_df, base_feature_columns):
        """X and y should have correct shapes for 5-min config."""
        feat_scaler, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "standard"
        )
        scaled = apply_scalers(
            sample_co2_df, feat_scaler, tgt_scaler,
            base_feature_columns, "CO2",
        )
        lookback, horizon = 24, 12
        X, y = create_sequences(scaled, lookback, horizon)
        n_seqs = len(scaled) - lookback - horizon + 1
        assert X.shape == (n_seqs, lookback, scaled.shape[1])
        assert y.shape == (n_seqs, horizon)

    def test_shapes_1h(self, sample_co2_df_1h, base_feature_columns):
        """Sequence creation should work with small lookback/horizon for 1h data."""
        feat_scaler, tgt_scaler = fit_scalers(
            sample_co2_df_1h, base_feature_columns, "CO2", "standard"
        )
        scaled = apply_scalers(
            sample_co2_df_1h, feat_scaler, tgt_scaler,
            base_feature_columns, "CO2",
        )
        lookback, horizon = 6, 1
        X, y = create_sequences(scaled, lookback, horizon)
        n_seqs = len(scaled) - lookback - horizon + 1
        assert X.shape == (n_seqs, lookback, scaled.shape[1])
        assert y.shape == (n_seqs, horizon)

    def test_stride(self, sample_co2_df, base_feature_columns):
        """Stride > 1 should reduce the number of sequences."""
        feat_scaler, tgt_scaler = fit_scalers(
            sample_co2_df, base_feature_columns, "CO2", "standard"
        )
        scaled = apply_scalers(
            sample_co2_df, feat_scaler, tgt_scaler,
            base_feature_columns, "CO2",
        )
        lookback, horizon = 24, 12
        X1, _ = create_sequences(scaled, lookback, horizon, stride=1)
        X2, _ = create_sequences(scaled, lookback, horizon, stride=3)
        assert len(X2) < len(X1)


# ---------------------------------------------------------------------------
#  load_and_parse_data
# ---------------------------------------------------------------------------

class TestLoadAndParseData:
    """Tests for CSV loading."""

    def test_loads_csv(self, tmp_csv):
        """Should load and parse the datetime column."""
        df = load_and_parse_data(tmp_csv, datetime_column="datetime")
        assert "datetime" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
        assert len(df) > 0
