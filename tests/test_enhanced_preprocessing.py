"""Tests for pipeline.py enhanced preprocessing — denoising, lags, rolling, weekday, outliers."""

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import (
    _clip_outliers,
    _compute_outlier_bounds,
    _denoise_co2,
    _interpolate_gaps,
    _log_nan_impact,
)
from src.data.preprocessing import compute_dco2


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _base_df(n: int = 200, freq: str = "5min") -> pd.DataFrame:
    """Build a DataFrame matching post-raw_processing output (with datetime column)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-06-01", periods=n, freq=freq, tz="Europe/Paris")
    co2 = 400 + 100 * np.sin(2 * np.pi * np.arange(n) / 50) + rng.normal(0, 15, n)
    df = pd.DataFrame({
        "datetime": dates.tz_localize(None),
        "CO2": co2,
        "TemperatureExt": 20 + rng.normal(0, 2, n),
        "Hrext": 50 + rng.normal(0, 5, n),
        "Noise": 40 + rng.normal(0, 3, n),
        "Pression": 1013 + rng.normal(0, 2, n),
    })
    return df


# ---------------------------------------------------------------------------
#  Tests — _denoise_co2
# ---------------------------------------------------------------------------

class TestDenoiseCO2:
    """Tests for Savitzky-Golay denoising."""

    def test_shape_preserved(self):
        """Output should have the same number of rows."""
        df = _base_df(200)
        result = _denoise_co2(df, window_length=11, polyorder=3)
        assert len(result) == 200

    def test_co2_raw_column_created(self):
        """Original CO2 should be saved as CO2_raw."""
        df = _base_df(200)
        original_co2 = df["CO2"].values.copy()
        result = _denoise_co2(df, window_length=11, polyorder=3)
        assert "CO2_raw" in result.columns
        np.testing.assert_array_equal(result["CO2_raw"].values, original_co2)

    def test_smoothing_reduces_noise(self):
        """Denoised signal should have less variation than the raw signal."""
        df = _base_df(500)
        raw_std = df["CO2"].diff().std()
        result = _denoise_co2(df, window_length=11, polyorder=3)
        smooth_std = result["CO2"].diff().std()
        assert smooth_std < raw_std, "Denoised signal should be smoother"

    def test_no_nan_introduced(self):
        """Denoising should not introduce NaN values."""
        df = _base_df(200)
        result = _denoise_co2(df, window_length=11, polyorder=3)
        assert result["CO2"].isna().sum() == 0

    def test_even_window_adjusted(self):
        """Even window_length should be auto-adjusted to odd."""
        df = _base_df(100)
        result = _denoise_co2(df, window_length=10, polyorder=3)
        assert len(result) == 100

    def test_skip_if_too_few_rows(self):
        """Should skip denoising gracefully if fewer rows than window."""
        df = _base_df(5)
        result = _denoise_co2(df, window_length=11, polyorder=3)
        assert "CO2_raw" not in result.columns


# ---------------------------------------------------------------------------
#  Tests — _interpolate_gaps
# ---------------------------------------------------------------------------

class TestInterpolateGaps:
    """Tests for NaN gap interpolation."""

    def test_short_gaps_filled(self):
        """Gaps <= max_gap_minutes should be interpolated."""
        df = _base_df(100)
        # Create a short gap (2 consecutive NaN = 10min at 5-min)
        df.loc[10:11, "CO2"] = np.nan
        result = _interpolate_gaps(df, max_gap_minutes=60, interval_minutes=5)
        assert result["CO2"].iloc[10:12].notna().all()

    def test_long_gaps_remain_nan(self):
        """Gaps > max_gap_minutes should remain NaN."""
        df = _base_df(100)
        # Create a long gap (20 consecutive NaN = 100min at 5-min)
        df.loc[10:29, "CO2"] = np.nan
        result = _interpolate_gaps(df, max_gap_minutes=60, interval_minutes=5)
        # Some should still be NaN (beyond the limit=12 consecutive)
        assert result["CO2"].iloc[10:30].isna().any()


# ---------------------------------------------------------------------------
#  Tests — compute_dco2
# ---------------------------------------------------------------------------

class TestComputeDCO2:
    """Tests for CO2 rate of change computation."""

    def test_first_row_nan(self):
        """First row should be NaN (no previous value)."""
        df = _base_df(50)
        result = compute_dco2(df, interval_minutes=5)
        assert pd.isna(result["dCO2"].iloc[0])

    def test_units_ppm_per_hour(self):
        """dCO2 should be in ppm/hour units."""
        df = pd.DataFrame({"CO2": [400.0, 410.0, 420.0]})
        result = compute_dco2(df, interval_minutes=5)
        # 10 ppm change over 5 min = 10 / (5/60) = 120 ppm/h
        assert abs(result["dCO2"].iloc[1] - 120.0) < 1e-6


# ---------------------------------------------------------------------------
#  Tests — outlier detection
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    """Tests for IQR-based outlier bounds and clipping."""

    def test_bounds_computed_correctly(self):
        """Bounds should be Q1 - 3*IQR and Q3 + 3*IQR."""
        df = _base_df(1000)
        bounds = _compute_outlier_bounds(df, ["CO2"], multiplier=3.0)
        assert "CO2" in bounds
        lower, upper = bounds["CO2"]
        q1 = df["CO2"].quantile(0.25)
        q3 = df["CO2"].quantile(0.75)
        iqr = q3 - q1
        assert abs(lower - (q1 - 3 * iqr)) < 1e-6
        assert abs(upper - (q3 + 3 * iqr)) < 1e-6

    def test_clipping_works(self):
        """Values outside bounds should be clipped."""
        df = pd.DataFrame({"CO2": [100, 200, 300, 400, 10000]})
        bounds: dict[str, tuple[float, float]] = {"CO2": (150.0, 500.0)}
        result = _clip_outliers(df, bounds)
        assert result["CO2"].min() >= 150
        assert result["CO2"].max() <= 500
