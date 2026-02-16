"""Tests for src/data/raw_processing.py — resampling and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.raw_processing import (
    remove_impossible_values,
    resample_to_5min,
    resample_to_interval,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _irregular_sensor_df(n: int = 300) -> pd.DataFrame:
    """Build a small DataFrame with an irregular datetime column (tz-aware)."""
    rng = np.random.default_rng(0)
    # Irregular timestamps: ~4-6 min apart
    base = pd.Timestamp("2021-07-01", tz="Europe/Paris")
    offsets = np.cumsum(rng.integers(4, 7, size=n))
    datetimes = [base + pd.Timedelta(minutes=int(m)) for m in offsets]

    return pd.DataFrame({
        "datetime": datetimes,
        "CO2": 400 + rng.normal(0, 30, n),
        "Temperature": 22 + rng.normal(0, 2, n),
        "Humidity": 55 + rng.normal(0, 5, n),
        "Noise": 40 + rng.normal(0, 3, n),
        "Pressure": 1013 + rng.normal(0, 2, n),
    })


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

class TestResampleToInterval:
    """Tests for the generalized resample_to_interval function."""

    def test_5min_regular_index(self):
        """Output should have a regular 5-min DatetimeIndex."""
        df = _irregular_sensor_df()
        result = resample_to_interval(df, interval_minutes=5)
        diffs = result.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(minutes=5)).all()

    def test_1h_regular_index(self):
        """Output should have a regular 1-hour DatetimeIndex."""
        df = _irregular_sensor_df(500)
        result = resample_to_interval(df, interval_minutes=60)
        diffs = result.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_output_columns_preserved(self):
        """Sensor columns survive resampling (datetime becomes the index)."""
        df = _irregular_sensor_df()
        result = resample_to_interval(df, interval_minutes=5)
        for col in ["CO2", "Temperature", "Humidity", "Noise", "Pressure"]:
            assert col in result.columns

    def test_invalid_interval_raises(self):
        """Non-positive interval should raise ValueError."""
        df = _irregular_sensor_df(50)
        with pytest.raises(ValueError, match="positive"):
            resample_to_interval(df, interval_minutes=0)


class TestResampleTo5min:
    """Test the backward-compatible wrapper."""

    def test_backward_compat(self):
        """resample_to_5min should produce the same as resample_to_interval(5)."""
        df = _irregular_sensor_df(100)
        a = resample_to_5min(df)
        b = resample_to_interval(df, interval_minutes=5)
        pd.testing.assert_frame_equal(a, b)


class TestRemoveImpossibleValues:
    """Tests for sensor range validation."""

    def test_out_of_range_become_nan(self):
        """Values outside valid ranges should become NaN."""
        df = pd.DataFrame({
            "CO2": [400, 200, 600, 6000],  # 200 and 6000 are out of range
            "Noise": [40, 50, 25, 60],     # 25 is below 30
        })
        result = remove_impossible_values(df)
        assert pd.isna(result.loc[1, "CO2"])  # 200 < 300
        assert pd.isna(result.loc[3, "CO2"])  # 6000 > 5000
        assert pd.isna(result.loc[2, "Noise"])  # 25 < 30
        # Valid values preserved
        assert result.loc[0, "CO2"] == 400
        assert result.loc[2, "CO2"] == 600

    def test_no_rows_dropped(self):
        """remove_impossible_values should NOT drop rows, only set NaN."""
        df = pd.DataFrame({
            "CO2": [100, 400, 600],
        })
        result = remove_impossible_values(df)
        assert len(result) == len(df)
