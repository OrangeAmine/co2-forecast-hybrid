"""Shared fixtures for the test suite.

Provides small synthetic DataFrames and configuration dicts that mirror
the project's real data schema, enabling fast, repeatable tests without
requiring actual sensor data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
#  Sample DataFrames
# ---------------------------------------------------------------------------

def _make_co2_df(n_rows: int, freq: str) -> pd.DataFrame:
    """Build a synthetic CO2 DataFrame at the requested frequency.

    Columns: datetime, CO2, Noise, Pression, TemperatureExt, Hrext,
             Day_sin, Day_cos, Year_sin, Year_cos, dCO2.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-06-01", periods=n_rows, freq=freq)

    # Realistic-ish sensor values
    co2 = 400 + 200 * np.sin(2 * np.pi * np.arange(n_rows) / (n_rows / 3)) + rng.normal(0, 10, n_rows)
    noise = 35 + rng.normal(0, 2, n_rows)
    pression = 1013 + rng.normal(0, 3, n_rows)
    temp_ext = 20 + 5 * np.sin(2 * np.pi * np.arange(n_rows) / n_rows) + rng.normal(0, 1, n_rows)
    hr_ext = 50 + 10 * rng.random(n_rows)

    hour = dates.hour + dates.minute / 60.0  # type: ignore[operator]
    day_of_year = dates.dayofyear + hour / 24.0  # type: ignore[operator]

    df = pd.DataFrame({
        "datetime": dates,
        "CO2": co2,
        "Noise": noise,
        "Pression": pression,
        "TemperatureExt": temp_ext,
        "Hrext": hr_ext,
        "Day_sin": np.sin(2 * np.pi * hour / 24.0),
        "Day_cos": np.cos(2 * np.pi * hour / 24.0),
        "Year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "Year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "dCO2": np.gradient(co2),
    })
    return df


@pytest.fixture
def sample_co2_df() -> pd.DataFrame:
    """500-row synthetic DataFrame at 5-min intervals."""
    return _make_co2_df(500, "5min")


@pytest.fixture
def sample_co2_df_1h() -> pd.DataFrame:
    """100-row synthetic DataFrame at 1-hour intervals."""
    return _make_co2_df(100, "1h")


# ---------------------------------------------------------------------------
#  Sample configurations
# ---------------------------------------------------------------------------

@pytest.fixture
def base_feature_columns() -> list[str]:
    """Baseline 9 feature columns."""
    return [
        "Noise", "Pression", "TemperatureExt", "Hrext",
        "Day_sin", "Day_cos", "Year_cos", "Year_sin", "dCO2",
    ]


@pytest.fixture
def enhanced_feature_columns() -> list[str]:
    """Enhanced 18 feature columns (exp3)."""
    return [
        "Noise", "Pression", "TemperatureExt", "Hrext",
        "Day_sin", "Day_cos", "Year_cos", "Year_sin", "dCO2",
        "Weekday_sin", "Weekday_cos",
        "CO2_lag_12", "CO2_lag_72", "CO2_lag_288",
        "CO2_rolling_mean_12", "CO2_rolling_std_12",
        "CO2_rolling_mean_72", "CO2_rolling_std_72",
    ]


@pytest.fixture
def sample_config(base_feature_columns) -> dict:
    """Minimal merged config dict matching the project schema."""
    return {
        "data": {
            "processed_csv": "data/processed/BM2021_22_5min.csv",
            "target_column": "CO2",
            "feature_columns": base_feature_columns,
            "datetime_column": "datetime",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "scaler_type": "standard",
            "lookback_hours": 2,       # small for tests (24 steps at 5-min)
            "forecast_horizon_hours": 1,  # 12 steps at 5-min
            "stride": 1,
            "samples_per_hour": 12,
        },
        "model": {
            "name": "LSTM",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
        },
        "training": {
            "seed": 42,
            "accelerator": "auto",
            "devices": 1,
            "precision": 32,
            "num_workers": 0,
            "pin_memory": False,
            "results_dir": "results",
            "log_every_n_steps": 1,
            "enable_progress_bar": False,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 8,
            "max_epochs": 2,
            "patience": 5,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
            "gradient_clip_val": 1.0,
        },
    }


@pytest.fixture
def sample_config_1h(sample_config) -> dict:
    """Config for 1-hour sampling experiments."""
    cfg = sample_config.copy()
    cfg["data"] = sample_config["data"].copy()
    cfg["data"]["samples_per_hour"] = 1
    cfg["data"]["lookback_hours"] = 6   # 6 steps at 1h
    cfg["data"]["forecast_horizon_hours"] = 1  # 1 step at 1h
    cfg["data"]["scaler_type"] = "standard"
    return cfg


# ---------------------------------------------------------------------------
#  Temp CSV fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv(tmp_path, sample_co2_df) -> Path:
    """Write sample_co2_df to a temp CSV and return its path."""
    csv_path = tmp_path / "test_data.csv"
    sample_co2_df.to_csv(csv_path, index=False)
    return csv_path
