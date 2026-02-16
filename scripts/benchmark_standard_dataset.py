"""Benchmark models on a standard synthetic time series.

Generates a deterministic sinusoidal signal (daily + weekly periodicities
with trend and Gaussian noise) to validate that each model architecture
can learn a structured pattern and produce reasonable forecasts.

The synthetic signal mimics indoor CO2 dynamics at 5-min resolution:
    y(t) = 600 + 50*sin(2*pi*t/288) + 15*sin(2*pi*t/2016) + eps
where:
    288  = samples per day  (24h x 12 samples/h)
    2016 = samples per week (7d x 288)
    eps ~ N(0, 5)

Expected outcomes for a 1-hour (12-step) forecast:
    - RMSE  < 25 ppm   (signal range ~550-700 ppm)
    - MAE   < 20 ppm
    - R2    > 0.50
    - MAPE  < 4%

Thresholds account for multi-step forecasting degradation and the
weekly component (15 ppm amplitude) that is invisible to the 24h
lookback window, plus irreducible Gaussian noise (std=5).

Models benchmarked:
    - LSTM, CNN-LSTM, HMM-LSTM (deep learning)
    - SARIMA (statistical baseline, univariate)
    - XGBoost, CatBoost (gradient boosting)

Usage:
    python scripts/benchmark_standard_dataset.py
    python scripts/benchmark_standard_dataset.py --models lstm cnn_lstm hmm_lstm sarima xgboost catboost
    python scripts/benchmark_standard_dataset.py --epochs 30
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import CO2DataModule
from src.data.preprocessing import inverse_scale_target
from src.evaluation.metrics import compute_metrics
from src.models.lstm import LSTMForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.hmm_lstm import HMMRegimeDetector, HMMLSTMForecaster
from src.models.sarima import SARIMAForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.catboost_model import CatBoostForecaster
from src.training.trainer import create_trainer
from src.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Acceptance thresholds for each metric (1h horizon on synthetic data)
# ---------------------------------------------------------------------------
# Realistic thresholds for multi-step (12-step) forecasting on a
# periodic signal with N(0,5) noise and a weekly component (15 ppm
# amplitude) invisible to the 24h lookback window.
# Irreducible error floor ~ sqrt(noise_var + weekly_component_var)
# = sqrt(25 + ~112) ~ 12 ppm RMSE at minimum.
THRESHOLDS = {
    "rmse": 25.0,   # ppm
    "mae": 20.0,    # ppm
    "r2": 0.50,     # dimensionless
    "mape": 4.0,    # percent
}


def generate_synthetic_dataset(
    n_days: int = 120,
    samples_per_hour: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic CO2-like time series with known structure.

    Signal components:
        - Base level:  600 ppm
        - Daily cycle:  50*sin(2*pi*t / samples_per_day)
        - Weekly cycle: 15*sin(2*pi*t / samples_per_week)
        - Noise:        N(0, 5)

    Additional synthetic features are generated as correlated or
    phase-shifted variants of the target to mimic the real feature set:
        Noise, Pression, TemperatureExt, Hrext, dCO2,
        Day_sin, Day_cos, Year_sin, Year_cos.

    Args:
        n_days: Number of days to generate.
        samples_per_hour: Number of samples per hour.
        seed: Random seed.

    Returns:
        DataFrame with datetime index and all required columns.
    """
    rng = np.random.default_rng(seed)

    samples_per_day = 24 * samples_per_hour   # 288
    samples_per_week = 7 * samples_per_day    # 2016
    n_samples = n_days * samples_per_day

    t = np.arange(n_samples, dtype=np.float64)

    # --- Target: CO2 ---
    daily_cycle = 50.0 * np.sin(2.0 * np.pi * t / samples_per_day)
    weekly_cycle = 15.0 * np.sin(2.0 * np.pi * t / samples_per_week)
    noise = rng.normal(0, 5, size=n_samples)
    co2 = 600.0 + daily_cycle + weekly_cycle + noise

    # --- Synthetic features ---
    # Noise (dB) — loosely correlated with CO2
    noise_db = 40.0 + 5.0 * np.sin(2.0 * np.pi * t / samples_per_day + 0.5) + rng.normal(0, 2, n_samples)

    # Pressure (hPa) — slow variation
    pression = 1013.0 + 5.0 * np.sin(2.0 * np.pi * t / (samples_per_week * 2)) + rng.normal(0, 1, n_samples)

    # External temperature — strong daily cycle
    temp_ext = 15.0 + 8.0 * np.sin(2.0 * np.pi * t / samples_per_day - np.pi / 4) + rng.normal(0, 1, n_samples)

    # External humidity — inverse correlation with temperature
    hr_ext = 60.0 - 10.0 * np.sin(2.0 * np.pi * t / samples_per_day - np.pi / 4) + rng.normal(0, 3, n_samples)

    # Temporal encodings
    day_phase = 2.0 * np.pi * (t % samples_per_day) / samples_per_day
    day_sin = np.sin(day_phase)
    day_cos = np.cos(day_phase)

    year_phase = 2.0 * np.pi * t / (365.25 * samples_per_day)
    year_sin = np.sin(year_phase)
    year_cos = np.cos(year_phase)

    # CO2 derivative (first difference, padded with 0 at start)
    dco2 = np.diff(co2, prepend=co2[0])

    # --- Build DataFrame ---
    start_date = pd.Timestamp("2021-01-01")
    freq = f"{60 // samples_per_hour}min"
    datetimes = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    df = pd.DataFrame({
        "datetime": datetimes,
        "CO2": co2,
        "Noise": noise_db,
        "Pression": pression,
        "TemperatureExt": temp_ext,
        "Hrext": hr_ext,
        "Day_sin": day_sin,
        "Day_cos": day_cos,
        "Year_sin": year_sin,
        "Year_cos": year_cos,
        "dCO2": dco2,
    })

    return df


def build_config(max_epochs: int = 25) -> dict:
    """Build a minimal config dict for benchmark testing.

    Uses GPU with reduced epochs and smaller lookback for faster iteration.
    The config mirrors the project's structure so all model constructors
    and the DataModule work without modification.

    Args:
        max_epochs: Maximum training epochs.

    Returns:
        Configuration dictionary.
    """
    return {
        "data": {
            "processed_csv": "__synthetic__",  # placeholder, not used directly
            "target_column": "CO2",
            "feature_columns": [
                "Noise", "Pression", "TemperatureExt", "Hrext",
                "Day_sin", "Day_cos", "Year_cos", "Year_sin", "dCO2",
            ],
            "datetime_column": "datetime",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "scaler_type": "standard",
            "lookback_hours": 24,      # 288 steps -- full daily cycle
            "forecast_horizon_hours": 1,  # 12 steps
            "stride": 6,              # 30-min stride, balances speed and data volume
            "samples_per_hour": 12,
        },
        "training": {
            "seed": 42,
            "accelerator": "gpu",
            "devices": 1,
            "precision": 32,
            "num_workers": 0,
            "pin_memory": True,
            "results_dir": "results/benchmark",
            "log_every_n_steps": 5,
            "enable_progress_bar": True,
            "gradient_clip_val": 1.0,
            "scheduler": "reduce_on_plateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
            "warmup_epochs": 0,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "max_epochs": max_epochs,
            "patience": 15,
        },
        "model": {},  # filled per model
    }


def prepare_datamodule_from_df(
    df: pd.DataFrame, config: dict
) -> CO2DataModule:
    """Create a CO2DataModule from a synthetic DataFrame.

    Performs chronological split, fits scalers, and creates sequences
    using the project's standard pipeline.

    Args:
        df: Synthetic DataFrame.
        config: Configuration dictionary.

    Returns:
        Configured CO2DataModule ready for training.
    """
    from src.data.preprocessing import (
        chronological_split,
        fit_scalers,
        apply_scalers,
        create_sequences,
    )
    from src.data.dataset import TimeSeriesDataset

    data_cfg = config["data"]
    feature_cols = data_cfg["feature_columns"]
    target_col = data_cfg["target_column"]

    train_df, val_df, test_df = chronological_split(
        df, data_cfg["train_ratio"], data_cfg["val_ratio"], data_cfg["test_ratio"]
    )

    dm = CO2DataModule(config)
    dm.feature_scaler, dm.target_scaler = fit_scalers(
        train_df, feature_cols, target_col, data_cfg["scaler_type"]
    )

    train_scaled = apply_scalers(train_df, dm.feature_scaler, dm.target_scaler, feature_cols, target_col)
    val_scaled = apply_scalers(val_df, dm.feature_scaler, dm.target_scaler, feature_cols, target_col)
    test_scaled = apply_scalers(test_df, dm.feature_scaler, dm.target_scaler, feature_cols, target_col)

    sph = data_cfg["samples_per_hour"]
    lookback = data_cfg["lookback_hours"] * sph
    horizon = data_cfg["forecast_horizon_hours"] * sph
    stride = data_cfg.get("stride", 1)

    X_train, y_train = create_sequences(train_scaled, lookback, horizon, stride)
    X_val, y_val = create_sequences(val_scaled, lookback, horizon, stride)
    X_test, y_test = create_sequences(test_scaled, lookback, horizon, stride)

    dm.train_dataset = TimeSeriesDataset(X_train, y_train)
    dm.val_dataset = TimeSeriesDataset(X_val, y_val)
    dm.test_dataset = TimeSeriesDataset(X_test, y_test)

    # Store split DataFrames for HMM usage
    dm._benchmark_splits = (train_df, val_df, test_df)

    return dm


def benchmark_lstm(config: dict, datamodule: CO2DataModule) -> dict:
    """Train and evaluate the LSTM baseline on synthetic data.

    Args:
        config: Configuration dictionary.
        datamodule: Prepared CO2DataModule.

    Returns:
        Dictionary of test metrics.
    """
    cfg = copy.deepcopy(config)
    cfg["model"] = {
        "name": "LSTM",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": False,
    }

    model = LSTMForecaster(cfg)
    trainer, run_dir = create_trainer(cfg, model_name="bench_LSTM")
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    predictions = trainer.predict(model, datamodule.test_dataloader(), ckpt_path="best")
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()
    y_true_scaled = datamodule.test_dataset.y.numpy()

    y_pred = inverse_scale_target(y_pred_scaled, datamodule.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, datamodule.target_scaler)

    return compute_metrics(y_true, y_pred)


def benchmark_cnn_lstm(config: dict, datamodule: CO2DataModule) -> dict:
    """Train and evaluate the CNN-LSTM model on synthetic data.

    Args:
        config: Configuration dictionary.
        datamodule: Prepared CO2DataModule.

    Returns:
        Dictionary of test metrics.
    """
    cfg = copy.deepcopy(config)
    cfg["model"] = {
        "name": "CNN-LSTM",
        "cnn_channels": [32, 64],
        "cnn_kernel_sizes": [7, 5],
        "cnn_pool_size": 2,
        "cnn_dropout": 0.1,
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.2,
        "fc_hidden_size": 64,
    }

    model = CNNLSTMForecaster(cfg)
    trainer, run_dir = create_trainer(cfg, model_name="bench_CNN-LSTM")
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    predictions = trainer.predict(model, datamodule.test_dataloader(), ckpt_path="best")
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()
    y_true_scaled = datamodule.test_dataset.y.numpy()

    y_pred = inverse_scale_target(y_pred_scaled, datamodule.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, datamodule.target_scaler)

    return compute_metrics(y_true, y_pred)


def benchmark_hmm_lstm(
    config: dict,
    df: pd.DataFrame,
) -> dict:
    """Train and evaluate the HMM-LSTM model on synthetic data.

    The HMM-LSTM requires a separate data pipeline because it augments
    the DataFrame with HMM posterior probabilities before windowing.

    Args:
        config: Configuration dictionary.
        df: Full synthetic DataFrame (before splitting).

    Returns:
        Dictionary of test metrics.
    """
    from src.data.preprocessing import (
        chronological_split,
        fit_scalers,
        apply_scalers,
        create_sequences,
    )
    from src.data.dataset import TimeSeriesDataset

    cfg = copy.deepcopy(config)
    cfg["model"] = {
        "name": "HMM-LSTM",
        "hmm_n_states": 3,
        "hmm_covariance_type": "full",
        "hmm_n_iter": 50,
        "hmm_features": ["CO2", "Noise", "TemperatureExt"],
        "hmm_append_mode": "probabilities",
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.2,
        "fc_hidden_size": 64,
    }

    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_columns"]
    target_col = data_cfg["target_column"]

    # Split
    train_df, val_df, test_df = chronological_split(
        df, data_cfg["train_ratio"], data_cfg["val_ratio"], data_cfg["test_ratio"]
    )

    # Fit HMM on training data and augment all splits
    hmm_detector = HMMRegimeDetector(
        n_states=cfg["model"]["hmm_n_states"],
        covariance_type=cfg["model"]["hmm_covariance_type"],
        n_iter=cfg["model"]["hmm_n_iter"],
        hmm_features=cfg["model"]["hmm_features"],
    )
    hmm_detector.fit(train_df)

    n_states = cfg["model"]["hmm_n_states"]
    state_cols = [f"hmm_state_{i}" for i in range(n_states)]

    for split_df in [train_df, val_df, test_df]:
        probs = hmm_detector.predict_proba(split_df)
        for i, col in enumerate(state_cols):
            split_df[col] = probs[:, i]

    # Augmented feature list: original features + HMM state probabilities
    augmented_features = feature_cols + state_cols
    cfg["data"]["feature_columns"] = augmented_features

    # Build DataModule from augmented DataFrames
    dm = CO2DataModule(cfg)
    dm.feature_scaler, dm.target_scaler = fit_scalers(
        train_df, augmented_features, target_col, data_cfg["scaler_type"]
    )

    train_scaled = apply_scalers(train_df, dm.feature_scaler, dm.target_scaler, augmented_features, target_col)
    val_scaled = apply_scalers(val_df, dm.feature_scaler, dm.target_scaler, augmented_features, target_col)
    test_scaled = apply_scalers(test_df, dm.feature_scaler, dm.target_scaler, augmented_features, target_col)

    sph = data_cfg["samples_per_hour"]
    lookback = data_cfg["lookback_hours"] * sph
    horizon = data_cfg["forecast_horizon_hours"] * sph
    stride = data_cfg.get("stride", 1)

    X_train, y_train = create_sequences(train_scaled, lookback, horizon, stride)
    X_val, y_val = create_sequences(val_scaled, lookback, horizon, stride)
    X_test, y_test = create_sequences(test_scaled, lookback, horizon, stride)

    dm.train_dataset = TimeSeriesDataset(X_train, y_train)
    dm.val_dataset = TimeSeriesDataset(X_val, y_val)
    dm.test_dataset = TimeSeriesDataset(X_test, y_test)

    # input_size = augmented features (9 original + 3 HMM states) + 1 target = 13
    # We must pass it explicitly because HMMLSTMForecaster.__init__ would
    # otherwise add hmm_n_states again on top of the already-augmented
    # feature_columns list.
    actual_input_size = len(augmented_features) + 1  # features + target
    model = HMMLSTMForecaster(cfg, input_size=actual_input_size)
    trainer, run_dir = create_trainer(cfg, model_name="bench_HMM-LSTM")
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    predictions = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()
    y_true_scaled = dm.test_dataset.y.numpy()

    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, dm.target_scaler)

    return compute_metrics(y_true, y_pred)


def benchmark_sarima(config: dict, datamodule: CO2DataModule) -> dict:
    """Train and evaluate SARIMA on synthetic data.

    SARIMA is univariate — it uses only the target column from each
    lookback window for forecasting. For the benchmark, a reduced
    seasonal order (1,0,1,24) is used instead of the full (1,1,1,288)
    to keep runtime reasonable while still testing seasonality handling.

    Args:
        config: Configuration dictionary.
        datamodule: Prepared CO2DataModule.

    Returns:
        Dictionary of test metrics.
    """
    cfg = copy.deepcopy(config)
    cfg["model"] = {
        "name": "SARIMA",
        # Reduced order for benchmark speed:
        # - Non-seasonal: AR(1), no differencing, MA(1)
        # - Seasonal: period=24 (2h at 5-min) to capture sub-daily pattern
        #   without the full 288-step seasonal period which is too slow
        "order": [1, 0, 1],
        "seasonal_order": [1, 0, 1, 24],
    }

    X_train = datamodule.train_dataset.X.numpy()
    y_train = datamodule.train_dataset.y.numpy()
    X_test = datamodule.test_dataset.X.numpy()
    y_test_scaled = datamodule.test_dataset.y.numpy()

    # Reconstruct a continuous training series from the target column
    target_idx = -1
    train_target_first_window = X_train[0, :, target_idx]
    train_target_remaining = y_train[:, 0]
    train_series = np.concatenate([train_target_first_window, train_target_remaining])

    model = SARIMAForecaster(cfg)
    model.fit(train_series)

    y_pred_scaled = model.predict_batch(X_test, target_idx=target_idx)

    y_pred = inverse_scale_target(y_pred_scaled, datamodule.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, datamodule.target_scaler)

    return compute_metrics(y_true, y_pred)


def benchmark_xgboost(config: dict, datamodule: CO2DataModule) -> dict:
    """Train and evaluate XGBoost on synthetic data.

    Uses the same sliding-window data as the neural models, flattened
    into feature vectors for the tree ensemble.

    Args:
        config: Configuration dictionary.
        datamodule: Prepared CO2DataModule.

    Returns:
        Dictionary of test metrics.
    """
    cfg = copy.deepcopy(config)
    cfg["model"] = {
        "name": "XGBoost",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "early_stopping_rounds": 20,
    }

    X_train = datamodule.train_dataset.X.numpy()
    y_train = datamodule.train_dataset.y.numpy()
    X_val = datamodule.val_dataset.X.numpy()
    y_val = datamodule.val_dataset.y.numpy()
    X_test = datamodule.test_dataset.X.numpy()
    y_test_scaled = datamodule.test_dataset.y.numpy()

    model = XGBoostForecaster(cfg)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred_scaled = model.predict(X_test)

    y_pred = inverse_scale_target(y_pred_scaled, datamodule.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, datamodule.target_scaler)

    return compute_metrics(y_true, y_pred)


def benchmark_catboost(config: dict, datamodule: CO2DataModule) -> dict:
    """Train and evaluate CatBoost on synthetic data.

    Uses the same sliding-window data as the neural models, flattened
    into feature vectors for the tree ensemble.

    Args:
        config: Configuration dictionary.
        datamodule: Prepared CO2DataModule.

    Returns:
        Dictionary of test metrics.
    """
    cfg = copy.deepcopy(config)
    cfg["model"] = {
        "name": "CatBoost",
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "early_stopping_rounds": 20,
    }

    X_train = datamodule.train_dataset.X.numpy()
    y_train = datamodule.train_dataset.y.numpy()
    X_val = datamodule.val_dataset.X.numpy()
    y_val = datamodule.val_dataset.y.numpy()
    X_test = datamodule.test_dataset.X.numpy()
    y_test_scaled = datamodule.test_dataset.y.numpy()

    model = CatBoostForecaster(cfg)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred_scaled = model.predict(X_test)

    y_pred = inverse_scale_target(y_pred_scaled, datamodule.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, datamodule.target_scaler)

    return compute_metrics(y_true, y_pred)


def check_thresholds(
    metrics: dict[str, float],
    model_name: str,
) -> tuple[bool, list[str]]:
    """Check whether metrics meet acceptance thresholds.

    Args:
        metrics: Dictionary of metric name -> value.
        model_name: Name of the model for reporting.

    Returns:
        Tuple of (all_passed, list of failure messages).
    """
    failures = []

    if metrics["rmse"] > THRESHOLDS["rmse"]:
        failures.append(
            f"  RMSE {metrics['rmse']:.2f} > {THRESHOLDS['rmse']} (threshold)"
        )
    if metrics["mae"] > THRESHOLDS["mae"]:
        failures.append(
            f"  MAE  {metrics['mae']:.2f} > {THRESHOLDS['mae']} (threshold)"
        )
    if metrics["r2"] < THRESHOLDS["r2"]:
        failures.append(
            f"  R2   {metrics['r2']:.4f} < {THRESHOLDS['r2']} (threshold)"
        )
    if metrics["mape"] > THRESHOLDS["mape"]:
        failures.append(
            f"  MAPE {metrics['mape']:.2f}% > {THRESHOLDS['mape']}% (threshold)"
        )

    return len(failures) == 0, failures


def print_report(all_results: dict[str, dict]) -> bool:
    """Print a formatted comparison report and return overall pass/fail.

    Args:
        all_results: Mapping of model_name -> metrics dict.

    Returns:
        True if all models passed all thresholds.
    """
    print("\n")
    print("=" * 72)
    print("  BENCHMARK RESULTS - Standard Synthetic Time Series")
    print("=" * 72)
    print(f"\n  Signal: y(t) = 600 + 50*sin(daily) + 15*sin(weekly) + N(0,5)")
    print(f"  Horizon: 1 hour (12 steps @ 5-min)")
    print(f"  Lookback: 24 hours (288 steps)")

    # Thresholds
    print(f"\n  Acceptance thresholds:")
    print(f"    RMSE  <= {THRESHOLDS['rmse']} ppm")
    print(f"    MAE   <= {THRESHOLDS['mae']} ppm")
    print(f"    R2    >= {THRESHOLDS['r2']}")
    print(f"    MAPE  <= {THRESHOLDS['mape']}%")

    # Results table
    print(f"\n  {'Model':<15s} {'RMSE':>8s} {'MAE':>8s} {'MAPE':>8s} {'R2':>8s} {'Status':>10s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    all_passed = True
    for model_name, metrics in all_results.items():
        passed, failures = check_thresholds(metrics, model_name)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(
            f"  {model_name:<15s} "
            f"{metrics['rmse']:>8.2f} "
            f"{metrics['mae']:>8.2f} "
            f"{metrics['mape']:>7.2f}% "
            f"{metrics['r2']:>8.4f} "
            f"{'   PASS' if passed else '** FAIL'}"
        )

        if failures:
            for f in failures:
                print(f"    {f}")

    print(f"\n  {'='*72}")
    if all_passed:
        print("  OVERALL: ALL MODELS PASSED")
    else:
        print("  OVERALL: SOME MODELS FAILED - investigate model architecture or training")
    print(f"  {'='*72}\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark models on a standard synthetic time series"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["lstm", "cnn_lstm", "hmm_lstm", "sarima", "xgboost", "catboost"],
        choices=["lstm", "cnn_lstm", "hmm_lstm", "sarima", "xgboost", "catboost"],
        help="Models to benchmark (default: all six)"
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Maximum training epochs (default: 25)"
    )
    parser.add_argument(
        "--n-days", type=int, default=120,
        help="Days of synthetic data to generate (default: 120)"
    )
    args = parser.parse_args()

    seed_everything(42)

    # Generate synthetic dataset
    print("\n[1/4] Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_days=args.n_days)
    print(f"  Generated {len(df)} samples ({args.n_days} days @ 5-min resolution)")
    print(f"  CO2 range: [{df['CO2'].min():.0f}, {df['CO2'].max():.0f}] ppm")
    print(f"  CO2 mean:  {df['CO2'].mean():.0f} ppm, std: {df['CO2'].std():.1f} ppm")

    config = build_config(max_epochs=args.epochs)

    # Prepare shared DataModule for LSTM and CNN-LSTM
    print("\n[2/4] Preparing data pipeline...")
    datamodule = prepare_datamodule_from_df(df, config)
    print(f"  Train: {len(datamodule.train_dataset)} sequences")
    print(f"  Val:   {len(datamodule.val_dataset)} sequences")
    print(f"  Test:  {len(datamodule.test_dataset)} sequences")

    # Run benchmarks
    all_results = {}

    print("\n[3/4] Running model benchmarks...")

    if "lstm" in args.models:
        print(f"\n{'─'*60}")
        print(f"  Benchmarking: LSTM")
        print(f"{'─'*60}")
        t0 = time.time()
        all_results["LSTM"] = benchmark_lstm(config, datamodule)
        elapsed = time.time() - t0
        print(f"  LSTM completed in {elapsed:.1f}s")

    if "cnn_lstm" in args.models:
        print(f"\n{'─'*60}")
        print(f"  Benchmarking: CNN-LSTM")
        print(f"{'─'*60}")
        t0 = time.time()
        all_results["CNN-LSTM"] = benchmark_cnn_lstm(config, datamodule)
        elapsed = time.time() - t0
        print(f"  CNN-LSTM completed in {elapsed:.1f}s")

    if "hmm_lstm" in args.models:
        print(f"\n{'─'*60}")
        print(f"  Benchmarking: HMM-LSTM")
        print(f"{'─'*60}")
        t0 = time.time()
        all_results["HMM-LSTM"] = benchmark_hmm_lstm(config, df)
        elapsed = time.time() - t0
        print(f"  HMM-LSTM completed in {elapsed:.1f}s")

    if "sarima" in args.models:
        print(f"\n{'─'*60}")
        print(f"  Benchmarking: SARIMA")
        print(f"{'─'*60}")
        t0 = time.time()
        all_results["SARIMA"] = benchmark_sarima(config, datamodule)
        elapsed = time.time() - t0
        print(f"  SARIMA completed in {elapsed:.1f}s")

    if "xgboost" in args.models:
        print(f"\n{'─'*60}")
        print(f"  Benchmarking: XGBoost")
        print(f"{'─'*60}")
        t0 = time.time()
        all_results["XGBoost"] = benchmark_xgboost(config, datamodule)
        elapsed = time.time() - t0
        print(f"  XGBoost completed in {elapsed:.1f}s")

    if "catboost" in args.models:
        print(f"\n{'─'*60}")
        print(f"  Benchmarking: CatBoost")
        print(f"{'─'*60}")
        t0 = time.time()
        all_results["CatBoost"] = benchmark_catboost(config, datamodule)
        elapsed = time.time() - t0
        print(f"  CatBoost completed in {elapsed:.1f}s")

    # Report
    print("\n[4/4] Generating report...")
    all_passed = print_report(all_results)

    # Save results to JSON
    output_dir = Path(config["training"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "thresholds": THRESHOLDS,
                "results": all_results,
                "all_passed": all_passed,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
