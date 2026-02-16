"""Phase 4: Incremental feature addition on best preprocessing variant.

Starting from the best variant (selected in Phase 3), adds feature groups
one at a time. Each addition is tested on ALL 7 models. Accept/reject
based on validation RMSE.

Feature groups (tested in order, cumulative):
    1. 12h harmonic (Day12h_sin/cos)       — deterministic
    2. Occupancy proxy (is_weekend, is_active_hours) — deterministic
    3. CO2 deviation from rolling baseline  — stateful (post-split)
    4. Meteorological rates (dPression, dTemperatureExt) — stateful (post-split)

Usage:
    python scripts/run_feature_ablation.py
    python scripts/run_feature_ablation.py --base-variant preproc_A
    python scripts/run_feature_ablation.py --models lstm xgboost --horizons 1
    python scripts/run_feature_ablation.py --epochs 30
"""

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import CO2DataModule
from src.data.pipeline import run_preprocessing_pipeline
from src.data.preprocessing import inverse_scale_target
from src.evaluation.metrics import compute_metrics
from src.models.lstm import LSTMForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.hmm_lstm import HMMLSTMForecaster, HMMRegimeDetector
from src.models.sarima import SARIMAForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.catboost_model import CatBoostForecaster
from src.training.trainer import create_trainer
from src.utils.config import load_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

VARIANT_CONFIGS = {
    "preproc_A": "configs/experiments/preproc_A_simple_5min.yaml",
    "preproc_B": "configs/experiments/preproc_B_simple_1h.yaml",
    "preproc_C": "configs/experiments/preproc_C_enhanced_5min.yaml",
    "preproc_D": "configs/experiments/preproc_D_enhanced_1h.yaml",
}

MODEL_CONFIGS = {
    "lstm": "configs/lstm.yaml",
    "cnn_lstm": "configs/cnn_lstm.yaml",
    "hmm_lstm": "configs/hmm_lstm.yaml",
    "tft": "configs/tft.yaml",
    "sarima": "configs/sarima.yaml",
    "xgboost": "configs/xgboost.yaml",
    "catboost": "configs/catboost.yaml",
}

ALL_MODELS = ["lstm", "cnn_lstm", "hmm_lstm", "tft", "sarima", "xgboost", "catboost"]


# ──────────────────────────────────────────────────────────────────────
#  Feature group definitions
# ──────────────────────────────────────────────────────────────────────

def add_group1_12h_harmonic(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Group 1: 12h harmonic — captures bimodal occupancy pattern.

    Deterministic (function of timestamp only), so can be computed on all splits.
    """
    new_cols = ["Day12h_sin", "Day12h_cos"]
    for df in [train_df, val_df, test_df]:
        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
        else:
            dt = pd.to_datetime(df.index)
        hour = dt.dt.hour + dt.dt.minute / 60.0
        df["Day12h_sin"] = np.sin(2 * np.pi * hour / 12.0)
        df["Day12h_cos"] = np.cos(2 * np.pi * hour / 12.0)
    return train_df, val_df, test_df, new_cols


def add_group2_occupancy_proxy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Group 2: Occupancy proxy features — binary indicators."""
    new_cols = ["is_weekend", "is_active_hours"]
    for df in [train_df, val_df, test_df]:
        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
        else:
            dt = pd.to_datetime(df.index)
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
        hour = dt.dt.hour
        df["is_active_hours"] = ((hour >= 7) & (hour < 23)).astype(float)
    return train_df, val_df, test_df, new_cols


def add_group3_co2_baseline_deviation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Group 3: CO2 deviation from rolling 24h minimum baseline.

    Stateful (rolling window) — computed within each split independently.
    """
    new_cols = ["CO2_above_baseline"]
    samples_24h = (24 * 60) // interval_minutes  # 288 at 5-min, 24 at 1h
    for df in [train_df, val_df, test_df]:
        rolling_min = df["CO2"].rolling(window=samples_24h, min_periods=1).min()
        df["CO2_above_baseline"] = df["CO2"] - rolling_min
    # Drop NaN rows introduced at start of each split
    for i, df in enumerate([train_df, val_df, test_df]):
        before = len(df)
        df_clean = df.dropna().reset_index(drop=True)
        dropped = before - len(df_clean)
        if dropped > 0:
            logger.info(f"  Group 3: Dropped {dropped} NaN rows from split {i}")
        if i == 0:
            train_df = df_clean
        elif i == 1:
            val_df = df_clean
        else:
            test_df = df_clean
    return train_df, val_df, test_df, new_cols


def add_group4_meteo_rates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Group 4: Meteorological rate features — pressure and temperature tendency.

    Stateful (diff) — computed within each split independently.
    """
    new_cols = ["dPression", "dTemperatureExt"]
    hours_per_step = interval_minutes / 60.0
    for df in [train_df, val_df, test_df]:
        df["dPression"] = df["Pression"].diff() / hours_per_step
        df["dTemperatureExt"] = df["TemperatureExt"].diff() / hours_per_step
    # Drop NaN rows from diff
    for i, df in enumerate([train_df, val_df, test_df]):
        df_clean = df.dropna().reset_index(drop=True)
        if i == 0:
            train_df = df_clean
        elif i == 1:
            val_df = df_clean
        else:
            test_df = df_clean
    return train_df, val_df, test_df, new_cols


FEATURE_GROUPS = [
    ("group1_12h_harmonic", add_group1_12h_harmonic),
    ("group2_occupancy_proxy", add_group2_occupancy_proxy),
    ("group3_co2_baseline", add_group3_co2_baseline_deviation),
    ("group4_meteo_rates", add_group4_meteo_rates),
]


# ──────────────────────────────────────────────────────────────────────
#  Model training (reused from comparison script logic)
# ──────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str,
) -> dict:
    """Train a model and return metrics.

    Returns dict with val_rmse and test metrics.
    """
    cfg = copy.deepcopy(config)
    cfg["training"]["results_dir"] = "results/feature_ablation/runs"

    seed_everything(cfg["training"]["seed"])

    if model_name == "hmm_lstm":
        return _run_hmm_lstm(cfg, train_df.copy(), val_df.copy(), test_df.copy(), run_name)
    elif model_name == "tft":
        return _run_tft(cfg, train_df.copy(), val_df.copy(), test_df.copy(), run_name)
    else:
        dm = CO2DataModule(cfg)
        dm._build_datasets(train_df, val_df, test_df)

        if model_name == "sarima":
            return _run_sarima(cfg, dm, run_name)
        elif model_name == "xgboost":
            return _run_xgboost(cfg, dm, run_name)
        elif model_name == "catboost":
            return _run_catboost(cfg, dm, run_name)
        elif model_name == "lstm":
            return _run_pl_model(cfg, dm, run_name, LSTMForecaster)
        elif model_name == "cnn_lstm":
            return _run_pl_model(cfg, dm, run_name, CNNLSTMForecaster)
        else:
            raise ValueError(f"Unknown model: {model_name}")


def _run_pl_model(config, dm, run_name, model_cls):
    """Train a PyTorch Lightning model and return metrics."""
    model = model_cls(config)
    trainer, run_dir = create_trainer(config, model_name=run_name)
    trainer.fit(model, datamodule=dm)

    # Validation RMSE from trainer
    val_metrics = trainer.callback_metrics
    val_rmse = float(val_metrics.get("val_loss", float("nan"))) ** 0.5

    predictions = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()
    y_true_scaled = dm.test_dataset.y.numpy()
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, dm.target_scaler)

    metrics = compute_metrics(y_true, y_pred)
    # Also evaluate on validation set for decision making
    val_preds = trainer.predict(model, dm.val_dataloader(), ckpt_path="best")
    y_val_pred_scaled = torch.cat(val_preds, dim=0).numpy()
    y_val_true_scaled = dm.val_dataset.y.numpy()
    y_val_pred = inverse_scale_target(y_val_pred_scaled, dm.target_scaler)
    y_val_true = inverse_scale_target(y_val_true_scaled, dm.target_scaler)
    val_metrics_dict = compute_metrics(y_val_true, y_val_pred)

    metrics["val_rmse"] = val_metrics_dict["rmse"]
    return metrics


def _run_hmm_lstm(config, train_df, val_df, test_df, run_name):
    """Train HMM-LSTM and return metrics."""
    model_cfg = config["model"]
    hmm_detector = HMMRegimeDetector(
        n_states=model_cfg["hmm_n_states"],
        covariance_type=model_cfg["hmm_covariance_type"],
        n_iter=model_cfg["hmm_n_iter"],
        hmm_features=model_cfg["hmm_features"],
    )
    hmm_detector.fit(train_df)

    n_states = model_cfg["hmm_n_states"]
    state_cols = [f"hmm_state_{i}" for i in range(n_states)]
    for split_df in [train_df, val_df, test_df]:
        probs = hmm_detector.predict_proba(split_df)
        for i, col in enumerate(state_cols):
            split_df[col] = probs[:, i]

    config["data"]["feature_columns"] = config["data"]["feature_columns"] + state_cols
    dm = CO2DataModule.from_dataframes(train_df, val_df, test_df, config)

    n_input = len(config["data"]["feature_columns"]) + 1
    model = HMMLSTMForecaster(config, input_size=n_input)
    trainer, run_dir = create_trainer(config, model_name=run_name)
    trainer.fit(model, datamodule=dm)

    predictions = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()
    y_true_scaled = dm.test_dataset.y.numpy()
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, dm.target_scaler)

    metrics = compute_metrics(y_true, y_pred)

    val_preds = trainer.predict(model, dm.val_dataloader(), ckpt_path="best")
    y_val_pred_scaled = torch.cat(val_preds, dim=0).numpy()
    y_val_true_scaled = dm.val_dataset.y.numpy()
    y_val_pred = inverse_scale_target(y_val_pred_scaled, dm.target_scaler)
    y_val_true = inverse_scale_target(y_val_true_scaled, dm.target_scaler)
    metrics["val_rmse"] = compute_metrics(y_val_true, y_val_pred)["rmse"]
    return metrics


def _run_tft(config, train_df, val_df, test_df, run_name):
    """Train TFT and return metrics."""
    import lightning.pytorch as lpl
    from lightning.pytorch.callbacks import (
        EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
    from src.models.tft import build_tft_model, create_tft_datasets, prepare_tft_dataframe

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    datetime_col = config["data"]["datetime_column"]
    if datetime_col in combined_df.columns:
        combined_df = combined_df.sort_values(datetime_col).reset_index(drop=True)

    combined_df = prepare_tft_dataframe(combined_df, config)
    training_data, validation_data, test_data, combined_df = create_tft_datasets(combined_df, config)

    batch_size = config["training"]["batch_size"]
    nw = config["training"].get("num_workers", 0)
    train_dl = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=nw)
    val_dl = validation_data.to_dataloader(train=False, batch_size=batch_size, num_workers=nw)
    test_dl = test_data.to_dataloader(train=False, batch_size=batch_size, num_workers=nw)

    tft = build_tft_model(training_data, config)

    training_cfg = config["training"]
    results_dir = Path(training_cfg["results_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"{run_name}_{timestamp}"

    trainer = lpl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        precision=training_cfg.get("precision", 32),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=training_cfg["patience"], mode="min"),
            ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="best-{epoch}-{val_loss:.4f}",
                            monitor="val_loss", mode="min", save_top_k=1),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=20),
        ],
        gradient_clip_val=training_cfg.get("gradient_clip_val", 0.1),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
        logger=TensorBoardLogger(save_dir=str(run_dir), name="tb_logs"),
        deterministic=True,
    )

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_tft = tft.__class__.load_from_checkpoint(best_ckpt)
    predictions = best_tft.predict(test_dl, mode="prediction", return_x=False)
    y_pred = torch.cat([p for p in predictions], dim=0).numpy()
    actuals = torch.cat([y[0] for x, y in iter(test_dl)], dim=0).numpy()

    metrics = compute_metrics(actuals.ravel(), y_pred.ravel())

    # Validation metrics
    val_predictions = best_tft.predict(val_dl, mode="prediction", return_x=False)
    y_val_pred = torch.cat([p for p in val_predictions], dim=0).numpy()
    val_actuals = torch.cat([y[0] for x, y in iter(val_dl)], dim=0).numpy()
    metrics["val_rmse"] = compute_metrics(val_actuals.ravel(), y_val_pred.ravel())["rmse"]
    return metrics


def _run_sarima(config, dm, run_name):
    """Train SARIMA and return metrics."""
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_val = dm.val_dataset.X.numpy()
    y_val_scaled = dm.val_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test_scaled = dm.test_dataset.y.numpy()

    target_idx = -1
    train_series = np.concatenate([
        X_train[0, :, target_idx],
        y_train[:, 0],
    ])

    model = SARIMAForecaster(config)
    model.fit(train_series)

    y_pred_scaled = model.predict_batch(X_test, target_idx=target_idx)
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, dm.target_scaler)

    metrics = compute_metrics(y_true, y_pred)

    # Validation metrics
    y_val_pred_scaled = model.predict_batch(X_val, target_idx=target_idx)
    y_val_pred = inverse_scale_target(y_val_pred_scaled, dm.target_scaler)
    y_val_true = inverse_scale_target(y_val_scaled, dm.target_scaler)
    metrics["val_rmse"] = compute_metrics(y_val_true, y_val_pred)["rmse"]
    return metrics


def _run_xgboost(config, dm, run_name):
    """Train XGBoost and return metrics."""
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_val = dm.val_dataset.X.numpy()
    y_val = dm.val_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test_scaled = dm.test_dataset.y.numpy()

    model = XGBoostForecaster(config)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred_scaled = model.predict(X_test)
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, dm.target_scaler)

    metrics = compute_metrics(y_true, y_pred)

    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = inverse_scale_target(y_val_pred_scaled, dm.target_scaler)
    y_val_true = inverse_scale_target(y_val, dm.target_scaler)
    metrics["val_rmse"] = compute_metrics(y_val_true, y_val_pred)["rmse"]
    return metrics


def _run_catboost(config, dm, run_name):
    """Train CatBoost and return metrics."""
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_val = dm.val_dataset.X.numpy()
    y_val = dm.val_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test_scaled = dm.test_dataset.y.numpy()

    model = CatBoostForecaster(config)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred_scaled = model.predict(X_test)
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, dm.target_scaler)

    metrics = compute_metrics(y_true, y_pred)

    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = inverse_scale_target(y_val_pred_scaled, dm.target_scaler)
    y_val_true = inverse_scale_target(y_val, dm.target_scaler)
    metrics["val_rmse"] = compute_metrics(y_val_true, y_val_pred)["rmse"]
    return metrics


def check_correlation_guardrail(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """Check for highly correlated feature pairs and return features to drop.

    Args:
        train_df: Training data.
        feature_columns: Current feature column names.
        threshold: Correlation threshold for flagging.

    Returns:
        List of feature names to drop (the less interpretable of each pair).
    """
    # Only check numeric feature columns that exist
    valid_cols = [c for c in feature_columns if c in train_df.columns]
    if len(valid_cols) < 2:
        return []

    corr_matrix = train_df[valid_cols].corr().abs()
    to_drop = set()
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the one that appears later in the list (less interpretable)
                drop_candidate = valid_cols[j]
                to_drop.add(drop_candidate)
                logger.info(f"  Correlation guardrail: |r({valid_cols[i]}, {valid_cols[j]})| = "
                            f"{corr_matrix.iloc[i, j]:.3f} > {threshold}. Flagging {drop_candidate}.")
    return list(to_drop)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Phase 4: Incremental feature ablation")
    parser.add_argument("--base-variant", type=str, default=None,
                        help="Base variant to start from. If not set, reads from Phase 3 results.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 24])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    # Determine base variant
    if args.base_variant:
        base_variant = args.base_variant
    else:
        best_variant_path = Path("results/preprocessing_comparison/best_variant.json")
        if best_variant_path.exists():
            with open(best_variant_path) as f:
                base_variant = json.load(f)["best_variant"]
            print(f"  Using best variant from Phase 3: {base_variant}")
        else:
            base_variant = "preproc_A"
            print(f"  No Phase 3 results found, defaulting to: {base_variant}")

    output_dir = Path("results/feature_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions_log = []
    all_results = []

    print(f"\n{'='*70}")
    print(f"  PHASE 4: INCREMENTAL FEATURE ABLATION")
    print(f"  Base variant: {base_variant}")
    print(f"  Models: {args.models}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Feature groups: {[name for name, _ in FEATURE_GROUPS]}")
    print(f"{'='*70}\n")

    # Run baseline (no additional features)
    print(f"\n{'─'*60}")
    print(f"  BASELINE: {base_variant} (no added features)")
    print(f"{'─'*60}")

    baseline_results = {}
    for horizon in args.horizons:
        for model_name in args.models:
            config = load_full_config(base_variant, model_name)
            config["data"]["forecast_horizon_hours"] = horizon
            if args.epochs:
                config["training"]["max_epochs"] = args.epochs

            raw_dir = Path(config["data"].get("raw_dir", "data/raw"))
            train_df, val_df, test_df = run_preprocessing_pipeline(raw_dir=raw_dir, variant_config=config)

            run_name = f"ablation_baseline_{model_name}_h{horizon}"
            t0 = time.time()
            try:
                metrics = train_and_evaluate(model_name, config, train_df, val_df, test_df, run_name)
                elapsed = time.time() - t0
                key = (model_name, horizon)
                baseline_results[key] = metrics.get("val_rmse", float("nan"))

                all_results.append({
                    "group": "baseline", "model": model_name, "horizon": horizon,
                    "val_rmse": metrics.get("val_rmse", float("nan")),
                    "test_rmse": metrics["rmse"], "test_mae": metrics["mae"],
                    "test_r2": metrics["r2"], "test_mape": metrics["mape"],
                })
                print(f"  {model_name} h{horizon}: val_RMSE={metrics.get('val_rmse', 'N/A'):.2f}  "
                      f"test_RMSE={metrics['rmse']:.2f}  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  {model_name} h{horizon}: FAILED ({e}) ({elapsed:.1f}s)")
                baseline_results[(model_name, horizon)] = float("nan")

    # Track accepted feature columns cumulatively
    # Start from base variant's feature columns
    sample_config = load_full_config(base_variant, "lstm")
    accepted_extra_features: list[str] = []
    consecutive_rejections = 0

    # Process each feature group incrementally
    for group_name, group_fn in FEATURE_GROUPS:
        if consecutive_rejections >= 2:
            print(f"\n  STOPPING: 2 consecutive rejections. Skipping remaining groups.")
            decisions_log.append({
                "group": group_name, "decision": "SKIPPED",
                "reason": "2 consecutive rejections",
            })
            break

        print(f"\n{'─'*60}")
        print(f"  TESTING GROUP: {group_name}")
        print(f"{'─'*60}")

        group_results = {}
        prev_val_rmses = {}

        for horizon in args.horizons:
            for model_name in args.models:
                config = load_full_config(base_variant, model_name)
                config["data"]["forecast_horizon_hours"] = horizon
                if args.epochs:
                    config["training"]["max_epochs"] = args.epochs

                raw_dir = Path(config["data"].get("raw_dir", "data/raw"))
                train_df, val_df, test_df = run_preprocessing_pipeline(raw_dir=raw_dir, variant_config=config)

                # Add all previously accepted extra features
                interval_minutes = config["data"]["interval_minutes"]
                for prev_name, prev_fn in FEATURE_GROUPS:
                    if prev_name == group_name:
                        break
                    if prev_name in [d["group"] for d in decisions_log if d["decision"] == "ACCEPT"]:
                        train_df, val_df, test_df, cols = prev_fn(
                            train_df, val_df, test_df, interval_minutes,
                        )

                # Add current group's features
                train_df, val_df, test_df, new_cols = group_fn(
                    train_df, val_df, test_df, interval_minutes,
                )

                # Update feature columns in config
                all_extra = []
                for prev_name, _ in FEATURE_GROUPS:
                    if prev_name == group_name:
                        break
                    if prev_name in [d["group"] for d in decisions_log if d["decision"] == "ACCEPT"]:
                        # Get feature names from the accepted group
                        for d in decisions_log:
                            if d["group"] == prev_name and d["decision"] == "ACCEPT":
                                all_extra.extend(d.get("features", []))
                all_extra.extend(new_cols)

                config["data"]["feature_columns"] = config["data"]["feature_columns"] + all_extra

                run_name = f"ablation_{group_name}_{model_name}_h{horizon}"
                t0 = time.time()
                try:
                    metrics = train_and_evaluate(model_name, config, train_df, val_df, test_df, run_name)
                    elapsed = time.time() - t0
                    key = (model_name, horizon)
                    group_results[key] = metrics.get("val_rmse", float("nan"))
                    prev_val_rmses[key] = baseline_results.get(key, float("nan"))

                    all_results.append({
                        "group": group_name, "model": model_name, "horizon": horizon,
                        "val_rmse": metrics.get("val_rmse", float("nan")),
                        "test_rmse": metrics["rmse"], "test_mae": metrics["mae"],
                        "test_r2": metrics["r2"], "test_mape": metrics["mape"],
                    })
                    print(f"  {model_name} h{horizon}: val_RMSE={metrics.get('val_rmse', 'N/A'):.2f}  "
                          f"test_RMSE={metrics['rmse']:.2f}  ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {model_name} h{horizon}: FAILED ({e}) ({elapsed:.1f}s)")

        # Decision: compare median val_RMSE
        current_vals = [v for v in group_results.values() if not np.isnan(v)]
        previous_vals = [prev_val_rmses.get(k, float("nan")) for k in group_results]
        previous_vals = [v for v in previous_vals if not np.isnan(v)]

        if current_vals and previous_vals:
            median_current = np.median(current_vals)
            median_previous = np.median(previous_vals)

            if median_current < median_previous:
                decision = "ACCEPT"
                consecutive_rejections = 0
                # Update baseline for next group
                for k, v in group_results.items():
                    baseline_results[k] = v
            else:
                decision = "REJECT"
                consecutive_rejections += 1
        else:
            decision = "REJECT"
            consecutive_rejections += 1

        decisions_log.append({
            "group": group_name,
            "decision": decision,
            "features": new_cols,
            "median_val_rmse_before": float(np.median(previous_vals)) if previous_vals else None,
            "median_val_rmse_after": float(np.median(current_vals)) if current_vals else None,
        })

        print(f"\n  Decision: {decision} ({group_name})")
        if current_vals and previous_vals:
            print(f"    Median val RMSE: {np.median(previous_vals):.2f} -> {np.median(current_vals):.2f}")

        # Correlation guardrail after acceptance
        if decision == "ACCEPT" and current_vals:
            sample_config_check = load_full_config(base_variant, "lstm")
            all_feature_cols = sample_config_check["data"]["feature_columns"] + new_cols
            # Check on any available train_df
            corr_drops = check_correlation_guardrail(train_df, all_feature_cols)
            if corr_drops:
                print(f"  Correlation guardrail: would drop {corr_drops}")
                decisions_log[-1]["correlation_drops"] = corr_drops

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "decisions.json", "w") as f:
        json.dump(decisions_log, f, indent=2, default=str)

    # Print final summary
    print(f"\n{'='*70}")
    print("  FEATURE ABLATION SUMMARY")
    print(f"{'='*70}")
    for d in decisions_log:
        status = d["decision"]
        features = d.get("features", [])
        print(f"  {d['group']:>30}: {status}  {features}")
    print(f"{'='*70}\n")

    accepted_features = []
    for d in decisions_log:
        if d["decision"] == "ACCEPT":
            accepted_features.extend(d.get("features", []))

    print(f"  Accepted features: {accepted_features}")
    print(f"  Results saved to: {output_dir}")


def load_full_config(variant_name: str, model_name: str) -> dict:
    """Load merged config for a variant + model combination."""
    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / MODEL_CONFIGS[model_name]),
        str(PROJECT_ROOT / VARIANT_CONFIGS[variant_name]),
    ]
    return load_config(config_files)


if __name__ == "__main__":
    main()
