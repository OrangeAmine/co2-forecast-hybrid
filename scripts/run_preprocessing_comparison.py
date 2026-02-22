"""Phase 3: Preprocessing variant comparison.

Runs 7 models x 4 variants x 2 horizons = 56 training runs to find
the best preprocessing configuration for CO2 forecasting.

Model hyperparameters are FIXED across all variants - only the
preprocessing and feature set changes.

Usage:
    python scripts/run_preprocessing_comparison.py
    python scripts/run_preprocessing_comparison.py --variants preproc_A preproc_C
    python scripts/run_preprocessing_comparison.py --models lstm xgboost --horizons 1
    python scripts/run_preprocessing_comparison.py --epochs 30
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import CO2DataModule
from src.data.pipeline import run_preprocessing_pipeline
from src.data.preprocessing import (
    chronological_split,
    fit_scalers,
    apply_scalers,
    create_sequences,
    inverse_scale_target,
)
from src.data.dataset import TimeSeriesDataset
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


VARIANT_CONFIGS = {
    "preproc_A": "configs/experiments/preproc_A_simple_5min.yaml",
    "preproc_B": "configs/experiments/preproc_B_simple_1h.yaml",
    "preproc_C": "configs/experiments/preproc_C_enhanced_5min.yaml",
    "preproc_D": "configs/experiments/preproc_D_enhanced_1h.yaml",
}

SUMMARY_CSV = Path("results/preprocessing_comparison/summary.csv")


def _load_completed_runs(summary_path: Path) -> set[tuple[str, str, int]]:
    """Load (variant, model, horizon) keys that already finished successfully.

    Args:
        summary_path: Path to summary.csv from a previous (possibly interrupted) run.

    Returns:
        Set of (variant, model, horizon) tuples with status == "OK".
    """
    if not summary_path.exists():
        return set()
    try:
        df = pd.read_csv(summary_path)
        ok = df[df["status"] == "OK"]
        return set(zip(ok["variant"], ok["model"], ok["horizon"]))
    except Exception:
        return set()


def _append_result_row(summary_path: Path, row: dict) -> None:
    """Append a single result row to summary.csv (creates file if needed).

    Args:
        summary_path: Path to summary.csv.
        row: Dict with column names → values.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    row_df = pd.DataFrame([row])
    row_df.to_csv(summary_path, mode="a", header=write_header, index=False)

ALL_MODELS = ["lstm", "cnn_lstm", "hmm_lstm", "tft", "xgboost", "catboost"]

# Model config YAML files
MODEL_CONFIGS = {
    "lstm": "configs/lstm.yaml",
    "cnn_lstm": "configs/cnn_lstm.yaml",
    "hmm_lstm": "configs/hmm_lstm.yaml",
    "tft": "configs/tft.yaml",
    "sarima": "configs/sarima.yaml",
    "xgboost": "configs/xgboost.yaml",
    "catboost": "configs/catboost.yaml",
}


def load_full_config(variant_name: str, model_name: str) -> dict:
    """Load merged config: training + data + model + experiment variant.

    Args:
        variant_name: Preprocessing variant key.
        model_name: Model key (lstm, cnn_lstm, etc.).

    Returns:
        Merged config dict.
    """
    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / MODEL_CONFIGS[model_name]),
        str(PROJECT_ROOT / VARIANT_CONFIGS[variant_name]),
    ]
    return load_config(config_files)


def _build_datamodule_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> CO2DataModule:
    """Build a CO2DataModule from pre-split DataFrames.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        config: Merged config dict.

    Returns:
        Configured CO2DataModule with datasets ready.
    """
    dm = CO2DataModule(config)
    dm.build_datasets(train_df, val_df, test_df)
    return dm


# ----------------------------------------------------------------------
#  Per-model train+evaluate functions
# ----------------------------------------------------------------------

def run_lstm(config: dict, dm: CO2DataModule, run_name: str) -> dict:
    """Train and evaluate LSTM."""
    model = LSTMForecaster(config)
    trainer, run_dir = create_trainer(config, model_name=run_name)
    trainer.fit(model, datamodule=dm)
    predictions = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    assert predictions is not None
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()  # type: ignore[arg-type]
    assert dm.test_dataset is not None
    y_true_scaled = dm.test_dataset.y.numpy()
    assert dm.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, dm.target_scaler)
    return compute_metrics(y_true, y_pred)


def run_cnn_lstm(config: dict, dm: CO2DataModule, run_name: str) -> dict:
    """Train and evaluate CNN-LSTM."""
    model = CNNLSTMForecaster(config)
    trainer, run_dir = create_trainer(config, model_name=run_name)
    trainer.fit(model, datamodule=dm)
    predictions = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    assert predictions is not None
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()  # type: ignore[arg-type]
    assert dm.test_dataset is not None
    y_true_scaled = dm.test_dataset.y.numpy()
    assert dm.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, dm.target_scaler)
    return compute_metrics(y_true, y_pred)


def run_hmm_lstm(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str,
) -> dict:
    """Train and evaluate HMM-LSTM.

    Requires separate data pipeline: fit HMM on train, augment all splits.
    """
    cfg = copy.deepcopy(config)
    model_cfg = cfg["model"]

    # Fit HMM on training data
    hmm_detector = HMMRegimeDetector(
        n_states=model_cfg["hmm_n_states"],
        covariance_type=model_cfg["hmm_covariance_type"],
        n_iter=model_cfg["hmm_n_iter"],
        hmm_features=model_cfg["hmm_features"],
    )
    hmm_detector.fit(train_df)

    # Augment all splits with HMM posterior probabilities
    n_states = model_cfg["hmm_n_states"]
    state_cols = [f"hmm_state_{i}" for i in range(n_states)]

    for split_df in [train_df, val_df, test_df]:
        probs = hmm_detector.predict_proba(split_df)
        for i, col in enumerate(state_cols):
            split_df[col] = probs[:, i]

    # Update feature columns
    cfg["data"]["feature_columns"] = cfg["data"]["feature_columns"] + state_cols

    # Build DataModule from augmented DataFrames
    dm = CO2DataModule.from_dataframes(train_df, val_df, test_df, cfg)

    # Train
    n_input = len(cfg["data"]["feature_columns"]) + 1
    model = HMMLSTMForecaster(cfg, input_size=n_input)
    trainer, run_dir = create_trainer(cfg, model_name=run_name)
    trainer.fit(model, datamodule=dm)

    predictions = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    assert predictions is not None
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()  # type: ignore[arg-type]
    assert dm.test_dataset is not None
    y_true_scaled = dm.test_dataset.y.numpy()
    assert dm.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, dm.target_scaler)
    return compute_metrics(y_true, y_pred)


def run_tft(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str,
) -> dict:
    """Train and evaluate TFT.

    TFT uses its own data pipeline (TimeSeriesDataSet). We recombine
    the splits into a single DF and let TFT's pipeline re-split.
    """
    import lightning.pytorch as lpl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
    from src.models.tft import build_tft_model, create_tft_datasets, prepare_tft_dataframe

    cfg = copy.deepcopy(config)

    # Recombine splits into a single DF for TFT's pipeline
    # TFT does its own splitting based on time_idx
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Ensure datetime is present and sorted
    datetime_col = cfg["data"]["datetime_column"]
    if datetime_col in combined_df.columns:
        combined_df = combined_df.sort_values(datetime_col).reset_index(drop=True)

    # TFT's prepare step
    combined_df = prepare_tft_dataframe(combined_df, cfg)

    # Create TFT datasets
    training_data, validation_data, test_data, combined_df = create_tft_datasets(combined_df, cfg)

    batch_size = cfg["training"]["batch_size"]
    train_dl = training_data.to_dataloader(
        train=True, batch_size=batch_size,
        num_workers=cfg["training"].get("num_workers", 0),
    )
    val_dl = validation_data.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=cfg["training"].get("num_workers", 0),
    )
    test_dl = test_data.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=cfg["training"].get("num_workers", 0),
    )

    # Build model
    tft = build_tft_model(training_data, cfg)

    # Trainer (lightning.pytorch for TFT compatibility)
    training_cfg = cfg["training"]
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
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="best-{epoch}-{val_loss:.4f}",
                monitor="val_loss", mode="min", save_top_k=1,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=20),
        ],
        gradient_clip_val=training_cfg.get("gradient_clip_val", 0.1),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
        logger=TensorBoardLogger(save_dir=str(run_dir), name="tb_logs"),
        deterministic="warn",
    )

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Predict with best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    best_tft = tft.__class__.load_from_checkpoint(best_ckpt)
    predictions = best_tft.predict(test_dl, mode="prediction", return_x=False)
    y_pred = torch.cat([p for p in predictions], dim=0).numpy()

    # Actuals from test dataloader
    actuals = torch.cat([y[0] for x, y in iter(test_dl)], dim=0).numpy()

    return compute_metrics(actuals.ravel(), y_pred.ravel())


def run_sarima(config: dict, dm: CO2DataModule, run_name: str) -> dict:
    """Train and evaluate SARIMA."""
    cfg = copy.deepcopy(config)

    assert dm.train_dataset is not None
    assert dm.test_dataset is not None
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test_scaled = dm.test_dataset.y.numpy()

    # Reconstruct continuous training series
    target_idx = -1
    train_target_first_window = X_train[0, :, target_idx]
    train_target_remaining = y_train[:, 0]
    train_series = np.concatenate([train_target_first_window, train_target_remaining])

    model = SARIMAForecaster(cfg)
    model.fit(train_series)
    y_pred_scaled = model.predict_batch(X_test, target_idx=target_idx)

    assert dm.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, dm.target_scaler)
    return compute_metrics(y_true, y_pred)


def run_xgboost(config: dict, dm: CO2DataModule, run_name: str) -> dict:
    """Train and evaluate XGBoost."""
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_val = dm.val_dataset.X.numpy()
    y_val = dm.val_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test_scaled = dm.test_dataset.y.numpy()

    model = XGBoostForecaster(config)
    model.fit(X_train, y_train, X_val, y_val)
    y_pred_scaled = model.predict(X_test)

    assert dm.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, dm.target_scaler)
    return compute_metrics(y_true, y_pred)


def run_catboost(config: dict, dm: CO2DataModule, run_name: str) -> dict:
    """Train and evaluate CatBoost."""
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    X_train = dm.train_dataset.X.numpy()
    y_train = dm.train_dataset.y.numpy()
    X_val = dm.val_dataset.X.numpy()
    y_val = dm.val_dataset.y.numpy()
    X_test = dm.test_dataset.X.numpy()
    y_test_scaled = dm.test_dataset.y.numpy()

    model = CatBoostForecaster(config)
    model.fit(X_train, y_train, X_val, y_val)
    y_pred_scaled = model.predict(X_test)

    assert dm.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, dm.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, dm.target_scaler)
    return compute_metrics(y_true, y_pred)


# Dispatch table
MODEL_RUNNERS = {
    "lstm": run_lstm,
    "cnn_lstm": run_cnn_lstm,
    "sarima": run_sarima,
    "xgboost": run_xgboost,
    "catboost": run_catboost,
    # hmm_lstm and tft handled specially (need raw DFs)
}


def run_single_experiment(
    variant_name: str,
    model_name: str,
    horizon: int,
    cached_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    epochs_override: int | None = None,
) -> dict:
    """Run a single variant x model x horizon experiment.

    Args:
        variant_name: Preprocessing variant key.
        model_name: Model key.
        horizon: Forecast horizon in hours.
        cached_splits: Pre-computed (train, val, test) DataFrames to avoid
            reloading raw XLS files for every run.
        epochs_override: Override max epochs if set.

    Returns:
        Metrics dictionary.
    """
    config = load_full_config(variant_name, model_name)
    config["data"]["forecast_horizon_hours"] = horizon

    if epochs_override is not None:
        config["training"]["max_epochs"] = epochs_override

    # Direct results into comparison-specific directory
    config["training"]["results_dir"] = f"results/preprocessing_comparison/per_variant/{variant_name}"

    run_name = f"{variant_name}_{model_name}_h{horizon}"
    seed_everything(config["training"]["seed"])

    # Use cached splits or compute from pipeline
    if cached_splits is not None:
        train_df, val_df, test_df = cached_splits
    else:
        raw_dir = Path(config["data"].get("raw_dir", "data/raw"))
        train_df, val_df, test_df = run_preprocessing_pipeline(
            raw_dir=raw_dir,
            variant_config=config,
        )

    if model_name == "hmm_lstm":
        return run_hmm_lstm(config, train_df.copy(), val_df.copy(), test_df.copy(), run_name)
    elif model_name == "tft":
        return run_tft(config, train_df.copy(), val_df.copy(), test_df.copy(), run_name)
    else:
        # Standard models: build DataModule from pre-split DFs
        dm = _build_datamodule_from_splits(train_df, val_df, test_df, config)
        runner = MODEL_RUNNERS[model_name]
        return runner(config, dm, run_name)


def generate_comparison_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Generate bar chart comparing RMSE across variants and models.

    Args:
        summary_df: DataFrame with columns: variant, model, horizon, rmse, mae, r2, mape.
        output_path: Path to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    horizons = sorted(summary_df["horizon"].unique())
    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=(8 * n_horizons, 6), squeeze=False)

    for idx, horizon in enumerate(horizons):
        ax = axes[0, idx]
        subset = summary_df[summary_df["horizon"] == horizon]
        pivot = subset.pivot(index="model", columns="variant", values="rmse")
        pivot.plot(kind="bar", ax=ax, rot=45)
        ax.set_title(f"Test RMSE by Variant - {horizon}h Horizon")
        ax.set_ylabel("RMSE (ppm)")
        ax.set_xlabel("")
        ax.legend(title="Variant", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Comparison plot saved to: {output_path}")


def select_best_variant(summary_df: pd.DataFrame) -> str:
    """Select the best preprocessing variant based on median RMSE.

    Args:
        summary_df: Full results DataFrame.

    Returns:
        Name of the best variant.
    """
    median_rmse = summary_df.groupby("variant")["rmse"].median()
    sorted_variants = median_rmse.sort_values()  # type: ignore[call-overload]

    print("\n  Median RMSE by variant:")
    for variant, rmse in sorted_variants.items():
        print(f"    {variant}: {rmse:.2f} ppm")

    best: str = str(sorted_variants.index[0])
    second: str | None = str(sorted_variants.index[1]) if len(sorted_variants) > 1 else None

    # If top 2 are within 5%, prefer the simpler one
    if second is not None:
        ratio = sorted_variants.iloc[1] / sorted_variants.iloc[0]
        if ratio < 1.05:
            # Prefer simpler variant (A < B < C < D)
            simplicity_order = ["preproc_A", "preproc_B", "preproc_C", "preproc_D"]
            for v in simplicity_order:
                if v in [best, second]:
                    best = v
                    break

    print(f"\n  Best variant: {best} (median RMSE: {median_rmse[best]:.2f})")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Preprocessing variant comparison (7 models x 4 variants x 2 horizons)"
    )
    parser.add_argument(
        "--variants", nargs="+",
        default=list(VARIANT_CONFIGS.keys()),
        choices=list(VARIANT_CONFIGS.keys()),
        help="Variants to compare (default: all 4)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Models to run (default: all 7)",
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int,
        default=[1, 24],
        help="Forecast horizons in hours (default: 1 24)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs for all models",
    )
    args = parser.parse_args()

    total_runs = len(args.variants) * len(args.horizons) * len(args.models)

    # Resume support: load already-completed runs from previous (interrupted) execution
    completed = _load_completed_runs(SUMMARY_CSV)
    skipped = 0

    print(f"\n{'='*70}")
    print(f"  PHASE 3: PREPROCESSING COMPARISON")
    print(f"  Variants: {args.variants}")
    print(f"  Models: {args.models}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Total runs: {total_runs}")
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed - will skip them")
    print(f"{'='*70}\n")

    import gc

    run_count = 0
    # Cache pipeline results per variant to avoid reloading XLS files
    pipeline_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for variant in args.variants:
        # Pre-compute splits for this variant (once, reused across models/horizons)
        if variant not in pipeline_cache:
            print(f"\n  Loading pipeline for {variant}...")
            _cfg = load_full_config(variant, args.models[0])
            raw_dir = Path(_cfg["data"].get("raw_dir", "data/raw"))
            pipeline_cache[variant] = run_preprocessing_pipeline(
                raw_dir=raw_dir, variant_config=_cfg,
            )
            print(f"  Pipeline loaded: train={len(pipeline_cache[variant][0])}, "
                  f"val={len(pipeline_cache[variant][1])}, "
                  f"test={len(pipeline_cache[variant][2])} rows")

        for horizon in args.horizons:
            for model_name in args.models:
                run_count += 1
                label = f"[{run_count}/{total_runs}]"

                # Skip runs that already succeeded in a previous execution
                if (variant, model_name, horizon) in completed:
                    skipped += 1
                    print(f"  {label} {variant} | {model_name} | h={horizon} - SKIPPED (already done)")
                    continue

                print(f"\n{'-'*60}")
                print(f"  {label} {variant} | {model_name} | h={horizon}")
                print(f"{'-'*60}")

                t0 = time.time()
                try:
                    metrics = run_single_experiment(
                        variant_name=variant,
                        model_name=model_name,
                        horizon=horizon,
                        cached_splits=pipeline_cache[variant],
                        epochs_override=args.epochs,
                    )
                    elapsed = time.time() - t0

                    row = {
                        "variant": variant,
                        "model": model_name,
                        "horizon": horizon,
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "r2": metrics["r2"],
                        "mape": metrics["mape"],
                        "elapsed_s": elapsed,
                        "status": "OK",
                    }
                    print(f"  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
                          f"R2={metrics['r2']:.4f}  ({elapsed:.1f}s)")

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  FAILED: {e} ({elapsed:.1f}s)")
                    row = {
                        "variant": variant,
                        "model": model_name,
                        "horizon": horizon,
                        "rmse": float("nan"),
                        "mae": float("nan"),
                        "r2": float("nan"),
                        "mape": float("nan"),
                        "elapsed_s": elapsed,
                        "status": str(e),
                    }

                # Write each result immediately so progress survives interruptions
                _append_result_row(SUMMARY_CSV, row)

                # Free GPU memory between runs to prevent OOM hangs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Reload full results from disk (previous + current runs combined)
    output_dir = Path("results/preprocessing_comparison")
    summary_path = output_dir / "summary.csv"
    summary_df = pd.read_csv(summary_path)

    if skipped:
        print(f"\n  Skipped {skipped} already-completed runs")
    print(f"  All results in: {summary_path}")

    # Print results table
    print(f"\n{'='*70}")
    print("  PREPROCESSING COMPARISON RESULTS")
    print(f"{'='*70}\n")

    for horizon in args.horizons:
        print(f"\n  --- Horizon: {horizon}h ---")
        subset = summary_df[summary_df["horizon"] == horizon]
        if subset.empty:
            continue

        pivot = subset.pivot(index="model", columns="variant", values="rmse")
        print(pivot.to_string())
        print()

    # Select best variant
    valid_results = summary_df[summary_df["status"] == "OK"]
    if not valid_results.empty:
        best = select_best_variant(valid_results)  # type: ignore[arg-type]

        # Save best variant selection
        selection = {
            "best_variant": best,
            "selection_criterion": "median_RMSE",
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "best_variant.json", "w") as f:
            json.dump(selection, f, indent=2)

        # Generate comparison plot
        generate_comparison_plot(valid_results, output_dir / "comparison_plot.png")  # type: ignore[arg-type]

    print(f"\n{'='*70}")
    print("  PHASE 3 COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
