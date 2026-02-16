"""Train the Temporal Fusion Transformer (TFT) for CO2 forecasting.

Uses the pytorch-forecasting library's TemporalFusionTransformer.
Uses lightning.pytorch (unified package) for compatibility with
pytorch-forecasting v1.6+.

Usage:
    python scripts/train_tft.py
    python scripts/train_tft.py --horizon 1
    python scripts/train_tft.py --horizon 24
"""

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Fix Windows unicode issues
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use lightning.pytorch (unified) for TFT compatibility
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.preprocessing import load_and_parse_data
from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.visualization import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_scatter,
    plot_training_curves,
)
from src.models.tft import build_tft_model, create_tft_datasets, prepare_tft_dataframe
from src.utils.config import load_config
from src.utils.seed import seed_everything


def create_tft_trainer(config: dict, model_name: str) -> tuple[pl.Trainer, Path]:
    """Create a Lightning Trainer for TFT using the unified lightning package."""
    training_cfg = config["training"]

    # Enforce GPU availability when configured
    accelerator = training_cfg.get("accelerator", "auto")
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required but CUDA is not available. "
            "Install CUDA-enabled PyTorch or set accelerator to 'auto'."
        )

    results_dir = Path(training_cfg["results_dir"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    run_dir = results_dir / run_name

    early_stop = EarlyStopping(
        monitor="val_loss", patience=training_cfg["patience"],
        mode="min", verbose=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb_logs")

    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        precision=training_cfg.get("precision", 32),
        callbacks=[early_stop, checkpoint, lr_monitor, progress_bar],
        gradient_clip_val=training_cfg.get("gradient_clip_val", 0.1),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
        logger=tb_logger,
        deterministic=True,
    )
    return trainer, run_dir


def train_single_horizon(config: dict) -> None:
    """Train TFT for a single forecast horizon."""
    horizon = config["data"]["forecast_horizon_hours"]
    exp_name = config.get("experiment", {}).get("name", "")
    base_name = f"TFT_h{horizon}"
    model_name = f"{exp_name}_{base_name}" if exp_name else base_name

    seed_everything(config["training"]["seed"])

    # Load and prepare data
    csv_path = Path(config["data"]["processed_csv"])
    df = load_and_parse_data(csv_path, config["data"]["datetime_column"])
    df = prepare_tft_dataframe(df, config)

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Lookback: {config['data']['lookback_hours']}h")
    print(f"  Horizon: {horizon}h")
    print(f"  Total records: {len(df)}")
    print(f"{'='*60}\n")

    # Create TFT datasets
    training_data, validation_data, test_data, df = create_tft_datasets(df, config)

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    train_dl = training_data.to_dataloader(
        train=True, batch_size=batch_size,
        num_workers=config["training"].get("num_workers", 0),
    )
    val_dl = validation_data.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=config["training"].get("num_workers", 0),
    )
    test_dl = test_data.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=config["training"].get("num_workers", 0),
    )

    # Build TFT model
    tft = build_tft_model(training_data, config)

    # Trainer (using lightning.pytorch for TFT compatibility)
    trainer, run_dir = create_tft_trainer(config, model_name=model_name)

    # Train
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Test
    trainer.test(tft, dataloaders=test_dl, ckpt_path="best")

    # Predictions — use best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_tft = tft.__class__.load_from_checkpoint(best_ckpt)

    predictions = best_tft.predict(test_dl, mode="prediction", return_x=False)
    y_pred = torch.cat([p for p in predictions], dim=0).numpy()

    # Get actual values from test dataloader
    actuals = torch.cat([y[0] for x, y in iter(test_dl)], dim=0).numpy()

    # Compute metrics (TFT handles its own scaling internally)
    metrics = compute_metrics(actuals.ravel(), y_pred.ravel())
    save_metrics(metrics, model_name, run_dir / "metrics.json",
                 experiment_info=config.get("experiment"))

    np.savez(run_dir / "predictions.npz", y_true=actuals, y_pred=y_pred)

    # Plots
    plots_dir = run_dir / "plots"
    plot_predictions_vs_actual(actuals, y_pred, model_name, plots_dir / "predictions.png")
    plot_scatter(actuals, y_pred, model_name, plots_dir / "scatter.png")
    plot_residuals(actuals, y_pred, model_name, plots_dir / "residuals.png")
    plot_training_curves(run_dir / "tb_logs", model_name, plots_dir / "training_curves.png")

    print(f"\nAll outputs saved to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / "tft.yaml"),
    ]
    if args.experiment:
        config_files.append(args.experiment)

    base_config = load_config(config_files)

    for horizon in args.horizon:
        # Full deep copy prevents cross-horizon mutation of nested dicts
        config = copy.deepcopy(base_config)
        config["data"]["forecast_horizon_hours"] = horizon

        if args.epochs:
            config["training"]["max_epochs"] = args.epochs
        if args.lookback:
            config["data"]["lookback_hours"] = args.lookback

        train_single_horizon(config)


if __name__ == "__main__":
    main()
