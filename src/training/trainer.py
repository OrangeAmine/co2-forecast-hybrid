"""PyTorch Lightning Trainer factory with callbacks configuration."""

import os
from datetime import datetime
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Fix Windows unicode issues with rich progress bar
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def create_trainer(config: dict, model_name: str) -> tuple[pl.Trainer, Path]:
    """Create a configured PyTorch Lightning Trainer.

    Sets up EarlyStopping, ModelCheckpoint, LearningRateMonitor,
    and TensorBoard logging with a run-specific directory.

    Args:
        config: Merged configuration dictionary.
        model_name: Name of the model (used for logging directory).

    Returns:
        Tuple of (pl.Trainer, run_dir Path).
    """
    training_cfg = config["training"]

    # Enforce GPU availability when configured
    accelerator = training_cfg.get("accelerator", "auto")
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required but CUDA is not available. "
            "Install CUDA-enabled PyTorch or set accelerator to 'auto'."
        )

    results_dir = Path(training_cfg["results_dir"])

    # Create run-specific directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    run_dir = results_dir / run_name

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=training_cfg["patience"],
        mode="min",
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Use TQDM progress bar to avoid Windows rich unicode issues
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="tb_logs",
    )

    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        precision=training_cfg.get("precision", 32),
        callbacks=[early_stop, checkpoint, lr_monitor, progress_bar],
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
        logger=tb_logger,
        deterministic=True,
    )

    return trainer, run_dir
