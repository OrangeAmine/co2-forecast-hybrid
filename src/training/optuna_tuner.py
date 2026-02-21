"""Optuna hyperparameter tuning integration for CO2 forecasting models.

Provides a generic tuning framework that works with any PyTorch Lightning
model in the project. Uses MedianPruner for early stopping of unpromising
trials and PyTorchLightningPruningCallback for per-epoch pruning.

Typical search spaces:
    - Learning rate: 1e-4 to 1e-2 (log scale)
    - Hidden size: 64, 128, 256, 512 (categorical)
    - Num layers: 1-3 (integer)
    - Dropout: 0.1-0.5 (float)
    - Lookback hours: 12-72 (integer)
    - Batch size: 32, 64, 128 (categorical)

Usage:
    tuner = OptunaTuner(
        base_config=config,
        model_class=LSTMForecaster,
        search_space_fn=lstm_search_space,
        n_trials=100,
    )
    best_params, study = tuner.run()
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable

import optuna
import pytorch_lightning as pl
import torch
from optuna_integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from src.data.datamodule import CO2DataModule
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _set_nested(config: dict, key_path: str, value: Any) -> None:
    """Set a nested config value using dot-separated path.

    Example: _set_nested(cfg, "model.hidden_size", 256)
    sets cfg["model"]["hidden_size"] = 256.

    Args:
        config: Configuration dictionary to modify in-place.
        key_path: Dot-separated path to the key (e.g., "model.hidden_size").
        value: Value to set.
    """
    keys = key_path.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


class OptunaTuner:
    """Generic Optuna hyperparameter tuner for PyTorch Lightning models.

    Runs Bayesian optimization over a user-defined search space, training
    and evaluating the model on each trial. Bad trials are pruned early
    via MedianPruner (a trial is pruned if its intermediate val_loss is
    worse than the median of completed trials at the same epoch).

    Args:
        base_config: Base config dict (deep-copied per trial).
        model_class: PyTorch Lightning module class to instantiate.
        search_space_fn: Callable(trial) -> dict mapping dot-separated
            config paths to suggested values.
        n_trials: Number of Optuna trials to run.
        timeout: Maximum total time in seconds (None = unlimited).
        study_name: Name for the Optuna study.
        storage: Optuna storage URL (None = in-memory).
        results_dir: Directory to save best params and study stats.
    """

    def __init__(
        self,
        base_config: dict,
        model_class: type,
        search_space_fn: Callable[[optuna.Trial], dict[str, Any]],
        n_trials: int = 100,
        timeout: int | None = None,
        study_name: str = "co2_tuning",
        storage: str | None = None,
        results_dir: Path | None = None,
    ) -> None:
        self.base_config = base_config
        self.model_class = model_class
        self.search_space_fn = search_space_fn
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage
        self.results_dir = results_dir or Path("results/optuna")

    def _objective(self, trial: optuna.Trial) -> float:
        """Single trial objective function.

        Creates a fresh config with suggested hyperparameters, trains the
        model, and returns the best validation loss.

        Args:
            trial: Optuna trial object for parameter suggestions.

        Returns:
            Best validation loss (lower is better).
        """
        config = copy.deepcopy(self.base_config)
        suggested_params = self.search_space_fn(trial)

        for key_path, value in suggested_params.items():
            _set_nested(config, key_path, value)

        seed_everything(config["training"]["seed"])

        # Create data module
        dm = CO2DataModule(config)
        dm.setup()

        # Create model
        model = self.model_class(config)

        # Pruning callback: reports val_loss to Optuna after each epoch
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss"
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=config["training"].get("patience", 10),
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=config["training"]["max_epochs"],
            accelerator=config["training"].get("accelerator", "auto"),
            devices=config["training"].get("devices", 1),
            callbacks=[early_stop, pruning_callback],
            gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
            enable_progress_bar=False,
            logger=False,
            deterministic=True,
        )

        trainer.fit(model, datamodule=dm)

        # Clean up GPU memory between trials
        del model, dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return trainer.callback_metrics["val_loss"].item()

    def run(self) -> tuple[dict[str, Any], optuna.Study]:
        """Run the hyperparameter search.

        Returns:
            Tuple of (best_params dict, Optuna Study object).
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        # Save results
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }

        output_path = self.results_dir / f"{self.study_name}_best_params.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Best params saved to {output_path}")
        logger.info(f"Best val_loss: {study.best_value:.6f}")
        logger.info(f"Trials: {results['n_complete']} complete, {results['n_pruned']} pruned")

        return study.best_params, study
