"""PyTorch Lightning DataModule for CO2 forecasting."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from src.data.dataset import TimeSeriesDataset
from src.data.preprocessing import (
    apply_scalers,
    chronological_split,
    create_sequences,
    fit_scalers,
)


class CO2DataModule(pl.LightningDataModule):
    """Lightning DataModule for CO2 time series forecasting.

    Handles loading, preprocessing, splitting, scaling, windowing,
    and DataLoader creation. Stores fitted scalers for inverse
    transformation at evaluation time.

    Supports two data loading paths:
        1. Pipeline-based: Uses ``pipeline.run_preprocessing_pipeline()``
           when ``data.pipeline_variant`` is set in config.
        2. CSV-based (legacy): Loads from ``data.processed_csv`` when no
           pipeline variant is specified. Kept for backward compatibility
           with existing training scripts that don't use experiment configs.

    Args:
        config: Merged configuration dictionary containing 'data' and
            'training' keys.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.data_cfg = config["data"]
        self.training_cfg = config["training"]

        self.feature_columns: list[str] = self.data_cfg["feature_columns"]
        self.target_column: str = self.data_cfg["target_column"]

        # Convert hours to steps
        sph = self.data_cfg["samples_per_hour"]
        self.lookback_steps = self.data_cfg["lookback_hours"] * sph
        self.horizon_steps = self.data_cfg["forecast_horizon_hours"] * sph

        self.feature_scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self.target_scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None

        self.train_dataset: TimeSeriesDataset | None = None
        self.val_dataset: TimeSeriesDataset | None = None
        self.test_dataset: TimeSeriesDataset | None = None

        # Store test dates for plotting
        self.test_dates: pd.DatetimeIndex | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data, split, scale, and create sequences.

        Uses the split-aware pipeline when ``data.pipeline_variant`` is set,
        otherwise falls back to CSV loading for backward compatibility.

        Args:
            stage: Lightning stage ('fit', 'validate', 'test', 'predict').
        """
        # Skip if datasets already created (e.g., via from_dataframes)
        if self.train_dataset is not None:
            return

        pipeline_variant = self.data_cfg.get("pipeline_variant")

        if pipeline_variant is not None:
            # New split-aware pipeline path
            self._setup_from_pipeline()
        else:
            # Legacy CSV-loading path (for benchmark and existing scripts)
            self._setup_from_csv()

    def _setup_from_pipeline(self) -> None:
        """Load data via the split-aware preprocessing pipeline."""
        from src.data.pipeline import run_preprocessing_pipeline

        raw_dir = Path(self.data_cfg.get("raw_dir", "data/raw"))
        train_df, val_df, test_df = run_preprocessing_pipeline(
            raw_dir=raw_dir,
            variant_config=self.config,
        )

        self.build_datasets(train_df, val_df, test_df)

    def _setup_from_csv(self) -> None:
        """Load data from a pre-processed CSV file (legacy path)."""
        from src.data.preprocessing import load_and_parse_data

        csv_path = Path(self.data_cfg["processed_csv"])
        df = load_and_parse_data(csv_path, self.data_cfg["datetime_column"])

        train_df, val_df, test_df = chronological_split(
            df,
            self.data_cfg["train_ratio"],
            self.data_cfg["val_ratio"],
            self.data_cfg["test_ratio"],
        )

        self.build_datasets(train_df, val_df, test_df)

    def build_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Fit scalers, scale data, create sequences and datasets.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
        """
        datetime_col = self.data_cfg.get("datetime_column", "datetime")

        # Store test dates for plotting
        if datetime_col in test_df.columns:
            self.test_dates = test_df[datetime_col]

        # Fit scalers on training data
        self.feature_scaler, self.target_scaler = fit_scalers(
            train_df,
            self.feature_columns,
            self.target_column,
            self.data_cfg["scaler_type"],
        )

        # Scale all splits
        train_scaled = apply_scalers(
            train_df, self.feature_scaler, self.target_scaler,
            self.feature_columns, self.target_column,
        )
        val_scaled = apply_scalers(
            val_df, self.feature_scaler, self.target_scaler,
            self.feature_columns, self.target_column,
        )
        test_scaled = apply_scalers(
            test_df, self.feature_scaler, self.target_scaler,
            self.feature_columns, self.target_column,
        )

        # Create sequences
        stride = self.data_cfg.get("stride", 1)

        X_train, y_train = create_sequences(
            train_scaled, self.lookback_steps, self.horizon_steps, stride,
        )
        X_val, y_val = create_sequences(
            val_scaled, self.lookback_steps, self.horizon_steps, stride,
        )
        X_test, y_test = create_sequences(
            test_scaled, self.lookback_steps, self.horizon_steps, stride,
        )

        self.train_dataset = TimeSeriesDataset(X_train, y_train)
        self.val_dataset = TimeSeriesDataset(X_val, y_val)
        self.test_dataset = TimeSeriesDataset(X_test, y_test)

    @classmethod
    def from_dataframes(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: dict,
    ) -> CO2DataModule:
        """Create DataModule from pre-split DataFrames.

        Used by HMM-LSTM where features are augmented before splitting.

        Args:
            train_df: Training DataFrame (with augmented features).
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            config: Merged configuration dictionary.

        Returns:
            Configured CO2DataModule with datasets ready.
        """
        dm = cls(config)

        dm.test_dates = test_df.get(
            config["data"]["datetime_column"],
            pd.Series(dtype="datetime64[ns]"),
        )

        feature_columns = dm.feature_columns
        target_column = dm.target_column

        # Fit scalers on training data
        dm.feature_scaler, dm.target_scaler = fit_scalers(
            train_df, feature_columns, target_column,
            dm.data_cfg["scaler_type"],
        )

        # Scale all splits
        train_scaled = apply_scalers(
            train_df, dm.feature_scaler, dm.target_scaler,
            feature_columns, target_column,
        )
        val_scaled = apply_scalers(
            val_df, dm.feature_scaler, dm.target_scaler,
            feature_columns, target_column,
        )
        test_scaled = apply_scalers(
            test_df, dm.feature_scaler, dm.target_scaler,
            feature_columns, target_column,
        )

        stride = dm.data_cfg.get("stride", 1)

        X_train, y_train = create_sequences(
            train_scaled, dm.lookback_steps, dm.horizon_steps, stride,
        )
        X_val, y_val = create_sequences(
            val_scaled, dm.lookback_steps, dm.horizon_steps, stride,
        )
        X_test, y_test = create_sequences(
            test_scaled, dm.lookback_steps, dm.horizon_steps, stride,
        )

        dm.train_dataset = TimeSeriesDataset(X_train, y_train)
        dm.val_dataset = TimeSeriesDataset(X_val, y_val)
        dm.test_dataset = TimeSeriesDataset(X_test, y_test)

        return dm

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_cfg["batch_size"],
            shuffle=True,
            num_workers=self.training_cfg.get("num_workers", 0),
            pin_memory=self.training_cfg.get("pin_memory", False),
            persistent_workers=self.training_cfg.get("num_workers", 0) > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_cfg["batch_size"],
            shuffle=False,
            num_workers=self.training_cfg.get("num_workers", 0),
            pin_memory=self.training_cfg.get("pin_memory", False),
            persistent_workers=self.training_cfg.get("num_workers", 0) > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_cfg["batch_size"],
            shuffle=False,
            num_workers=self.training_cfg.get("num_workers", 0),
            pin_memory=self.training_cfg.get("pin_memory", False),
            persistent_workers=self.training_cfg.get("num_workers", 0) > 0,
        )

    def save_scalers(self, run_dir: Path) -> None:
        """Persist fitted scalers to disk for reproducible inference.

        Lightning checkpoints save the model weights but not the scalers.
        Without saved scalers, inference would require re-fitting on the
        training split, introducing a reproducibility risk.

        Args:
            run_dir: Run directory where scaler files will be saved.
        """
        scalers_dir = run_dir / "scalers"
        scalers_dir.mkdir(parents=True, exist_ok=True)

        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, scalers_dir / "feature_scaler.joblib")
        if self.target_scaler is not None:
            joblib.dump(self.target_scaler, scalers_dir / "target_scaler.joblib")

        logger.info(f"Scalers saved to {scalers_dir}")

    @staticmethod
    def load_scalers(
        run_dir: Path,
    ) -> tuple[StandardScaler | MinMaxScaler, StandardScaler | MinMaxScaler]:
        """Load previously saved scalers from a run directory.

        Args:
            run_dir: Run directory containing the ``scalers/`` subdirectory.

        Returns:
            Tuple of (feature_scaler, target_scaler).

        Raises:
            FileNotFoundError: If scaler files do not exist.
        """
        scalers_dir = run_dir / "scalers"
        feature_scaler = joblib.load(scalers_dir / "feature_scaler.joblib")
        target_scaler = joblib.load(scalers_dir / "target_scaler.joblib")
        return feature_scaler, target_scaler
