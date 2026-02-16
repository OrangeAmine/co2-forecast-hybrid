"""Tests for src/data/datamodule.py — CO2DataModule setup and shapes."""

import numpy as np
import pandas as pd
import pytest

from src.data.datamodule import CO2DataModule
from src.data.preprocessing import chronological_split


class TestCO2DataModuleFromDataFrames:
    """Test the from_dataframes classmethod (used by HMM-LSTM)."""

    def test_setup_creates_datasets(self, sample_co2_df, sample_config):
        """DataModule.from_dataframes should populate train/val/test datasets."""
        train_df, val_df, test_df = chronological_split(sample_co2_df)
        dm = CO2DataModule.from_dataframes(train_df, val_df, test_df, sample_config)
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

    def test_shapes(self, sample_co2_df, sample_config):
        """X and y tensors should have correct dimensions."""
        train_df, val_df, test_df = chronological_split(sample_co2_df)
        dm = CO2DataModule.from_dataframes(train_df, val_df, test_df, sample_config)

        n_features = len(sample_config["data"]["feature_columns"]) + 1  # +1 for target
        lookback = sample_config["data"]["lookback_hours"] * sample_config["data"]["samples_per_hour"]
        horizon = sample_config["data"]["forecast_horizon_hours"] * sample_config["data"]["samples_per_hour"]

        X, y = dm.train_dataset[0]
        assert X.shape == (lookback, n_features)
        assert y.shape == (horizon,)

    def test_scalers_fitted(self, sample_co2_df, sample_config):
        """Feature and target scalers should be fitted."""
        train_df, val_df, test_df = chronological_split(sample_co2_df)
        dm = CO2DataModule.from_dataframes(train_df, val_df, test_df, sample_config)
        assert dm.feature_scaler is not None
        assert dm.target_scaler is not None


class TestCO2DataModuleSetup:
    """Test the standard setup flow with a CSV file."""

    def test_setup_from_csv(self, tmp_csv, sample_config):
        """DataModule should load from CSV and create datasets."""
        sample_config["data"]["processed_csv"] = str(tmp_csv)
        dm = CO2DataModule(sample_config)
        dm.setup(stage="fit")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

    def test_dataloader_creation(self, tmp_csv, sample_config):
        """DataLoaders should be created with correct batch size."""
        sample_config["data"]["processed_csv"] = str(tmp_csv)
        dm = CO2DataModule(sample_config)
        dm.setup(stage="fit")

        train_dl = dm.train_dataloader()
        assert train_dl.batch_size == sample_config["training"]["batch_size"]

    def test_pin_memory_config(self, tmp_csv, sample_config):
        """pin_memory should respect config setting."""
        sample_config["data"]["processed_csv"] = str(tmp_csv)
        sample_config["training"]["pin_memory"] = True
        sample_config["training"]["num_workers"] = 0
        dm = CO2DataModule(sample_config)
        dm.setup(stage="fit")

        train_dl = dm.train_dataloader()
        assert train_dl.pin_memory is True
