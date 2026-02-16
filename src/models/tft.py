"""Temporal Fusion Transformer (TFT) wrapper using pytorch-forecasting.

Provides helper functions for building TFT-compatible datasets and
instantiating the TFT model from the pytorch-forecasting library.

Includes auto-classification of features into TFT's known vs unknown
reals, enabling automatic adaptation to experiment-specific feature sets.

TFT Key Equations (Lim et al., 2021 — "Temporal Fusion Transformers
for Interpretable Multi-horizon Time Series Forecasting"):

1. Gated Residual Network (GRN) — core building block:
    η₁ = W₁ · x + b₁                     (linear projection)
    η₂ = W₂ · ELU(η₁) + b₂               (nonlinear transformation)
    GRN(x) = LayerNorm(x + GLU(η₂))       (skip connection + gating)

    GLU(γ) = σ(W₃ · γ + b₃) ⊙ (W₄ · γ + b₄)   (Gated Linear Unit)

2. Variable Selection Network (VSN):
    v_χ = Softmax(GRN_v(Ξ))               (per-variable importance weights)
    ξ̃_t = Σⱼ v_χ^(j) · GRN_ξ^(j)(ξ_t^(j))  (weighted combination)
    where Ξ is the flattened vector of all transformed inputs.

3. Interpretable Multi-Head Attention:
    Attention(Q, K, V) = Softmax(Q · Kᵀ / √d_k) · V
    MultiHead(Q, K, V) = W_H · [head₁; ...; head_m]
    head_i = Attention(Q · W_Q^i, K · W_K^i, V · W_V^i)

    TFT uses additive aggregation (shared V across heads) for interpretability.

4. Static Enrichment:
    θ(t, n) = GRN_θ(ξ̃_t, c_e)    (enrich temporal features with static context c_e)

5. Temporal Self-Attention + Gated Residual:
    B(t, n) = InterpretableMultiHead(θ(t, n))
    δ(t, n) = LayerNorm(θ(t, n) + GLU(B(t, n)))

6. Position-wise Feed-Forward:
    ψ(t, n) = GRN_ψ(δ(t, n))
    ỹ(t, n) = LayerNorm(δ(t, n) + GLU(ψ(t, n)))  (final gated residual)

7. Quantile Outputs:
    ŷ(q, t, τ) = W_q · ỹ(t, τ) + b_q   (linear projection per quantile q)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, QuantileLoss, RMSE


# Temporal cyclical patterns are deterministic (knowable in advance)
_KNOWN_REAL_PATTERNS = {"_sin", "_cos"}
# The target column is always unknown (stochastic)
_TARGET_COLUMN = "CO2"


def classify_tft_features(
    feature_columns: list[str],
    target: str = "CO2",
) -> tuple[list[str], list[str]]:
    """Auto-classify features into TFT known vs unknown reals.

    Known reals are deterministic temporal features (sin/cos encodings
    for day, year, weekday cycles). Unknown reals are stochastic sensor
    readings, lags, rolling stats, etc.

    Args:
        feature_columns: All feature column names from the data config.
        target: The target column name (always classified as unknown).

    Returns:
        Tuple of (known_reals, unknown_reals).
    """
    known_reals: list[str] = []
    unknown_reals: list[str] = []

    for col in feature_columns:
        if any(pattern in col for pattern in _KNOWN_REAL_PATTERNS):
            known_reals.append(col)
        else:
            unknown_reals.append(col)

    # Target is always unknown
    if target not in unknown_reals:
        unknown_reals.insert(0, target)

    return known_reals, unknown_reals


def prepare_tft_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare DataFrame for pytorch-forecasting TimeSeriesDataSet.

    Adds required columns:
    - time_idx: monotonically increasing integer time index
    - group: constant group identifier (single time series)

    If the model config has an ``auto_classify_features`` flag or if the
    experiment config provides extra features not in the hardcoded TFT
    lists, auto-classification is used to update known/unknown reals.

    Args:
        df: Processed DataFrame with datetime column and features.
        config: Configuration dictionary.

    Returns:
        DataFrame ready for TimeSeriesDataSet construction.
    """
    df = df.copy()
    datetime_col = config["data"]["datetime_column"]

    # Ensure sorted by time
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # Add time_idx (monotonic integer index)
    df["time_idx"] = np.arange(len(df))

    # Add constant group identifier (single series)
    df["group"] = "0"

    # Auto-classify features from the data config if experiment provides them
    data_features = config["data"].get("feature_columns", [])
    model_cfg = config["model"]
    target = model_cfg.get("target", "CO2")

    if data_features:
        known_reals, unknown_reals = classify_tft_features(data_features, target)
        model_cfg["time_varying_known_reals"] = known_reals
        model_cfg["time_varying_unknown_reals"] = unknown_reals

    # Ensure target and features are float
    all_cols = model_cfg["time_varying_known_reals"] + model_cfg["time_varying_unknown_reals"]
    for col in all_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def create_tft_datasets(
    df: pd.DataFrame,
    config: dict,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    """Create train/val/test TimeSeriesDataSet objects for TFT.

    Args:
        df: Full prepared DataFrame with time_idx and group columns.
        config: Configuration dictionary with model and data keys.

    Returns:
        Tuple of (training_dataset, validation_dataset, test_dataset, df).
    """
    data_cfg = config["data"]
    model_cfg = config["model"]

    sph = data_cfg["samples_per_hour"]
    max_encoder_length = data_cfg["lookback_hours"] * sph
    max_prediction_length = data_cfg["forecast_horizon_hours"] * sph

    n = len(df)
    train_end = int(n * data_cfg["train_ratio"])
    val_end = train_end + int(n * data_cfg["val_ratio"])

    # Training dataset
    train_mask: pd.Series = df["time_idx"] <= train_end  # type: ignore[assignment]
    training_data = TimeSeriesDataSet(
        df.loc[train_mask].reset_index(drop=True),
        time_idx="time_idx",
        target=model_cfg["target"],
        group_ids=model_cfg["group_ids"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=model_cfg["time_varying_known_reals"],
        time_varying_unknown_reals=model_cfg["time_varying_unknown_reals"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Validation dataset (shares encoders/scalers with training)
    val_mask: pd.Series = (df["time_idx"] > train_end) & (df["time_idx"] <= val_end)  # type: ignore[assignment]
    validation_data = TimeSeriesDataSet.from_dataset(
        training_data,
        df.loc[val_mask].reset_index(drop=True),
        stop_randomization=True,
    )

    # Test dataset
    test_mask: pd.Series = df["time_idx"] > val_end  # type: ignore[assignment]
    test_data = TimeSeriesDataSet.from_dataset(
        training_data,
        df.loc[test_mask].reset_index(drop=True),
        stop_randomization=True,
    )

    return training_data, validation_data, test_data, df


def build_tft_model(
    training_dataset: TimeSeriesDataSet,
    config: dict,
) -> TemporalFusionTransformer:
    """Build TFT model from pytorch-forecasting.

    Args:
        training_dataset: pytorch-forecasting TimeSeriesDataSet for training.
        config: Configuration dictionary.

    Returns:
        Configured TemporalFusionTransformer instance.
    """
    model_cfg = config["model"]
    training_cfg = config["training"]

    # Loss function selection:
    #   MAE:          L = (1/N) Σ |y - ŷ|
    #   QuantileLoss: L = (1/N) Σ_q Σ_t max(q·(y-ŷ), (q-1)·(y-ŷ))
    #   RMSE:         L = √((1/N) Σ (y - ŷ)²)
    loss_fn = MAE()
    if model_cfg.get("loss") == "quantile":
        loss_fn = QuantileLoss()
    elif model_cfg.get("loss") == "rmse":
        loss_fn = RMSE()

    # TFT internally creates a ReduceLROnPlateau scheduler.
    # scheduler_patience controls how many epochs of no val_loss improvement
    # before the LR is reduced. This is separate from early stopping patience.
    scheduler_patience = training_cfg.get("scheduler_patience", 5)

    tft: TemporalFusionTransformer = TemporalFusionTransformer.from_dataset(  # type: ignore[assignment]
        training_dataset,
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        hidden_continuous_size=model_cfg["hidden_continuous_size"],
        loss=loss_fn,
        learning_rate=training_cfg["learning_rate"],
        reduce_on_plateau_patience=scheduler_patience,
    )

    return tft
