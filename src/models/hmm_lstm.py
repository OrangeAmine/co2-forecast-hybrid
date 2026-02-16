"""HMM-LSTM hybrid model for time series forecasting.

Two-stage approach:
1. HMMRegimeDetector identifies latent regime states using Gaussian HMM
2. HMMLSTMForecaster uses regime posterior probabilities as additional
   features for LSTM-based prediction

HMM Equations (Gaussian Hidden Markov Model):
    Transition:   P(z_t = j | z_{t-1} = i) = A_{ij}     (state transition matrix)
    Emission:     P(x_t | z_t = k) = N(x_t; μ_k, Σ_k)   (Gaussian per state)
    Forward:      α_t(j) = P(x_t | z_t=j) · Σ_i α_{t-1}(i) · A_{ij}
    Posterior:     γ_t(k) = P(z_t = k | x_{1:T})          (via forward-backward)

    Parameters (μ_k, Σ_k, A) are estimated via Expectation-Maximization (EM).
    The posterior probabilities γ_t(k) are appended as features for the LSTM.

LSTM Gate Equations: see lstm.py module docstring.
"""

import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """Gaussian HMM for identifying latent regime states in sensor data.

    Fits a Hidden Markov Model on training data to discover latent states
    (e.g., low/medium/high CO2 regimes). Produces posterior state
    probabilities that augment the feature set for downstream LSTM.

    Args:
        n_states: Number of hidden states.
        covariance_type: HMM covariance type ("full", "diag", "tied", "spherical").
        n_iter: Maximum EM iterations.
        hmm_features: Column names used for HMM training.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        hmm_features: list[str] | None = None,
    ) -> None:
        self.n_states = n_states
        self.hmm_features = hmm_features or ["CO2", "Noise", "TemperatureExt"]
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
        )
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit HMM on training data (unscaled).

        Args:
            train_df: Training DataFrame with HMM feature columns.

        Returns:
            self for method chaining.
        """
        data = train_df[self.hmm_features].values
        self.hmm.fit(data)
        self._is_fitted = True

        logger.info(f"HMM fitted with {self.n_states} states")
        logger.info(f"  Converged: {self.hmm.monitor_.converged}")
        logger.info(f"  Final log-likelihood: {self.hmm.score(data):.2f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime state posterior probabilities.

        Args:
            df: DataFrame with the HMM feature columns.

        Returns:
            Array of shape (n_samples, n_states) with posterior
            probabilities for each state at each timestep.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before calling predict_proba")

        data = df[self.hmm_features].values
        return self.hmm.predict_proba(data)

    def predict_states(self, df: pd.DataFrame) -> np.ndarray:
        """Predict most likely regime state sequence (Viterbi).

        Args:
            df: DataFrame with the HMM feature columns.

        Returns:
            Array of shape (n_samples,) with integer state labels.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before calling predict_states")

        data = df[self.hmm_features].values
        return self.hmm.predict(data)


class HMMLSTMForecaster(pl.LightningModule):
    """LSTM forecaster augmented with HMM regime state information.

    Architecture identical to LSTMForecaster but with additional input
    features from HMM posterior probabilities.

    Architecture:
        Input (batch, lookback, features + n_hmm_states)
        -> LSTM (multi-layer with dropout)
        -> last hidden state h_n[-1]
        -> Linear -> ReLU -> Dropout -> Linear
        -> Output (batch, output_size)

    Args:
        config: Merged configuration dictionary.
        input_size: Number of input features (including HMM states).
        output_size: Number of output steps. If None, inferred from config.
    """

    def __init__(
        self,
        config: dict,
        input_size: int | None = None,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        data_cfg = config["data"]

        if input_size is None:
            # features + target + hmm states
            n_base = len(data_cfg["feature_columns"]) + 1
            input_size = n_base + model_cfg["hmm_n_states"]
        if output_size is None:
            sph = data_cfg["samples_per_hour"]
            output_size = data_cfg["forecast_horizon_hours"] * sph

        self.save_hyperparameters(ignore=["config"])

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=model_cfg["lstm_hidden_size"],
            num_layers=model_cfg["lstm_num_layers"],
            dropout=model_cfg["lstm_dropout"] if model_cfg["lstm_num_layers"] > 1 else 0.0,
            batch_first=True,
        )

        fc_hidden = model_cfg["fc_hidden_size"]
        self.fc1 = nn.Linear(model_cfg["lstm_hidden_size"], fc_hidden)
        self.fc_dropout = nn.Dropout(model_cfg["lstm_dropout"])
        self.fc2 = nn.Linear(fc_hidden, output_size)

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, lookback, features + hmm_states).
                The last n_hmm_states columns are γ_t(k) = P(z_t=k | x_{1:T}),
                the HMM posterior probabilities providing regime context.

        Returns:
            Predictions of shape (batch, output_size).
        """
        # x: (batch, lookback, original_features + target + hmm_state_probs)
        # LSTM processes the full augmented sequence; gate equations (see lstm.py):
        #   f_t, i_t, o_t = σ(...), g_t = tanh(...), c_t = f_t⊙c_{t-1} + i_t⊙g_t
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, lstm_hidden_size)
        out = h_n[-1]  # Last layer's final hidden state: (batch, lstm_hidden_size)

        # FC head: two-layer MLP
        out = F.relu(self.fc1(out))   # -> (batch, fc_hidden_size)
        out = self.fc_dropout(out)
        out = self.fc2(out)           # -> (batch, output_size)

        return out

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> dict:
        """Configure Adam optimizer with optional warm-up + ReduceLROnPlateau."""
        training_cfg = self.config["training"]
        self._warmup_epochs = training_cfg.get("warmup_epochs", 0)
        self._target_lr = training_cfg["learning_rate"]

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg.get("weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_cfg.get("scheduler_factor", 0.5),
            patience=training_cfg.get("scheduler_patience", 5),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_start(self) -> None:
        """Linear LR warm-up during the first ``warmup_epochs`` epochs."""
        warmup = getattr(self, "_warmup_epochs", 0)
        if warmup > 0 and self.current_epoch < warmup:
            scale = (self.current_epoch + 1) / warmup
            new_lr = self._target_lr * scale
            for pg in self.optimizers().param_groups:
                pg["lr"] = new_lr
