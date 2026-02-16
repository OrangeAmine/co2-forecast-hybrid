"""LSTM baseline model for time series forecasting.

LSTM Gate Equations (Hochreiter & Schmidhuber, 1997):
    At each timestep t, given input x_t and previous hidden state h_{t-1}:

    Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    Candidate:    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
    Cell update:  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    Hidden state: h_t = o_t ⊙ tanh(c_t)

    where σ is the sigmoid function and ⊙ is element-wise multiplication.
    The forget gate controls how much of the old cell state to retain,
    the input gate controls how much of the new candidate to add, and
    the output gate controls how much of the cell state to expose.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LSTMForecaster(pl.LightningModule):
    """Standard multi-layer LSTM for time series forecasting.

    Architecture:
        Input (batch, lookback, features)
        -> LSTM (multi-layer with dropout)
        -> last hidden state h_n[-1] (batch, hidden_size)
        -> Linear(hidden_size, output_size)
        -> Output (batch, output_size)

    Args:
        config: Merged configuration dictionary with 'model' and 'training' keys.
        input_size: Number of input features. If None, inferred from config.
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
            # features + target
            input_size = len(data_cfg["feature_columns"]) + 1
        if output_size is None:
            sph = data_cfg["samples_per_hour"]
            output_size = data_cfg["forecast_horizon_hours"] * sph

        self.save_hyperparameters(ignore=["config"])

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"] if model_cfg["num_layers"] > 1 else 0.0,
            batch_first=True,
            bidirectional=model_cfg.get("bidirectional", False),
        )

        lstm_output_size = model_cfg["hidden_size"]
        if model_cfg.get("bidirectional", False):
            lstm_output_size *= 2

        self.fc = nn.Linear(lstm_output_size, output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, lookback, features).

        Returns:
            Predictions of shape (batch, output_size).
        """
        # x: (batch, lookback, features)
        # The LSTM internally computes at each timestep t:
        #   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   — forget gate
        #   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   — input gate
        #   g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) — candidate cell
        #   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       — cell state update
        #   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   — output gate
        #   h_t = o_t ⊙ tanh(c_t)                   — hidden state
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        if self.config["model"].get("bidirectional", False):
            # Concatenate last forward h_n[-2] and backward h_n[-1] hidden states
            # Result: (batch, hidden_size * 2)
            out = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            # Use last layer's final hidden state: (batch, hidden_size)
            out = h_n[-1]

        # Linear projection: (batch, hidden_size) -> (batch, output_size)
        return self.fc(out)

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
        """Configure Adam optimizer with optional warm-up + ReduceLROnPlateau.

        When warmup_epochs > 0, the LR linearly ramps from ~0 to
        learning_rate over that many epochs before ReduceLROnPlateau
        takes over. This stabilizes early training for large output horizons.
        """
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
        """Linear LR warm-up during the first ``warmup_epochs`` epochs.

        Linearly scales LR from target_lr / warmup_epochs to target_lr.
        After warm-up, ReduceLROnPlateau manages the LR normally.
        """
        warmup = getattr(self, "_warmup_epochs", 0)
        if warmup > 0 and self.current_epoch < warmup:
            # Linear ramp: epoch 0 → lr/warmup, epoch warmup-1 → lr
            scale = (self.current_epoch + 1) / warmup
            new_lr = self._target_lr * scale
            for pg in self.optimizers().param_groups:
                pg["lr"] = new_lr
