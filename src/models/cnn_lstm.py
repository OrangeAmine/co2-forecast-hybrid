"""CNN-LSTM hybrid model for time series forecasting.

The CNN layers extract local temporal patterns from the input sequences,
and the LSTM captures long-range dependencies from the CNN's learned
representations.

Conv1D Equation:
    For each output channel c_out and temporal position t:
    y(c_out, t) = bias(c_out) + Σ_{c_in} Σ_{k=0}^{K-1} W(c_out, c_in, k) · x(c_in, t + k - pad)
    where K is the kernel size and pad = (K-1)//2 for 'same' padding.

    With 'same' padding, temporal dimension is preserved:
    L_out = L_in  (when stride=1 and pad=(K-1)//2)

MaxPool1D Equation:
    y(c, t) = max_{k=0}^{P-1} x(c, t·P + k)
    where P is the pool size. Reduces temporal dimension by factor P:
    L_out = ⌊L_in / P⌋

BatchNorm1D (Ioffe & Szegedy, 2015):
    y = (x - E[x]) / sqrt(Var[x] + ε) · γ + β
    where γ and β are learned affine parameters.

LSTM Gate Equations: see lstm.py module docstring.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMForecaster(pl.LightningModule):
    """Hybrid CNN-LSTM model for time series forecasting.

    Architecture:
        Input (batch, lookback, features)
        -> Permute to (batch, features, lookback) for Conv1D
        -> Conv1D blocks with BatchNorm and ReLU
        -> MaxPool1D
        -> Permute back to (batch, seq_len, channels)
        -> LSTM layers
        -> Last hidden state
        -> FC head with ReLU and Dropout
        -> Output (batch, output_size)

    Args:
        config: Merged configuration dictionary.
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
            input_size = len(data_cfg["feature_columns"]) + 1
        if output_size is None:
            sph = data_cfg["samples_per_hour"]
            output_size = data_cfg["forecast_horizon_hours"] * sph

        self.save_hyperparameters(ignore=["config"])

        # CNN block
        cnn_channels = model_cfg["cnn_channels"]  # e.g., [32, 64]
        cnn_kernels = model_cfg["cnn_kernel_sizes"]  # e.g., [7, 5]

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_channels = input_size
        for out_channels, kernel_size in zip(cnn_channels, cnn_kernels):
            padding = (kernel_size - 1) // 2
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        # MaxPool1d: y(c, t) = max_{k=0}^{P-1} x(c, t·P + k)
        # Reduces temporal dimension: L_out = ⌊L_in / P⌋
        self.pool = nn.MaxPool1d(kernel_size=model_cfg["cnn_pool_size"])
        self.cnn_dropout = nn.Dropout(model_cfg.get("cnn_dropout", 0.1))

        # Guard: verify sequence length survives convolution + pooling
        # With 'same' padding, conv preserves length; pool divides by pool_size
        sph = data_cfg["samples_per_hour"]
        lookback_steps = data_cfg["lookback_hours"] * sph
        pooled_length = lookback_steps // model_cfg["cnn_pool_size"]
        if pooled_length < 1:
            raise ValueError(
                f"Sequence length after pooling is {pooled_length} (< 1). "
                f"lookback_steps={lookback_steps}, pool_size={model_cfg['cnn_pool_size']}. "
                f"Reduce pool_size or increase lookback_hours."
            )

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=model_cfg["lstm_hidden_size"],
            num_layers=model_cfg["lstm_num_layers"],
            dropout=model_cfg["lstm_dropout"] if model_cfg["lstm_num_layers"] > 1 else 0.0,
            batch_first=True,
        )

        # FC head
        fc_hidden = model_cfg["fc_hidden_size"]
        self.fc1 = nn.Linear(model_cfg["lstm_hidden_size"], fc_hidden)
        self.fc_dropout = nn.Dropout(model_cfg["lstm_dropout"])
        assert output_size is not None
        self.fc2 = nn.Linear(fc_hidden, output_size)

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN -> LSTM -> FC.

        Args:
            x: Input tensor of shape (batch, lookback, features).

        Returns:
            Predictions of shape (batch, output_size).
        """
        # x: (batch, lookback, features)
        # Conv1d expects (batch, channels, seq_len) — permute features to channel dim
        x = x.permute(0, 2, 1)  # -> (batch, features, lookback)

        # Conv1D blocks with BatchNorm and ReLU activation:
        #   y(c_out, t) = ReLU(BN(Σ_{c_in} Σ_k W(c_out, c_in, k) · x(c_in, t+k-pad) + b))
        # With 'same' padding, temporal dimension L is preserved after each conv
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))
        # x: (batch, cnn_channels[-1], lookback)

        # MaxPool1d: reduces temporal dim by pool_size
        # y(c, t) = max_{k=0}^{P-1} x(c, t·P + k)
        x = self.pool(x)   # -> (batch, cnn_channels[-1], lookback // pool_size)
        x = self.cnn_dropout(x)

        # Back to (batch, seq_len, channels) for LSTM
        x = x.permute(0, 2, 1)  # -> (batch, lookback // pool_size, cnn_channels[-1])

        # LSTM processes the pooled sequence; gate equations applied at each step
        # (see lstm.py module docstring for f_t, i_t, g_t, o_t, c_t, h_t)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, lstm_hidden_size)
        out = h_n[-1]  # Last layer's final hidden state: (batch, lstm_hidden_size)

        # FC head: two-layer MLP for final projection
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

    def configure_optimizers(self) -> dict:  # type: ignore[override]
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
            optimizer = self.optimizers()
            if not isinstance(optimizer, list):
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
