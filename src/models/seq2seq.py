"""Seq2Seq Encoder-Decoder LSTM with Bahdanau attention for multi-step forecasting.

Addresses the failure of direct forecasting at long horizons (e.g., 24h) by
generating predictions autoregressively — each future step is conditioned on
previous predictions via attention over the encoder's output sequence.

Encoder-Decoder Architecture:
    Encoder:
        Input (batch, lookback, features)
        -> LSTM (multi-layer with dropout)
        -> encoder_outputs (batch, lookback, hidden_size)
        -> h_n, c_n (final encoder states)

    Decoder (autoregressive loop):
        For each future step t in [1, ..., horizon]:
            1. Attention context (Bahdanau, 1414):
                score_i = v^T tanh(W_h · h_{t-1} + W_s · encoder_output_i)
                alpha = softmax(scores)
                context = sum(alpha_i · encoder_output_i)
            2. Decoder input: [previous_prediction; context]
            3. Decoder LSTM step:
                h_t, c_t = LSTMCell(decoder_input, (h_{t-1}, c_{t-1}))
            4. Output projection:
                y_hat_t = Linear(h_t)

    Teacher forcing (Bengio et al., 2015):
        During training with probability teacher_forcing_ratio:
            decoder uses y_true_{t-1} instead of y_hat_{t-1}
        At inference: always uses y_hat_{t-1} (free-running)
        Ratio annealed linearly to 0 over first half of training to smooth
        the transition from teacher-forced to free-running regime.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism.

    Computes alignment scores between the decoder hidden state and each
    encoder output using a learned additive scoring function:

        score(h_t, s_i) = v^T tanh(W_h · h_t + W_s · s_i)
        alpha = softmax(score)
        context = sum(alpha_i · s_i)

    This allows the decoder to focus on different parts of the input
    sequence at each generation step, critical for long-horizon forecasting
    where different lookback positions carry information for different
    future timesteps (e.g., same hour yesterday → predict same hour today).

    Args:
        hidden_size: Decoder hidden state dimension.
        encoder_size: Encoder output dimension.
    """

    def __init__(self, hidden_size: int, encoder_size: int) -> None:
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(encoder_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention context and weights.

        Args:
            decoder_hidden: (batch, decoder_hidden_size) — current decoder state.
            encoder_outputs: (batch, lookback, encoder_hidden_size) — all encoder outputs.

        Returns:
            context: (batch, encoder_hidden_size) — weighted sum of encoder outputs.
            alpha: (batch, lookback) — attention weights (sum to 1).
        """
        # score_i = v^T tanh(W_h · h + W_s · s_i)
        # Expand decoder_hidden to (batch, 1, hidden) for broadcasting
        scores = self.v(torch.tanh(
            self.W_h(decoder_hidden.unsqueeze(1)) +
            self.W_s(encoder_outputs)
        )).squeeze(-1)  # (batch, lookback)

        alpha = torch.softmax(scores, dim=-1)  # (batch, lookback)

        # Weighted sum: (batch, 1, lookback) @ (batch, lookback, enc_hidden) → (batch, 1, enc_hidden)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, alpha


class Seq2SeqForecaster(pl.LightningModule):
    """Encoder-Decoder LSTM with Bahdanau attention for multi-step CO2 forecasting.

    Instead of predicting all horizon steps from a single hidden state vector
    (direct forecasting), this model generates predictions one step at a time.
    Each step attends to the full encoder output sequence, allowing the model
    to route information from relevant lookback positions to each future step.

    Args:
        config: Merged configuration dictionary with 'model', 'data', 'training' keys.
        input_size: Number of input features. If None, inferred from config.
        output_size: Number of forecast steps. If None, inferred from config.
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

        self.output_size = output_size

        self.save_hyperparameters(ignore=["config"])

        enc_hidden = model_cfg.get("encoder_hidden_size", 128)
        enc_layers = model_cfg.get("encoder_num_layers", 2)
        enc_dropout = model_cfg.get("encoder_dropout", 0.2)
        dec_hidden = model_cfg.get("decoder_hidden_size", 128)

        # --- Encoder ---
        # Multi-layer LSTM processes the full lookback window.
        # Returns all hidden states (for attention) and final state (for decoder init).
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            dropout=enc_dropout if enc_layers > 1 else 0.0,
            batch_first=True,
        )

        # --- Attention ---
        self.attention = BahdanauAttention(
            hidden_size=dec_hidden,
            encoder_size=enc_hidden,
        )

        # --- Bridge layers ---
        # Map encoder final hidden/cell to decoder initial hidden/cell.
        # Allows encoder and decoder to have different hidden sizes.
        self.bridge_h = nn.Linear(enc_hidden, dec_hidden)
        self.bridge_c = nn.Linear(enc_hidden, dec_hidden)

        # --- Decoder ---
        # LSTMCell (not LSTM) because we step one timestep at a time.
        # Input: previous prediction (1) + attention context (enc_hidden)
        decoder_input_size = 1 + enc_hidden
        self.decoder_cell = nn.LSTMCell(
            input_size=decoder_input_size,
            hidden_size=dec_hidden,
        )

        # Dropout between decoder steps
        self.decoder_dropout = nn.Dropout(model_cfg.get("decoder_dropout", 0.1))

        # Output projection: decoder hidden → single scalar prediction
        self.output_projection = nn.Linear(dec_hidden, 1)

        # --- Training strategy ---
        self.teacher_forcing_ratio = model_cfg.get("teacher_forcing_ratio", 0.5)
        self.teacher_forcing_anneal = model_cfg.get("teacher_forcing_anneal", True)
        self.criterion = nn.MSELoss()

    def _get_current_tf_ratio(self) -> float:
        """Get the current teacher forcing ratio, annealed if configured.

        Linearly decays from initial ratio to 0 over the first 50% of
        max_epochs. This smoothly transitions the model from teacher-forced
        training (stable gradients) to free-running (matches inference).
        """
        if not self.teacher_forcing_anneal:
            return self.teacher_forcing_ratio

        max_epochs = self.config["training"].get("max_epochs", 100)
        anneal_end = max_epochs // 2

        if self.current_epoch >= anneal_end:
            return 0.0

        # Linear decay: ratio * (1 - epoch / anneal_end)
        decay = 1.0 - (self.current_epoch / anneal_end)
        return self.teacher_forcing_ratio * decay

    def forward(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through encoder-decoder.

        Args:
            x: (batch, lookback, features) — encoder input sequence.
            y_true: (batch, horizon) — ground truth for teacher forcing.
                Only used during training; ignored at inference.

        Returns:
            Predictions of shape (batch, horizon).
        """
        batch_size = x.size(0)

        # --- Encode ---
        # encoder_outputs: (batch, lookback, enc_hidden) — all timestep outputs
        # h_n: (num_layers, batch, enc_hidden) — final hidden per layer
        # c_n: (num_layers, batch, enc_hidden) — final cell per layer
        encoder_outputs, (h_n, c_n) = self.encoder(x)

        # --- Bridge: encoder final state → decoder initial state ---
        # Use last layer's hidden/cell state, transform through bridge
        # tanh bounds the bridged state to [-1, 1], matching LSTM conventions
        decoder_h = torch.tanh(self.bridge_h(h_n[-1]))  # (batch, dec_hidden)
        decoder_c = torch.tanh(self.bridge_c(c_n[-1]))  # (batch, dec_hidden)

        # --- Initialize decoder input with last observed target ---
        # Target is the last column in x by convention (feature_cols + target)
        decoder_input_pred = x[:, -1, -1:].clone()  # (batch, 1)

        # --- Decode autoregressively ---
        predictions = []
        tf_ratio = self._get_current_tf_ratio() if self.training else 0.0
        use_tf = self.training and y_true is not None and torch.rand(1).item() < tf_ratio

        for t in range(self.output_size):
            # Attention: focus on relevant encoder positions
            context, _alpha = self.attention(decoder_h, encoder_outputs)

            # Decoder input: [previous_pred (1), context (enc_hidden)]
            decoder_input = torch.cat([decoder_input_pred, context], dim=-1)

            # Single decoder LSTM step
            decoder_h, decoder_c = self.decoder_cell(
                decoder_input, (decoder_h, decoder_c)
            )

            # Apply dropout to decoder hidden state
            decoder_h_dropped = self.decoder_dropout(decoder_h)

            # Project to scalar prediction
            pred_t = self.output_projection(decoder_h_dropped)  # (batch, 1)
            predictions.append(pred_t)

            # Next input: teacher forcing uses ground truth, otherwise use prediction
            if use_tf:
                decoder_input_pred = y_true[:, t:t + 1]  # (batch, 1)
            else:
                # Detach to prevent backprop through entire prediction chain
                # (would be O(horizon^2) memory without detach)
                decoder_input_pred = pred_t.detach()

        # Stack: list of (batch, 1) → (batch, horizon)
        return torch.cat(predictions, dim=-1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x, y_true=y)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x, y_true=None)  # No teacher forcing during validation
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x, y_true=None)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        return self(x, y_true=None)

    def configure_optimizers(self) -> dict:  # type: ignore[override]
        """Configure Adam optimizer with optional warm-up + ReduceLROnPlateau.

        When warmup_epochs > 0, the LR linearly ramps from ~0 to
        learning_rate over that many epochs before ReduceLROnPlateau
        takes over. Warm-up is especially useful for the autoregressive
        decoder which can be unstable in early training.
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
            scale = (self.current_epoch + 1) / warmup
            new_lr = self._target_lr * scale
            optimizer = self.optimizers()
            if not isinstance(optimizer, list):
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
