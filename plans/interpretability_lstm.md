# LSTM Interpretability Study Plan

## Overview

Deep interpretability analysis of the vanilla LSTM forecaster, focusing on gate dynamics,
gradient-based feature attribution, hidden state geometry, and temporal pattern discovery.
Mirrors the TFT study structure (Sections A-E) but adapted to LSTM-specific internals.

**Model:** `LSTMForecaster` (`src/models/lstm.py`)
**Architecture:** 2-layer LSTM (hidden=128) -> Linear projection
**Data variant:** preproc_D (Enhanced 1h) -- consistent with TFT study
**Horizons:** 1h (12 steps) and 24h (288 steps)
**Output directory:** `results/lstm_interpretability/`

---

## Section A: Gate Dynamics Analysis

**Goal:** Understand how the LSTM's 4 gates (forget, input, output, cell candidate) behave
across the lookback window and how they respond to different CO2 regimes.

### A1. Gate Extraction via Forward Hooks

Since `nn.LSTM` is a fused kernel and doesn't expose individual gates, we need a custom
hook strategy:

1. **Register a forward hook** on `model.lstm` that captures:
   - Full output sequence: `lstm_out` of shape `(batch, lookback, hidden_size)`
   - Final hidden state: `h_n` of shape `(num_layers, batch, hidden_size)`
   - Final cell state: `c_n` of shape `(num_layers, batch, hidden_size)`

2. **Gate reconstruction** -- To get actual gate activations, we must manually recompute
   them from the LSTM weight matrices. For each layer `l`:
   ```
   W_ih, W_hh, b_ih, b_hh = model.lstm.weight_ih_l{l}, weight_hh_l{l}, bias_ih_l{l}, bias_hh_l{l}
   gates = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh
   i_t, f_t, g_t, o_t = gates.chunk(4)  # PyTorch order: (i, f, g, o)
   i_t = sigmoid(i_t)
   f_t = sigmoid(f_t)
   g_t = tanh(g_t)
   o_t = sigmoid(o_t)
   ```

3. **Alternative approach** (simpler, recommended): Use a **manual LSTM unrolling** during
   analysis only -- iterate timestep by timestep using the same weights, capturing gates at
   each step. This avoids fighting the fused CUDA kernel.

### A2. Gate Activation Plots

- **Plot A1: Gate activation distributions** -- Histograms of f_t, i_t, o_t values across
  all test samples and timesteps. Expect forget gates near 1.0 (long memory) or bimodal.
- **Plot A2: Gate statistics table** -- Mean, std, min, max, sparsity for each gate type.
- **Plot A3: Gate evolution over lookback window** -- Line plot of mean gate activation at
  each lookback position (t-L, ..., t-1). Shows whether the model attends more to recent vs
  distant past.
- **Plot A4: Forget gate heatmap** -- (lookback_step x hidden_dim) heatmap of forget gate
  values for a representative batch. Reveals which hidden dimensions maintain long-term memory.
- **Plot A5: Gate activation by CO2 regime** -- Split test samples by CO2 level
  (Low/Medium/High), compute mean gate activations per regime. Do forget gates behave
  differently during high-CO2 events?

### A3. Cell State Dynamics

- **Plot A6: Cell state magnitude over time** -- Track ||c_t|| across lookback steps.
  Growing magnitude suggests gradient issues; stable magnitude suggests healthy training.
- **Plot A7: Cell state update ratio** -- Plot `||i_t * g_t|| / ||c_t||` to see how much
  new information is injected relative to accumulated state.

---

## Section B: Gradient-Based Feature Attribution

**Goal:** Identify which input features and lookback timesteps drive predictions most.

### B1. Input Gradient Saliency

1. Set the model to eval mode but enable gradients on the input tensor.
2. For each test batch:
   - `x.requires_grad_(True)` -- shape `(batch, lookback, n_features)`
   - Forward pass -> scalar loss = `prediction.mean()`
   - Backward pass -> `x.grad` has shape `(batch, lookback, n_features)`
   - Store `|x.grad|` (absolute gradient magnitude)
3. Average across all test samples to get `(lookback, n_features)` attribution map.

### B2. Gradient x Input (Integrated-like)

- Compute `x * grad(x)` to weight gradients by input magnitude.
- This better captures the actual contribution (a large gradient on a zero-valued input
  is irrelevant).

### B3. SmoothGrad

- Add Gaussian noise (sigma = 0.1 * input_std) to inputs N=50 times.
- Average the resulting gradient maps to reduce noise.
- Produces a cleaner attribution map than vanilla gradients.

### B4. Plots

- **Plot B1: Gradient attribution heatmap** -- `(lookback_steps x features)`, same format
  as TFT Section B. Color = mean |gradient|.
- **Plot B2: Per-feature gradient importance bar chart** -- Sum |gradient| across timesteps.
  Horizontal bar chart ranking features by importance (%).
- **Plot B3: Temporal gradient profile** -- Sum |gradient| across features at each lookback
  step. Bar chart showing which past hours matter most.
- **Plot B4: Gradient x Input heatmap** -- Same as B1 but using `x * grad` product.
- **Plot B5: SmoothGrad vs Vanilla comparison** -- Side-by-side heatmaps.

---

## Section C: Hidden State Structural Analysis

**Goal:** Explore the geometry of the LSTM's learned representation space via PCA and
clustering, comparing how different CO2 regimes and temporal patterns are encoded.

### C1. Hidden State Extraction

- Hook on `model.lstm` to capture:
  - `h_n[-1]` (last layer, final timestep): `(batch, 128)` -- the state used for prediction
  - Full sequence `lstm_out`: `(batch, lookback, 128)` -- all timestep hidden states

### C2. PCA Analysis

- Apply PCA to `h_n[-1]` across all test samples.
- **Plot C1: Explained variance** -- Bar + cumulative line for top 20 PCs.
- **Plot C2: PCA scatter by CO2 level** -- 2D scatter (PC1 vs PC2), colored by
  Low/Medium/High CO2 categories. Does the LSTM separate CO2 regimes in hidden space?
- **Plot C3: PCA scatter by hour of day** -- Same scatter, colored by hour (twilight cmap).
  Reveals if the LSTM learns diurnal structure.
- **Plot C4: PCA scatter by day of week** -- Colored by weekday vs weekend.
- **Plot C5: 3D PCA** -- PC1/PC2/PC3 3D scatter for richer visualization.

### C3. K-Means Clustering

- Cluster hidden states (K=4, same as TFT study).
- **Plot C6: Cluster scatter** -- PCA 2D colored by cluster.
- **Plot C7: Cluster x CO2 regime crosstab** -- Table showing cluster composition.
- **Plot C8: Cluster centroids in PCA space** -- Annotated centroids with mean CO2 per cluster.

### C4. Hidden State Trajectories

- For selected test windows, plot the trajectory of `h_t` through PCA space across the
  lookback window (from t-L to t-1).
- **Plot C9: Hidden state trajectories** -- 5-10 example trajectories in PC1-PC2 space,
  colored by timestep progression. Shows how the hidden state evolves as it processes
  the input sequence.

### C5. Cross-Layer Comparison

- Compare hidden states from layer 0 vs layer 1:
- **Plot C10: PCA layer comparison** -- Side-by-side PCA scatter plots for layer 0 h_n
  vs layer 1 h_n. Does the second layer learn more abstract representations?

---

## Section D: Temporal Pattern Analysis

**Goal:** Detect periodic patterns in predictions, hidden states, and model behavior.

### D1. FFT of Hidden State PCs

- Extract principal components of hidden states across sequential test samples.
- Apply FFT to each PC time series.
- **Plot D1: FFT of top 3 PCs** -- Frequency spectrum showing dominant periodicities
  (expect peaks at 24h and possibly 12h).

### D2. Autocorrelation of Hidden States

- Compute autocorrelation function (ACF) of each PC.
- **Plot D2: ACF of top 3 PCs** -- Bar plots with significance bounds.
  Shows how quickly hidden state memory decays.

### D3. Prediction Error Periodicity

- Compute FFT of the residual time series.
- **Plot D3: FFT of residuals** -- Reveals systematic periodic errors the model fails to
  capture (e.g., if there's a 24h peak, the model misses daily patterns).

### D4. Rolling Window Analysis

- Compute rolling RMSE over 24h windows across the test set.
- **Plot D4: Rolling RMSE** -- Time series of local performance.
  Identifies periods where the model struggles (e.g., regime transitions).

---

## Section E: LSTM-Specific Analyses

### E1. Weight Matrix Visualization

- **Plot E1: Input weight heatmap** -- Visualize `W_ih` of shape `(4*hidden, input_size)`.
  Shows how each input feature connects to each gate.
- **Plot E2: Recurrent weight heatmap** -- Visualize `W_hh` of shape `(4*hidden, hidden)`.
  Reveals hidden-to-hidden connectivity patterns.
- **Plot E3: Weight matrix singular values** -- SVD of `W_ih` and `W_hh` to assess
  effective rank and redundancy.

### E2. Prediction Quality Analysis

(Same framework as TFT Section E3, for cross-model comparability)

- **Plot E4: Predictions overlay** -- Actual vs predicted time series.
- **Plot E5: Scatter with R2** -- With metrics annotation.
- **Plot E6: 4-panel residual analysis** -- Distribution, over time, by hour, by day.
- **Plot E7: Error by CO2 level** -- Boxplot + statistics table.

### E3. Cross-Horizon Comparison

- If both horizons run: side-by-side feature importance, metrics table, hidden state
  geometry comparison.
- **Plot E8: Metrics summary table** -- RMSE, MAE, R2, MAPE for both horizons.
- **Plot E9: Feature attribution shift** -- How does the gradient importance ranking
  change between 1h and 24h? Expect longer lag features to gain importance at 24h.

---

## Technical Notes

### Hook Implementation

```python
class LSTMHookManager:
    """Forward hooks for LSTM internal state extraction."""

    def __init__(self, model):
        self.captures = {}
        self._hooks = []
        # Hook on the nn.LSTM module
        self._hooks.append(
            model.lstm.register_forward_hook(self._lstm_hook)
        )

    def _lstm_hook(self, module, inp, output):
        # output = (lstm_out, (h_n, c_n))
        self.captures["lstm_out"] = output[0].detach()
        self.captures["h_n"] = output[1][0].detach()
        self.captures["c_n"] = output[1][1].detach()
```

### Gate Reconstruction (Manual Unrolling)

For gate-level analysis, unroll the LSTM manually using `model.lstm` weights:
```python
def unroll_lstm_gates(model, x):
    """Manually unroll LSTM to capture per-timestep gate activations."""
    W_ih = model.lstm.weight_ih_l0  # (4*H, input_size)
    W_hh = model.lstm.weight_hh_l0  # (4*H, H)
    b_ih = model.lstm.bias_ih_l0
    b_hh = model.lstm.bias_hh_l0
    H = model.lstm.hidden_size
    batch, seq_len, _ = x.shape

    h_t = torch.zeros(batch, H, device=x.device)
    c_t = torch.zeros(batch, H, device=x.device)
    gates_history = {"forget": [], "input": [], "cell_cand": [], "output": []}

    for t in range(seq_len):
        gates = x[:, t] @ W_ih.T + b_ih + h_t @ W_hh.T + b_hh
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        gates_history["forget"].append(f_t.detach())
        gates_history["input"].append(i_t.detach())
        gates_history["cell_cand"].append(g_t.detach())
        gates_history["output"].append(o_t.detach())
    return gates_history
```

### Memory Considerations

- Full gate unrolling for 288 timesteps x 128 hidden x batch_size=64 generates ~9.4M floats
  per gate per batch. Limit to 20-30 test batches.
- Store only statistics (mean, std) per timestep, not full tensors, to avoid OOM.

### Dependencies

- Existing: torch, numpy, matplotlib, sklearn (PCA, KMeans), scipy (fft)
- No new dependencies required.

---

## Expected Output

```
results/lstm_interpretability/
  h1/
    gate_dynamics/       -- Plots A1-A7
    gradient_attribution/ -- Plots B1-B5 + CSV
    hidden_state_analysis/ -- Plots C1-C10
    temporal_patterns/   -- Plots D1-D4
    lstm_specific/       -- Plots E1-E3
    predictions/         -- Plots E4-E7
    metrics.json
    predictions.npz
  h24/
    (same structure)
  comparison/           -- Cross-horizon plots E8-E9
  summary.png
  study_results.json
```

## Total: ~30 plots per horizon + cross-horizon comparison
