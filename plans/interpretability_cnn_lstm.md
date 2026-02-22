# CNN-LSTM Interpretability Study Plan

## Overview

Deep interpretability analysis of the CNN-LSTM hybrid model, exploiting both the CNN's
spatial/temporal feature extraction and the LSTM's sequential processing. This model
uniquely offers interpretability at two levels: CNN filters (local pattern detectors) and
LSTM hidden states (global context).

**Model:** `CNNLSTMForecaster` (`src/models/cnn_lstm.py`)
**Architecture:** Conv1D(10->32, k=7) -> Conv1D(32->64, k=5) -> MaxPool(2) -> LSTM(64->128, 2 layers) -> FC head
**Data variant:** preproc_D (Enhanced 1h)
**Horizons:** 1h (12 steps) and 24h (288 steps)
**Output directory:** `results/cnn_lstm_interpretability/`

---

## Section A: CNN Filter Analysis

**Goal:** Understand what temporal patterns the convolutional filters have learned to detect.

### A1. Filter Weight Visualization

1. Extract learned filter weights from both Conv1D layers:
   - Layer 1: `(32 filters, 10 input_channels, kernel_size=7)` -- each filter is a 10x7 pattern
   - Layer 2: `(64 filters, 32 input_channels, kernel_size=5)` -- each filter is a 32x5 pattern

2. **Plot A1: Layer 1 filter bank** -- Grid of 32 heatmaps, each (10 features x 7 timesteps).
   Shows what local temporal pattern each filter detects in the raw input space.
   Annotate feature names on y-axis.

3. **Plot A2: Layer 1 filter norms** -- Bar chart of L2-norm per filter. Filters with
   higher norms contribute more to the output. Identifies dormant vs active filters.

4. **Plot A3: Layer 2 filter summary** -- Since layer 2 operates on learned representations
   (32 channels), visualize as (64 x 32) heatmap of filter norms collapsed across kernel dim.

### A2. Activation Map Analysis

1. Register forward hooks on both Conv1D layers and the MaxPool1D layer to capture
   intermediate activations:
   - After Conv1D_1 + BN + ReLU: shape `(batch, 32, lookback)`
   - After Conv1D_2 + BN + ReLU: shape `(batch, 64, lookback)`
   - After MaxPool: shape `(batch, 64, lookback//2)`

2. **Plot A4: Layer 1 activation heatmap** -- For a representative sample, show
   `(32 filters x lookback)` activation map. Which filters fire at which timesteps?

3. **Plot A5: Layer 2 activation heatmap** -- Same for the deeper layer, `(64 x lookback)`.

4. **Plot A6: Activation statistics per filter** -- Mean activation, sparsity (% near zero),
   and max activation per filter. Table format.

5. **Plot A7: Filter activation by CO2 regime** -- For each CO2 level (Low/Medium/High),
   compute mean activation per filter. Which filters specialize in detecting high-CO2 patterns?

### A3. Temporal Receptive Field Analysis

1. The effective receptive field of the CNN after 2 layers + pooling:
   - Layer 1 (k=7): RF = 7 timesteps
   - Layer 2 (k=5): RF = 7 + (5-1) = 11 timesteps
   - MaxPool(2): RF unchanged but stride doubles
   At 1h resolution, the CNN "sees" local windows of ~11 hours.

2. **Plot A8: Receptive field diagram** -- Schematic showing the temporal coverage of each
   CNN layer relative to the lookback window.

---

## Section B: Gradient-Based Feature Attribution

**Goal:** Trace how gradients flow from predictions back through LSTM -> Pool -> CNN -> Input.

### B1. Input Gradient Saliency

Same protocol as LSTM study Section B1, but gradients now flow through the full CNN-LSTM
pipeline. The CNN creates a non-trivial gradient landscape due to max-pooling (sparse gradients)
and batch normalization.

1. Enable gradients on input tensor `(batch, lookback, n_features)`.
2. Forward pass -> scalar loss -> backward pass.
3. Collect `|grad_input|` of shape `(batch, lookback, n_features)`.

### B2. Guided Backpropagation

Standard backprop through ReLU kills gradients where activations are negative. Guided
backprop also kills gradients where the gradient itself is negative, producing sharper
attribution maps.

Implementation: Replace all `ReLU` forward hooks with guided versions during analysis.

### B3. Layer-wise Gradient Analysis

Hook gradients at intermediate points to see where information flows:
1. Gradient at CNN Layer 1 output -> shows which filters carry prediction-relevant signal
2. Gradient at CNN Layer 2 output -> shows post-refinement signal
3. Gradient at LSTM input (= MaxPool output) -> shows what the LSTM receives

### B4. Plots

- **Plot B1: Input gradient heatmap** -- `(lookback x features)`.
- **Plot B2: Per-feature gradient importance** -- Bar chart.
- **Plot B3: Temporal gradient profile** -- Which lookback positions matter.
- **Plot B4: Guided backprop heatmap** -- Sharper version of B1.
- **Plot B5: Layer-wise gradient magnitude** -- 3 heatmaps showing gradient flow through
  CNN_L1 -> CNN_L2 -> LSTM input.
- **Plot B6: Gradient flow ratio** -- Ratio of gradient magnitude at each layer to quantify
  information bottlenecks.

---

## Section C: Hidden State Structural Analysis

**Goal:** Analyze both CNN feature maps and LSTM hidden states geometrically.

### C1. CNN Feature Map Embeddings

- Extract CNN output (after MaxPool): `(batch, 64, lookback//2)`.
- Flatten to `(batch, 64 * lookback//2)` and apply PCA/t-SNE.
- **Plot C1: CNN embedding scatter by CO2 level** -- Does the CNN alone separate regimes?
- **Plot C2: CNN embedding scatter by hour** -- Does the CNN encode temporal context?

### C2. LSTM Hidden State Analysis

Same as LSTM study Section C, but now the LSTM processes CNN-extracted features rather
than raw inputs. This reveals a different representation geometry.

- Hook on `model.lstm` to capture `h_n[-1]` (final hidden state).
- **Plot C3: PCA explained variance** -- Top 20 PCs.
- **Plot C4: PCA scatter by CO2 level** -- Compare to pure LSTM scatter.
- **Plot C5: PCA scatter by hour of day**.
- **Plot C6: K-Means clustering** -- 4 clusters + CO2 regime crosstab.

### C3. Representation Comparison: CNN vs LSTM

- **Plot C7: CNN vs LSTM explained variance** -- Overlay the PCA explained variance curves
  for CNN feature maps and LSTM hidden states. Which representation is more compact?
- **Plot C8: Canonical Correlation Analysis (CCA)** -- Measure the linear relationship
  between CNN feature map and LSTM hidden state spaces. High correlation means the LSTM
  preserves CNN information; low means it transforms it significantly.

---

## Section D: Temporal Pattern Analysis

### D1. CNN Activation Periodicity

- For each CNN filter, compute the activation time series across sequential test samples.
- Apply FFT to detect periodic activation patterns.
- **Plot D1: Filter activation FFT** -- Top 6 most periodic filters with their frequency spectra.

### D2. LSTM Hidden State Periodicity

Same as LSTM study:
- **Plot D2: FFT of hidden state PCs** -- Frequency spectrum of top 3 PCs.
- **Plot D3: ACF of hidden state PCs** -- Autocorrelation with lags.

### D3. Prediction Error Analysis

- **Plot D4: FFT of residuals** -- Periodic error patterns.
- **Plot D5: Rolling RMSE** -- 24h window across test set.
- **Plot D6: Phase-aligned error** -- Average error profile aligned to 24h cycles.

---

## Section E: CNN-LSTM-Specific Analyses

### E1. CNN-to-LSTM Information Flow

- Measure the mutual information between CNN filter activations and LSTM output.
- **Plot E1: Filter-output correlation matrix** -- Pearson correlation between each of
  64 CNN filters' mean activation and the LSTM's predicted CO2 value.
  Reveals which CNN filters most directly influence predictions.

### E2. MaxPool Attention Proxy

- The MaxPool1D layer selects one value per 2-timestep window per filter. The selected
  indices reveal which timesteps the model considers most important.
- Hook on MaxPool to capture indices via `return_indices=True`.
- **Plot E2: MaxPool selection frequency** -- Histogram of selected timestep positions
  across all filters and samples. Shows temporal attention pattern of the CNN.

### E3. Ablation by CNN Layer

- Evaluate model performance when:
  1. Layer 1 filters are frozen/zeroed
  2. Layer 2 filters are frozen/zeroed
  3. Both CNN layers bypassed (LSTM on raw input for comparison)
- **Plot E3: Layer ablation impact** -- Bar chart of RMSE with each configuration.

### E4. Prediction Quality Analysis

(Same framework as TFT/LSTM studies)

- **Plot E4: Predictions overlay**.
- **Plot E5: Scatter with R2**.
- **Plot E6: 4-panel residual analysis**.
- **Plot E7: Error by CO2 level**.

### E5. Cross-Horizon Comparison

- **Plot E8: Metrics summary table**.
- **Plot E9: Feature attribution shift between horizons**.
- **Plot E10: CNN filter importance shift** -- Do different filters become important at 24h?

---

## Technical Notes

### Hook Implementation

```python
class CNNLSTMHookManager:
    """Forward hooks for CNN-LSTM internal state extraction."""

    def __init__(self, model):
        self.captures = {}
        self._hooks = []

        # CNN Layer 1 (after BN + ReLU)
        # Hook after the first conv block - need to identify exact module
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                self._hooks.append(
                    module.register_forward_hook(self._tensor_hook(f"conv_{name}"))
                )
            elif isinstance(module, nn.MaxPool1d):
                # Use return_indices hook
                self._hooks.append(
                    module.register_forward_hook(self._pool_hook("maxpool"))
                )

        # LSTM
        self._hooks.append(
            model.lstm.register_forward_hook(self._lstm_hook)
        )
```

### MaxPool Index Extraction

To capture MaxPool indices, temporarily modify the MaxPool layer:
```python
# Before analysis
model.pool.return_indices = True
# After analysis, restore
model.pool.return_indices = False
```
Note: this changes the forward pass return type. Handle in the hook.

### Memory Considerations

- CNN activations at full resolution: `(batch, 64, lookback)` = significant memory.
  For lookback=24 (1h resolution), this is manageable. For lookback=288 (5min), limit batches.
- Layer 1 filter visualization is straightforward (32 x 10 x 7 = 2240 parameters).

### Dependencies

- Existing: torch, numpy, matplotlib, sklearn (PCA, KMeans), scipy (fft)
- Optional: `sklearn.cross_decomposition.CCA` for canonical correlation analysis
- No new pip installs required.

---

## Expected Output

```
results/cnn_lstm_interpretability/
  h1/
    cnn_filters/         -- Plots A1-A8
    gradient_attribution/ -- Plots B1-B6 + CSV
    hidden_state_analysis/ -- Plots C1-C8
    temporal_patterns/   -- Plots D1-D6
    cnn_lstm_specific/   -- Plots E1-E3
    predictions/         -- Plots E4-E7
    metrics.json
    predictions.npz
  h24/
    (same structure)
  comparison/           -- Cross-horizon plots E8-E10
  summary.png
  study_results.json
```

## Total: ~35 plots per horizon + cross-horizon comparison
