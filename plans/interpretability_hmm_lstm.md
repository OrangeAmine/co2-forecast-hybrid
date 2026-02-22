# HMM-LSTM Interpretability Study Plan

## Overview

Deep interpretability analysis of the HMM-LSTM hybrid model. This is the most naturally
interpretable model in the study due to its explicit two-stage architecture: the HMM provides
discrete, human-readable regime states, while the LSTM adds sequential modeling. The study
exploits both stages.

**Model:** `HMMRegimeDetector` + `HMMLSTMForecaster` (`src/models/hmm_lstm.py`)
**Architecture:**
  - Stage 1: GaussianHMM(n_states=3) on [CO2, Noise, TemperatureExt]
  - Stage 2: LSTM(input=13, hidden=128, layers=2) -> FC head
    (13 = 10 original features + 3 HMM state posterior probabilities)
**Data variant:** preproc_D (Enhanced 1h)
**Horizons:** 1h (12 steps) and 24h (288 steps)
**Output directory:** `results/hmm_lstm_interpretability/`

---

## Section A: HMM Regime Analysis

**Goal:** Fully characterize the 3 learned hidden states -- what do they represent physically?

### A1. State Parameter Visualization

After fitting the HMM, extract its learned parameters:
- `hmm.means_`: `(3, 3)` -- mean of [CO2, Noise, TemperatureExt] per state
- `hmm.covars_`: `(3, 3, 3)` -- full covariance matrix per state
- `hmm.transmat_`: `(3, 3)` -- state transition probabilities
- `hmm.startprob_`: `(3,)` -- initial state distribution

1. **Plot A1: Transition matrix heatmap** -- 3x3 heatmap with probabilities annotated.
   Self-loop strength reveals regime persistence. Off-diagonal elements show transition paths.

2. **Plot A2: State emission distributions** -- For each of the 3 HMM features, plot
   per-state Gaussian PDFs overlaid. Shows how states partition the CO2/Noise/Temperature
   space. Use colored shading per state.

3. **Plot A3: State means table** -- Table with state index, mean CO2, mean Noise,
   mean TemperatureExt, and a human-readable label (e.g., "Ventilated/Low occupancy",
   "Occupied/Active", "Stale air/High CO2").

4. **Plot A4: State covariance ellipses** -- 2D scatter of CO2 vs Noise colored by Viterbi
   state, with 95% confidence ellipses for each state's Gaussian. Shows separation quality.

### A2. Viterbi State Sequence Visualization

1. Decode the Viterbi (most likely) state sequence on the full dataset.
2. **Plot A5: CO2 time series with state overlay** -- Full test set CO2 with background
   color-coded by Viterbi state. This is the most visually compelling plot -- directly shows
   how the HMM segments occupancy/ventilation patterns.

3. **Plot A6: State sequence detail (1-week zoom)** -- Zoomed view of 1 representative week
   showing clear daily occupancy cycles.

4. **Plot A7: State duration histogram** -- Distribution of how long each state persists
   before transitioning. Log-scale x-axis. Shows if states are stable (hours) or flickering (minutes).

5. **Plot A8: State transition times** -- Histogram of transitions by hour-of-day.
   E.g., "Low->High" transitions peak at morning occupancy onset.

### A3. Posterior Probability Analysis

1. Compute soft posterior probabilities `gamma_t(k)` via forward-backward on test data.
2. **Plot A9: Posterior probability time series** -- 3 stacked line plots showing
   `P(state=k | observations)` over time for each state. Overlaid with actual CO2 (secondary axis).

3. **Plot A10: Posterior entropy** -- `H(gamma_t) = -sum_k gamma_t(k) * log(gamma_t(k))`.
   High entropy = model is uncertain about the regime. Plot over time. Expect high entropy
   during transitions (regime-switching moments).

4. **Plot A11: Entropy vs prediction error** -- Scatter of posterior entropy vs absolute
   prediction error. Tests hypothesis that regime uncertainty causes forecasting errors.

---

## Section B: LSTM Gate Dynamics (Augmented Input)

**Goal:** Understand how the LSTM processes the 13-dimensional input (10 sensor features +
3 HMM posteriors) and whether the HMM channels receive special treatment by the gates.

### B1. Gate Extraction

Same manual unrolling approach as LSTM study Section A, but now input_size=13.
After gate reconstruction, we can analyze how each gate responds to the 3 HMM channels
specifically.

### B2. HMM Channel Gate Response

1. Compute input gate activation `i_t` and partition the weight connections:
   - Columns 0-9 of `W_ih`: connections from original sensor features
   - Columns 10-12 of `W_ih`: connections from HMM state posteriors

2. **Plot B1: Input gate weight decomposition** -- Heatmap of `W_ih` (forget gate rows)
   with a clear separator between sensor features and HMM channels. Shows how strongly
   the HMM information influences gating.

3. **Plot B2: Gate activation correlation with HMM states** -- For each test sample,
   compute correlation between forget gate activations and the 3 HMM posterior values.
   If high, the LSTM modulates its memory based on regime context.

### B3. Standard Gate Plots

(Same as LSTM study Section A)
- **Plot B3: Gate activation distributions** (f, i, g, o).
- **Plot B4: Forget gate evolution over lookback**.
- **Plot B5: Gate activation by CO2 regime**.

---

## Section C: Gradient-Based Feature Attribution

**Goal:** Quantify the contribution of each input channel, with special focus on
separating the importance of HMM channels from original sensor features.

### C1. Input Gradient Saliency

Same protocol as LSTM study, but the 13-dimensional input allows us to decompose:
- Gradient importance of features 0-9 (sensor channels)
- Gradient importance of features 10-12 (HMM posteriors)

### C2. HMM Channel Ablation Gradient

1. Compute gradients with full input (baseline).
2. Compute gradients with HMM channels zeroed out (ablated).
3. Difference reveals the HMM's causal contribution to prediction.

### C3. Plots

- **Plot C1: Full gradient attribution heatmap** -- `(lookback x 13 features)` with
  HMM channels clearly labeled.
- **Plot C2: Feature importance bar chart** -- All 13 features, with HMM channels
  highlighted in a distinct color.
- **Plot C3: Temporal gradient profile** -- Split into sensor vs HMM contributions.
- **Plot C4: HMM channel importance over time** -- Per-sample importance of the 3 HMM
  channels. Does HMM importance spike during regime transitions?
- **Plot C5: HMM ablation impact** -- Side-by-side heatmaps: full model vs HMM-ablated.

---

## Section D: Hidden State Structural Analysis

**Goal:** Compare the LSTM hidden state geometry with and without HMM augmentation.

### D1. Standard Hidden State Analysis

Same as LSTM study Section C:
- **Plot D1: PCA explained variance**.
- **Plot D2: PCA scatter by CO2 level**.
- **Plot D3: PCA scatter by hour of day**.
- **Plot D4: K-Means clustering** + CO2 regime crosstab.

### D2. Regime-Conditioned Analysis

- Split test samples by dominant HMM state (argmax of posterior).
- **Plot D5: Hidden state PCA by HMM regime** -- 2D scatter colored by HMM state (3 colors).
  Shows whether the LSTM's hidden states correlate with HMM's discrete states.

- **Plot D6: Hidden state cluster vs HMM state crosstab** -- K-Means clusters vs HMM states.
  Measures alignment between unsupervised LSTM clustering and HMM's explicit segmentation.

### D3. With vs Without HMM Comparison

If a vanilla LSTM has been trained on the same data (from the main experiment):
- **Plot D7: PCA overlay** -- LSTM hidden states vs HMM-LSTM hidden states in the same
  PCA projection. Shows how HMM augmentation reshapes the representation space.

---

## Section E: Temporal Pattern Analysis

### E1. HMM State Periodicity

- **Plot E1: HMM state autocorrelation** -- ACF of each posterior probability channel.
  Expect ~24h periodicity reflecting daily occupancy cycles.
- **Plot E2: State transition frequency spectrum** -- FFT of the state transition indicator
  time series. Peaks reveal the dominant switching frequency.

### E2. Prediction Error by Regime

- **Plot E3: Per-regime error boxplot** -- Split residuals by HMM state. RMSE, MAE, and
  mean bias per state. Reveals if certain regimes are harder to forecast.
- **Plot E4: Error during transitions** -- Compute prediction error at timesteps within
  +/-3 steps of a state transition vs far from transitions. Tests hypothesis that regime
  boundaries cause large errors.

### E3. Standard Temporal Analysis

- **Plot E5: FFT of hidden state PCs**.
- **Plot E6: Rolling RMSE (24h window)**.
- **Plot E7: Phase-aligned average error**.

---

## Section F: HMM-LSTM-Specific Analyses

### F1. HMM Value-Add Assessment

The key question: does the HMM actually help?

1. **Plot F1: Permutation importance of HMM channels** -- Shuffle only the 3 HMM posterior
   columns and measure RMSE increase. Compare to shuffling each sensor feature individually.
   Directly quantifies the marginal value of regime information.

2. **Plot F2: Model comparison table** -- Side-by-side metrics:
   | | LSTM (no HMM) | HMM-LSTM | Delta |
   Shows the concrete benefit of HMM augmentation.

### F2. HMM Sensitivity Analysis

- **Plot F3: HMM posterior perturbation** -- Systematically increase/decrease each HMM
  posterior by 0.1 and measure prediction change. Shows model sensitivity to regime beliefs.

### F3. Prediction Quality Analysis

- **Plot F4: Predictions overlay**.
- **Plot F5: Scatter with R2**.
- **Plot F6: 4-panel residual analysis**.
- **Plot F7: Error by CO2 level**.

### F4. Cross-Horizon Comparison

- **Plot F8: Metrics summary table**.
- **Plot F9: HMM regime importance shift** -- Does the HMM become more/less useful at 24h?
- **Plot F10: State distribution shift** -- Are state proportions different in the
  lookback windows that precede 1h vs 24h forecasts?

---

## Technical Notes

### HMM Access

The `HMMRegimeDetector` stores the fitted HMM as `self.hmm` (a `hmmlearn.GaussianHMM`).
Key attributes:
```python
detector.hmm.means_         # (n_states, n_hmm_features)
detector.hmm.covars_        # (n_states, n_hmm_features, n_hmm_features)
detector.hmm.transmat_      # (n_states, n_states)
detector.hmm.startprob_     # (n_states,)
detector.hmm.score(X)       # log-likelihood
detector.hmm.predict(X)     # Viterbi decoding
detector.hmm.predict_proba(X)  # posterior probabilities
```

### Two-Stage Training

The HMM must be fitted first on the training data (unscaled), then posteriors are computed
and appended to all splits before LSTM training. The existing `CO2DataModule.from_dataframes()`
handles this. For the interpretability study, we need to preserve the fitted `HMMRegimeDetector`
object to access HMM parameters after LSTM training.

### Dependencies

- Existing: hmmlearn, torch, numpy, matplotlib, sklearn, scipy
- No new dependencies.

---

## Expected Output

```
results/hmm_lstm_interpretability/
  h1/
    hmm_regime_analysis/  -- Plots A1-A11
    gate_dynamics/        -- Plots B1-B5
    gradient_attribution/ -- Plots C1-C5 + CSV
    hidden_state_analysis/ -- Plots D1-D7
    temporal_patterns/    -- Plots E1-E7
    hmm_lstm_specific/    -- Plots F1-F3
    predictions/          -- Plots F4-F7
    metrics.json
    predictions.npz
  h24/
    (same structure)
  comparison/            -- Cross-horizon plots F8-F10
  summary.png
  study_results.json
```

## Total: ~40 plots per horizon + cross-horizon comparison
## (Most plots of any model, reflecting the richest interpretability surface)
