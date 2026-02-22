# SARIMA Interpretability Study Plan

## Overview

Deep interpretability analysis of the SARIMA baseline model. As a classical statistical model,
SARIMA offers the most transparent interpretability of any model in the study: all parameters
have direct mathematical meaning, residual diagnostics are well-established, and the model's
behavior can be fully characterized analytically. This study emphasizes statistical rigor
and time series theory.

**Model:** `SARIMAForecaster` (`src/models/sarima.py`)
**Architecture:** SARIMA(1,1,1)(1,1,1,s) where s depends on resolution
  - preproc_D (1h): s=24 (24h seasonal period)
**Input:** Univariate CO2 time series only (no exogenous features)
**Data variant:** preproc_D (Enhanced 1h)
**Horizons:** 1h (12 steps) and 24h (288 steps)
**Output directory:** `results/sarima_interpretability/`

---

## Section A: Model Parameter Analysis

**Goal:** Fully characterize the fitted SARIMA parameters and their physical interpretation
for indoor CO2 dynamics.

### A1. Parameter Extraction

From `SARIMAForecaster.result_` (a `SARIMAXResults` object):
```python
result.params      # All parameters as array
result.pvalues     # P-values for significance testing
result.bse         # Standard errors
result.conf_int()  # 95% confidence intervals
result.aic         # Akaike Information Criterion
result.bic         # Bayesian Information Criterion
```

SARIMA(1,1,1)(1,1,1,24) has the following parameters:
- `ar.L1` (phi_1): AR(1) coefficient -- persistence of CO2 deviations
- `ma.L1` (theta_1): MA(1) coefficient -- short-term shock response
- `ar.S.L24` (Phi_1): Seasonal AR -- "same hour yesterday" influence
- `ma.S.L24` (Theta_1): Seasonal MA -- daily periodic shock response
- `sigma2`: Innovation variance

### A2. Parameter Plots

- **Plot A1: Parameter summary table** -- All coefficients with values, std errors,
  z-statistics, p-values, and 95% CIs. Traffic-light coloring for significance.

- **Plot A2: Coefficient bar chart with CIs** -- Visual representation of each parameter's
  magnitude with error bars. Clearly shows which terms dominate the model.

- **Plot A3: AR/MA polynomial roots** -- Plot the roots of phi(B) and theta(B) polynomials
  in the complex plane. Roots near the unit circle indicate near-nonstationarity or
  near-noninvertibility. All roots should be outside the unit circle for stability.

- **Plot A4: Seasonal polynomial roots** -- Same for Phi(B^s) and Theta(B^s).

### A3. Physical Interpretation Table

Create a human-readable interpretation:
- **Plot A5: Parameter interpretation** -- Table mapping each coefficient to its physical
  meaning:
  | Parameter | Value | Interpretation |
  |-----------|-------|----------------|
  | phi_1 | 0.XX | "XX% of the previous hour's deviation from trend persists" |
  | Phi_1 | 0.XX | "XX% of yesterday's same-hour deviation persists" |
  | theta_1 | 0.XX | "XX% of the previous shock is corrected in 1 hour" |
  | Theta_1 | 0.XX | "XX% of yesterday's shock is corrected after 24 hours" |
  | sigma^2 | XX | "Innovation variance = XX ppm^2, typical shock = +/- YY ppm" |

---

## Section B: Impulse Response Analysis

**Goal:** Characterize how the model propagates a unit shock through time, revealing
the system's dynamic response to CO2 disturbances.

### B1. Impulse Response Function (IRF)

The IRF shows the effect of a 1-unit innovation at t=0 on future values y_{t+h}:
```python
irf = result.impulse_responses(steps=72)  # 72 hours = 3 days
```

- **Plot B1: Impulse response function** -- Line plot of IRF over 72 hours with 95%
  confidence bands. Key features to annotate:
  - Initial response magnitude (at h=0)
  - Half-life: when does the response decay to 50%?
  - 24h echo: does a spike at h=24 show seasonal propagation?
  - Convergence to zero (stationarity verification)

- **Plot B2: Cumulative impulse response** -- Cumulative sum of IRF. Shows the total
  long-run effect of a unit shock. For differenced models, this converges to a finite value.

### B2. Step Response

The step response shows how the model responds to a permanent level shift (e.g., sudden
increase in background CO2):
```python
step_response = np.cumsum(irf)
```

- **Plot B3: Step response function** -- Shows model adaptation to persistent changes.
  A flat asymptote means the model fully adapts; oscillation means seasonal effects.

### B3. Variance Decomposition

- **Plot B4: Forecast error variance decomposition** -- How much of the h-step-ahead
  forecast uncertainty comes from AR vs MA vs seasonal AR vs seasonal MA components.
  Computed analytically from the MA(infinity) representation.

---

## Section C: Residual Diagnostics

**Goal:** Comprehensive residual analysis to verify model adequacy and identify what
SARIMA fails to capture.

### C1. Standard Residual Diagnostics

```python
result.plot_diagnostics()  # statsmodels built-in 4-panel plot
```

But we create custom, higher-quality versions:

- **Plot C1: Residual time series** -- Full residual sequence with +/-2*sigma bands.
  Highlights periods of model failure (large residuals).

- **Plot C2: Residual distribution** -- Histogram + KDE + Q-Q plot against normal
  distribution. Tests normality assumption (WN ~ N(0, sigma^2)).

- **Plot C3: ACF of residuals** -- Autocorrelation function with 95% significance bounds
  (Bartlett bands). Any significant spikes indicate model inadequacy.
  Particular attention to lags 1, 24, 48, 168 (weekly).

- **Plot C4: PACF of residuals** -- Partial autocorrelation function. Significant spikes
  suggest missing AR terms.

- **Plot C5: Ljung-Box test** -- Plot p-values of the Ljung-Box Q-statistic at lags
  1, 6, 12, 24, 48, 72. All should be > 0.05 if the model is adequate.
  Table of test statistics and p-values.

### C2. Advanced Residual Analysis

- **Plot C6: Squared residual ACF** -- Autocorrelation of squared residuals. Significant
  autocorrelation indicates heteroscedasticity (conditional variance changes over time).
  If present, suggests GARCH extension might help.

- **Plot C7: Residuals by hour of day** -- Boxplot. Tests whether certain hours have
  systematically larger residuals (e.g., morning occupancy transitions).

- **Plot C8: Residuals by day of week** -- Boxplot. Weekend vs weekday error patterns.

- **Plot C9: Rolling residual variance** -- 24h rolling window of residual variance.
  Non-constant variance = heteroscedasticity.

### C3. Spectral Residual Analysis

- **Plot C10: FFT of residuals** -- Power spectrum of residuals. Peaks indicate periodic
  patterns the model failed to capture. Compare to FFT of the original series.

- **Plot C11: Original vs residual spectrum** -- Overlay FFT of raw CO2 and FFT of residuals.
  Frequencies that disappear in residuals were successfully modeled by SARIMA.
  Frequencies that persist are the model's blind spots.

---

## Section D: Forecast Uncertainty Analysis

**Goal:** Characterize SARIMA's forecast uncertainty, which is analytically available
(unlike neural models that need bootstrap or MC dropout).

### D1. Forecast Confidence Intervals

SARIMA provides exact confidence intervals via the MA(infinity) representation:
```python
forecast = result.get_forecast(steps=12)
ci = forecast.conf_int(alpha=0.05)  # 95% CI
```

- **Plot D1: Fan chart** -- Forecast with nested confidence intervals (50%, 80%, 95%).
  Multiple test windows overlaid to show typical uncertainty growth.

- **Plot D2: CI width vs horizon** -- Plot the width of the 95% CI as a function of
  forecast horizon (1 to 12 steps for 1h, 1 to 288 for 24h). Shows how quickly
  uncertainty grows.

### D2. Prediction Interval Coverage

- Compute empirical coverage: what fraction of actual values fall within the 95% CI?
- **Plot D3: Prediction interval coverage** -- Expected: 95%. Overcoverage (>95%) means
  the model is too uncertain; undercoverage (<95%) means overconfident.
  Bar chart of coverage at each horizon step.

- **Plot D4: Calibration plot** -- For nominal levels 10%, 20%, ..., 90%, plot actual
  coverage vs nominal coverage. Perfect calibration = 45-degree line.

### D3. One-Step-Ahead Analysis

- **Plot D5: One-step-ahead predictions** -- For each test point, the model's
  1-step prediction with CI. This is where SARIMA should perform best.
  Overlay with actual values.

- **Plot D6: One-step vs multi-step error** -- Compare 1-step RMSE to 6-step and 12-step.
  Shows error accumulation rate specific to SARIMA's recursive forecasting.

---

## Section E: Seasonal Decomposition Comparison

**Goal:** Compare SARIMA's implicit decomposition with explicit STL decomposition.

### E1. STL Decomposition

```python
from statsmodels.tsa.seasonal import STL
stl = STL(co2_series, period=24)
decomposition = stl.fit()
```

- **Plot E1: STL decomposition** -- 4-panel (observed, trend, seasonal, residual).

### E2. SARIMA's Implicit Decomposition

SARIMA's differencing removes trend and seasonality:
- `(1-B)^1`: removes trend
- `(1-B^24)^1`: removes 24h seasonality
- The AR/MA model on the differenced series captures remaining structure.

- **Plot E2: Differenced series** -- Plot the twice-differenced (trend + seasonal) CO2
  series that SARIMA actually models. Should look stationary.

### E3. Comparison

- **Plot E3: STL seasonal vs SARIMA seasonal** -- Overlay STL's extracted seasonal component
  with the 24h pattern implied by SARIMA's seasonal parameters. Measures agreement.

- **Plot E4: STL residual vs SARIMA residual** -- Compare what's "left over" after each
  decomposition method. Are the residual patterns similar?

---

## Section F: SARIMA-Specific Analyses

### F1. Model Selection Diagnostics

- **Plot F1: AIC/BIC for alternative orders** -- Fit a grid of SARIMA orders:
  (p,d,q) in {0,1,2} x {0,1} x {0,1,2}, (P,D,Q) in {0,1} x {0,1} x {0,1}
  (reduced grid due to computation cost). Plot AIC and BIC for each.
  Verify that (1,1,1)(1,1,1,24) is indeed optimal.

- **Plot F2: Information criteria table** -- Top 10 models by AIC, with BIC and
  log-likelihood for comparison.

### F2. Stationarity Tests

- **Plot F3: ADF and KPSS tests** -- Run Augmented Dickey-Fuller and KPSS tests on:
  1. Original series (expect non-stationary)
  2. Once-differenced series (expect stationary or borderline)
  3. Seasonally differenced series (expect stationary)
  4. Twice-differenced (trend + seasonal) series (should be stationary)
  Table of test statistics, critical values, and conclusions.

### F3. SARIMA Limitations Visualization

This section explicitly documents what SARIMA cannot capture:

- **Plot F4: Exogenous feature impact** -- Show correlation between SARIMA residuals and
  available exogenous features (Noise, TemperatureExt, Pression). Significant correlations
  reveal information that SARIMA misses but multivariate models can exploit.

- **Plot F5: Nonlinear regime effects** -- Compare SARIMA residual magnitude in different
  CO2 regimes (Low/Medium/High). If residuals are larger in specific regimes, SARIMA's
  linear assumption fails there.

### F4. Prediction Quality Analysis

- **Plot F6: Predictions overlay**.
- **Plot F7: Scatter with R2**.
- **Plot F8: 4-panel residual analysis** (same framework as other models).
- **Plot F9: Error by CO2 level**.

### F5. Cross-Horizon Comparison

- **Plot F10: Metrics summary table**.
- **Plot F11: Error growth curve** -- RMSE at each forecast step (1 to 12 for 1h, 1 to 288
  for 24h). SARIMA's error growth is analytically predictable via the IRF.
- **Plot F12: CI width vs actual error** -- Overlay the predicted CI width with the
  realized RMSE at each step. Tests whether analytical uncertainty matches reality.

---

## Technical Notes

### Accessing the Fitted Model

```python
sarima = SARIMAForecaster(config)
sarima.fit(train_X, train_y, val_X, val_y)

# The fitted statsmodels result object
result = sarima.result_

# Key methods
result.params           # Fitted parameters
result.pvalues          # P-values
result.summary()        # Full statistical summary
result.impulse_responses(steps=72)
result.plot_diagnostics()
result.get_forecast(steps=12)
result.resid            # In-sample residuals
result.fittedvalues     # In-sample fitted values
```

### Prediction Protocol

SARIMA uses `filter()` for each test window (re-initializes state without re-estimating):
```python
filtered = sarima.result_.apply(test_history, refit=False)
forecast = filtered.forecast(steps=horizon)
```

### Computation Time

- SARIMA fitting: 5-15 minutes depending on series length and seasonal period.
- IRF computation: instant.
- Model selection grid: slow (each SARIMA fit = minutes). Limit to a small grid
  or parallelize with joblib.
- All diagnostic plots: fast (analytical computations).

### Dependencies

- Existing: statsmodels, numpy, matplotlib, pandas, scipy
- No new dependencies required.

---

## Expected Output

```
results/sarima_interpretability/
  h1/
    parameters/          -- Plots A1-A5
    impulse_response/    -- Plots B1-B4
    residual_diagnostics/ -- Plots C1-C11
    forecast_uncertainty/ -- Plots D1-D6
    seasonal_decomposition/ -- Plots E1-E4
    sarima_specific/     -- Plots F1-F5
    predictions/         -- Plots F6-F9
    metrics.json
    predictions.npz
  h24/
    (same structure)
  comparison/           -- Cross-horizon plots F10-F12
  summary.png
  study_results.json
```

## Total: ~38 plots per horizon + cross-horizon comparison
## (Highest plot count among non-HMM models, reflecting the depth of classical statistical diagnostics)
