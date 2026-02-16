# Plan: Preprocessing Pipeline Experiments

## Objective

Systematically test 4 preprocessing variants, then incrementally add domain-driven
features to the best-performing variant. Finally, run feature importance diagnostics
on all 7 models. The goal is to find the optimal preprocessing + feature set for
indoor CO2 forecasting.

---

## Design Decisions (resolved)

| # | Decision | Choice |
|---|----------|--------|
| 1 | NaN handling before/after split | Keep NaN rows in the resampled grid. Split first, then interpolate within each split (linear, max 60 min). Drop rows only for gaps > 60 min. |
| 2 | dCO2 in simple variants | Include. dCO2 is domain-driven and physically meaningful. Computed post-split within each split. |
| 3 | Savitzky-Golay edge effects | Use `mode='nearest'` (scipy default). Minor edge distortion is acceptable; all data points preserved. |
| 4 | Backward compatibility | Replace old pipeline entirely. Delete old exp1/exp2/exp3 configs and processed CSVs. Clean break. |
| 5 | Max interpolation gap | 60 minutes. Gaps > 1h are dropped. Indoor CO2 changes slowly enough for linear interpolation over 1h. |

---

## PHASE 1: Replace preprocessing with a split-aware pipeline

### What gets deleted

- `src/data/enhanced_preprocessing.py` — replaced by post-split logic in new pipeline
- `configs/experiments/exp1_baseline.yaml` — replaced by variant A
- `configs/experiments/exp2_hourly.yaml` — replaced by variant B
- `configs/experiments/exp3_enhanced_5min.yaml` — replaced by variant C
- `configs/experiments/exp3_enhanced_1h.yaml` — replaced by variant D
- `data/processed/BM2021_22_5min.csv` — will be regenerated
- `data/processed/BM2021_22_1h.csv` — will be regenerated
- `data/processed/BM2021_22_5min_enhanced.csv` — will be regenerated
- `data/processed/BM2021_22_1h_enhanced.csv` — will be regenerated

### What gets refactored

- `src/data/raw_processing.py` — Keep steps 1-5 (load, dedupe, sort, remove
  impossible, resample). Remove step 6 (`add_engineered_features`). Remove the
  NaN-dropping from resampling (NaN rows are now kept for post-split interpolation).
  The `run_pipeline()` function stops after resampling and returns a DataFrame
  with NaN gaps intact.

- `src/data/datamodule.py` — The `setup()` method now calls the new pipeline
  instead of loading a pre-made CSV. The `from_dataframes()` classmethod stays
  (used by HMM-LSTM and the benchmark). Remove CSV-loading path since we no
  longer produce monolithic CSVs.

- `src/data/preprocessing.py` — Add `compute_dco2()` as a reusable function.
  Everything else (split, scalers, sequences) stays.

- `scripts/process_raw_data.py` — Update to call the new pipeline. Still the
  main entry point for data preprocessing, but now produces split-aware output.

### New file: `src/data/pipeline.py`

Orchestrates preprocessing with a strict before/after split boundary.

```python
# Public API:

def run_preprocessing_pipeline(
    raw_dir: Path,
    variant_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: raw XLS → split-aware (train, val, test) DataFrames.

    Returns 3 DataFrames with all features computed, NaN-free, ready for
    scaling and sequencing by the DataModule.
    """

def before_split(
    raw_dir: Path,
    interval_minutes: int,
    max_ffill_minutes: int,
) -> pd.DataFrame:
    """Steps safe to run on the full dataset before splitting.

    1. Load raw XLS files           (from raw_processing)
    2. Remove duplicates            (from raw_processing)
    3. Sort by time                 (from raw_processing)
    4. Remove impossible values     (from raw_processing)
    5. Resample to regular grid     (from raw_processing, mean aggregation)
    6. Forward-fill ONLY very short gaps (<=max_ffill_minutes)
       Longer gaps stay as NaN — will be interpolated post-split.
    7. Add DETERMINISTIC temporal features (pure functions of timestamp):
       - Day_sin, Day_cos  (hour-of-day cycle)
       - Year_sin, Year_cos (day-of-year cycle)
       NOTE: dCO2 is NOT added here (depends on CO2 values → post-split).
    8. Rename columns (Temperature→TemperatureExt, etc.)

    Returns DataFrame WITH NaN gaps preserved (no dropna).
    """

def after_split_simple(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
    max_interp_gap_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Post-split processing for simple variants (A, B).

    Applied independently within each split:
    9.  Interpolate NaN gaps (linear, up to max_interp_gap_minutes)
    10. Drop rows with remaining NaN (gaps > max_interp_gap_minutes)
    11. Compute dCO2 from raw CO2 within each split

    No denoising, no outlier detection, no lags, no rolling stats.
    """

def after_split_enhanced(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
    config: dict,
    max_interp_gap_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Post-split processing for enhanced variants (C, D).

    Applied independently within each split:
    9.  Interpolate NaN gaps (linear, up to max_interp_gap_minutes)
    10. Drop rows with remaining NaN (gaps > max_interp_gap_minutes)
    11. Denoise CO2 (Savitzky-Golay, mode='nearest', per split independently)
    12. Outlier detection [train-fit]:
        - Compute IQR bounds from TRAIN split only
        - Clip values in ALL splits to those bounds
        - Use multiplier=3.0 (conservative, avoids clipping real CO2 spikes)
        - Apply to columns: CO2, TemperatureExt, Hrext, Noise, Pression
    13. Compute dCO2 from DENOISED CO2 within each split
    14. Feature engineering within each split:
        - Lag features (CO2 at t-1h, t-6h, t-24h)
        - Rolling statistics (mean, std over 1h, 6h windows)
        - Weekday sin/cos encoding
    15. Drop NaN rows from lags/rolling (first max_lag rows of each split)
    """
```

### Key implementation details

**NaN interpolation within each split (step 9):**
```python
def _interpolate_gaps(df: pd.DataFrame, max_gap_minutes: int, interval_minutes: int):
    """Linearly interpolate NaN gaps up to max_gap_minutes within a single split.

    - max_gap_minutes=60 at 5-min resolution → limit=12 consecutive NaN
    - max_gap_minutes=60 at 1h resolution → limit=1 consecutive NaN
    - Gaps exceeding the limit remain NaN (dropped in step 10)
    - Split boundaries are real observations, NOT gaps
    """
    max_consecutive = max_gap_minutes // interval_minutes
    # Only interpolate sensor columns (not temporal sin/cos which are already complete)
    sensor_cols = ["CO2", "TemperatureExt", "Hrext", "Noise", "Pression"]
    for col in sensor_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=max_consecutive)
    return df
```

**Outlier detection (step 12) — train-fit, apply to all:**
```python
def _compute_outlier_bounds(train_df, columns, multiplier=3.0):
    """Compute IQR-based clip bounds from training data only."""
    bounds = {}
    for col in columns:
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - multiplier * iqr, q3 + multiplier * iqr)
    return bounds

def _clip_outliers(df, bounds):
    """Clip values to bounds (computed from train). Applied to any split."""
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df
```

**dCO2 computation (step 11 or 13):**
```python
def _compute_dco2(df, interval_minutes):
    """Compute CO2 rate of change (ppm/hour) within a single split.

    First row of the split gets NaN (no previous value), which is
    dropped later. This avoids the 1-sample cross-split leakage
    present in the old pipeline.
    """
    hours_per_step = interval_minutes / 60.0
    df["dCO2"] = df["CO2"].diff() / hours_per_step
    return df
```

**NaN row tracking:**
```python
# After each NaN-producing step, log how many rows will be dropped:
def _log_nan_impact(split_name, df, step_name):
    n_nan_rows = df.isna().any(axis=1).sum()
    logger.info(f"  [{split_name}] After {step_name}: {n_nan_rows} rows with NaN "
                f"(will be dropped), {len(df) - n_nan_rows} clean rows remain")
```

### Pipeline flow diagram

```
Raw XLS files
  │
  ▼
before_split()
  ├─ load_raw_xls_files()
  ├─ remove_duplicates()
  ├─ sort_by_time()
  ├─ remove_impossible_values()   → NaN for bad values
  ├─ resample_to_interval()       → regular grid, ffill short gaps
  ├─ rename columns
  └─ add Day_sin/cos, Year_sin/cos (deterministic)
  │
  ▼
DataFrame WITH NaN gaps (no dropna!)
  │
  ▼
chronological_split()  →  train (70%) | val (15%) | test (15%)
  │
  ▼                          ▼                       ▼
┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   TRAIN split   │   │    VAL split     │   │   TEST split     │
└────────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                     │                       │
         ▼                     ▼                       ▼
   after_split_simple() or after_split_enhanced()
   (each split processed independently)
         │
         ├─ Interpolate NaN gaps (linear, ≤60 min)
         ├─ Drop remaining NaN rows (gaps > 60 min)
         │
         │  [Enhanced only:]
         ├─ Denoise CO2 (Savitzky-Golay, mode='nearest')
         ├─ Clip outliers (bounds from TRAIN, applied to all)
         │
         ├─ Compute dCO2 (from raw or denoised CO2)
         │
         │  [Enhanced only:]
         ├─ Add lag features (per split)
         ├─ Add rolling stats (per split)
         ├─ Add Weekday sin/cos
         ├─ Drop NaN from lags/rolling
         │
         ▼
   Clean DataFrames (NaN-free)
         │
         ▼
   CO2DataModule
   ├─ fit_scalers(train)
   ├─ apply_scalers(train, val, test)
   ├─ create_sequences()
   └─ DataLoaders
```

---

## PHASE 2: Define the 4 preprocessing variants

### Variant A: Simple 5-min
```yaml
# configs/experiments/preproc_A_simple_5min.yaml
experiment:
  name: "preproc_A"
  label: "Simple (5-min)"
  description: "Minimal preprocessing, 5-min resolution, no denoising/lags"

data:
  interval_minutes: 5
  pipeline_variant: "simple"
  max_interp_gap_minutes: 60
  samples_per_hour: 12
  scaler_type: "standard"
  feature_columns:
    - "Noise"
    - "Pression"
    - "TemperatureExt"
    - "Hrext"
    - "Day_sin"
    - "Day_cos"
    - "Year_sin"
    - "Year_cos"
    - "dCO2"
```

### Variant B: Simple 1h
```yaml
# configs/experiments/preproc_B_simple_1h.yaml
experiment:
  name: "preproc_B"
  label: "Simple (1h)"
  description: "Minimal preprocessing, 1h resolution, no denoising/lags"

data:
  interval_minutes: 60
  pipeline_variant: "simple"
  max_interp_gap_minutes: 60
  samples_per_hour: 1
  scaler_type: "standard"
  feature_columns:
    - "Noise"
    - "Pression"
    - "TemperatureExt"
    - "Hrext"
    - "Day_sin"
    - "Day_cos"
    - "Year_sin"
    - "Year_cos"
    - "dCO2"
```

### Variant C: Enhanced 5-min
```yaml
# configs/experiments/preproc_C_enhanced_5min.yaml
experiment:
  name: "preproc_C"
  label: "Enhanced (5-min)"
  description: "Denoised, outlier-clipped, with lags/rolling/weekday features, 5-min"

data:
  interval_minutes: 5
  pipeline_variant: "enhanced"
  max_interp_gap_minutes: 60
  samples_per_hour: 12
  scaler_type: "robust"
  feature_columns:
    - "Noise"
    - "Pression"
    - "TemperatureExt"
    - "Hrext"
    - "Day_sin"
    - "Day_cos"
    - "Year_sin"
    - "Year_cos"
    - "dCO2"
    - "Weekday_sin"
    - "Weekday_cos"
    - "CO2_lag_12"
    - "CO2_lag_72"
    - "CO2_lag_288"
    - "CO2_rolling_mean_12"
    - "CO2_rolling_std_12"
    - "CO2_rolling_mean_72"
    - "CO2_rolling_std_72"

preprocessing:
  denoising:
    method: "savgol"
    window_length: 11
    polyorder: 3
    mode: "nearest"
  outlier_detection:
    method: "iqr"
    multiplier: 3.0
    columns: ["CO2", "TemperatureExt", "Hrext", "Noise", "Pression"]
  lag_steps: [12, 72, 288]      # 1h, 6h, 24h at 5-min
  rolling_windows: [12, 72]     # 1h, 6h at 5-min
```

### Variant D: Enhanced 1h
```yaml
# configs/experiments/preproc_D_enhanced_1h.yaml
experiment:
  name: "preproc_D"
  label: "Enhanced (1h)"
  description: "Denoised, outlier-clipped, with lags/rolling/weekday features, 1h"

data:
  interval_minutes: 60
  pipeline_variant: "enhanced"
  max_interp_gap_minutes: 60
  samples_per_hour: 1
  scaler_type: "robust"
  feature_columns:
    - "Noise"
    - "Pression"
    - "TemperatureExt"
    - "Hrext"
    - "Day_sin"
    - "Day_cos"
    - "Year_sin"
    - "Year_cos"
    - "dCO2"
    - "Weekday_sin"
    - "Weekday_cos"
    - "CO2_lag_1"
    - "CO2_lag_6"
    - "CO2_lag_24"
    - "CO2_rolling_mean_3"
    - "CO2_rolling_std_3"
    - "CO2_rolling_mean_6"
    - "CO2_rolling_std_6"

preprocessing:
  denoising:
    method: "savgol"
    window_length: 5
    polyorder: 2
    mode: "nearest"
  outlier_detection:
    method: "iqr"
    multiplier: 3.0
    columns: ["CO2", "TemperatureExt", "Hrext", "Noise", "Pression"]
  lag_steps: [1, 6, 24]         # 1h, 6h, 24h at 1h
  rolling_windows: [3, 6]       # 3h, 6h at 1h
```

---

## PHASE 3: Run preprocessing comparison (7 models x 4 variants x 2 horizons)

### Script: `scripts/run_preprocessing_comparison.py`

```
For each variant in [A, B, C, D]:
  Run pipeline → (train_df, val_df, test_df)
  For each horizon in [1h, 24h]:
    For each model in [LSTM, CNN-LSTM, HMM-LSTM, TFT, SARIMA, XGBoost, CatBoost]:
      Train model
      Evaluate on test set
      Record: RMSE, MAE, R2, MAPE
```

Total runs: 4 variants x 2 horizons x 7 models = **56 training runs**.

### Selection criterion
- Primary metric: test RMSE (lower is better)
- Secondary metric: test MAE
- Best variant = lowest **median** RMSE across all models (median is robust to
  one model being an outlier)
- If two variants are within 5% of each other, prefer the simpler one

### Model hyperparameters: FIXED across all variants
Use the existing config files (lstm.yaml, cnn_lstm.yaml, etc.) unchanged.
Only the preprocessing and feature set changes. This ensures a fair comparison.

### Output
```
results/preprocessing_comparison/
  summary.csv                    # variant x model x horizon → metrics
  per_variant/
    A_simple_5min/               # Predictions, plots per model
    B_simple_1h/
    C_enhanced_5min/
    D_enhanced_1h/
  comparison_plot.png            # Bar chart: RMSE by variant, grouped by model
```

---

## PHASE 4: Incremental feature addition on best variant

Starting from the best variant from Phase 3, add feature groups one at a time.
Each addition is tested on ALL 7 models. Accept/reject based on validation RMSE.

### Feature groups (tested in this order, cumulative):

#### Group 1: 12h harmonic (bimodal occupancy)
```python
df["Day12h_sin"] = np.sin(2 * np.pi * hour / 12.0)
df["Day12h_cos"] = np.cos(2 * np.pi * hour / 12.0)
```
- Deterministic (function of timestamp only) → safe to compute before split
- Captures morning + evening CO2 peaks that 24h sin/cos cannot represent
- +2 features

#### Group 2: Occupancy proxy features
```python
df["is_weekend"] = (df.index.dayofweek >= 5).astype(float)
df["is_active_hours"] = ((hour >= 7) & (hour < 23)).astype(float)
```
- Deterministic → safe to compute before split
- Binary features: directly useful for tree models (XGBoost/CatBoost)
- Also helps NNs learn sharp occupancy transitions
- +2 features

#### Group 3: CO2 deviation from rolling baseline
```python
# MUST be computed WITHIN each split (rolling window is stateful)
rolling_min_24h = df["CO2"].rolling(window=samples_24h, min_periods=1).min()
df["CO2_above_baseline"] = df["CO2"] - rolling_min_24h
```
- Stateful (rolling window) → computed post-split within each split
- Rolling 24h minimum ≈ "empty room" CO2 baseline (~400-500 ppm)
- Deviation = accumulated occupancy effect, normalized across drift
- Creates NaN at start of each split (first 288 rows at 5-min) → already
  handled by the lag/rolling NaN drop logic
- +1 feature

#### Group 4: Meteorological rate features
```python
# Computed within each split, same approach as dCO2
hours_per_step = interval_minutes / 60.0
df["dPression"] = df["Pression"].diff() / hours_per_step
df["dTemperatureExt"] = df["TemperatureExt"].diff() / hours_per_step
```
- Stateful (diff) → computed post-split within each split
- Pressure tendency → weather fronts, ventilation behavior correlation
- Temperature rate → window opening/closing events
- +2 features

### Decision rule for each group
1. Train all 7 models with the candidate feature set (base + new group)
2. Compare **validation RMSE** against previous best for each model
3. **ACCEPT** if median val RMSE across models improves (even slightly)
4. **REJECT** if median val RMSE worsens or is unchanged
5. Log per-model results: if one model gets significantly worse (>10% RMSE
   increase) while others improve, note this as a model-specific sensitivity

### Guardrails
- Maximum total features: ~15 for LSTM-family, ~20 for TFT, unlimited for trees
- After each accepted group, compute correlation matrix on training data:
  if any feature pair has |r| > 0.95, drop one (keep the more interpretable one)
- Stop adding features if 2 consecutive groups are rejected

### Script: `scripts/run_feature_ablation.py`

Output:
```
results/feature_ablation/
  summary.csv                    # group x model → val_RMSE, test_RMSE
  correlation_matrices/          # Heatmaps per step
  decisions.log                  # Accept/reject log with justification
```

---

## PHASE 5: Feature importance diagnostics

Run on the final feature set from Phase 4, using models trained in that phase.

### Per-model methods:

**XGBoost:**
- Built-in: `model.feature_importances_` (gain-based)
- Permutation importance: `sklearn.inspection.permutation_importance()`

**CatBoost:**
- Built-in: `model.get_feature_importance(type="PredictionValuesChange")`
- SHAP: `model.get_feature_importance(type="ShapValues", data=test_pool)`

**TFT:**
- Built-in Variable Selection Network weights from `interpret_output()`
- Temporal attention weights for understanding which timesteps matter

**LSTM / CNN-LSTM / HMM-LSTM:**
- Permutation importance (no built-in method):
  For each feature index i, shuffle that feature across samples (not within
  sequences), re-predict, measure RMSE increase. Repeat 10 times.

**SARIMA:**
- Univariate → no feature importance. Report AR/MA coefficients and p-values.

### Script: `scripts/run_feature_importance.py`

Output:
```
results/feature_importance/
  xgboost_importance.csv
  catboost_importance.csv
  catboost_shap.csv
  tft_variable_importance.csv
  lstm_perm_importance.csv
  cnn_lstm_perm_importance.csv
  hmm_lstm_perm_importance.csv
  sarima_coefficients.csv
  combined_importance_heatmap.png   # features (rows) x models (cols)
  combined_importance_barplot.png   # per-model horizontal bar charts
```

### Action on results:
- If any feature has near-zero importance across ALL models → remove it,
  re-validate to confirm no regression
- If a feature is critical for one model but harmful for another → note for
  discussion (future work: per-model feature selection)

---

## PHASE 6: Final validation

1. Take final feature set from Phase 4, pruned by Phase 5 insights
2. Run benchmark: `python scripts/benchmark_standard_dataset.py`
   - Benchmark uses synthetic data and its own pipeline → won't break from
     preprocessing changes
   - But verify that any model code changes (input_size, etc.) still pass
3. Train all 7 models on final pipeline, both horizons (1h, 24h)
4. Produce final results table

### Output:
```
results/final/
  preprocessing_comparison.csv     # Phase 3 results
  feature_ablation.csv             # Phase 4 results
  feature_importance/              # Phase 5 diagnostics
  final_metrics.csv                # model x horizon → RMSE, MAE, R2, MAPE
  final_plots/                     # Prediction vs actual time series plots
```

---

## FILE CHANGES SUMMARY

### Files to DELETE:
- `src/data/enhanced_preprocessing.py`
- `configs/experiments/exp1_baseline.yaml`
- `configs/experiments/exp2_hourly.yaml`
- `configs/experiments/exp3_enhanced_5min.yaml`
- `configs/experiments/exp3_enhanced_1h.yaml`
- `data/processed/BM2021_22_5min.csv`
- `data/processed/BM2021_22_1h.csv`
- `data/processed/BM2021_22_5min_enhanced.csv`
- `data/processed/BM2021_22_1h_enhanced.csv`

### Files to CREATE:
- `src/data/pipeline.py` — Split-aware preprocessing orchestrator
- `configs/experiments/preproc_A_simple_5min.yaml`
- `configs/experiments/preproc_B_simple_1h.yaml`
- `configs/experiments/preproc_C_enhanced_5min.yaml`
- `configs/experiments/preproc_D_enhanced_1h.yaml`
- `scripts/run_preprocessing_comparison.py` — Phase 3 runner
- `scripts/run_feature_ablation.py` — Phase 4 runner
- `scripts/run_feature_importance.py` — Phase 5 diagnostics

### Files to MODIFY:
- `src/data/raw_processing.py`
  - Keep: load, dedupe, sort, remove_impossible, resample functions
  - Remove: `add_engineered_features()` function
  - Modify: `run_pipeline()` to stop after resampling (no feature engineering,
    no dropna). Return DataFrame with NaN gaps preserved.
  - Modify: `resample_to_interval()` — keep forward-fill of short gaps but
    do NOT drop NaN rows. The post-split interpolation handles the rest.

- `src/data/datamodule.py`
  - Modify `setup()`: call `pipeline.run_preprocessing_pipeline()` instead of
    loading a monolithic CSV. Receives pre-split DataFrames.
  - Keep `from_dataframes()` unchanged (used by HMM-LSTM and benchmark).

- `src/data/preprocessing.py`
  - Add: `compute_dco2(df, interval_minutes)` reusable function
  - Keep everything else unchanged (split, scalers, sequences)

- `scripts/process_raw_data.py`
  - Update to call the new pipeline. Still the CLI entry point.
  - Now outputs a validation log (rows per split, NaN counts, etc.)
    instead of a monolithic CSV.

### Files NOT to modify:
- `scripts/benchmark_standard_dataset.py` — must pass unchanged
- All model files in `src/models/` — input_size is already dynamic
- `src/data/dataset.py` — no changes needed
- `src/evaluation/` — no changes needed
- `src/training/` — no changes needed

---

## EXECUTION ORDER

```
Phase 1 → Refactor pipeline (delete old, create new, modify existing)
        → Unit test: pipeline produces NaN-free DataFrames for all 4 variants
        → Verify: benchmark still passes

Phase 2 → Create 4 experiment config files
        → Verify: configs load correctly and match pipeline output columns

Phase 3 → Run 56 training runs (4 variants x 2 horizons x 7 models)
        → Analyze results, select best variant
        → STOP: report results before proceeding

Phase 4 → Incremental feature addition (4 groups, on best variant only)
        → Accept/reject each group
        → STOP: report results before proceeding

Phase 5 → Feature importance diagnostics on final model set
        → Prune zero-importance features if found
        → STOP: report results before proceeding

Phase 6 → Final validation: benchmark + full training + results table
```

Each phase depends on the previous one. Do not skip ahead.
STOP after Phases 3, 4, and 5 to report results and get confirmation
before proceeding to the next phase.
