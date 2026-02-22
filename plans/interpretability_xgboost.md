# XGBoost Interpretability Study Plan

## Overview

Deep interpretability analysis of the XGBoost forecaster, leveraging tree-based models'
unique advantages: exact SHAP values (TreeSHAP), built-in importance metrics, partial
dependence, and inspectable decision trees. This study exploits the flattened
`(lookback * n_features)` input structure to create both per-feature and per-lag attribution maps.

**Model:** `XGBoostForecaster` (`src/models/xgboost_model.py`)
**Architecture:** MultiOutputRegressor wrapping XGBRegressor (one tree ensemble per forecast step)
**Input:** Flattened `(lookback * n_features)` = e.g., `(24 * 10) = 240` features at 1h resolution
**Data variant:** preproc_D (Enhanced 1h)
**Horizons:** 1h (12 steps) and 24h (288 steps)
**Output directory:** `results/xgboost_interpretability/`

---

## Section A: Built-in Feature Importance Analysis

**Goal:** Extract and visualize XGBoost's native importance metrics across the flattened
feature space, then reshape to reveal temporal and feature-level patterns.

### A1. Gain-Based Importance

XGBoost provides `feature_importances_` (gain-based) for each estimator in the
MultiOutputRegressor. Since there are 12 (or 288) independent sub-models:

1. Extract importance from each sub-model: `model.model.estimators_[step].feature_importances_`
2. Average across forecast steps for aggregate importance.
3. **Reshape** from `(lookback * n_features,)` to `(lookback, n_features)` matrix.

- **Plot A1: Aggregated gain importance heatmap** -- `(lookback x features)` showing which
  lag positions and features are most used for splits. Color = mean gain importance.
- **Plot A2: Per-feature importance (collapsed across lags)** -- Horizontal bar chart
  summing importance across all lag positions per feature.
- **Plot A3: Per-lag importance (collapsed across features)** -- Bar chart summing
  importance across features at each lag position. Shows temporal attention pattern.
- **Plot A4: Per-step importance variation** -- Heatmap `(forecast_step x n_features)` of
  feature importance. Does step 1 (next hour) use different features than step 12?

### A2. Additional Importance Types

XGBoost supports multiple importance types:
- `weight`: number of times feature is used in splits
- `gain`: average gain from splits using the feature
- `cover`: average number of samples affected by splits on the feature
- `total_gain`: total gain from all splits

- **Plot A5: Importance type comparison** -- 4-panel heatmap comparing weight/gain/cover/total_gain
  for top 20 features. Different metrics can give different rankings.

---

## Section B: SHAP Analysis (TreeSHAP)

**Goal:** Exact Shapley value decomposition for every prediction, enabling both global
and local interpretability.

### B1. Global SHAP Analysis

1. Compute SHAP values using `shap.TreeExplainer(model)` for each sub-model.
   TreeSHAP is exact and polynomial-time for tree ensembles.
2. SHAP output shape per sub-model: `(n_test_samples, lookback * n_features)`.
3. Reshape to `(n_test_samples, lookback, n_features)` for 3D attribution.

- **Plot B1: SHAP summary (beeswarm)** -- Top 20 flattened features sorted by mean |SHAP|.
  Each dot = one sample, x = SHAP value, color = feature value.
- **Plot B2: SHAP importance bar chart** -- Mean |SHAP| per original feature (summed across
  lags). Direct comparison to gradient importance from neural models.
- **Plot B3: SHAP temporal heatmap** -- `(lookback x features)` of mean |SHAP|. Same format
  as neural model gradient heatmaps, enabling cross-model comparison.
- **Plot B4: SHAP temporal profile** -- Mean |SHAP| per lag position (summed across features).
  Which past hours drive XGBoost's predictions?

### B2. Per-Step SHAP Variation

Since MultiOutputRegressor trains independent models per forecast step:
- **Plot B5: SHAP importance by forecast step** -- Line plot of top 5 features' SHAP
  importance across the 12 forecast steps. Do features gain/lose importance for longer horizons?
- **Plot B6: Step 1 vs Step 12 SHAP comparison** -- Side-by-side beeswarm plots.

### B3. Local SHAP Explanations

Select 3-5 representative test samples (one per CO2 regime + one with large error):
- **Plot B7: Waterfall plots** -- SHAP waterfall for individual predictions showing how
  each feature pushes the prediction up/down from the base value.
- **Plot B8: Force plots** -- SHAP force plots for the same samples.

### B4. SHAP Interaction Analysis

- Compute SHAP interaction values: `shap.TreeExplainer(model).shap_interaction_values(X)`
  Shape: `(n_samples, n_features, n_features)`. Only feasible for a single sub-model.
- **Plot B9: SHAP interaction heatmap** -- Top 10x10 feature interaction matrix.
  Reveals which feature pairs have synergistic effects (e.g., CO2_lag_0 x dCO2_lag_0).

---

## Section C: Partial Dependence Analysis

**Goal:** Show the marginal effect of each feature on predictions, holding others constant.

### C1. 1D Partial Dependence Plots

For the top 6 most important features:
- **Plot C1: Partial dependence grid** -- 2x3 grid of PDP curves showing how the predicted
  CO2 changes as each feature varies across its observed range.
  Most interesting: CO2 at lag 0 (most recent value), dCO2 at lag 0, and temporal features.

### C2. 2D Partial Dependence

- **Plot C2: 2D PDP (CO2_lag0 x dCO2_lag0)** -- Contour plot showing interaction surface.
  Reveals whether XGBoost's response to current CO2 depends on the rate of change.
- **Plot C3: 2D PDP (CO2_lag0 x Day_sin_lag0)** -- Shows time-of-day modulation of the
  CO2 response.

### C3. Individual Conditional Expectation (ICE)

- **Plot C4: ICE plots for CO2_lag0** -- Individual lines (one per sample) showing
  how predictions change. Reveals heterogeneity that PDP averages mask.
  If lines cross, there's an interaction effect.

---

## Section D: Tree Structure Analysis

**Goal:** Inspect the actual decision trees to understand XGBoost's learned rules.

### D1. Tree Visualization

- Export 3 representative trees: the first tree, the most important tree (highest gain),
  and the last tree from the step-1 sub-model.
- **Plot D1: Decision tree diagrams** -- Using `xgboost.plot_tree()` or graphviz export.
  Annotate split features with their original names (feature_lag format).

### D2. Split Statistics

- Collect split statistics across all trees in the step-1 sub-model:
  - Most frequently split features
  - Distribution of split thresholds for each feature
  - Average depth at which each feature appears

- **Plot D2: Split feature frequency** -- Bar chart of how often each feature is used
  for splits (top 20).
- **Plot D3: Split threshold distributions** -- For top 5 features, histogram of split
  thresholds. Shows the decision boundaries XGBoost learned.
- **Plot D4: Split depth distribution** -- For top 5 features, histogram of the tree depth
  at which they appear. Features split near the root are globally important.

### D3. Leaf Analysis

- **Plot D5: Leaf value distribution** -- Histogram of leaf weights across all trees.
  Narrow distribution = conservative predictions; wide = diverse predictions.
- **Plot D6: Number of leaves per tree over boosting rounds** -- Shows model complexity growth.

---

## Section E: XGBoost-Specific Analyses

### E1. Boosting Dynamics

- **Plot E1: Training/validation loss curve** -- If stored during training, plot the
  MSE over boosting rounds. Shows convergence and early stopping point.
- **Plot E2: Feature importance evolution** -- Track how the top 10 features' importance
  changes across boosting rounds (early trees vs late trees).

### E2. Multi-Output Comparison

- **Plot E3: Per-step model complexity** -- Number of trees, average depth, total leaves
  for each of the 12 (or 288) sub-models.
- **Plot E4: Per-step RMSE** -- RMSE at each forecast step. Shows error growth with
  horizon within a single model run.

### E3. Prediction Quality Analysis

- **Plot E5: Predictions overlay**.
- **Plot E6: Scatter with R2**.
- **Plot E7: 4-panel residual analysis**.
- **Plot E8: Error by CO2 level**.

### E4. Cross-Horizon Comparison

- **Plot E9: Metrics summary table**.
- **Plot E10: SHAP importance shift between 1h and 24h**.
- **Plot E11: Tree complexity comparison** -- Do 24h sub-models need deeper/more trees?

---

## Technical Notes

### Feature Name Mapping

The flattened feature vector has `lookback * n_features` entries. Map them back:
```python
def get_feature_name(flat_idx, lookback, feature_names):
    feat_idx = flat_idx % len(feature_names)
    lag_idx = flat_idx // len(feature_names)
    return f"{feature_names[feat_idx]}_lag{lag_idx}"
```

### SHAP Performance

- TreeSHAP on XGBoost is fast (exact, polynomial time).
- For `n_test=1000` samples and `240` features: ~10 seconds per sub-model.
- Interaction values are O(n_features^2) per sample -- only compute for step 1 sub-model
  and limit to 500 samples.

### MultiOutputRegressor Access

```python
# Access individual XGBRegressor for step i
sub_model = model.model.estimators_[i]
# sub_model is an XGBRegressor with full API access
sub_model.feature_importances_
sub_model.get_booster().get_score(importance_type='gain')
```

### Dependencies

- Existing: xgboost, numpy, matplotlib, sklearn
- New: `shap` (pip install shap) -- required for SHAP analysis
- Optional: `graphviz` for tree visualization (can fallback to matplotlib)

---

## Expected Output

```
results/xgboost_interpretability/
  h1/
    built_in_importance/  -- Plots A1-A5
    shap_analysis/        -- Plots B1-B9 + CSV
    partial_dependence/   -- Plots C1-C4
    tree_structure/       -- Plots D1-D6
    xgboost_specific/     -- Plots E1-E4
    predictions/          -- Plots E5-E8
    metrics.json
    predictions.npz
  h24/
    (same structure)
  comparison/            -- Cross-horizon plots E9-E11
  summary.png
  study_results.json
```

## Total: ~35 plots per horizon + cross-horizon comparison
