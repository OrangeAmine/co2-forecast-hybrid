# CatBoost Interpretability Study Plan

## Overview

Deep interpretability analysis of the CatBoost forecaster. CatBoost shares tree-based
interpretability with XGBoost but offers unique advantages: native SHAP via `get_feature_importance`,
built-in feature interaction detection, and oblivious (symmetric) tree structures that are
inherently more interpretable than standard CART trees.

**Model:** `CatBoostForecaster` (`src/models/catboost_model.py`)
**Architecture:** MultiOutputRegressor wrapping CatBoostRegressor (symmetric trees, ordered boosting)
**Input:** Flattened `(lookback * n_features)` = e.g., `(24 * 10) = 240` features at 1h resolution
**Data variant:** preproc_D (Enhanced 1h)
**Horizons:** 1h (12 steps) and 24h (288 steps)
**Output directory:** `results/catboost_interpretability/`

---

## Section A: Built-in Feature Importance Analysis

**Goal:** Extract CatBoost's native importance metrics and compare its multi-type importance system.

### A1. PredictionValuesChange Importance

CatBoost's default importance: measures how much each feature changes predictions on average.

1. Extract from each sub-model:
   `model.model.estimators_[step].get_feature_importance(type="PredictionValuesChange")`
2. Reshape from flattened to `(lookback, n_features)`.

- **Plot A1: Importance heatmap** -- `(lookback x features)`, same format as XGBoost A1.
- **Plot A2: Per-feature importance (collapsed across lags)**.
- **Plot A3: Per-lag importance (collapsed across features)**.

### A2. LossFunctionChange Importance

More robust metric: measures how much the loss increases when a feature is excluded.
Requires a validation pool.

`model.get_feature_importance(type="LossFunctionChange", data=pool)`

- **Plot A4: LossFunctionChange importance bar chart** -- Top 20 features.
- **Plot A5: PredictionValuesChange vs LossFunctionChange comparison** -- Scatter plot
  of the two importance metrics. Discrepancies reveal features that change predictions
  but don't hurt accuracy (potential noise features).

### A3. Feature Interaction Strengths

CatBoost uniquely provides built-in pairwise interaction detection:
`model.get_feature_importance(type="Interaction")`

Returns pairs `(feature_i, feature_j, strength)`.

- **Plot A6: Top 20 feature interactions** -- Horizontal bar chart of interaction strengths
  with feature pair labels (e.g., "CO2_lag0 x dCO2_lag0").
- **Plot A7: Interaction matrix heatmap** -- `(top_15 x top_15)` symmetric matrix of
  pairwise interaction strengths. Reveals feature clusters that work together.

---

## Section B: SHAP Analysis

**Goal:** Exact SHAP decomposition using CatBoost's native implementation.

### B1. CatBoost Native SHAP

CatBoost has its own SHAP implementation, which is faster and more accurate than the
generic `shap` library for CatBoost models:

```python
shap_values = model.get_feature_importance(type="ShapValues", data=pool)
# Shape: (n_samples, n_features + 1), last column is base value
```

### B2. Global SHAP Plots

- **Plot B1: SHAP summary (beeswarm)** -- Top 20 features by mean |SHAP|.
- **Plot B2: SHAP importance bar chart** -- Per original feature (summed across lags).
- **Plot B3: SHAP temporal heatmap** -- `(lookback x features)` of mean |SHAP|.
- **Plot B4: SHAP temporal profile** -- Mean |SHAP| per lag position.

### B3. Per-Step SHAP Variation

- **Plot B5: SHAP importance by forecast step** -- How importance shifts across the
  12 forecast steps.
- **Plot B6: Step 1 vs Step 12 SHAP comparison**.

### B4. Local SHAP Explanations

- **Plot B7: Waterfall plots** -- 3-5 representative samples.
- **Plot B8: Force plots** -- Same samples.

### B5. SHAP Interaction Values (CatBoost-native)

CatBoost can also compute SHAP interaction values natively:
`model.get_feature_importance(type="ShapInteractionValues", data=pool)`
Shape: `(n_samples, n_features + 1, n_features + 1)`.

- **Plot B9: SHAP interaction matrix** -- Mean |SHAP interaction| for top 10 features.
- **Plot B10: SHAP interaction comparison with built-in interactions** -- Scatter plot
  comparing CatBoost's built-in `Interaction` metric to SHAP interaction values.
  Tests consistency between the two approaches.

---

## Section C: Oblivious Tree Analysis

**Goal:** Exploit CatBoost's symmetric tree structure for unique interpretability insights
not possible with XGBoost's asymmetric trees.

### C1. Oblivious Tree Structure

In CatBoost, all nodes at the same depth use the same split feature and threshold.
A tree of depth d has exactly d split conditions and 2^d leaves.

1. Extract tree structure from the model's internal representation.
2. For each tree, the split conditions form a simple lookup table:
   ```
   If feature_a > threshold_a AND feature_b > threshold_b AND ...
   -> leaf value
   ```

- **Plot C1: Split condition frequency** -- Which features appear most often as split
  conditions in the symmetric trees? Bar chart.
- **Plot C2: Split depth distribution** -- At which tree depth does each feature typically
  appear? Features at depth 0 are the most globally important.

### C2. Tree Complexity Analysis

- **Plot C3: Tree depth distribution** -- Histogram of depths across all trees.
  CatBoost's depth param (6) should dominate, but some trees may be shallower.
- **Plot C4: Leaf value distributions** -- Per-depth histogram of leaf values.
  Deeper leaves may have more extreme values.

### C3. Decision Rules

- Extract the most common decision paths (split condition combinations) from the
  first few trees.
- **Plot C5: Top 10 decision rules** -- Table listing the most frequently activated
  leaf paths with their conditions and average prediction. Human-readable rules like:
  "IF CO2_lag0 > 750 AND dCO2_lag0 > 5 AND Day_sin_lag0 > 0.3 THEN predict +45 ppm"

---

## Section D: Partial Dependence Analysis

**Goal:** Marginal effect of features on predictions.

### D1. 1D Partial Dependence

- **Plot D1: PDP grid (top 6 features)** -- Same approach as XGBoost Section C1.
- **Plot D2: CatBoost vs XGBoost PDP comparison** -- Overlay PDP curves from both models
  for top 3 features. Reveals whether they learn similar or different response shapes.

### D2. 2D Partial Dependence

- **Plot D3: 2D PDP (CO2_lag0 x dCO2_lag0)**.
- **Plot D4: 2D PDP (CO2_lag0 x Day_sin_lag0)**.

### D3. ICE Plots

- **Plot D5: ICE plots for CO2_lag0** -- Individual conditional expectation.

---

## Section E: CatBoost-Specific Analyses

### E1. Ordered Boosting Analysis

CatBoost's ordered boosting uses permutation-driven gradient computation to reduce
overfitting. We can analyze:

- **Plot E1: Training/validation loss curve** -- Over boosting iterations.
- **Plot E2: Overfitting gap** -- Train loss - val loss over iterations. Compare to
  XGBoost's gap at the same iteration count. CatBoost should show smaller gap due to
  ordered boosting.

### E2. CatBoost vs XGBoost Direct Comparison

Since both models have the same input format (flattened lookback window):
- **Plot E3: Importance correlation** -- Scatter of CatBoost vs XGBoost feature importance
  rankings. High correlation = models agree; divergence = model-specific insights.
- **Plot E4: SHAP correlation** -- Same for SHAP values (per-sample correlation between
  CatBoost and XGBoost SHAP for the same test instances).
- **Plot E5: Prediction disagreement** -- Where do CatBoost and XGBoost differ most?
  Scatter plot of |CatBoost_pred - XGBoost_pred| vs actual CO2.

### E3. Multi-Output Analysis

- **Plot E6: Per-step RMSE** -- RMSE at each forecast step.
- **Plot E7: Per-step model complexity** -- Trees, depth, leaves per sub-model.

### E4. Prediction Quality Analysis

- **Plot E8: Predictions overlay**.
- **Plot E9: Scatter with R2**.
- **Plot E10: 4-panel residual analysis**.
- **Plot E11: Error by CO2 level**.

### E5. Cross-Horizon Comparison

- **Plot E12: Metrics summary table**.
- **Plot E13: SHAP importance shift (1h vs 24h)**.
- **Plot E14: Interaction structure shift** -- Do feature interactions change at 24h?

---

## Technical Notes

### CatBoost Pool Object

CatBoost uses `Pool` objects for efficient data handling:
```python
from catboost import Pool
pool = Pool(data=X_test, label=y_test)
# Required for LossFunctionChange, ShapValues, ShapInteractionValues
```

### CatBoost Importance API

```python
# Access individual CatBoostRegressor for step i
sub_model = model.model.estimators_[i]

# Different importance types
sub_model.get_feature_importance(type="PredictionValuesChange")
sub_model.get_feature_importance(type="LossFunctionChange", data=pool)
sub_model.get_feature_importance(type="ShapValues", data=pool)
sub_model.get_feature_importance(type="Interaction")
sub_model.get_feature_importance(type="ShapInteractionValues", data=pool)
```

### SHAP Interaction Performance

- CatBoost's native `ShapInteractionValues` is O(n_samples * n_features^2 * n_trees).
- For 240 features and 500 trees: significant computation. Limit to step-1 sub-model
  and 500 test samples.

### GPU Considerations

- CatBoost was trained with `task_type="GPU"`, but SHAP computation happens on CPU.
- `get_feature_importance(type="ShapValues")` automatically runs on CPU regardless of training device.

### Dependencies

- Existing: catboost, numpy, matplotlib, sklearn
- New: `shap` (for visualization utilities -- beeswarm, waterfall plots).
  CatBoost's native SHAP can be used without the `shap` package for values,
  but `shap` provides the best visualization.

---

## Expected Output

```
results/catboost_interpretability/
  h1/
    built_in_importance/  -- Plots A1-A7
    shap_analysis/        -- Plots B1-B10 + CSV
    tree_analysis/        -- Plots C1-C5
    partial_dependence/   -- Plots D1-D5
    catboost_specific/    -- Plots E1-E7
    predictions/          -- Plots E8-E11
    metrics.json
    predictions.npz
  h24/
    (same structure)
  comparison/            -- Cross-horizon plots E12-E14
  summary.png
  study_results.json
```

## Total: ~38 plots per horizon + cross-horizon comparison
