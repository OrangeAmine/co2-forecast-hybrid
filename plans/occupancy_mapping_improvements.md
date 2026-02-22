# Switching AR Occupancy Mapping — Improvement Suggestions

## Context

The Switching AR-HMM model achieves excellent dynamics detection (κ=0.70 vs rate_of_change detector at K=8) but lower consensus agreement (best bitmask κ=0.36) compared to Gaussian HMM baseline (κ=0.53). The mapping step (Otsu on μ) is the identified weak link.

## Current Approach

- Otsu thresholding on μ (per-step CO2 drift)
- Physical floor: μ ≤ 0 → unoccupied
- Bitmask ceiling: best possible κ is 0.34–0.36

## Potential Improvements

### 1. Relaxed μ Threshold
- Use μ > 0 as the occupied criterion (positive drift = CO2 generation = people)
- Simpler, more physically grounded than Otsu
- Risk: may over-classify borderline states

### 2. Multi-Feature Mapping
- Use both μ and c (or τ) together for the binary decision
- A state with moderate μ but very slow ventilation (high c) is more likely occupied
- Could use a simple decision boundary in (μ, c) space

### 3. Supervised Calibration
- Use a small labeled window to learn which states are occupied
- Semi-supervised: leverage physics-based features but calibrate the threshold from data
- Requires labeled data availability

### 4. Weighted Consensus Mapping
- Use the consensus (or a subset of detectors) on a validation set to calibrate
- Essentially finds the Otsu-equivalent threshold that maximizes κ vs consensus
- More principled than exhaustive bitmask but less interpretable than pure physics

## Key Insight

Even with optimal mapping (bitmask search), the switching AR κ ceiling is ~0.36. The fundamental issue is that AR state segmentation captures CO2 *dynamics* (rate of change) rather than CO2 *level* (which dominates the consensus through absolute_threshold and state_machine detectors). Improving the mapping alone won't bridge this gap — it would require changing the model's feature space or the consensus definition.
