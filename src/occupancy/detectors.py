"""Seven occupancy detectors for CO2 time series.

Each detector implements a different physical or statistical heuristic
for inferring room occupancy from indoor CO2 measurements. All share
a common ``BaseDetector`` interface returning binary arrays.

Detectors:
    1. AbsoluteThresholdDetector  — CO2 > fixed threshold
    2. RateOfChangeDetector       — smoothed dCO2 > positive threshold
    3. AdaptiveThresholdDetector  — CO2 > per-hour learned threshold
    4. HybridDetector             — OR-gate of (1) and (2)
    5. StateMachineDetector       — latched FSM: onset → sustain → timeout
    6. DiarraDetector             — four-state rules (Diarra et al., 2023)
    7. SwitchingARDetector        — physics-informed switching AR-HMM

References:
    Diarra et al., "A Methodology for Indoor CO2-Based Occupancy Detection,"
    Sensors 2023, 23(23):9603.  DOI: 10.3390/s23239603

    Esmaieeli-Sikaroudi et al., "Physics-Informed Building Occupancy Detection:
    a Switching Process with Markov Regime," arXiv:2409.11743 (2024).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """Abstract base for occupancy detectors.

    Subclasses must implement ``detect()`` which returns a binary array
    (1 = occupied, 0 = unoccupied) of length ``len(df)``.
    """

    name: str = "base"

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """Run detection on a DataFrame.

        Args:
            df: Must contain at least 'CO2' column. Some detectors also
                require 'dCO2', 'Noise', or datetime information.

        Returns:
            Binary int array of shape (n_samples,).
        """

    def _ensure_dco2(self, df: pd.DataFrame) -> pd.Series:
        """Return dCO2 column, computing it if absent."""
        if "dCO2" in df.columns:
            return df["dCO2"]
        # Finite-difference approximation (backward difference)
        return df["CO2"].diff().fillna(0.0)

    def _get_hours(self, df: pd.DataFrame) -> np.ndarray:
        """Extract hour-of-day from DataFrame index or 'datetime' column."""
        if "datetime" in df.columns:
            return pd.to_datetime(df["datetime"]).dt.hour.values
        return pd.DatetimeIndex(df.index).hour  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. Absolute Threshold Detector
# ---------------------------------------------------------------------------

class AbsoluteThresholdDetector(BaseDetector):
    """Occupied when CO2 exceeds a fixed threshold.

    The simplest detector — uses a single global threshold. Effective
    when ventilation and baseline are stable, but sensitive to seasonal
    drift in outdoor CO2 and HVAC changes.

    Decision rule:
        occupied(t) = 1  if CO2(t) > co2_threshold_ppm
    """

    name = "absolute_threshold"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        cfg = config.get("absolute_threshold", {})
        self.threshold = cfg.get("co2_threshold_ppm", 500)

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        result = (df["CO2"].values > self.threshold).astype(np.int32)
        logger.info(
            f"  [{self.name}] {result.sum()}/{len(result)} occupied "
            f"(threshold={self.threshold} ppm)"
        )
        return result


# ---------------------------------------------------------------------------
# 2. Rate of Change Detector
# ---------------------------------------------------------------------------

class RateOfChangeDetector(BaseDetector):
    """Occupied when the smoothed CO2 derivative is positive.

    Rising CO2 implies active generation by occupants. A rolling mean
    smooths transient sensor noise. This detector captures *arrival*
    events well but misses sustained low-activity presence (steady-state).

    Decision rule:
        dCO2_smooth(t) = rolling_mean(dCO2, window)
        occupied(t) = 1  if dCO2_smooth(t) > dco2_threshold
    """

    name = "rate_of_change"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        cfg = config.get("rate_of_change", {})
        self.dco2_threshold = cfg.get("dco2_threshold_ppm_per_hour", 5.0)
        self.smoothing_window = cfg.get("smoothing_window", 3)

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        dco2 = self._ensure_dco2(df)
        # Smooth with rolling mean to reduce false positives from noise
        dco2_smooth = dco2.rolling(
            window=self.smoothing_window, min_periods=1, center=True
        ).mean()
        result = (dco2_smooth.values > self.dco2_threshold).astype(np.int32)
        logger.info(
            f"  [{self.name}] {result.sum()}/{len(result)} occupied "
            f"(dCO2_threshold={self.dco2_threshold} ppm/h, "
            f"window={self.smoothing_window})"
        )
        return result


# ---------------------------------------------------------------------------
# 3. Adaptive Threshold Detector
# ---------------------------------------------------------------------------

class AdaptiveThresholdDetector(BaseDetector):
    """Occupied when CO2 exceeds a per-hour learned threshold.

    Learns the typical CO2 distribution for each hour of the day from
    training data. The threshold is set at a high percentile + margin,
    so only *abnormally elevated* CO2 (above normal diurnal variation)
    triggers occupancy.

    Must call ``fit(train_df)`` before ``detect()``.

    Decision rule:
        threshold(h) = quantile(CO2_train[hour=h], baseline_percentile) + margin
        occupied(t)  = 1  if CO2(t) > threshold(hour(t))
    """

    name = "adaptive_threshold"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        cfg = config.get("adaptive_threshold", {})
        self.baseline_percentile = cfg.get("baseline_percentile", 75)
        self.margin = cfg.get("margin_ppm", 50)
        self.hourly_thresholds_: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Learn per-hour CO2 thresholds from training data.

        Args:
            train_df: Training DataFrame with CO2 and datetime/index.
        """
        hours = self._get_hours(train_df)
        # Compute per-hour percentile of CO2 distribution
        hourly_q = train_df.assign(hour=hours).groupby("hour")["CO2"].quantile(
            self.baseline_percentile / 100.0
        )
        self.hourly_thresholds_ = hourly_q + self.margin
        logger.info(
            f"  [{self.name}] Fitted per-hour thresholds "
            f"(p{self.baseline_percentile} + {self.margin} ppm): "
            f"range [{self.hourly_thresholds_.min():.0f}, "
            f"{self.hourly_thresholds_.max():.0f}] ppm"
        )

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        if self.hourly_thresholds_ is None:
            raise RuntimeError(
                "AdaptiveThresholdDetector must be fit() on training data first."
            )
        hours = self._get_hours(df)
        # Map each sample's hour to its learned threshold
        thresholds = pd.Series(hours).map(self.hourly_thresholds_).values
        result = (df["CO2"].values > thresholds).astype(np.int32)
        logger.info(
            f"  [{self.name}] {result.sum()}/{len(result)} occupied "
            f"(percentile={self.baseline_percentile}, margin={self.margin} ppm)"
        )
        return result


# ---------------------------------------------------------------------------
# 4. Hybrid Detector (OR-gate)
# ---------------------------------------------------------------------------

class HybridDetector(BaseDetector):
    """OR-gate fusion of absolute threshold and rate-of-change.

    Combines the *level-based* sensitivity of the absolute threshold
    with the *dynamics-based* sensitivity of the rate-of-change
    detector. Reduces false negatives at the cost of slightly higher
    false positive rate.

    Decision rule:
        occupied(t) = 1  if (CO2(t) > threshold) OR (dCO2_smooth(t) > dco2_threshold)
    """

    name = "hybrid"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.abs_detector = AbsoluteThresholdDetector(config)
        self.roc_detector = RateOfChangeDetector(config)

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        abs_result = self.abs_detector.detect(df)
        roc_result = self.roc_detector.detect(df)
        # OR-gate: occupied if EITHER condition is met
        result = np.maximum(abs_result, roc_result)
        logger.info(
            f"  [{self.name}] {result.sum()}/{len(result)} occupied "
            f"(absolute={abs_result.sum()}, rate_of_change={roc_result.sum()}, "
            f"union={result.sum()})"
        )
        return result


# ---------------------------------------------------------------------------
# 5. State Machine Detector
# ---------------------------------------------------------------------------

class StateMachineDetector(BaseDetector):
    """Latched finite-state machine mimicking occupancy lifecycle.

    Models the physical arrival → presence → departure cycle:

    States:
        UNOCCUPIED: default state, no one present
        ONSET:      dCO2 rising — someone may be arriving
        OCCUPIED:   sustained presence, CO2 above sustain threshold
        DECAY:      CO2 dropping but not yet below threshold long enough

    Transitions:
        UNOCCUPIED → ONSET:    dCO2 > onset_threshold for min_onset_steps
        ONSET → OCCUPIED:      after min_onset_steps of sustained rising dCO2
        OCCUPIED → DECAY:      CO2 drops below sustain_threshold
        DECAY → UNOCCUPIED:    timeout_steps consecutive below threshold
        DECAY → OCCUPIED:      CO2 rises back above sustain_threshold

    The latch prevents rapid on/off toggling during brief ventilation
    events or door openings.
    """

    name = "state_machine"

    # FSM states
    UNOCCUPIED = 0
    ONSET = 1
    OCCUPIED = 2
    DECAY = 3

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        cfg = config.get("state_machine", {})
        self.onset_threshold = cfg.get("onset_dco2_threshold", 5.0)
        self.sustain_threshold = cfg.get("sustain_co2_threshold", 500)
        self.timeout_steps = cfg.get("timeout_steps", 6)
        self.min_onset_steps = cfg.get("min_onset_steps", 2)

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        co2 = df["CO2"].values
        dco2 = self._ensure_dco2(df).values
        n = len(co2)

        state = self.UNOCCUPIED
        onset_count = 0
        decay_count = 0
        result = np.zeros(n, dtype=np.int32)

        for t in range(n):
            if state == self.UNOCCUPIED:
                if dco2[t] > self.onset_threshold:
                    onset_count += 1
                    if onset_count >= self.min_onset_steps:
                        state = self.OCCUPIED
                        result[t] = 1
                    else:
                        state = self.ONSET
                        result[t] = 0
                else:
                    onset_count = 0
                    result[t] = 0

            elif state == self.ONSET:
                if dco2[t] > self.onset_threshold:
                    onset_count += 1
                    if onset_count >= self.min_onset_steps:
                        state = self.OCCUPIED
                        result[t] = 1
                    else:
                        result[t] = 0
                else:
                    # Rising CO2 not sustained — return to unoccupied
                    state = self.UNOCCUPIED
                    onset_count = 0
                    result[t] = 0

            elif state == self.OCCUPIED:
                if co2[t] < self.sustain_threshold:
                    # CO2 dropped below sustain level — start decay timer
                    state = self.DECAY
                    decay_count = 1
                    result[t] = 1  # still occupied during decay
                else:
                    result[t] = 1

            elif state == self.DECAY:
                if co2[t] >= self.sustain_threshold:
                    # CO2 back above threshold — re-latch as occupied
                    state = self.OCCUPIED
                    decay_count = 0
                    result[t] = 1
                else:
                    decay_count += 1
                    if decay_count >= self.timeout_steps:
                        # Timed out — transition to unoccupied
                        state = self.UNOCCUPIED
                        decay_count = 0
                        onset_count = 0
                        result[t] = 0
                    else:
                        result[t] = 1  # still in grace period

        logger.info(
            f"  [{self.name}] {result.sum()}/{len(result)} occupied "
            f"(onset_dCO2={self.onset_threshold}, "
            f"sustain_CO2={self.sustain_threshold}, "
            f"timeout={self.timeout_steps})"
        )
        return result


# ---------------------------------------------------------------------------
# 6. Diarra et al. Detector
# ---------------------------------------------------------------------------

class DiarraDetector(BaseDetector):
    """Four-state classification from Diarra et al. (Sensors 2023).

    Uses three signals — CO2 concentration, CO2 derivative sign, and
    noise level — to classify each timestep into one of four states:

    State 1 — Prolonged absence:
        CO2 low, dCO2 <= 0, noise low → binary 0
        Room empty and decaying towards baseline.

    State 2 — Presence:
        CO2 high OR dCO2 > 0 OR noise high → binary 1
        At least one indicator of human activity.

    State 3 — Absence:
        CO2 declining (dCO2 < 0), noise low → binary 0
        Recent departure, CO2 still decaying.

    State 4 — Inactive presence:
        CO2 high, dCO2 <= 0, noise low → binary 1
        People present but inactive (e.g., sleeping, quiet work).

    Reference:
        Diarra et al., "A Methodology for Indoor CO2-Based Occupancy
        Detection," Sensors 2023, 23(23):9603.
    """

    name = "diarra"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        cfg = config.get("diarra", {})
        self.co2_high = cfg.get("co2_high_ppm", 500)
        self.dco2_positive = cfg.get("dco2_positive_threshold", 0.0)
        self.noise_high = cfg.get("noise_high_db", 45.0)

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        co2 = df["CO2"].values
        dco2 = self._ensure_dco2(df).values

        # Noise is optional — if absent, disable noise-based rules
        has_noise = "Noise" in df.columns
        if has_noise:
            noise = df["Noise"].values
        else:
            logger.warning(
                f"  [{self.name}] 'Noise' column not found — "
                "noise-based rules disabled"
            )
            noise = np.zeros(len(co2))

        n = len(co2)
        states = np.zeros(n, dtype=np.int32)
        binary = np.zeros(n, dtype=np.int32)

        for t in range(n):
            co2_high = co2[t] > self.co2_high
            dco2_rising = dco2[t] > self.dco2_positive
            noise_high = noise[t] > self.noise_high if has_noise else False

            if co2_high or dco2_rising or noise_high:
                if co2_high and not dco2_rising and not noise_high:
                    # State 4: Inactive presence — CO2 elevated but no active signs
                    states[t] = 4
                    binary[t] = 1
                else:
                    # State 2: Active presence — at least one dynamic indicator
                    states[t] = 2
                    binary[t] = 1
            else:
                if dco2[t] < -abs(self.dco2_positive):
                    # State 3: Absence — CO2 actively declining
                    states[t] = 3
                    binary[t] = 0
                else:
                    # State 1: Prolonged absence — stable low CO2
                    states[t] = 1
                    binary[t] = 0

        # Store states for potential analysis
        self.states_ = states

        state_counts = {s: int((states == s).sum()) for s in [1, 2, 3, 4]}
        logger.info(
            f"  [{self.name}] {binary.sum()}/{len(binary)} occupied "
            f"(states: {state_counts})"
        )
        return binary


# ---------------------------------------------------------------------------
# 7. Switching AR Detector (Physics-Informed)
# ---------------------------------------------------------------------------

class SwitchingARDetector(BaseDetector):
    """Physics-informed switching AR-HMM occupancy detector.

    Models CO2 as a switching autoregressive process where each hidden
    state encodes a (ventilation regime, occupancy level) pair.  The
    AR(1) structure directly captures the CO2 mass-balance ODE:

        y_t = c_k · y_{t-1} + μ_k + ε_t

    where c_k = exp(-Δt/τ_k) encodes ventilation and μ_k encodes
    occupancy-driven CO2 generation. States with high effective
    generation rate μ_k/(1-c_k) are labelled "occupied".

    Must call ``fit(train_df)`` before ``detect()``.

    Reference:
        Esmaieeli-Sikaroudi et al. (2024), arXiv:2409.11743
    """

    name = "switching_ar"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        cfg = config.get("switching_ar", {})
        self.n_states = cfg.get("n_states", 6)
        self.co2_ambient = cfg.get("co2_ambient_ppm", 420.0)
        self.delta_t_hours = cfg.get("delta_t_hours", 1.0)
        self.n_iter = cfg.get("n_iter", 200)
        self.constrain_ar = cfg.get("constrain_ar", True)
        self._model = None
        self._result = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the switching AR-HMM on training CO2 data.

        Args:
            train_df: Training DataFrame with 'CO2' column.
        """
        from src.models.switching_ar import SwitchingARHMM

        self._model = SwitchingARHMM(
            n_states=self.n_states,
            co2_ambient=self.co2_ambient,
            delta_t_hours=self.delta_t_hours,
            n_iter=self.n_iter,
            constrain_ar=self.constrain_ar,
        )
        co2_train = train_df["CO2"].values
        self._result = self._model.fit(co2_train)
        logger.info(
            f"  [{self.name}] Fitted on {len(co2_train)} training samples, "
            f"converged={self._result.converged}, "
            f"LL={self._result.log_likelihood:.1f}"
        )

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None or not self._model._is_fitted:
            raise RuntimeError(
                "SwitchingARDetector must be fit() on training data first."
            )

        co2 = df["CO2"].values
        # Run inference on test data using fitted parameters
        result = self._model.fit(co2)
        self._result = result

        occ = result.occupancy_binary
        logger.info(
            f"  [{self.name}] {occ.sum()}/{len(occ)} occupied "
            f"(K={self.n_states}, converged={result.converged})"
        )
        return occ


# ---------------------------------------------------------------------------
# Helper: run all detectors
# ---------------------------------------------------------------------------

def run_all_detectors(
    df: pd.DataFrame,
    config: dict,
    train_df: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Run all 6 occupancy detectors on a DataFrame.

    Args:
        df: Test/evaluation DataFrame with CO2 (and optionally dCO2, Noise).
        config: Detector configuration dict (from occupancy.yaml 'detectors' key).
        train_df: Training DataFrame for fitting AdaptiveThresholdDetector.
            If None, the adaptive detector is skipped.

    Returns:
        Dictionary mapping detector name to binary occupancy array.
    """
    results: dict[str, np.ndarray] = {}

    detectors: list[BaseDetector] = [
        AbsoluteThresholdDetector(config),
        RateOfChangeDetector(config),
        HybridDetector(config),
        StateMachineDetector(config),
        DiarraDetector(config),
    ]

    # AdaptiveThresholdDetector requires fitting on training data
    adaptive = AdaptiveThresholdDetector(config)
    if train_df is not None:
        adaptive.fit(train_df)
        detectors.insert(2, adaptive)  # insert after rate_of_change
    else:
        logger.warning(
            "No training data provided — skipping AdaptiveThresholdDetector"
        )

    # SwitchingARDetector requires fitting on training data
    if config.get("switching_ar", {}).get("enabled", False) and train_df is not None:
        switching_ar = SwitchingARDetector(config)
        switching_ar.fit(train_df)
        detectors.append(switching_ar)

    for detector in detectors:
        logger.info(f"Running detector: {detector.name}")
        results[detector.name] = detector.detect(df)

    return results
