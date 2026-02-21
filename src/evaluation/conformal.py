"""Split conformal prediction for calibrated uncertainty intervals.

Implements the split conformal prediction method (Vovk et al., 2005)
which wraps any point forecaster with calibrated prediction intervals:

    1. Hold out a calibration set from validation/test data
    2. Compute nonconformity scores: |y_true - y_pred| on calibration set
    3. For a desired coverage level (1 - alpha):
        quantile_level = ceil((n_cal + 1)(1 - alpha)) / n_cal
        q_hat = quantile(scores, quantile_level)
    4. Prediction interval: [y_pred - q_hat, y_pred + q_hat]

Key properties:
    - Finite-sample valid: empirical coverage >= (1 - alpha) regardless
      of the underlying distribution (no distributional assumptions)
    - Per-step calibration: each forecast step gets its own interval width,
      so later steps naturally get wider intervals
    - Model-agnostic: works as a post-hoc wrapper around any point forecast

Usage:
    conformal = SplitConformalPredictor(alpha=0.1)  # 90% coverage
    conformal.calibrate(cal_true, cal_pred)
    lower, upper = conformal.predict_intervals(test_pred)
    coverage = conformal.evaluate_coverage(test_true, test_pred)
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SplitConformalPredictor:
    """Split conformal prediction wrapper for any forecaster.

    Calibrates per-step nonconformity quantiles on a held-out calibration
    set, then produces prediction intervals for new data.

    Args:
        alpha: Miscoverage rate. Default 0.1 means 90% target coverage.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat_: np.ndarray | None = None  # (horizon,) per-step quantiles

    def calibrate(
        self,
        cal_true: np.ndarray,
        cal_pred: np.ndarray,
    ) -> "SplitConformalPredictor":
        """Calibrate on a held-out calibration set.

        Computes per-step nonconformity quantiles so each forecast step
        gets its own interval width. The finite-sample correction
        ceil((n+1)(1-alpha))/n ensures valid coverage even with small
        calibration sets.

        Args:
            cal_true: (n_cal, horizon) ground truth on calibration set.
            cal_pred: (n_cal, horizon) predictions on calibration set.

        Returns:
            Self for method chaining.
        """
        scores = np.abs(cal_true - cal_pred)  # (n_cal, horizon)
        n_cal = scores.shape[0]

        # Finite-sample valid quantile level (Vovk et al., 2005)
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)

        # Per-step quantile: later forecast steps typically have larger scores
        self.q_hat_ = np.quantile(scores, quantile_level, axis=0)  # (horizon,)

        logger.info(
            f"  Conformal calibration: n_cal={n_cal}, alpha={self.alpha}, "
            f"q_hat range=[{self.q_hat_.min():.2f}, {self.q_hat_.max():.2f}]"
        )

        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals.

        Args:
            y_pred: (n_samples, horizon) point predictions.

        Returns:
            Tuple of (lower, upper) each of shape (n_samples, horizon).
        """
        if self.q_hat_ is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        lower = y_pred - self.q_hat_
        upper = y_pred + self.q_hat_
        return lower, upper

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float | list[float]]:
        """Evaluate empirical coverage and average interval width.

        Args:
            y_true: (n_samples, horizon) ground truth.
            y_pred: (n_samples, horizon) point predictions.

        Returns:
            Dictionary with coverage and width statistics.
        """
        lower, upper = self.predict_intervals(y_pred)
        covered = (y_true >= lower) & (y_true <= upper)

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "avg_interval_width": float((upper - lower).mean()),
            "per_step_coverage": covered.mean(axis=0).tolist(),
            "per_step_width": (upper - lower).mean(axis=0).tolist(),
        }

    def save(self, path: Path) -> None:
        """Save calibration results to JSON.

        Args:
            path: Output file path.
        """
        if self.q_hat_ is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "q_hat": self.q_hat_.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
