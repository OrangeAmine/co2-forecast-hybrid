"""Cross-method consensus evaluation for occupancy detectors.

Since no ground-truth occupancy labels exist, we evaluate detector
quality through inter-detector agreement metrics:

- **Cohen's kappa**: pairwise chance-corrected agreement between two raters
- **Fleiss' kappa**: multi-rater generalization for all detectors simultaneously
- **Pairwise agreement %**: raw proportion of matching predictions
- **Majority vote**: consensus label (≥50% of detectors agree)
- **Consistency score**: per-timestep fraction of detectors agreeing

High inter-detector agreement suggests the detectors are capturing the
same underlying occupancy signal rather than producing random noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Container for cross-method consensus evaluation results.

    Attributes:
        cohens_kappa_matrix: Pairwise Cohen's kappa (n_detectors, n_detectors).
        fleiss_kappa: Fleiss' kappa for multi-rater agreement.
        agreement_matrix: Pairwise raw agreement % (n_detectors, n_detectors).
        majority_vote: Consensus binary labels (n_samples,).
        occupancy_rates: Fraction of time each detector classifies as occupied.
        detector_names: Ordered list of detector names.
        consistency_scores: Per-timestep agreement fraction (n_samples,).
    """

    cohens_kappa_matrix: np.ndarray
    fleiss_kappa: float
    agreement_matrix: np.ndarray
    majority_vote: np.ndarray
    occupancy_rates: dict[str, float]
    detector_names: list[str]
    consistency_scores: np.ndarray

    def save(self, path: Path) -> None:
        """Save consensus results to JSON.

        Args:
            path: Output JSON file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "detector_names": self.detector_names,
            "fleiss_kappa": float(self.fleiss_kappa),
            "occupancy_rates": {
                k: float(v) for k, v in self.occupancy_rates.items()
            },
            "cohens_kappa_matrix": self.cohens_kappa_matrix.tolist(),
            "agreement_matrix": self.agreement_matrix.tolist(),
            "majority_vote_occupancy_rate": float(self.majority_vote.mean()),
            "mean_consistency": float(self.consistency_scores.mean()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Consensus results saved to {path}")


class ConsensusEvaluator:
    """Evaluate inter-detector agreement without ground truth.

    Computes pairwise and multi-rater agreement metrics across
    all detectors to assess whether they capture a common signal.
    """

    def evaluate(
        self, predictions: dict[str, np.ndarray]
    ) -> ConsensusResult:
        """Run consensus evaluation across all detector outputs.

        Args:
            predictions: Dict mapping detector name to binary array (n_samples,).

        Returns:
            ConsensusResult with all agreement metrics.
        """
        names = list(predictions.keys())
        n_detectors = len(names)
        arrays = [predictions[name] for name in names]

        # Stack into matrix: (n_samples, n_detectors)
        prediction_matrix = np.column_stack(arrays)
        n_samples = prediction_matrix.shape[0]

        # --- Pairwise Cohen's kappa ---
        kappa_matrix = np.ones((n_detectors, n_detectors), dtype=np.float64)
        agreement_matrix = np.ones(
            (n_detectors, n_detectors), dtype=np.float64
        )

        for i in range(n_detectors):
            for j in range(i + 1, n_detectors):
                # Cohen's kappa: chance-corrected agreement
                # κ = (p_o - p_e) / (1 - p_e)
                # where p_o = observed agreement, p_e = expected by chance
                # Handle degenerate cases where one or both arrays are constant
                try:
                    kappa = cohen_kappa_score(arrays[i], arrays[j])
                    if np.isnan(kappa):
                        kappa = 0.0
                except Exception:
                    kappa = 0.0
                kappa_matrix[i, j] = kappa
                kappa_matrix[j, i] = kappa

                # Raw agreement: fraction of matching predictions
                agree = np.mean(arrays[i] == arrays[j])
                agreement_matrix[i, j] = agree
                agreement_matrix[j, i] = agree

        # --- Fleiss' kappa (multi-rater) ---
        fleiss = self._compute_fleiss_kappa(prediction_matrix)

        # --- Majority vote consensus ---
        # Occupied if >50% of detectors agree
        vote_fractions = prediction_matrix.mean(axis=1)
        majority_vote = (vote_fractions > 0.5).astype(np.int32)

        # --- Per-detector occupancy rates ---
        occupancy_rates = {
            name: float(arr.mean()) for name, arr in predictions.items()
        }

        # --- Per-timestep consistency score ---
        # How many detectors agree with the majority at each timestep
        # consistency = max(fraction_occupied, fraction_unoccupied)
        consistency_scores = np.maximum(
            vote_fractions, 1.0 - vote_fractions
        )

        result = ConsensusResult(
            cohens_kappa_matrix=kappa_matrix,
            fleiss_kappa=fleiss,
            agreement_matrix=agreement_matrix,
            majority_vote=majority_vote,
            occupancy_rates=occupancy_rates,
            detector_names=names,
            consistency_scores=consistency_scores,
        )

        # Log summary
        logger.info(f"Fleiss' kappa: {fleiss:.4f}")
        logger.info(
            f"Mean pairwise Cohen's kappa: "
            f"{kappa_matrix[np.triu_indices(n_detectors, k=1)].mean():.4f}"
        )
        logger.info(
            f"Majority vote occupancy rate: {majority_vote.mean():.2%}"
        )
        logger.info(
            f"Mean consistency: {consistency_scores.mean():.2%}"
        )

        return result

    @staticmethod
    def _compute_fleiss_kappa(ratings: np.ndarray) -> float:
        """Compute Fleiss' kappa for multiple binary raters.

        Fleiss' kappa generalizes Cohen's kappa to >2 raters. For binary
        classification (occupied/unoccupied), it measures how much
        agreement exceeds what would be expected by chance alone.

        Formula:
            κ = (P̄ - P̄_e) / (1 - P̄_e)

            where P̄  = mean observed agreement per sample
                  P̄_e = expected agreement by chance

        Args:
            ratings: Binary matrix (n_samples, n_raters).

        Returns:
            Fleiss' kappa statistic (-1 to 1).
        """
        n_samples, n_raters = ratings.shape

        # Count how many raters assigned each category per sample
        # For binary: categories are 0 and 1
        n_ones = ratings.sum(axis=1)   # count of 1s per sample
        n_zeros = n_raters - n_ones     # count of 0s per sample

        # P_i = (1 / (n*(n-1))) * sum_j(n_ij * (n_ij - 1))
        # For binary: P_i = (n_ones*(n_ones-1) + n_zeros*(n_zeros-1)) / (n*(n-1))
        p_i = (
            n_ones * (n_ones - 1) + n_zeros * (n_zeros - 1)
        ) / (n_raters * (n_raters - 1))

        p_bar = p_i.mean()

        # P_e = sum_j(p_j^2) where p_j = proportion of all ratings in category j
        total_ratings = n_samples * n_raters
        p_one = n_ones.sum() / total_ratings
        p_zero = n_zeros.sum() / total_ratings
        p_e = p_one**2 + p_zero**2

        if abs(1.0 - p_e) < 1e-10:
            # Perfect agreement or all same category — kappa undefined
            return 1.0

        kappa = (p_bar - p_e) / (1.0 - p_e)
        return float(kappa)
