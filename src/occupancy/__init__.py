"""Occupancy inference from CO2 signals.

Provides 6 rule-based occupancy detection methods that can be applied
to both actual and predicted CO2 time series, with cross-method
consensus evaluation (no ground truth required).

Detectors:
    1. AbsoluteThresholdDetector — fixed CO2 threshold
    2. RateOfChangeDetector — smoothed dCO2 threshold
    3. AdaptiveThresholdDetector — per-hour learned thresholds
    4. HybridDetector — OR-gate fusion of threshold + rate-of-change
    5. StateMachineDetector — latched FSM with onset/sustain/timeout
    6. DiarraDetector — four-state classification (Diarra et al., 2023)
"""

from src.occupancy.detectors import (
    AbsoluteThresholdDetector,
    AdaptiveThresholdDetector,
    BaseDetector,
    DiarraDetector,
    HybridDetector,
    RateOfChangeDetector,
    StateMachineDetector,
    run_all_detectors,
)
from src.occupancy.evaluation import ConsensusEvaluator, ConsensusResult

__all__ = [
    "AbsoluteThresholdDetector",
    "AdaptiveThresholdDetector",
    "BaseDetector",
    "ConsensusEvaluator",
    "ConsensusResult",
    "DiarraDetector",
    "HybridDetector",
    "RateOfChangeDetector",
    "StateMachineDetector",
    "run_all_detectors",
]
