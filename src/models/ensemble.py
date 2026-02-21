"""Ensemble methods for combining multiple forecaster predictions.

Implements two post-hoc strategies that work on pre-computed predictions
(no coupling to specific model APIs):

1. WeightedAverageEnsemble: Optimizes convex combination weights on
   validation predictions to minimize MSE. Weights are constrained to
   be non-negative and sum to 1.

2. StackingEnsemble: Trains one Ridge regression meta-learner per
   forecast step on stacked model predictions. Ridge regularization
   handles the collinearity between correlated model outputs.

Both strategies are fit on validation set predictions and evaluated on
test set predictions. Models must be pre-trained and predictions saved
as numpy arrays.

Usage:
    # Load predictions from N models
    val_preds = [np.load(f"model_{i}/val_predictions.npz")["y_pred"] for i in range(N)]
    test_preds = [np.load(f"model_{i}/test_predictions.npz")["y_pred"] for i in range(N)]
    val_true = np.load("model_0/val_predictions.npz")["y_true"]

    # Weighted average
    ensemble = WeightedAverageEnsemble()
    ensemble.fit(val_preds, val_true)
    y_test_ensemble = ensemble.predict(test_preds)

    # Stacking
    stacker = StackingEnsemble(alpha=1.0)
    stacker.fit(val_preds, val_true)
    y_test_stacked = stacker.predict(test_preds)
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class WeightedAverageEnsemble:
    """Optimizes convex combination weights on validation predictions.

    Given N model predictions P_1, ..., P_N on the validation set:
        y_hat = sum(w_i * P_i)  subject to w_i >= 0, sum(w_i) = 1

    The non-negativity constraint prevents short-selling models (which
    would amplify errors), and the sum-to-one constraint ensures the
    ensemble output is on the same scale as individual models.

    Weights are optimized using SLSQP (Sequential Least Squares
    Programming) to minimize MSE on the validation set.
    """

    def __init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.model_names_: list[str] | None = None

    def fit(
        self,
        val_predictions: list[np.ndarray],
        val_true: np.ndarray,
        model_names: list[str] | None = None,
    ) -> "WeightedAverageEnsemble":
        """Optimize ensemble weights on validation data.

        Args:
            val_predictions: List of (n_val, horizon) arrays, one per model.
            val_true: (n_val, horizon) ground truth.
            model_names: Optional list of model names for logging.

        Returns:
            Self for method chaining.
        """
        n_models = len(val_predictions)
        self.model_names_ = model_names or [f"model_{i}" for i in range(n_models)]

        # Stack: (n_models, n_val, horizon)
        P = np.stack(val_predictions, axis=0)

        def objective(w: np.ndarray) -> float:
            w_expanded = w.reshape(-1, 1, 1)
            y_hat = (w_expanded * P).sum(axis=0)
            return float(np.mean((y_hat - val_true) ** 2))

        # Constraints: sum(w) = 1, w_i >= 0
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * n_models
        x0 = np.ones(n_models) / n_models  # Equal weights as starting point

        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        self.weights_ = result.x

        for name, w in zip(self.model_names_, self.weights_):
            logger.info(f"  Ensemble weight [{name}]: {w:.4f}")

        return self

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Apply learned weights to new predictions.

        Args:
            predictions: List of (n_samples, horizon) arrays, one per model.

        Returns:
            Weighted ensemble predictions of shape (n_samples, horizon).
        """
        if self.weights_ is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        P = np.stack(predictions, axis=0)
        w = self.weights_.reshape(-1, 1, 1)
        return (w * P).sum(axis=0)

    def save(self, path: Path) -> None:
        """Save ensemble weights and metadata to JSON.

        Args:
            path: Output file path.
        """
        if self.weights_ is None or self.model_names_ is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": "weighted_average",
            "weights": {
                name: float(w) for name, w in zip(
                    self.model_names_, self.weights_.tolist()
                )
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class StackingEnsemble:
    """Ridge regression meta-learner on stacked model predictions.

    For each forecast step t in [0, horizon):
        y_hat_t = Ridge([P_1_t, P_2_t, ..., P_N_t])

    Each forecast step gets its own Ridge model because the difficulty
    and optimal weighting of models may change across the horizon
    (e.g., LSTM may be best at h=1, Seq2Seq at h=24).

    Ridge (L2 regularization) is used instead of OLS because model
    predictions are typically highly correlated, making the design
    matrix ill-conditioned.

    Args:
        alpha: Ridge regularization strength (default 1.0).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.meta_learners_: list[Ridge] | None = None
        self.model_names_: list[str] | None = None

    def fit(
        self,
        val_predictions: list[np.ndarray],
        val_true: np.ndarray,
        model_names: list[str] | None = None,
    ) -> "StackingEnsemble":
        """Fit one Ridge model per forecast step.

        Args:
            val_predictions: List of (n_val, horizon) arrays, one per model.
            val_true: (n_val, horizon) ground truth.
            model_names: Optional list of model names for logging.

        Returns:
            Self for method chaining.
        """
        self.model_names_ = model_names or [f"model_{i}" for i in range(len(val_predictions))]
        horizon = val_true.shape[1]

        # (n_val, horizon, n_models) — rearranged for per-step fitting
        P = np.stack(val_predictions, axis=-1)

        self.meta_learners_ = []
        for t in range(horizon):
            ridge = Ridge(alpha=self.alpha)
            ridge.fit(P[:, t, :], val_true[:, t])
            self.meta_learners_.append(ridge)

        logger.info(f"  Fitted {horizon} Ridge meta-learners (alpha={self.alpha})")
        return self

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Apply meta-learners to new predictions.

        Args:
            predictions: List of (n_samples, horizon) arrays, one per model.

        Returns:
            Stacked ensemble predictions of shape (n_samples, horizon).
        """
        if self.meta_learners_ is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        P = np.stack(predictions, axis=-1)
        n_samples = P.shape[0]
        horizon = P.shape[1]

        result = np.zeros((n_samples, horizon))
        for t in range(horizon):
            result[:, t] = self.meta_learners_[t].predict(P[:, t, :])
        return result

    def save(self, path: Path) -> None:
        """Save stacking ensemble metadata to JSON.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": "stacking",
            "alpha": self.alpha,
            "n_meta_learners": len(self.meta_learners_ or []),
            "model_names": self.model_names_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
