"""CatBoost model for time series forecasting.

CatBoost (Category Boosting) is a gradient boosting library that uses
ordered boosting and oblivious decision trees to reduce prediction shift.

Key algorithmic differences from standard gradient boosting:

1. Ordered Boosting (reduces overfitting from target leakage):
    Instead of computing residuals on the same data used for tree
    construction, CatBoost uses a permutation-driven approach:
        - For each sample i, gradients are computed using a model
          trained only on samples with index < sigma(i), where sigma
          is a random permutation.
        - This prevents the model from implicitly using y_i when
          computing the gradient for sample i.

2. Oblivious Decision Trees (symmetric trees):
    All nodes at the same depth use the same split condition:
        f(x) = sum_j w_j * I(x in R_j)

    where regions R_j are defined by the same split feature and
    threshold at each level. This acts as implicit regularization
    and enables efficient inference via bitwise operations.

3. Leaf value computation:
    w_j = - sum_{i in I_j} g_i / (sum_{i in I_j} h_i + lambda)

    Same as XGBoost, but applied to symmetric tree leaves.

For multi-step forecasting, the lookback window is flattened into a
feature vector and MultiOutputRegressor fits one CatBoost model per
forecast step.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


class CatBoostForecaster:
    """CatBoost forecaster for CO2 time series.

    Flattens the lookback window into a feature vector and trains one
    CatBoostRegressor per forecast step via MultiOutputRegressor.

    Args:
        config: Merged configuration dictionary with 'model' and 'data' keys.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_cfg = config["model"]
        self.data_cfg = config["data"]

        sph = self.data_cfg["samples_per_hour"]
        self.horizon_steps = self.data_cfg["forecast_horizon_hours"] * sph
        self.lookback_steps = self.data_cfg["lookback_hours"] * sph

        # CatBoost hyperparameters from config
        self.iterations = self.model_cfg.get("iterations", 500)
        self.depth = self.model_cfg.get("depth", 6)
        self.learning_rate = self.model_cfg.get("learning_rate", 0.05)
        self.l2_leaf_reg = self.model_cfg.get("l2_leaf_reg", 3.0)
        self.subsample = self.model_cfg.get("subsample", 0.8)
        self.early_stopping_rounds = self.model_cfg.get(
            "early_stopping_rounds", 20
        )

        self.model_: MultiOutputRegressor | None = None

    @staticmethod
    def _resolve_task_type() -> tuple[str, str | None]:
        """Select the best available device for CatBoost.

        CatBoost uses ``task_type="GPU"`` with ``devices="0"`` for GPU
        training. Falls back to CPU if CUDA is not available.

        Returns:
            Tuple of (task_type, devices) — e.g. ("GPU", "0") or ("CPU", None).
        """
        if torch.cuda.is_available():
            return "GPU", "0"
        return "CPU", None

    def _build_estimator(self) -> CatBoostRegressor:
        """Create a single CatBoostRegressor with configured hyperparameters.

        Uses GPU when available via ``task_type="GPU"``.

        Returns:
            Configured CatBoostRegressor instance.
        """
        task_type, devices = self._resolve_task_type()

        kwargs: dict = {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": "RMSE",
            "random_seed": self.config["training"].get("seed", 42),
            "verbose": 0,
            "allow_writing_files": False,
            "task_type": task_type,
        }
        if devices is not None:
            kwargs["devices"] = devices

        # GPU mode uses Bayesian bootstrap by default, which doesn't support
        # the subsample parameter. Switch to MVS (Minimum Variance Sampling)
        # bootstrap on GPU to enable subsampling; on CPU use Bernoulli.
        if task_type == "GPU":
            kwargs["bootstrap_type"] = "MVS"
            kwargs["subsample"] = self.subsample
        else:
            kwargs["bootstrap_type"] = "Bernoulli"
            kwargs["subsample"] = self.subsample

        return CatBoostRegressor(**kwargs)

    @staticmethod
    def _flatten_windows(X: np.ndarray) -> np.ndarray:
        """Flatten 3-D sliding windows to 2-D feature matrix.

        Args:
            X: Array of shape (n_samples, lookback, n_features).

        Returns:
            Array of shape (n_samples, lookback * n_features).
        """
        return X.reshape(X.shape[0], -1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> CatBoostForecaster:
        """Fit the CatBoost model.

        Args:
            X_train: Training input windows (n_samples, lookback, n_features).
            y_train: Training targets (n_samples, horizon).
            X_val: Validation input windows for early stopping.
            y_val: Validation targets for early stopping.

        Returns:
            self
        """
        X_train_flat = self._flatten_windows(X_train)
        logger.info(
            f"Fitting CatBoost: {X_train_flat.shape[0]} samples, "
            f"{X_train_flat.shape[1]} features, "
            f"{self.horizon_steps}-step output"
        )

        base_estimator = self._build_estimator()

        if X_val is not None and y_val is not None:
            X_val_flat = self._flatten_windows(X_val)

            # Fit each output dimension separately for per-step early stopping
            self.model_ = MultiOutputRegressor(base_estimator)
            self.model_.estimators_ = []
            for step in range(y_train.shape[1]):
                est = self._build_estimator()
                est.fit(
                    X_train_flat,
                    y_train[:, step],
                    eval_set=(X_val_flat, y_val[:, step]),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
                self.model_.estimators_.append(est)
        else:
            self.model_ = MultiOutputRegressor(base_estimator)
            self.model_.fit(X_train_flat, y_train)

        logger.info("CatBoost fitting complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Produce multi-step forecasts.

        Args:
            X: Input windows of shape (n_samples, lookback, n_features).

        Returns:
            Array of shape (n_samples, horizon_steps).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_flat = self._flatten_windows(X)

        predictions = np.column_stack(
            [est.predict(X_flat) for est in self.model_.estimators_]  # type: ignore[union-attr]
        )
        return predictions
