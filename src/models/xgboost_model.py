"""XGBoost model for time series forecasting.

XGBoost (eXtreme Gradient Boosting) is a gradient-boosted decision tree
ensemble that iteratively fits residuals using second-order Taylor
expansion of the loss function.

For a regression task with squared error loss:

    Objective at iteration t:
        L^(t) = sum_i [ l(y_i, y_hat_i^(t-1) + f_t(x_i)) ] + Omega(f_t)

    Second-order Taylor approximation:
        L^(t) ≈ sum_i [ g_i * f_t(x_i) + 0.5 * h_i * f_t(x_i)^2 ] + Omega(f_t)

    where:
        g_i = d l / d y_hat |_{y_hat=y_hat^(t-1)}     (gradient)
        h_i = d^2 l / d y_hat^2 |_{y_hat=y_hat^(t-1)} (Hessian)

    For MSE loss: g_i = 2*(y_hat_i - y_i), h_i = 2

    Regularization term:
        Omega(f) = gamma * T + 0.5 * lambda * sum_j w_j^2

    where T = number of leaves, w_j = leaf weights, gamma = min split
    loss reduction, lambda = L2 regularization on leaf weights.

    Optimal leaf weight for leaf j:
        w_j* = - sum_{i in I_j} g_i / (sum_{i in I_j} h_i + lambda)

    Split gain for splitting node into left (L) and right (R):
        Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                       - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

For multi-step time series forecasting, the lookback window is flattened
into a single feature vector:
    x_i = [x_{t-L}, x_{t-L+1}, ..., x_{t-1}]  (all features concatenated)

Multi-output is handled via sklearn's MultiOutputRegressor, which fits
one independent XGBoost model per forecast step.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """XGBoost forecaster for CO2 time series.

    Flattens the lookback window into a feature vector and trains one
    XGBRegressor per forecast step via MultiOutputRegressor.

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

        # XGBoost hyperparameters from config
        self.n_estimators = self.model_cfg.get("n_estimators", 500)
        self.max_depth = self.model_cfg.get("max_depth", 6)
        self.learning_rate = self.model_cfg.get("learning_rate", 0.05)
        self.subsample = self.model_cfg.get("subsample", 0.8)
        self.colsample_bytree = self.model_cfg.get("colsample_bytree", 0.8)
        self.reg_alpha = self.model_cfg.get("reg_alpha", 0.01)
        self.reg_lambda = self.model_cfg.get("reg_lambda", 1.0)
        self.min_child_weight = self.model_cfg.get("min_child_weight", 5)
        self.early_stopping_rounds = self.model_cfg.get(
            "early_stopping_rounds", 20
        )

        self.model_: MultiOutputRegressor | None = None

    @staticmethod
    def _resolve_device() -> str:
        """Select the best available device for XGBoost.

        XGBoost >= 2.0 uses device="cuda" instead of tree_method="gpu_hist".
        Falls back to CPU if CUDA is not available.

        Returns:
            Device string: "cuda" if GPU available, "cpu" otherwise.
        """
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_estimator(self) -> XGBRegressor:
        """Create a single XGBRegressor with the configured hyperparameters.

        Uses GPU (CUDA) when available via ``device="cuda"`` (XGBoost >= 2.0).

        Returns:
            Configured XGBRegressor instance.
        """
        device = self._resolve_device()
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            tree_method="hist",
            device=device,
            random_state=self.config["training"].get("seed", 42),
            n_jobs=-1,
            verbosity=0,
        )

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
    ) -> XGBoostForecaster:
        """Fit the XGBoost model.

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
            f"Fitting XGBoost: {X_train_flat.shape[0]} samples, "
            f"{X_train_flat.shape[1]} features, "
            f"{self.horizon_steps}-step output"
        )

        base_estimator = self._build_estimator()

        if X_val is not None and y_val is not None:
            X_val_flat = self._flatten_windows(X_val)

            # With early stopping, fit each output dimension separately
            # to allow per-step early stopping on validation data
            self.model_ = MultiOutputRegressor(base_estimator)

            # Override fit to pass eval_set per estimator
            self.model_.estimators_ = []
            for step in range(y_train.shape[1]):
                est = self._build_estimator()
                est.fit(
                    X_train_flat,
                    y_train[:, step],
                    eval_set=[(X_val_flat, y_val[:, step])],
                    verbose=False,
                )
                self.model_.estimators_.append(est)
        else:
            self.model_ = MultiOutputRegressor(base_estimator)
            self.model_.fit(X_train_flat, y_train)

        logger.info("XGBoost fitting complete.")
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

        # MultiOutputRegressor.predict returns (n_samples, n_outputs)
        predictions = np.column_stack(
            [est.predict(X_flat) for est in self.model_.estimators_]
        )
        return predictions
