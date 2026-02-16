"""SARIMA (Seasonal ARIMA) model for time series forecasting.

SARIMA combines autoregressive (AR), integrated (I), and moving average (MA)
components with seasonal counterparts for periodic time series:

    SARIMA(p, d, q) x (P, D, Q, s)

Non-seasonal component:
    phi(B)(1 - B)^d * y_t = theta(B) * eps_t

    where:
        phi(B)   = 1 - phi_1*B - ... - phi_p*B^p        (AR polynomial)
        theta(B) = 1 + theta_1*B + ... + theta_q*B^q     (MA polynomial)
        B        = backshift operator: B*y_t = y_{t-1}
        d        = order of non-seasonal differencing
        eps_t    ~ WN(0, sigma^2)                         (white noise)

Seasonal component (period s):
    Phi(B^s)(1 - B^s)^D * y_t = Theta(B^s) * eps_t

    where:
        Phi(B^s)   = 1 - Phi_1*B^s - ... - Phi_P*B^{P*s}  (seasonal AR)
        Theta(B^s) = 1 + Theta_1*B^s + ... + Theta_Q*B^{Q*s} (seasonal MA)
        D          = order of seasonal differencing

Combined model:
    phi(B) * Phi(B^s) * (1-B)^d * (1-B^s)^D * y_t
        = theta(B) * Theta(B^s) * eps_t

For multi-step forecasting, SARIMA naturally produces sequential forecasts
by iterating forward one step at a time, conditioning on its own predictions.

Note:
    SARIMA is a univariate model — it uses only the target variable (CO2).
    Exogenous features are not used, making this a pure baseline comparison.
    This is an intentional design choice: SARIMA captures temporal structure
    (trend, seasonality, autocorrelation) without requiring feature engineering.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


class SARIMAForecaster:
    """SARIMA forecaster for CO2 time series.

    Fits a SARIMA model on training data and produces multi-step forecasts
    using the same sliding-window evaluation protocol as the neural models.

    Because SARIMA fitting is expensive, this class fits a single model on
    the full training series and then generates forecasts for each test window
    by conditioning on the lookback window (using the ``apply`` method to
    re-initialize state without re-estimating parameters).

    Args:
        config: Merged configuration dictionary with 'model' and 'data' keys.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_cfg = config["model"]
        self.data_cfg = config["data"]

        # SARIMA order
        self.order: tuple[int, int, int] = tuple(
            self.model_cfg.get("order", [1, 1, 1])
        )
        self.seasonal_order: tuple[int, int, int, int] = tuple(
            self.model_cfg.get("seasonal_order", [1, 1, 1, 288])
        )

        # Forecast horizon in steps
        sph = self.data_cfg["samples_per_hour"]
        self.horizon_steps = self.data_cfg["forecast_horizon_hours"] * sph
        self.lookback_steps = self.data_cfg["lookback_hours"] * sph

        self.fitted_params_: dict[str, Any] | None = None
        self.result_: Any | None = None  # Full statsmodels SARIMAXResults object
        self.train_series_: np.ndarray | None = None

    def fit(
        self,
        train_series: np.ndarray,
        val_series: np.ndarray | None = None,
    ) -> SARIMAForecaster:
        """Fit SARIMA on the training time series.

        Args:
            train_series: 1-D array of target values (original scale).
            val_series: Ignored (kept for API consistency).

        Returns:
            self
        """
        self.train_series_ = train_series.ravel()

        logger.info(
            f"Fitting SARIMA{self.order}x{self.seasonal_order} "
            f"on {len(self.train_series_)} samples..."
        )

        # Suppress convergence warnings during fitting — SARIMA on long
        # series with large seasonal periods may hit iteration limits but
        # still produce usable parameter estimates.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                self.train_series_,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)

        self.fitted_params_ = result.params
        self.result_ = result  # Keep full result for diagnostics (coefficients, p-values, AIC, etc.)
        logger.info("SARIMA fitting complete.")
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Produce a multi-step forecast conditioned on a history window.

        Re-initializes the SARIMA state machine from the given history
        using the previously estimated parameters (no re-fitting).

        Args:
            history: 1-D array of observed values preceding the forecast
                window. Length should be >= lookback_steps.

        Returns:
            1-D array of shape (horizon_steps,) with forecasted values.
        """
        if self.fitted_params_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        history = history.ravel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                history,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            # apply() re-uses estimated parameters without re-fitting
            result = model.filter(self.fitted_params_)
            forecast = result.forecast(steps=self.horizon_steps)

        return forecast

    def predict_batch(
        self,
        X: np.ndarray,
        target_idx: int = -1,
    ) -> np.ndarray:
        """Produce forecasts for a batch of lookback windows.

        Extracts the target column from each sliding window and calls
        predict() for each one. This mirrors the neural model evaluation
        protocol where each test sample is a (lookback, features) window.

        Args:
            X: Array of shape (n_samples, lookback, n_features).
            target_idx: Column index of the target variable in X.

        Returns:
            Array of shape (n_samples, horizon_steps).
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.horizon_steps))

        for i in range(n_samples):
            # Extract target column from this lookback window
            history = X[i, :, target_idx]
            predictions[i] = self.predict(history)

        return predictions
