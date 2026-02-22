"""Mixed Gaussian + von Mises Hidden Markov Model.

Subclasses hmmlearn.base.BaseHMM to support mixed emission distributions:
- Gaussian emissions for continuous features (CO2, dCO2, Noise)
- Von Mises emissions for circular/angular features (hour_angle, dow_angle)

Standard Gaussian HMM treats sin/cos time encodings as linear variables,
which causes EM to model temporal patterns instead of physical occupancy
regimes. By using the von Mises distribution — the "circular Gaussian" —
we let the HMM model time properly on the circle while preserving
Gaussian modeling for the physical sensor features.

Emission model (conditional independence given state k):
    log P(x_t | z_t = k) = SUM_j log N(x_t^g_j; mu^g_{k,j}, sigma^2_{k,j})
                          + SUM_j log VM(x_t^c_j; mu^c_{k,j}, kappa_{k,j})

Von Mises PDF:
    f(theta; mu, kappa) = exp(kappa * cos(theta - mu)) / (2*pi * I_0(kappa))

    mu in [-pi, pi]: circular mean (direction of max probability)
    kappa >= 0:      concentration parameter (0 = uniform, large = tight)

M-step (weighted MLE):
    Gaussian:
        mu_k = SUM_t gamma_t(k)*x_t / SUM_t gamma_t(k)
        sigma^2_k = SUM_t gamma_t(k)*x_t^2 / SUM_t gamma_t(k) - mu_k^2

    Von Mises:
        mu_k = arctan2(S_k, C_k)                     (circular mean)
        kappa_k: solve I_1(kappa)/I_0(kappa) = R_bar  (Newton iteration)
        where C_k = SUM_t gamma_t(k)*cos(theta_t)
              S_k = SUM_t gamma_t(k)*sin(theta_t)
              R_bar = sqrt(C_k^2 + S_k^2) / SUM_t gamma_t(k)

References:
    - Mardia & Jupp (2000), "Directional Statistics", ch. 9
    - Banerjee et al. (2005), "Clustering on the unit hypersphere"
    - hmmlearn BaseHMM subclassing: hmmlearn.readthedocs.io
"""

from __future__ import annotations

import logging

import numpy as np
from hmmlearn.base import BaseHMM
from scipy.special import i0, i1
from scipy.stats import norm, vonmises
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)


def _estimate_kappa_vectorized(R_bar: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Estimate von Mises concentration kappa from mean resultant length.

    Uses the Banerjee et al. (2005) initial approximation:
        kappa_0 = R_bar * (2 - R_bar^2) / (1 - R_bar^2)

    Refined by Newton-Raphson on the equation A(kappa) = R_bar,
    where A(kappa) = I_1(kappa) / I_0(kappa) is the ratio of modified
    Bessel functions. The derivative is:
        A'(kappa) = 1 - A(kappa)/kappa - A(kappa)^2

    Args:
        R_bar: Mean resultant lengths, shape (n_states, n_circular).
        max_iter: Maximum Newton iterations.

    Returns:
        Estimated kappa values, same shape as R_bar.
    """
    # Banerjee et al. initial estimate (good for R_bar < 0.9)
    kappa = R_bar * (2.0 - R_bar ** 2) / (1.0 - R_bar ** 2)

    for _ in range(max_iter):
        A_k = i1(kappa) / i0(kappa)
        A_prime = 1.0 - A_k / np.maximum(kappa, 1e-10) - A_k ** 2
        delta = (A_k - R_bar) / np.maximum(np.abs(A_prime), 1e-10)
        kappa = np.maximum(kappa - delta, 1e-8)
        if np.all(np.abs(delta) < 1e-10):
            break

    return kappa


class MixedGaussianVonMisesHMM(BaseHMM):
    """HMM with mixed Gaussian (continuous) + von Mises (circular) emissions.

    Feature layout in X: [gaussian_features..., circular_features...]
    - First n_gaussian columns: continuous features modeled by diagonal Gaussian
    - Next n_circular columns: angular features (radians in [-pi, pi])
      modeled by independent von Mises distributions

    The total emission log-likelihood factorizes (conditional independence):
        log P(x_t | z_t=k) = SUM_j log N(x^g_j; mu^g_{k,j}, sigma^2_{k,j})
                            + SUM_j log VM(x^c_j; mu^c_{k,j}, kappa_{k,j})

    Args:
        n_components: Number of hidden states.
        n_gaussian: Number of Gaussian (continuous) features.
        n_circular: Number of von Mises (circular) features.
        min_var: Floor for Gaussian variances (prevents degeneracy).
        min_kappa: Floor for von Mises concentration (prevents uniform collapse).
        n_iter: Maximum EM iterations.
        tol: Convergence threshold on log-likelihood relative improvement.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_gaussian: int = 3,
        n_circular: int = 2,
        min_var: float = 1e-4,
        min_kappa: float = 1e-2,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        verbose: bool = False,
        implementation: str = "log",
    ):
        # 'g' = Gaussian params, 'v' = von Mises params
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            params="stgv",
            init_params="stgv",
            implementation=implementation,
        )
        self.n_gaussian = n_gaussian
        self.n_circular = n_circular
        self.min_var = min_var
        self.min_kappa = min_kappa

    def _get_n_fit_scalars_per_param(self) -> dict:
        """Count free parameters for AIC/BIC computation."""
        nc = self.n_components
        ng = self.n_gaussian
        nv = self.n_circular
        return {
            "s": nc - 1,           # start probs (simplex constraint)
            "t": nc * (nc - 1),    # transition matrix (row simplex constraints)
            "g": nc * ng * 2,      # Gaussian: means + variances
            "v": nc * nv * 2,      # von Mises: circular means + concentrations
        }

    def _init(self, X, lengths=None):
        """Initialize emission parameters from data statistics.

        Gaussian means: data mean + small random perturbation per state.
        Gaussian variances: data variance (uniform across states initially).
        Von Mises means: spread uniformly around the circle.
        Von Mises kappas: moderate initial concentration (kappa=2.0).
        """
        super()._init(X, lengths)
        rs = check_random_state(self.random_state)
        nc = self.n_components
        ng = self.n_gaussian
        nv = self.n_circular

        X_gauss = X[:, :ng]
        X_circ = X[:, ng:ng + nv]

        # Gaussian initialization: means spread around data center
        if self._needs_init("g", "gauss_means_"):
            offsets = rs.randn(nc, ng) * 0.5
            self.gauss_means_ = X_gauss.mean(axis=0) + offsets * X_gauss.std(axis=0)

        if self._needs_init("g", "gauss_vars_"):
            self.gauss_vars_ = np.tile(X_gauss.var(axis=0), (nc, 1))

        # Von Mises initialization: spread means uniformly around the circle
        if self._needs_init("v", "vm_mus_"):
            # Evenly spaced initial means (e.g., for 4 states: 0, pi/2, pi, 3pi/2)
            base_angles = np.linspace(-np.pi, np.pi, nc, endpoint=False)
            # Small perturbation per circular feature
            self.vm_mus_ = np.column_stack([
                base_angles + rs.randn(nc) * 0.1
                for _ in range(nv)
            ])

        if self._needs_init("v", "vm_kappas_"):
            # Moderate initial concentration (not too tight, not too diffuse)
            self.vm_kappas_ = np.full((nc, nv), 2.0)

    def _check(self):
        """Validate and clamp emission parameters to prevent degeneracy."""
        super()._check()
        self.gauss_vars_ = np.maximum(self.gauss_vars_, self.min_var)
        self.vm_kappas_ = np.maximum(self.vm_kappas_, self.min_kappa)

    def _compute_log_likelihood(self, X):
        """Compute log P(x_t | z_t = k) for all t and k.

        Combines Gaussian and von Mises log-likelihoods additively
        (conditional independence of features given the hidden state).

        Args:
            X: Observation matrix, shape (n_samples, n_gaussian + n_circular).

        Returns:
            Log-likelihood matrix, shape (n_samples, n_components).
        """
        nc = self.n_components
        ng = self.n_gaussian
        nv = self.n_circular
        n_samples = X.shape[0]

        X_gauss = X[:, :ng]
        X_circ = X[:, ng:ng + nv]

        log_prob = np.zeros((n_samples, nc))

        for k in range(nc):
            # --- Gaussian contribution ---
            # log N(x; mu, sigma^2) = -0.5*log(2*pi*sigma^2) - (x-mu)^2/(2*sigma^2)
            for j in range(ng):
                log_prob[:, k] += norm.logpdf(
                    X_gauss[:, j],
                    loc=self.gauss_means_[k, j],
                    scale=np.sqrt(self.gauss_vars_[k, j]),
                )

            # --- Von Mises contribution ---
            # log VM(theta; mu, kappa) = kappa*cos(theta-mu) - log(2*pi*I_0(kappa))
            for j in range(nv):
                log_prob[:, k] += vonmises.logpdf(
                    X_circ[:, j],
                    kappa=self.vm_kappas_[k, j],
                    loc=self.vm_mus_[k, j],
                )

        return log_prob

    def _initialize_sufficient_statistics(self):
        """Initialize accumulators for E-step sufficient statistics.

        Gaussian: weighted sums of x and x^2 (for mean/variance MLE).
        Von Mises: weighted sums of cos(theta) and sin(theta) (for circular MLE).
        """
        stats = super()._initialize_sufficient_statistics()
        nc = self.n_components
        ng = self.n_gaussian
        nv = self.n_circular

        # Posterior weight per state (shared denominator for all updates)
        stats["post"] = np.zeros(nc)

        # Gaussian sufficient statistics
        stats["gauss_obs"] = np.zeros((nc, ng))    # SUM_t gamma_t(k) * x_t
        stats["gauss_obs2"] = np.zeros((nc, ng))   # SUM_t gamma_t(k) * x_t^2

        # Von Mises circular sufficient statistics
        stats["vm_cos"] = np.zeros((nc, nv))       # SUM_t gamma_t(k) * cos(theta_t)
        stats["vm_sin"] = np.zeros((nc, nv))       # SUM_t gamma_t(k) * sin(theta_t)

        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
    ):
        """Accumulate weighted statistics from the E-step.

        posteriors[t, k] = gamma_t(k) = P(z_t = k | x_{1:T})
        These are the soft assignments used to weight the sufficient statistics.
        """
        # Parent accumulates start/transition probability statistics
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice
        )

        ng = self.n_gaussian
        nv = self.n_circular
        X_gauss = X[:, :ng]
        X_circ = X[:, ng:ng + nv]

        # Total posterior mass per state
        stats["post"] += posteriors.sum(axis=0)

        # Gaussian: weighted sums for MLE of mean and variance
        if "g" in self.params:
            # posteriors.T: (nc, n_samples) @ X_gauss: (n_samples, ng) -> (nc, ng)
            stats["gauss_obs"] += posteriors.T @ X_gauss
            stats["gauss_obs2"] += posteriors.T @ (X_gauss ** 2)

        # Von Mises: weighted circular sums for MLE of mu and kappa
        if "v" in self.params:
            stats["vm_cos"] += posteriors.T @ np.cos(X_circ)
            stats["vm_sin"] += posteriors.T @ np.sin(X_circ)

    def _do_mstep(self, stats):
        """M-step: update emission parameters from accumulated statistics.

        Gaussian:
            mu_k,j = gauss_obs_{k,j} / post_k
            sigma^2_{k,j} = gauss_obs2_{k,j} / post_k - mu_k,j^2

        Von Mises:
            mu_k,j = arctan2(vm_sin_{k,j}, vm_cos_{k,j})
            R_bar_{k,j} = sqrt(vm_cos_{k,j}^2 + vm_sin_{k,j}^2) / post_k
            kappa_{k,j}: Newton-Raphson on I_1(kappa)/I_0(kappa) = R_bar
        """
        # Update startprob and transmat
        super()._do_mstep(stats)

        denom = stats["post"][:, None]             # (nc, 1)
        denom_safe = np.maximum(denom, 1e-10)

        # --- Gaussian M-step ---
        if "g" in self.params:
            self.gauss_means_ = stats["gauss_obs"] / denom_safe
            self.gauss_vars_ = (
                stats["gauss_obs2"] / denom_safe - self.gauss_means_ ** 2
            )
            self.gauss_vars_ = np.maximum(self.gauss_vars_, self.min_var)

        # --- Von Mises M-step ---
        if "v" in self.params:
            C = stats["vm_cos"]  # (nc, nv)
            S = stats["vm_sin"]  # (nc, nv)

            # Circular mean direction
            self.vm_mus_ = np.arctan2(S, C)

            # Mean resultant length (0 = dispersed, 1 = concentrated)
            R_bar = np.sqrt(C ** 2 + S ** 2) / denom_safe
            R_bar = np.clip(R_bar, 1e-10, 1.0 - 1e-10)

            # Estimate kappa from R_bar via Newton iteration
            self.vm_kappas_ = _estimate_kappa_vectorized(R_bar)
            self.vm_kappas_ = np.maximum(self.vm_kappas_, self.min_kappa)

    def _generate_sample_from_state(self, state, random_state):
        """Generate one observation from the emission distribution of a state.

        Used by hmmlearn's sample() method for generating synthetic data.
        """
        ng = self.n_gaussian
        nv = self.n_circular
        sample = np.empty(ng + nv)

        # Gaussian features
        for j in range(ng):
            sample[j] = random_state.normal(
                self.gauss_means_[state, j],
                np.sqrt(self.gauss_vars_[state, j]),
            )

        # Circular features
        for j in range(nv):
            sample[ng + j] = vonmises.rvs(
                self.vm_kappas_[state, j],
                loc=self.vm_mus_[state, j],
                random_state=random_state,
            )

        return sample

    def score(self, X, lengths=None) -> float:
        """Compute total log-likelihood of the data under the model.

        Useful for model comparison (higher = better fit).
        """
        return super().score(X, lengths)

    def get_state_summary(self) -> list[dict]:
        """Return interpretable summary of each state's emission parameters.

        For Gaussian features: mean and standard deviation.
        For von Mises features: circular mean (converted to interpretable units)
            and concentration kappa.
        """
        summaries = []
        for k in range(self.n_components):
            state_info = {"state": k}

            # Gaussian parameters
            state_info["gaussian_means"] = self.gauss_means_[k].tolist()
            state_info["gaussian_stds"] = np.sqrt(self.gauss_vars_[k]).tolist()

            # Von Mises parameters (convert mu from radians to hours/days)
            state_info["vm_mus_rad"] = self.vm_mus_[k].tolist()
            state_info["vm_kappas"] = self.vm_kappas_[k].tolist()

            summaries.append(state_info)

        return summaries


class MixedHMMRegimeDetector:
    """Regime detector using mixed Gaussian + von Mises HMM.

    Drop-in replacement for HMMRegimeDetector that properly handles
    circular temporal features via von Mises distributions, rather than
    treating sin/cos encodings as linear Gaussian variables.

    The detector converts raw temporal columns (hour, dayofweek) into
    angular representations internally, so the caller provides column
    names for both Gaussian features and temporal columns.

    Args:
        n_states: Number of hidden states.
        gaussian_features: Column names for continuous (Gaussian) features.
        circular_features: List of dicts specifying circular features:
            [{"column": "hour", "period": 24}, {"column": "dayofweek", "period": 7}]
            If column is "hour" or "dayofweek", it's extracted from the datetime.
            The angle is computed as: 2 * pi * value / period.
        n_iter: Maximum EM iterations.
        min_var: Gaussian variance floor.
        min_kappa: Von Mises concentration floor.
    """

    def __init__(
        self,
        n_states: int = 4,
        gaussian_features: list[str] | None = None,
        circular_features: list[dict] | None = None,
        n_iter: int = 200,
        min_var: float = 1e-4,
        min_kappa: float = 1e-2,
    ) -> None:
        self.n_states = n_states
        self.gaussian_features = gaussian_features or ["CO2", "dCO2", "Noise"]
        self.circular_features = circular_features or [
            {"column": "hour", "period": 24},
            {"column": "dayofweek", "period": 7},
        ]
        self.n_iter = n_iter
        self.min_var = min_var
        self.min_kappa = min_kappa

        n_gauss = len(self.gaussian_features)
        n_circ = len(self.circular_features)

        self.hmm = MixedGaussianVonMisesHMM(
            n_components=n_states,
            n_gaussian=n_gauss,
            n_circular=n_circ,
            min_var=min_var,
            min_kappa=min_kappa,
            n_iter=n_iter,
            random_state=42,
        )
        self._is_fitted = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build the observation matrix [gaussian_cols | circular_angles].

        Gaussian features are taken directly from the DataFrame.
        Circular features are computed from temporal columns:
            angle = 2 * pi * column_value / period

        For "hour" and "dayofweek", values are extracted from the datetime column
        if they don't exist as standalone columns.

        Args:
            df: DataFrame with required columns.

        Returns:
            X: array of shape (n_samples, n_gaussian + n_circular).
        """
        # Gaussian part
        X_gauss = df[self.gaussian_features].values.astype(np.float64)

        # Circular part
        circ_arrays = []
        for circ_cfg in self.circular_features:
            col = circ_cfg["column"]
            period = circ_cfg["period"]

            if col in df.columns:
                raw_values = df[col].values.astype(np.float64)
            elif col == "hour" and "datetime" in df.columns:
                dt = pd.to_datetime(df["datetime"])
                raw_values = (dt.dt.hour + dt.dt.minute / 60.0).values.astype(np.float64)
            elif col == "dayofweek" and "datetime" in df.columns:
                dt = pd.to_datetime(df["datetime"])
                raw_values = dt.dt.dayofweek.values.astype(np.float64)
            else:
                raise ValueError(
                    f"Circular feature column '{col}' not found in DataFrame "
                    f"and cannot be derived from datetime."
                )

            # Map to angle on [-pi, pi]
            # angle = 2*pi*(value / period) - pi   maps [0, period) -> [-pi, pi)
            angle = 2.0 * np.pi * (raw_values / period) - np.pi
            circ_arrays.append(angle)

        X_circ = np.column_stack(circ_arrays) if circ_arrays else np.empty((len(df), 0))

        return np.hstack([X_gauss, X_circ])

    def fit(self, train_df: pd.DataFrame) -> "MixedHMMRegimeDetector":
        """Fit mixed HMM on training data.

        Args:
            train_df: Training DataFrame with required feature columns.

        Returns:
            self for method chaining.
        """
        X = self._extract_features(train_df)
        self.hmm.fit(X)
        self._is_fitted = True

        circ_summary = []
        for k in range(self.n_states):
            for j, circ_cfg in enumerate(self.circular_features):
                mu_rad = self.hmm.vm_mus_[k, j]
                kappa = self.hmm.vm_kappas_[k, j]
                # Convert back to original units for interpretability
                period = circ_cfg["period"]
                col = circ_cfg["column"]
                mu_orig = (mu_rad + np.pi) / (2.0 * np.pi) * period
                circ_summary.append(
                    f"    State {k}: {col} mean={mu_orig:.1f} "
                    f"(kappa={kappa:.2f})"
                )

        logger.info(f"Mixed HMM fitted with {self.n_states} states")
        logger.info(f"  Converged: {self.hmm.monitor_.converged}")
        logger.info(f"  Gaussian features: {self.gaussian_features}")
        logger.info(f"  Circular features: {[c['column'] for c in self.circular_features]}")
        for line in circ_summary:
            logger.info(line)

        return self

    def predict_states(self, df: pd.DataFrame) -> np.ndarray:
        """Predict most likely state sequence (Viterbi decoding).

        Args:
            df: DataFrame with required feature columns.

        Returns:
            Array of shape (n_samples,) with integer state labels.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before calling predict_states")
        X = self._extract_features(df)
        return self.hmm.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict posterior state probabilities (forward-backward).

        Args:
            df: DataFrame with required feature columns.

        Returns:
            Array of shape (n_samples, n_states) with gamma_t(k).
        """
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before calling predict_proba")
        X = self._extract_features(df)
        return self.hmm.predict_proba(X)

    def score(self, df: pd.DataFrame) -> float:
        """Compute total log-likelihood of data under the model."""
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before calling score")
        X = self._extract_features(df)
        return self.hmm.score(X)


# Needed for _extract_features to access pd
import pandas as pd
