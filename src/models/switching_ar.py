"""Physics-informed Switching Auto-Regressive HMM for occupancy detection.

Implements the model from Esmaieeli-Sikaroudi et al. (2024):
    "Physics-Informed Building Occupancy Detection: a Switching Process
     with Markov Regime"  (arXiv:2409.11743)

The model encodes the CO2 mass-balance ODE into a switching AR(1) process
where each hidden state corresponds to a (ventilation, occupancy) regime.

CO2 Mass-Balance ODE:
    dy/dt = -(1/τ) · y(t) + n(t) · r

    y(t) = CO2(t) - CO2_ambient   (excess concentration above outdoor baseline)
    τ     = V / Q                  (room volume / ventilation flow rate, seconds)
    n(t)  = number of occupants
    r     = CO2 generation rate per person (ppm/s equivalent)

Discretized (Euler, interval Δt):
    y_t = c · y_{t-1} + μ + ε_t

    c   = exp(-Δt/τ)               (AR coefficient, depends on ventilation regime)
    μ   = (1 - exp(-Δt/τ)) · r · n (drift, depends on occupancy count AND ventilation)
    ε_t ~ N(0, σ²)                 (observation noise)

Switching AR-HMM:
    Each hidden state S_t ∈ {1, ..., K} represents a distinct (τ_k, n_k) pair.
    The observation model for state k is:
        y_t | S_t=k ~ N(c_k · y_{t-1} + μ_k,  σ_k²)

    State transitions follow a K×K Markov chain with transition matrix A.

EM Algorithm:
    E-step:  Forward-backward algorithm to compute γ_t(k) = P(S_t=k | y_{1:T})
             and ξ_t(j,k) = P(S_{t-1}=j, S_t=k | y_{1:T})
    M-step:  Update (c_k, μ_k, σ_k²) via weighted least squares on AR residuals
             Update A_{jk} from expected transition counts
             Update π_k from γ_1(k)

Physical constraints (optional):
    c_k ∈ (0, 1)    because exp(-Δt/τ) ∈ (0,1) for positive τ, Δt
    μ_k ≥ 0          for occupied states (CO2 generation is non-negative)
    μ_k ≈ 0          for unoccupied states (no CO2 source)

References:
    [1] Esmaieeli-Sikaroudi et al., "Physics-Informed Building Occupancy
        Detection: a Switching Process with Markov Regime," arXiv:2409.11743
    [2] Hamilton (1989), "A New Approach to the Economic Analysis of
        Nonstationary Time Series and the Business Cycle," Econometrica 57(2)
    [3] Kim (1994), "Dynamic linear models with Markov-switching," J. Econometrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


@dataclass
class SwitchingARParams:
    """Parameters for the Switching AR-HMM.

    Attributes:
        c: AR(1) coefficients per state, shape (K,).
           Physically: c_k = exp(-Δt/τ_k), encodes ventilation time constant.
        mu: Drift terms per state, shape (K,).
           Physically: μ_k = (1 - c_k) · r · n_k, encodes occupancy level.
        sigma2: Observation noise variance per state, shape (K,).
        A: Transition matrix, shape (K, K). A[j,k] = P(S_t=k | S_{t-1}=j).
        pi: Initial state distribution, shape (K,).
    """
    c: np.ndarray       # (K,) AR coefficients
    mu: np.ndarray      # (K,) drift terms
    sigma2: np.ndarray  # (K,) noise variances
    A: np.ndarray       # (K, K) transition matrix
    pi: np.ndarray      # (K,) initial distribution

    @property
    def K(self) -> int:
        return len(self.c)

    def copy(self) -> "SwitchingARParams":
        return SwitchingARParams(
            c=self.c.copy(),
            mu=self.mu.copy(),
            sigma2=self.sigma2.copy(),
            A=self.A.copy(),
            pi=self.pi.copy(),
        )


@dataclass
class SwitchingARResult:
    """Results from the switching AR-HMM inference.

    Attributes:
        states: Most likely state sequence (Viterbi), shape (T,).
        gamma: Posterior state probabilities, shape (T, K).
        log_likelihood: Final log-likelihood of the data.
        params: Fitted model parameters.
        n_iter: Number of EM iterations performed.
        converged: Whether EM converged.
        occupancy_binary: Binary occupancy labels, shape (T,).
        occupancy_map: Mapping from state index to occupancy level.
    """
    states: np.ndarray
    gamma: np.ndarray
    log_likelihood: float
    params: SwitchingARParams
    n_iter: int
    converged: bool
    occupancy_binary: np.ndarray = field(default_factory=lambda: np.array([]))
    occupancy_map: dict = field(default_factory=dict)


class SwitchingARHMM:
    """Physics-informed Switching AR(1) Hidden Markov Model.

    Models indoor CO2 as a switching autoregressive process where each
    hidden state encodes a distinct (ventilation regime, occupancy level)
    pair. The AR structure captures the physical CO2 decay dynamics that
    a standard HMM ignores.

    Key advantage over standard HMM:
        Standard HMM:     P(y_t | S_t=k) = N(y_t; μ_k, σ²_k)
        Switching AR-HMM: P(y_t | y_{t-1}, S_t=k) = N(y_t; c_k·y_{t-1} + μ_k, σ²_k)

    The AR term c_k·y_{t-1} explicitly models the exponential decay of CO2
    towards ambient, which is the dominant physical process in the signal.
    This lets the drift μ_k cleanly capture the occupancy contribution.

    Args:
        n_states: Number of hidden states K.
        co2_ambient: Ambient (outdoor) CO2 concentration in ppm.
        delta_t_hours: Time step between observations in hours.
        n_iter: Maximum EM iterations.
        tol: Convergence tolerance on relative log-likelihood change.
        min_sigma2: Floor for noise variance (prevents degeneracy).
        constrain_ar: If True, clamp c_k to (0, 1) per physics.
        random_state: Random seed for initialization.
    """

    def __init__(
        self,
        n_states: int = 6,
        co2_ambient: float = 420.0,
        delta_t_hours: float = 1.0,
        n_iter: int = 200,
        tol: float = 1e-6,
        min_sigma2: float = 1.0,
        constrain_ar: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.co2_ambient = co2_ambient
        self.delta_t_hours = delta_t_hours
        self.n_iter = n_iter
        self.tol = tol
        self.min_sigma2 = min_sigma2
        self.constrain_ar = constrain_ar
        self.random_state = random_state
        self.params_: SwitchingARParams | None = None
        self._is_fitted = False

    def _initialize_params(self, y: np.ndarray) -> SwitchingARParams:
        """Initialize parameters using physics-informed heuristics.

        Strategy:
        1. Estimate a global AR(1) coefficient via Yule-Walker:
           c_global = corr(y_t, y_{t-1}) / var(y)
        2. Spread AR coefficients around c_global to represent different
           ventilation rates (faster/slower decay).
        3. Set drift terms to capture different occupancy levels:
           μ_k ∝ k (higher states = more people = more CO2 generation).
        4. Initialize transition matrix with strong self-transitions
           (occupancy regimes tend to persist for hours).

        Args:
            y: CO2 excess time series (CO2 - ambient), shape (T,).

        Returns:
            Initial parameter set.
        """
        rng = np.random.RandomState(self.random_state)
        K = self.n_states
        T = len(y)

        # Yule-Walker estimate of global AR(1) coefficient
        # c = autocorrelation at lag 1
        y_mean = y.mean()
        y_var = np.var(y) + 1e-10
        y_centered = y - y_mean
        autocov_1 = np.mean(y_centered[:-1] * y_centered[1:])
        c_global = np.clip(autocov_1 / y_var, 0.5, 0.99)

        logger.info(f"  Yule-Walker global AR(1) estimate: c = {c_global:.4f}")

        # Spread AR coefficients: represent different ventilation time constants
        # c = exp(-Δt/τ), so smaller c = faster ventilation
        # Range from c_global - 0.1 to c_global + 0.05
        c_spread = np.linspace(
            max(0.3, c_global - 0.15),
            min(0.99, c_global + 0.05),
            K,
        )
        # Add small random perturbation for symmetry breaking
        c_init = c_spread + rng.randn(K) * 0.02
        c_init = np.clip(c_init, 0.01, 0.999)

        # Drift terms: represent increasing occupancy levels
        # Unoccupied states have μ ≈ 0, occupied have μ > 0
        # Estimate a reasonable scale from data dynamics
        dy = np.diff(y)
        dy_positive = dy[dy > 0]
        if len(dy_positive) > 0:
            typical_rise = np.percentile(dy_positive, 75)
        else:
            typical_rise = 10.0

        # Half states unoccupied (μ ≈ 0), half occupied (μ > 0)
        mu_init = np.zeros(K)
        n_occ = K // 2
        n_unocc = K - n_occ
        # Unoccupied states: small μ around 0
        mu_init[:n_unocc] = rng.uniform(-2, 2, n_unocc)
        # Occupied states: positive μ, spread over observed dynamics
        mu_init[n_unocc:] = np.linspace(
            typical_rise * 0.3,
            typical_rise * 1.5,
            n_occ,
        ) + rng.randn(n_occ) * 1.0
        mu_init[n_unocc:] = np.maximum(mu_init[n_unocc:], 1.0)

        # Noise variance: start from global AR residual variance
        residuals = y[1:] - c_global * y[:-1]
        global_sigma2 = np.var(residuals) + 1e-4
        sigma2_init = np.full(K, global_sigma2)

        # Transition matrix: strong self-transitions (persistence)
        # Occupancy states typically last multiple hours
        self_prob = 0.85
        A_init = np.full((K, K), (1 - self_prob) / (K - 1))
        np.fill_diagonal(A_init, self_prob)
        # Ensure each row sums to 1
        A_init /= A_init.sum(axis=1, keepdims=True)

        # Initial distribution: uniform
        pi_init = np.ones(K) / K

        return SwitchingARParams(
            c=c_init, mu=mu_init, sigma2=sigma2_init,
            A=A_init, pi=pi_init,
        )

    def _compute_log_emission(
        self, y: np.ndarray, params: SwitchingARParams,
    ) -> np.ndarray:
        """Compute log P(y_t | y_{t-1}, S_t=k) for all t and k.

        The AR(1) emission model for state k is:
            y_t | y_{t-1}, S_t=k ~ N(c_k · y_{t-1} + μ_k, σ²_k)

        So:
            log P(y_t | y_{t-1}, S_t=k) = -0.5·log(2πσ²_k)
                                           - (y_t - c_k·y_{t-1} - μ_k)² / (2σ²_k)

        For t=0, we use the marginal: y_0 ~ N(μ_k/(1-c_k), σ²_k/(1-c_k²)).

        Args:
            y: Observation sequence, shape (T,).
            params: Current model parameters.

        Returns:
            Log-emission matrix, shape (T, K).
        """
        T = len(y)
        K = params.K
        log_emit = np.zeros((T, K))

        for k in range(K):
            c_k = params.c[k]
            mu_k = params.mu[k]
            s2_k = params.sigma2[k]

            # t=0: marginal distribution of stationary AR(1)
            # Stationary mean: μ/(1-c), stationary variance: σ²/(1-c²)
            if abs(1 - c_k) > 1e-6:
                stat_mean = mu_k / (1.0 - c_k)
            else:
                stat_mean = y[0]  # degenerate: use observation
            if abs(1 - c_k**2) > 1e-6:
                stat_var = s2_k / (1.0 - c_k**2)
            else:
                stat_var = s2_k * 100.0  # large variance for near-unit-root
            stat_var = max(stat_var, self.min_sigma2)

            log_emit[0, k] = (
                -0.5 * np.log(2.0 * np.pi * stat_var)
                - 0.5 * (y[0] - stat_mean)**2 / stat_var
            )

            # t=1,...,T-1: conditional AR(1) distribution
            # y_t | y_{t-1}, S_t=k ~ N(c_k * y_{t-1} + μ_k, σ²_k)
            predicted = c_k * y[:-1] + mu_k            # (T-1,)
            residuals = y[1:] - predicted               # (T-1,)
            log_emit[1:, k] = (
                -0.5 * np.log(2.0 * np.pi * s2_k)
                - 0.5 * residuals**2 / s2_k
            )

        return log_emit

    def _forward_backward(
        self, log_emit: np.ndarray, params: SwitchingARParams,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm in log-space for numerical stability.

        E-step of the Baum-Welch algorithm. Computes:
        - γ_t(k) = P(S_t = k | y_{1:T}, θ)         [posterior state probs]
        - ξ_t(j,k) = P(S_{t-1}=j, S_t=k | y_{1:T}) [pairwise posteriors]
        - log P(y_{1:T} | θ)                         [total log-likelihood]

        All computations done in log-space using logsumexp to avoid underflow.

        Args:
            log_emit: Log emission probabilities, shape (T, K).
            params: Current parameters.

        Returns:
            gamma: Posterior state probabilities, shape (T, K).
            xi_sum: Expected transition counts, shape (K, K).
                    xi_sum[j,k] = SUM_{t=1}^{T-1} ξ_t(j,k)
            log_likelihood: Total log-likelihood.
        """
        T, K = log_emit.shape
        log_A = np.log(params.A + 1e-300)
        log_pi = np.log(params.pi + 1e-300)

        # --- Forward pass ---
        # log_alpha[t, k] = log P(y_{1:t}, S_t=k | θ)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_emit[0]

        for t in range(1, T):
            for k in range(K):
                # log P(y_{1:t}, S_t=k) = log_emit[t,k] +
                #   logsumexp_j( log_alpha[t-1, j] + log_A[j, k] )
                log_alpha[t, k] = log_emit[t, k] + logsumexp(
                    log_alpha[t - 1] + log_A[:, k]
                )

        # Total log-likelihood
        log_likelihood = float(logsumexp(log_alpha[-1]))

        # --- Backward pass ---
        # log_beta[t, k] = log P(y_{t+1:T} | S_t=k, θ)
        log_beta = np.zeros((T, K))
        # log_beta[T-1] = 0 (i.e., beta[T-1] = 1)

        for t in range(T - 2, -1, -1):
            for k in range(K):
                # log P(y_{t+1:T} | S_t=k) =
                #   logsumexp_j( log_A[k, j] + log_emit[t+1, j] + log_beta[t+1, j] )
                log_beta[t, k] = logsumexp(
                    log_A[k, :] + log_emit[t + 1] + log_beta[t + 1]
                )

        # --- Posterior state probabilities ---
        # γ_t(k) = P(S_t=k | y_{1:T}) ∝ α_t(k) · β_t(k)
        log_gamma = log_alpha + log_beta
        # Normalize: subtract logsumexp across states
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # --- Expected transition counts ---
        # ξ_t(j,k) = P(S_{t-1}=j, S_t=k | y_{1:T})
        #           ∝ α_{t-1}(j) · A[j,k] · emit(y_t|k) · β_t(k)
        xi_sum = np.zeros((K, K))
        for t in range(1, T):
            # log_xi[j, k] = log_alpha[t-1,j] + log_A[j,k]
            #                + log_emit[t,k] + log_beta[t,k]
            log_xi = (
                log_alpha[t - 1, :, None]   # (K, 1)
                + log_A                      # (K, K)
                + log_emit[t, None, :]       # (1, K)
                + log_beta[t, None, :]       # (1, K)
            )
            # Normalize
            log_xi -= logsumexp(log_xi.ravel())
            xi_sum += np.exp(log_xi)

        return gamma, xi_sum, log_likelihood

    def _m_step(
        self, y: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray,
        params: SwitchingARParams,
    ) -> SwitchingARParams:
        """M-step: update parameters from sufficient statistics.

        Updates:
            π_k = γ_1(k)                                    (initial probs)
            A[j,k] = ξ_sum[j,k] / Σ_k' ξ_sum[j,k']        (transition matrix)
            c_k, μ_k = weighted least squares on:
                y_t ≈ c_k · y_{t-1} + μ_k  with weights γ_t(k)
            σ²_k = weighted residual variance

        The WLS for (c_k, μ_k) solves the 2×2 system:
            [Σ w·y²_{t-1}   Σ w·y_{t-1}] [c_k]   [Σ w·y_t·y_{t-1}]
            [Σ w·y_{t-1}    Σ w         ] [μ_k] = [Σ w·y_t         ]

        where w = γ_t(k) for t=1,...,T-1.

        Args:
            y: Observation sequence, shape (T,).
            gamma: Posterior state probs from E-step, shape (T, K).
            xi_sum: Expected transition counts, shape (K, K).
            params: Current parameters (for structure).

        Returns:
            Updated parameters.
        """
        T = len(y)
        K = params.K
        new_params = params.copy()

        # --- Initial distribution ---
        new_params.pi = gamma[0] + 1e-10
        new_params.pi /= new_params.pi.sum()

        # --- Transition matrix ---
        row_sums = xi_sum.sum(axis=1, keepdims=True) + 1e-10
        new_params.A = xi_sum / row_sums

        # --- AR parameters: weighted least squares per state ---
        y_prev = y[:-1]     # y_{t-1}, shape (T-1,)
        y_curr = y[1:]       # y_t,     shape (T-1,)

        for k in range(K):
            # Weights from posterior (skip t=0, AR model starts at t=1)
            w = gamma[1:, k]           # (T-1,)
            w_sum = w.sum() + 1e-10

            # Weighted sums for the 2×2 normal equations
            w_y2_prev = np.sum(w * y_prev**2)       # Σ w · y²_{t-1}
            w_y_prev = np.sum(w * y_prev)            # Σ w · y_{t-1}
            w_y_curr_prev = np.sum(w * y_curr * y_prev)  # Σ w · y_t · y_{t-1}
            w_y_curr = np.sum(w * y_curr)            # Σ w · y_t

            # Solve: E @ [c_k, μ_k]^T = D
            E = np.array([
                [w_y2_prev, w_y_prev],
                [w_y_prev,  w_sum   ],
            ])
            D = np.array([w_y_curr_prev, w_y_curr])

            det = E[0, 0] * E[1, 1] - E[0, 1] * E[1, 0]
            if abs(det) > 1e-12:
                # Cramer's rule for 2×2
                c_k_new = (D[0] * E[1, 1] - D[1] * E[0, 1]) / det
                mu_k_new = (E[0, 0] * D[1] - E[1, 0] * D[0]) / det
            else:
                # Near-singular: keep previous values
                c_k_new = params.c[k]
                mu_k_new = params.mu[k]

            # Physics constraint: c_k ∈ (0, 1) (exponential decay)
            if self.constrain_ar:
                c_k_new = np.clip(c_k_new, 0.01, 0.999)

            new_params.c[k] = c_k_new
            new_params.mu[k] = mu_k_new

            # --- Noise variance ---
            # σ²_k = Σ_t w_t · (y_t - c_k·y_{t-1} - μ_k)² / Σ_t w_t
            residuals = y_curr - c_k_new * y_prev - mu_k_new
            new_params.sigma2[k] = max(
                np.sum(w * residuals**2) / w_sum,
                self.min_sigma2,
            )

        return new_params

    def _viterbi(
        self, log_emit: np.ndarray, params: SwitchingARParams,
    ) -> np.ndarray:
        """Viterbi algorithm: find the most likely state sequence.

        Finds: S*_{1:T} = argmax_{S_{1:T}} P(S_{1:T}, y_{1:T} | θ)

        Args:
            log_emit: Log emission probabilities, shape (T, K).
            params: Model parameters.

        Returns:
            Most likely state sequence, shape (T,).
        """
        T, K = log_emit.shape
        log_A = np.log(params.A + 1e-300)
        log_pi = np.log(params.pi + 1e-300)

        # Viterbi log-probabilities
        V = np.zeros((T, K))
        B = np.zeros((T, K), dtype=int)  # backpointers

        V[0] = log_pi + log_emit[0]

        for t in range(1, T):
            for k in range(K):
                scores = V[t - 1] + log_A[:, k]
                B[t, k] = np.argmax(scores)
                V[t, k] = scores[B[t, k]] + log_emit[t, k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            states[t] = B[t + 1, states[t + 1]]

        return states

    def fit(self, co2: np.ndarray) -> SwitchingARResult:
        """Fit the switching AR-HMM via EM and decode the state sequence.

        Args:
            co2: Raw CO2 concentration time series in ppm, shape (T,).

        Returns:
            SwitchingARResult with fitted parameters, state sequence,
            posteriors, and occupancy labels.
        """
        # Convert to excess above ambient
        y = co2.astype(np.float64) - self.co2_ambient
        T = len(y)

        logger.info(
            f"Fitting Switching AR-HMM: K={self.n_states}, T={T}, "
            f"Δt={self.delta_t_hours}h, CO2_ambient={self.co2_ambient} ppm"
        )
        logger.info(
            f"  CO2 range: [{co2.min():.1f}, {co2.max():.1f}] ppm, "
            f"excess range: [{y.min():.1f}, {y.max():.1f}] ppm"
        )

        # Initialize
        params = self._initialize_params(y)
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.n_iter):
            # E-step
            log_emit = self._compute_log_emission(y, params)
            gamma, xi_sum, log_ll = self._forward_backward(log_emit, params)

            # Check convergence
            rel_change = abs(log_ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if iteration % 20 == 0 or iteration < 5:
                logger.info(
                    f"  EM iter {iteration:4d}: "
                    f"LL = {log_ll:.2f}, "
                    f"ΔLL = {rel_change:.2e}"
                )
            if iteration > 0 and rel_change < self.tol:
                converged = True
                logger.info(
                    f"  Converged at iteration {iteration} "
                    f"(ΔLL = {rel_change:.2e} < {self.tol})"
                )
                break
            prev_ll = log_ll

            # M-step
            params = self._m_step(y, gamma, xi_sum, params)

        if not converged:
            logger.warning(
                f"  EM did not converge after {self.n_iter} iterations "
                f"(final ΔLL = {rel_change:.2e})"
            )

        # Final E-step for posteriors
        log_emit = self._compute_log_emission(y, params)
        gamma, _, log_ll = self._forward_backward(log_emit, params)

        # Viterbi decoding for hard state sequence
        states = self._viterbi(log_emit, params)

        self.params_ = params
        self._is_fitted = True

        # Log learned parameters
        logger.info(f"\n  Learned parameters (K={self.n_states}):")
        for k in range(self.n_states):
            # Recover physical quantities
            c_k = params.c[k]
            mu_k = params.mu[k]
            dt_s = self.delta_t_hours * 3600.0
            if c_k > 0 and c_k < 1:
                tau_k = -dt_s / np.log(c_k)
                tau_h = tau_k / 3600.0
            else:
                tau_h = np.inf
            count_k = int((states == k).sum())
            pct_k = count_k / T * 100
            logger.info(
                f"    State {k}: c={c_k:.4f} (τ={tau_h:.1f}h), "
                f"μ={mu_k:.2f} ppm, σ={np.sqrt(params.sigma2[k]):.1f} ppm, "
                f"count={count_k} ({pct_k:.1f}%)"
            )

        # Map states to occupancy
        occupancy_map, occupancy_binary = self._map_states_to_occupancy(
            states, params, gamma
        )

        return SwitchingARResult(
            states=states,
            gamma=gamma,
            log_likelihood=log_ll,
            params=params,
            n_iter=iteration + 1,
            converged=converged,
            occupancy_binary=occupancy_binary,
            occupancy_map=occupancy_map,
        )

    def _map_states_to_occupancy(
        self, states: np.ndarray, params: SwitchingARParams,
        gamma: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        """Map HMM states to binary occupancy using physical interpretation.

        Strategy — Robust mapping based on the drift term μ_k:

        The CO2 mass-balance gives: y_t = c_k · y_{t-1} + μ_k + ε_t
        where μ_k = (1 - c_k) · r · n_k encodes occupancy.

        Key insight: μ_k is the *net CO2 drift per step*. When people are
        present they generate CO2, so μ_k > 0. When absent, CO2 decays
        toward ambient, so μ_k ≤ 0 (or very small).

        Previous approach using generation = μ/(1-c) fails when c → 1
        because (1-c) → 0 causes numerical explosion. Instead, we use μ_k
        directly for the binary split:

        1. Compute Otsu threshold on μ values (bounded, well-behaved)
        2. Enforce a minimum threshold of 0 — states with negative μ
           (CO2 declining) are never "occupied"
        3. Log the effective generation rate for interpretability, but
           clip it to prevent extreme outliers from c → 1

        Returns:
            occupancy_map: dict mapping state index → {"label": str, "occupied": bool}
            occupancy_binary: Binary array, shape (T,).
        """
        K = params.K
        occupancy_map = {}

        mu_values = params.mu.copy()

        # Compute effective generation rate for interpretability only
        # Clip c to prevent division-by-near-zero explosion
        c_clipped = np.clip(params.c, 0.0, 0.98)
        generation = mu_values / (1.0 - c_clipped)

        # Otsu threshold on μ (not on generation) — μ is bounded and
        # directly represents per-step CO2 drift in ppm
        mu_threshold = self._otsu_threshold(mu_values)

        # Physical floor: states with μ ≤ 0 are never occupied
        # (negative drift = CO2 decaying = no occupants generating CO2)
        mu_threshold = max(mu_threshold, 0.0)

        logger.info(f"\n  Occupancy mapping (μ threshold = {mu_threshold:.2f} ppm):")
        for k in range(K):
            is_occ = bool(mu_values[k] > mu_threshold)
            label = "occupied" if is_occ else "unoccupied"

            # Physical time constant (for logging)
            c_k = params.c[k]
            dt_s = self.delta_t_hours * 3600.0
            if 0 < c_k < 1:
                tau_h = (-dt_s / np.log(c_k)) / 3600.0
            else:
                tau_h = np.inf

            occupancy_map[k] = {
                "label": label,
                "occupied": is_occ,
                "c": float(c_k),
                "mu": float(mu_values[k]),
                "generation": float(generation[k]),
                "tau_hours": round(float(tau_h), 1),
            }
            marker = "●" if is_occ else "○"
            logger.info(
                f"    State {k}: μ={mu_values[k]:>+7.2f} ppm, "
                f"c={c_k:.4f} (τ={tau_h:.1f}h), "
                f"gen={generation[k]:>+8.1f} → {marker} {label}"
            )

        # Build binary array
        occupied_states = [k for k, v in occupancy_map.items() if v["occupied"]]
        occupancy_binary = np.isin(states, occupied_states).astype(int)

        occ_rate = occupancy_binary.mean()
        logger.info(
            f"  Occupancy rate: {occ_rate:.1%} "
            f"(occupied states: {occupied_states})"
        )

        return occupancy_map, occupancy_binary

    @staticmethod
    def _otsu_threshold(values: np.ndarray) -> float:
        """Otsu's method for finding the optimal binary split threshold.

        Minimizes the weighted within-class variance (or equivalently
        maximizes between-class variance) for a 1D array of values.

        Falls back to the mean if there are fewer than 2 distinct values.
        """
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        if n < 2:
            return float(sorted_vals.mean())

        best_threshold = float(sorted_vals.mean())
        best_variance = -1.0

        for i in range(1, n):
            class0 = sorted_vals[:i]
            class1 = sorted_vals[i:]
            w0 = len(class0) / n
            w1 = len(class1) / n
            # Between-class variance
            between_var = w0 * w1 * (class0.mean() - class1.mean())**2
            if between_var > best_variance:
                best_variance = between_var
                best_threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0

        return best_threshold

    def predict_states(self, co2: np.ndarray) -> np.ndarray:
        """Predict state sequence for new CO2 data using fitted parameters.

        Args:
            co2: CO2 concentration in ppm, shape (T,).

        Returns:
            State labels, shape (T,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        y = co2.astype(np.float64) - self.co2_ambient
        log_emit = self._compute_log_emission(y, self.params_)
        return self._viterbi(log_emit, self.params_)

    def predict_occupancy(self, co2: np.ndarray) -> np.ndarray:
        """Predict binary occupancy for new CO2 data.

        Args:
            co2: CO2 concentration in ppm, shape (T,).

        Returns:
            Binary occupancy array, shape (T,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        states = self.predict_states(co2)
        occupied_states = [
            k for k, v in self._occupancy_map.items() if v["occupied"]
        ]
        return np.isin(states, occupied_states).astype(int)

    @property
    def _occupancy_map(self) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        # Re-derive from stored params using μ-based threshold
        # (consistent with _map_states_to_occupancy logic)
        K = self.params_.K
        mu_values = self.params_.mu
        mu_threshold = max(self._otsu_threshold(mu_values), 0.0)
        occ_map = {}
        for k in range(K):
            c_clipped = min(self.params_.c[k], 0.98)
            generation = mu_values[k] / (1.0 - c_clipped)
            occ_map[k] = {
                "occupied": bool(mu_values[k] > mu_threshold),
                "mu": float(mu_values[k]),
                "generation": float(generation),
            }
        return occ_map
