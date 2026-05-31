"""
gp_scratch.py
=============
Gaussian Process Regression — numpy only, no sklearn GP modules.

Classes
-------
GaussianProcessScratch
    RBF or Matérn-5/2 kernel.
    Hyperparameters (signal variance σ², length-scale ℓ) fitted by
    maximising the log marginal likelihood via L-BFGS-B.
    Prediction via Cholesky solve.

Constants
---------
KERNELS : dict  {kernel_name: kernel_function}
    Passed to GaussianProcessScratch(kernel=...) as a string key.

Usage
-----
    from gp_scratch import GaussianProcessScratch, KERNELS

    gp = GaussianProcessScratch(kernel='matern52', n_restarts=10)
    gp.fit(X_train, y_train, noise_var=alpha_array)   # noise_var = σ² per point
    mu, std = gp.predict(X_test, return_std=True)
"""

import numpy as np
from scipy.optimize import minimize


# ── Kernel functions ──────────────────────────────────────────────────────────
# All kernels have signature:  k(X1, X2, sigma2, ell) → (n1, n2) matrix
# X1, X2 : (n, 1) arrays of Cr fractions

def _rbf(X1, X2, sigma2, ell):
    """
    Squared-exponential (RBF) kernel.
        k(x, x') = σ² · exp( −||x−x'||² / (2ℓ²) )
    Implies infinitely smooth functions.
    """
    d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)  # (n1,n2)
    return sigma2 * np.exp(-0.5 * d2 / (ell ** 2))


def _matern52(X1, X2, sigma2, ell):
    """
    Matérn-5/2 kernel.
        r      = ||x−x'||
        k(x,x')= σ² · (1 + √5·r/ℓ + 5r²/(3ℓ²)) · exp(−√5·r/ℓ)
    Implies twice-differentiable functions — more realistic than RBF
    for physical alloy systems with composition-dependent anomalies.
    """
    r = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))  # (n1,n2)
    s = np.sqrt(5.0) * r / ell
    return sigma2 * (1.0 + s + s**2 / 3.0) * np.exp(-s)


KERNELS = {
    'rbf':       _rbf,
    'matern52':  _matern52,
}


# ── GP class ──────────────────────────────────────────────────────────────────

class GaussianProcessScratch:
    """
    Gaussian Process Regression from scratch.

    Parameters
    ----------
    kernel : str
        Key into KERNELS dict — 'rbf' or 'matern52'.
    n_restarts : int
        Number of random restarts for L-BFGS-B hyperparameter optimisation.
        More restarts → less likely to land in a bad local optimum.
        10 is enough for 1-D input; increase if log_ml values vary widely.
    seed : int
        Random seed for restart initialisation. Set for reproducibility.

    Fitted attributes (available after .fit())
    ------------------------------------------
    sigma2   : float  — signal variance (amplitude²)
    ell      : float  — length-scale
    log_ml   : float  — log marginal likelihood at optimum
    X_train  : (N,1)
    alpha_vec: (N,)   — K_noisy⁻¹ · y (precomputed for fast prediction)
    L        : (N,N)  — Cholesky factor of K_noisy
    """

    def __init__(self, kernel: str = 'matern52', n_restarts: int = 10, seed: int = 42):
        assert kernel in KERNELS, f"Unknown kernel '{kernel}'. Choose from {list(KERNELS)}"
        self.kernel_name = kernel
        self.kernel_fn   = KERNELS[kernel]
        self.n_restarts  = n_restarts
        self.seed        = seed
        # Fitted attributes — set by .fit()
        self.sigma2    = None
        self.ell       = None
        self.log_ml    = None
        self.X_train   = None
        self.y_train   = None
        self.noise_var = None
        self.alpha_vec = None
        self.L         = None

    # ── Internal: log marginal likelihood ────────────────────────────────────

    def _log_marginal_likelihood(self, log_theta, X, y, noise_var):
        """
        Negative log marginal likelihood (minimised by L-BFGS-B).

        log p(y|X,θ) = −½ yᵀ K_noisy⁻¹ y − ½ log|K_noisy| − n/2 log(2π)

        Parameters
        ----------
        log_theta : (2,) array  — [log σ², log ℓ]  (log-space avoids positivity constraints)
        """
        sigma2 = np.exp(log_theta[0])
        ell    = np.exp(log_theta[1])
        n      = len(y)
        K      = self.kernel_fn(X, X, sigma2, ell)
        K_n    = K + np.diag(noise_var) + 1e-8 * np.eye(n)   # jitter for numerical stability
        try:
            L   = np.linalg.cholesky(K_n)
        except np.linalg.LinAlgError:
            return 1e10   # not positive-definite — penalise heavily
        v       = np.linalg.solve(L, y)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        lml     = -0.5 * (np.dot(v, v) + log_det + n * np.log(2 * np.pi))
        return -lml   # negate because scipy minimises

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, noise_var: np.ndarray):
        """
        Fit GP hyperparameters by maximising log marginal likelihood.

        Parameters
        ----------
        X         : (N, 1) float — input (Cr fraction)
        y         : (N,)   float — target (elastic constant, GPa)
        noise_var : (N,)   float — per-point noise VARIANCE σ²  (not σ)
                    From data_utils: alpha[target]
        """
        X         = np.atleast_2d(X).reshape(-1, 1)
        y         = np.asarray(y, dtype=float)
        noise_var = np.asarray(noise_var, dtype=float)
        assert len(y) == len(noise_var), "y and noise_var must have the same length"

        self.X_train   = X
        self.y_train   = y
        self.noise_var = noise_var

        # L-BFGS-B with n_restarts random starting points
        rng       = np.random.default_rng(self.seed)
        best_nll  = np.inf
        best_theta = None

        # First restart: data-informed starting point
        init_guesses = [np.array([np.log(np.var(y) + 1e-6), np.log(0.3)])]
        # Remaining restarts: random in log-space
        init_guesses += [
            rng.uniform([-2, -3], [3, 1])
            for _ in range(self.n_restarts - 1)
        ]

        for theta0 in init_guesses:
            res = minimize(
                self._log_marginal_likelihood,
                theta0,
                args=(X, y, noise_var),
                method='L-BFGS-B',
                bounds=[(-5, 10), (-4, 2)],  
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.fun < best_nll:
                best_nll   = res.fun
                best_theta = res.x

        self.sigma2  = float(np.exp(best_theta[0]))
        self.ell     = float(np.exp(best_theta[1]))
        self.log_ml  = float(-best_nll)

        # Precompute Cholesky and alpha_vec for prediction
        n   = len(y)
        K   = self.kernel_fn(X, X, self.sigma2, self.ell)
        K_n = K + np.diag(noise_var) + 1e-8 * np.eye(n)
        self.L         = np.linalg.cholesky(K_n)
        v              = np.linalg.solve(self.L, y)
        self.alpha_vec = np.linalg.solve(self.L.T, v)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X_test: np.ndarray, return_std: bool = True):
        """
        Posterior mean (and optionally std) at X_test.

        μ(x*) = k(x*,X) · K_noisy⁻¹ · y
        σ²(x*)= k(x*,x*)− k(x*,X) · K_noisy⁻¹ · k(X,x*)

        Parameters
        ----------
        X_test     : (M, 1) or (M,) — prediction points
        return_std : bool — if True returns (mu, std), else returns mu

        Returns
        -------
        mu  : (M,) posterior mean
        std : (M,) posterior standard deviation  [only if return_std=True]
        """
        assert self.alpha_vec is not None, "Call .fit() before .predict()"
        X_test = np.atleast_2d(X_test).reshape(-1, 1)

        K_star  = self.kernel_fn(X_test, self.X_train, self.sigma2, self.ell)  # (M, N)
        mu      = K_star @ self.alpha_vec                                        # (M,)

        if not return_std:
            return mu

        K_ss    = self.kernel_fn(X_test, X_test, self.sigma2, self.ell)         # (M, M)
        v       = np.linalg.solve(self.L, K_star.T)                             # (N, M)
        var     = np.diag(K_ss) - np.sum(v**2, axis=0)                          # (M,)
        var     = np.maximum(var, 0.0)                                           # clip numerical negatives
        return mu, np.sqrt(var)