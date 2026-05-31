"""
gp_scratch.py
=============
Gaussian Process Regression — numpy only, no sklearn GP modules.

Classes
-------
GaussianProcessScratch
    RBF or Matérn-5/2 kernel.
    Hyperparameters fitted by maximising the log marginal likelihood via L-BFGS-B.
    For small datasets (N < 30), fix sigma2 to data variance and optimise ell only.
    Prediction via Cholesky solve.

Constants
---------
KERNELS : dict  {kernel_name: kernel_function}

Usage
-----
    from gp_scratch import GaussianProcessScratch, KERNELS

    gp = GaussianProcessScratch(kernel='matern52', n_restarts=10)
    gp.fit(X_train, y_train, noise_var=alpha_array, fix_sigma2=np.var(y_train))
    mu, std = gp.predict(X_test, return_std=True)
"""

import numpy as np
from scipy.optimize import minimize


# ── Kernel functions ──────────────────────────────────────────────────────────

def _rbf(X1, X2, sigma2, ell):
    """
    Squared-exponential (RBF) kernel.
        k(x, x') = σ² · exp( −||x−x'||² / (2ℓ²) )
    Implies infinitely smooth functions.
    """
    d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return sigma2 * np.exp(-0.5 * d2 / (ell ** 2))


def _matern52(X1, X2, sigma2, ell):
    """
    Matérn-5/2 kernel.
        k(x,x') = σ² · (1 + √5·r/ℓ + 5r²/(3ℓ²)) · exp(−√5·r/ℓ)
    Implies twice-differentiable functions — more realistic than RBF
    for physical alloy systems with composition-dependent anomalies.
    """
    r = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
    s = np.sqrt(5.0) * r / ell
    return sigma2 * (1.0 + s + s**2 / 3.0) * np.exp(-s)


KERNELS = {
    'rbf':      _rbf,
    'matern52': _matern52,
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
    seed : int
        Random seed for restart initialisation.

    Fitted attributes (available after .fit())
    ------------------------------------------
    sigma2    : float  — signal variance (amplitude²)
    ell       : float  — length-scale
    log_ml    : float  — log marginal likelihood at optimum
    X_train   : (N,1)
    alpha_vec : (N,)   — K_noisy⁻¹ · y
    L         : (N,N)  — Cholesky factor of K_noisy
    """

    def __init__(self, kernel: str = 'matern52', n_restarts: int = 10, seed: int = 42):
        assert kernel in KERNELS, f"Unknown kernel '{kernel}'. Choose from {list(KERNELS)}"
        self.kernel_name = kernel
        self.kernel_fn   = KERNELS[kernel]
        self.n_restarts  = n_restarts
        self.seed        = seed
        self.sigma2    = None
        self.ell       = None
        self.log_ml    = None
        self.X_train   = None
        self.y_train   = None
        self.noise_var = None
        self.alpha_vec = None
        self.L         = None

    # ── Internal: joint NLL (sigma2 + ell) ───────────────────────────────────

    def _log_marginal_likelihood(self, log_theta, X, y, noise_var):
        """
        Negative log marginal likelihood for joint optimisation.
        log_theta : (2,) — [log σ², log ℓ]
        """
        sigma2 = np.exp(log_theta[0])
        ell    = np.exp(log_theta[1])
        n      = len(y)
        K      = self.kernel_fn(X, X, sigma2, ell)
        K_n    = K + np.diag(noise_var) + 1e-8 * np.eye(n)
        try:
            L = np.linalg.cholesky(K_n)
        except np.linalg.LinAlgError:
            return 1e10
        v       = np.linalg.solve(L, y)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        lml     = -0.5 * (np.dot(v, v) + log_det + n * np.log(2 * np.pi))
        return -lml

    # ── Internal: NLL with fixed sigma2 (ell only) ───────────────────────────

    def _nll_fixed_sigma2(self, log_ell, X, y, noise_var, sigma2):
        """
        Negative log marginal likelihood with sigma2 fixed.
        Only ell is optimised. Used for small datasets where joint
        optimisation is poorly conditioned.
        log_ell : (1,) — [log ℓ]
        """
        ell = np.exp(log_ell[0])
        n   = len(y)
        K   = self.kernel_fn(X, X, sigma2, ell)
        K_n = K + np.diag(noise_var) + 1e-8 * np.eye(n)
        try:
            L = np.linalg.cholesky(K_n)
        except np.linalg.LinAlgError:
            return 1e10
        v       = np.linalg.solve(L, y)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        lml     = -0.5 * (np.dot(v, v) + log_det + n * np.log(2 * np.pi))
        return -lml

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, noise_var: np.ndarray,
            fix_sigma2: float = None):
        """
        Fit GP hyperparameters by maximising log marginal likelihood.

        Parameters
        ----------
        X          : (N, 1) float — input (Cr fraction)
        y          : (N,)   float — target (elastic constant, GPa)
        noise_var  : (N,)   float — per-point noise VARIANCE σ²
        fix_sigma2 : float or None
            If provided, sigma2 is fixed to this value and only ell is
            optimised. Recommended for N < 30 where joint optimisation is
            poorly conditioned. Pass np.var(y) as a data-informed prior.
        """
        X         = np.atleast_2d(X).reshape(-1, 1)
        y         = np.asarray(y, dtype=float)
        noise_var = np.asarray(noise_var, dtype=float)
        assert len(y) == len(noise_var), "y and noise_var must have the same length"

        self.X_train   = X
        self.y_train   = y
        self.noise_var = noise_var

        rng = np.random.default_rng(self.seed)

        if fix_sigma2 is not None:
            # ── Fixed sigma2: optimise ell only ──────────────────────────────
            self.sigma2  = float(fix_sigma2)
            init_guesses = [np.array([np.log(0.3)])]
            init_guesses += [rng.uniform([-3], [1])
                             for _ in range(self.n_restarts - 1)]
            best_nll = np.inf
            best_ell = None
            for ell0 in init_guesses:
                res = minimize(
                    self._nll_fixed_sigma2,
                    ell0,
                    args=(X, y, noise_var, self.sigma2),
                    method='L-BFGS-B',
                    bounds=[(-4, 2)],
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_ell = res.x
            self.ell    = float(np.exp(best_ell[0]))
            self.log_ml = float(-best_nll)

        else:
            # ── Joint optimisation: sigma2 + ell ─────────────────────────────
            init_guesses = [np.array([np.log(np.var(y) + 1e-6), np.log(0.3)])]
            init_guesses += [rng.uniform([-2, -3], [3, 1])
                             for _ in range(self.n_restarts - 1)]
            best_nll   = np.inf
            best_theta = None
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

        # ── Precompute Cholesky and alpha_vec for prediction ──────────────────
        n   = len(y)
        K   = self.kernel_fn(X, X, self.sigma2, self.ell)
        K_n = K + np.diag(noise_var) + 1e-8 * np.eye(n)
        self.L         = np.linalg.cholesky(K_n)
        v              = np.linalg.solve(self.L, y)
        self.alpha_vec = np.linalg.solve(self.L.T, v)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X_test: np.ndarray, return_std: bool = True):
        """
        Posterior mean and optionally std at X_test.

        μ(x*) = k(x*,X) · K_noisy⁻¹ · y
        σ²(x*)= k(x*,x*)− k(x*,X) · K_noisy⁻¹ · k(X,x*)

        Parameters
        ----------
        X_test     : (M, 1) or (M,) — prediction points
        return_std : bool

        Returns
        -------
        mu  : (M,) posterior mean
        std : (M,) posterior std  [only if return_std=True]
        """
        assert self.alpha_vec is not None, "Call .fit() before .predict()"
        X_test = np.atleast_2d(X_test).reshape(-1, 1)

        K_star = self.kernel_fn(X_test, self.X_train, self.sigma2, self.ell)
        mu     = K_star @ self.alpha_vec

        if not return_std:
            return mu

        K_ss = self.kernel_fn(X_test, X_test, self.sigma2, self.ell)
        v    = np.linalg.solve(self.L, K_star.T)
        var  = np.diag(K_ss) - np.sum(v**2, axis=0)
        var  = np.maximum(var, 0.0)
        return mu, np.sqrt(var)