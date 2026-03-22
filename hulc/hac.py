"""HAC variance estimation matching R's sandwich::vcovHAC defaults.

Defaults: Quadratic Spectral kernel, Andrews (1991) automatic bandwidth,
VAR(1) prewhitening, n/(n-k) finite-sample adjustment.

Reference: Andrews DWK (1991). "Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix Estimation." Econometrica, 59, 817-858.
"""

import numpy as np
from scipy import stats


# ── Kernels ──────────────────────────────────────────────────────────────────

def _qs_kernel(x):
    """Quadratic Spectral kernel (Andrews 1991, Table II)."""
    out = np.ones_like(x, dtype=float)
    nz = np.abs(x) > 1e-10
    z = x[nz]
    piz = 6.0 * np.pi * z / 5.0
    out[nz] = (25.0 / (12.0 * np.pi**2 * z**2)) * (np.sin(piz) / piz - np.cos(piz))
    return out


def _bartlett_kernel(x):
    return np.where(np.abs(x) <= 1, 1.0 - np.abs(x), 0.0)


_KERNELS = {"qs": (_qs_kernel, 2), "bartlett": (_bartlett_kernel, 1)}


# ── Andrews bandwidth ────────────────────────────────────────────────────────

def _bw_andrews(U, q):
    """Andrews (1991) data-driven bandwidth from estimating functions U."""
    n, d = U.shape
    rho, sig2 = np.zeros(d), np.zeros(d)
    for j in range(d):
        u = U[:, j]
        denom = np.sum(u[:-1]**2)
        rho[j] = np.clip(np.sum(u[1:] * u[:-1]) / denom, -0.99, 0.99) if denom > 1e-15 else 0.0
        sig2[j] = np.mean((u[1:] - rho[j] * u[:-1])**2)

    num = den = 0.0
    for j in range(d):
        s4, r = sig2[j]**2, rho[j]
        den += s4 / max((1 - r)**4, 1e-30)
        num += 4 * r**2 * s4 / max((1 - r)**(4 + 4*(q - 1)), 1e-30)

    alpha = num / max(den, 1e-30)
    if q == 1:
        return max(1.1447 * (alpha * n) ** (1/3), 1.0)
    return max(1.3221 * (alpha * n) ** 0.2, 1.0)


# ── Prewhitening ──────────────────────────────────────────────────────────────

def _prewhiten(U):
    A_T = np.linalg.lstsq(U[:-1], U[1:], rcond=None)[0]
    return U[1:] - U[:-1] @ A_T, A_T.T


def _recolor(S, A):
    inv = np.linalg.inv(np.eye(A.shape[0]) - A)
    return inv @ S @ inv.T


# ── Variance estimators ──────────────────────────────────────────────────────

def vcov_hac(X, residuals, kernel="qs", bw=None, prewhite=True, adjust=True):
    """HAC variance-covariance matrix for OLS beta_hat."""
    n, d = X.shape
    kfunc, q = _KERNELS[kernel]
    U = X * residuals[:, None]

    A = None
    if prewhite:
        U_pw, A = _prewhiten(U)
    else:
        U_pw = U

    n_eff = len(U_pw)
    if bw is None:
        bw = _bw_andrews(U_pw, q)

    S = np.zeros((d, d))
    for j in range(-n_eff + 1, n_eff):
        w = kfunc(np.array([j / bw]))[0]
        if abs(w) < 1e-15:
            continue
        if j >= 0:
            S += w * U_pw[j:].T @ U_pw[:n_eff - j] / n_eff
        else:
            S += w * U_pw[:n_eff + j].T @ U_pw[-j:] / n_eff

    if A is not None:
        S = _recolor(S, A)
    if adjust:
        S *= n / (n - d)

    bread = np.linalg.inv(X.T @ X / n)
    return bread @ S @ bread / n


def vcov_hc(X, residuals, adjust=True):
    """HC (White) variance — ignores autocorrelation."""
    n, d = X.shape
    Xu = X * residuals[:, None]
    meat = Xu.T @ Xu / n
    if adjust:
        meat *= n / (n - d)
    bread = np.linalg.inv(X.T @ X / n)
    return bread @ meat @ bread / n


# ── CI functions ─────────────────────────────────────────────────────────────

def _wald_ci(X, Y, c, V, alpha):
    """Wald CI from a variance-covariance matrix."""
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    se = np.sqrt(max(c @ V @ c, 0))
    z = stats.norm.ppf(1 - alpha / 2)
    pt = c @ beta
    return pt - z * se, pt + z * se, 2 * z * se


def hc_ci(X, Y, c, alpha=0.05):
    """Classical HC (White) sandwich CI — ignores autocorrelation."""
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    V = vcov_hc(X, Y - X @ beta)
    return _wald_ci(X, Y, c, V, alpha)


def hac_ci(X, Y, c, alpha=0.05, kernel="qs", prewhite=True):
    """HAC Wald CI for c'beta (sandwich::vcovHAC defaults)."""
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    V = vcov_hac(X, Y - X @ beta, kernel=kernel, prewhite=prewhite)
    return _wald_ci(X, Y, c, V, alpha)
