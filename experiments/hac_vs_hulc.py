"""
Monte Carlo study: HC vs HAC vs HulC for regression with AR(1) errors.

    Y = X beta_0 + eps,   Cov(eps)_{i,j} = rho^|i-j|
    X: fixed AR(0.8) design (nearby rows correlated, differences O(1))

Methods: HC (White), HAC (QS+Andrews+prewhite), HulC (raw), HulC (differenced).

Usage:
    python experiments/hac_vs_hulc.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hulc.hulc import HulC, split_batches
from hulc.hac import hc_ci, hac_ci

# ── Config ────────────────────────────────────────────────────────────────────

N, D = 500, 2
BETA = np.array([1.0, -2.0])
ALPHA = 0.05
N_SIMS = 2000
B, GAP = 8, 10

RHOS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
C_VECS = {
    "c=(1,0) [β₁]": np.array([1.0, 0.0]),
    "c=(0,1) [β₂]": np.array([0.0, 1.0]),
    "c=(1,1)/√2":   np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
}


def ar1_errors(n, rho, rng):
    if abs(rho) < 1e-12:
        return rng.normal(0, 1, n)
    if abs(rho - 1) < 1e-12:
        return np.cumsum(rng.normal(0, 1, n))
    eta = rng.normal(0, np.sqrt(1 - rho**2), n)
    eps = np.zeros(n)
    eps[0] = rng.normal()
    for t in range(1, n):
        eps[t] = rho * eps[t-1] + eta[t]
    return eps


def hulc_raw_ci(X, Y, c, alpha, B, gap):
    n = len(Y)
    batches = split_batches(n, B, gap)
    def proj(idx):
        Xb, Yb = X[idx], Y[idx]
        if len(idx) < X.shape[1]:
            return np.nan
        return float(c @ np.linalg.lstsq(Xb, Yb, rcond=None)[0])
    theta = np.array([proj(b) for b in batches])
    if np.any(np.isnan(theta)):
        return np.nan, np.nan, np.nan
    lo, hi = HulC(alpha, B)._ci_from_estimates(theta)
    return lo, hi, hi - lo


def hulc_diff_ci(X, Y, c, alpha, B, gap):
    Xd, Yd = X[1:] - X[:-1], Y[1:] - Y[:-1]
    batches = split_batches(len(Yd), B, gap)
    def proj(idx):
        Xb, Yb = Xd[idx], Yd[idx]
        if len(idx) < Xd.shape[1]:
            return np.nan
        return float(c @ np.linalg.lstsq(Xb, Yb, rcond=None)[0])
    theta = np.array([proj(b) for b in batches])
    if np.any(np.isnan(theta)):
        return np.nan, np.nan, np.nan
    lo, hi = HulC(alpha, B)._ci_from_estimates(theta)
    return lo, hi, hi - lo


METHODS = {
    "hc":        lambda X, Y, c: hc_ci(X, Y, c, ALPHA),
    "hac":       lambda X, Y, c: hac_ci(X, Y, c, ALPHA),
    "hulc_raw":  lambda X, Y, c: hulc_raw_ci(X, Y, c, ALPHA, B, GAP),
    "hulc_diff": lambda X, Y, c: hulc_diff_ci(X, Y, c, ALPHA, B, gap=5),
}


def run():
    results = {}
    rng = np.random.default_rng(42)
    # Fixed AR(0.8) design: nearby rows are correlated (HC fails),
    # but differences are O(1) (differenced OLS works).
    rng_x = np.random.default_rng(999)
    X_cols = []
    for _ in range(D):
        x = np.zeros(N)
        x[0] = rng_x.normal()
        for t in range(1, N):
            x[t] = 0.8 * x[t-1] + rng_x.normal(0, np.sqrt(1 - 0.64))
        X_cols.append(x)
    X_fixed = np.column_stack(X_cols)

    total = len(C_VECS) * len(RHOS)
    idx = 0
    for c_name, c_vec in C_VECS.items():
        target = float(c_vec @ BETA)
        results[c_name] = {"target": target, "rho_results": []}
        for rho in RHOS:
            idx += 1
            t0 = time.time()
            counters = {m: {"covers": 0, "widths": [], "valid": 0} for m in METHODS}
            for _ in range(N_SIMS):
                eps = ar1_errors(N, rho, rng)
                Y = X_fixed @ BETA + eps
                for m, fn in METHODS.items():
                    lo, hi, w = fn(X_fixed, Y, c_vec)
                    if not np.isnan(w):
                        counters[m]["valid"] += 1
                        counters[m]["widths"].append(w)
                        if lo <= target <= hi:
                            counters[m]["covers"] += 1
            dt = time.time() - t0
            covs = " ".join(f"{m}:{counters[m]['covers']/max(counters[m]['valid'],1):.3f}" for m in METHODS)
            print(f"[{idx}/{total}] {c_name} ρ={rho:.2f} ({dt:.0f}s) {covs}")
            entry = {"rho": rho}
            for m in METHODS:
                cm = counters[m]
                entry[m] = {
                    "coverage": cm["covers"] / max(cm["valid"], 1),
                    "mean_width": float(np.mean(cm["widths"])) if cm["widths"] else 0,
                    "median_width": float(np.median(cm["widths"])) if cm["widths"] else 0,
                    "valid_sims": cm["valid"],
                }
            results[c_name]["rho_results"].append(entry)
    return results


if __name__ == "__main__":
    results = run()
    results_path = os.path.join(os.path.dirname(__file__), "..", "reports", "study_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
