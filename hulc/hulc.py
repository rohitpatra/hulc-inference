"""HulC (Hull Confidence) inference.

Distribution-free confidence intervals via batch splitting and order statistics.
No knowledge of convergence rates, limiting distributions, or asymptotic variance required.

Reference: Kuchibhotla, Balakrishnan, Wasserman.
  "The HulC: Confidence Regions from Convex Hulls." JRSS-B, 2024.
"""

import math

import numpy as np
from scipy import stats


def _min_batches(alpha):
    return math.ceil(math.log2(2.0 / alpha))


def _binom_critical(B, alpha):
    return int(B // 2 - stats.binom.ppf(alpha / 2, B, 0.5))


def split_batches(n, B, gap=0):
    """Split {0,...,n-1} into B contiguous batches, optionally with gaps."""
    if gap == 0:
        return [a for a in np.array_split(np.arange(n), B)]
    batch_size = (n - (B - 1) * gap) // B
    if batch_size <= 0:
        raise ValueError(f"n={n} too small for B={B} batches with gap={gap}")
    batches, start = [], 0
    for j in range(B):
        end = start + batch_size if j < B - 1 else n
        batches.append(np.arange(start, min(end, n)))
        start = end + gap
    return batches


class HulC:
    """Univariate HulC confidence interval."""

    def __init__(self, alpha=0.05, B=None, gap=0):
        self.alpha = alpha
        self.B = B or _min_batches(alpha)
        self.gap = gap
        if self.B < _min_batches(alpha):
            raise ValueError(f"B={self.B} < log2(2/alpha)={_min_batches(alpha)}")

    def ci(self, data, estimator):
        """Compute HulC CI. estimator(data_subset) -> scalar."""
        batches = split_batches(len(data), self.B, self.gap)
        theta = np.array([estimator(data[b]) for b in batches])
        return self._ci_from_estimates(theta)

    def _ci_from_estimates(self, theta):
        B = len(theta)
        c = _binom_critical(B, self.alpha)
        s = np.sort(theta)
        lo = max(0, B // 2 - c - 1)
        hi = min(B - 1, (B + 1) // 2 + c)
        return float(s[lo]), float(s[hi])


def hulc_mean(data, alpha=0.05, B=None, gap=0):
    """HulC CI for the mean."""
    return HulC(alpha, B, gap).ci(data, np.mean)


def hulc_regression(X, Y, c, alpha=0.05, B=None, gap=0):
    """HulC CI for c'beta in Y = X beta + eps."""
    n = len(Y)
    hulc = HulC(alpha, B, gap)
    batches = split_batches(n, hulc.B, gap)

    def proj(idx):
        return float(c @ np.linalg.lstsq(X[idx], Y[idx], rcond=None)[0])

    theta = np.array([proj(b) for b in batches])
    return hulc._ci_from_estimates(theta)
