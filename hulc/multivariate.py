"""Multivariate HulC confidence sets.

Two extensions:
  1. TukeyHulC — half-space (angular) symmetry, dimension-free.
  2. RectilinearHulC — coordinate-wise median, L-inf norm.

Reference: Kuchibhotla, Balakrishnan, Wasserman.
"""

import numpy as np

from .hulc import _binom_critical, _min_batches, split_batches


class TukeyHulC:
    """Confidence set via Tukey (half-space) depth. Uses B+1 batches."""

    def __init__(self, alpha=0.05, B=None, gap=0):
        self.alpha = alpha
        self.B = B or max(_min_batches(alpha), int(np.ceil(np.log2(1.0 / alpha))))
        self.gap = gap

    def confidence_set(self, data, estimator, V_inv=None):
        batches = split_batches(len(data), self.B + 1, self.gap)
        theta = np.array([estimator(data[b]) for b in batches])
        return self._build(theta, V_inv)

    def _build(self, theta, V_inv=None):
        B = len(theta) - 1
        c = _binom_critical(B, 2 * self.alpha)
        threshold = B // 2 - c
        anchor = theta[B]
        estimates = theta[:B]

        def contains(phi):
            count = 0
            for i in range(B):
                d_anchor = V_inv @ (phi - anchor) if V_inv is not None else (phi - anchor)
                if np.dot(d_anchor, estimates[i] - phi) >= 0:
                    count += 1
            return count >= threshold

        balls = []
        for i in range(B):
            mid = (estimates[i] + anchor) / 2
            diff = estimates[i] - anchor
            r = 0.5 * (np.sqrt(diff @ V_inv @ diff) if V_inv is not None else np.linalg.norm(diff))
            balls.append((mid, r))

        return {"theta_estimates": theta, "anchor": anchor, "threshold": threshold,
                "contains": contains, "bounding_balls": balls}


class RectilinearHulC:
    """Confidence set via rectilinear (coordinate-wise) median. Uses B+1 batches."""

    def __init__(self, alpha=0.05, B=None, gap=0):
        self.alpha = alpha
        self.B = B or max(_min_batches(alpha), int(np.ceil(np.log2(1.0 / alpha))))
        self.gap = gap

    def confidence_set(self, data, estimator, norm=None):
        batches = split_batches(len(data), self.B + 1, self.gap)
        theta = np.array([estimator(data[b]) for b in batches])
        return self._build(theta, norm)

    def _build(self, theta, norm=None):
        if norm is None:
            norm = lambda x: np.max(np.abs(x))

        B = len(theta) - 1
        c = _binom_critical(B, 2 * self.alpha)
        threshold = B // 2 - c
        anchor = theta[B]
        estimates = theta[:B]
        radii = np.array([norm(estimates[i] - anchor) for i in range(B)])

        def contains(phi):
            return int(np.sum(norm(phi - anchor) <= radii)) >= threshold

        sorted_r = np.sort(radii)
        eff_r = sorted_r[B - threshold] if 0 < threshold <= B else sorted_r[-1]

        return {"theta_estimates": theta, "anchor": anchor, "threshold": threshold,
                "radii": radii, "contains": contains,
                "bounding_box": (anchor, np.full(len(anchor), eff_r))}
