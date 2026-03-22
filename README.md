# HulC Inference

Implementation of the **HulC (Hull Confidence)** method for uniformly valid inference, based on:

> Kuchibhotla, Balakrishnan, Wasserman. *"Uniformly Valid Inference: Beyond CLT and Beyond Independence."*

## What is HulC?

HulC constructs confidence sets that are **uniformly valid** without requiring knowledge of convergence rates, limiting distributions, or asymptotic variance. It works by:

1. Splitting data into B batches
2. Computing an estimator on each batch
3. Using order statistics of batch estimates as a confidence interval

Valid under heavy tails, heteroscedasticity, and dependence — settings where CLT-based methods (Wald, bootstrap) fail.

## Package structure

```
hulc/
  hulc.py          # Univariate HulC (batch splitting + order statistics)
  multivariate.py  # Tukey HulC (half-space) + Rectilinear HulC (L-inf)
  hac.py           # HAC variance estimation (sandwich::vcovHAC defaults)
experiments/
  hac_vs_hulc.py   # Monte Carlo: HC vs HAC vs HulC for AR(1) regression
reports/
  study_report.md    # Detailed findings
  study_report.html  # Interactive plots
  study_results.json # Raw simulation results
```

## Quick start

```python
import numpy as np
from hulc import HulC, hulc_mean, hulc_regression

# CI for the mean (works even for Cauchy!)
data = 5.0 + np.random.standard_cauchy(1000)
lo, hi = hulc_mean(data, alpha=0.05)

# CI for regression coefficient c'beta
X = np.random.normal(0, 1, (500, 3))
Y = X @ [1, -2, 0.5] + np.random.normal(0, 1, 500)
c = np.array([0, 1, 0])  # inference for beta_2
lo, hi = hulc_regression(X, Y, c, alpha=0.05)
```

## Running the experiment

```bash
python experiments/hac_vs_hulc.py    # ~7 min
```

## Key findings

With fixed AR(0.8) design and AR(1) errors:

| rho | HC (White) | HAC (Andrews) | HulC (raw) | HulC (diff) |
|-----|-----------|---------------|------------|-------------|
| 0.0 | 94% | 94% | 99% | 99% |
| 0.5 | 80% | 94% | 99% | 99% |
| 0.9 | 55% | 93% | 99% | 99% |
| 0.99 | 35% | 81% | 99% | **99% (width 0.09)** |
| 1.0 | **19%** | **68%** | **100%** | **99% (width 0.63)** |

HC collapses. HAC degrades at high rho. HulC-diff maintains uniform validity AND gets *narrower* as rho increases.

## References

- Kuchibhotla, Balakrishnan, Wasserman (2024). *The HulC: Confidence Regions from Convex Hulls.* JRSS-B.
- Andrews (1991). *Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation.* Econometrica.
- Potscher, Preinerstorfer (2021). *Failure of the Wald Test under Non-Regular Conditions.* Econometrica.
