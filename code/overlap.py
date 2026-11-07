"""Gaussian-kernel distributional discrepancy using the unbiased U-statistic."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def _kernel_sum(
    x: np.ndarray, y: np.ndarray, bandwidth: float, block_size: int
) -> float:
    total = 0.0
    for i in range(0, len(x), block_size):
        for j in range(0, len(y), block_size):
            d = cdist(
                x[i : i + block_size], y[j : j + block_size], metric="sqeuclidean"
            )
            total += float(np.exp(-d / (2 * bandwidth**2)).sum())
    return total


def _statistic(
    x: np.ndarray, y: np.ndarray, bandwidth: float, block_size: int
) -> float:
    nx, ny = len(x), len(y)
    return (
        (_kernel_sum(x, x, bandwidth, block_size) - nx) / (nx * (nx - 1))
        + (_kernel_sum(y, y, bandwidth, block_size) - ny) / (ny * (ny - 1))
        - 2 * _kernel_sum(x, y, bandwidth, block_size) / (nx * ny)
    )


def gaussian_mmd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int = 200,
    seed: int = 2026,
    block_size: int = 512,
) -> dict[str, float | int]:
    """Return unbiased MMD squared, its clipped square root, and a label-test p.

    Uses the unequal-sample-size U-statistic of Gretton et al. (JMLR, 2012),
    https://jmlr.org/papers/v13/gretton12a.html. The kernel is
    exp(-||x-y||^2 / (2*bandwidth^2)). Standardization uses the full pooled
    mean and population standard deviation (ddof=0); zero-SD columns are
    dropped. Bandwidth is the median off-diagonal Euclidean distance on a
    fixed-seed sample of at most 2,000 pooled standardized observations.

    The reported MMD uses both complete samples, with blockwise kernel sums.
    The permutation test uses the full smaller group and a random subsample
    of min(n_larger, 2000) from the larger group. Standardization and bandwidth
    remain fixed across label permutations. Its observed statistic uses this
    same test subset, not the full-sample statistic. The p-value is
    (1 + number of permuted statistics >= observed)/(1+B). Kernels and all
    patient-level arrays remain in memory. If all columns are constant or the
    median distance is zero, a unit bandwidth gives a well-defined kernel.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or min(len(x), len(y)) < 2
    ):
        raise ValueError(
            "Two finite matrices with matching columns and at least two rows are required"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite features")
    if permutations < 1 or block_size < 1:
        raise ValueError("Positive permutations and block_size required")
    nx, ny = len(x), len(y)
    pool = np.vstack([x, y])
    mean = pool.mean(0)
    sd = pool.std(0)
    keep = sd > 0
    pool = (pool[:, keep] - mean[keep]) / sd[keep]
    rng = np.random.default_rng(seed)
    sample = pool[rng.choice(len(pool), min(2000, len(pool)), replace=False)]
    bandwidth = float(np.median(pdist(sample)))
    if bandwidth <= 0:
        bandwidth = 1.0
    x, y = pool[:nx], pool[nx:]
    mmd2 = _statistic(x, y, bandwidth, block_size)
    if nx >= ny:
        x_test = x[rng.choice(nx, min(nx, 2000), replace=False)]
        y_test = y
    else:
        x_test = x
        y_test = y[rng.choice(ny, min(ny, 2000), replace=False)]
    test = np.vstack([x_test, y_test])
    n = len(test)
    a = len(x_test)
    b = len(y_test)
    # Compute quadratic forms for all label permutations in bounded blocks.
    labels = np.zeros((n, permutations + 1))
    labels[:a, 0] = 1
    for p in range(1, permutations + 1):
        labels[rng.permutation(n)[:a], p] = 1
    xx = np.zeros(permutations + 1)
    xy = xx.copy()
    yy = xx.copy()
    for i in range(0, n, block_size):
        li = labels[i : i + block_size]
        for j in range(0, n, block_size):
            lj = labels[j : j + block_size]
            k = np.exp(
                -cdist(
                    test[i : i + block_size], test[j : j + block_size], "sqeuclidean"
                )
                / (2 * bandwidth**2)
            )
            kl = k @ lj
            kother = k.sum(axis=1)[:, None] - kl
            xx += (li * kl).sum(0)
            xy += (li * kother).sum(0)
            yy += ((1 - li) * kother).sum(0)
    stats = (xx - a) / (a * (a - 1)) + (yy - b) / (b * (b - 1)) - 2 * xy / (a * b)
    pvalue = float((1 + np.count_nonzero(stats[1:] >= stats[0])) / (permutations + 1))
    return dict(
        mmd2=mmd2,
        mmd=float(np.sqrt(max(mmd2, 0))),
        bandwidth=bandwidth,
        n_x=nx,
        n_y=ny,
        permutation_p=pvalue,
        permutation_mmd2=float(stats[0]),
        permutation_n_x=a,
        permutation_n_y=b,
        permutations=permutations,
        retained_columns=int(keep.sum()),
    )
