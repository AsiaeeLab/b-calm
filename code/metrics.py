from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def posterior_metrics(mean: np.ndarray, var: np.ndarray, truth: np.ndarray, level: float = 0.90) -> dict[str, float]:
    z = norm.ppf((1.0 + level) / 2.0)
    sd = np.sqrt(np.maximum(var, 1.0e-12))
    lo = mean - z * sd
    hi = mean + z * sd
    return {
        "cate_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "iae": float(np.mean(np.abs(mean - truth))),
        "coverage_90": float(np.mean((truth >= lo) & (truth <= hi))),
        "ci_width": float(np.mean(hi - lo)),
        "ci_width_mean": float(np.mean(hi - lo)),
    }


def coverage_at_level(mean: np.ndarray, var: np.ndarray, truth: np.ndarray, level: float) -> float:
    z = norm.ppf((1.0 + level) / 2.0)
    sd = np.sqrt(np.maximum(var, 1.0e-12))
    lo = mean - z * sd
    hi = mean + z * sd
    return float(np.mean((truth >= lo) & (truth <= hi)))


def negative_transfer(cate_rmse: float, rct_rmse: float) -> int:
    return int(float(cate_rmse) > float(rct_rmse))


def empirical_os_ess(sigma_o2: float, var_full: float, var_rct: float) -> float:
    if not np.isfinite(var_full) or not np.isfinite(var_rct):
        return float("nan")
    contribution = (1.0 / max(var_full, 1.0e-12)) - (1.0 / max(var_rct, 1.0e-12))
    return float(max(0.0, sigma_o2 * contribution))


def theory_os_ess(n_O: int, sigma_delta: float, sigma_o2: float) -> float:
    if math.isinf(float(sigma_delta)):
        return 0.0
    if float(sigma_delta) <= 0.0:
        return float(n_O)
    return float(sigma_o2 / (sigma_o2 / float(n_O) + float(sigma_delta) ** 2))


def interval(mean: np.ndarray, var: np.ndarray, level: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    z = norm.ppf((1.0 + level) / 2.0)
    sd = np.sqrt(np.maximum(var, 1.0e-12))
    return mean - z * sd, mean + z * sd
