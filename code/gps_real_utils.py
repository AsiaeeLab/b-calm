from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import fit_method
from bcalm import BCalmLinear
from dgp import basis
from utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, ensure_dirs, sigma_label


# GPS external-control B-CALM uses sigma_0 (baseline-bias prior on b_0) as the
# borrowing-strength knob. With one-arm OS data (A_O = 0 for every EHR row),
# the comparative-bias parameter sigma_delta is not informed by the
# observational sample and we hold it diffuse (sigma_delta = inf) throughout.
# Borrowing flows through the shared baseline outcome surface mu(z) and the
# baseline-bias function b_0(z); sigma_0 controls how strictly EHR-control
# outcomes are shrunk toward the RCT control surface.
SIGMA0_MAIN = [0.0, 0.025, 0.05, 0.1, 0.2, 0.5, math.inf]  # main grid (concentrated where borrowing tails off)
SIGMA0_BIAS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, math.inf]  # finer grid for sensitivity
# BMA prior weights, peaked at moderate-to-low sigma_0 (mild baseline commensurability):
# Heavier mass below 0.1 reflects the clinically reasonable stance that EHR controls
# are at least somewhat exchangeable with RCT controls; substantial mass on
# diffuse end (sigma_0 = inf) hedges against prior misspecification.
BMA_PRIOR_SIGMA0 = {
    0.0: 0.05, 0.025: 0.20, 0.05: 0.30, 0.1: 0.20, 0.2: 0.10, 0.5: 0.05, math.inf: 0.10,
}
# Default sigma_delta value for "B-CALM" calls in this GPS context: diffuse,
# since b_delta is unidentified from one-arm OS and we don't want a tight
# prior to leak through into anything that does happen to depend on it.
SIGMA_DELTA_DIFFUSE = math.inf


def prepare_gps(loaded: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    D_R = loaded["D_R"]
    D_O = loaded["D_O"]
    shared_cols = loaded["shared_cols"]
    rct = {
        "X": D_R[shared_cols].to_numpy(dtype=float),
        "A": D_R["A"].to_numpy(dtype=float),
        "Y": D_R["Y"].to_numpy(dtype=float),
    }
    ehr = {
        "X": D_O[shared_cols].to_numpy(dtype=float),
        "A": D_O["A"].to_numpy(dtype=float),
        "Y": D_O["Y"].to_numpy(dtype=float),
    }
    X_rct = rct["X"]
    return rct, ehr, X_rct


def fit_baseline_bcalm(
    rct: dict[str, np.ndarray],
    ehr: dict[str, np.ndarray],
    sigma_0: float,
    seed: int,
    basis_kind: str = "linear",
) -> BCalmLinear:
    """Literal B-CALM linear-Gaussian posterior, parameterized for one-arm
    external-control borrowing. ``sigma_delta`` is held diffuse because
    A_O is constant zero in the EHR cohort (b_delta is unidentified).
    ``sigma_0`` is the baseline-bias prior scale: small ``sigma_0`` shrinks
    the EHR-control outcome surface toward the RCT-control surface
    (aggressive borrowing); large ``sigma_0`` decouples the two surfaces
    (no borrowing, recovers RCT-only)."""
    model, _ = fit_method(
        "B-CALM", rct, ehr, basis_kind, seed,
        sigma_delta=SIGMA_DELTA_DIFFUSE,
        sigma_0=sigma_0,
    )
    return model


def fit_plain_bcalm(
    rct: dict[str, np.ndarray],
    ehr: dict[str, np.ndarray],
    sigma_delta: float,
    seed: int,
    basis_kind: str = "linear",
) -> BCalmLinear:
    """Literal B-CALM with a sigma_delta argument and the default sigma_0=0.6.
    In the GPS application this is equivalent to fit_baseline_bcalm with
    sigma_0=0.6 because sigma_delta has no effect on the tau posterior under
    one-arm OS."""
    model, _ = fit_method("B-CALM", rct, ehr, basis_kind, seed, sigma_delta=sigma_delta)
    return model


def ate_draws(model: object, X_eval: np.ndarray, seed: int, n_draws: int = 800) -> np.ndarray:
    """Per-bootstrap-or-posterior-draw ATE on the trial covariate distribution.

    For Bayesian models (BCalmLinear, BcfHahnCate) this returns posterior
    draws of (1/n_eval) sum_i tau(x_i). For bootstrap-wrapped point models
    (PointCateModel with bootstrap_models populated) this returns the
    bootstrap distribution of ATE."""
    cate = model.posterior_cate(X_eval, n_samples=n_draws, seed=seed)
    return np.mean(cate, axis=1)


def summarize(draws: np.ndarray) -> dict[str, float]:
    draws = np.asarray(draws, dtype=float)
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return {
        "ate_mean": float(np.mean(draws)),
        "ate_lo": float(lo),
        "ate_hi": float(hi),
        "ate_width": float(hi - lo),
    }


def summarize_model(model: object, X_eval: np.ndarray, seed: int, n_draws: int = 800) -> dict[str, float]:
    return summarize(ate_draws(model, X_eval, seed=seed, n_draws=n_draws))


def select_sigma0(summary: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """BMA-style selection over the sigma_0 grid using a width-and-prior score
    (interpretable as: prefer shorter intervals if the prior weight justifies
    it)."""
    bcalm = summary[summary["method"] == "B-CALM"].copy()
    scores: dict[str, float] = {}
    for sigma, prior in BMA_PRIOR_SIGMA0.items():
        lab = sigma_label(sigma)
        row = bcalm[bcalm["sigma_0"] == lab]
        if row.empty:
            continue
        width = float(row["ate_width"].iloc[0])
        mean = float(row["ate_mean"].iloc[0])
        score = math.log(float(prior)) - math.log(max(width, 1.0e-8)) - 0.02 * abs(mean)
        scores[lab] = score
    selected = max(scores, key=scores.get)
    return selected, scores


def pct_reduction(width: float, rct_width: float) -> float:
    return 100.0 * (1.0 - float(width) / float(rct_width))


def method_summary_from_rows(rows: list[dict], selected_sigma: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep = []
    for (method, sigma), group in df.groupby(["method", "sigma_0"], dropna=False):
        keep.append(
            {
                "method": method,
                "sigma_0": sigma,
                "ate_mean": float(group["ate_mean"].mean()),
                "ate_lo": float(group["ate_lo"].mean()),
                "ate_hi": float(group["ate_hi"].mean()),
                "ate_width": float(group["ate_width"].mean()),
                "n_fits": int(len(group)),
            }
        )
    summary = pd.DataFrame(keep)
    rct_width = float(summary.loc[summary["method"] == "RCT-only", "ate_width"].iloc[0])
    summary["width_reduction_pct"] = summary["ate_width"].map(lambda w: pct_reduction(float(w), rct_width))
    summary["selected"] = (summary["method"].eq("B-CALM") & summary["sigma_0"].eq(selected_sigma))
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def tex_num(x: float, digits: int = 3) -> str:
    if not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def ci_text(lo: float, hi: float) -> str:
    if not np.isfinite(float(lo)) or not np.isfinite(float(hi)):
        return "--"
    return f"[{lo:.3f}, {hi:.3f}]"


def common_setup() -> None:
    ensure_dirs()
    for path in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def b0_group_summary(model: BCalmLinear, X: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    Phi = basis(X, model.basis_kind)
    sl = slice(2 * model.k_, 3 * model.k_)
    mean_b0 = model.mean_[sl]
    cov_b0 = model.cov_[sl, sl]
    rows = []
    for label, idx in groups.groupby(groups, observed=True).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        if loc.size < 20:
            continue
        phi_bar = Phi[loc].mean(axis=0)
        mean = float(phi_bar @ mean_b0)
        var = float(phi_bar @ cov_b0 @ phi_bar)
        sd = math.sqrt(max(var, 1.0e-12))
        rows.append(
            {
                "stratum": str(label),
                "n": int(loc.size),
                "b0_mean": mean,
                "b0_lo": mean - 1.6448536269514722 * sd,
                "b0_hi": mean + 1.6448536269514722 * sd,
                "b0_width": 2.0 * 1.6448536269514722 * sd,
                "abs_b0": abs(mean),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_b0", ascending=False)
