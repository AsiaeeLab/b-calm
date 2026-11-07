from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

try:
    from .baselines import fit_method
    from .bcalm import BCalmLinear
    from .dgp import basis
    from .gps_data import load_gps
    from .gps_real_utils import (
        ate_draws,
        ci_text,
        common_setup,
        fit_baseline_bcalm,
        prepare_gps,
        summarize,
        tex_num,
    )
    from .utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, format_seconds, log, sigma_label
except ImportError:
    from baselines import fit_method
    from bcalm import BCalmLinear
    from dgp import basis
    from gps_data import load_gps
    from gps_real_utils import (
        ate_draws,
        ci_text,
        common_setup,
        fit_baseline_bcalm,
        prepare_gps,
        summarize,
        tex_num,
    )
    from utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, format_seconds, log, sigma_label


SELECTED_SIGMA0 = 0.05
SKEPTICAL_SIGMA0 = 0.6
DELTA = 0.1
SEED = 20260503
Z90 = 1.6448536269514722

SUMMARY_OUT = RESULTS_DIR / "gps_bias_susceptibility.csv"
STRATA_OUT = RESULTS_DIR / "gps_bias_susceptibility_strata.csv"
TABLE_OUT = TABLES_DIR / "gps_bias_susceptibility.tex"
FIGURE_OUT = FIGURES_DIR / "gps_bias_susceptibility.pdf"

FINITE_METHODS = ["R-OSCAR", "MR-OSCAR", "CALM", "CF"]
LINEAR_SPECS = [
    ("B-CALM", SELECTED_SIGMA0, "B-CALM ($\\sigma_0=0.05$)"),
    ("B-CALM", SKEPTICAL_SIGMA0, "B-CALM ($\\sigma_0=0.6$ diagnostic)"),
    ("B-CALM", math.inf, "B-CALM ($\\sigma_0=\\infty$ diagnostic)"),
    ("RCT-only", math.inf, "RCT-only"),
    ("pooled", 0.0, "pooled"),
]


def _method_tmp_out(method: str) -> Path:
    slug = method.lower().replace("-", "_")
    return RESULTS_DIR / f"gps_bias_susceptibility_{slug}_tmp.csv"


def _finite_seed(method: str) -> int:
    return SEED + 1000 + len(LINEAR_SPECS) + FINITE_METHODS.index(method)


def _copy_source(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in source.items()}


def _shift_ehr(ehr: dict[str, np.ndarray], delta: float, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    shifted = _copy_source(ehr)
    if mask is None:
        shifted["Y"] = shifted["Y"] + float(delta)
    else:
        shifted["Y"][np.asarray(mask, dtype=bool)] += float(delta)
    return shifted


def _theta_selector(model: BCalmLinear, X_eval: np.ndarray) -> np.ndarray:
    phi_bar = basis(X_eval, model.basis_kind).mean(axis=0)
    ell = np.zeros(4 * model.k_)
    ell[model.tau_slice] = phi_bar
    return ell


def _ehr_design(model: BCalmLinear, ehr: dict[str, np.ndarray]) -> np.ndarray:
    phi = basis(ehr["X"], model.basis_kind)
    a = ehr["A"][:, None]
    return np.column_stack([phi, a * phi, phi, a * phi])


def bcalm_ehr_weights(model: BCalmLinear, ehr: dict[str, np.ndarray], X_eval: np.ndarray) -> np.ndarray:
    """Derivative of trial-distribution ATE posterior mean wrt each EHR outcome."""
    ell = _theta_selector(model, X_eval)
    return (_ehr_design(model, ehr) @ (model.cov_ @ ell)) / (model.sigma_y**2)


def bcalm_b0_draws(model: BCalmLinear, X: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = basis(X, model.basis_kind)
    sl = slice(2 * model.k_, 3 * model.k_)
    cov = 0.5 * (model.cov_[sl, sl] + model.cov_[sl, sl].T)
    coeff = rng.multivariate_normal(model.mean_[sl], cov + 1.0e-10 * np.eye(model.k_), size=n_draws)
    return coeff @ phi.T


def _b0_coeff_draws(model: BCalmLinear, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sl = slice(2 * model.k_, 3 * model.k_)
    cov = 0.5 * (model.cov_[sl, sl] + model.cov_[sl, sl].T)
    return rng.multivariate_normal(model.mean_[sl], cov + 1.0e-10 * np.eye(model.k_), size=n_draws)


def _b0_draws_for_strata(
    model: BCalmLinear,
    X: np.ndarray,
    strata: pd.Series,
    labels: list[str],
    n_draws: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    phi = basis(X, model.basis_kind)
    coeff = _b0_coeff_draws(model, n_draws=n_draws, seed=seed)
    rows = []
    out = []
    for label in labels:
        mask = (strata.astype(str) == str(label)).to_numpy()
        phi_bar = phi[mask].mean(axis=0)
        draws = coeff @ phi_bar
        out.append(draws)
        q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95])
        rows.append(
            {
                "stratum": str(label),
                "n_ehr": int(mask.sum()),
                "b0_mean": float(np.mean(draws)),
                "b0_q05": float(q05),
                "b0_median": float(q50),
                "b0_q95": float(q95),
                "abs_b0_mean": abs(float(np.mean(draws))),
            }
        )
    return np.column_stack(out), pd.DataFrame(rows)


def _b0_summary_all_strata(model: BCalmLinear, X: np.ndarray, strata: pd.Series) -> pd.DataFrame:
    phi = basis(X, model.basis_kind)
    sl = slice(2 * model.k_, 3 * model.k_)
    mean_b0 = model.mean_[sl]
    rows = []
    for label, idx in strata.groupby(strata.astype(str), observed=True).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        phi_bar = phi[loc].mean(axis=0)
        rows.append(
            {
                "stratum": str(label),
                "n_ehr": int(loc.size),
                "b0_mean": float(phi_bar @ mean_b0),
                "abs_b0_mean": abs(float(phi_bar @ mean_b0)),
            }
        )
    return pd.DataFrame(rows)


def _ate_summary(model: object, X_eval: np.ndarray, seed: int, n_draws: int) -> dict[str, float]:
    return summarize(ate_draws(model, X_eval, seed=seed, n_draws=n_draws))


def _bcalm_mean_ate(model: BCalmLinear, X_eval: np.ndarray) -> float:
    phi_bar = basis(X_eval, model.basis_kind).mean(axis=0)
    return float(phi_bar @ model.mean_[model.tau_slice])


def _point_ate(model: object, X_eval: np.ndarray) -> float:
    if isinstance(model, BCalmLinear):
        return _bcalm_mean_ate(model, X_eval)
    if hasattr(model, "point_cate"):
        return float(np.mean(model.point_cate(X_eval)))
    if hasattr(model, "posterior_cate_moments"):
        mean, _ = model.posterior_cate_moments(X_eval)
        return float(np.mean(mean))
    return float(np.mean(model.posterior_cate(X_eval, n_samples=200, seed=SEED)))


def _uniform_fd_bcalm(
    rct: dict[str, np.ndarray],
    ehr: dict[str, np.ndarray],
    X_eval: np.ndarray,
    sigma0: float,
    seed: int,
    delta: float = DELTA,
) -> dict[str, float]:
    base = fit_baseline_bcalm(rct, ehr, sigma0, seed)
    plus = fit_baseline_bcalm(rct, _shift_ehr(ehr, delta), sigma0, seed)
    minus = fit_baseline_bcalm(rct, _shift_ehr(ehr, -delta), sigma0, seed)
    theta0 = _bcalm_mean_ate(base, X_eval)
    theta_plus = _bcalm_mean_ate(plus, X_eval)
    theta_minus = _bcalm_mean_ate(minus, X_eval)
    weights = bcalm_ehr_weights(base, ehr, X_eval)
    return {
        "theta0": theta0,
        "theta_plus": theta_plus,
        "theta_minus": theta_minus,
        "slope_plus": (theta_plus - theta0) / delta,
        "slope_minus": (theta0 - theta_minus) / delta,
        "slope_central": (theta_plus - theta_minus) / (2.0 * delta),
        "analytic_uniform_slope": float(np.sum(weights)),
        "ate_change_plus_0.1": theta_plus - theta0,
    }


def _raw_b0_strata(raw_ehr: pd.DataFrame) -> pd.Series:
    wflz_bin = pd.qcut(raw_ehr["wflz_baseline"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    return raw_ehr["site"].astype(str) + "; " + raw_ehr["insurance"].astype(str) + "; baseline " + wflz_bin.astype(str)


def _selected_strata(raw_ehr: pd.DataFrame, skeptical_b0: pd.DataFrame, top_k: int) -> pd.DataFrame:
    strata = _raw_b0_strata(raw_ehr)
    counts = strata.value_counts()
    by_n = pd.DataFrame({"stratum": counts.index.astype(str), "n_ehr": counts.to_numpy()})
    by_n["rank_n"] = np.arange(1, len(by_n) + 1)
    by_b0 = skeptical_b0.sort_values("abs_b0_mean", ascending=False).copy()
    by_b0["rank_abs_b0"] = np.arange(1, len(by_b0) + 1)
    merged = by_n.merge(by_b0[["stratum", "b0_mean", "abs_b0_mean", "rank_abs_b0"]], on="stratum", how="left")
    merged["selected_by_n"] = merged["rank_n"] <= int(top_k)
    merged["selected_by_abs_b0"] = merged["rank_abs_b0"] <= int(top_k)
    selected = merged[merged["selected_by_n"] | merged["selected_by_abs_b0"]].copy()
    selected = selected.sort_values(["selected_by_n", "rank_n", "rank_abs_b0"], ascending=[False, True, True])
    return selected.reset_index(drop=True)


def _fit_linear_method(
    method: str,
    sigma0: float,
    rct: dict[str, np.ndarray],
    ehr: dict[str, np.ndarray],
    seed: int,
) -> BCalmLinear:
    if method == "B-CALM":
        return fit_baseline_bcalm(rct, ehr, sigma0, seed)
    model, _ = fit_method(method, rct, ehr, "linear", seed)
    if not isinstance(model, BCalmLinear):
        raise TypeError(f"Expected BCalmLinear for {method}; got {type(model).__name__}")
    return model


def _linear_weights_for_method(
    method: str,
    model: BCalmLinear,
    ehr: dict[str, np.ndarray],
    X_eval: np.ndarray,
) -> np.ndarray:
    if method == "RCT-only":
        return np.zeros(ehr["Y"].shape[0], dtype=float)
    return bcalm_ehr_weights(model, ehr, X_eval)


def _nominal_lookup(n_draws: int) -> dict[tuple[str, str], dict[str, float]]:
    path = RESULTS_DIR / "gps_main_summary.csv"
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return lookup
    df = pd.read_csv(path)
    def norm_sigma(value: object) -> str:
        text = str(value)
        if text.lower() == "inf":
            return "inf"
        if text.lower() == "external":
            return "external"
        return sigma_label(float(text))
    for row in df.itertuples(index=False):
        lookup[(str(row.method), norm_sigma(row.sigma_0))] = {
            "nominal_ate": float(row.ate_mean),
            "nominal_lo": float(row.ate_lo),
            "nominal_hi": float(row.ate_hi),
            "nominal_width": float(row.ate_width),
            "nominal_source": "results/gps_main_summary.csv",
        }
    return lookup


def _nominal_for(
    method: str,
    sigma0: float | str,
    label: str,
    lookup: dict[tuple[str, str], dict[str, float]],
    rct: dict[str, np.ndarray],
    ehr: dict[str, np.ndarray],
    X_eval: np.ndarray,
    n_draws: int,
    seed: int,
    workers: int,
) -> dict[str, float | str]:
    sigma = sigma_label(sigma0)
    key = (method, sigma)
    if method in {"R-OSCAR", "MR-OSCAR", "CALM", "CF", "BCF"}:
        key = (method, "external")
    if key in lookup:
        out = dict(lookup[key])
        out["nominal_label"] = label
        return out

    fit_name = method
    if method == "B-CALM":
        model = fit_baseline_bcalm(rct, ehr, float(sigma0), seed)
    else:
        model, _ = fit_method(
            fit_name,
            rct,
            ehr,
            "linear",
            seed,
            bootstrap_b=0,
            workers=workers,
        )
    item = _ate_summary(model, X_eval, seed=seed + 17, n_draws=n_draws)
    return {
        "nominal_ate": item["ate_mean"],
        "nominal_lo": item["ate_lo"],
        "nominal_hi": item["ate_hi"],
        "nominal_width": item["ate_width"],
        "nominal_source": "computed_by_bias_susceptibility",
        "nominal_label": label,
    }


def _risk_summary(bias_draws: np.ndarray, nominal: dict[str, float | str]) -> dict[str, float]:
    q05, q50, q95 = np.quantile(bias_draws, [0.05, 0.5, 0.95])
    width = float(nominal["nominal_width"])
    return {
        "bias_risk_mean": float(np.mean(bias_draws)),
        "bias_risk_q05": float(q05),
        "bias_risk_q95": float(q95),
        "bias_risk_abs_median": float(np.median(np.abs(bias_draws))),
        "bias_risk_to_width": float((q95 - q05) / width) if width > 0 else math.nan,
        "bias_expanded_lo": float(nominal["nominal_lo"]) + float(q05),
        "bias_expanded_hi": float(nominal["nominal_hi"]) + float(q95),
    }


def _summary_row(
    method: str,
    label: str,
    sigma0: str,
    calibration: str,
    nominal: dict[str, float | str],
    uniform: dict[str, float],
    bias_draws: np.ndarray,
    risk_source: str,
    seconds: float,
    seed: int,
    n_bias_draws: int,
    n_strata: int,
) -> dict[str, float | str]:
    risk = _risk_summary(bias_draws, nominal)
    return {
        "method": method,
        "method_label": label,
        "sigma_0": sigma0,
        "b0_posterior": calibration,
        "nominal_ate": nominal["nominal_ate"],
        "nominal_lo": nominal["nominal_lo"],
        "nominal_hi": nominal["nominal_hi"],
        "nominal_ci": ci_text(float(nominal["nominal_lo"]), float(nominal["nominal_hi"])),
        "nominal_width": nominal["nominal_width"],
        "uniform_shift_ate_change_plus_0p1": uniform["ate_change_plus_0.1"],
        "uniform_shift_slope_per_1.0": uniform["slope_central"],
        "uniform_slope_plus": uniform["slope_plus"],
        "uniform_slope_minus": uniform["slope_minus"],
        "bias_risk_mean": risk["bias_risk_mean"],
        "bias_risk_q05": risk["bias_risk_q05"],
        "bias_risk_q95": risk["bias_risk_q95"],
        "bias_risk_90_range": ci_text(risk["bias_risk_q05"], risk["bias_risk_q95"]),
        "bias_risk_abs_median": risk["bias_risk_abs_median"],
        "bias_risk_to_width": risk["bias_risk_to_width"],
        "bias_expanded_lo": risk["bias_expanded_lo"],
        "bias_expanded_hi": risk["bias_expanded_hi"],
        "bias_expanded_interval": ci_text(risk["bias_expanded_lo"], risk["bias_expanded_hi"]),
        "risk_source": risk_source,
        "nominal_source": nominal["nominal_source"],
        "seconds": seconds,
        "seed": seed,
        "n_bias_draws": n_bias_draws,
        "n_strata": n_strata,
    }


def _finite_method_susceptibility(
    method: str,
    rct: dict[str, np.ndarray],
    ehr: dict[str, np.ndarray],
    X_eval: np.ndarray,
    strata: pd.Series,
    stratum_labels: list[str],
    seed: int,
    workers: int,
) -> tuple[dict[str, float], pd.DataFrame, float]:
    started = time.time()
    base_model, _ = fit_method(method, rct, ehr, "linear", seed, bootstrap_b=0, workers=workers)
    theta0 = _point_ate(base_model, X_eval)

    plus_model, _ = fit_method(method, rct, _shift_ehr(ehr, DELTA), "linear", seed, bootstrap_b=0, workers=workers)
    minus_model, _ = fit_method(method, rct, _shift_ehr(ehr, -DELTA), "linear", seed, bootstrap_b=0, workers=workers)
    theta_plus = _point_ate(plus_model, X_eval)
    theta_minus = _point_ate(minus_model, X_eval)
    uniform = {
        "theta0": theta0,
        "theta_plus": theta_plus,
        "theta_minus": theta_minus,
        "slope_plus": (theta_plus - theta0) / DELTA,
        "slope_minus": (theta0 - theta_minus) / DELTA,
        "slope_central": (theta_plus - theta_minus) / (2.0 * DELTA),
        "analytic_uniform_slope": math.nan,
        "ate_change_plus_0.1": theta_plus - theta0,
    }
    rows = []
    for idx, label in enumerate(stratum_labels, start=1):
        mask = (strata.astype(str) == str(label)).to_numpy()
        plus_s, _ = fit_method(method, rct, _shift_ehr(ehr, DELTA, mask), "linear", seed, bootstrap_b=0, workers=workers)
        minus_s, _ = fit_method(method, rct, _shift_ehr(ehr, -DELTA, mask), "linear", seed, bootstrap_b=0, workers=workers)
        theta_plus_s = _point_ate(plus_s, X_eval)
        theta_minus_s = _point_ate(minus_s, X_eval)
        rows.append(
            {
                "method": method,
                "stratum": str(label),
                "n_ehr": int(mask.sum()),
                "theta0": theta0,
                "theta_plus": theta_plus_s,
                "theta_minus": theta_minus_s,
                "slope_plus": (theta_plus_s - theta0) / DELTA,
                "slope_minus": (theta0 - theta_minus_s) / DELTA,
                "slope_central": (theta_plus_s - theta_minus_s) / (2.0 * DELTA),
                "ate_change_plus_0.1": theta_plus_s - theta0,
                "seed": seed,
                "stratum_index": idx,
            }
        )
        log(f"{method} stratum finite diff {idx}/{len(stratum_labels)}")
    return uniform, pd.DataFrame(rows), time.time() - started


def _write_smoke(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "gps_bias_susceptibility_smoke.csv", index=False)


def _short_method(label: str) -> str:
    return (
        label.replace("B-CALM ($\\sigma_0=0.05$)", "B-CALM")
        .replace("B-CALM ($\\sigma_0=0.6$ diagnostic)", "B-CALM $\\sigma_0=0.6$")
        .replace("B-CALM ($\\sigma_0=\\infty$ diagnostic)", "B-CALM $\\sigma_0=\\infty$")
    )


def _write_table(summary: pd.DataFrame) -> None:
    selected = summary[summary["b0_posterior"] == "selected_sigma0_0.05"].copy()
    skeptical = summary[summary["b0_posterior"] == "skeptical_sigma0_0.6"].copy()
    keep_order = [
        "B-CALM ($\\sigma_0=0.05$)",
        "RCT-only",
        "pooled",
        "CF",
        "R-OSCAR",
        "MR-OSCAR",
        "CALM",
        "B-CALM ($\\sigma_0=0.6$ diagnostic)",
        "B-CALM ($\\sigma_0=\\infty$ diagnostic)",
    ]
    selected["order"] = selected["method_label"].map({name: i for i, name in enumerate(keep_order)})
    selected = selected.sort_values("order")
    skeptical = skeptical.set_index("method_label")
    rows = []
    for _, row in selected.iterrows():
        sk = skeptical.loc[row["method_label"]] if row["method_label"] in skeptical.index else row
        rows.append(
            f"{_short_method(row['method_label'])} & {tex_num(row['nominal_ate'])} & "
            f"{ci_text(row['nominal_lo'], row['nominal_hi'])} & {tex_num(row['nominal_width'])} & "
            f"{tex_num(row['uniform_shift_ate_change_plus_0p1'])} & "
            f"{ci_text(row['bias_risk_q05'], row['bias_risk_q95'])} & {tex_num(row['bias_risk_to_width'], 2)} & "
            f"{ci_text(sk['bias_risk_q05'], sk['bias_risk_q95'])} & {tex_num(sk['bias_risk_to_width'], 2)} \\\\"
        )
    body = "\n".join(rows)
    TABLE_OUT.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\caption{B-CALM-calibrated hidden-bias susceptibility in the GPS external-control analysis. "
        "The uniform column is the ATE change under a $+0.1$ WFLz shift of all EHR control outcomes. "
        "Bias-risk ranges are posterior 5th--95th percentiles of $\\sum_s L_{m,s}b_{0,s}$ or row-level analogues.}\n"
        "\\label{tab:gps-bias-susceptibility}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}lcccccccc@{}}\n"
        "\\toprule\n"
        "Method & ATE & Nominal 90\\% CI & Width & $+0.1$ shift & "
        "Selected risk & Sel./width & Skeptical risk & Skep./width \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


def _write_figure(summary: pd.DataFrame) -> None:
    panel = summary[
        (summary["b0_posterior"] == "skeptical_sigma0_0.6")
        & summary["method_label"].isin(
            ["B-CALM ($\\sigma_0=0.05$)", "RCT-only", "pooled", "CF", "R-OSCAR", "MR-OSCAR", "CALM"]
        )
    ].copy()
    order = ["RCT-only", "B-CALM ($\\sigma_0=0.05$)", "pooled", "CF", "R-OSCAR", "MR-OSCAR", "CALM"]
    panel["order"] = panel["method_label"].map({name: i for i, name in enumerate(order)})
    panel = panel.sort_values("order", ascending=False)
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    y = np.arange(len(panel))
    ax.hlines(y, panel["bias_expanded_lo"], panel["bias_expanded_hi"], color="#8b4a62", linewidth=3.0, alpha=0.55, label="skeptical hidden-bias interval")
    ax.hlines(y, panel["nominal_lo"], panel["nominal_hi"], color="#1f6f8b", linewidth=1.8, label="nominal 90% interval")
    ax.scatter(panel["nominal_ate"], y, color="#1f6f8b", s=20, zorder=3)
    ax.axvline(0.0, color="0.25", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_method(x).replace("$\\sigma_0=0.6$", "0.6").replace("$\\sigma_0=\\infty$", "inf") for x in panel["method_label"]])
    ax.set_xlabel("ATE")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--n-draws", type=int, default=800)
    parser.add_argument("--bias-draws", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--finite-methods", nargs="*", default=FINITE_METHODS)
    parser.add_argument(
        "--append-finite-only",
        action="store_true",
        help="Load existing outputs and recompute only --finite-methods rows before regenerating artifacts.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    common_setup()
    started = time.time()
    loaded = load_gps()
    rct, ehr, X_rct = prepare_gps(loaded)
    log(f"GPS bias-susceptibility data: n_R={len(rct['Y'])}, n_O={len(ehr['Y'])}")

    selected_model = fit_baseline_bcalm(rct, ehr, SELECTED_SIGMA0, SEED)
    selected_summary = _ate_summary(selected_model, X_rct, seed=SEED + 1, n_draws=args.n_draws)
    fd = _uniform_fd_bcalm(rct, ehr, X_rct, SELECTED_SIGMA0, SEED)
    smoke = {
        "method": "B-CALM",
        "sigma_0": SELECTED_SIGMA0,
        "nominal_ate": selected_summary["ate_mean"],
        "nominal_lo": selected_summary["ate_lo"],
        "nominal_hi": selected_summary["ate_hi"],
        "nominal_width": selected_summary["ate_width"],
        **fd,
        "seconds": time.time() - started,
    }
    _write_smoke([smoke])
    log(
        "smoke B-CALM sigma_0=0.05: "
        f"ATE={selected_summary['ate_mean']:.3f} {ci_text(selected_summary['ate_lo'], selected_summary['ate_hi'])}, "
        f"+0.1 change={fd['ate_change_plus_0.1']:.3f}, "
        f"analytic slope={fd['analytic_uniform_slope']:.3f}, fd slope={fd['slope_central']:.3f}"
    )
    if args.smoke_only:
        return

    raw_ehr = loaded["meta"]["raw_D_O"]
    strata = _raw_b0_strata(raw_ehr)
    skeptical_model = fit_baseline_bcalm(rct, ehr, SKEPTICAL_SIGMA0, SEED + 2)
    skeptical_all = _b0_summary_all_strata(skeptical_model, ehr["X"], strata)
    strata_pick = _selected_strata(raw_ehr, skeptical_all, top_k=args.top_k)
    stratum_labels = strata_pick["stratum"].astype(str).tolist()
    log(f"selected {len(stratum_labels)} strata for finite differences")

    b0_row_draws = {
        "selected_sigma0_0.05": bcalm_b0_draws(selected_model, ehr["X"], args.bias_draws, SEED + 101),
        "skeptical_sigma0_0.6": bcalm_b0_draws(skeptical_model, ehr["X"], args.bias_draws, SEED + 202),
    }
    b0_stratum_draws = {}
    b0_stratum_summaries = {}
    for cal, model, seed in [
        ("selected_sigma0_0.05", selected_model, SEED + 303),
        ("skeptical_sigma0_0.6", skeptical_model, SEED + 404),
    ]:
        draws, b0sum = _b0_draws_for_strata(model, ehr["X"], strata, stratum_labels, args.bias_draws, seed)
        b0_stratum_draws[cal] = draws
        b0_stratum_summaries[cal] = b0sum

    nominal = _nominal_lookup(args.n_draws)
    finite_method_set = set(args.finite_methods)
    if args.append_finite_only and not SUMMARY_OUT.exists():
        raise FileNotFoundError(f"--append-finite-only requires existing {SUMMARY_OUT}")
    if args.append_finite_only and not STRATA_OUT.exists():
        raise FileNotFoundError(f"--append-finite-only requires existing {STRATA_OUT}")

    summary_rows: list[dict] = []
    strata_rows: list[dict] = []
    method_seconds: dict[str, float] = {}
    method_seeds: dict[str, int] = {}

    if args.append_finite_only:
        existing_summary = pd.read_csv(SUMMARY_OUT)
        existing_strata = pd.read_csv(STRATA_OUT)
        keep_summary = existing_summary[~existing_summary["method"].isin(finite_method_set)].copy()
        keep_strata = existing_strata[~existing_strata["method"].isin(finite_method_set)].copy()
        summary_rows = keep_summary.to_dict("records")
        strata_rows = keep_strata.to_dict("records")
        for row in keep_summary.drop_duplicates("method_label").itertuples(index=False):
            method_seconds[str(row.method_label)] = float(row.seconds)
            method_seeds[str(row.method_label)] = int(row.seed)
        log(
            "append finite-only mode: kept "
            f"{len(summary_rows)} summary rows and {len(strata_rows)} stratum rows before recomputing "
            f"{sorted(finite_method_set)}"
        )

    selection_meta = strata_pick[["stratum", "selected_by_n", "selected_by_abs_b0", "rank_n", "rank_abs_b0"]].copy()
    selected_b0_cols = b0_stratum_summaries["selected_sigma0_0.05"].rename(
        columns={
            "b0_mean": "selected_b0_mean",
            "b0_q05": "selected_b0_q05",
            "b0_q95": "selected_b0_q95",
            "abs_b0_mean": "selected_abs_b0_mean",
        }
    )
    skeptical_b0_cols = b0_stratum_summaries["skeptical_sigma0_0.6"].rename(
        columns={
            "b0_mean": "skeptical_b0_mean",
            "b0_q05": "skeptical_b0_q05",
            "b0_q95": "skeptical_b0_q95",
            "abs_b0_mean": "skeptical_abs_b0_mean",
        }
    )
    b0_meta = (
        selection_meta.merge(
            selected_b0_cols[
                ["stratum", "selected_b0_mean", "selected_b0_q05", "selected_b0_q95", "selected_abs_b0_mean"]
            ],
            on="stratum",
            how="left",
        )
        .merge(
            skeptical_b0_cols[
                ["stratum", "skeptical_b0_mean", "skeptical_b0_q05", "skeptical_b0_q95", "skeptical_abs_b0_mean"]
            ],
            on="stratum",
            how="left",
        )
    )

    if not args.append_finite_only:
        for method, sigma0, label in LINEAR_SPECS:
            method_key = f"{label}"
            seed = SEED + len(method_seeds) + 10
            method_seeds[method_key] = seed
            m_start = time.time()
            model = _fit_linear_method(method, sigma0, rct, ehr, seed)
            weights = _linear_weights_for_method(method, model, ehr, X_rct)
            theta0 = _point_ate(model, X_rct)
            if method == "RCT-only":
                uniform = {
                    "theta0": theta0,
                    "theta_plus": theta0,
                    "theta_minus": theta0,
                    "slope_plus": 0.0,
                    "slope_minus": 0.0,
                    "slope_central": 0.0,
                    "analytic_uniform_slope": 0.0,
                    "ate_change_plus_0.1": 0.0,
                }
            elif method == "B-CALM":
                uniform = _uniform_fd_bcalm(rct, ehr, X_rct, sigma0, seed)
            else:
                plus_model = _fit_linear_method(method, sigma0, rct, _shift_ehr(ehr, DELTA), seed)
                minus_model = _fit_linear_method(method, sigma0, rct, _shift_ehr(ehr, -DELTA), seed)
                theta_plus = _point_ate(plus_model, X_rct)
                theta_minus = _point_ate(minus_model, X_rct)
                uniform = {
                    "theta0": theta0,
                    "theta_plus": theta_plus,
                    "theta_minus": theta_minus,
                    "slope_plus": (theta_plus - theta0) / DELTA,
                    "slope_minus": (theta0 - theta_minus) / DELTA,
                    "slope_central": (theta_plus - theta_minus) / (2.0 * DELTA),
                    "analytic_uniform_slope": float(np.sum(weights)),
                    "ate_change_plus_0.1": theta_plus - theta0,
                }
            elapsed = time.time() - m_start
            method_seconds[method_key] = elapsed

            nom = _nominal_for(method, sigma0, label, nominal, rct, ehr, X_rct, args.n_draws, seed, args.workers)
            for cal, row_draws in b0_row_draws.items():
                bias_draws = row_draws @ weights
                summary_rows.append(
                    _summary_row(
                        method=method,
                        label=label,
                        sigma0=sigma_label(sigma0),
                        calibration=cal,
                        nominal=nom,
                        uniform=uniform,
                        bias_draws=bias_draws,
                        risk_source="row_level_analytic_influence",
                        seconds=elapsed,
                        seed=seed,
                        n_bias_draws=args.bias_draws,
                        n_strata=len(stratum_labels),
                    )
                )

            lin_rows = []
            for idx, label_s in enumerate(stratum_labels, start=1):
                mask = (strata.astype(str) == str(label_s)).to_numpy()
                slope = float(np.sum(weights[mask]))
                lin_rows.append(
                    {
                        "method": method,
                        "method_label": label,
                        "sigma_0": sigma_label(sigma0),
                        "stratum": label_s,
                        "n_ehr": int(mask.sum()),
                        "theta0": theta0,
                        "theta_plus": theta0 + DELTA * slope,
                        "theta_minus": theta0 - DELTA * slope,
                        "slope_plus": slope,
                        "slope_minus": slope,
                        "slope_central": slope,
                        "ate_change_plus_0p1": DELTA * slope,
                        "seed": seed,
                        "stratum_index": idx,
                        "risk_source": "row_level_analytic_influence",
                    }
                )
            strata_rows.extend(pd.DataFrame(lin_rows).merge(b0_meta, on="stratum", how="left").to_dict("records"))
            log(f"completed analytic susceptibility for {label}")

    for method in args.finite_methods:
        if method == "BCF":
            log("skipping BCF finite differences: each BCF refit is too slow for the stratum grid")
            continue
        if method not in FINITE_METHODS:
            raise ValueError(f"Unknown finite method: {method}")
        seed = _finite_seed(method)
        method_seeds[method] = seed
        log(f"starting finite-difference susceptibility for {method}")
        uniform, method_strata, elapsed = _finite_method_susceptibility(
            method, rct, ehr, X_rct, strata, stratum_labels, seed=seed, workers=args.workers
        )
        method_seconds[method] = elapsed
        nom = _nominal_for(method, "external", method, nominal, rct, ehr, X_rct, args.n_draws, seed, args.workers)
        slopes = method_strata.set_index("stratum").loc[stratum_labels, "slope_central"].to_numpy(dtype=float)
        method_summary_rows = []
        for cal, draws in b0_stratum_draws.items():
            bias_draws = draws @ slopes
            row = _summary_row(
                method=method,
                label=method,
                sigma0="external",
                calibration=cal,
                nominal=nom,
                uniform=uniform,
                bias_draws=bias_draws,
                risk_source="top_strata_finite_difference",
                seconds=elapsed,
                seed=seed,
                n_bias_draws=args.bias_draws,
                n_strata=len(stratum_labels),
            )
            summary_rows.append(row)
            method_summary_rows.append(row)
        pd.DataFrame(method_summary_rows).to_csv(_method_tmp_out(method), index=False)
        if method == "CF":
            log(f"wrote CF temp summary to {_method_tmp_out(method)}")
        tmp = method_strata.copy()
        tmp["method_label"] = method
        tmp["sigma_0"] = "external"
        tmp["ate_change_plus_0p1"] = tmp["ate_change_plus_0.1"]
        tmp = tmp.drop(columns=["ate_change_plus_0.1"])
        tmp["risk_source"] = "top_strata_finite_difference"
        strata_rows.extend(tmp.merge(b0_meta, on="stratum", how="left").to_dict("records"))

        pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
        pd.DataFrame(strata_rows).to_csv(STRATA_OUT, index=False)
        log(f"completed finite-difference susceptibility for {method} in {format_seconds(elapsed)}")

    summary = pd.DataFrame(summary_rows)
    strata_df = pd.DataFrame(strata_rows)
    summary.to_csv(SUMMARY_OUT, index=False)
    strata_df.to_csv(STRATA_OUT, index=False)
    _write_table(summary)
    _write_figure(summary)
    elapsed_all = time.time() - started
    log(f"gps_bias_susceptibility finished in {format_seconds(elapsed_all)}")


if __name__ == "__main__":
    main()
