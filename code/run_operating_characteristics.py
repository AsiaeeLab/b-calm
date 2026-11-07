#!/usr/bin/env python3
"""Go/No-Go operating characteristics for B-CALM and comparators."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import norm


from baselines import SIGMA_GRID, SIGMA_MODEL_PRIOR, fit_bcalm_selected
from bcalm import BCalmLinear, BCalmMixture
from dgp import basis, generate_data, true_tau
from utils import seed_list, RESULTS_DIR, TABLES_DIR, log, sigma_label
from run_main import basis_for_dgp, kind_for_dgp
from experiment_io import compare_frames, write_table


N_R = 500
CALIBRATION_N_R = 200000
CALIBRATION_SEED = 1000000007
RAW_COLUMNS = [
    "delta",
    "bias_regime",
    "n_O",
    "method",
    "seed",
    "post_prob_positive",
    "go",
]
SUMMARY_COLUMNS = [
    "delta",
    "bias_regime",
    "n_O",
    "method",
    "quantity",
    "go_rate",
    "go_rate_se",
    "n_seeds",
]
METHOD_ORDER = [
    "B-CALM (function-valued)",
    "Pooled",
    "RCT-only",
]
BIAS_CELLS = [
    ("none", "none", 0.0),
    ("baseline-only", "baseline-only", 0.0),
    ("comparative@0.2", "comparative", 0.2),
    ("comparative@0.5", "comparative", 0.5),
]


def dgp_seed(
    loop_seed: int,
    delta: float,
    n_O: int,
    bias_strength: float,
) -> int:
    """Extend the paper convention with collision-free cell identifiers."""
    return (
        20000
        + 1000000 * loop_seed
        + 10 * n_O
        + 1000 * int(round(100 * delta))
        + N_R
        + int(round(100 * bias_strength))
    )


def estimate_tau_bar(regime: str = "balanced") -> float:
    """Estimate a regime-specific trial-population ATE before the grid."""
    calibration = generate_data(
        n_R=CALIBRATION_N_R,
        n_O=1,
        seed=CALIBRATION_SEED,
        regime=regime,
        bias="none",
        bias_strength=0.0,
        kind=kind_for_dgp(regime),
        n_eval=1,
    )
    tau = true_tau(calibration.D_R["X"], kind=kind_for_dgp(regime))
    estimate = float(np.mean(tau))
    if not np.isfinite(estimate):
        raise RuntimeError("Non-finite tau_bar calibration estimate.")
    return estimate


def shifted_copy(data: dict[str, np.ndarray], shift: float) -> dict[str, np.ndarray]:
    return {
        "X": np.asarray(data["X"]).copy(),
        "A": np.asarray(data["A"]).copy(),
        "Y": np.asarray(data["Y"]) - shift * np.asarray(data["A"]),
    }


def linear_functional_moments(
    model: BCalmLinear, X_eval: np.ndarray
) -> tuple[float, float]:
    phi_bar = np.mean(basis(X_eval, model.basis_kind), axis=0)
    tau_mean = model.mean_[model.tau_slice]
    tau_cov = model.cov_[model.tau_slice, model.tau_slice]
    mean = float(phi_bar @ tau_mean)
    var = float(phi_bar @ tau_cov @ phi_bar)
    return mean, max(var, 1.0e-12)


def posterior_probability_positive(model: object, X_eval: np.ndarray) -> float:
    """Exact probability for linear models and their finite BMA mixtures."""
    if isinstance(model, BCalmMixture):
        probability = 0.0
        for weight, component in zip(model.weights, model.components):
            mean, var = linear_functional_moments(component, X_eval)
            probability += float(weight) * float(norm.cdf(mean / math.sqrt(var)))
        return float(np.clip(probability, 0.0, 1.0))
    if isinstance(model, BCalmLinear):
        mean, var = linear_functional_moments(model, X_eval)
        return float(np.clip(norm.cdf(mean / math.sqrt(var)), 0.0, 1.0))

    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def run_config(
    delta: float,
    bias_regime: str,
    bias: str,
    bias_strength: float,
    n_O: int,
    seed: int,
    tau_bar: float,
    regime: str = "balanced",
) -> dict[str, object]:
    data = generate_data(
        n_R=N_R,
        n_O=n_O,
        seed=dgp_seed(seed, delta, n_O, bias_strength),
        regime=regime,
        bias=bias,
        bias_strength=bias_strength,
        kind=kind_for_dgp(regime),
        n_eval=600,
    )
    shift = tau_bar - delta
    D_R = shifted_copy(data.D_R, shift)
    D_O = shifted_copy(data.D_O, shift)
    shifted_truth = data.tau_eval - shift
    shifted_truth_mean = float(np.mean(shifted_truth))

    basis_kind = basis_for_dgp(regime)
    bcalm_model, selection = fit_bcalm_selected(D_R, D_O, basis_kind, seed, sigma_y=1.0)
    if selection != "bma":
        raise RuntimeError(f"Unexpected B-CALM selector result: {selection}")
    pooled_model = BCalmLinear(
        sigma_delta=0.0,
        sigma0=0.0,
        sigma_y=1.0,
        basis_kind=basis_kind,
    ).fit(D_R, D_O)
    rct_model = BCalmLinear(
        sigma_delta=math.inf,
        sigma0=math.inf,
        sigma_y=1.0,
        basis_kind=basis_kind,
    ).fit(D_R, None)
    models = [
        ("B-CALM (function-valued)", bcalm_model),
        ("Pooled", pooled_model),
        ("RCT-only", rct_model),
    ]
    models.extend(
        (f"B-CALM sigma={sigma_label(sigma)}", model)
        for sigma, model in zip(SIGMA_GRID, bcalm_model.components)
    )
    rows = []
    for method, model in models:
        probability = posterior_probability_positive(model, data.X_eval)
        rows.append(
            {
                "dgp": regime,
                "truth_mean": shifted_truth_mean,
                "tau_bar": tau_bar,
                "delta": delta,
                "bias_regime": bias_regime,
                "n_O": n_O,
                "method": method,
                "seed": seed,
                "post_prob_positive": probability,
                "go": int(probability >= 0.975),
            }
        )
    return {
        "rows": rows,
        "delta": delta,
        "bias_regime": bias_regime,
        "n_O": n_O,
        "seed": seed,
        "shifted_truth_mean": shifted_truth_mean,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["delta", "bias_regime", "n_O", "method"]
    summary = (
        raw.groupby(key_columns, sort=False, dropna=False)["go"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "go_rate", "count": "n_seeds"})
    )
    summary["go_rate_se"] = np.sqrt(
        summary["go_rate"] * (1.0 - summary["go_rate"]) / summary["n_seeds"]
    )
    summary["quantity"] = np.where(
        np.isclose(summary["delta"], 0.0), "type_I_error", "power"
    )
    return summary[SUMMARY_COLUMNS]


def tables(summary: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for regime in ["shared-only", "balanced", "mismatch-heavy"]:
        rows.append(r"\multicolumn{5}{l}{" + regime + "}")
        subset = summary[summary.dgp.eq(regime)]
        for (bias, no), group in subset.groupby(["bias_regime", "n_O"], sort=False):
            cells = []
            for method in METHOD_ORDER:
                g = group[group.method.eq(method)].set_index("delta")
                cells.append(f"{g.loc[0.0, 'go_rate']:.3f}/{g.loc[0.2, 'go_rate']:.3f}")
            rows.append(f"{bias} & {no // 1000}k & " + " & ".join(cells))
    write_table(
        out_dir / "oc_main.tex",
        "Go/No-Go operating characteristics over 200 seeds. Entries are Type I error/power; Go requires posterior probability of a positive trial-population ATE at least 0.975.",
        "tab:oc-main",
        "llrrr",
        "Bias & $n_O$ & B-CALM EB & Pooled & RCT-only",
        rows,
    )
    rows = []
    subset = summary[summary.dgp.eq("balanced")]
    for bias, group in subset.groupby("bias_regime", sort=False):
        for sigma in SIGMA_GRID:
            g = group[group.method.eq(f"B-CALM sigma={sigma_label(sigma)}")].set_index(
                ["n_O", "delta"]
            )
            cells = [
                f"{g.loc[(no, delta), 'go_rate']:.3f}"
                for no in [5000, 20000]
                for delta in [0.0, 0.2]
            ]
            rows.append(f"{bias} & {sigma_label(sigma)} & " + " & ".join(cells))
    write_table(
        out_dir / "oc_fixed_sigma.tex",
        "Fixed discrepancy scales in the balanced regime, 200 seeds. Each sample-size block gives Type I error and power. Component posteriors are reused from the empirical-Bayes mixture.",
        "tab:oc-fixed-sigma",
        "llrrrr",
        r"Bias & $\sigma_\Delta$ & I (5k) & Power (5k) & I (20k) & Power (20k)",
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.tables_only:
        tables(
            pd.read_csv(args.out_dir / "operating_characteristics_summary.csv"),
            args.table_dir,
        )
        return
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"branch=contrast BMA; prior={SIGMA_MODEL_PRIOR}; sigma_grid={SIGMA_GRID}; fixed arms reuse mixture components"
    )
    calibration = {
        r: estimate_tau_bar(r) for r in ["shared-only", "balanced", "mismatch-heavy"]
    }
    log(
        f"Population calibration n={CALIBRATION_N_R}, seed={CALIBRATION_SEED}: {calibration}"
    )

    def run(tasks: list) -> pd.DataFrame:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=args.workers, verbose=10)(
                delayed(run_config)(*t) for t in tasks
            )
        return pd.DataFrame([row for result in results for row in result["rows"]])

    def tasks(regimes: list[str], biases: list) -> list:
        return [
            (delta, label, bias, strength, no, seed, calibration[r], r)
            for r in regimes
            for delta in [0.0, 0.2]
            for label, bias, strength in biases
            for no in [5000, 20000]
            for seed in range(200)
        ]

    original_biases = [cell for cell in BIAS_CELLS if cell[0] != "baseline-only"]
    first = run(tasks(["balanced"], original_biases))
    if args.reference:
        old = pd.read_csv(
            args.reference, keep_default_na=False, float_precision="round_trip"
        )
        reproduced = summarize(first[first.method.isin(METHOD_ORDER)])
        check = compare_frames(
            reproduced, old, ["delta", "bias_regime", "n_O", "method"]
        )
        if not check["exact"]:
            first.to_csv(
                args.out_dir / "operating_characteristics_check.csv", index=False
            )
            raise AssertionError(
                "Balanced operating characteristics did not reproduce; extension stopped."
            )
    rest = run(
        tasks(["shared-only", "mismatch-heavy"], BIAS_CELLS)
        + tasks(["balanced"], [BIAS_CELLS[1]])
    )
    raw = pd.concat([first, rest], ignore_index=True)
    keys = ["dgp", "delta", "bias_regime", "n_O", "method"]
    summary = (
        raw.groupby(keys, sort=False)
        .agg(
            go_rate=("go", "mean"),
            n_seeds=("go", "size"),
            realized_mean_truth=("truth_mean", "mean"),
            tau_bar=("tau_bar", "first"),
        )
        .reset_index()
    )
    summary["go_rate_se"] = np.sqrt(
        summary.go_rate * (1 - summary.go_rate) / summary.n_seeds
    )
    summary["quantity"] = np.where(summary.delta.eq(0), "type_I_error", "power")
    raw.to_csv(args.out_dir / "operating_characteristics_raw.csv", index=False)
    summary.to_csv(args.out_dir / "operating_characteristics_summary.csv", index=False)
    truth = (
        raw.drop_duplicates(["dgp", "delta", "bias_regime", "n_O", "seed"])
        .groupby(["dgp", "delta"])
        .truth_mean.mean()
    )
    log("Realized trial-evaluation truth:\n" + truth.to_string())
    if (summary.realized_mean_truth - summary.delta).abs().max() >= 0.005:
        raise AssertionError("Population-centering check failed")
    tables(summary, args.table_dir)
    log(f"wall_clock_seconds={time.time() - start:.6f}")


if __name__ == "__main__":
    main()
