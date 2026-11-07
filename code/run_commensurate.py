#!/usr/bin/env python3
"""Scalar commensurate-prior and robust-MAP baselines for B-CALM.

All model and metric definitions are imported from the B-CALM modules in this
directory. The scalar variants differ only by pinning non-intercept bias coefficients to
the same 1e-10 prior variance used by the source model for a zero prior scale.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.special import logsumexp


from utils import RESULTS_DIR, log

from baselines import (
    SIGMA_GRID,
    SIGMA_MODEL_PRIOR,
    _contrast_log_marginal,
    fit_bcalm_selected,
)
from bcalm import BCalmLinear, BCalmMixture
from dgp import generate_data
from metrics import negative_transfer, posterior_metrics
from utils import seed_list


RAW_COLUMNS = [
    "bias_shape",
    "bias_strength",
    "method",
    "seed",
    "cate_rmse",
    "coverage_90",
    "ci_width",
    "neg_transfer",
]
KEY_COLUMNS = ["bias_shape", "bias_strength", "method"]
METHOD_ORDER = [
    "B-CALM (function-valued)",
    "Commensurate (scalar b_delta)",
    "Commensurate (scalar b_delta + scalar b_0)",
    "Robust MAP (mixture on scalar b_delta)",
    "RCT-only",
    "Pooled",
]
CELLS = [
    ("heterogeneous", 0.2, False, "comparative"),
    ("heterogeneous", 0.4, False, "comparative"),
    ("constant", 0.2, True, "comparative"),
    ("constant", 0.4, True, "comparative"),
    ("n/a", 0.0, False, "none"),
]


def dgp_seed(loop_seed: int, bias_strength: float) -> int:
    """Match the seed derivation in the paper's run_main.py driver."""
    n_R = 500
    return 20000 + 1000 * loop_seed + n_R + int(round(100 * bias_strength))


class ScalarDeltaBCalm(BCalmLinear):
    """BCalmLinear with only the intercept of b_delta left unpinned."""

    def _system(self, D_R, D_O):
        X_design, y, prior_var, prior_mean, k = super()._system(D_R, D_O)
        prior_var[3 * k + 1 : 4 * k] = 1.0e-10
        return X_design, y, prior_var, prior_mean, k


class ScalarDeltaScalarB0BCalm(BCalmLinear):
    """BCalmLinear with only the intercepts of b_0 and b_delta unpinned."""

    def _system(self, D_R, D_O):
        X_design, y, prior_var, prior_mean, k = super()._system(D_R, D_O)
        prior_var[2 * k + 1 : 3 * k] = 1.0e-10
        prior_var[3 * k + 1 : 4 * k] = 1.0e-10
        return X_design, y, prior_var, prior_mean, k


def fit_bma_for_class(
    model_class: Type[BCalmLinear],
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    basis_kind: str,
    sigma_y: float = 1.0,
) -> BCalmMixture:
    """Apply the paper's exact BMA weighting rule to a candidate model class."""
    rct_contrast = BCalmLinear(
        sigma_delta=math.inf,
        sigma0=math.inf,
        sigma_y=sigma_y,
        basis_kind=basis_kind,
    ).fit(D_R, None)
    os_contrast = BCalmLinear(
        sigma_delta=0.0,
        sigma0=0.6,
        sigma_y=sigma_y,
        basis_kind=basis_kind,
    ).fit(None, D_O)
    tau_r = rct_contrast.mean_[rct_contrast.tau_slice]
    cov_r = rct_contrast.cov_[rct_contrast.tau_slice, rct_contrast.tau_slice]
    tau_o = os_contrast.mean_[os_contrast.tau_slice]
    cov_o = os_contrast.cov_[os_contrast.tau_slice, os_contrast.tau_slice]

    components: list[BCalmLinear] = []
    logps: list[float] = []
    for sigma in SIGMA_GRID:
        component = model_class(
            sigma_delta=sigma,
            sigma0=0.6,
            sigma_y=sigma_y,
            basis_kind=basis_kind,
        ).fit(D_R, D_O)
        components.append(component)
        logps.append(_contrast_log_marginal(tau_r, cov_r, tau_o, cov_o, sigma))

    logps_arr = np.asarray(logps, dtype=float) + np.log(SIGMA_MODEL_PRIOR)
    weights = np.exp(logps_arr - logsumexp(logps_arr))
    mixture = BCalmMixture(components, weights)
    mixture.sigma_grid_ = tuple(SIGMA_GRID)
    mixture.log_marginal_likelihoods_ = tuple(float(x) for x in logps_arr)
    mixture.weight_summary_ = {
        sigma: float(weight) for sigma, weight in zip(SIGMA_GRID, weights)
    }
    return mixture


def fit_robust_map(
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    basis_kind: str,
) -> BCalmMixture:
    """Two-component robust MAP prior with data-updated mixture weights."""
    components = [
        ScalarDeltaBCalm(
            sigma_delta=sigma,
            sigma0=0.6,
            sigma_y=1.0,
            basis_kind=basis_kind,
        ).fit(D_R, D_O)
        for sigma in (0.05, 10.0)
    ]
    log_weights = np.asarray(
        [
            math.log(0.5) + component.log_marginal_likelihood_
            for component in components
        ],
        dtype=float,
    )
    weights = np.exp(log_weights - logsumexp(log_weights))
    return BCalmMixture(components, weights)


def evaluate_model(
    model: object,
    X_eval: np.ndarray,
    tau_eval: np.ndarray,
    rct_rmse: float,
) -> dict[str, float | int]:
    mean, var = model.posterior_cate_moments(X_eval)
    metrics = posterior_metrics(mean, var, tau_eval)
    return {
        "cate_rmse": metrics["cate_rmse"],
        "coverage_90": metrics["coverage_90"],
        "ci_width": metrics["ci_width"],
        "neg_transfer": negative_transfer(metrics["cate_rmse"], rct_rmse),
    }


def run_cell_seed(
    bias_shape: str,
    bias_strength: float,
    constant_comparative: bool,
    bias: str,
    seed: int,
) -> list[dict[str, object]]:
    data = generate_data(
        n_R=500,
        n_O=5000,
        seed=dgp_seed(seed, bias_strength),
        regime="balanced",
        bias=bias,
        bias_strength=bias_strength,
        kind="mildly_nonlinear",
        n_eval=600,
        constant_comparative=constant_comparative,
    )
    basis_kind = "mild"

    # Fit RCT-only first so every method's negative-transfer indicator uses
    # exactly the same seed-specific reference.
    rct_model = BCalmLinear(
        sigma_delta=math.inf,
        sigma0=math.inf,
        sigma_y=1.0,
        basis_kind=basis_kind,
    ).fit(data.D_R, None)
    rct_mean, rct_var = rct_model.posterior_cate_moments(data.X_eval)
    rct_rmse = posterior_metrics(rct_mean, rct_var, data.tau_eval)["cate_rmse"]

    bcalm_model, selection = fit_bcalm_selected(
        data.D_R, data.D_O, basis_kind, seed, sigma_y=1.0
    )
    if selection != "bma":
        raise RuntimeError(f"Unexpected B-CALM selector result: {selection}")

    models = [
        ("B-CALM (function-valued)", bcalm_model),
        (
            "Commensurate (scalar b_delta)",
            fit_bma_for_class(ScalarDeltaBCalm, data.D_R, data.D_O, basis_kind),
        ),
        (
            "Commensurate (scalar b_delta + scalar b_0)",
            fit_bma_for_class(
                ScalarDeltaScalarB0BCalm,
                data.D_R,
                data.D_O,
                basis_kind,
            ),
        ),
        (
            "Robust MAP (mixture on scalar b_delta)",
            fit_robust_map(data.D_R, data.D_O, basis_kind),
        ),
        ("RCT-only", rct_model),
        (
            "Pooled",
            BCalmLinear(
                sigma_delta=0.0,
                sigma0=0.0,
                sigma_y=1.0,
                basis_kind=basis_kind,
            ).fit(data.D_R, data.D_O),
        ),
    ]

    rows = []
    for method, model in models:
        row = {
            "bias_shape": bias_shape,
            "bias_strength": bias_strength,
            "method": method,
            "seed": seed,
        }
        row.update(evaluate_model(model, data.X_eval, data.tau_eval, rct_rmse))
        rows.append(row)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["cate_rmse", "coverage_90", "ci_width", "neg_transfer"]
    for keys, group in raw.groupby(KEY_COLUMNS, sort=False, dropna=False):
        row = dict(zip(KEY_COLUMNS, keys))
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_se"] = values.std(ddof=1) / math.sqrt(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_summary(summary: pd.DataFrame) -> str:
    lines = [
        "# Scalar commensurate-prior baselines",
        "",
        "Means over 50 seeds; intervals are 90% posterior intervals.",
        "",
    ]
    method_rank = {method: i for i, method in enumerate(METHOD_ORDER)}
    for bias_shape, strength, _, _ in CELLS:
        arm = summary[
            (summary["bias_shape"] == bias_shape)
            & np.isclose(summary["bias_strength"], strength)
        ].copy()
        arm["_rank"] = arm["method"].map(method_rank)
        arm = arm.sort_values("_rank")
        label = (
            "No comparative bias"
            if bias_shape == "n/a"
            else f"{bias_shape.capitalize()} comparative bias, strength {strength:g}"
        )
        lines.extend(
            [
                f"## {label}",
                "",
                "| Method | Coverage | Width | RMSE | Neg. transfer |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in arm.iterrows():
            lines.append(
                f"| {row['method']} | {row['coverage_90_mean']:.3f} | "
                f"{row['ci_width_mean']:.3f} | {row['cate_rmse_mean']:.3f} | "
                f"{row['neg_transfer_mean']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run one heterogeneous seed and assert six finite result rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR = args.out_dir
    log(
        "B-CALM: frozen six-scale contrast BMA; scalar models: same grid; robust MAP: scalar scales 0.05/10, prior 0.5/0.5, joint marginal likelihood; NT: seed-specific Gaussian RCT-only"
    )
    start = time.time()
    if args.validate_only:
        rows = run_cell_seed("heterogeneous", 0.2, False, "comparative", 0)
        frame = pd.DataFrame(rows)
        if (
            len(frame) != 6
            or not np.isfinite(
                frame[["cate_rmse", "coverage_90", "ci_width", "neg_transfer"]]
            )
            .all()
            .all()
        ):
            raise RuntimeError("Validation did not produce six finite method rows.")
        print(frame[RAW_COLUMNS].to_string(index=False))
        print(f"validation_seconds={time.time() - start:.3f}")
        return

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (shape, strength, constant, bias, seed)
        for shape, strength, constant, bias in CELLS
        for seed in seed_list(50)
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        nested_rows = Parallel(n_jobs=args.workers, verbose=10)(
            delayed(run_cell_seed)(*task) for task in tasks
        )
    raw = pd.DataFrame(
        [row for task_rows in nested_rows for row in task_rows],
        columns=RAW_COLUMNS,
    )
    raw["method"] = pd.Categorical(raw["method"], categories=METHOD_ORDER, ordered=True)
    shape_order = {"heterogeneous": 0, "constant": 1, "n/a": 2}
    raw["_shape_order"] = raw["bias_shape"].map(shape_order)
    raw = (
        raw.sort_values(
            ["_shape_order", "bias_strength", "method", "seed"],
            kind="stable",
        )
        .drop(columns="_shape_order")
        .reset_index(drop=True)
    )
    raw["method"] = raw["method"].astype("object")
    expected_rows = len(CELLS) * len(seed_list(50)) * len(METHOD_ORDER)
    if len(raw) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, produced {len(raw)}.")
    if raw[RAW_COLUMNS[4:]].isna().any().any():
        raise RuntimeError("Non-finite metric detected in raw results.")

    summary = summarize(raw)
    expected_summary_rows = len(CELLS) * len(METHOD_ORDER)
    if len(summary) != expected_summary_rows:
        raise RuntimeError(
            f"Expected {expected_summary_rows} summary rows, produced {len(summary)}."
        )

    raw_path = TABLE_DIR / "commensurate_baseline.csv"
    summary_path = TABLE_DIR / "commensurate_baseline_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    elapsed = time.time() - start
    print(f"wrote {raw_path} ({len(raw)} rows)")
    print(f"wrote {summary_path} ({len(summary)} rows)")
    print(f"wall_clock_seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
