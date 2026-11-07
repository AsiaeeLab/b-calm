#!/usr/bin/env python3
"""Benchmark the official mixed QR-learner on the scalar-baseline cells.

The benchmark uses the two balanced, comparative-bias-strength 0.4 cells from
the scalar-discrepancy comparison: heterogeneous and constant
comparative bias. It reports only point-estimation metrics because the official
QR-learner implementation does not expose a general prediction-variance
interface.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegressionCV


from utils import RESULTS_DIR, log

UPSTREAM_COMMIT = "6a62bd6b0bbf8e90c43d4108c3be227b3e4c0e6a"

from bcalm import BCalmLinear
from dgp import generate_data
from metrics import negative_transfer, posterior_metrics
from third_party.qr_learner.official_learners import QuasiOptimizedLearner
from utils import seed_list


METHOD = "QR-learner (official mixed)"
CELLS = [
    ("heterogeneous", False),
    ("constant", True),
]
RAW_COLUMNS = [
    "bias_shape",
    "bias_strength",
    "method",
    "seed",
    "cate_rmse",
    "neg_transfer",
    "rct_only_rmse",
    "fit_seconds",
    "upstream_commit",
]


def dgp_seed(loop_seed: int, bias_strength: float) -> int:
    """Match the seed derivation in the paper's run_main.py driver."""
    n_R = 500
    return 20000 + 1000 * loop_seed + n_R + int(round(100 * bias_strength))


def stack_sources(
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert the paper DGP dictionaries to the official QR-learner schema."""
    X = np.vstack([D_R["X"], D_O["X"]])
    S = np.concatenate([np.ones(len(D_R["X"])), np.zeros(len(D_O["X"]))]).reshape(-1, 1)
    A = np.concatenate([D_R["A"], D_O["A"]]).reshape(-1, 1)
    Y = np.concatenate([D_R["Y"], D_O["Y"]]).reshape(-1, 1)
    return X, S, A, Y


def official_mixed_qr() -> QuasiOptimizedLearner:
    """Instantiate the official repository's mixed QR-learner configuration."""
    return QuasiOptimizedLearner(
        propensity_score=0.5,
        regressor_cate=LinearRegression(),
        regressor_control=HistGradientBoostingRegressor(),
        regressor_treated=HistGradientBoostingRegressor(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=2,
    )


def run_cell_seed(
    bias_shape: str,
    constant_comparative: bool,
    seed: int,
) -> dict[str, object]:
    bias_strength = 0.4
    generated_seed = dgp_seed(seed, bias_strength)
    data = generate_data(
        n_R=500,
        n_O=5000,
        seed=generated_seed,
        regime="balanced",
        bias="comparative",
        bias_strength=bias_strength,
        kind="mildly_nonlinear",
        n_eval=600,
        constant_comparative=constant_comparative,
    )

    rct_model = BCalmLinear(
        sigma_delta=math.inf,
        sigma0=math.inf,
        sigma_y=1.0,
        basis_kind="mild",
    ).fit(data.D_R, None)
    rct_mean, rct_var = rct_model.posterior_cate_moments(data.X_eval)
    rct_rmse = posterior_metrics(rct_mean, rct_var, data.tau_eval)["cate_rmse"]

    # The upstream cross-fitting splitters use NumPy's global RNG when no
    # random_state is supplied. Reset it per task for reproducible parallel
    # execution without changing the official learner source.
    np.random.seed(generated_seed)
    X, S, A, Y = stack_sources(data.D_R, data.D_O)
    learner = official_mixed_qr()
    fit_start = time.time()
    learner.fit(X, S, A, Y)
    predictions = np.asarray(learner.predict(data.X_eval), dtype=float)
    fit_seconds = time.time() - fit_start

    if predictions.shape != data.tau_eval.shape:
        raise RuntimeError(
            f"Prediction shape {predictions.shape} does not match "
            f"truth shape {data.tau_eval.shape}."
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("QR-learner produced a non-finite prediction.")

    cate_rmse = float(np.sqrt(np.mean((predictions - data.tau_eval) ** 2)))
    return {
        "bias_shape": bias_shape,
        "bias_strength": bias_strength,
        "method": METHOD,
        "seed": seed,
        "cate_rmse": cate_rmse,
        "neg_transfer": negative_transfer(cate_rmse, rct_rmse),
        "rct_only_rmse": rct_rmse,
        "fit_seconds": fit_seconds,
        "upstream_commit": UPSTREAM_COMMIT,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bias_shape, group in raw.groupby("bias_shape", sort=False):
        row: dict[str, object] = {
            "bias_shape": bias_shape,
            "bias_strength": float(group["bias_strength"].iloc[0]),
            "method": METHOD,
            "n_seeds": len(group),
        }
        for metric in ("cate_rmse", "neg_transfer", "fit_seconds"):
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_se"] = values.std(ddof=1) / math.sqrt(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_summary(summary: pd.DataFrame) -> str:
    lines = [
        "# QR-learner benchmark",
        "",
        "Official mixed configuration on the balanced DGP at comparative-bias "
        "strength 0.4; means over 50 seeds.",
        "",
        "| Comparative bias | CATE RMSE | Neg. transfer |",
        "|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['bias_shape'].capitalize()} | "
            f"{row['cate_rmse_mean']:.3f} | "
            f"{row['neg_transfer_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Negative transfer uses the same seed-specific B-CALM RCT-only "
            "reference as the scalar-discrepancy comparison.",
            "",
            f"Upstream commit: `{UPSTREAM_COMMIT}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR = args.out_dir
    log(
        f"QR-learner: official mixed configuration (upstream commit {UPSTREAM_COMMIT}); NT: seed-specific Gaussian RCT-only"
    )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive.")

    start = time.time()
    if args.validate_only:
        row = run_cell_seed("heterogeneous", False, 0)
        frame = pd.DataFrame([row], columns=RAW_COLUMNS)
        if (
            not np.isfinite(
                frame[["cate_rmse", "neg_transfer", "rct_only_rmse", "fit_seconds"]]
            )
            .all()
            .all()
        ):
            raise RuntimeError("Validation did not produce finite metrics.")
        print(frame.to_string(index=False))
        print(f"validation_seconds={time.time() - start:.3f}")
        return

    tasks = [
        (bias_shape, constant_comparative, seed)
        for bias_shape, constant_comparative in CELLS
        for seed in seed_list(args.seeds)
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=args.workers, verbose=10)(
            delayed(run_cell_seed)(*task) for task in tasks
        )
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    shape_order = {"heterogeneous": 0, "constant": 1}
    raw["_shape_order"] = raw["bias_shape"].map(shape_order)
    raw = (
        raw.sort_values(["_shape_order", "seed"], kind="stable")
        .drop(columns="_shape_order")
        .reset_index(drop=True)
    )

    expected_rows = len(CELLS) * args.seeds
    if len(raw) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, produced {len(raw)}.")
    if raw[["cate_rmse", "neg_transfer", "rct_only_rmse"]].isna().any().any():
        raise RuntimeError("Missing metric detected in raw results.")
    if raw.groupby("bias_shape")["seed"].nunique().ne(args.seeds).any():
        raise RuntimeError("A benchmark cell is missing seeds.")

    summary = summarize(raw)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = TABLE_DIR / "qr_learner_benchmark.csv"
    summary_path = TABLE_DIR / "qr_learner_benchmark_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    elapsed = time.time() - start
    print(f"wrote {raw_path} ({len(raw)} rows)")
    print(f"wrote {summary_path} ({len(summary)} rows)")
    print(f"wall_clock_seconds={elapsed:.6f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
