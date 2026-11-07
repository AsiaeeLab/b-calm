"""Distributional discrepancy for the synthetic alignment regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from overlap import gaussian_mmd
from dgp import generate_data
from run_main import kind_for_dgp
from utils import RESULTS_DIR, TABLES_DIR, log
from experiment_io import write_table


def run_seed(regime: str, seed: int) -> list[dict]:
    data = generate_data(
        500,
        5000,
        20000 + 1000 * seed + 500 + 20,
        regime=regime,
        bias="comparative",
        bias_strength=0.2,
        kind=kind_for_dgp(regime),
        n_eval=600,
    )
    rows = []
    for name, columns in [("working", slice(None)), ("shared", slice(0, 3))]:
        result = gaussian_mmd(data.D_R["X"][:, columns], data.D_O["X"][:, columns])
        rows.append(dict(dgp=regime, seed=seed, representation=name, **result))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(
        "Working representation: all five X columns; operational shared block: X[:, :3]. No encoder transform. Covariates are generated before bias is used."
    )
    with parallel_config(backend="loky", inner_max_num_threads=1):
        nested = Parallel(n_jobs=args.workers, verbose=10)(
            delayed(run_seed)(regime, seed)
            for regime in ["shared-only", "balanced", "mismatch-heavy"]
            for seed in range(50)
        )
    raw = pd.DataFrame([row for rows in nested for row in rows])
    raw["reject"] = raw.permutation_p < 0.05
    raw.to_csv(args.out_dir / "overlap_ladder_raw.csv", index=False)
    summary = (
        raw.groupby(["dgp", "representation"], sort=False)
        .agg(
            mmd2_mean=("mmd2", "mean"),
            mmd2_sd=("mmd2", "std"),
            mmd_mean=("mmd", "mean"),
            mmd_sd=("mmd", "std"),
            bandwidth_mean=("bandwidth", "mean"),
            bandwidth_sd=("bandwidth", "std"),
            reject_fraction=("reject", "mean"),
            seeds=("seed", "size"),
            n_x=("n_x", "first"),
            n_y=("n_y", "first"),
        )
        .reset_index()
    )
    summary.to_csv(args.out_dir / "overlap_ladder.csv", index=False)
    rows = [
        f"{r.dgp} & {r.representation} & {r.mmd2_mean:.4f} $\\pm$ {r.mmd2_sd:.4f} & {r.mmd_mean:.4f} $\\pm$ {r.mmd_sd:.4f} & {r.bandwidth_mean:.3f} & {r.reject_fraction:.2f}"
        for r in summary.itertuples()
    ]
    caption = "Gaussian-kernel discrepancy. Synthetic entries show mean and SD over 50 seeds; last column is the fraction with permutation $p<0.05$."
    caption += " Tests subsample the larger group to 2,000 observations; discrepancy estimates use the full groups."
    write_table(
        args.table_dir / "overlap.tex",
        caption,
        "tab:overlap",
        "llrrrr",
        "Regime & Features & MMD$^2$ & MMD & Bandwidth & Test",
        rows,
    )
    log(summary.to_string(index=False))
    log(f"wall_clock_seconds={time.time() - start:.6f}")


if __name__ == "__main__":
    main()
