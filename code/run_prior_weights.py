"""Sensitivity of the synthetic factorial study to discrepancy-scale weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from baselines import SIGMA_MODEL_PRIOR, SIGMA_GRID, fit_bcalm_selected
from dgp import generate_data
from metrics import posterior_metrics, negative_transfer
from run_main import config_for_mode, kind_for_dgp, basis_for_dgp
from utils import RESULTS_DIR, TABLES_DIR, log
from experiment_io import compare_frames, write_table

PRIORS = {
    "frozen": SIGMA_MODEL_PRIOR,
    "uniform": np.ones(6) / 6,
    "concentrated": np.array([0.01, 0.01, 0.65, 0.15, 0.10, 0.08]),
}
KEYS = ["dgp", "bias", "bias_strength", "n_R", "n_O", "seed"]
METRICS = ["cate_rmse", "coverage_90", "ci_width", "neg_transfer"]


def run_one(
    dgp: str, bias: str, strength: float, nr: int, seed: int, name: str, rct_rmse: float
) -> dict:
    data = generate_data(
        nr,
        5000,
        20000 + 1000 * seed + nr + int(round(100 * strength)),
        regime=dgp,
        bias=bias,
        bias_strength=strength,
        kind=kind_for_dgp(dgp),
        n_eval=600,
    )
    model, branch = fit_bcalm_selected(
        data.D_R, data.D_O, basis_for_dgp(dgp), seed, prior_weights=PRIORS[name]
    )
    assert branch == "bma" and tuple(model.sigma_grid_) == tuple(SIGMA_GRID)
    assert all(m.sigma0 == 0.6 for m in model.components)
    mean, var = model.posterior_cate_moments(data.X_eval)
    met = posterior_metrics(mean, var, data.tau_eval)
    return dict(
        prior=name,
        dgp=dgp,
        bias=bias,
        bias_strength=strength,
        n_R=nr,
        n_O=5000,
        seed=seed,
        cate_rmse=met["cate_rmse"],
        iae=met["iae"],
        coverage_90=met["coverage_90"],
        ci_width=met["ci_width"],
        neg_transfer=negative_transfer(met["cate_rmse"], rct_rmse),
    )


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
    old = pd.read_csv(RESULTS_DIR / "main.csv", keep_default_na=False)
    ref = old[old.method.eq("RCT-only")].set_index(KEYS).cate_rmse
    cells, nrs, seeds, _ = config_for_mode("full")
    raw = []
    summary = []
    for name, prior in PRIORS.items():
        log(
            f"prior={name}: {prior}; branch=contrast BMA; candidate sigma0=0.6; grid={SIGMA_GRID}"
        )
        tasks = [
            (d, b, s, n, seed, name, float(ref.loc[(d, b, s, n, 5000, seed)]))
            for d, b, s in cells
            for n in nrs
            for seed in seeds
        ]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=args.workers, verbose=10)(
                delayed(run_one)(*t) for t in tasks
            )
        df = pd.DataFrame(rows)
        raw.append(df)
        if name == "frozen":
            columns = KEYS + METRICS + ["iae"]
            check = compare_frames(
                df[columns], old[old.method.eq("B-CALM")][columns], KEYS, digits=10
            )
            if not check["exact"]:
                df.to_csv(args.out_dir / "prior_weights_failed_check.csv", index=False)
                raise AssertionError(
                    "Frozen main-grid reproduction failed; sensitivity stopped."
                )
        grouped = df.groupby(KEYS[:-1]).coverage_90.mean()
        minimum = grouped.idxmin()
        row = {
            "prior": name,
            "group": "all",
            **df[METRICS].mean().to_dict(),
            "minimum_cell_coverage": grouped.min(),
            "minimum_cell": str(minimum),
        }
        summary.append(row)
        df["bias_group"] = np.where(
            df.bias.eq("comparative"),
            "comparative@" + df.bias_strength.map(lambda x: f"{x:g}"),
            df.bias,
        )
        for group, g in df.groupby("bias_group", sort=False):
            summary.append(
                {"prior": name, "group": group, **g[METRICS].mean().to_dict()}
            )
    pd.concat(raw).to_csv(args.out_dir / "prior_weights_raw.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(args.out_dir / "prior_weights.csv", index=False)
    rows = [
        f"{r.prior} & {r.group} & {r.cate_rmse:.3f} & {r.coverage_90:.3f} & {r.ci_width:.3f} & {r.neg_transfer:.3f}"
        for r in result.itertuples()
    ]
    write_table(
        args.table_dir / "prior_weights.tex",
        "Discrepancy-scale prior sensitivity over the full factorial grid. NT compares each fit to its seed-matched RCT-only result.",
        "tab:prior-weights",
        "llrrrr",
        "Prior & Bias & RMSE & Coverage & Width & NT",
        rows,
    )
    log(result.to_string(index=False))
    log(f"wall_clock_seconds={time.time() - start:.6f}")


if __name__ == "__main__":
    main()
