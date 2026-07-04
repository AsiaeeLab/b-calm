from __future__ import annotations

import argparse
import math
import time

import numpy as np

from baselines import bcf_hahn_available, external_available, fit_method
from dgp import generate_star_like
from metrics import interval, posterior_metrics
from utils import RESULTS_DIR, Timer, append_csv_rows, dump_failure, ensure_dirs, log, progress, record_wallclock, read_done, seed_list, sigma_label


OSCAR_FAMILY = {"MR-OSCAR", "R-OSCAR", "CALM"}


def filter_methods(methods: list[str]) -> list[str]:
    """Drop OSCAR/CALM-family baselines when their companion repo is absent."""
    if external_available():
        return methods
    dropped = [m for m in methods if m in OSCAR_FAMILY]
    if dropped:
        log(
            "skipping OSCAR/CALM-family baselines (companion repo not found; "
            "set BCALM_EXTERNAL_CALM_PATH to enable): " + ", ".join(sorted(set(dropped)))
        )
    return [m for m in methods if m not in OSCAR_FAMILY]


SEMI_FIELDS = ["method", "seed", "cate_rmse", "coverage_90"]
SENS_FIELDS = ["sigma_delta", "x_slice_id", "posterior_cate_mean", "posterior_cate_lo", "posterior_cate_hi"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "cheap", "full"], default="full")
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    timer = Timer(f"semi {args.mode}")
    n_seeds = {"smoke": 2, "cheap": 10, "full": 50}[args.mode]
    n_R = 500 if args.mode != "smoke" else 100
    n_O = 5000 if args.mode != "smoke" else 200
    semi_file = RESULTS_DIR / {"smoke": "smoke_semi.csv", "cheap": "cheap_semi.csv", "full": "semi.csv"}[args.mode]
    sens_file = RESULTS_DIR / {"smoke": "smoke_semi_sensitivity.csv", "cheap": "cheap_semi_sensitivity.csv", "full": "semi_sensitivity.csv"}[args.mode]
    done = read_done(semi_file, ["method", "seed"])
    # B-CALM uses the same empirical-Bayes model averaging over sigma_delta as
    # the main factorial grid ("B-CALM-selected"), reported under "B-CALM".
    methods = filter_methods(["B-CALM-selected", "RCT-only", "CF", "MR-OSCAR", "pooled", "OS-only"])
    # BCF (Hahn) intentionally skipped; see comment in run_main.py.
    seeds = seed_list(n_seeds)
    start = time.time()
    for index, seed in enumerate(seeds, start=1):
        try:
            data = generate_star_like(n_R=n_R, n_O=n_O, seed=40000 + seed, n_eval=600)
            rows = []
            for method in methods:
                method_name = "B-CALM" if method == "B-CALM-selected" else method
                key = (method_name, str(seed))
                if key in done:
                    continue
                use_bootstrap = method in {"MR-OSCAR", "R-OSCAR", "CALM"}
                try:
                    model, _ = fit_method(
                        method,
                        data.D_R,
                        data.D_O,
                        "star",
                        seed,
                        sigma_delta=0.2,
                        bootstrap_b=args.bootstrap_b if use_bootstrap else 0,
                        workers=args.workers,
                    )
                except ImportError as imp_exc:
                    # CF requires econml; OSCAR/CALM require the companion repo.
                    # Skip gracefully if a dependency is missing at fit time.
                    log(f"skipping {method} (missing dependency): {imp_exc}")
                    continue
                mean, var = model.posterior_cate_moments(data.X_eval)
                met = posterior_metrics(mean, var, data.tau_eval)
                rows.append(
                    {
                        "method": method_name,
                        "seed": seed,
                        "cate_rmse": met["cate_rmse"],
                        "coverage_90": met["coverage_90"],
                    }
                )
            append_csv_rows(semi_file, rows, SEMI_FIELDS)
        except Exception as exc:
            dump_failure(seed, {"script": "run_semi"}, exc)
            raise
        progress(index, len(seeds), start, "semi", every=10)

    data = generate_star_like(n_R=n_R, n_O=n_O, seed=49999, n_eval=600)
    order = np.argsort(data.X_eval[:, 0])
    slice_ids = order[np.linspace(0, len(order) - 1, 7, dtype=int)]
    rows = []
    for sigma in [0.0, 0.1, 0.2, 0.4, math.inf]:
        model, _ = fit_method("B-CALM", data.D_R, data.D_O, "star", 999, sigma_delta=sigma)
        mean, var = model.posterior_cate_moments(data.X_eval[slice_ids])
        lo, hi = interval(mean, var)
        for j, idx in enumerate(slice_ids):
            rows.append(
                {
                    "sigma_delta": sigma_label(sigma),
                    "x_slice_id": j,
                    "posterior_cate_mean": mean[j],
                    "posterior_cate_lo": lo[j],
                    "posterior_cate_hi": hi[j],
                }
            )
    if sens_file.exists():
        sens_file.unlink()
    append_csv_rows(sens_file, rows, SENS_FIELDS)
    seconds = timer.finish()
    record_wallclock(f"semi_{args.mode}", seconds, {"rows_file": str(semi_file), "sensitivity_file": str(sens_file), "seeds": n_seeds})
    log(f"wrote {semi_file} and {sens_file}")


if __name__ == "__main__":
    main()
