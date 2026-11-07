from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from baselines import bcf_hahn_available, external_available, fit_method
from dgp import generate_data
from metrics import coverage_at_level
from utils import RESULTS_DIR, Timer, append_csv_rows, dump_failure, ensure_dirs, log, progress, record_wallclock, seed_list


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


FIELDNAMES = ["method", "alpha", "empirical_coverage", "n_seeds"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "cheap", "full"], default="full")
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    timer = Timer(f"calibration {args.mode}")
    n_seeds = {"smoke": 2, "cheap": 20, "full": 200}[args.mode]
    filename = {"smoke": "smoke_calibration.csv", "cheap": "cheap_calibration.csv", "full": "calibration.csv"}[args.mode]
    n_R = 500 if args.mode != "smoke" else 100
    n_O = 5000 if args.mode != "smoke" else 200
    # B-CALM uses empirical-Bayes model averaging ("B-CALM-selected"),
    # accumulated and reported under the display name "B-CALM".
    methods = filter_methods(["B-CALM-selected", "R-OSCAR", "MR-OSCAR", "CALM", "CF", "pooled"])
    # BCF (Hahn) intentionally skipped; see comment in run_main.py.
    alphas = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]
    cover_sum = defaultdict(float)
    cover_n = defaultdict(int)
    seeds = seed_list(n_seeds)
    start = time.time()
    for index, seed in enumerate(seeds, start=1):
        try:
            data = generate_data(
                n_R=n_R,
                n_O=n_O,
                seed=30000 + seed,
                regime="balanced",
                bias="comparative",
                kind="mildly_nonlinear",
                n_eval=500,
            )
            for method in methods:
                method_name = "B-CALM" if method == "B-CALM-selected" else method
                use_bootstrap = method in {"MR-OSCAR", "R-OSCAR", "CALM"}
                try:
                    model, _ = fit_method(
                        method,
                        data.D_R,
                        data.D_O,
                        "mild",
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
                for alpha in alphas:
                    cover_sum[(method_name, alpha)] += coverage_at_level(mean, var, data.tau_eval, alpha)
                    cover_n[(method_name, alpha)] += 1
        except Exception as exc:
            dump_failure(seed, {"script": "run_calibration"}, exc)
            raise
        progress(index, len(seeds), start, "calibration", every=10)
    method_names = ["B-CALM" if m == "B-CALM-selected" else m for m in methods]
    # A method skipped for a missing optional dependency has no fits; omit it
    # rather than writing zero-coverage rows.
    skipped = [m for m in method_names if not any(cover_n[(m, alpha)] for alpha in alphas)]
    if skipped:
        log("no fits for " + ", ".join(skipped) + "; omitted from the output")
    rows = [
        {
            "method": method,
            "alpha": alpha,
            "empirical_coverage": cover_sum[(method, alpha)] / cover_n[(method, alpha)],
            "n_seeds": n_seeds,
        }
        for method in method_names
        if method not in skipped
        for alpha in alphas
    ]
    output = RESULTS_DIR / filename
    if output.exists():
        output.unlink()
    append_csv_rows(output, rows, FIELDNAMES)
    seconds = timer.finish()
    record_wallclock(f"calibration_{args.mode}", seconds, {"rows_file": str(output), "seeds": n_seeds})
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
