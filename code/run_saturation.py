from __future__ import annotations

import argparse
import math
import time

from baselines import fit_method
from dgp import generate_data
from metrics import empirical_os_ess, posterior_metrics, theory_os_ess
from utils import RESULTS_DIR, Timer, append_csv_rows, dump_failure, ensure_dirs, log, progress, record_wallclock, read_done, seed_list, sigma_label


FIELDNAMES = [
    "method",
    "n_R",
    "n_O",
    "sigma_delta",
    "seed",
    "cate_rmse",
    "ci_width_mean",
    "coverage_90",
    "ess_O_empirical",
    "ess_O_theory",
]


def config_for_mode(mode: str) -> tuple[list[int], list[int], list[float], list[int], str]:
    if mode == "smoke":
        return [80], [120], [0.2, math.inf], [0], "smoke_saturation.csv"
    if mode == "cheap":
        return [250, 500], [500, 5000], [0.1, 0.2, 0.4, math.inf], seed_list(5), "cheap_saturation.csv"
    return [250, 500], [500, 1000, 2000, 5000, 10000, 20000, 50000], [0.1, 0.2, 0.4, math.inf], seed_list(50), "saturation.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "cheap", "full"], default="full")
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    timer = Timer(f"saturation {args.mode}")
    n_R_grid, n_O_grid, sigmas, seeds, filename = config_for_mode(args.mode)
    output = RESULTS_DIR / filename
    done = read_done(output, ["method", "n_R", "n_O", "sigma_delta", "seed"])
    total = len(n_R_grid) * len(n_O_grid) * len(seeds)
    start = time.time()
    counter = 0
    sigma_o2 = 4.0
    for n_R in n_R_grid:
        for n_O in n_O_grid:
            for seed in seeds:
                counter += 1
                try:
                    data = generate_data(
                        n_R=n_R,
                        n_O=n_O,
                        seed=10000 + seed,
                        regime="shared-only",
                        bias="comparative",
                        kind="linear",
                        n_eval=400,
                        constant_tau=True,
                        constant_comparative=True,
                    )
                    rct_model, _ = fit_method("RCT-only", data.D_R, data.D_O, "linear", seed)
                    var_rct = rct_model.tau_intercept_var()
                    rows = []
                    for sigma in sigmas:
                        key = ("B-CALM", str(n_R), str(n_O), sigma_label(sigma), str(seed))
                        if key not in done:
                            model, _ = fit_method("B-CALM", data.D_R, data.D_O, "linear", seed, sigma_delta=sigma, m_delta=0.2)
                            mean, var = model.posterior_cate_moments(data.X_eval)
                            met = posterior_metrics(mean, var, data.tau_eval)
                            rows.append(
                                {
                                    "method": "B-CALM",
                                    "n_R": n_R,
                                    "n_O": n_O,
                                    "sigma_delta": sigma_label(sigma),
                                    "seed": seed,
                                    "cate_rmse": met["cate_rmse"],
                                    "ci_width_mean": met["ci_width_mean"],
                                    "coverage_90": met["coverage_90"],
                                    "ess_O_empirical": empirical_os_ess(sigma_o2, model.tau_intercept_var(), var_rct),
                                    "ess_O_theory": theory_os_ess(n_O, sigma, sigma_o2),
                                }
                            )
                    for method in ["pooled", "OS-only"]:
                        key = (method, str(n_R), str(n_O), "0", str(seed))
                        if key not in done:
                            model, sigma = fit_method(method, data.D_R, data.D_O, "linear", seed, sigma_delta=0.0)
                            mean, var = model.posterior_cate_moments(data.X_eval)
                            met = posterior_metrics(mean, var, data.tau_eval)
                            rows.append(
                                {
                                    "method": method,
                                    "n_R": n_R,
                                    "n_O": n_O,
                                    "sigma_delta": sigma_label(sigma),
                                    "seed": seed,
                                    "cate_rmse": met["cate_rmse"],
                                    "ci_width_mean": met["ci_width_mean"],
                                    "coverage_90": met["coverage_90"],
                                    "ess_O_empirical": float(n_O),
                                    "ess_O_theory": float(n_O),
                                }
                            )
                    append_csv_rows(output, rows, FIELDNAMES)
                except Exception as exc:
                    dump_failure(seed, {"script": "run_saturation", "n_R": n_R, "n_O": n_O}, exc)
                    raise
                progress(counter, total, start, "saturation", every=20)
    seconds = timer.finish()
    record_wallclock(f"saturation_{args.mode}", seconds, {"rows_file": str(output), "seeds": len(seeds)})
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
