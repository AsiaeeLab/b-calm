from __future__ import annotations

import argparse
import time

from baselines import external_available, fit_ablation, fit_method
from dgp import generate_data
from metrics import negative_transfer, posterior_metrics
from utils import RESULTS_DIR, Timer, append_csv_rows, dump_failure, ensure_dirs, log, progress, record_wallclock, read_done, seed_list


OSCAR_FAMILY = {"MR-OSCAR", "R-OSCAR", "CALM"}


def filter_methods(methods: list[str]) -> list[str]:
    """Drop OSCAR/CALM-family baselines when their companion repo is absent.

    The ablation grid is composed of B-CALM variants only, so this is a no-op
    here; it is kept for symmetry with the other runners.
    """
    if external_available():
        return methods
    dropped = [m for m in methods if m in OSCAR_FAMILY]
    if dropped:
        log(
            "skipping OSCAR/CALM-family baselines (companion repo not found; "
            "set BCALM_EXTERNAL_CALM_PATH to enable): " + ", ".join(sorted(set(dropped)))
        )
    return [m for m in methods if m not in OSCAR_FAMILY]


FIELDNAMES = ["variant", "seed", "cate_rmse", "coverage_90", "ci_width", "neg_transfer"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "cheap", "full"], default="full")
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    timer = Timer(f"ablation {args.mode}")
    n_seeds = {"smoke": 2, "cheap": 10, "full": 50}[args.mode]
    n_R = 500 if args.mode != "smoke" else 100
    n_O = 5000 if args.mode != "smoke" else 200
    output = RESULTS_DIR / {"smoke": "smoke_ablation.csv", "cheap": "cheap_ablation.csv", "full": "ablation.csv"}[args.mode]
    done = read_done(output, ["variant", "seed"])
    variants = filter_methods(["Full B-CALM", "No b_delta", "No b0", "Deterministic encoders", "Point plus bootstrap"])
    seeds = seed_list(n_seeds)
    start = time.time()
    for index, seed in enumerate(seeds, start=1):
        try:
            data = generate_data(
                n_R=n_R,
                n_O=n_O,
                seed=50000 + seed,
                regime="balanced",
                bias="comparative",
                kind="mildly_nonlinear",
                n_eval=600,
            )
            rct_model, _ = fit_method("RCT-only", data.D_R, data.D_O, "mild", seed)
            rct_mean, rct_var = rct_model.posterior_cate_moments(data.X_eval)
            rct_rmse = posterior_metrics(rct_mean, rct_var, data.tau_eval)["cate_rmse"]
            rows = []
            for variant in variants:
                if (variant, str(seed)) in done:
                    continue
                model = fit_ablation(variant, data.D_R, data.D_O, "mild", seed=seed)
                mean, var = model.posterior_cate_moments(data.X_eval)
                met = posterior_metrics(mean, var, data.tau_eval)
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "cate_rmse": met["cate_rmse"],
                        "coverage_90": met["coverage_90"],
                        "ci_width": met["ci_width"],
                        "neg_transfer": negative_transfer(met["cate_rmse"], rct_rmse),
                    }
                )
            append_csv_rows(output, rows, FIELDNAMES)
        except Exception as exc:
            dump_failure(seed, {"script": "run_ablation"}, exc)
            raise
        progress(index, len(seeds), start, "ablation", every=10)
    seconds = timer.finish()
    record_wallclock(f"ablation_{args.mode}", seconds, {"rows_file": str(output), "seeds": n_seeds})
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
