from __future__ import annotations

import argparse
import time
from pathlib import Path

from baselines import fit_ablation, fit_method
from dgp import generate_data
from metrics import negative_transfer, posterior_metrics
from utils import RESULTS_DIR, Timer, append_csv_rows, dump_failure, ensure_dirs, log, progress, record_wallclock, read_done, seed_list


FIELDNAMES = ["variant", "seed", "cate_rmse", "coverage_90", "ci_width", "neg_transfer"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "cheap", "full"], default="full")
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    ensure_dirs()
    timer = Timer(f"ablation {args.mode}")
    n_seeds = {"smoke": 2, "cheap": 10, "full": 50}[args.mode]
    n_R = 500 if args.mode != "smoke" else 100
    n_O = 5000 if args.mode != "smoke" else 200
    output = args.out_dir / {"smoke": "smoke_ablation.csv", "cheap": "cheap_ablation.csv", "full": "ablation.csv"}[args.mode]
    done = read_done(output, ["variant", "seed"])
    variants = ["Full B-CALM", "No b_delta", "No b0"]
    log("Full and No b0: contrast-weighted six-component mixture; sigma0=0.6 and 0; No b_delta: sigma_delta=0, sigma0=0.6")
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
                bias_strength=0.2,
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
            log(f"seed={seed} failed: {exc}")
            raise
        progress(index, len(seeds), start, "ablation", every=10)
    seconds = timer.finish()
    record_wallclock(f"ablation_{args.mode}", seconds, {"rows_file": str(output), "seeds": n_seeds}, out_dir=args.out_dir)
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
