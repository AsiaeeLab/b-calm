from __future__ import annotations

import argparse
import csv
import time

from baselines import bcf_hahn_available, external_available, fit_method
from dgp import generate_data
from metrics import negative_transfer, posterior_metrics
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


FIELDNAMES = [
    "method",
    "dgp",
    "bias",
    "bias_strength",
    "n_R",
    "n_O",
    "seed",
    "cate_rmse",
    "iae",
    "coverage_90",
    "ci_width",
    "neg_transfer",
    "chosen_sigma_delta",
]

BMA_FIELDNAMES = ["dgp", "bias", "bias_strength", "n_R", "seed", "sigma_delta", "weight"]


def dump_bma_weights(model: object, dgp: str, bias: str, bias_strength: float, n_R: int, seed: int) -> None:
    """Persist per-replicate BMA weights so the retreat behavior is inspectable."""
    summary = getattr(model, "weight_summary_", None)
    if not summary:
        return
    rows = [
        {
            "dgp": dgp,
            "bias": bias,
            "bias_strength": bias_strength,
            "n_R": n_R,
            "seed": seed,
            "sigma_delta": sigma_label(sigma),
            "weight": weight,
        }
        for sigma, weight in summary.items()
    ]
    append_csv_rows(RESULTS_DIR / "bma_weights.csv", rows, BMA_FIELDNAMES)


def config_for_mode(mode: str) -> tuple[list[tuple[str, str, float]], list[int], list[int], str]:
    if mode == "smoke":
        return [("balanced", "comparative", 0.2)], [80], [0], "smoke_main.csv"
    if mode == "cheap":
        cells = [(dgp, bias, 0.2 if bias == "comparative" else 0.0) for dgp in ["shared-only", "balanced"] for bias in ["none", "comparative"]]
        return cells, [250], seed_list(5), "cheap_main.csv"
    cells = [
        (dgp, bias, 0.2 if bias == "comparative" else 0.0)
        for dgp in ["shared-only", "balanced", "mismatch-heavy"]
        for bias in ["none", "baseline-only", "comparative"]
    ]
    cells.extend(("balanced", "comparative", strength) for strength in [0.0, 0.5, 1.0])
    return cells, [250, 500], seed_list(50), "main.csv"


def kind_for_dgp(dgp: str) -> str:
    return {"shared-only": "linear", "balanced": "mildly_nonlinear", "mismatch-heavy": "nonlinear"}[dgp]


def basis_for_dgp(dgp: str) -> str:
    return {"shared-only": "linear", "balanced": "mild", "mismatch-heavy": "nonlinear"}[dgp]


def bias_strength_key(value: float) -> str:
    return str(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "cheap", "full"], default="full")
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    timer = Timer(f"main {args.mode}")
    cells, n_R_grid, seeds, filename = config_for_mode(args.mode)
    output = RESULTS_DIR / filename
    if output.exists() and output.stat().st_size > 0:
        with output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDNAMES:
                raise ValueError(f"{output} has unexpected columns; move it aside before running.")
    done = read_done(output, ["method", "dgp", "bias", "bias_strength", "n_R", "n_O", "seed"])
    methods = filter_methods(["B-CALM-selected", "RCT-only", "CF", "MR-OSCAR", "R-OSCAR", "CALM", "pooled", "OS-only"])
    sweep_methods = filter_methods(["B-CALM-selected", "RCT-only", "CF", "MR-OSCAR", "R-OSCAR", "CALM", "pooled"])
    # BCF (Hahn 2020) is skipped in the synthetic factorial grid: each fit takes
    # roughly 30 s to 2 min depending on n, which is prohibitive over the full
    # grid of cells and seeds. BCF is reported in the paper's real-data
    # application, where a single fit is feasible.
    n_O = 5000 if args.mode != "smoke" else 160
    total = len(cells) * len(n_R_grid) * len(seeds)
    start = time.time()
    counter = 0
    for dgp, bias, bias_strength in cells:
        for n_R in n_R_grid:
            for seed in seeds:
                counter += 1
                try:
                    data = generate_data(
                        n_R=n_R,
                        n_O=n_O,
                        seed=20000 + 1000 * seed + n_R + int(round(100 * bias_strength)),
                        regime=dgp,
                        bias=bias,
                        bias_strength=bias_strength,
                        kind=kind_for_dgp(dgp),
                        n_eval=600,
                    )
                    basis_kind = basis_for_dgp(dgp)
                    rct_model, _ = fit_method("RCT-only", data.D_R, data.D_O, basis_kind, seed)
                    rct_mean, rct_var = rct_model.posterior_cate_moments(data.X_eval)
                    rct_rmse = posterior_metrics(rct_mean, rct_var, data.tau_eval)["cate_rmse"]
                    rows = []
                    cell_methods = sweep_methods if (dgp, bias, bias_strength) in {
                        ("balanced", "comparative", 0.0),
                        ("balanced", "comparative", 0.5),
                        ("balanced", "comparative", 1.0),
                    } else methods
                    for method in cell_methods:
                        method_name = "B-CALM" if method == "B-CALM-selected" else method
                        key = (method_name, dgp, bias, bias_strength_key(bias_strength), str(n_R), str(n_O), str(seed))
                        if key in done:
                            continue
                        # BCF (Hahn) has Bayesian credible intervals from posterior MCMC; not bootstrapped.
                        # CF (Wager-Athey) has EconML's asymptotic IF-based SE; not bootstrapped here.
                        # The OSCAR family gets bootstrap intervals.
                        use_bootstrap = method in {"MR-OSCAR", "R-OSCAR", "CALM"}
                        try:
                            model, chosen_sigma = fit_method(
                                method,
                                data.D_R,
                                data.D_O,
                                basis_kind,
                                seed,
                                bootstrap_b=args.bootstrap_b if use_bootstrap else 0,
                                workers=args.workers,
                            )
                        except ImportError as imp_exc:
                            # CF requires econml; OSCAR/CALM require the companion
                            # repo. If a dependency is missing at fit time, skip the
                            # method gracefully instead of aborting the run.
                            log(f"skipping {method} (missing dependency): {imp_exc}")
                            continue
                        if method == "B-CALM-selected":
                            dump_bma_weights(model, dgp, bias, bias_strength, n_R, seed)
                        mean, var = model.posterior_cate_moments(data.X_eval)
                        met = posterior_metrics(mean, var, data.tau_eval)
                        rows.append(
                            {
                                "method": method_name,
                                "dgp": dgp,
                                "bias": bias,
                                "bias_strength": bias_strength,
                                "n_R": n_R,
                                "n_O": n_O,
                                "seed": seed,
                                "cate_rmse": met["cate_rmse"],
                                "iae": met["iae"],
                                "coverage_90": met["coverage_90"],
                                "ci_width": met["ci_width"],
                                "neg_transfer": negative_transfer(met["cate_rmse"], rct_rmse),
                                "chosen_sigma_delta": sigma_label(chosen_sigma),
                            }
                        )
                    append_csv_rows(output, rows, FIELDNAMES)
                except Exception as exc:
                    dump_failure(seed, {"script": "run_main", "dgp": dgp, "bias": bias, "bias_strength": bias_strength, "n_R": n_R}, exc)
                    raise
                progress(counter, total, start, "main", every=20)
    seconds = timer.finish()
    record_wallclock(f"main_{args.mode}", seconds, {"rows_file": str(output), "seeds": len(seeds)})
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
