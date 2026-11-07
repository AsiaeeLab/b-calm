"""Tables and reproduction checks for ablations and additional baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import pandas as pd
from utils import RESULTS_DIR, TABLES_DIR, log
from experiment_io import compare_frames, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--input-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_dir / "ablation.csv")
    old = pd.read_csv(RESULTS_DIR / "ablation.csv")
    keep = ["Full B-CALM", "No b_delta"]
    if not compare_frames(
        df[df.variant.isin(keep)], old[old.variant.isin(keep)], ["variant", "seed"]
    )["exact"]:
        raise AssertionError("Ablation reproduction failed")
    agg = df.groupby("variant").agg(
        rmse=("cate_rmse", "mean"),
        rmse_sd=("cate_rmse", "std"),
        coverage=("coverage_90", "mean"),
        width=("ci_width", "mean"),
        nt=("neg_transfer", "mean"),
    )
    labels = {
        "Full B-CALM": "Full B-CALM",
        "No b0": r"No $b_0$ ($\sigma_0=0$)",
        "No b_delta": r"No $b_\Delta$ ($\sigma_\Delta=0$)",
    }
    rows = []
    for variant, label in labels.items():
        r = agg.loc[variant]
        rows.append(
            f"{label} & {r.coverage:.3f} & {r.nt:.3f} & {r.rmse:.3f} $\\pm$ {r.rmse_sd:.3f} & {r.width:.3f}"
        )
    write_table(
        args.out_dir / "ablation.tex",
        "B-CALM ablations on the balanced comparative-bias setting. Coverage and negative-transfer metrics are shown before RMSE to emphasize calibrated borrowing.",
        "tab:ablation",
        "lrrrr",
        "Variant & Coverage & Neg. transfer & RMSE & Width",
        rows,
    )
    log("Ablation summary:\n" + agg.to_string())
    scalar = pd.read_csv(
        args.input_dir / "commensurate_baseline_summary.csv", keep_default_na=False
    )
    if args.reference_dir:
        old = pd.read_csv(
            args.reference_dir / "commensurate_baseline_summary.csv",
            keep_default_na=False,
        )
        if not compare_frames(scalar, old, ["bias_shape", "bias_strength", "method"])[
            "exact"
        ]:
            log("Scalar reproduction failed; scalar table stopped.")
            scalar = None
    if scalar is not None:
        labels = {
            "B-CALM (function-valued)": "B-CALM",
            "Commensurate (scalar b_delta)": r"Scalar $b_\Delta$",
            "Commensurate (scalar b_delta + scalar b_0)": r"Scalar $b_\Delta,b_0$",
            "Robust MAP (mixture on scalar b_delta)": "Robust MAP",
        }
        rows = []
        for r in scalar.itertuples():
            if r.method not in labels or r.bias_shape == "n/a":
                continue
            rows.append(
                f"{r.bias_shape} & {r.bias_strength:g} & {labels[r.method]} & {r.coverage_90_mean:.3f} & {r.ci_width_mean:.3f} & {r.cate_rmse_mean:.3f} & {r.neg_transfer_mean:.3f}"
            )
        write_table(
            args.out_dir / "scalar_bias.tex",
            "Function-valued and scalar discrepancy models on the balanced DGP. Means over 50 seeds; NT uses the seed-specific Gaussian RCT-only CATE RMSE.",
            "tab:scalar-bias",
            "llrrrrr",
            "Bias & Strength & Method & Coverage & Width & RMSE & NT",
            rows,
        )
        log(scalar.to_string(index=False))
    qr = pd.read_csv(args.input_dir / "qr_learner_benchmark_summary.csv")
    if args.reference_dir:
        old = pd.read_csv(args.reference_dir / "qr_learner_benchmark_summary.csv")
        columns = [c for c in old if not c.startswith("fit_seconds")]
        check = compare_frames(
            qr[columns], old[columns], ["bias_shape", "bias_strength", "method"]
        )
        log(
            "Runtime-column differences:\n"
            + (
                qr.filter(like="fit_seconds") - old.filter(like="fit_seconds")
            ).to_string(index=False)
        )
        if not check["exact"]:
            log("QR metric reproduction failed; QR table stopped.")
            qr = None
    if qr is not None and scalar is not None:
        rows = []
        for r in qr.itertuples():
            b = scalar[
                scalar.bias_shape.eq(r.bias_shape)
                & scalar.bias_strength.eq(0.4)
                & scalar.method.eq("B-CALM (function-valued)")
            ].iloc[0]
            rows.append(
                f"{r.bias_shape} & {r.cate_rmse_mean:.3f} & {r.neg_transfer_mean:.3f} & {b.cate_rmse_mean:.3f} & {b.neg_transfer_mean:.3f}"
            )
        write_table(
            args.out_dir / "qr_learner.tex",
            "Official mixed QR-learner and B-CALM at comparative-bias strength 0.4. Both NT columns use the same seed-specific Gaussian RCT-only CATE RMSE; 50 seeds per cell.",
            "tab:qr-learner",
            "lrrrr",
            "Bias & QR RMSE & QR NT & B-CALM RMSE & B-CALM NT",
            rows,
        )
        log(qr.to_string(index=False))
    log(f"wall_clock_seconds={time.time() - start:.6f}")


if __name__ == "__main__":
    main()
