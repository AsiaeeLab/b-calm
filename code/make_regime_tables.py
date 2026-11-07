"""Summarize bias regimes, alignment regimes, and calibrated baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from utils import RESULTS_DIR, TABLES_DIR, log
from experiment_io import compare_frames, write_table

FAM = ["R-OSCAR", "MR-OSCAR", "CALM"]
METRICS = ["cate_rmse", "coverage_90", "ci_width", "neg_transfer", "iae"]
CELL = ["dgp", "bias", "bias_strength", "n_R", "n_O"]
METHODS = ["B-CALM", "OSCAR/CALM", "pooled", "CF", "RCT-only"]
REGIMES = [
    "none",
    "baseline-only",
    "comparative b=0",
    "comparative b=0.2",
    "comparative b=0.5",
    "comparative b=1",
]


def metrics(d: pd.DataFrame) -> dict:
    return dict(
        n_rows=len(d),
        coverage_90=d.coverage_90.mean(),
        coverage_se=d.coverage_90.std() / np.sqrt(len(d)),
        ci_width=d.ci_width.mean(),
        cate_rmse=d.cate_rmse.mean(),
        neg_transfer=d.neg_transfer.mean(),
    )


def precision(df: pd.DataFrame, out: Path, table: Path, reference: Path | None) -> None:
    gm = pd.DataFrame(
        [
            dict(
                method=m,
                n_rows=len(df[df.method.eq(m)]),
                **df[df.method.eq(m)][METRICS].mean().to_dict(),
            )
            for m in FAM
        ]
    )
    summary = {}
    seedlevel = {}
    se_rows = []
    piv = df[df.method.isin(FAM)].pivot_table(
        index=CELL + ["seed"], columns="method", values=METRICS
    )
    for k in METRICS:
        vals = gm[k].to_numpy(float)
        spread = float(vals.max() - vals.min())
        ses = []
        for m in FAM:
            d = df[df.method.eq(m)]
            pc = d.groupby(CELL)[k].agg(["mean", "std", "count"])
            se_cell = pc["std"] / np.sqrt(pc["count"])
            se = float(np.sqrt((se_cell.to_numpy() ** 2).sum()) / len(pc))
            ses.append(se)
            se_rows.append(
                dict(
                    method=m,
                    metric=k,
                    mean=float(gm.loc[gm.method.eq(m), k].iloc[0]),
                    mc_se=se,
                )
            )
        summary[k] = dict(
            values={m: float(v) for m, v in zip(FAM, vals)},
            between_method_spread=spread,
            mc_se_of_grid_mean=float(np.mean(ses)),
            spread_over_se=spread / float(np.mean(ses)),
        )
        sub = piv[k][FAM].dropna()
        any_diff = np.zeros(len(sub), dtype=bool)
        dmax = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                d = (sub[FAM[i]] - sub[FAM[j]]).abs()
                dmax = max(dmax, float(d.max()))
                any_diff |= (d > 0).to_numpy()
        seedlevel[k] = dict(
            max_abs_pairwise_diff=dmax,
            n_seed_cells=len(sub),
            frac_seed_cells_with_any_diff=float(any_diff.mean()),
        )
    ref = {}
    pivall = df.pivot_table(index=CELL + ["seed"], columns="method", values=METRICS)
    for other in ["RCT-only", "pooled", "B-CALM"]:
        ref[other] = {}
        for k in METRICS:
            sub = pivall[k][["MR-OSCAR", other]].dropna()
            d = (sub["MR-OSCAR"] - sub[other]).abs()
            ref[other][k] = dict(
                max_abs_diff=float(d.max()),
                mean_abs_diff=float(d.mean()),
                frac_nonzero=float((d > 0).mean()),
            )
    result = dict(grid_means=summary, seed_level=seedlevel, vs_other_methods=ref)
    (out / "baseline_distinctness.json").write_text(json.dumps(result, indent=2))
    gm.to_csv(out / "baseline_distinctness_gridmeans.csv", index=False)
    se_frame = pd.DataFrame(se_rows)
    se_frame.to_csv(out / "oscar_precision.csv", index=False)
    seeds = pd.DataFrame(
        [
            dict(
                metric=k,
                **v,
                n_differing=round(
                    v["frac_seed_cells_with_any_diff"] * v["n_seed_cells"]
                ),
            )
            for k, v in seedlevel.items()
        ]
    )
    seeds.to_csv(out / "oscar_seed_differences.csv", index=False)
    if reference:
        old = json.loads((reference / "baseline_distinctness.json").read_text())
        log(f"baseline_distinctness JSON exact={result == old}")
        if result != old:
            raise AssertionError("Baseline precision did not reproduce")
    # The real-data row needs the restricted-data output of run_gps_main.py.
    gps_path = RESULTS_DIR / "gps_main_summary.csv"
    gps = None
    if gps_path.exists():
        gps = pd.read_csv(gps_path)
        gps = gps[gps.method.isin(FAM)][["method", "ate_mean"]]
        gps.to_csv(out / "oscar_realdata.csv", index=False)
    else:
        log(f"{gps_path.name} not found; omitting the real-data ATE row")
    rows = []
    for k in METRICS[:4]:
        means = " & ".join(f"{summary[k]['values'][m]:.9g}" for m in FAM)
        ses = " & ".join(
            f"{se_frame[(se_frame.metric == k) & (se_frame.method == m)].mc_se.iloc[0]:.9g}"
            for m in FAM
        )
        rows.append(k.replace("_", r"\_") + " mean & " + means)
        rows.append("MC SE & " + ses)
    for r in seeds.itertuples():
        rows.append(
            f"{r.metric.replace('_', chr(92) + '_')} max diff & {r.max_abs_pairwise_diff:.9g} & {r.n_differing}/{r.n_seed_cells} & --"
        )
    if gps is not None:
        rows.append(
            "GPS ATE & "
            + " & ".join(
                f"{gps.loc[gps.method.eq(m), 'ate_mean'].iloc[0]:.7f}" for m in FAM
            )
        )
    write_table(
        table / "oscar_precision.tex",
        "OSCAR-family precision. MC SE combines within-cell seed variances with equal cell weights. Difference rows give the maximum pairwise difference and the number of differing seed-cells.",
        "tab:oscar-precision",
        "lrrr",
        "Metric & R-OSCAR & MR-OSCAR & CALM",
        rows,
    )
    log(gm.to_string(index=False))
    log(seeds.to_string(index=False))
    if gps is not None:
        log(gps.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "main.csv")
    df["bias"] = df.bias.fillna("none")
    df["regime"] = np.where(
        df.bias.eq("comparative"),
        "comparative b=" + df.bias_strength.map(lambda v: f"{v:g}"),
        df.bias,
    )
    df["disp"] = df.method.replace({m: "OSCAR/CALM" for m in FAM})
    rows = []
    for reg in REGIMES:
        for m in METHODS:
            d = df[df.regime.eq(reg) & df.disp.eq(m)]
            rows.append(
                dict(
                    regime=reg,
                    method=m,
                    n_cells=d.groupby(["dgp", "n_R", "n_O"]).ngroups,
                    **metrics(d),
                )
            )
    bias = pd.DataFrame(rows)
    main = df[
        df.bias.isin(["none", "baseline-only"])
        | (df.bias.eq("comparative") & df.bias_strength.eq(0.2))
    ]
    rows = []
    for b in ["none", "baseline-only", "comparative"]:
        for reg in ["shared-only", "balanced", "mismatch-heavy"]:
            for m in METHODS:
                d = main[main.bias.eq(b) & main.dgp.eq(reg) & main.disp.eq(m)]
                rows.append(dict(bias=b, dgp=reg, method=m, **metrics(d)))
    alignment = pd.DataFrame(rows)
    for name, frame, keys in [
        ("by_bias_regime", bias, ["regime", "method"]),
        ("by_alignment_regime", alignment, ["bias", "dgp", "method"]),
    ]:
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
        if args.reference_dir:
            # Match the decimal parser used by the source aggregation script.
            serialized = pd.read_csv(args.out_dir / f"{name}.csv")
            expected = pd.read_csv(args.reference_dir / f"{name}.csv")
            if not compare_frames(serialized, expected, keys)["exact"]:
                raise AssertionError(f"{name} did not reproduce")
    rows = [
        f"{r.regime} & {r.method} & {r.coverage_90:.3f} & {r.ci_width:.3f} & {r.cate_rmse:.3f} & {r.neg_transfer:.3f} & "
        + (f"{r.coverage_se:.4f}" if r.method == "B-CALM" else "--")
        for r in bias.itertuples()
        if r.regime != "comparative b=0"
    ]
    write_table(
        args.table_dir / "by_regime.tex",
        "Results by bias regime. The OSCAR/CALM family averages its three members, which agree within $10^{-5}$ on all saved metrics. SE is the B-CALM coverage standard error over seed-cell rows.",
        "tab:by-regime",
        "llrrrrr",
        "Bias & Method & Coverage & Width & RMSE & NT & SE",
        rows,
    )
    rows = []
    for (b, reg), g in alignment.groupby(["bias", "dgp"], sort=False):
        if b == "baseline-only":
            continue
        g = g.set_index("method")
        bc = g.loc["B-CALM"]
        rct = g.loc["RCT-only"]
        p = g.loc["pooled"]
        rows.append(
            f"{b} & {reg} & {bc.coverage_90:.3f} & {bc.ci_width:.3f} & {bc.ci_width / rct.ci_width:.3f} & {bc.cate_rmse:.3f} & {bc.neg_transfer:.3f} & {p.coverage_90:.3f} & {rct.ci_width:.3f}"
        )
    write_table(
        args.table_dir / "alignment_ladder.tex",
        "Alignment ladder. B-CALM coverage, width, width relative to RCT-only, RMSE, and negative transfer; pooled coverage and RCT-only width are included for comparison. Comparative bias has strength 0.2.",
        "tab:alignment-ladder",
        "llrrrrrrr",
        "Bias & Regime & Cov. & Width & Ratio & RMSE & NT & Pool cov. & RCT width",
        rows,
    )
    precision(df, args.out_dir, args.table_dir, args.reference_dir)
    log(bias.to_string(index=False))
    log(alignment.to_string(index=False))
    log(f"wall_clock_seconds={time.time() - start:.6f}")


if __name__ == "__main__":
    main()
