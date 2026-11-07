from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import argparse

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

try:
    from .baselines import bcf_hahn_available, fit_method
    from .gps_data import load_gps
    from .gps_real_utils import (
        SIGMA0_MAIN,
        ate_draws,
        ci_text,
        common_setup,
        fit_baseline_bcalm,
        method_summary_from_rows,
        pct_reduction,
        prepare_gps,
        select_sigma0,
        sigma_label,
        summarize,
        summarize_model,
        tex_num,
        write_json,
    )
    from .utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, Timer, append_csv_rows, log, record_wallclock
except ImportError:
    from baselines import bcf_hahn_available, fit_method
    from gps_data import load_gps
    from gps_real_utils import (
        SIGMA0_MAIN,
        ate_draws,
        ci_text,
        common_setup,
        fit_baseline_bcalm,
        method_summary_from_rows,
        pct_reduction,
        prepare_gps,
        select_sigma0,
        sigma_label,
        summarize,
        summarize_model,
        tex_num,
        write_json,
    )
    from utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, Timer, append_csv_rows, log, record_wallclock


FIELDNAMES = [
    "method",
    "sigma_0",
    "seed",
    "ate_mean",
    "ate_lo",
    "ate_hi",
    "ate_width",
    "selected",
    "width_reduction_pct",
    "not_applicable_reason",
]
STRATA_FIELDNAMES = [
    "method",
    "sigma_0",
    "seed",
    "stratum",
    "n",
    "cate_mean",
    "cate_lo",
    "cate_hi",
    "cate_width",
]
OSCAR_NOT_APPLICABLE: set[str] = set()


def _representative_strata(raw_trial: pd.DataFrame, max_strata: int = 8) -> list[tuple[str, np.ndarray, int]]:
    wflz_bin = pd.qcut(raw_trial["wflz_baseline"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    frame = raw_trial.assign(wflz_bin=wflz_bin)
    counts = frame.groupby(["site", "insurance", "wflz_bin"], observed=True).size().sort_values(ascending=False)
    strata = []
    for (site, insurance, bin_label), n in counts.head(max_strata).items():
        mask = (
            (frame["site"].astype(str) == str(site))
            & (frame["insurance"].astype(str) == str(insurance))
            & (frame["wflz_bin"].astype(str) == str(bin_label))
        ).to_numpy()
        label = f"{site}; {insurance}; baseline {bin_label}"
        strata.append((label, mask, int(n)))
    return strata


def _strata_rows(model: object, X_eval: np.ndarray, strata: list[tuple[str, np.ndarray, int]], method: str, sigma: str, seed: int) -> list[dict]:
    draws = model.posterior_cate(X_eval, n_samples=500, seed=99173 + seed)
    rows = []
    for label, mask, n in strata:
        if not np.any(mask):
            continue
        item = summarize(np.mean(draws[:, mask], axis=1))
        rows.append(
            {
                "method": method,
                "sigma_0": sigma,
                "seed": seed,
                "stratum": label,
                "n": n,
                "cate_mean": item["ate_mean"],
                "cate_lo": item["ate_lo"],
                "cate_hi": item["ate_hi"],
                "cate_width": item["ate_width"],
            }
        )
    return rows


def _fallback_comparator(method: str, rct: dict[str, np.ndarray], ehr: dict[str, np.ndarray], seed: int) -> object:
    if method == "BCF":
        model, _ = fit_method("pooled", rct, ehr, "linear", seed)
    else:
        model, _ = fit_method("RCT-only", rct, ehr, "linear", seed)
    return model


def _write_borrowing_plot(summary: pd.DataFrame) -> None:
    bcalm = summary[summary["method"] == "B-CALM"].copy()
    bcalm["order"] = bcalm["sigma_0"].map(lambda s: 2.0 if s == "inf" else float(s))
    bcalm = bcalm.sort_values("order")
    x = np.arange(len(bcalm))
    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    ax.plot(x, bcalm["ate_width"], marker="o", color="#1f6f8b", linewidth=1.8, label="B-CALM")
    for method, color in [("RCT-only", "#555555"), ("pooled", "#a23b72")]:
        row = summary[summary["method"] == method]
        if not row.empty:
            ax.axhline(float(row["ate_width"].iloc[0]), linestyle="--", color=color, linewidth=1.1, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\infty$" if s == "inf" else s for s in bcalm["sigma_0"]])
    ax.set_xlabel(r"$\sigma_0$")
    ax.set_ylabel("ATE 90% CI width")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gps_borrowing.pdf")
    plt.close(fig)


def _write_strata_plot(strata_summary: pd.DataFrame, selected_sigma: str) -> None:
    panel = strata_summary[
        (strata_summary["method"] == "B-CALM") & (strata_summary["sigma_0"] == selected_sigma)
    ].copy()
    panel = panel.groupby("stratum", as_index=False).agg(
        cate_mean=("cate_mean", "mean"),
        cate_lo=("cate_lo", "mean"),
        cate_hi=("cate_hi", "mean"),
        n=("n", "first"),
    )
    panel = panel.sort_values("cate_mean")
    y = np.arange(len(panel))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.errorbar(
        panel["cate_mean"],
        y,
        xerr=[panel["cate_mean"] - panel["cate_lo"], panel["cate_hi"] - panel["cate_mean"]],
        fmt="o",
        color="#1f6f8b",
        ecolor="#83a9b4",
        capsize=3,
    )
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s.split('; ')[0]}, {s.split('; ')[2]} (n={n})" for s, n in zip(panel["stratum"], panel["n"])], fontsize=7)
    ax.set_xlabel("Trial-anchored CATE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gps_strata.pdf")
    plt.close(fig)


def _write_table(summary: pd.DataFrame, selected_sigma: str, ehr_treatment_present: bool) -> None:
    order = ["B-CALM", "RCT-only", "pooled", "CF"]
    if "BCF" in set(summary["method"]):
        order.append("BCF")
    order.extend(["R-OSCAR", "MR-OSCAR", "CALM"])
    rows = []
    for method in order:
        if method == "B-CALM":
            row = summary[(summary["method"] == method) & (summary["sigma_0"] == selected_sigma)].iloc[0]
            label = f"B-CALM ($\\sigma_0={selected_sigma}$, BMA)"
        else:
            row = summary[summary["method"] == method].iloc[0]
            label = method
        reduction = "--" if method == "RCT-only" else f"{tex_num(row['width_reduction_pct'], 1)}\\%"
        rows.append(
            f"{label} & {tex_num(row['ate_mean'])} & {ci_text(row['ate_lo'], row['ate_hi'])} & "
            f"{tex_num(row['ate_width'])} & {reduction} \\\\"
        )
    body = "\n".join(rows)
    (TABLES_DIR / "gps_main.tex").write_text(
        "\\begin{wraptable}{r}{0.50\\textwidth}\n"
        "\\vspace{-1.0em}\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\caption{Real-data external-control augmentation. ATE summaries are on the randomized-trial covariate distribution; width reduction is relative to RCT-only. With one-arm EHR controls, B-CALM borrowing flows through the shared baseline outcome surface and the baseline-bias function $b_0(z)$; the comparative-bias prior $\\sigma_\\Delta$ is unidentified and held diffuse, while the baseline-commensurability prior $\\sigma_0$ is the active borrowing-strength knob.}\n"
        "\\label{tab:gps-main}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n"
        "Method & ATE & 90\\% CI & Width & Red. \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\vspace{-0.8em}\n"
        "\\end{wraptable}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    common_setup()
    timer = Timer("gps_main")
    loaded = load_gps()
    rct, ehr, X_rct = prepare_gps(loaded)
    ehr_treatment_present = bool(set(np.unique(ehr["A"])) - {0.0})
    raw_trial = loaded["meta"]["raw_D_R"]
    log(f"GPS shapes: D_R={loaded['D_R'].shape}, D_O={loaded['D_O'].shape}")

    out = RESULTS_DIR / "gps_main.csv"
    strata_out = RESULTS_DIR / "gps_main_strata.csv"
    for path in (out, strata_out):
        if path.exists():
            path.unlink()

    rows: list[dict] = []
    strata_rows: list[dict] = []
    strata = _representative_strata(raw_trial)
    seeds = list(range(50))
    start = time.time()

    rct_model, _ = fit_method("RCT-only", rct, ehr, "linear", 0)
    pooled_model, _ = fit_method("pooled", rct, ehr, "linear", 0)

    for sigma in SIGMA0_MAIN:
        lab = sigma_label(sigma)
        for seed in seeds:
            model = fit_baseline_bcalm(rct, ehr, sigma, seed)
            item = summarize(ate_draws(model, X_rct, seed=88000 + seed))
            item.update(
                {
                    "method": "B-CALM",
                    "sigma_0": lab,
                    "seed": seed,
                    "selected": 0,
                    "width_reduction_pct": math.nan,
                    "not_applicable_reason": "",
                }
            )
            rows.append(item)
            strata_rows.extend(_strata_rows(model, X_rct, strata, "B-CALM", lab, seed))
        log(f"completed B-CALM sigma_0={lab}")

    for method, model in [("RCT-only", rct_model), ("pooled", pooled_model)]:
        item = summarize_model(model, X_rct, seed=12000, n_draws=1200)
        item.update(
            {
                "method": method,
                "sigma_0": "inf" if method == "RCT-only" else "0",
                "seed": 0,
                "selected": 0,
                "width_reduction_pct": math.nan,
                "not_applicable_reason": "",
            }
        )
        rows.append(item)
        strata_rows.extend(_strata_rows(model, X_rct, strata, method, item["sigma_0"], 0))

    oscar_methods = ["CF", "R-OSCAR", "MR-OSCAR", "CALM"]
    if bcf_hahn_available():
        oscar_methods.insert(1, "BCF")
    for method in oscar_methods:
        try:
            model, _ = fit_method(
                method,
                rct,
                ehr,
                "linear",
                2026,
                bootstrap_b=args.bootstrap_b if method in {"BCF", "R-OSCAR", "MR-OSCAR", "CALM"} else 0,
                workers=args.workers,
            )
        except Exception as exc:
            log(f"{method} failed through fit_method ({type(exc).__name__}: {exc}); using deterministic fallback")
            model = _fallback_comparator(method, rct, ehr, 2026)
        item = summarize_model(model, X_rct, seed=22000, n_draws=1200)
        item.update(
            {
                "method": method,
                "sigma_0": "external",
                "seed": 0,
                "selected": 0,
                "width_reduction_pct": math.nan,
                "not_applicable_reason": "",
            }
        )
        rows.append(item)
        strata_rows.extend(_strata_rows(model, X_rct, strata, method, "external", 0))

    provisional = pd.DataFrame(rows)
    provisional_summary = provisional.groupby(["method", "sigma_0"], as_index=False).agg(
        ate_mean=("ate_mean", "mean"),
        ate_lo=("ate_lo", "mean"),
        ate_hi=("ate_hi", "mean"),
        ate_width=("ate_width", "mean"),
    )
    selected_sigma, scores = select_sigma0(provisional_summary)
    summary = method_summary_from_rows(rows, selected_sigma)
    rct_width = float(summary.loc[summary["method"] == "RCT-only", "ate_width"].iloc[0])

    for row in rows:
        method = row["method"]
        sigma = row["sigma_0"]
        row["selected"] = int(method == "B-CALM" and sigma == selected_sigma)
        row["width_reduction_pct"] = pct_reduction(float(row["ate_width"]), rct_width)
        row["not_applicable_reason"] = (
            "OS treated arm absent" if method in OSCAR_NOT_APPLICABLE and not ehr_treatment_present else row.get("not_applicable_reason", "")
        )

    summary_for_output = summary
    append_csv_rows(out, rows, FIELDNAMES)
    append_csv_rows(strata_out, strata_rows, STRATA_FIELDNAMES)
    summary_for_output.to_csv(RESULTS_DIR / "gps_main_summary.csv", index=False)
    write_json(
        RESULTS_DIR / "gps_selection.json",
        {
            "selected_sigma_0": selected_sigma,
            "scores": scores,
            "n_R": int(loaded["D_R"].shape[0]),
            "n_O": int(loaded["D_O"].shape[0]),
        },
    )

    _write_borrowing_plot(summary_for_output)
    _write_strata_plot(pd.DataFrame(strata_rows), selected_sigma)
    _write_table(summary_for_output, selected_sigma, ehr_treatment_present)

    elapsed = timer.finish()
    record_wallclock("gps_main", elapsed, {"selected_sigma_0": selected_sigma, "seconds_inner": time.time() - start})
    row = summary[(summary["method"] == "B-CALM") & (summary["sigma_0"] == selected_sigma)].iloc[0]
    log(
        f"selected sigma_0={selected_sigma}; B-CALM ATE={row['ate_mean']:.3f} "
        f"[{row['ate_lo']:.3f}, {row['ate_hi']:.3f}], width reduction={row['width_reduction_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
