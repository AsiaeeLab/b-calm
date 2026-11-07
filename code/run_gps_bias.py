from __future__ import annotations

import json
import math
import argparse

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gps_data import load_gps
from gps_real_utils import (
    SIGMA0_BIAS,
    b0_group_summary,
    ci_text,
    common_setup,
    fit_baseline_bcalm,
    fit_plain_bcalm,
    prepare_gps,
    sigma_label,
    summarize,
    tex_num,
)
from utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, Timer, append_csv_rows, log, record_wallclock


BIAS_FIELDS = ["stratum", "n", "b0_mean", "b0_lo", "b0_hi", "b0_width", "abs_b0"]
SWEEP_FIELDS = ["sigma_0", "ate_mean", "ate_lo", "ate_hi", "ate_width"]


def _selected_sigma() -> float:
    path = RESULTS_DIR / "gps_selection.json"
    if not path.exists():
        return 0.2
    lab = json.loads(path.read_text())["selected_sigma_0"]
    return math.inf if lab == "inf" else float(lab)


def _b0_groups(raw: pd.DataFrame) -> pd.Series:
    wflz_bin = pd.qcut(raw["wflz_baseline"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    return raw["site"].astype(str) + "; " + raw["insurance"].astype(str) + "; baseline " + wflz_bin.astype(str)


def _ate_summary(model: object, X_rct: np.ndarray) -> dict[str, float]:
    """ATE on the trial covariate distribution from posterior draws.

    For BCalmLinear, posterior_cate returns (n_draws, n_x) draws of tau(x);
    we average over x then take posterior quantiles."""
    draws = model.posterior_cate(X_rct, n_samples=8000, seed=90210)
    ate = draws.mean(axis=1)
    lo, hi = np.quantile(ate, [0.05, 0.95])
    return {
        "ate_mean": float(ate.mean()),
        "ate_lo": float(lo),
        "ate_hi": float(hi),
        "ate_width": float(hi - lo),
    }


def _write_plot(b0: pd.DataFrame, sweep: pd.DataFrame) -> None:
    top = b0.head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    ax = axes[0]
    y = np.arange(len(top))
    ax.errorbar(
        top["b0_mean"],
        y,
        xerr=[top["b0_mean"] - top["b0_lo"], top["b0_hi"] - top["b0_mean"]],
        fmt="o",
        color="#1f6f8b",
        ecolor="#83a9b4",
        capsize=3,
    )
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["stratum"], fontsize=6)
    ax.set_xlabel(r"$b_0(z)$")
    ax.set_title("Baseline discrepancy")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    labels = sweep["sigma_0"].tolist()
    x = np.arange(len(labels))
    ax.plot(x, sweep["ate_mean"], marker="o", linewidth=1.5, color="#1f6f8b")
    ax.fill_between(x, sweep["ate_lo"], sweep["ate_hi"], color="#1f6f8b", alpha=0.12)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\infty$" if s == "inf" else s for s in labels])
    ax.set_xlabel(r"$\sigma_0$")
    ax.set_ylabel("Trial-population ATE")
    ax.set_title("Sensitivity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gps_bias.pdf")
    plt.close(fig)


def _write_table(b0: pd.DataFrame) -> None:
    def short_label(label: str) -> str:
        return (
            str(label)
            .replace("Unavailable/No Payer Specified/Blank (EHR), NA/Null (EHR), or None (GLP)", "No/other payer")
            .replace("baseline ", "base ")
        )

    rows = []
    for row in b0.head(3).itertuples():
        rows.append(f"{short_label(row.stratum)} & {row.n} & {tex_num(row.b0_mean)} & {ci_text(row.b0_lo, row.b0_hi)} \\\\")
    body = "\n".join(rows)
    (TABLES_DIR / "gps_bias.tex").write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\caption{Largest EHR-versus-RCT control-outcome discrepancies by posterior $|b_0(z)|$.}\n"
        "\\label{tab:gps-bias}\n"
        "\\begin{tabular}{@{}p{0.46\\textwidth}rcc@{}}\n"
        "\\toprule\n"
        "Stratum & $n$ & Mean & 90\\% CI \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-b", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    parser.parse_args()
    common_setup()
    timer = Timer("gps_bias")
    selected = _selected_sigma()
    loaded = load_gps()
    rct, ehr, X_rct = prepare_gps(loaded)
    raw_ehr = loaded["meta"]["raw_D_O"]

    plain = fit_plain_bcalm(rct, ehr, selected, seed=2026)
    b0 = b0_group_summary(plain, ehr["X"], _b0_groups(raw_ehr))
    b0_out = RESULTS_DIR / "gps_bias_strata.csv"
    if b0_out.exists():
        b0_out.unlink()
    append_csv_rows(b0_out, b0.to_dict("records"), BIAS_FIELDS)

    sweep_rows = []
    for sigma in SIGMA0_BIAS:
        lab = sigma_label(sigma)
        model = fit_baseline_bcalm(rct, ehr, sigma, seed=2026)
        item = _ate_summary(model, X_rct)
        item["sigma_0"] = lab
        sweep_rows.append(item)
        log(f"completed sigma={lab}")

    sweep_out = RESULTS_DIR / "gps_bias_sigma_sweep.csv"
    if sweep_out.exists():
        sweep_out.unlink()
    append_csv_rows(sweep_out, sweep_rows, SWEEP_FIELDS)
    sweep = pd.DataFrame(sweep_rows)
    _write_plot(b0, sweep)
    _write_table(b0)
    elapsed = timer.finish()
    record_wallclock("gps_bias", elapsed, {"selected_sigma_0": sigma_label(selected)})


if __name__ == "__main__":
    main()
