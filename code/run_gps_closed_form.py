"""Exact Gaussian functionals for the GPS external-control analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import fit_method
from bcalm import BCalmLinear
from dgp import basis
from gps_data import load_gps
from gps_real_utils import (
    SIGMA0_MAIN,
    SIGMA0_BIAS,
    prepare_gps,
    fit_baseline_bcalm,
    select_sigma0,
    sigma_label,
    ci_text,
    pct_reduction,
)
import run_gps_main as gm
import run_gps_loo as gl
import run_gps_bias as gb
import run_gps_bias_susceptibility as gs
from utils import RESULTS_DIR, TABLES_DIR, FIGURES_DIR, log
from experiment_io import write_table

Z90 = 1.645

# Publication layout (everything outside the numeric table body) of the four
# real-data tables. Only the rows between \midrule and \bottomrule are computed.
PUBLICATION_LAYOUT = {
    "gps_main.tex": (
        r"""\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Real-data external-control augmentation. ATE on the trial covariate distribution; width reduction is relative to RCT-only. B-CALM uses $\sigma_0=0.05$, selected over a $\sigma_0$ grid (Appendix~\ref{app:reproducibility}). With one-arm EHR controls, $\sigma_\Delta$ is held diffuse and $\sigma_0$ is the active borrowing knob. The OSCAR/CALM row merges R-OSCAR, MR-OSCAR, and CALM, which coincide here (identical one-arm external-control calibration). The last column is the bias-susceptibility ratio under the skeptical $\sigma_0=0.6$ prior: 90\% range of the implied B-CALM-calibrated bias risk on each method's ATE, divided by that method's nominal width. A value above $1$ means a method's nominal interval is narrower than the bias it would absorb under a skeptical EHR-control prior.}
\label{tab:gps-main}
\begin{tabular}{@{}lccc@{}}
\toprule
Method & ATE [90\% CI] & Width (Red.) & Skep./w. \\
\midrule
""",
        r"""\bottomrule
\end{tabular}
\end{table}
""",
    ),
    "gps_loo.tex": (
        r"""\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Leave-one-covariate masking in the EHR controls. The final column checks whether the full-model 90\% CI contains the masked posterior mean.}
\label{tab:gps-loo}
\begin{tabular}{@{}lcccc@{}}
\toprule
Mask & ATE & 90\% CI & Width & Hit \\
\midrule
""",
        r"""\bottomrule
\end{tabular}
\end{table}
""",
    ),
    "gps_bias.tex": (
        r"""\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Largest EHR-versus-RCT control-outcome discrepancies by posterior $|b_0(z)|$ under the skeptical commensurability prior $\sigma_0=0.6$. The skeptical prior gives $b_0$ enough room to absorb between-source baseline disagreement so that the diagnostic surfaces the strata where pooling would be most misleading; the trial-anchored ATE in Table~\ref{tab:gps-main} uses the BMA-selected $\sigma_0=0.05$ and yields much smaller posterior $b_0$ shrinkage.}
\label{tab:gps-bias}
\begin{tabular}{@{}p{0.46\textwidth}rcc@{}}
\toprule
Stratum & $n$ & Mean & 90\% CI \\
\midrule
""",
        r"""\bottomrule
\end{tabular}
\end{table}
""",
    ),
    "gps_bias_susceptibility.tex": (
        r"""\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{B-CALM-calibrated hidden-bias susceptibility in the GPS external-control analysis. The uniform column is the ATE change under a $+0.1$ WFLz shift of all EHR control outcomes. Bias-risk ranges are posterior 5th--95th percentiles of $\sum_s L_{m,s}b_{0,s}$ or row-level analogues.}
\label{tab:gps-bias-susceptibility}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lcccccccc@{}}
\toprule
Method & ATE & Nominal 90\% CI & Width & $+0.1$ shift & Selected risk & Sel./width & Skeptical risk & Skep./width \\
\midrule
""",
        r"""\bottomrule
\end{tabular}%
}
\end{table}
""",
    ),
}


def functional(model: BCalmLinear, vector: np.ndarray, sl: slice) -> dict[str, float]:
    mean = float(vector @ model.mean_[sl])
    variance = float(vector @ model.cov_[sl, sl] @ vector)
    sd = math.sqrt(max(variance, 0.0))
    return dict(
        ate_mean=mean,
        ate_sd=sd,
        ate_lo=mean - Z90 * sd,
        ate_hi=mean + Z90 * sd,
        ate_width=2 * Z90 * sd,
    )


def ate(model: BCalmLinear, x: np.ndarray) -> dict[str, float]:
    return functional(model, basis(x, model.basis_kind).mean(0), model.tau_slice)


def compare_cells(
    name: str, new: pd.DataFrame, old: pd.DataFrame, keys: list[str], ledger: list[dict]
) -> None:
    merged = new.merge(old, on=keys, suffixes=("_exact", "_mc"))
    for col in new.columns:
        if col in keys or col not in old or not pd.api.types.is_numeric_dtype(new[col]):
            continue
        if not pd.api.types.is_numeric_dtype(old[col]):
            continue
        for _, row in merged.iterrows():
            a, b = row[col + "_exact"], row[col + "_mc"]
            ledger.append(
                dict(
                    source=name,
                    configuration="; ".join(f"{k}={row[k]}" for k in keys),
                    quantity=col,
                    closed_form=a,
                    old_mc=b,
                    abs_difference=abs(float(a) - float(b)),
                )
            )
    differences = {
        c: float((merged[c + "_exact"] - merged[c + "_mc"]).abs().max())
        for c in ["ate_mean", "ate_lo", "ate_hi", "ate_width"]
        if c in new and c in old
    }
    log(f"{name}: max absolute differences {differences}")


def normalized_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_DIR / name, keep_default_na=False)
    if "sigma_0" in frame:
        frame["sigma_0"] = frame.sigma_0.map(
            lambda v: (
                sigma_label(float(v)) if str(v) not in ["external", ""] else str(v)
            )
        )
    return frame


def preserve_table_layout(name: str, body: str | None = None) -> None:
    """Wrap a computed numeric body in the publication layout of ``name``."""
    prefix, suffix = PUBLICATION_LAYOUT[name]
    target = gm.TABLES_DIR / name
    if body is None:
        generated = target.read_text()
        body = generated.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
    target.write_text(prefix + body + suffix)


def main_table_body(summary: pd.DataFrame, risk: pd.DataFrame, selected: str) -> str:
    rows = []
    for label, method in [
        ("B-CALM", "B-CALM"),
        ("RCT-only", "RCT-only"),
        ("pooled", "pooled"),
        ("CF", "CF"),
        ("BCF", "BCF"),
        ("OSCAR/CALM", "MR-OSCAR"),
    ]:
        panel = summary[summary.method.eq(method)]
        if method == "B-CALM":
            panel = panel[panel.sigma_0.eq(selected)]
        row = panel.iloc[0]
        width = f"{row.ate_width:.3f}"
        if method != "RCT-only":
            width += f"\\,({row.width_reduction_pct:.1f}\\%)"
        sk = risk[risk.method.eq(method) & risk.b0_posterior.eq("skeptical_sigma0_0.6")]
        if method == "B-CALM":
            sk = sk[sk.sigma_0.eq(selected)]
        ratio = "--" if sk.empty else f"${sk.iloc[0].bias_risk_to_width:.2f}$"
        rows.append(
            f"{label} & ${row.ate_mean:.3f}\\,[{row.ate_lo:.3f},{row.ate_hi:.3f}]$ & ${width}$ & {ratio} "
            + r"\\"
        )
    return "\n".join(rows) + "\n"


def susceptibility(
    rct: dict, ehr: dict, x: np.ndarray, loaded: dict, nominal: pd.DataFrame
) -> pd.DataFrame:
    result = normalized_csv("gps_bias_susceptibility.csv")
    strata = normalized_csv("gps_bias_susceptibility_strata.csv")
    groups = gs._raw_b0_strata(loaded["meta"]["raw_D_O"]).astype(str)
    phi = basis(ehr["X"], "linear")
    calibrations = {
        name: fit_baseline_bcalm(rct, ehr, sigma, 0)
        for name, sigma in [
            ("selected_sigma0_0.05", 0.05),
            ("skeptical_sigma0_0.6", 0.6),
        ]
    }
    for i, row in result.iterrows():
        method = row["method"]
        sigma = row["sigma_0"]
        label = row["method_label"]
        if method in ["B-CALM", "RCT-only", "pooled"]:
            model = gs._fit_linear_method(method, float(sigma), rct, ehr, 0)
            item = ate(model, x)
            for col, src in [
                ("nominal_ate", "ate_mean"),
                ("nominal_lo", "ate_lo"),
                ("nominal_hi", "ate_hi"),
                ("nominal_width", "ate_width"),
            ]:
                result.loc[i, col] = item[src]
            weights = gs._linear_weights_for_method(method, model, ehr, x)
            vector = phi.T @ weights
        else:
            g = strata[strata.method_label.eq(label)]
            vector = np.zeros(phi.shape[1])
            for row_s in g.itertuples():
                mask = groups.eq(row_s.stratum).to_numpy()
                vector += float(row_s.slope_central) * phi[mask].mean(0)
        cal = calibrations[row["b0_posterior"]]
        risk = functional(cal, vector, slice(2 * cal.k_, 3 * cal.k_))
        mean, sd = risk["ate_mean"], risk["ate_sd"]
        lo, hi = risk["ate_lo"], risk["ate_hi"]
        if sd == 0:
            abs_median = abs(mean)
        else:
            abs_median = brentq(
                lambda t: norm.cdf((t - mean) / sd) - norm.cdf((-t - mean) / sd) - 0.5,
                0,
                abs(mean) + 10 * sd,
            )
        width = float(result.loc[i, "nominal_width"])
        updates = dict(
            bias_risk_mean=mean,
            bias_risk_q05=lo,
            bias_risk_q95=hi,
            bias_risk_abs_median=abs_median,
            bias_risk_to_width=(hi - lo) / width,
            bias_expanded_lo=float(result.loc[i, "nominal_lo"]) + lo,
            bias_expanded_hi=float(result.loc[i, "nominal_hi"]) + hi,
            n_bias_draws=0,
        )
        for key, value in updates.items():
            result.loc[i, key] = value
        result.loc[i, "nominal_ci"] = ci_text(
            result.loc[i, "nominal_lo"], result.loc[i, "nominal_hi"]
        )
        result.loc[i, "bias_risk_90_range"] = ci_text(lo, hi)
        result.loc[i, "bias_expanded_interval"] = ci_text(
            updates["bias_expanded_lo"], updates["bias_expanded_hi"]
        )
    log(
        "Susceptibility: exact correlated b0 coefficient functionals; nonlinear method slopes and nominal intervals read unchanged from existing aggregate CSVs."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    start = time.time()
    for p in [args.out_dir, args.table_dir, args.figure_dir]:
        p.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    for module in [gm, gl, gb]:
        module.TABLES_DIR = args.table_dir
        module.FIGURES_DIR = args.figure_dir
    gs.TABLE_OUT = args.table_dir / "gps_bias_susceptibility.tex"
    gs.FIGURE_OUT = args.figure_dir / "gps_bias_susceptibility.pdf"
    loaded = load_gps()
    rct, ehr, x = prepare_gps(loaded)
    log(
        f"n_R={len(x)}, n_O={len(ehr['Y'])}; linear basis; sigma_delta=inf; Gaussian functional mean/variance; z={Z90}; no posterior sampling"
    )
    rct_model, _ = fit_method("RCT-only", rct, ehr, "linear", 0)
    rct_summary = ate(rct_model, x)
    grids = {
        "paper": SIGMA0_BIAS,
        "fine": list(np.arange(101) / 100) + [2.0, 5.0, 10.0, math.inf],
    }
    cache = {}
    rows = []
    for grid, values in grids.items():
        for sigma in values:
            if sigma not in cache:
                cache[sigma] = ate(fit_baseline_bcalm(rct, ehr, sigma, 0), x)
            item = cache[sigma]
            rows.append(
                dict(
                    grid=grid,
                    method="B-CALM",
                    sigma_0=sigma_label(sigma),
                    **item,
                    width_reduction_pct=pct_reduction(
                        item["ate_width"], rct_summary["ate_width"]
                    ),
                )
            )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.out_dir / "gps_closed_form_sweep.csv", index=False)
    selected, scores = select_sigma0(sweep[sweep.grid.eq("paper")])
    log(f"selected_sigma0={selected}; stays_0.05={selected == '0.05'}; scores={scores}")
    crossing = [s for s, item in cache.items() if item["ate_hi"] >= 0]
    log(
        f"smallest evaluated sigma0 with upper >= 0: {min(crossing) if crossing else None}"
    )
    for sigma in [0.5, 1.0, math.inf]:
        log(f"sigma0={sigma}: upper={cache[sigma]['ate_hi']:.12g}")
    ledger = []
    summary = normalized_csv("gps_main_summary.csv")
    old = summary.copy()
    for i, row in summary.iterrows():
        method = row["method"]
        if method == "B-CALM":
            item = cache[float(row["sigma_0"])]
        elif method == "RCT-only":
            item = rct_summary
        elif method == "pooled":
            model, _ = fit_method("pooled", rct, ehr, "linear", 0)
            item = ate(model, x)
        else:
            continue
        for key, value in item.items():
            summary.loc[i, key] = value
    summary["width_reduction_pct"] = summary.ate_width.map(
        lambda w: pct_reduction(w, rct_summary["ate_width"])
    )
    summary["selected"] = summary.method.eq("B-CALM") & summary.sigma_0.eq(selected)
    compare_cells("gps_main_summary", summary, old, ["method", "sigma_0"], ledger)
    compare_cells(
        "gps_bias_sigma_sweep",
        sweep[sweep.grid.eq("paper")],
        normalized_csv("gps_bias_sigma_sweep.csv"),
        ["sigma_0"],
        ledger,
    )
    summary.to_csv(args.out_dir / "gps_main_summary.csv", index=False)
    strata = []
    for sigma in SIGMA0_MAIN:
        model = fit_baseline_bcalm(rct, ehr, sigma, 0)
        for label, mask, n in gm._representative_strata(loaded["meta"]["raw_D_R"]):
            item = ate(model, x[mask])
            strata.append(
                dict(
                    method="B-CALM",
                    sigma_0=sigma_label(sigma),
                    stratum=label,
                    n=n,
                    **{k.replace("ate_", "cate_"): v for k, v in item.items()},
                )
            )
    strata = pd.DataFrame(strata)
    strata.to_csv(args.out_dir / "gps_closed_form_strata.csv", index=False)
    oldstr = (
        normalized_csv("gps_main_strata.csv")
        .groupby(["method", "sigma_0", "stratum"], as_index=False)
        .mean(numeric_only=True)
    )
    compare_cells(
        "gps_main_strata", strata, oldstr, ["method", "sigma_0", "stratum"], ledger
    )
    loo = []
    full = cache[float(selected)]
    for name in ["full"] + gl.MASKS:
        if name == "full":
            item = full.copy()
        else:
            rm, om, xm = prepare_gps(gl._masked_loaded(loaded, name))
            item = ate(fit_baseline_bcalm(rm, om, float(selected), 0), xm)
        loo.append(
            dict(
                scenario=name,
                sigma_0=selected,
                **item,
                full_ci_contains_mean=full["ate_lo"]
                <= item["ate_mean"]
                <= full["ate_hi"],
            )
        )
    loo = pd.DataFrame(loo)
    compare_cells(
        "gps_loo", loo, normalized_csv("gps_loo.csv"), ["scenario", "sigma_0"], ledger
    )
    loo.to_csv(args.out_dir / "gps_loo.csv", index=False)
    risk = susceptibility(rct, ehr, x, loaded, summary)
    compare_cells(
        "gps_bias_susceptibility",
        risk,
        normalized_csv("gps_bias_susceptibility.csv"),
        ["method_label", "b0_posterior"],
        ledger,
    )
    risk.to_csv(args.out_dir / "gps_bias_susceptibility.csv", index=False)
    # These discrepancy and profile intervals already use Gaussian closed forms.
    b0 = normalized_csv("gps_bias_strata.csv")
    compare_cells("gps_bias_strata", b0, b0, ["stratum"], ledger)
    pd.DataFrame(ledger).to_csv(
        args.out_dir / "gps_closed_form_tables.csv", index=False
    )
    gm._write_table(summary, selected, False)
    gl._write_table(loo)
    gb._write_table(b0)
    gs._write_table(risk)
    preserve_table_layout("gps_main.tex", main_table_body(summary, risk, selected))
    for name in ["gps_loo.tex", "gps_bias.tex", "gps_bias_susceptibility.tex"]:
        preserve_table_layout(name)
    gm._write_borrowing_plot(summary)
    log("gps_borrowing.pdf: B-CALM, RCT-only, pooled")
    gm._write_strata_plot(strata, selected)
    log("gps_strata.pdf: selected B-CALM, eight trial strata, zero reference")
    gl._write_plot(loo, loo.iloc[0])
    log("gps_loo.pdf: five masks, full-model interval and mean")
    log("gps_bias.pdf is not regenerated here; this script writes aggregate outputs only.")
    gs._write_figure(risk)
    log(
        "gps_bias_susceptibility.pdf: RCT-only, B-CALM, pooled, CF, R-OSCAR, MR-OSCAR, CALM; nominal and skeptical intervals"
    )
    rows = []
    for sigma in SIGMA0_MAIN:
        item = cache[sigma]
        rows.append(
            f"{sigma_label(sigma)} & {item['ate_mean']:.3f} & {ci_text(item['ate_lo'], item['ate_hi'])} & {item['ate_width']:.3f} & {pct_reduction(item['ate_width'], rct_summary['ate_width']):.1f}\\%"
        )
    write_table(
        args.table_dir / "gps_sensitivity.tex",
        "Exact Gaussian trial-population ATE intervals across baseline-bias prior scales. Width reduction is relative to the exact RCT-only interval.",
        "tab:gps-sensitivity",
        "lrrrr",
        r"$\sigma_0$ & ATE & 90\% interval & Width & Reduction",
        rows,
    )
    fine = sweep[sweep.grid.eq("fine")].copy()
    fine["plot_x"] = fine.sigma_0.map(lambda v: 20 if v == "inf" else float(v))
    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    ax.plot(fine.plot_x, fine.ate_mean, label="B-CALM", color="#1f6f8b")
    ax.fill_between(
        fine.plot_x.to_numpy(),
        fine.ate_lo.to_numpy(),
        fine.ate_hi.to_numpy(),
        alpha=0.15,
        color="#1f6f8b",
    )
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(0, 25)
    ax.set_xticks([0, 0.05, 0.1, 0.5, 1, 5, 20])
    ax.set_xticklabels(["0", ".05", ".1", ".5", "1", "5", r"$\infty$"])
    ax.set_xlabel(r"$\sigma_0$")
    ax.set_ylabel("Trial-population ATE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "gps_sensitivity.pdf")
    plt.close(fig)
    log(
        "gps_sensitivity.pdf: exact B-CALM ATE and 90% interval on fine grid, zero reference"
    )
    log(sweep[sweep.grid.eq("paper")].to_string(index=False))
    log(f"wall_clock_seconds={time.time() - start:.6f}")


if __name__ == "__main__":
    main()
