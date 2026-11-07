from __future__ import annotations

import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils import RESULTS_DIR, TABLES_DIR, Timer, ensure_dirs, log, record_wallclock


def pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def tex_text(value: object) -> str:
    return str(value).replace("_", "\\_")


def wrap_cell(text: str, *, bold: bool = False, underline: bool = False) -> str:
    out = text
    if underline:
        out = f"\\underline{{{out}}}"
    if bold:
        out = f"\\textbf{{{out}}}"
    return out


def table_header(caption: str, label: str) -> list[str]:
    return ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]


def table_footer() -> list[str]:
    return ["\\end{table}", ""]


def write_saturation() -> None:
    df = pd.read_csv(RESULTS_DIR / "saturation.csv")
    sub = df[(df["method"] == "B-CALM") & (df["n_O"] == df["n_O"].max())].copy()
    sub["sigma_delta"] = sub["sigma_delta"].astype(str)
    agg = (
        sub.groupby(["n_R", "sigma_delta"], as_index=False)
        .agg(
            ess_emp=("ess_O_empirical", "mean"),
            ess_theory=("ess_O_theory", "mean"),
            width=("ci_width_mean", "mean"),
            coverage=("coverage_90", "mean"),
        )
        .sort_values(["n_R", "sigma_delta"])
    )
    lines = table_header(
        "Saturation diagnostic at the largest observational sample size. Empirical $\\ess^O$ is computed from posterior precision after subtracting the RCT contribution.",
        "tab:saturation",
    )
    lines += ["\\begin{tabular}{ccrrrr}", "\\toprule", "$n_R$ & $\\sigma_\\Delta$ & Emp. $\\ess^O$ & Theory $\\ess^O$ & Width & Coverage \\\\", "\\midrule"]
    for _, row in agg.iterrows():
        lines.append(
            f"{int(row.n_R)} & {row.sigma_delta} & {row.ess_emp:.1f} & {row.ess_theory:.1f} & {row.width:.3f} & {row.coverage:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"] + table_footer()
    (TABLES_DIR / "saturation.tex").write_text("\n".join(lines))


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    try:
        if np.allclose(a, b):
            return 1.0
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def write_main() -> None:
    df = pd.read_csv(RESULTS_DIR / "main.csv")
    if "bias_strength" not in df.columns:
        df["bias_strength"] = np.where(df["bias"].eq("comparative"), 0.2, 0.0)
    df["bias_strength"] = df["bias_strength"].astype(float)
    # R-OSCAR, MR-OSCAR, and CALM coincide to within Monte Carlo noise in this
    # harness; report them as a single OSCAR/CALM row (MR-OSCAR rows).
    df = df[~df["method"].isin(["R-OSCAR", "CALM"])].copy()
    df["method"] = df["method"].replace({"MR-OSCAR": "OSCAR/CALM"})
    # The full main-results table cannot fit on a single page,
    # so we emit it as a longtable that page-breaks naturally. The synthetic
    # main body of the paper references a summary subset (tables/main_summary.tex);
    # this longtable is the appendix-only full reproduction.
    caption = (
        "Full main simulation results over 50 seeds per cell. Within each cell we bold the method "
        "with the lowest negative-transfer rate and underline the method with coverage closest to "
        "the 90\\% target from below. Pooled and CF (the Wager causal forest) achieve lower RMSE "
        "under no bias because pooling is variance-optimal there, but their coverage collapses "
        "under comparative bias. R-OSCAR, MR-OSCAR, and CALM coincide to within Monte Carlo noise "
        "in this harness and are merged into a single OSCAR/CALM row; the family sits just below "
        "nominal coverage on average while declining to borrow under bias. The Hahn Bayesian "
        "causal forest (BCF) is run only in the real-data analysis. Bias-limited width saturation "
        "is guaranteed by Corollary~\\ref{thm:scalar}."
    )
    lines = [
        "{\\tiny",
        "\\setlength\\tabcolsep{3pt}",
        "\\begin{longtable}{lllrlrrrr}",
        f"\\caption{{{caption}}}",
        "\\label{tab:main} \\\\",
        "\\toprule",
        "DGP & Bias & $b_\\Delta$ & $n^r$ & Method & Coverage & Neg. transfer & RMSE & Width \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{9}{l}{\\emph{(continued)}} \\\\",
        "\\toprule",
        "DGP & Bias & $b_\\Delta$ & $n^r$ & Method & Coverage & Neg. transfer & RMSE & Width \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{9}{r}{\\emph{continued on next page}} \\\\",
        "\\endfoot",
        "\\endlastfoot",
    ]
    summary = (
        df.groupby(["dgp", "bias", "bias_strength", "n_R", "method"])
        .agg(
            rmse_mean=("cate_rmse", "mean"),
            rmse_std=("cate_rmse", "std"),
            cov_mean=("coverage_90", "mean"),
            width_mean=("ci_width", "mean"),
            neg_mean=("neg_transfer", "mean"),
        )
        .reset_index()
    )
    method_order = ["B-CALM", "RCT-only", "BCF", "OSCAR/CALM", "pooled", "OS-only"]
    for dgp in sorted(summary["dgp"].unique()):
        dgp_summary = summary[summary["dgp"] == dgp]
        for (bias, strength, nr), cell in dgp_summary.groupby(["bias", "bias_strength", "n_R"], sort=True):
            cell = cell.copy()
            cell["score"] = cell["cov_mean"] / np.maximum(cell["width_mean"], 1.0e-8)
            eligible = cell[cell["cov_mean"] <= 0.9000001]
            if len(eligible):
                best_cov_idx = (0.9 - eligible["cov_mean"]).idxmin()
            else:
                best_cov_idx = (cell["cov_mean"] - 0.9).abs().idxmin()
            best_neg_idx = cell.sort_values(["neg_mean", "cov_mean"], ascending=[True, False]).index[0]
            cell["method_rank"] = cell["method"].map({m: i for i, m in enumerate(method_order)}).fillna(99)
            cell = cell.sort_values(["score", "method_rank"], ascending=[False, True])
            strength_text = f"{strength:.1f}" if bias == "comparative" else ""
            for _, row in cell.iterrows():
                is_best_neg = row.name == best_neg_idx
                is_best_cov = row.name == best_cov_idx
                method = wrap_cell(tex_text(row.method), bold=is_best_neg, underline=is_best_cov)
                coverage = wrap_cell(f"{row.cov_mean:.3f}", bold=is_best_neg, underline=is_best_cov)
                neg = wrap_cell(f"{row.neg_mean:.3f}", bold=is_best_neg)
                rmse = wrap_cell(pm(row.rmse_mean, row.rmse_std), bold=is_best_neg)
                width = wrap_cell(f"{row.width_mean:.3f}", bold=is_best_neg)
                lines.append(
                    f"{tex_text(dgp)} & {tex_text(bias)} & {strength_text} & {int(nr)} & {method} & {coverage} & {neg} & {rmse} & {width} \\\\"
                )
        avg = dgp_summary.groupby("method", as_index=False).agg(cov=("cov_mean", "mean"), neg=("neg_mean", "mean"))
        avg["method_rank"] = avg["method"].map({m: i for i, m in enumerate(method_order)}).fillna(99)
        avg = avg.sort_values("method_rank")
        cov_text = "; ".join(f"{tex_text(row['method'])} {row['cov']:.3f}" for _, row in avg.iterrows())
        neg_text = "; ".join(f"{tex_text(row['method'])} {row['neg']:.3f}" for _, row in avg.iterrows())
        lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{9}}{{l}}{{Avg coverage: {cov_text}}} \\\\")
        lines.append(f"\\multicolumn{{9}}{{l}}{{Avg neg-transfer: {neg_text}}} \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{longtable}", "}"]
    (TABLES_DIR / "main.tex").write_text("\n".join(lines))


def write_semi() -> None:
    df = pd.read_csv(RESULTS_DIR / "semi.csv")
    agg = df.groupby("method").agg(rmse_mean=("cate_rmse", "mean"), rmse_std=("cate_rmse", "std"), cov_mean=("coverage_90", "mean")).reset_index()
    lines = table_header(
        "STAR-like semi-synthetic benchmark with known potential outcomes "
        "(50 replicates; construction in Appendix~\\ref{app:semi-construction}). "
        "MR-OSCAR represents the OSCAR/CALM family, whose members coincide in this harness; "
        "it is the family member run in this study.",
        "tab:semi",
    )
    lines += ["\\begin{tabular}{lrr}", "\\toprule", "Method & CATE RMSE & 90\\% coverage \\\\", "\\midrule"]
    for _, row in agg.sort_values("rmse_mean").iterrows():
        lines.append(f"{tex_text(row.method)} & {pm(row.rmse_mean, row.rmse_std)} & {row.cov_mean:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"] + table_footer()
    (TABLES_DIR / "semi.tex").write_text("\n".join(lines))


def write_ablation() -> None:
    df = pd.read_csv(RESULTS_DIR / "ablation.csv")
    agg = (
        df.groupby("variant")
        .agg(
            rmse_mean=("cate_rmse", "mean"),
            rmse_std=("cate_rmse", "std"),
            cov_mean=("coverage_90", "mean"),
            width_mean=("ci_width", "mean"),
            neg_mean=("neg_transfer", "mean"),
        )
        .reset_index()
    )
    lines = table_header(
        "B-CALM ablations on the balanced comparative-bias setting. Coverage and negative-transfer metrics are shown before RMSE to emphasize calibrated borrowing.",
        "tab:ablation",
    )
    lines += ["\\begin{tabular}{lrrrr}", "\\toprule", "Variant & Coverage & Neg. transfer & RMSE & Width \\\\", "\\midrule"]
    # No per-cell emphasis; Full B-CALM is listed first.
    variant_order = ["Full B-CALM", "No b0", "No b_delta"]
    agg["variant_rank"] = agg["variant"].map({v: i for i, v in enumerate(variant_order)}).fillna(99)
    for _, row in agg.sort_values("variant_rank").iterrows():
        lines.append(
            f"{tex_text(row.variant)} & {row.cov_mean:.3f} & {row.neg_mean:.3f} & {pm(row.rmse_mean, row.rmse_std)} & {row.width_mean:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"] + table_footer()
    (TABLES_DIR / "ablation.tex").write_text("\n".join(lines))


MAIN_SUMMARY_CAPTION = (
    "Synthetic factorial study: metric averages over the full condition grid ($24$ cells $\\times$ $50$ replicates; "
    "Appendix~\\ref{app:dgp-synthetic}). Cov.\\ and Width refer to pointwise 90\\% credible intervals evaluated on "
    "trial-distribution covariates; NT is the negative-transfer rate relative to RCT-only. R-OSCAR, MR-OSCAR, and CALM "
    "are reported separately. Per-cell results: Appendix~\\ref{app:main-table}, Table~\\ref{tab:main}."
)
MAIN_SUMMARY_ORDER = ["B-CALM", "R-OSCAR", "MR-OSCAR", "CALM", "RCT-only", "pooled", "OS-only", "CF"]


def write_main_summary(out_dir: Path = TABLES_DIR) -> None:
    """Write the factorial summary with a separate row for each method."""
    df = pd.read_csv(RESULTS_DIR / "main.csv")
    summary = df.groupby("method")[["cate_rmse", "coverage_90", "ci_width", "neg_transfer"]].mean()
    lines = table_header(MAIN_SUMMARY_CAPTION, "tab:main_summary")
    lines += ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Method   & RMSE  & Cov. & Width & NT \\\\", "\\midrule"]
    for method in MAIN_SUMMARY_ORDER:
        if method not in summary.index:
            continue
        r = summary.loc[method]
        lines.append(f"{method:<18}& {r.cate_rmse:.3f} & {r.coverage_90:.3f} & {r.ci_width:.3f} & {r.neg_transfer:.3f} " + r"\\")
    lines += ["\\bottomrule", "\\end{tabular}"] + table_footer()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main_summary.tex").write_text("\n".join(lines))
    log(f"main_summary: {len(summary)} methods")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        timer = Timer("main summary")
        write_main_summary(args.out_dir)
        log(f"wall_clock_seconds={timer.finish():.6f}")
        return
    if args.out_dir != TABLES_DIR:
        raise ValueError("Use --summary-only with a custom output directory")
    ensure_dirs()
    timer = Timer("tables")
    write_saturation()
    write_main()
    write_semi()
    write_ablation()
    seconds = timer.finish()
    record_wallclock("tables", seconds)
    log(f"wrote tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
