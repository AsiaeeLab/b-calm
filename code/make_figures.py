from __future__ import annotations

import math
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIGURES_DIR, RESULTS_DIR, Timer, ensure_dirs, log, record_wallclock


def _sigma_sort(value: str) -> float:
    return math.inf if str(value) == "inf" else float(value)


def saturation() -> None:
    df = pd.read_csv(RESULTS_DIR / "saturation.csv")
    df["sigma_delta"] = df["sigma_delta"].astype(str)
    nrs = sorted(df["n_R"].unique())
    fig, axes = plt.subplots(1, len(nrs), figsize=(7.2, 3.0), sharey=True)
    if len(nrs) == 1:
        axes = [axes]
    for ax, nr in zip(axes, nrs):
        sub = df[df["n_R"] == nr]
        for sigma in sorted(sub.loc[sub["method"] == "B-CALM", "sigma_delta"].unique(), key=_sigma_sort):
            g = (
                sub[(sub["method"] == "B-CALM") & (sub["sigma_delta"] == sigma)]
                .groupby("n_O", as_index=False)["ci_width_mean"]
                .mean()
            )
            sigma_txt = "$\\infty$" if str(sigma) == "inf" else str(sigma)
            ax.plot(g["n_O"], g["ci_width_mean"], marker="o", linewidth=1.2, label=f"B-CALM $\\sigma_\\Delta{{=}}${sigma_txt}")
        for method, style in [("pooled", "--"), ("OS-only", ":")]:
            g = sub[sub["method"] == method].groupby("n_O", as_index=False)["ci_width_mean"].mean()
            ax.plot(g["n_O"], g["ci_width_mean"], linestyle=style, linewidth=1.3, label=method)
        floor = sub[
            (sub["method"] == "B-CALM")
            & (sub["sigma_delta"] == "0.2")
            & (sub["n_O"] == sub["n_O"].max())
        ]["ci_width_mean"].mean()
        ax.axhline(floor, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="width plateau ($\\sigma_\\Delta{=}$0.2)")
        ax.set_xscale("log")
        ax.set_title(f"$n^r={nr}$")
        ax.set_xlabel("$n^o$")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Mean 90% CI width")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    log("saturation.pdf: " + repr([ax.get_legend_handles_labels()[1] for ax in axes]))
    fig.savefig(FIGURES_DIR / "saturation.pdf", bbox_inches="tight")
    plt.close(fig)


def calibration() -> None:
    df = pd.read_csv(RESULTS_DIR / "calibration.csv")
    oscar = {"R-OSCAR", "MR-OSCAR", "CALM"}
    df = df.copy()
    df["display_method"] = df["method"].where(~df["method"].isin(oscar), "OSCAR/CALM")
    df = df.drop_duplicates(subset=["display_method", "alpha"])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for method, g in df.groupby("display_method"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g["empirical_coverage"], marker="o", linewidth=1.2, markersize=3, label=method)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal credible level")
    ax.set_ylabel("Empirical coverage")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    log("calibration.pdf: " + repr(ax.get_legend_handles_labels()[1]))
    fig.savefig(FIGURES_DIR / "calibration.pdf")
    plt.close(fig)


def bias_sweep() -> None:
    df = pd.read_csv(RESULTS_DIR / "main.csv")
    if "bias_strength" not in df.columns:
        df["bias_strength"] = np.where(df["bias"].eq("comparative"), 0.2, 0.0)
    df["bias_strength"] = df["bias_strength"].astype(float)
    sub = df[(df["dgp"] == "balanced") & (df["bias"] == "comparative") & (df["n_R"] == df["n_R"].max())].copy()
    # MR-OSCAR, R-OSCAR, and CALM differ by less than 1e-4 at every point in this
    # implementation; plot MR-OSCAR rows under the collapsed OSCAR/CALM label.
    methods = ["B-CALM", "OSCAR/CALM", "pooled", "CF", "RCT-only"]
    source = {"OSCAR/CALM": "MR-OSCAR"}
    styles = {
        "B-CALM": dict(color="#1f77b4", linestyle="-", marker="o", linewidth=1.7, zorder=5),
        "OSCAR/CALM": dict(color="#9467bd", linestyle="-.", marker="^", linewidth=1.4, zorder=4),
        "pooled": dict(color="#d62728", linestyle="--", marker="s", linewidth=1.4, zorder=3),
        "CF": dict(color="#ff7f0e", linestyle=":", marker="D", linewidth=1.6, zorder=3),
        "RCT-only": dict(color="#111111", linestyle="-", marker="x", linewidth=0.9, zorder=2),
    }
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 1.9))
    for metric, ax, ylabel in [("cate_rmse", axes[0], "CATE RMSE"), ("coverage_90", axes[1], "90% coverage")]:
        for method in methods:
            g = sub[sub["method"] == source.get(method, method)].groupby("bias_strength", as_index=False)[metric].mean()
            if len(g) == 0:
                continue
            g = g.sort_values("bias_strength")
            ax.plot(g["bias_strength"], g[metric], markersize=4, label=method, **styles[method])
        ax.set_xticks([0.0, 0.2, 0.5, 1.0])
        ax.set_xlabel("Comparative-bias strength", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
    axes[1].axhline(0.90, color="#555555", linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.8)
    axes[1].set_ylim(0.0, 1.02)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    log("bias_sweep.pdf: " + repr([ax.get_legend_handles_labels()[1] for ax in axes]))
    fig.savefig(FIGURES_DIR / "bias_sweep.pdf")
    plt.close(fig)
    plt.rcParams.update({"font.size": 10})


def semi_sensitivity() -> None:
    df = pd.read_csv(RESULTS_DIR / "semi_sensitivity.csv")
    # sigma_delta arrives as strings; normalize so "0.0" and "0" both map.
    mapping = {"0": 0.0, "0.0": 0.0, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "inf": 0.7}
    df["sigma_plot"] = df["sigma_delta"].astype(str).map(mapping)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for sid, g in df.groupby("x_slice_id"):
        g = g.sort_values("sigma_plot")
        ax.plot(g["sigma_plot"], g["posterior_cate_mean"], marker="o", linewidth=1.0, alpha=0.8, label=f"profile {int(sid) + 1}")
        ax.fill_between(g["sigma_plot"], g["posterior_cate_lo"], g["posterior_cate_hi"], alpha=0.08)
    ax.set_xticks([0.0, 0.1, 0.2, 0.4, 0.7])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.4", "$\\infty$"])
    ax.set_xlabel("$\\sigma_\\Delta$")
    ax.set_ylabel("Posterior CATE")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    log("semi_sensitivity.pdf: " + repr(ax.get_legend_handles_labels()[1]))
    fig.savefig(FIGURES_DIR / "semi_sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--truetype", action="store_true")
    args = parser.parse_args()
    FIGURES_DIR = args.out_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if args.truetype:
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
    ensure_dirs()
    timer = Timer("figures")
    saturation()
    bias_sweep()
    calibration()
    semi_sensitivity()
    seconds = timer.finish()
    record_wallclock("figures", seconds, out_dir=args.out_dir if args.out_dir != RESULTS_DIR.parent / "figures" else None)
    log(f"wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
