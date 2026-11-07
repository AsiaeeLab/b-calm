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
    ate_draws,
    ci_text,
    common_setup,
    fit_baseline_bcalm,
    prepare_gps,
    sigma_label,
    summarize,
    tex_num,
)
from utils import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, Timer, append_csv_rows, log, record_wallclock


MASKS = ["sex", "race_ethnicity", "language", "insurance", "wflz_baseline"]
FIELDNAMES = ["scenario", "sigma_0", "ate_mean", "ate_lo", "ate_hi", "ate_width", "full_ci_contains_mean"]


def _selected_sigma() -> float:
    path = RESULTS_DIR / "gps_selection.json"
    if not path.exists():
        return 0.2
    lab = json.loads(path.read_text())["selected_sigma_0"]
    return math.inf if lab == "inf" else float(lab)


def _columns_for(shared_cols: list[str], raw_name: str) -> list[str]:
    return [col for col in shared_cols if col == raw_name or col.startswith(f"{raw_name}=")]


def _masked_loaded(loaded: dict, raw_name: str) -> dict:
    out = dict(loaded)
    D_O = loaded["D_O"].copy()
    cols = _columns_for(loaded["shared_cols"], raw_name)
    if not cols:
        raise ValueError(f"No encoded columns found for {raw_name}")
    for col in cols:
        D_O[col] = float(D_O[col].mean())
    out["D_O"] = D_O
    return out


def _write_plot(df: pd.DataFrame, full: pd.Series) -> None:
    panel = df[df["scenario"] != "full"].copy()
    panel = panel.sort_values("ate_mean")
    y = np.arange(len(panel))
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.axvspan(float(full["ate_lo"]), float(full["ate_hi"]), color="#c7d9df", alpha=0.6)
    ax.scatter(panel["ate_mean"], y, color="#1f6f8b", zorder=3)
    for i, row in enumerate(panel.itertuples()):
        ax.plot([row.ate_lo, row.ate_hi], [i, i], color="#83a9b4", linewidth=1.1)
    ax.axvline(float(full["ate_mean"]), color="#234", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(panel["scenario"], fontsize=8)
    ax.set_xlabel("ATE posterior")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gps_loo.pdf")
    plt.close(fig)


def _write_table(df: pd.DataFrame) -> None:
    rows = []
    for row in df.itertuples():
        name = str(row.scenario).replace("_", "\\_")
        hit = "--" if row.scenario == "full" else ("\\checkmark" if row.full_ci_contains_mean else "$\\times$")
        rows.append(
            f"{name} & {tex_num(row.ate_mean)} & {ci_text(row.ate_lo, row.ate_hi)} & {tex_num(row.ate_width)} & {hit} \\\\"
        )
    body = "\n".join(rows)
    (TABLES_DIR / "gps_loo.tex").write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\caption{Leave-one-covariate masking in the EHR controls. The final column checks whether the full-model 90\\% CI contains the masked posterior mean.}\n"
        "\\label{tab:gps-loo}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n"
        "Mask & ATE & 90\\% CI & Width & Hit \\\\\n"
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
    timer = Timer("gps_loo")
    sigma = _selected_sigma()
    lab = sigma_label(sigma)
    loaded = load_gps()
    rct, ehr, X_rct = prepare_gps(loaded)
    rows = []

    model = fit_baseline_bcalm(rct, ehr, sigma, seed=2026)
    full = summarize(ate_draws(model, X_rct, seed=3026, n_draws=1600))
    full.update({"scenario": "full", "sigma_0": lab, "full_ci_contains_mean": True})
    rows.append(full)

    for raw_name in MASKS:
        masked = _masked_loaded(loaded, raw_name)
        rct_m, ehr_m, X_rct_m = prepare_gps(masked)
        model_m = fit_baseline_bcalm(rct_m, ehr_m, sigma, seed=2026)
        item = summarize(ate_draws(model_m, X_rct_m, seed=4026 + len(rows), n_draws=1600))
        hit = bool(full["ate_lo"] <= item["ate_mean"] <= full["ate_hi"])
        item.update({"scenario": raw_name, "sigma_0": lab, "full_ci_contains_mean": hit})
        rows.append(item)
        log(f"completed mask={raw_name}; hit={hit}")

    out = RESULTS_DIR / "gps_loo.csv"
    if out.exists():
        out.unlink()
    append_csv_rows(out, rows, FIELDNAMES)
    df = pd.DataFrame(rows)
    _write_plot(df, df[df["scenario"] == "full"].iloc[0])
    _write_table(df)
    elapsed = timer.finish()
    record_wallclock("gps_loo", elapsed, {"sigma_0": lab})


if __name__ == "__main__":
    main()
