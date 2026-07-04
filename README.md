# B-CALM: Bias-Limited Bayesian Borrowing for RCT-Anchored Conditional Treatment Effects under Covariate Mismatch — experiment code

This repository contains the experiment code for B-CALM. B-CALM estimates a conditional average treatment effect (CATE) by anchoring on a randomized controlled trial (RCT) and borrowing strength from an observational study whose covariates only partially overlap the trial's. A Gaussian-prior contrast model places a calibrated prior on the trial-vs-observational effect discrepancy, so the amount of borrowing is limited by how much the two sources actually disagree: when the observational arm is unbiased the posterior tightens toward the pooled estimate, and when it is biased the borrowing saturates and the posterior falls back toward the RCT-only estimate. A Bayesian model average over the discrepancy-prior scale selects the borrowing strength automatically, yielding calibrated credible intervals and bounded negative transfer. The main, semi-synthetic, and calibration studies all run B-CALM with this empirical-Bayes model averaging under fixed asymmetric prior weights over the $\sigma_\Delta$ grid (`SIGMA_MODEL_PRIOR` in `code/baselines.py`); the main grid additionally exports each replicate's posterior weights to `results/bma_weights.csv`.

## Install

Requires **Python 3.10**.

```bash
pip install -r code/requirements.txt
```

`econml` is only needed for the CF (causal forest) baseline; every B-CALM, RCT-only, pooled, and OS-only experiment runs without it.

## Reproduce

The easiest path is the notebook:

```bash
jupyter notebook notebooks/reproduce.ipynb
```

It installs dependencies, runs each experiment, and builds the figures and tables in order.

To run the pieces directly from the command line, work inside `code/` (the runners resolve `results/`, `figures/`, and `tables/` relative to the package root):

```bash
cd code
python run_saturation.py  --mode full
python run_main.py        --mode full
python run_calibration.py --mode full
python run_semi.py        --mode full
python run_ablation.py    --mode full
python make_figures.py
python make_tables.py
```

Every runner accepts `--mode {smoke,cheap,full}`. Use `smoke` for a fast end-to-end check (seconds), `cheap` for an intermediate run, and `full` to reproduce the paper numbers. The figure and table builders read the `full`-mode CSV filenames, so run the experiments in `full` mode before building artifacts.

## What is and isn't included

- **Synthetic experiments** (saturation, main grid, calibration, ablations): fully reproducible from this package.
- **Semi-synthetic STAR-like benchmark**: fully reproducible from this package.
- **Real-data pediatric-obesity analysis**: **excluded.** It uses patient-level EHR/trial data that cannot be released for IRB/DUA reasons. The corresponding analysis code is also excluded; it is available on request to the authors for readers with appropriate data access.
- **OSCAR/CALM-family baselines** (R-OSCAR, MR-OSCAR, CALM): these are prior works whose code lives in separate companion repositories. They are skipped automatically unless you check out the companion code and set the environment variable `BCALM_EXTERNAL_CALM_PATH` to its path. References:
  - R-OSCAR: arXiv:2306.17478
  - MR-OSCAR: arXiv:2603.17066
  - CALM: arXiv:2603.19186

## Outputs → paper

| Result CSV | Figures / tables produced |
|---|---|
| `results/main.csv` | Table 1, the full main results `tables/main.tex`, and `figures/bias_sweep.pdf` |
| `results/saturation.csv` | `figures/saturation.pdf` and saturation table `tables/saturation.tex` |
| `results/calibration.csv` | `figures/calibration.pdf` |
| `results/semi.csv`, `results/semi_sensitivity.csv` | semi table `tables/semi.tex` and `figures/semi_sensitivity.pdf` |
| `results/ablation.csv` | ablation table `tables/ablation.tex` |
| `results/bma_weights.csv` | per-replicate BMA weight diagnostic (written by `run_main.py`; not a paper artifact) |
