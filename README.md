# B-CALM: Bias-Limited Bayesian Borrowing for RCT-Defined Conditional Average Treatment Effects under Covariate Mismatch

Code for the NeurIPS 2026 paper by Amir Asiaee and Samhita Pal.

B-CALM estimates the conditional average treatment effect (CATE) defined by a randomized controlled trial (RCT) and borrows information from an observational study (OS) whose covariates only partly overlap the trial's. Baseline-bias and comparative-bias functions describe how the OS departs from the trial, and the prior on the comparative-bias function limits how much OS treatment-contrast information can reach the trial CATE. All experiments use the closed-form Gaussian version of the model. Unless a script says otherwise, B-CALM averages over six comparative-bias prior scales with fixed prior weights (`SIGMA_GRID` and `SIGMA_MODEL_PRIOR` in `code/baselines.py`) and posterior weights from the trial-versus-OS contrast.

## Citation

```bibtex
@inproceedings{asiaee2026bcalm,
  title         = {{B-CALM}: Bias-Limited Bayesian Borrowing for {RCT}-Defined Conditional Average Treatment Effects under Covariate Mismatch},
  author        = {Asiaee, Amir and Pal, Samhita},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2607.04036},
  archivePrefix = {arXiv}
}
```

Preprint: [arXiv:2607.04036](https://arxiv.org/abs/2607.04036).

## Repository layout

- `code/`: the model (`bcalm.py`), data-generating processes (`dgp.py`), metrics, baselines (`baselines.py`), experiment runners (`run_*.py`), and table and figure builders (`make_*.py`).
- `code/third_party/qr_learner/`: the vendored QR-learner used for one comparison (see Third-party code).
- `results/`: the result CSVs behind the synthetic and semi-synthetic results in the paper.
- `notebooks/reproduce.ipynb`: runs the core synthetic and semi-synthetic experiments in order and builds their figures and tables.

Tables and figures are written to `tables/` and `figures/` at the repository root. Both directories are created on the first run and are not tracked.

## Installation

The code was run with Python 3.10.

```bash
pip install -r code/requirements.txt
```

`econml` is needed only for the causal-forest (CF) baseline. If it is missing, the runners skip CF and log a message.

The R-OSCAR, MR-OSCAR, and CALM baselines are prior work (arXiv:2306.17478, arXiv:2603.17066, and arXiv:2603.19186). Their implementation is in the companion repository [AsiaeeLab/calm-cate](https://github.com/AsiaeeLab/calm-cate), which this code imports through an environment variable:

```bash
git clone https://github.com/AsiaeeLab/calm-cate.git
export BCALM_EXTERNAL_CALM_PATH=/path/to/calm-cate
```

`BCALM_EXTERNAL_CALM_PATH` must point at the root of the checkout, the directory that contains `src/`. Without it, the synthetic runners skip these three baselines and log a message, and everything else runs.

## Reproducing the results

Run the scripts from `code/`. Each script writes to `results/`, `tables/`, or `figures/` at the repository root regardless of the working directory; the newer scripts also accept `--out-dir` (and `--table-dir` or `--figure-dir` where relevant).

`results/` already contains the paper's CSVs, so the table and figure builders in steps 2 and 4 can run immediately. The core runners `run_main.py`, `run_saturation.py`, `run_semi.py`, and `run_ablation.py` resume from rows already present in their output file. To regenerate everything, move `results/` aside first. Steps 3 and 4 read `results/main.csv` and `results/ablation.csv`, so regenerate those before running them.

The times below are approximate wall-clock times from the paper runs on a single workstation. The core runners used their default `--workers 10` for the OSCAR/CALM bootstrap; the additional experiments in step 3 used 5 workers.

**Step 1. Core synthetic and semi-synthetic experiments.** Each runner accepts `--mode {smoke,cheap,full}`: `smoke` is an end-to-end check that takes seconds, and `full` uses the paper's settings.

```bash
cd code
python run_saturation.py  --mode full   # about 25 s
python run_main.py        --mode full   # about 5 h with all baselines; the B-CALM rows alone take about 2 min
python run_calibration.py --mode full   # about 25 min
python run_semi.py        --mode full   # about 5 min
python run_ablation.py    --mode full   # a few seconds
```

Most of the time in `run_main.py`, `run_calibration.py`, and `run_semi.py` goes to the 200-resample bootstrap for the OSCAR/CALM baselines and to CF.

**Step 2. Figures and tables for the core experiments** (a few seconds in total).

```bash
python make_figures.py                 # saturation, bias sweep, calibration, semi-synthetic sensitivity
python make_tables.py                  # per-cell main table, saturation table, semi-synthetic table
python make_tables.py --summary-only   # main simulation summary table
```

**Step 3. Additional synthetic experiments.**

```bash
python run_commensurate.py              # scalar commensurate and robust-MAP comparison, about 5 s
python run_qr_learner.py                # QR-learner benchmark, about 15 s
python run_operating_characteristics.py # Go/No-Go operating characteristics, about 100 s
python run_prior_weights.py             # prior-weight sensitivity over the full grid, about 17 s
python run_overlap.py                   # MMD along the alignment ladder, about 20 s
```

`run_prior_weights.py` stops with an error unless its fit under the default prior weights reproduces the B-CALM rows of `results/main.csv`. `run_commensurate.py --validate-only` and `run_qr_learner.py --validate-only` fit a single seed as a quick check. The raw per-seed output of `run_operating_characteristics.py` (about 10 MB) is not shipped; `python run_operating_characteristics.py --tables-only` rebuilds its tables from the shipped summary.

**Step 4. Regrouped and remaining tables** (under a second each).

```bash
python make_regime_tables.py      # results by bias regime, alignment ladder, OSCAR-family precision
python make_additional_tables.py  # ablation, scalar-discrepancy, and QR-learner tables
```

Run `make_additional_tables.py` after `make_tables.py`: both write `tables/ablation.tex`, and the version from `make_additional_tables.py` is the one in the paper.

## Paper-to-code map

Some tables in the paper combine or reformat the generated LaTeX. For example, the alignment-ladder table in the paper merges `tables/alignment_ladder.tex` and `tables/overlap.tex`.

| Paper content | Scripts | Result files | Generated tables and figures |
|---|---|---|---|
| Main simulation summary and per-cell table | `run_main.py`, `make_tables.py` | `results/main.csv` | `tables/main_summary.tex`, `tables/main.tex` |
| Per-replicate posterior weights over prior scales | `run_main.py` | `results/bma_weights.csv` | none |
| Comparative-bias sweep figure | `run_main.py`, `make_figures.py` | `results/main.csv` | `figures/bias_sweep.pdf` |
| Saturation (effective sample size) | `run_saturation.py`, `make_figures.py`, `make_tables.py` | `results/saturation.csv` | `figures/saturation.pdf`, `tables/saturation.tex` |
| Ablation | `run_ablation.py`, `make_additional_tables.py` | `results/ablation.csv` | `tables/ablation.tex` |
| Calibration curves | `run_calibration.py`, `make_figures.py` | `results/calibration.csv` | `figures/calibration.pdf` |
| STAR-like semi-synthetic benchmark and sensitivity figure | `run_semi.py`, `make_tables.py`, `make_figures.py` | `results/semi.csv`, `results/semi_sensitivity.csv` | `tables/semi.tex`, `figures/semi_sensitivity.pdf` |
| OSCAR-family precision | `make_regime_tables.py` | `results/oscar_precision.csv`, `results/oscar_seed_differences.csv`, `results/baseline_distinctness.json`, `results/baseline_distinctness_gridmeans.csv` | `tables/oscar_precision.tex` |
| Results by bias regime | `make_regime_tables.py` | `results/by_bias_regime.csv` | `tables/by_regime.tex` |
| Alignment ladder and MMD diagnostic | `make_regime_tables.py`, `run_overlap.py` | `results/by_alignment_regime.csv`, `results/overlap_ladder.csv`, `results/overlap_ladder_raw.csv` | `tables/alignment_ladder.tex`, `tables/overlap.tex` |
| Scalar and robust-MAP comparison | `run_commensurate.py`, `make_additional_tables.py` | `results/commensurate_baseline.csv`, `results/commensurate_baseline_summary.csv` | `tables/scalar_bias.tex` |
| QR-learner benchmark | `run_qr_learner.py`, `make_additional_tables.py` | `results/qr_learner_benchmark.csv`, `results/qr_learner_benchmark_summary.csv` | `tables/qr_learner.tex` |
| Go/No-Go operating characteristics | `run_operating_characteristics.py` | `results/operating_characteristics_summary.csv` | `tables/oc_main.tex`, `tables/oc_fixed_sigma.tex` |
| Prior-weight sensitivity | `run_prior_weights.py` | `results/prior_weights.csv`, `results/prior_weights_raw.csv` | `tables/prior_weights.tex` |

`results/run_times.json` records the wall-clock time of the last run of each core runner and builder.

## Data availability

The synthetic data (`generate_data` in `code/dgp.py`) and the STAR-like semi-synthetic data (`generate_star_like` in `code/dgp.py`) are generated by the code from fixed seeds, so no data files are needed for those experiments.

The analysis code and outputs for the paper's real-data application use restricted data and are not part of this repository.

## Third-party code

`code/third_party/qr_learner/official_learners.py` is `src/randomization_aware/learners.py` from [RickardKarl/robust-trial-CATE-augmentation](https://github.com/RickardKarl/robust-trial-CATE-augmentation) at commit `6a62bd6b0bbf8e90c43d4108c3be227b3e4c0e6a`, included unmodified under its MIT license (`code/third_party/qr_learner/LICENSE`, Copyright (c) 2025 Rickard Karlsson). It is used only by `run_qr_learner.py` for the QR-learner comparison. The method is described in Karlsson et al., "Robust Estimation of Heterogeneous Treatment Effects in Randomized Trials Leveraging External Data," AISTATS 2026.

## License

MIT. See `LICENSE`. The vendored QR-learner keeps its own MIT license.
