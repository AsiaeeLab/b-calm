"""Greenlight Plus trial and EHR external-control data loader.

EHR records are used as external controls for outcome modeling: every EHR
unit is assigned ``A = 0``. The comparative-bias prior ``sigma_delta`` in the
downstream B-CALM fits controls robustness to departures from this untreated
control convention.

The data are restricted and are not distributed with this code. Set the
environment variable ``GPS_CLEAN_DIR`` to a directory that contains

- ``cate_rct_cross_sectional_data.rds``: one row per trial participant, with
  columns ``site``, ``treatment`` (the treated arm is coded
  ``"Clinic + Digital Intervention"``), ``sex``, ``race_ethnicity``,
  ``language``, ``insurance``, ``wflz_baseline``, ``wflz_12m``,
  ``food_security``, ``education``, ``income``, ``health_literacy``, and
  ``birth_weight``;
- ``cate_ehr_cross_sectional_data.rds``: one row per EHR usual-care child,
  with columns ``site``, ``sex``, ``race_ethnicity``, ``language``,
  ``insurance``, ``wflz_baseline``, and ``wflz_12m``.

Only trial rows from the sites in ``OVERLAP_SITES`` are kept, and rows with a
missing shared covariate or outcome are dropped (complete cases).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr


SHARED_COVARIATES = [
    "site",
    "sex",
    "race_ethnicity",
    "language",
    "insurance",
    "wflz_baseline",
]
RCT_ONLY_COVARIATES = [
    "food_security",
    "education",
    "income",
    "health_literacy",
    "birth_weight",
]
OUTCOME_VAR = "wflz_12m"
OVERLAP_SITES = ["Duke", "UNC", "Vanderbilt"]
EXPECTED_N_R = 385
EXPECTED_N_O = 15150


def _clean_dir() -> Path:
    value = os.environ.get("GPS_CLEAN_DIR")
    if not value:
        raise RuntimeError(
            "GPS_CLEAN_DIR is not set. The real-data analysis needs the restricted "
            "Greenlight Plus trial and EHR files (see the module docstring); set "
            "GPS_CLEAN_DIR to the directory that contains them."
        )
    clean = Path(value).expanduser()
    if not clean.is_dir():
        raise FileNotFoundError(f"GPS_CLEAN_DIR={clean} is not a directory.")
    return clean


def _read_rds(path: Path) -> pd.DataFrame:
    out = pyreadr.read_r(path)
    if not out:
        raise ValueError(f"No dataframe found in {path}")
    return next(iter(out.values())).copy()


def _recode_sex(x: pd.Series) -> pd.Series:
    return x.astype("string").replace({"F": "Female", "M": "Male"})


def _recode_race(x: pd.Series) -> pd.Series:
    return x.astype("string").replace(
        {
            "Non-Hispanic White": "White, non-Hispanic",
            "Non-Hispanic Black": "Black, non-Hispanic",
            "Non-Hispanic Other": "Other, non-Hispanic",
            "Unknown": "Other, non-Hispanic",
        }
    )


def _recode_insurance(x: pd.Series) -> pd.Series:
    long_other = "Unavailable/No Payer Specified/Blank (EHR), NA/Null (EHR), or None (GLP)"
    return x.astype("string").replace({"Other": long_other})


def _clip_wflz(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").clip(-5.0, 5.0)


def _categorical_levels(trial: pd.DataFrame, external: pd.DataFrame) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {"site": list(OVERLAP_SITES)}
    for col in ["sex", "race_ethnicity", "language", "insurance"]:
        vals = pd.concat([trial[col], external[col]], axis=0).dropna().astype(str).unique()
        levels[col] = sorted(vals.tolist())
    return levels


def _dummy_frame(
    df: pd.DataFrame, cols: list[str], levels: dict[str, list[str]]
) -> tuple[pd.DataFrame, list[str]]:
    pieces: list[pd.DataFrame] = []
    names: list[str] = []
    for col in cols:
        if col in {"wflz_baseline", "birth_weight"}:
            num = pd.to_numeric(df[col], errors="coerce").astype(float)
            pieces.append(num.to_frame(col))
            names.append(col)
            continue
        if col in levels:
            cats = pd.Categorical(df[col].astype("string"), categories=levels[col])
        else:
            vals = sorted(df[col].dropna().astype(str).unique())
            cats = pd.Categorical(df[col].astype("string"), categories=vals)
        dummies = pd.get_dummies(cats, prefix=col, prefix_sep="=", dtype=float)
        dummies.index = df.index
        pieces.append(dummies)
        names.extend(dummies.columns.tolist())
    if not pieces:
        return pd.DataFrame(index=df.index), []
    return pd.concat(pieces, axis=1), names


def load_gps(seed: int | None = None) -> dict:
    """Load and harmonize the GPS trial and EHR external-control cohort.

    Returns ``{"D_R": D_R, "D_O": D_O, "shared_cols": [...],
    "X_R_only": [...], "meta": {...}}``. ``D_R`` contains the randomized
    trial treatment indicator and outcome. ``D_O`` contains EHR outcomes and
    sets ``A = 0.0`` for every row, so downstream outcome models use these
    records as external controls while B-CALM's comparative-bias prior governs
    how strongly the resulting control information informs trial efficacy.
    """

    clean = _clean_dir()
    rct = _read_rds(clean / "cate_rct_cross_sectional_data.rds")
    ehr = _read_rds(clean / "cate_ehr_cross_sectional_data.rds")

    rct = rct.loc[rct["site"].isin(OVERLAP_SITES)].copy()
    for df in (rct, ehr):
        df["sex"] = _recode_sex(df["sex"])
        df["race_ethnicity"] = _recode_race(df["race_ethnicity"])
        df["insurance"] = _recode_insurance(df["insurance"])
        df["wflz_baseline"] = _clip_wflz(df["wflz_baseline"])
        df[OUTCOME_VAR] = pd.to_numeric(df[OUTCOME_VAR], errors="coerce")

    keep_trial = rct[SHARED_COVARIATES + [OUTCOME_VAR]].notna().all(axis=1)
    keep_external = ehr[SHARED_COVARIATES + [OUTCOME_VAR]].notna().all(axis=1)
    trial = rct.loc[keep_trial].copy()
    external = ehr.loc[keep_external].copy()

    if seed is not None:
        n_subsample = int(os.environ.get("GPS_SUBSAMPLE_N", "5000"))
        if len(external) > n_subsample:
            external = external.sample(n=n_subsample, random_state=int(seed)).sort_index()

    levels = _categorical_levels(trial, external)
    for col, vals in levels.items():
        trial[col] = pd.Categorical(trial[col].astype("string"), categories=vals)
        external[col] = pd.Categorical(external[col].astype("string"), categories=vals)

    shared_trial, shared_cols = _dummy_frame(trial, SHARED_COVARIATES, levels)
    shared_external, shared_cols_external = _dummy_frame(external, SHARED_COVARIATES, levels)
    if shared_cols != shared_cols_external:
        raise AssertionError("Shared one-hot columns are inconsistent across trial and EHR.")

    rct_only, rct_only_cols = _dummy_frame(trial, RCT_ONLY_COVARIATES, {})
    D_R = pd.concat(
        [
            pd.DataFrame(
                {
                    "A": (trial["treatment"].astype(str) == "Clinic + Digital Intervention").astype(float),
                    "Y": trial[OUTCOME_VAR].astype(float),
                },
                index=trial.index,
            ),
            shared_trial.astype(float),
            rct_only.astype(float),
        ],
        axis=1,
    ).reset_index(drop=True)
    D_O = pd.concat(
        [
            pd.DataFrame({"A": 0.0, "Y": external[OUTCOME_VAR].astype(float)}, index=external.index),
            shared_external.astype(float),
        ],
        axis=1,
    ).reset_index(drop=True)

    raw_trial = trial.reset_index(drop=True)
    raw_external = external.reset_index(drop=True)
    meta = {
        "clean_dir": str(clean),
        "raw_shared_covariates": list(SHARED_COVARIATES),
        "raw_rct_only_covariates": list(RCT_ONLY_COVARIATES),
        "levels": levels,
        "raw_D_R": raw_trial,
        "raw_D_O": raw_external,
        "expected_n_R": EXPECTED_N_R,
        "expected_n_O": EXPECTED_N_O,
    }

    if D_R[shared_cols].isna().any().any() or D_O[shared_cols].isna().any().any():
        raise AssertionError("NaNs remain in shared encoded columns.")
    if not set(D_R["A"].unique()).issubset({0.0, 1.0}):
        raise AssertionError("RCT treatment indicator is not binary.")
    if not np.array_equal(D_O["A"].unique(), np.array([0.0])):
        raise AssertionError("EHR treatment indicator must be constant zero.")
    for col in shared_cols:
        if D_R[col].dtype.kind not in "biufc" or D_O[col].dtype.kind not in "biufc":
            raise AssertionError(f"Non-numeric shared encoded column: {col}")

    return {"D_R": D_R, "D_O": D_O, "shared_cols": shared_cols, "X_R_only": rct_only_cols, "meta": meta}


if __name__ == "__main__":
    loaded = load_gps()
    D_R = loaded["D_R"]
    D_O = loaded["D_O"]
    shared_cols = loaded["shared_cols"]
    print(f"D_R shape: {D_R.shape}")
    print(f"D_O shape: {D_O.shape}")
    print(f"shared encoded columns: {len(shared_cols)}")
    print(f"D_R A values: {sorted(D_R['A'].unique().tolist())}")
    print(f"D_O A values: {sorted(D_O['A'].unique().tolist())}")
    assert D_R[shared_cols].notna().all().all()
    assert D_O[shared_cols].notna().all().all()
    assert set(D_R["A"].unique()).issubset({0.0, 1.0})
    assert np.array_equal(D_O["A"].unique(), np.array([0.0]))
    assert 0.9 * EXPECTED_N_R <= D_R.shape[0] <= 1.1 * EXPECTED_N_R
    assert 0.9 * EXPECTED_N_O <= D_O.shape[0] <= 1.1 * EXPECTED_N_O
