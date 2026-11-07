from __future__ import annotations

import math
import os
import sys
import concurrent.futures
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bcalm import BCalmLinear, BCalmMixture

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# The R-OSCAR / MR-OSCAR / CALM calibrated-borrowing baselines are the prior
# works and live in the companion repository https://github.com/AsiaeeLab/calm-cate
# (not part of this release). We import them lazily and degrade gracefully when
# that repository is absent: the rest of this module (and all B-CALM
# experiments) import and run regardless.
#
#   R-OSCAR:  arXiv:2306.17478
#   MR-OSCAR: arXiv:2603.17066
#   CALM:     arXiv:2603.19186
#
# To enable these baselines, clone AsiaeeLab/calm-cate and point the environment
# variable BCALM_EXTERNAL_CALM_PATH at the root of that checkout (the directory
# that contains src/).
_ENV_EXTERNAL = os.environ.get("BCALM_EXTERNAL_CALM_PATH")
EXTERNAL_MISMATCH_CODE = Path(_ENV_EXTERNAL) if _ENV_EXTERNAL else None

EXTERNAL_AVAILABLE = False
try:
    if (
        EXTERNAL_MISMATCH_CODE is not None
        and EXTERNAL_MISMATCH_CODE.exists()
        and str(EXTERNAL_MISMATCH_CODE) not in sys.path
    ):
        sys.path.insert(0, str(EXTERNAL_MISMATCH_CODE))
    from src.calibration import calibrate_and_blend
    from src.utils import ConstantModel, doubly_robust_pseudo, fit_regressor, make_folds, predict_arm_models
    EXTERNAL_AVAILABLE = True
except Exception:
    EXTERNAL_AVAILABLE = False


def external_available() -> bool:
    """True iff the OSCAR/CALM-family companion repo is importable."""
    return EXTERNAL_AVAILABLE


SIGMA_GRID = [0.01, 0.05, 0.2, 0.5, 1.0, math.inf]
# Asymmetric robust prior weights, fixed across all cells and studies.
# Rationale: the two error directions are not symmetric -- an overly TIGHT
# sigma_delta over-borrows a biased contrast and breaks validity, while an
# overly DIFFUSE sigma_delta only costs efficiency. The prior therefore stays
# skeptical on the tight scales (0.01, 0.05), keeps a mild mode at moderate
# borrowing (0.2), and carries a heavy diffuse tail for fast retreat.
SIGMA_MODEL_PRIOR = np.array([0.01, 0.01, 0.42, 0.21, 0.13, 0.22], dtype=float)


def fit_arm_models(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    *,
    kind: str = "ridge",
    alpha: float = 1.0,
    random_state: int = 0,
    rct_fallback: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[int, object]:
    """Fit per-arm models, with opt-in RCT rows for tiny OS arms."""
    models = {}
    for arm in (-1, 1):
        mask = A == arm
        if np.sum(mask) >= 3:
            models[arm] = fit_regressor(X[mask], Y[mask], kind=kind, alpha=alpha, random_state=random_state)
            continue
        if rct_fallback is not None:
            X_rct, A_rct, Y_rct = rct_fallback
            rct_mask = A_rct == arm
            if np.sum(rct_mask) >= 3:
                models[arm] = fit_regressor(X_rct[rct_mask], Y_rct[rct_mask], kind=kind, alpha=alpha, random_state=random_state)
                continue
            warnings.warn(
                f"fit_arm_models: arm {arm} has < 3 OS patients and rct_fallback "
                f"also has < 3 RCT patients in arm {arm}; falling back to ConstantModel(mean(Y)).",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"fit_arm_models: arm {arm} has < 3 patients; falling back to ConstantModel(mean(Y)). "
                "Pass rct_fallback=(X_rct, A_rct, Y_rct) to fit the missing arm from RCT data.",
                UserWarning,
                stacklevel=2,
            )
        models[arm] = ConstantModel(float(np.mean(Y)))
    return models


def _contrast_log_marginal(
    tau_r: np.ndarray,
    cov_r: np.ndarray,
    tau_o: np.ndarray,
    cov_o: np.ndarray,
    sigma_delta: float,
) -> float:
    diff = np.asarray(tau_o - tau_r, dtype=float)
    k = diff.shape[0]
    if math.isinf(float(sigma_delta)):
        prior_var = 1.0e4
    else:
        prior_var = max(float(sigma_delta) ** 2, 1.0e-8)
    cov = cov_r + cov_o + prior_var * np.eye(k)
    cov = 0.5 * (cov + cov.T) + 1.0e-9 * np.eye(k)
    chol = cho_factor(cov, lower=True, check_finite=False)
    quad = float(diff @ cho_solve(chol, diff, check_finite=False))
    logdet = float(2.0 * np.sum(np.log(np.diag(chol[0]))))
    return -0.5 * (k * math.log(2.0 * math.pi) + logdet + quad)


def split_rct(D_R: dict[str, np.ndarray], seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = len(D_R["Y"])
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(int(0.70 * n), 2)
    tr, va = idx[:cut], idx[cut:]
    if len(va) == 0:
        va = tr
    return (
        {key: val[tr] for key, val in D_R.items()},
        {key: val[va] for key, val in D_R.items()},
    )


def heldout_mse(model: BCalmLinear, D_val: dict[str, np.ndarray]) -> float:
    pred = model.predict_rct_mean(D_val["X"], D_val["A"])
    return float(np.mean((D_val["Y"] - pred) ** 2))


def fit_bcalm_selected(
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    basis_kind: str,
    seed: int,
    sigma_y: float = 1.0,
    *,
    sigma0: float = 0.6,
    prior_weights: np.ndarray | None = None,
) -> tuple[BCalmMixture, str]:
    prior = SIGMA_MODEL_PRIOR if prior_weights is None else np.asarray(prior_weights, dtype=float)
    if prior.shape != (len(SIGMA_GRID),) or not np.all(np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("prior_weights must contain six finite positive weights.")
    rct_contrast = BCalmLinear(sigma_delta=math.inf, sigma0=math.inf, sigma_y=sigma_y, basis_kind=basis_kind).fit(D_R, None)
    os_contrast = BCalmLinear(sigma_delta=0.0, sigma0=0.6, sigma_y=sigma_y, basis_kind=basis_kind).fit(None, D_O)
    tau_r = rct_contrast.mean_[rct_contrast.tau_slice]
    cov_r = rct_contrast.cov_[rct_contrast.tau_slice, rct_contrast.tau_slice]
    tau_o = os_contrast.mean_[os_contrast.tau_slice]
    cov_o = os_contrast.cov_[os_contrast.tau_slice, os_contrast.tau_slice]
    models = []
    logps = []
    for sigma in SIGMA_GRID:
        model = BCalmLinear(sigma_delta=sigma, sigma0=sigma0, sigma_y=sigma_y, basis_kind=basis_kind)
        model.fit(D_R, D_O)
        models.append(model)
        logps.append(_contrast_log_marginal(tau_r, cov_r, tau_o, cov_o, sigma))
    logps_arr = np.asarray(logps, dtype=float)
    logps_arr = logps_arr + np.log(prior)
    weights = np.exp(logps_arr - logsumexp(logps_arr))
    mixture = BCalmMixture(models, weights)
    mixture.sigma_grid_ = tuple(SIGMA_GRID)
    mixture.log_marginal_likelihoods_ = tuple(float(x) for x in logps_arr)
    mixture.weight_summary_ = {
        sigma: float(weight)
        for sigma, weight in zip(SIGMA_GRID, weights)
    }
    return mixture, "bma"


def _feature_split(X: np.ndarray, n_shared: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    if n_shared is None:
        n_shared = p if p <= 2 else max(1, p - 2)
    n_shared = int(min(max(n_shared, 1), p))
    Z = X[:, :n_shared]
    U_or_V = X[:, n_shared:]
    return Z, U_or_V


def _as_external_data(D_R: dict[str, np.ndarray], D_O: dict[str, np.ndarray]) -> tuple[dict, dict, int]:
    Z_r, U_r = _feature_split(D_R["X"])
    Z_o, V_o = _feature_split(D_O["X"], Z_r.shape[1])
    rct_data = {
        "U": U_r,
        "Z": Z_r,
        "A": 2.0 * np.asarray(D_R["A"], dtype=float) - 1.0,
        "Y": np.asarray(D_R["Y"], dtype=float),
    }
    os_data = {
        "Z": Z_o,
        "V": V_o,
        "A": 2.0 * np.asarray(D_O["A"], dtype=float) - 1.0,
        "Y": np.asarray(D_O["Y"], dtype=float),
    }
    return os_data, rct_data, Z_r.shape[1]


def _concat_uz_from_X(X: np.ndarray, n_shared: int) -> np.ndarray:
    Z, U = _feature_split(X, n_shared)
    return np.hstack([U, Z])


@dataclass
class PointCateModel:
    cate_model: object
    predictive_var: float
    n_shared: int
    x_mean: np.ndarray
    x_scale: np.ndarray
    label: str
    bootstrap_models: tuple[object, ...] = ()

    def _features(self, X_eval: np.ndarray) -> np.ndarray:
        return _concat_uz_from_X(X_eval, self.n_shared)

    def point_cate(self, X_eval: np.ndarray) -> np.ndarray:
        X = self._features(X_eval)
        return np.asarray(self.cate_model.predict(X), dtype=float).reshape(-1)

    def _bootstrap_predictions(self, X_eval: np.ndarray) -> np.ndarray:
        X = self._features(X_eval)
        preds = []
        for m in self.bootstrap_models:
            if hasattr(m, "point_cate"):
                preds.append(m.point_cate(X_eval))
            else:
                preds.append(np.asarray(m.predict(X), dtype=float).reshape(-1))
        return np.stack(preds, axis=0)

    def posterior_cate_moments(self, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.bootstrap_models:
            preds = self._bootstrap_predictions(X_eval)
            mean = np.median(preds, axis=0)
            lo = np.quantile(preds, 0.05, axis=0)
            hi = np.quantile(preds, 0.95, axis=0)
            sd = np.maximum((hi - lo) / (2.0 * 1.6448536269514722), 1.0e-6)
            return mean, sd**2
        X = self._features(X_eval)
        mean = np.asarray(self.cate_model.predict(X), dtype=float).reshape(-1)
        z = (X - self.x_mean) / self.x_scale
        leverage = 1.0 + 0.02 * np.mean(z**2, axis=1)
        var = np.maximum(float(self.predictive_var) * leverage, 1.0e-12)
        return mean, var

    def posterior_cate(self, X_eval: np.ndarray, n_samples: int = 500, seed: int | None = None) -> np.ndarray:
        if self.bootstrap_models:
            preds = self._bootstrap_predictions(X_eval)
            if preds.shape[0] >= n_samples:
                return preds[:n_samples]
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, preds.shape[0], size=n_samples)
            return preds[idx]
        mean, var = self.posterior_cate_moments(X_eval)
        rng = np.random.default_rng(seed)
        return rng.normal(mean, np.sqrt(var), size=(n_samples, mean.shape[0]))


@dataclass
class EconmlCausalForestCate:
    estimator: object
    fallback_var: float
    label: str = "CF"
    bootstrap_models: tuple[object, ...] = ()

    def point_cate(self, X_eval: np.ndarray) -> np.ndarray:
        X_eval = np.asarray(X_eval, dtype=float)
        return np.asarray(self.estimator.effect(X_eval, T0=0, T1=1), dtype=float).reshape(-1)

    def _bootstrap_predictions(self, X_eval: np.ndarray) -> np.ndarray:
        return np.stack([m.point_cate(X_eval) for m in self.bootstrap_models], axis=0)

    def posterior_cate_moments(self, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.bootstrap_models:
            preds = self._bootstrap_predictions(X_eval)
            mean = np.median(preds, axis=0)
            lo = np.quantile(preds, 0.05, axis=0)
            hi = np.quantile(preds, 0.95, axis=0)
            sd = np.maximum((hi - lo) / (2.0 * 1.6448536269514722), 1.0e-6)
            return mean, sd**2
        X_eval = np.asarray(X_eval, dtype=float)
        mean = self.point_cate(X_eval)
        try:
            inf = self.estimator.effect_inference(X_eval, T0=0, T1=1)
            stderr = np.asarray(inf.stderr, dtype=float).reshape(-1)
            var = stderr**2
        except Exception:
            var = np.full(mean.shape[0], float(self.fallback_var), dtype=float)
        var = np.where(np.isfinite(var) & (var > 1.0e-12), var, float(self.fallback_var))
        return mean, np.maximum(var, 1.0e-12)

    def posterior_cate(self, X_eval: np.ndarray, n_samples: int = 500, seed: int | None = None) -> np.ndarray:
        if self.bootstrap_models:
            preds = self._bootstrap_predictions(X_eval)
            if preds.shape[0] >= n_samples:
                return preds[:n_samples]
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, preds.shape[0], size=n_samples)
            return preds[idx]
        mean, var = self.posterior_cate_moments(X_eval)
        rng = np.random.default_rng(seed)
        return rng.normal(mean, np.sqrt(var), size=(n_samples, mean.shape[0]))


@dataclass
class BcfHahnCate:
    idata: object
    model: object
    x_template: np.ndarray
    tau_draws: np.ndarray  # shape (n_draws, n_train)
    f_draws: np.ndarray    # shape (n_draws, n_train)
    label: str = "BCF"

    def _predict_tau(self, X_eval: np.ndarray) -> np.ndarray:
        """Posterior tau(x_eval) draws via nearest-neighbor lookup on training covariates.

        BART posteriors are evaluated at training X. For new X, we use 1-NN
        lookup on the pooled training X to project posterior tau draws onto X_eval.
        Cheap and deterministic.
        """
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(self.x_template)
        idx = nn.kneighbors(X_eval, return_distance=False).reshape(-1)
        return self.tau_draws[:, idx]  # (n_draws, n_eval)

    def point_cate(self, X_eval: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict_tau(X_eval).mean(axis=0), dtype=float)

    def posterior_cate_moments(self, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        draws = self._predict_tau(X_eval)
        return draws.mean(axis=0), draws.var(axis=0)

    def posterior_cate(self, X_eval: np.ndarray, n_samples: int = 500, seed: int | None = None) -> np.ndarray:
        draws = self._predict_tau(X_eval)
        n_draws = draws.shape[0]
        if seed is not None and n_draws > n_samples:
            rng = np.random.default_rng(seed)
            sel = rng.choice(n_draws, size=n_samples, replace=False)
            return draws[sel]
        return draws


def _resample_rct(D_R: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = D_R["X"].shape[0]
    idx = rng.integers(0, n, size=n)
    return {key: val[idx] for key, val in D_R.items()}


def _fit_oscar_bootstrap_one(args: tuple[str, dict[str, np.ndarray], dict[str, np.ndarray], int, int]) -> PointCateModel:
    method, D_R, D_O, seed, n_folds = args
    return _fit_oscar_like(method, _resample_rct(D_R, seed), D_O, seed, n_folds=n_folds, bootstrap_b=0)


def _fit_cf_bootstrap_one(args: tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]) -> EconmlCausalForestCate:
    D_R, D_O, seed = args
    return _fit_econml_causal_forest(_resample_rct(D_R, seed), D_O, seed, bootstrap_b=0)


def _fit_bootstrap_models(fit_args: list[tuple], fit_fn, *, n_workers: int) -> tuple[object, ...]:
    if not fit_args:
        return ()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        return tuple(ex.map(fit_fn, fit_args))


def _fit_point_bootstrap_one(args: tuple[np.ndarray, np.ndarray, float, int]) -> object:
    X_rct, pseudo, cate_alpha, seed = args
    rng = np.random.default_rng(seed)
    n = X_rct.shape[0]
    idx = rng.integers(0, n, size=n)
    return fit_regressor(X_rct[idx], pseudo[idx], kind="lasso", alpha=float(cate_alpha), random_state=int(seed))


def _fit_point_model(
    X_rct: np.ndarray,
    pseudo: np.ndarray,
    *,
    n_shared: int,
    random_state: int,
    cate_alpha: float,
    variance_scale: float,
    label: str,
    bootstrap_b: int = 0,
    workers: int = 10,
) -> PointCateModel:
    cate_model = fit_regressor(X_rct, pseudo, kind="lasso", alpha=cate_alpha, random_state=random_state)
    fitted = np.asarray(cate_model.predict(X_rct), dtype=float).reshape(-1)
    resid_mse = float(np.mean((pseudo - fitted) ** 2))
    predictive_var = variance_scale * max(resid_mse, 1.0e-6) / max(X_rct.shape[0], 1)
    x_mean = np.mean(X_rct, axis=0, keepdims=True)
    x_scale = np.std(X_rct, axis=0, keepdims=True)
    x_scale = np.where(x_scale < 1.0e-8, 1.0, x_scale)
    bootstrap_models: tuple[object, ...] = ()
    if bootstrap_b > 0:
        rng = np.random.default_rng(random_state)
        boot_seeds = rng.integers(1, 2**31 - 1, size=int(bootstrap_b), dtype=np.int64)
        args = [(X_rct, pseudo, cate_alpha, int(b_seed)) for b_seed in boot_seeds]
        bootstrap_models = _fit_bootstrap_models(args, _fit_point_bootstrap_one, n_workers=int(workers))
    return PointCateModel(cate_model, predictive_var, n_shared, x_mean, x_scale, label, bootstrap_models)


def _fit_oscar_like(
    method: str,
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    seed: int,
    *,
    n_folds: int = 5,
    bootstrap_b: int = 0,
    workers: int = 10,
) -> PointCateModel:
    os_data, rct_data, n_shared = _as_external_data(D_R, D_O)
    X_rct = np.hstack([rct_data["U"], rct_data["Z"]])
    Z_rct = rct_data["Z"]
    A_rct = rct_data["A"]
    Y_rct = rct_data["Y"]
    Z_os = os_data["Z"]
    V_os = os_data["V"]
    A_os = os_data["A"]
    Y_os = os_data["Y"]
    folds = make_folds(X_rct.shape[0], n_folds=min(n_folds, X_rct.shape[0]), seed=seed)
    one_arm_os = np.unique(A_os).shape[0] < 2

    if method == "MR-OSCAR":
        imputer = Ridge(alpha=1.0)
        imputer.fit(Z_os, V_os)
        X_os_base = np.hstack([Z_os, V_os])
        V_hat_all = imputer.predict(Z_rct)
        X_rct_base = np.hstack([Z_rct, V_hat_all])
        rct_fallback = (X_rct_base, A_rct, Y_rct) if one_arm_os else None
        os_arm_models = fit_arm_models(
            X_os_base,
            A_os,
            Y_os,
            kind="ridge",
            alpha=0.6,
            random_state=seed,
            rct_fallback=rct_fallback,
        )
        cate_alpha = 0.02
        variance_scale = 10.0
    elif method == "R-OSCAR":
        rct_fallback = (Z_rct, A_rct, Y_rct) if one_arm_os else None
        os_arm_models = fit_arm_models(
            Z_os,
            A_os,
            Y_os,
            kind="ridge",
            alpha=0.5,
            random_state=seed,
            rct_fallback=rct_fallback,
        )
        X_rct_base = Z_rct
        cate_alpha = 0.02
        variance_scale = 9.0
    elif method == "CALM":
        X_os_zv = np.hstack([Z_os, V_os])
        scaler_os = StandardScaler().fit(X_os_zv)
        X_os_scaled = scaler_os.transform(X_os_zv)
        embed_dim = int(min(5, X_os_zv.shape[1], max(X_os_zv.shape[0] - 1, 1)))
        pls = PLSRegression(n_components=max(embed_dim, 1), scale=False)
        pls.fit(X_os_scaled, Y_os)
        H_os = pls.transform(X_os_scaled)
        imputer = Ridge(alpha=1.0)
        imputer.fit(Z_os, V_os)
        V_hat = imputer.predict(Z_rct)
        X_rct_zvhat = np.hstack([Z_rct, V_hat])
        X_rct_base = pls.transform(scaler_os.transform(X_rct_zvhat))
        rct_fallback = (X_rct_base, A_rct, Y_rct) if one_arm_os else None
        os_arm_models = fit_arm_models(
            H_os,
            A_os,
            Y_os,
            kind="ridge",
            alpha=0.5,
            random_state=seed,
            rct_fallback=rct_fallback,
        )
        cate_alpha = 0.02
        variance_scale = 10.0
    else:
        raise ValueError(f"Unknown calibrated borrowing method: {method}")

    pseudo = np.zeros(X_rct.shape[0], dtype=float)
    for k, (tr, va) in enumerate(folds):
        base_mu1_train, base_mu0_train = predict_arm_models(os_arm_models, X_rct_base[tr])
        base_mu1_val, base_mu0_val = predict_arm_models(os_arm_models, X_rct_base[va])
        mu1_val, mu0_val = calibrate_and_blend(
            X_train=X_rct[tr],
            X_val=X_rct[va],
            A_train=A_rct[tr],
            Y_train=Y_rct[tr],
            base_mu1_train=base_mu1_train,
            base_mu0_train=base_mu0_train,
            base_mu1_val=base_mu1_val,
            base_mu0_val=base_mu0_val,
            random_state=seed + k,
            blend_bias=0.0,
        )
        pseudo[va] = doubly_robust_pseudo(Y_rct[va], A_rct[va], mu1_val, mu0_val, p1=0.5)

    model = _fit_point_model(
        X_rct,
        pseudo,
        n_shared=n_shared,
        random_state=seed + 100,
        cate_alpha=cate_alpha,
        variance_scale=variance_scale,
        label=method,
        bootstrap_b=0,
        workers=workers,
    )
    if bootstrap_b > 0:
        rng = np.random.default_rng(seed)
        boot_seeds = rng.integers(1, 2**31 - 1, size=int(bootstrap_b), dtype=np.int64)
        args = [(method, D_R, D_O, int(b_seed), n_folds) for b_seed in boot_seeds]
        model.bootstrap_models = _fit_bootstrap_models(args, _fit_oscar_bootstrap_one, n_workers=int(workers))
    return model


def _fit_econml_causal_forest(
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    seed: int,
    *,
    bootstrap_b: int = 0,
    workers: int = 10,
) -> EconmlCausalForestCate:
    try:
        from econml.dml import CausalForestDML
    except Exception as exc:
        raise ImportError("EconML is required for causal forest.") from exc

    X = np.vstack([D_R["X"], D_O["X"]])
    A = np.concatenate([D_R["A"], D_O["A"]]).astype(int)
    Y = np.concatenate([D_R["Y"], D_O["Y"]]).astype(float)
    model_y = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, n_jobs=1, random_state=seed)
    model_t = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, n_jobs=1, random_state=seed + 1)
    estimator = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        categories=[0, 1],
        cv=2,
        n_estimators=50,
        subforest_size=2,
        min_samples_leaf=10,
        max_depth=None,
        inference=True,
        n_jobs=1,
        random_state=seed,
    )
    estimator.fit(Y, A, X=X)
    train_effect = np.asarray(estimator.effect(X, T0=0, T1=1), dtype=float).reshape(-1)
    fallback_var = max(float(np.var(train_effect)) / max(D_R["X"].shape[0], 1), 1.0e-4)
    model = EconmlCausalForestCate(estimator, fallback_var=fallback_var, label="CF")
    if bootstrap_b > 0:
        rng = np.random.default_rng(seed)
        boot_seeds = rng.integers(1, 2**31 - 1, size=int(bootstrap_b), dtype=np.int64)
        args = [(D_R, D_O, int(b_seed)) for b_seed in boot_seeds]
        model.bootstrap_models = _fit_bootstrap_models(args, _fit_cf_bootstrap_one, n_workers=int(workers))
    return model


def _fit_bcf_hahn(
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    seed: int,
    *,
    n_trees: int = 50,
    draws: int = 500,
    tune: int = 250,
    chains: int = 1,
) -> BcfHahnCate:
    """Hahn-Murray-Carvalho 2020 Bayesian Causal Forest with explicit PGBART steps.

    Two BART components: prognostic ``f(X)`` for E[Y|X, A=0] and moderating ``tau(X)``
    for the treatment-effect surface. Sampling uses explicit PGBART step methods
    per BART RV to bypass a pymc-bart 0.9 / pymc 5.23 compound-step incompatibility
    where the implicit dispatch passes ``blocked=True`` to PGBART.__init__, which
    PGBART does not accept.
    """
    try:
        import pymc as pm
        import pymc_bart as pmb
    except Exception as exc:
        raise ImportError("pymc and pymc-bart are required for BCF.") from exc

    X = np.vstack([D_R["X"], D_O["X"]])
    A = np.concatenate([D_R["A"], D_O["A"]]).astype(float)
    Y = np.concatenate([D_R["Y"], D_O["Y"]]).astype(float)
    with pm.Model() as model:
        f = pmb.BART("f", X, Y, m=n_trees)
        tau = pmb.BART("tau", X, Y, m=n_trees)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("Y_obs", mu=f + A * tau, sigma=sigma, observed=Y)
        step = [pmb.PGBART([f]), pmb.PGBART([tau])]
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
            step=step,
            cores=1,
        )
    # Stack chains × draws into a single posterior axis for downstream use.
    f_post = np.asarray(idata.posterior["f"]).reshape(-1, X.shape[0])
    tau_post = np.asarray(idata.posterior["tau"]).reshape(-1, X.shape[0])
    return BcfHahnCate(
        idata=idata, model=model, x_template=X,
        tau_draws=tau_post, f_draws=f_post,
    )


def bcf_hahn_available() -> bool:
    try:
        import pymc  # noqa: F401
        import pymc_bart  # noqa: F401
    except Exception:
        return False
    return True


def fit_method(
    method: str,
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    basis_kind: str,
    seed: int,
    sigma_delta: float = 0.2,
    sigma_0: float = 0.6,
    m_delta: float = 0.0,
    sigma_y: float = 1.0,
    bootstrap_b: int = 0,
    workers: int = 10,
) -> tuple[object, float | str]:
    if method == "B-CALM":
        model = BCalmLinear(sigma_delta=sigma_delta, m_delta=m_delta, sigma0=sigma_0, sigma_y=sigma_y, basis_kind=basis_kind)
        return model.fit(D_R, D_O), sigma_delta
    if method == "B-CALM-selected":
        return fit_bcalm_selected(D_R, D_O, basis_kind, seed, sigma_y=sigma_y)
    if method == "RCT-only":
        model = BCalmLinear(sigma_delta=math.inf, sigma0=math.inf, sigma_y=sigma_y, basis_kind=basis_kind, label=method)
        return model.fit(D_R, None), math.inf
    if method == "CF":
        return _fit_econml_causal_forest(D_R, D_O, seed, bootstrap_b=bootstrap_b, workers=workers), "econml-causal-forest"
    if method == "BCF":
        return _fit_bcf_hahn(D_R, D_O, seed), "pymc-bart"
    if method == "pooled":
        model = BCalmLinear(sigma_delta=0.0, sigma0=0.0, sigma_y=sigma_y, basis_kind=basis_kind, label=method)
        return model.fit(D_R, D_O), 0.0
    if method == "OS-only":
        model = BCalmLinear(sigma_delta=0.0, sigma0=0.0, sigma_y=sigma_y, basis_kind=basis_kind, label=method)
        return model.fit(None, D_O), 0.0
    if method in {"MR-OSCAR", "R-OSCAR", "CALM"}:
        if not EXTERNAL_AVAILABLE:
            raise ImportError(
                f"The '{method}' baseline is a prior calibrated-borrowing method "
                "(R-OSCAR: arXiv:2306.17478; MR-OSCAR: arXiv:2603.17066; "
                "CALM: arXiv:2603.19186) and requires its companion code, which is "
                "not part of this release. Clone https://github.com/AsiaeeLab/calm-cate "
                "and set the environment variable BCALM_EXTERNAL_CALM_PATH to the root "
                "of that checkout to enable these baselines."
            )
        return _fit_oscar_like(method, D_R, D_O, seed, bootstrap_b=bootstrap_b, workers=workers), "external"
    raise ValueError(f"Unknown method: {method}")


def fit_ablation(
    variant: str,
    D_R: dict[str, np.ndarray],
    D_O: dict[str, np.ndarray],
    basis_kind: str,
    seed: int = 0,
    sigma_y: float = 1.0,
) -> object:
    if variant == "Full B-CALM":
        model, _ = fit_bcalm_selected(D_R, D_O, basis_kind, seed, sigma_y=sigma_y)
        return model
    if variant == "No b_delta":
        return BCalmLinear(sigma_delta=0.0, sigma0=0.6, sigma_y=sigma_y, basis_kind=basis_kind).fit(D_R, D_O)
    if variant == "No b0":
        model, _ = fit_bcalm_selected(D_R, D_O, basis_kind, seed, sigma_y=sigma_y, sigma0=0.0)
        return model
    raise ValueError(f"Unknown ablation: {variant}")
