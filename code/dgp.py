from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DgpKind = Literal["linear", "mildly_nonlinear", "nonlinear"]
Regime = Literal["shared-only", "balanced", "mismatch-heavy"]
Bias = Literal["none", "baseline-only", "comparative"]


@dataclass(frozen=True)
class GeneratedData:
    D_R: dict[str, np.ndarray]
    D_O: dict[str, np.ndarray]
    X_eval: np.ndarray
    tau_eval: np.ndarray
    mu_eval: np.ndarray
    meta: dict[str, object]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def balanced_binary(n: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros(n, dtype=float)
    a[: n // 2] = 1.0
    if n % 2:
        a[-1] = rng.binomial(1, 0.5)
    rng.shuffle(a)
    return a


def source_covariates(
    n: int,
    regime: Regime,
    source: Literal["R", "O"],
    rng: np.random.Generator,
    star_like: bool = False,
) -> np.ndarray:
    if star_like:
        female = rng.binomial(1, 0.49, size=n)
        black = rng.binomial(1, 0.32 if source == "R" else 0.38, size=n)
        free_lunch = rng.binomial(1, 0.42 if source == "R" else 0.50, size=n)
        small_class = rng.binomial(1, 0.33 if source == "R" else 0.24, size=n)
        teacher_exp = rng.normal(8.0 if source == "R" else 7.0, 3.0, size=n)
        baseline_score = rng.normal(0.0, 1.0, size=n) - 0.35 * free_lunch + 0.18 * teacher_exp / 10.0
        school_quality = rng.normal(0.0, 0.7, size=n) + 0.2 * (1.0 - free_lunch)
        return np.column_stack(
            [
                baseline_score,
                female - 0.5,
                black - 0.32,
                free_lunch - 0.45,
                small_class - 0.30,
                (teacher_exp - 8.0) / 3.0,
                school_quality,
            ]
        )

    shift_map = {
        "shared-only": np.array([0.05, -0.02, 0.00]),
        "balanced": np.array([0.25, -0.15, 0.10]),
        "mismatch-heavy": np.array([0.60, -0.40, 0.25]),
    }
    noise_map = {"shared-only": 0.10, "balanced": 0.25, "mismatch-heavy": 0.45}
    z = rng.normal(size=(n, 3))
    if source == "O":
        z = z + shift_map[regime]
    eps = rng.normal(scale=noise_map[regime], size=(n, 5))
    x0 = z[:, 0] + eps[:, 0]
    x1 = z[:, 1] + eps[:, 1]
    if regime == "shared-only":
        x2 = z[:, 2] + eps[:, 2]
        x3 = 0.6 * z[:, 0] - 0.2 * z[:, 1] + eps[:, 3]
        x4 = 0.4 * z[:, 2] + eps[:, 4]
    elif regime == "balanced":
        x2 = z[:, 2] + eps[:, 2]
        x3 = (0.7 if source == "R" else 0.45) * z[:, 0] + eps[:, 3]
        x4 = (0.4 if source == "O" else 0.25) * z[:, 1] + eps[:, 4]
    else:
        x2 = (0.9 if source == "R" else 0.35) * z[:, 2] + eps[:, 2]
        x3 = (0.9 if source == "R" else 0.15) * z[:, 0] + eps[:, 3]
        x4 = (0.8 if source == "O" else 0.10) * z[:, 1] + eps[:, 4]
    return np.column_stack([x0, x1, x2, x3, x4])


def true_mu(X: np.ndarray, kind: DgpKind = "linear", star_like: bool = False) -> np.ndarray:
    if star_like:
        return (
            0.15
            + 0.60 * X[:, 0]
            + 0.12 * X[:, 1]
            - 0.22 * X[:, 3]
            + 0.16 * X[:, 5]
            + 0.20 * X[:, 6]
            + 0.08 * np.sin(X[:, 0])
        )
    mu = 0.25 + 0.45 * X[:, 0] - 0.25 * X[:, 1] + 0.18 * X[:, 2] + 0.08 * X[:, 3]
    if kind in ("mildly_nonlinear", "nonlinear"):
        mu = mu + 0.12 * np.sin(X[:, 0]) - 0.08 * X[:, 1] * X[:, 2]
    if kind == "nonlinear":
        mu = mu + 0.10 * np.cos(X[:, 3]) + 0.08 * np.tanh(X[:, 0] * X[:, 4])
    return mu


def true_tau(
    X: np.ndarray,
    kind: DgpKind = "linear",
    constant: bool = False,
    star_like: bool = False,
) -> np.ndarray:
    if constant:
        return np.full(X.shape[0], 0.40)
    if star_like:
        return 0.22 + 0.18 * (1.0 - X[:, 3]) + 0.12 * X[:, 4] + 0.08 * np.tanh(X[:, 0])
    tau = 0.35 + 0.16 * X[:, 0] - 0.10 * X[:, 2]
    if kind in ("mildly_nonlinear", "nonlinear"):
        tau = tau + 0.08 * np.sin(X[:, 1]) + 0.05 * X[:, 0] * X[:, 3]
    if kind == "nonlinear":
        tau = tau + 0.10 * np.tanh(X[:, 4]) - 0.06 * np.cos(X[:, 0] + X[:, 2])
    return tau


def bias_terms(
    X: np.ndarray,
    bias: Bias,
    constant_comparative: bool = False,
    star_like: bool = False,
    bias_strength: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if bias == "none":
        return np.zeros(n), np.zeros(n)
    if star_like:
        b0 = 0.22 + 0.18 * X[:, 3] - 0.12 * X[:, 6]
        template = 0.18 + 0.10 * X[:, 3] - 0.08 * X[:, 4] + 0.05 * np.tanh(X[:, 0])
        b_delta = (float(bias_strength) / 0.2) * template
        if bias == "baseline-only":
            b_delta = np.zeros(n)
        return b0, b_delta
    b0 = 0.20 + 0.15 * X[:, 0] - 0.10 * X[:, 2]
    if bias == "baseline-only":
        return b0, np.zeros(n)
    if constant_comparative:
        return b0, np.full(n, float(bias_strength))
    template = 0.20 + 0.12 * X[:, 0] - 0.08 * X[:, 1] + 0.04 * np.tanh(X[:, 3])
    b_delta = (float(bias_strength) / 0.2) * template
    return b0, b_delta


def generate_data(
    n_R: int,
    n_O: int,
    seed: int,
    regime: Regime = "balanced",
    bias: Bias = "comparative",
    kind: DgpKind = "mildly_nonlinear",
    sigma_y: float = 1.0,
    n_eval: int = 600,
    constant_tau: bool = False,
    constant_comparative: bool = False,
    bias_strength: float = 0.2,
) -> GeneratedData:
    rng = np.random.default_rng(seed)
    X_R = source_covariates(n_R, regime, "R", rng)
    X_O = source_covariates(n_O, regime, "O", rng)
    A_R = balanced_binary(n_R, rng)
    logits = -0.05 + 0.85 * X_O[:, 0] - 0.55 * X_O[:, 1] + 0.35 * X_O[:, 3]
    if bias == "comparative":
        logits = logits + 0.35 * X_O[:, 2]
    A_O = rng.binomial(1, sigmoid(logits)).astype(float)
    mu_R = true_mu(X_R, kind)
    tau_R = true_tau(X_R, kind, constant=constant_tau)
    mu_O = true_mu(X_O, kind)
    tau_O = true_tau(X_O, kind, constant=constant_tau)
    b0_O, bd_O = bias_terms(X_O, bias, constant_comparative=constant_comparative, bias_strength=bias_strength)
    Y_R = mu_R + A_R * tau_R + rng.normal(scale=sigma_y, size=n_R)
    Y_O = mu_O + A_O * tau_O + b0_O + A_O * bd_O + rng.normal(scale=sigma_y, size=n_O)

    X_eval = source_covariates(n_eval, regime, "R", rng)
    tau_eval = true_tau(X_eval, kind, constant=constant_tau)
    mu_eval = true_mu(X_eval, kind)
    return GeneratedData(
        D_R={"X": X_R, "A": A_R, "Y": Y_R},
        D_O={"X": X_O, "A": A_O, "Y": Y_O},
        X_eval=X_eval,
        tau_eval=tau_eval,
        mu_eval=mu_eval,
        meta={
            "n_R": n_R,
            "n_O": n_O,
            "seed": seed,
            "regime": regime,
            "bias": bias,
            "kind": kind,
            "sigma_y": sigma_y,
            "bias_strength": bias_strength if bias == "comparative" else 0.0,
        },
    )


def generate_star_like(
    n_R: int,
    n_O: int,
    seed: int,
    sigma_y: float = 1.0,
    n_eval: int = 700,
) -> GeneratedData:
    rng = np.random.default_rng(seed)
    X_R = source_covariates(n_R, "balanced", "R", rng, star_like=True)
    X_O = source_covariates(n_O, "balanced", "O", rng, star_like=True)
    A_R = balanced_binary(n_R, rng)
    logits = -0.15 + 0.75 * X_O[:, 0] - 0.55 * X_O[:, 3] + 0.30 * X_O[:, 6]
    A_O = rng.binomial(1, sigmoid(logits)).astype(float)
    mu_R = true_mu(X_R, star_like=True)
    tau_R = true_tau(X_R, star_like=True)
    mu_O = true_mu(X_O, star_like=True)
    tau_O = true_tau(X_O, star_like=True)
    b0_O, bd_O = bias_terms(X_O, "comparative", star_like=True)
    Y_R = mu_R + A_R * tau_R + rng.normal(scale=sigma_y, size=n_R)
    Y_O = mu_O + A_O * tau_O + b0_O + A_O * bd_O + rng.normal(scale=sigma_y, size=n_O)
    X_eval = source_covariates(n_eval, "balanced", "R", rng, star_like=True)
    return GeneratedData(
        D_R={"X": X_R, "A": A_R, "Y": Y_R},
        D_O={"X": X_O, "A": A_O, "Y": Y_O},
        X_eval=X_eval,
        tau_eval=true_tau(X_eval, star_like=True),
        mu_eval=true_mu(X_eval, star_like=True),
        meta={"n_R": n_R, "n_O": n_O, "seed": seed, "regime": "star-like", "bias": "comparative"},
    )


def basis(X: np.ndarray, kind: str = "linear") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    parts = [np.ones((X.shape[0], 1))]
    if kind == "constant":
        return parts[0]
    parts.append(X)
    if kind in ("mild", "mildly_nonlinear", "nonlinear", "star"):
        cols = [
            np.sin(X[:, [0]]),
            X[:, [1]] ** 2,
            X[:, [0]] * X[:, [2]],
        ]
        parts.extend(cols)
    if kind in ("nonlinear", "star"):
        parts.extend(
            [
                np.cos(X[:, [min(3, X.shape[1] - 1)]]),
                np.tanh(X[:, [0]] * X[:, [min(4, X.shape[1] - 1)]]),
            ]
        )
    return np.column_stack(parts)
