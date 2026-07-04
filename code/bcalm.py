from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from dgp import basis


def _prior_variance(scale: float, size: int) -> np.ndarray:
    if math.isinf(float(scale)):
        return np.full(size, 1.0e10)
    if float(scale) <= 0.0:
        return np.full(size, 1.0e-10)
    return np.full(size, float(scale) ** 2)


@dataclass
class BCalmLinear:
    sigma_delta: float = 0.2
    m_delta: float = 0.0
    sigma0: float = 0.6
    sigma_mu: float = 3.0
    sigma_tau: float = 3.0
    sigma_y: float = 1.0
    basis_kind: str = "linear"
    label: str = "B-CALM"

    def _system(
        self,
        D_R: dict[str, np.ndarray] | None,
        D_O: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        blocks: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        k = None
        if D_R is not None and len(D_R.get("Y", [])) > 0:
            Phi_R = basis(D_R["X"], self.basis_kind)
            k = Phi_R.shape[1]
            A_R = D_R["A"][:, None]
            z = np.zeros_like(Phi_R)
            blocks.append(np.column_stack([Phi_R, A_R * Phi_R, z, z]))
            y_parts.append(D_R["Y"])
        if D_O is not None and len(D_O.get("Y", [])) > 0:
            Phi_O = basis(D_O["X"], self.basis_kind)
            if k is None:
                k = Phi_O.shape[1]
            A_O = D_O["A"][:, None]
            blocks.append(np.column_stack([Phi_O, A_O * Phi_O, Phi_O, A_O * Phi_O]))
            y_parts.append(D_O["Y"])
        if not blocks or k is None:
            raise ValueError("At least one non-empty data source is required.")
        X_design = np.vstack(blocks)
        y = np.concatenate(y_parts)
        prior_var = np.concatenate(
            [
                _prior_variance(self.sigma_mu, k),
                _prior_variance(self.sigma_tau, k),
                _prior_variance(self.sigma0, k),
                _prior_variance(self.sigma_delta, k),
            ]
        )
        prior_mean = np.zeros(4 * k)
        prior_mean[3 * k] = self.m_delta
        return X_design, y, prior_var, prior_mean, k

    def fit(self, D_R: dict[str, np.ndarray] | None, D_O: dict[str, np.ndarray] | None) -> "BCalmLinear":
        X_design, y, prior_var, prior_mean, k = self._system(D_R, D_O)
        prior_prec = np.diag(1.0 / prior_var)
        precision = prior_prec + (X_design.T @ X_design) / (self.sigma_y**2)
        rhs = prior_prec @ prior_mean + (X_design.T @ y) / (self.sigma_y**2)
        jitter = 1.0e-8 * np.eye(precision.shape[0])
        chol = cho_factor(precision + jitter, lower=True, check_finite=False)
        mean = cho_solve(chol, rhs, check_finite=False)
        cov = cho_solve(chol, np.eye(precision.shape[0]), check_finite=False)
        self.k_ = k
        self.mean_ = mean
        self.cov_ = 0.5 * (cov + cov.T)
        self.precision_ = precision
        self.n_R_ = 0 if D_R is None else len(D_R.get("Y", []))
        self.n_O_ = 0 if D_O is None else len(D_O.get("Y", []))
        self.resid_sd_ = self._residual_sd(X_design, y)
        logdet_prior_prec = float(np.sum(np.log(1.0 / prior_var)))
        logdet_post_prec = float(2.0 * np.sum(np.log(np.diag(chol[0]))))
        y_quad = float((y @ y) / (self.sigma_y**2))
        prior_quad = float(prior_mean @ (prior_prec @ prior_mean))
        post_quad = float(rhs @ mean)
        n = len(y)
        self.log_marginal_likelihood_ = (
            -0.5 * n * math.log(2.0 * math.pi * self.sigma_y**2)
            + 0.5 * logdet_prior_prec
            - 0.5 * logdet_post_prec
            - 0.5 * (y_quad + prior_quad - post_quad)
        )
        return self

    def _residual_sd(self, X_design: np.ndarray, y: np.ndarray) -> float:
        resid = y - X_design @ self.mean_
        df = max(len(y) - X_design.shape[1], 1)
        return float(np.sqrt(np.sum(resid**2) / df))

    @property
    def tau_slice(self) -> slice:
        return slice(self.k_, 2 * self.k_)

    def posterior_cate_moments(self, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Phi = basis(X_eval, self.basis_kind)
        mean_tau = Phi @ self.mean_[self.tau_slice]
        cov_tau = self.cov_[self.tau_slice, self.tau_slice]
        var_tau = np.einsum("ij,jk,ik->i", Phi, cov_tau, Phi)
        return mean_tau, np.maximum(var_tau, 1.0e-12)

    def posterior_cate(self, X_eval: np.ndarray, n_samples: int = 500, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        Phi = basis(X_eval, self.basis_kind)
        cov_tau = self.cov_[self.tau_slice, self.tau_slice]
        cov_tau = 0.5 * (cov_tau + cov_tau.T) + 1.0e-10 * np.eye(cov_tau.shape[0])
        draws = rng.multivariate_normal(self.mean_[self.tau_slice], cov_tau, size=n_samples)
        return draws @ Phi.T

    def predict_rct_mean(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        Phi = basis(X, self.basis_kind)
        mu = Phi @ self.mean_[: self.k_]
        tau = Phi @ self.mean_[self.tau_slice]
        return mu + A * tau

    def tau_intercept_var(self) -> float:
        return float(self.cov_[self.k_, self.k_])

    def scaled_covariance(self, scale: float) -> "BCalmLinear":
        self.cov_ = self.cov_ * float(scale)
        return self


@dataclass
class BCalmMixture:
    components: list[BCalmLinear]
    weights: np.ndarray
    label: str = "B-CALM"

    def posterior_cate_moments(self, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = []
        variances = []
        for component in self.components:
            mean, var = component.posterior_cate_moments(X_eval)
            means.append(mean)
            variances.append(var)
        mean_stack = np.vstack(means)
        var_stack = np.vstack(variances)
        weights = self.weights[:, None]
        mean = np.sum(weights * mean_stack, axis=0)
        second = np.sum(weights * (var_stack + mean_stack**2), axis=0)
        return mean, np.maximum(second - mean**2, 1.0e-12)

    def posterior_cate(self, X_eval: np.ndarray, n_samples: int = 500, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n_samples, self.weights)
        draws = []
        for component, count in zip(self.components, counts):
            if count:
                draws.append(component.posterior_cate(X_eval, n_samples=int(count), seed=int(rng.integers(1, 2**31 - 1))))
        out = np.vstack(draws) if draws else np.empty((0, X_eval.shape[0]))
        if out.shape[0] > 1:
            rng.shuffle(out, axis=0)
        return out

    def predict_rct_mean(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        preds = np.vstack([component.predict_rct_mean(X, A) for component in self.components])
        return np.sum(self.weights[:, None] * preds, axis=0)

    def tau_intercept_var(self) -> float:
        means = np.array([component.mean_[component.k_] for component in self.components])
        variances = np.array([component.tau_intercept_var() for component in self.components])
        mean = float(np.sum(self.weights * means))
        second = float(np.sum(self.weights * (variances + means**2)))
        return max(second - mean**2, 1.0e-12)

    def scaled_covariance(self, scale: float) -> "BCalmMixture":
        for component in self.components:
            component.scaled_covariance(scale)
        return self


def empty_source() -> dict[str, np.ndarray]:
    return {"X": np.empty((0, 1)), "A": np.empty(0), "Y": np.empty(0)}
