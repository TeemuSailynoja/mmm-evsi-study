"""Analytic conjugate Gaussian-Gaussian toy (Stage 2, gates G2.0-G2.3).

Scalar model: prior ``theta ~ N(0, tau^2)``; Q1 likelihood under allocation
``a`` is ``y*_t | theta, a ~ N(theta, sigma^2 / (1 + a))`` for ``t = 1..t1``
(higher ``a`` = more precise observation = more information). Q2 utility
``u2(d, theta) = 2 d theta - d^2`` (linear-quadratic; maximizer ``d* = theta``,
realized value ``theta^2``). All quantities have exact closed forms (see
docs/contracts/stage2-weighted.md section 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

LOG_2PI = np.log(2 * np.pi)


@dataclass(frozen=True)
class ToySpec:
    tau: float = 2.0  # prior sd
    sigma: float = 3.0  # observation noise at a=0
    t1: int = 13  # quarter length (weeks), == Q1 length
    n_prior_draws: int = 50_000
    seed: int = 0


@dataclass(frozen=True)
class ToyEvsiResult:
    a: float
    utility_mc: float  # mean over outcomes of (resampled-mean)^2
    utility_analytic: float
    value_of_exploration_mc: float
    value_of_exploration_analytic: float
    khats: np.ndarray  # (n_outcomes,)
    weight_ess: np.ndarray  # (n_outcomes,)
    n_skipped: int


def noise_sd(spec: ToySpec, a: float) -> float:
    """Observation sd under allocation a: sigma / sqrt(1 + a)."""
    return spec.sigma / np.sqrt(1.0 + a)


def prior_draws(spec: ToySpec) -> np.ndarray:
    """(S,) prior draws theta_s ~ N(0, tau^2)."""
    rng = np.random.default_rng(spec.seed)
    return rng.normal(0.0, spec.tau, size=spec.n_prior_draws)


def simulate_quarter(spec: ToySpec, a: float, rng: np.random.Generator) -> np.ndarray:
    """(t1,) simulated quarter: draw theta ~ prior, y*_t = theta + noise_sd(a) z_t.

    NOTE: for CRN across arms use ``_simulate_with`` with a shared (theta, z);
    this standalone form is for single-arm simulation.
    """
    theta, z = _draw_outcome(spec, rng)
    return theta + noise_sd(spec, a) * z


def _draw_outcome(
    spec: ToySpec, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    """Shared (theta, standardized noise) for CRN across arms."""
    theta = float(rng.normal(0.0, spec.tau))
    z = rng.standard_normal(spec.t1)
    return theta, z


def _simulate_with(spec: ToySpec, a: float, theta: float, z: np.ndarray) -> np.ndarray:
    return theta + noise_sd(spec, a) * z


def joint_log_likelihood(
    spec: ToySpec, a: float, y_star: np.ndarray
) -> np.ndarray:
    """(S,) joint log-likelihood of the whole quarter per prior draw.

    ``ell_s = sum_t log Normal(y*_t | theta_s, sigma^2/(1+a))``, vectorized
    over the prior draws ``theta_s``.
    """
    y_star = np.asarray(y_star, dtype=float)
    if y_star.ndim != 1 or y_star.shape[0] < 1:
        raise ValueError(f"y_star must be a 1-D vector, got shape {y_star.shape}")
    theta = prior_draws(spec)  # (S,)
    sd = noise_sd(spec, a)
    var = sd * sd
    t = y_star.shape[0]
    ybar = y_star.mean()
    ss = np.sum((y_star - ybar) ** 2)
    # sum_t (y_t - theta_s)^2 = t*(theta_s - ybar)^2 + sum_t (y_t - ybar)^2
    return -0.5 * t * np.log(LOG_2PI * var) - 0.5 * (
        t * (theta - ybar) ** 2 + ss
    ) / var


def posterior_sufficient(
    spec: ToySpec, a: float, y_star: np.ndarray
) -> tuple[float, float]:
    """Analytic posterior (mu*, v) of theta | y*, a."""
    ybar = float(np.asarray(y_star).mean())
    m = spec.t1 * (1.0 + a) / (spec.sigma**2)
    v = 1.0 / (1.0 / (spec.tau**2) + m)
    mu_star = v * m * ybar
    return mu_star, v


def opt_q2_analytic(spec: ToySpec, a: float, y_star: np.ndarray) -> float:
    """Closed-form OptQ2(y*, a) = (mu*)^2."""
    mu_star, _ = posterior_sufficient(spec, a, y_star)
    return mu_star**2


def _u_analytic(spec: ToySpec, a: float) -> float:
    z = spec.t1 * (1.0 + a) * (spec.tau**2) / (spec.sigma**2)
    return spec.tau**2 * z / (1.0 + z)


def value_of_exploration_analytic(spec: ToySpec, a: float, a0: float = 0.0) -> float:
    """Analytic VoE(a) = U(a) - U(a0)."""
    return _u_analytic(spec, a) - _u_analytic(spec, a0)


def posterior_dataset(spec: ToySpec, draws: np.ndarray) -> xr.Dataset:
    """Wrap prior draws as a (chain=1, draw=S) posterior Dataset for
    ``resample_posterior``."""
    draws = np.atleast_2d(np.asarray(draws, dtype=float))  # (1, S)
    return xr.Dataset(
        {"theta": xr.DataArray(draws, dims=("chain", "draw"), coords={"chain": [0]})}
    )


def run_toy_evsi(
    spec: ToySpec,
    a: float,
    a0: float = 0.0,
    n_outcomes: int = 5_000,
    seed: int = 0,
    resample_size: int | None = None,
) -> ToyEvsiResult:
    """Toy end-to-end VoE with COMMON RANDOM NUMBERS (CRN, mandatory).

    Each outcome draws ONE shared (theta, standardized z); the a-arm and the
    a0-arm differ only by their observation sd, so the MC SE of the VoE
    difference is ~2-3% at N=5000 (the G2.0 tolerance assumes CRN).

    Pipeline per outcome: simulate y* -> joint log-likelihood -> PSIS weights
    (``importance.psis_weights``) -> resample prior draws (``importance.
    resample_posterior``) -> weighted Q2 value = (resampled mean)^2.
    """
    from mmm_evsi.importance import psis_weights, resample_posterior

    rng = np.random.default_rng(seed)
    theta = prior_draws(spec)  # shared (S,) prior draws for weighting
    post = posterior_dataset(spec, theta)

    sum_a = 0.0
    sum_a0 = 0.0
    n_a = 0
    n_a0 = 0
    khats = np.empty(n_outcomes)
    ess = np.empty(n_outcomes)
    n_skipped = 0

    for i in range(n_outcomes):
        t, z = _draw_outcome(spec, rng)
        y_a = _simulate_with(spec, a, t, z)
        y_a0 = _simulate_with(spec, a0, t, z)

        psis_a = psis_weights(joint_log_likelihood(spec, a, y_a))
        psis_a0 = psis_weights(joint_log_likelihood(spec, a0, y_a0))
        acc_a = not (psis_a.khat > 0.7 or not np.isfinite(psis_a.khat))
        acc_a0 = not (psis_a0.khat > 0.7 or not np.isfinite(psis_a0.khat))
        khats[i] = psis_a.khat
        ess[i] = psis_a.weight_ess
        if not acc_a:
            n_skipped += 1

        if acc_a and acc_a0:
            # Paired (CRN-preserving): both arms use the SAME outcome and the
            # SAME resample seed; skipping either arm would break the pairing.
            pr = resample_posterior(
                post, np.exp(psis_a.smoothed_log_weights), n=resample_size, seed=seed + i
            )
            pr0 = resample_posterior(
                post, np.exp(psis_a0.smoothed_log_weights), n=resample_size, seed=seed + i
            )
            sum_a += float(pr["theta"].mean()) ** 2
            sum_a0 += float(pr0["theta"].mean()) ** 2
            n_a += 1
            n_a0 += 1
        elif not acc_a:
            n_skipped += 1

    utility_mc = sum_a / n_a if n_a else float("nan")
    utility_a0 = sum_a0 / n_a0 if n_a0 else float("nan")
    return ToyEvsiResult(
        a=a,
        utility_mc=utility_mc,
        utility_analytic=_u_analytic(spec, a),
        value_of_exploration_mc=utility_mc - utility_a0,
        value_of_exploration_analytic=value_of_exploration_analytic(spec, a, a0),
        khats=khats,
        weight_ess=ess,
        n_skipped=n_skipped,
    )
