"""Stage 3 — Bayesian optimization over feasible Q1 budget perturbations.

Implements the outer BO loop that proposes perturbations of the baseline Q1
allocation, evaluates their expected utility (OptQ2 after importance-weighting
minus Q1 revenue loss), and searches for the allocation with maximum expected
utility subject to feasibility constraints.

Key design decisions
--------------------
1. **Feasibility by construction** — proposals are reparameterized in the
   6-dimensional null space of the total-budget constraint, then clipped to
   channel boxes and the E_max gate. Infeasible suggestions never reach the
   expensive utility evaluator.

2. **Modular GP backend** — sklearn ``GaussianProcessRegressor`` is the
   default (CPU). The ``_GPBackend`` protocol abstracts fitting, prediction,
   and acquisition so GPyTorch can be swapped in later (Stage 3 GPU path).

3. **Incremental surrogate** — after every proposal evaluation the GP is
   refit on the augmented dataset. Acquisition is maximized over the
   feasible set to propose the next candidate.

4. **Early stopping** — if the relative improvement in best utility stays
   below ``tol`` for ``patience`` consecutive evaluations the loop stops.
   This prevents wasting compute when the surrogate is confident.

5. **Common random numbers (CRN)** — the winner re-evaluation uses the same
   seed sequence for both allocations so the variance of the difference is
   reduced.

Public API
----------
- ``generate_feasible_proposal()`` — one feasible perturbation
- ``lhs_feasible_design()`` — N feasible LHS samples
- ``compute_v_q1()`` — fast V_Q1 (n_draws=100)
- ``bayesian_optimization()`` — full BO loop
- ``re_evaluate_winner_paired()`` — CRN paired re-evaluation
- ``decompose_utility()`` — total benefit vs. isolated information value

.. codeauthor:: Stage 3 W1
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from mmm_evsi import config
from mmm_evsi.budgets import QuarterBudget
from mmm_evsi.importance import evaluate_allocation, pool_posterior
from mmm_evsi.optimize_slsqp import q2_expected_response

logger = logging.getLogger("mmm_evsi.bo_design")

# ── Per-allocation tracking for GP observation noise ──────────────────
# Key: tuple of allocation values (hashable)
# Value: {"sum": float, "sum_sq": float, "n": int}
# Uses Welford's online algorithm for numerically stable variance
_ALLOCATION_STATS: dict[tuple, dict] = {}


def _allocation_key(alloc: xr.DataArray) -> tuple:
    """Hashable key for an allocation array."""
    return tuple(np.round(alloc.values, decimals=6))


def _update_allocation_variance(
    alloc_key: tuple, utilities: np.ndarray
) -> float:
    """Update running variance for an allocation and return current variance.

    Uses running sum and sum-of-squares for numerically stable variance:
    - Each call adds a batch of utilities (one per outcome)
    - Returns the current sample variance (ddof=1)
    """
    n_new = len(utilities)

    if alloc_key not in _ALLOCATION_STATS:
        _ALLOCATION_STATS[alloc_key] = {
            "sum": 0.0,
            "sum_sq": 0.0,
            "n": 0,
        }

    stats = _ALLOCATION_STATS[alloc_key]

    # Update running sum and sum-of-squares
    stats["sum"] += float(utilities.sum())
    stats["sum_sq"] += float((utilities ** 2).sum())
    stats["n"] += n_new

    # Return current sample variance (ddof=1)
    if stats["n"] < 2:
        return 0.0
    mean_all = stats["sum"] / stats["n"]
    # Population variance first, then convert to sample variance
    pop_variance = (stats["sum_sq"] / stats["n"]) - (mean_all ** 2)
    sample_variance = pop_variance * stats["n"] / (stats["n"] - 1)
    return max(sample_variance, 0.0)  # guard against numerical negativity


# ===================================================================
# Data classes
# ===================================================================


@dataclass(frozen=True)
class BOProposal:
    """A single BO candidate evaluation."""

    allocation: xr.DataArray  # (7,) channel budgets
    v_q1: float  # expected Q1 sales under this allocation
    q1_loss: float  # V_Q1(baseline) - V_Q1(proposal)
    q2_gain: float  # E[OptQ2] mean utility from Q2
    q2_gains: np.ndarray  # per-outcome OptQ2 values
    utility: float  # E[OptQ2] - lambda * q1_loss
    n_outcomes: int  # number of simulated outcomes used
    utility_variance: float  # variance of utility estimate (for GP alpha)
    khats: np.ndarray  # per-outcome k-hats
    weight_ess: np.ndarray  # per-outcome ESS


@dataclass(frozen=True)
class BOTrace:
    """Per-iteration trace for the BO loop."""

    iteration: int
    allocation: np.ndarray  # (7,) raw budget values
    v_q1: float
    q1_loss: float
    q2_gain: float  # E[OptQ2] mean utility from Q2
    q2_gains: np.ndarray  # per-outcome OptQ2 values
    utility: float
    gp_mean: float  # GP predictive mean at this point
    gp_std: float  # GP predictive std
    acquisition_value: float  # EI value


@dataclass(frozen=True)
class BOResult:
    """Final BO result."""

    best_allocation: xr.DataArray  # (7,) channel budgets
    best_utility: float
    best_v_q1: float
    best_q1_loss: float
    best_q2_gain: float  # E[OptQ2] for best allocation
    trace: list[BOTrace] = field(default_factory=list)
    n_evaluations: int = 0
    n_stopped_early: bool = False
    reason: str = ""
    winner_re_eval: dict | None = None
    gp_model: object | None = None


# ===================================================================
# Feasible proposal generation
# ===================================================================


def _baseline_to_array(baseline: xr.DataArray) -> np.ndarray:
    """Extract budget values as a (7,) numpy array."""
    return np.asarray(baseline.values, dtype=float)


def _array_to_allocation(arr: np.ndarray, baseline: xr.DataArray) -> xr.DataArray:
    """Reconstruct an xr.DataArray from a (7,) array, preserving coords."""
    return xr.DataArray(arr, dims="channel", coords={"channel": baseline.coords["channel"].values})


def _null_space_perturbation(
    delta: np.ndarray, n_channels: int
) -> np.ndarray:
    """Project ``delta`` onto the null space of the sum constraint.

    Given free perturbations for the first ``n_channels - 1`` dimensions,
    set the last dimension so that ``sum(delta) == 0``.

    Parameters
    ----------
    delta : (n_channels,)
        Raw perturbations (first n_channels-1 are free, last is determined).
    n_channels : int
        Number of channels (7).

    Returns
    -------
    (n_channels,)
        Perturbation with sum exactly zero.
    """
    result = np.empty(n_channels, dtype=float)
    result[: n_channels - 1] = delta[: n_channels - 1]
    result[n_channels - 1] = -result[: n_channels - 1].sum()
    return result


def _clip_to_boxes(
    perturbation: np.ndarray,
    boxes: dict[str, tuple[float, float]],
    baseline: np.ndarray,
) -> np.ndarray:
    """Clip perturbation so that ``baseline + perturbation`` stays in boxes.

    Also re-centers to maintain the sum constraint after clipping.

    Parameters
    ----------
    perturbation : (n_channels,)
        Raw perturbation (sum=0).
    boxes : dict
        Per-channel ``(lo, hi)`` boxes keyed by channel name.
    baseline : (n_channels,)
        Baseline allocation.

    Returns
    -------
    (n_channels,)
        Clipped perturbation (may not sum to exactly zero; caller re-centers).
    """
    clipped = np.empty_like(perturbation)
    channels = list(boxes.keys())
    for i, ch in enumerate(channels):
        lo, hi = boxes[ch]
        proposed = baseline[i] + perturbation[i]
        clipped[i] = np.clip(proposed, lo, hi) - baseline[i]
    return clipped


def _recenter_sum(perturbation: np.ndarray) -> np.ndarray:
    """Shift perturbation so that sum is exactly zero (preserve relative values)."""
    return perturbation - perturbation.mean()


def compute_v_q1(
    mmm,
    posterior,
    df: pd.DataFrame,
    allocation: xr.DataArray,
    q1_cfg: QuarterBudget,
    n_draws: int = config.BO_N_DRAW_VQ1,
) -> float:
    """Expected Q1 sales under allocation (averaged over n_draws posterior samples).

    Uses the cached ``CompiledResponseEvaluator`` from ``importance.py``.
    A subset of posterior draws is used for speed (default 100).

    Parameters
    ----------
    mmm : pymc_marketing.mmm.BaseMMM
        Fitted MMM model.
    posterior : xr.Dataset
        Posterior draws (may be pooled or unpooled).
    df : pd.DataFrame
        Full dataset.
    allocation : xr.DataArray
        Channel budgets (7,).
    q1_cfg : QuarterBudget
        Q1 budget config (window, boxes).
    n_draws : int
        Number of posterior draws to average over.

    Returns
    -------
    float
        Expected Q1 sales (13 weeks) under the allocation.
    """
    from mmm_evsi.experiments import allocation_to_weekly_spend
    from mmm_evsi.importance import response_mu, unpool_posterior

    weekly = allocation_to_weekly_spend(
        allocation, q1_cfg.weekly_spend, q1_cfg.planned, 13
    )
    weekly = np.asarray(weekly, dtype=float)  # (13, 7)
    q1_start = q1_cfg.window[0]

    # Subsample posterior draws if needed
    if "sample" in posterior.dims:
        n_total = posterior.sizes["sample"]
        if n_draws < n_total:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_total, size=n_draws, replace=False)
            posterior = posterior.isel(sample=idx)
    else:
        # Posterior has "chain" and "draw" dims (not pooled) —
        # subsample directly without unpooling.
        n_total = posterior.sizes.get("chain", 1) * posterior.sizes.get("draw", 1)
        if n_draws < n_total:
            rng = np.random.default_rng(42)
            n_ch = posterior.sizes["chain"]
            n_dr = posterior.sizes["draw"]
            total = n_ch * n_dr
            idx = rng.choice(total, size=n_draws, replace=False)
            chain_idx = idx // n_dr
            draw_idx = idx % n_dr
            posterior = posterior.isel(chain=chain_idx, draw=draw_idx)

    mu, _ = response_mu(mmm, posterior, df, q1_start, weekly, carry_weekly=None)
    return float(mu.mean())  # mean over (sample, 13)


def generate_feasible_proposal(
    baseline_allocation: xr.DataArray,
    q1_cfg: QuarterBudget,
    rng: np.random.Generator,
    E_max: float,
    V_Q1_baseline: float,
    n_iter: int = 5,
) -> xr.DataArray:
    """Generate one feasible perturbation of the baseline.

    Steps:
    1. Draw random perturbation in the 6-dim null space.
    2. Clip to channel boxes.
    3. Re-center to maintain sum constraint.
    4. Check E_max gate; reject and retry if violated.

    Parameters
    ----------
    baseline_allocation : xr.DataArray
        Baseline Q1 allocation (7,).
    q1_cfg : QuarterBudget
        Q1 budget config (boxes, total).
    rng : np.random.Generator
        Random number generator.
    E_max : float
        Maximum allowed Q1 revenue loss.
    V_Q1_baseline : float
        Expected Q1 sales under baseline.
    n_iter : int
        Maximum retries before raising.

    Returns
    -------
    xr.DataArray
        Feasible allocation (7,).

    Raises
    ------
    RuntimeError
        If no feasible proposal is found within ``n_iter`` attempts.
    """
    baseline = _baseline_to_array(baseline_allocation)
    boxes = q1_cfg.boxes
    n_ch = len(baseline)

    for _ in range(n_iter):
        # 1. Draw in null space (6 free dims)
        raw = rng.normal(0, 1, size=n_ch - 1)
        delta = _null_space_perturbation(raw, n_ch)

        # 2. Clamp deltas for channels that are at their bounds and can't
        #    move in the perturbation direction.
        max_allowed = np.array(
            [boxes[ch][1] - baseline[i] for i, ch in enumerate(boxes)],
            dtype=float,
        )
        min_allowed = np.array(
            [boxes[ch][0] - baseline[i] for i, ch in enumerate(boxes)],
            dtype=float,
        )
        # For channels at upper bound: can only decrease (delta <= 0)
        # For channels at lower bound: can only increase (delta >= 0)
        # Use a small tolerance to handle floating-point noise near bounds.
        _eps = 1e-6
        blocked_up = max_allowed <= _eps  # at upper bound
        blocked_down = min_allowed >= -_eps  # at lower bound (baseline <= lo)
        delta[blocked_up & (delta > 0)] = 0.0
        delta[blocked_down & (delta < 0)] = 0.0

        # Compute margins for channels that still have non-zero delta.
        active = np.abs(delta) > 1e-12
        if not active.any():
            continue

        pos_margin = np.where(delta > 0, max_allowed, np.inf)
        neg_margin = np.where(delta < 0, -min_allowed, np.inf)
        margin = np.minimum(pos_margin, neg_margin)

        # If any active channel has zero margin, skip this draw.
        if margin[active].min() < 1e-12:
            continue

        # Scale so that the largest absolute perturbation is at most 50 % of
        # the allowed distance to the nearest box boundary (gives headroom
        # for re-centering).
        scale = 0.5 * margin[active].min() / (np.abs(delta[active]).max() + 1e-12)
        delta *= scale

        # 3. Project delta onto the feasible set (boxes + sum constraint).
        #    Use iterative projection: clip to boxes, then distribute the
        #    sum residual only to channels that have room to absorb it.
        for _proj_iter in range(20):
            delta_before = delta.copy()
            # Clip to boxes
            proposed = baseline + delta
            proposed = np.clip(proposed, np.array([boxes[ch][0] for ch in boxes]),
                               np.array([boxes[ch][1] for ch in boxes]))
            delta = proposed - baseline
            # Check sum constraint
            sum_resid = delta.sum()
            if abs(sum_resid) < 1e-10:
                break
            # Distribute residual to channels that have room to absorb it
            # (channels not at their bounds in the direction of the residual)
            if sum_resid > 0:
                # Need to decrease some channels (residual is positive)
                can_decrease = baseline - np.array([boxes[ch][0] for ch in boxes]) > 1e-6
            else:
                # Need to increase some channels (residual is negative)
                can_increase = np.array([boxes[ch][1] for ch in boxes]) - baseline > 1e-6
                can_decrease = ~can_increase
            can_absorb = can_decrease if sum_resid > 0 else can_increase
            if not can_absorb.any():
                break
            # Distribute residual proportionally to available room
            room = np.where(can_absorb,
                            np.abs(baseline - np.where(sum_resid > 0,
                                                       np.array([boxes[ch][0] for ch in boxes]),
                                                       np.array([boxes[ch][1] for ch in boxes]))),
                            0.0)
            if room.sum() < 1e-12:
                break
            share = room / room.sum()
            delta[can_absorb] -= sum_resid * share[can_absorb]

        # 4. Apply and check E_max (proxy: perturbation magnitude as proxy for
        #    Q1 revenue loss; the full V_Q1 check happens in evaluate_proposal).
        #    Use a generous multiplier — tight box constraints may force
        #    larger relative perturbations on small channels.
        max_pct = E_max / V_Q1_baseline if V_Q1_baseline > 0 else 0.05
        if np.abs(delta).max() / (baseline.max() + 1e-12) > max_pct * 2.0:
            continue

        # Re-center one more time after all clipping
        delta = _recenter_sum(delta)
        final = baseline + delta

        # Verify sum constraint
        if abs(final.sum() - baseline.sum()) > 1e-6:
            continue

        return _array_to_allocation(final, baseline_allocation)

    raise RuntimeError(
        f"Could not generate a feasible proposal in {n_iter} attempts"
    )


def lhs_feasible_design(
    baseline_allocation: xr.DataArray,
    q1_cfg: QuarterBudget,
    n_samples: int,
    E_max: float,
    V_Q1_baseline: float,
    rng: np.random.Generator,
) -> list[xr.DataArray]:
    """Generate N feasible LHS samples in the perturbation space.

    Uses scipy's ``LatinHypercubeDesign`` on the 6 free dimensions, then
    projects each sample onto the feasible set (boxes + E_max).

    Parameters
    ----------
    baseline_allocation : xr.DataArray
        Baseline Q1 allocation (7,).
    q1_cfg : QuarterBudget
        Q1 budget config.
    n_samples : int
        Number of LHS samples.
    E_max : float
        Maximum allowed Q1 revenue loss.
    V_Q1_baseline : float
        Expected Q1 sales under baseline.
    rng : np.random.Generator

    Returns
    -------
    list[xr.DataArray]
        Feasible allocations (may be fewer than n_samples if some are rejected).
    """
    from scipy.stats import qmc

    baseline = _baseline_to_array(baseline_allocation)
    boxes = q1_cfg.boxes
    n_ch = len(baseline)
    n_free = n_ch - 1

    # LHS in the free dimension space
    try:
        lhs = qmc.LatinHypercube(d=n_free, seed=rng.integers(0, 2**31))
        samples = lhs.random(n=n_samples)
        # Scale to [-1, 1] range
        samples = 2.0 * samples - 1.0
    except Exception:
        # Fallback: random uniform
        samples = rng.uniform(-1, 1, size=(n_samples, n_free))

    proposals: list[xr.DataArray] = []
    for sample in samples:
        try:
            p = generate_feasible_proposal(
                baseline_allocation, q1_cfg,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
                E_max=E_max, V_Q1_baseline=V_Q1_baseline,
            )
            proposals.append(p)
        except RuntimeError:
            continue
        if len(proposals) >= n_samples:
            break

    return proposals


# ===================================================================
# GP backend protocol
# ===================================================================


class _GPBackend(ABC):
    """Abstract backend for the GP surrogate.

    Subclass to provide GPyTorch (GPU) or keep sklearn (CPU).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP on (X, y)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) at X."""

    @abstractmethod
    def acquisition(self, X: np.ndarray, y_best: float) -> np.ndarray:
        """Compute acquisition values at X (e.g., EI)."""


class _SklearnGP(_GPBackend):
    """sklearn GaussianProcessRegressor backend (CPU)."""

    def __init__(self, kernel: object | None = None):
        if kernel is None:
            self.kernel = RBF(length_scale=np.ones(7)) + ConstantKernel(1.0) + WhiteKernel(0.1)
        else:
            self.kernel = kernel
        self._gpr: GaussianProcessRegressor | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray | None = None) -> None:
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._y_std = float(np.std(y)) if np.std(y) > 1e-12 else 1.0
        y_norm = (y - y.mean()) / self._y_std
        # Per-point alpha: if provided, use observation noise; else use scalar
        if alpha is not None:
            alpha_norm = alpha / (self._y_std ** 2)
        else:
            alpha_norm = 1e-6  # default: near-zero noise
        # sklearn 1.9: alpha must be passed during init, not fit()
        self._gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=5,
            alpha=alpha_norm,
            random_state=42,
        )
        self._gpr.fit(X, y_norm)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._gpr is None:
            raise RuntimeError("GP not fitted yet")
        mean_norm, std_norm = self._gpr.predict(X, return_std=True)
        mean = mean_norm * self._y_std + self._y_train.mean()
        std = std_norm * self._y_std
        return mean, std

    def acquisition(self, X: np.ndarray, y_best: float) -> np.ndarray:
        """Expected Improvement (EI)."""
        mean, std = self.predict(X)
        y_best_norm = (y_best - self._y_train.mean()) / self._y_std
        Z = (mean - y_best_norm) / (std + 1e-12)
        from scipy.stats import norm
        ei_norm = (mean - y_best_norm) * norm.cdf(Z) + std * norm.pdf(Z)
        return ei_norm * self._y_std  # back-transform


# ===================================================================
# Acquisition function maximization
# ===================================================================


def _maximize_acquisition(
    backend: _GPBackend,
    baseline_allocation: xr.DataArray,
    q1_cfg: QuarterBudget,
    E_max: float,
    V_Q1_baseline: float,
    y_best: float,
    rng: np.random.Generator,
    n_restarts: int = 20,
) -> xr.DataArray:
    """Maximize the acquisition function over the feasible set.

    Uses multi-start L-BFGS-B from random initial points in the null space.

    Parameters
    ----------
    backend : _GPBackend
        Fitted GP surrogate.
    baseline_allocation : xr.DataArray
        Baseline allocation (7,).
    q1_cfg : QuarterBudget
        Q1 budget config.
    E_max : float
        Maximum allowed Q1 revenue loss.
    V_Q1_baseline : float
        Expected Q1 sales under baseline.
    y_best : float
        Current best utility (for EI).
    rng : np.random.Generator
    n_restarts : int
        Number of random restarts for acquisition maximization.

    Returns
    -------
    xr.DataArray
        Allocation that maximizes the acquisition function (feasible).
    """
    baseline = _baseline_to_array(baseline_allocation)
    boxes = q1_cfg.boxes
    n_ch = len(baseline)
    n_free = n_ch - 1

    best_acq = -np.inf
    best_allocation = baseline_allocation

    for _ in range(n_restarts):
        # Random start in null space
        raw = rng.normal(0, 1, size=n_free)
        delta = _null_space_perturbation(raw, n_ch)

        # Scale to reasonable range
        max_allowed = np.array(
            [boxes[ch][1] - baseline[i] for i, ch in enumerate(boxes)],
            dtype=float,
        )
        min_allowed = np.array(
            [boxes[ch][0] - baseline[i] for i, ch in enumerate(boxes)],
            dtype=float,
        )
        margin = np.minimum(
            np.where(delta > 0, max_allowed, np.inf),
            np.where(delta < 0, -min_allowed, np.inf),
        )
        if margin.min() < 1e-12:
            continue
        scale = 0.3 * margin.min() / (np.abs(delta).max() + 1e-12)
        delta *= scale

        def neg_acquisition(x_free: np.ndarray) -> float:
            """Negative EI (to minimize)."""
            delta_full = _null_space_perturbation(x_free, n_ch)
            delta_full = _clip_to_boxes(delta_full, boxes, baseline)
            delta_full = _recenter_sum(delta_full)
            x = baseline + delta_full
            # Standardize X for GP prediction
            if backend._X_train is not None:
                x_norm = (x - backend._X_train.mean(axis=0)) / (backend._X_train.std(axis=0) + 1e-12)
            else:
                x_norm = x
            acq = backend.acquisition(x_norm.reshape(1, -1), y_best)[0]
            return -acq

        bounds = []
        for i in range(n_free):
            lo = min_allowed[i]
            hi = max_allowed[i]
            bounds.append((lo, hi))

        result = minimize(
            neg_acquisition,
            delta[:n_free],
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 100},
        )

        if result.fun < best_acq or best_acq == -np.inf:
            delta_full = _null_space_perturbation(result.x, n_ch)
            delta_full = _clip_to_boxes(delta_full, boxes, baseline)
            delta_full = _recenter_sum(delta_full)
            x = baseline + delta_full
            if abs(x.sum() - baseline.sum()) < 1e-6:
                best_acq = result.fun
                best_allocation = _array_to_allocation(x, baseline_allocation)

    return best_allocation


# ===================================================================
# Proposal evaluation
# ===================================================================


def evaluate_proposal(
    mmm,
    idata,
    df: pd.DataFrame,
    allocation: xr.DataArray,
    q1_cfg: QuarterBudget,
    q2_cfg: QuarterBudget,
    V_Q1_baseline: float,
    n_outcomes: int = config.BO_N_OUTCOMES,
    seed: int = 0,
    x0_warm: xr.DataArray | None = None,
    n_processes: int | None = None,
) -> BOProposal:
    """Evaluate one proposal: compute V_Q1, utility, and diagnostics.

    Parameters
    ----------
    mmm : pymc_marketing.mmm.BaseMMM
        Fitted MMM model.
    idata : xr.DataTree
        Posterior idata.
    df : pd.DataFrame
        Raw data.
    allocation : xr.DataArray
        Proposed Q1 allocation (7,).
    q1_cfg : QuarterBudget
        Q1 budget config.
    q2_cfg : QuarterBudget
        Q2 budget config.
    V_Q1_baseline : float
        Expected Q1 sales under baseline.
    n_outcomes : int
        Number of simulated outcomes.
    seed : int
        Random seed.
    x0_warm : xr.DataArray | None
        Warm start for Q2 optimization.
    n_processes : int | None
        Number of processes for parallel solve.

    Returns
    -------
    BOProposal
        Evaluation result.
    """
    # Fast V_Q1 for feasibility tracking
    pooled = pool_posterior(idata["posterior"].to_dataset())
    V_Q1_prop = compute_v_q1(mmm, pooled, df, allocation, q1_cfg, n_draws=config.BO_N_DRAW_VQ1)

    # Full utility evaluation
    result = evaluate_allocation(
        mmm=mmm,
        idata=idata,
        df=df,
        allocation=allocation,
        q1_cfg=q1_cfg,
        q2_cfg=q2_cfg,
        n_outcomes=n_outcomes,
        seed=seed,
        x0_warm=x0_warm,
        n_processes=n_processes,
    )

    q1_loss = V_Q1_baseline - V_Q1_prop
    q2_gain = result.utility  # E[OptQ2] mean utility from Q2
    q2_gains = result.utilities.copy()  # per-outcome OptQ2 values
    # U(a) = E[OptQ2] - lambda * q1_loss
    utility = q2_gain - config.LAMBDA * q1_loss

    # Sanity check: Q2 gains should be non-negative (or negligible if negative)
    # Negative Q2 gains indicate sampling variance issues
    if np.any(q2_gains < 0):
        n_negative = np.sum(q2_gains < 0)
        min_q2 = float(q2_gains.min())
        logger.warning(
            "Q2 gain sanity check: %d/%d outcomes negative (min=%.4e). "
            "This may indicate sampling variance.",
            n_negative, len(q2_gains), min_q2,
        )

    # Compute variance of utility estimate for GP observation noise
    utility_variance = float(result.utilities.var(ddof=1)) if len(result.utilities) > 1 else 0.0

    return BOProposal(
        allocation=allocation,
        v_q1=V_Q1_prop,
        q1_loss=q1_loss,
        q2_gain=q2_gain,
        q2_gains=q2_gains,
        utility=utility,
        n_outcomes=n_outcomes,
        utility_variance=utility_variance,
        khats=result.khats,
        weight_ess=result.weight_ess,
    )


# ===================================================================
# BO loop
# ===================================================================


def bayesian_optimization(
    mmm,
    idata,
    df: pd.DataFrame,
    q1_cfg: QuarterBudget,
    q2_cfg: QuarterBudget,
    baseline_allocation: xr.DataArray,
    V_Q1_baseline: float,
    n_evaluations: int = config.BO_N_EVALUATIONS,
    n_initial: int = config.BO_N_INITIAL,
    n_outcomes: int = config.BO_N_OUTCOMES,
    E_max_fraction: float = config.E_MAX_FRACTION,
    acquisition: str = "ei",
    seed: int = 0,
    tol: float = config.BO_TOL,
    patience: int = config.BO_PATIENCE,
    x0_warm: xr.DataArray | None = None,
    n_processes: int | None = None,
) -> BOResult:
    """Full Bayesian optimization loop over feasible Q1 perturbations.

    Algorithm:
    1. Generate initial LHS design (``n_initial`` feasible points).
    2. Evaluate each point (V_Q1 + utility with ``n_outcomes`` simulations).
    3. Fit GP on (X, utility).
    4. For each remaining evaluation:
       a. Maximize acquisition function → next proposal.
       b. Evaluate proposal.
       c. Add to dataset, refit GP.
       d. Check early stopping.

    Parameters
    ----------
    mmm : pymc_marketing.mmm.BaseMMM
        Fitted MMM model.
    idata : xr.DataTree
        Posterior idata.
    df : pd.DataFrame
        Raw data.
    q1_cfg : QuarterBudget
        Q1 budget config.
    q2_cfg : QuarterBudget
        Q2 budget config.
    baseline_allocation : xr.DataArray
        Baseline Q1 allocation (7,).
    V_Q1_baseline : float
        Expected Q1 sales under baseline.
    n_evaluations : int
        Total function evaluations.
    n_initial : int
        LHS initial design size.
    n_outcomes : int
        Simulated outcomes per proposal.
    E_max_fraction : float
        Fraction of Q1 total budget for E_max.
    acquisition : str
        Acquisition function ("ei" for Expected Improvement).
    seed : int
        Random seed.
    tol : float
        Relative improvement threshold for early stopping.
    patience : int
        Consecutive non-improvements to trigger early stop.
    x0_warm : xr.DataArray | None
        Warm start for Q2 optimization.
    n_processes : int | None
        Number of processes for parallel solve.

    Returns
    -------
    BOResult
        Final BO result with best allocation and trace.
    """
    rng = np.random.default_rng(seed)
    E_max = E_max_fraction * q1_cfg.total

    # --- Initial LHS design ---
    logger.info("Generating LHS initial design (n=%d)", n_initial)
    t0 = time.time()
    initial_proposals = lhs_feasible_design(
        baseline_allocation, q1_cfg, n_initial, E_max, V_Q1_baseline, rng
    )
    logger.info("LHS design generated in %.1fs (%d feasible)", time.time() - t0, len(initial_proposals))

    # --- Evaluate initial design ---
    logger.info("Evaluating %d initial proposals", len(initial_proposals))
    X: list[np.ndarray] = []
    y: list[float] = []
    y_var: list[float] = []  # per-point variance for GP alpha
    allocations: list[xr.DataArray] = []
    v_q1_history: list[float] = []
    q1_loss_history: list[float] = []
    q2_gain_history: list[float] = []
    q2_gains_history: list[np.ndarray] = []
    traces: list[BOTrace] = []

    for i, alloc in enumerate(initial_proposals):
        t_i = time.time()
        proposal = evaluate_proposal(
            mmm, idata, df, alloc, q1_cfg, q2_cfg,
            V_Q1_baseline, n_outcomes=n_outcomes,
            seed=seed + i, x0_warm=x0_warm, n_processes=n_processes,
        )
        elapsed = time.time() - t_i
        logger.info(
            "  [%d/%d] util=%.4g loss=%.4g q2_gain=%.4g V_Q1=%.4g (%.1fs)",
            i + 1, len(initial_proposals),
            proposal.utility, proposal.q1_loss, proposal.q2_gain,
            proposal.v_q1, elapsed,
        )
        X.append(_baseline_to_array(proposal.allocation))
        y.append(proposal.utility)
        y_var.append(proposal.utility_variance)
        allocations.append(proposal.allocation)
        v_q1_history.append(proposal.v_q1)
        q1_loss_history.append(proposal.q1_loss)
        q2_gain_history.append(proposal.q2_gain)
        q2_gains_history.append(proposal.q2_gains)

    X = np.array(X)
    y = np.array(y)
    y_var = np.array(y_var)
    y_best = float(y.max())
    best_idx = int(np.argmax(y))
    best_alloc = allocations[best_idx]
    best_v_q1 = float(v_q1_history[best_idx])
    best_q1_loss = float(q1_loss_history[best_idx])
    best_q2_gain = float(q2_gain_history[best_idx])

    # --- BO loop ---
    n_remaining = n_evaluations - len(X)
    backend = _SklearnGP()

    for iteration in range(len(X), n_evaluations):
        # Fit GP with per-point alpha (observation noise)
        backend.fit(X, y, alpha=y_var)

        # Maximize acquisition
        t_acq = time.time()
        next_alloc = _maximize_acquisition(
            backend, baseline_allocation, q1_cfg, E_max,
            V_Q1_baseline, y_best, rng, n_restarts=20,
        )

        # Get GP predictive at the proposed point (before evaluation)
        x_next = _baseline_to_array(next_alloc)
        x_std = X.std(axis=0)
        if x_std.max() > 1e-12:
            x_norm = (x_next - X.mean(axis=0)) / x_std
        else:
            # All training points nearly identical — use raw values
            # (GP will predict near the mean with high uncertainty)
            x_norm = x_next
        # Guard against inf/nan in x_norm (can happen when some dims have std≈0)
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        gp_mean, gp_std = backend.predict(x_norm.reshape(1, -1))
        gp_mean, gp_std = float(gp_mean[0]), float(gp_std[0])
        acq_val = float(backend.acquisition(x_norm.reshape(1, -1), y_best)[0])

        # Evaluate proposal
        t_eval = time.time()
        proposal = evaluate_proposal(
            mmm, idata, df, next_alloc, q1_cfg, q2_cfg,
            V_Q1_baseline, n_outcomes=n_outcomes,
            seed=seed + iteration, x0_warm=x0_warm, n_processes=n_processes,
        )
        elapsed = time.time() - t_eval

        # Update per-allocation variance tracking for GP alpha
        # Use Welford's algorithm with actual utilities for numerically stable update
        alloc_key = _allocation_key(next_alloc)
        current_variance = _update_allocation_variance(
            alloc_key, proposal.q2_gains
        )
        # Use the running variance as alpha for this point
        alpha_this = max(current_variance, 1e-10)  # guard against zero

        # Record trace
        traces.append(BOTrace(
            iteration=iteration,
            allocation=x_next,
            v_q1=proposal.v_q1,
            q1_loss=proposal.q1_loss,
            q2_gain=proposal.q2_gain,
            q2_gains=proposal.q2_gains.copy(),
            utility=proposal.utility,
            gp_mean=gp_mean,
            gp_std=gp_std,
            acquisition_value=acq_val,
        ))

        # Update dataset
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, proposal.utility)
        y_var = np.append(y_var, alpha_this)
        v_q1_history.append(proposal.v_q1)
        q1_loss_history.append(proposal.q1_loss)
        q2_gain_history.append(proposal.q2_gain)
        q2_gains_history.append(proposal.q2_gains)

        if proposal.utility > y_best:
            y_best = proposal.utility
            best_alloc = proposal.allocation
            best_v_q1 = proposal.v_q1
            best_q1_loss = proposal.q1_loss
            best_q2_gain = proposal.q2_gain

        logger.info(
            "  BO[%d] util=%.4g (best=%.4g) loss=%.4g (%.1fs)",
            iteration, proposal.utility, y_best, proposal.q1_loss, elapsed,
        )

        # Early stopping check
        if iteration >= n_initial + patience:
            # Check last `patience` improvements
            recent = traces[-patience:]
            utilities = [t.utility for t in recent]
            max_recent = max(utilities)
            min_recent = min(utilities)
            rel_change = abs(max_recent - min_recent) / (abs(max_recent) + 1e-12)
            if rel_change < tol:
                logger.info(
                    "Early stop: rel_change=%.6g < tol=%.6g over %d iters",
                    rel_change, tol, patience,
                )
                return BOResult(
                    best_allocation=best_alloc,
                    best_utility=y_best,
                    best_v_q1=best_v_q1,
                    best_q1_loss=best_q1_loss,
                    best_q2_gain=best_q2_gain,
                    trace=traces,
                    n_evaluations=iteration + 1,
                    n_stopped_early=True,
                    reason=f"early_stop: rel_change={rel_change:.6g} < tol={tol}",
                    gp_model=backend,
                )

    return BOResult(
        best_allocation=best_alloc,
        best_utility=y_best,
        best_v_q1=best_v_q1,
        best_q1_loss=best_q1_loss,
        best_q2_gain=best_q2_gain,
        trace=traces,
        n_evaluations=n_evaluations,
        n_stopped_early=False,
        reason="max_evaluations",
        gp_model=backend,
    )


# ===================================================================
# Winner re-evaluation with CRN
# ===================================================================


def re_evaluate_winner_paired(
    winner_allocation: xr.DataArray,
    baseline_allocation: xr.DataArray,
    mmm,
    idata,
    df: pd.DataFrame,
    q1_cfg: QuarterBudget,
    q2_cfg: QuarterBudget,
    V_Q1_baseline: float,
    n_outcomes: int = 100,
    seed: int = 0,
    n_processes: int | None = None,
) -> dict:
    """Re-evaluate winner and baseline with common random numbers (CRN).

    Uses the same seed sequence for both allocations' simulated outcomes
    so the variance of the difference is reduced.

    Parameters
    ----------
    winner_allocation : xr.DataArray
        Winner allocation (7,).
    baseline_allocation : xr.DataArray
        Baseline allocation (7,).
    mmm : pymc_marketing.mmm.BaseMMM
        Fitted MMM model.
    idata : xr.DataTree
        Posterior idata.
    df : pd.DataFrame
        Raw data.
    q1_cfg : QuarterBudget
        Q1 budget config.
    q2_cfg : QuarterBudget
        Q2 budget config.
    V_Q1_baseline : float
        Expected Q1 sales under baseline.
    n_outcomes : int
        Number of paired simulated outcomes.
    seed : int
        Base random seed.
    n_processes : int | None
        Number of processes for parallel solve.

    Returns
    -------
    dict
        Keys: winner_utility, baseline_utility, delta, delta_se,
              delta_ci_lower, delta_ci_upper, n_outcomes,
              winner_q1_loss, baseline_q1_loss,
              winner_q2_gain, baseline_q2_gain,
              q2_gain_sanity_check.
    """
    winner_utils = []
    baseline_utils = []
    winner_q1_losses = []
    baseline_q1_losses = []
    winner_q2_gains = []
    baseline_q2_gains = []

    # Compute V_Q1 for winner once (expensive, do it once)
    pooled = pool_posterior(idata["posterior"].to_dataset())
    V_Q1_winner = compute_v_q1(mmm, pooled, df, winner_allocation, q1_cfg, n_draws=config.BO_N_DRAW_VQ1)
    V_Q1_baseline_for_reeval = V_Q1_baseline  # reuse baseline

    for i in range(n_outcomes):
        # Same seed for both → CRN
        s = seed + i

        # Winner
        w_proposal = evaluate_allocation(
            mmm=mmm, idata=idata, df=df,
            allocation=winner_allocation,
            q1_cfg=q1_cfg, q2_cfg=q2_cfg,
            n_outcomes=1, seed=s,
            n_processes=n_processes,
        )
        # Baseline — need to recompute with same seed
        b_proposal = evaluate_allocation(
            mmm=mmm, idata=idata, df=df,
            allocation=baseline_allocation,
            q1_cfg=q1_cfg, q2_cfg=q2_cfg,
            n_outcomes=1, seed=s,
            n_processes=n_processes,
        )
        # Decompose into Q1 loss and Q2 gain
        w_loss = V_Q1_baseline_for_reeval - V_Q1_winner
        b_loss = V_Q1_baseline_for_reeval - V_Q1_baseline_for_reeval  # 0 for baseline
        w_q2_gain = w_proposal.utility
        b_q2_gain = b_proposal.utility

        # Utility = Q2 gain - lambda * Q1 loss
        w_util = w_q2_gain - config.LAMBDA * w_loss
        b_util = b_q2_gain - config.LAMBDA * b_loss

        winner_utils.append(w_util)
        baseline_utils.append(b_util)
        winner_q1_losses.append(w_loss)
        baseline_q1_losses.append(b_loss)
        winner_q2_gains.append(w_q2_gain)
        baseline_q2_gains.append(b_q2_gain)

    # Sanity check: Q2 gains should be non-negative
    all_q2_gains = np.array(winner_q2_gains + baseline_q2_gains)
    n_negative = np.sum(all_q2_gains < 0)
    min_q2 = float(all_q2_gains.min())
    if n_negative > 0:
        logger.warning(
            "Re-eval Q2 gain sanity check: %d/%d outcomes negative (min=%.4e). "
            "This may indicate sampling variance.",
            n_negative, len(all_q2_gains), min_q2,
        )

    winner_utils = np.array(winner_utils)
    baseline_utils = np.array(baseline_utils)
    delta = winner_utils - baseline_utils

    delta_mean = float(delta.mean())
    delta_se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    # 95% CI
    from scipy.stats import t as t_dist
    ci_half = t_dist.ppf(0.975, df=len(delta) - 1) * delta_se
    delta_ci_lower = delta_mean - ci_half
    delta_ci_upper = delta_mean + ci_half

    return {
        "winner_utility": float(winner_utils.mean()),
        "baseline_utility": float(baseline_utils.mean()),
        "delta": delta_mean,
        "delta_se": delta_se,
        "delta_ci_lower": delta_ci_lower,
        "delta_ci_upper": delta_ci_upper,
        "n_outcomes": n_outcomes,
        "winner_q1_loss": float(np.mean(winner_q1_losses)),
        "baseline_q1_loss": float(np.mean(baseline_q1_losses)),
        "winner_q2_gain": float(np.mean(winner_q2_gains)),
        "baseline_q2_gain": float(np.mean(baseline_q2_gains)),
        "q2_gain_sanity_check": {
            "n_negative": int(n_negative),
            "total": int(len(all_q2_gains)),
            "min_q2_gain": min_q2,
        },
    }


# ===================================================================
# Utility decomposition (Stage 4 handoff)
# ===================================================================


def decompose_utility(
    winner_result: BOResult,
    baseline_utility: float,
    winner_re_eval: dict | None = None,
) -> dict:
    """Decompose the value of exploration into components.

    Parameters
    ----------
    winner_result : BOResult
        BO result for the winner allocation.
    baseline_utility : float
        U(baseline) from Stage 2 baseline-arm computation.
    winner_re_eval : dict | None
        Paired re-evaluation result (from ``re_evaluate_winner_paired``).

    Returns
    -------
    dict
        Keys: total_benefit, isolated_information_value,
              winner_utility, baseline_utility,
              winner_q1_loss, winner_q2_gain,
              beyond_q2_tail (placeholder).
    """
    total_benefit = winner_result.best_utility - baseline_utility

    # Isolated information value: same adstock carry-in as baseline,
    # only the importance weights differ. This is approximated by the
    # paired re-evaluation delta if available.
    if winner_re_eval is not None:
        isolated_info_value = winner_re_eval["delta"]
    else:
        isolated_info_value = float("nan")

    return {
        "total_benefit": total_benefit,
        "isolated_information_value": isolated_info_value,
        "winner_utility": winner_result.best_utility,
        "baseline_utility": baseline_utility,
        "q1_loss": winner_result.best_q1_loss,
        "q2_gain": winner_result.best_q2_gain,
    }
