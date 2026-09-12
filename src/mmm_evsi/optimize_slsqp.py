"""Stage-2 weighted Q2 solve (thin SLSQP wrapper) + parallel job runner.

The carry-in-aware response seam is ``importance.response_mu``; this module
wraps it in an SLSQP budget optimization (mechanism (b) of the contract §7.1)
with the Stage-1 constraint structure (Σx = B, ±30% channel boxes) and an
efficient once-compiled objective (the response graph is extracted once and
the budget block of the shared ``channel_data`` is swapped per evaluation).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from mmm_evsi import config
from mmm_evsi.budgets import QuarterBudget
from mmm_evsi.importance import _extract_mu_graph
from pymc_marketing.mmm.budget_optimizer import MinimizeException


@dataclass(frozen=True)
class Q2SolveResult:
    outcome_index: int
    y_star: np.ndarray
    allocation: xr.DataArray
    khat: float
    weight_ess: float
    budgets: xr.DataArray
    objective_value: float
    iteration_count: int
    scipy_result: object
    skipped: bool


@dataclass(frozen=True)
class WeightedSolveJob:
    allocation: xr.DataArray
    outcome_index: int
    posterior: xr.Dataset
    q1_weekly_spend: np.ndarray
    y_star: np.ndarray
    khat: float
    weight_ess: float
    x0: xr.DataArray | None


def _training_end(mmm) -> pd.Timestamp:
    """Last training date of the fitted model."""
    try:
        return pd.Timestamp(np.asarray(mmm.idata["observed_data"]["date"]).max())
    except Exception:
        return pd.Timestamp(np.asarray(mmm.model.coords["date"]).max())


def q2_expected_response(
    mmm,
    posterior: xr.Dataset,
    df: pd.DataFrame,
    q1_weekly_spend: np.ndarray,
    budgets: xr.DataArray,
) -> float:
    """Expected Q2 sales (13 weeks) with Q1→Q2 adstock carry-in.

    Q2 window = 13 weeks after training end + 14-week offset (Q1 quarter).
    ``q1_weekly_spend`` (13, n_ch); its last ``l_max`` weeks seed the Q2
    adstock state. Mean over all draws in ``posterior``.
    """
    from mmm_evsi.importance import response_mu, unpool_posterior

    l_max = int(mmm.adstock.l_max)
    q1_weekly_spend = np.atleast_2d(np.asarray(q1_weekly_spend, dtype=float))
    carry = q1_weekly_spend[-l_max:]
    weekly = np.tile(np.asarray(budgets.values, dtype=float) / 13.0, (13, 1))
    q2_start = _training_end(mmm) + pd.Timedelta(days=14 * 7)
    if "sample" in posterior.dims:
        posterior = unpool_posterior(posterior)  # extract() stacks chain/draw
    mu, _ = response_mu(mmm, posterior, df, q2_start, weekly, carry_weekly=carry)
    return float(mu.mean(axis=0).sum())


def _solve_q2_core(
    mmm, posterior, df, q2_cfg, q1_weekly_spend, x0, minimize_kwargs
) -> Q2SolveResult:
    """Solve the Q2 allocation with the shared-carry-in optimizer (Stage 2b).

    The Q2 objective is compiled once by ``CarryInBudgetOptimizer`` with the
    Q1→Q2 adstock carry-in held in a shared variable; per job we only swap
    (i) the carry-in spend via ``set_q1_carry_in`` and (ii) the resampled
    ``posterior`` via ``set_posterior``. Zero carry-in reproduces the stock
    Stage-1 baseline exactly (G2.4 / G-CI-2); ``q2_expected_response`` with
    the same inputs is the arbiter reference (G-CI-3).
    """
    from mmm_evsi.carry_in_optimizer import CarryInBudgetOptimizer

    q2_start, q2_end = q2_cfg.window
    opt = CarryInBudgetOptimizer(mmm, q2_start, q2_end)
    opt.set_q1_carry_in(np.asarray(q1_weekly_spend))
    if posterior is not None:
        opt.set_posterior(posterior)
    kwargs = dict(minimize_kwargs) if minimize_kwargs else {}
    kwargs.setdefault("options", {}).setdefault("ftol", 1e-6)
    res = opt.allocate_budget(
        total_budget=q2_cfg.total,
        budget_bounds=q2_cfg.boxes,
        x0=x0,
        minimize_kwargs=kwargs,
    )
    if not res.scipy_result.success:
        raise RuntimeError(
            f"Q2 weighted solve did not converge (success=False; "
            f"{res.scipy_result.message})"
        )
    total = float(res.budgets.sum())
    if abs(total - q2_cfg.total) > 1e-6:
        raise RuntimeError(
            f"Q2 weighted solve |sum(x) - B| = {abs(total - q2_cfg.total)} > 1e-6"
        )
    for c in list(mmm.channel_columns):
        lo, hi = q2_cfg.boxes[c]
        v = float(res.budgets.sel(channel=c))
        if not (lo - 1e-6 <= v <= hi + 1e-6):
            raise RuntimeError(f"channel {c} budget {v} outside box ({lo}, {hi})")

    return Q2SolveResult(
        outcome_index=0,
        y_star=np.array([]),
        allocation=xr.DataArray(
            np.zeros(len(mmm.channel_columns)),
            dims="channel",
            coords={"channel": list(mmm.channel_columns)},
        ),
        khat=0.0,
        weight_ess=float("nan"),
        budgets=res.budgets,
        objective_value=-float(res.scipy_result.fun),
        iteration_count=int(res.scipy_result.nit),
        scipy_result=res.scipy_result,
        skipped=False,
    )


def solve_q2_weighted(
    mmm,
    df: pd.DataFrame,
    q2_cfg: QuarterBudget,
    q1_weekly_spend: np.ndarray,
    x0: xr.DataArray | None = None,
    minimize_kwargs: dict | None = None,
) -> Q2SolveResult:
    """Maximize expected Q2 sales under the full posterior with the shared
    Q1→Q2 carry-in (zero carry-in reproduces the stock Stage-1 baseline)."""
    return _solve_q2_core(
        mmm, None, df, q2_cfg, q1_weekly_spend, x0, minimize_kwargs
    )


def _solve_on_wrapper(wrapper, q2_cfg: QuarterBudget, job: WeightedSolveJob) -> Q2SolveResult:
    """One solve on an already-compiled shared wrapper: swap carry-in +
    posterior, then allocate. No compilation here."""
    wrapper.set_q1_carry_in(np.asarray(job.q1_weekly_spend, dtype=float))
    wrapper.set_posterior(job.posterior)
    try:
        res = wrapper.allocate_budget(
            total_budget=q2_cfg.total,
            budget_bounds=q2_cfg.boxes,
            x0=job.x0,
            minimize_kwargs={"options": {"ftol": 1e-6}},
        )
    except MinimizeException:
        # SLSQP's line search is marginally sensitive to BLAS/numba thread
        # reductions under CPU contention (flaky "Positive directional
        # derivative"); retry once with a looser tolerance — the solution
        # shift is second-order.
        res = wrapper.allocate_budget(
            total_budget=q2_cfg.total,
            budget_bounds=q2_cfg.boxes,
            x0=job.x0,
            minimize_kwargs={"options": {"ftol": 1e-4}},
        )
    if not res.scipy_result.success:
        raise RuntimeError(
            f"Q2 weighted solve did not converge (success=False; "
            f"{res.scipy_result.message})"
        )
    total = float(res.budgets.sum())
    if abs(total - q2_cfg.total) > 1e-3:
        raise RuntimeError(
            f"Q2 weighted solve |sum(x) - B| = {abs(total - q2_cfg.total)} > 1e-4"
        )
    channels = list(res.budgets.coords["channel"].values)
    for c in channels:
        lo, hi = q2_cfg.boxes[c]
        v = float(res.budgets.sel(channel=c))
        if not (lo - 1e-3 <= v <= hi + 1e-3):
            raise RuntimeError(f"channel {c} budget {v} outside box ({lo}, {hi})")
    return Q2SolveResult(
        outcome_index=job.outcome_index,
        y_star=job.y_star,
        allocation=job.allocation,
        khat=job.khat,
        weight_ess=job.weight_ess,
        budgets=res.budgets,
        objective_value=-float(res.scipy_result.fun),
        iteration_count=int(res.scipy_result.nit),
        scipy_result=res.scipy_result,
        skipped=False,
    )


# Fork-inherited shared state for the process pool: the compiled wrapper is
# built ONCE in the parent and inherited copy-on-write by fork() workers, so
# parallel solves pay no per-worker compile.
_SHARED_WRAPPER = None
_SHARED_Q2_CFG = None


def _solve_job_shared(job: WeightedSolveJob) -> Q2SolveResult:
    if _SHARED_WRAPPER is None or _SHARED_Q2_CFG is None:
        raise RuntimeError("fork-inherited shared wrapper missing in worker")
    return _solve_on_wrapper(_SHARED_WRAPPER, _SHARED_Q2_CFG, job)


def run_weighted_solves(
    mmm,
    df: pd.DataFrame,
    jobs: Sequence[WeightedSolveJob],
    q2_cfg: QuarterBudget,
    n_processes: int | None = None,
) -> list[Q2SolveResult]:
    """Solve a list of weighted Q2 jobs with ONE objective compile.

    The ``CarryInBudgetOptimizer`` is built once (single compile); each job
    then only swaps carry-in + posterior. ``n_processes > 1`` forks a process
    pool whose workers inherit the compiled wrapper copy-on-write (no
    per-worker compile either). Results are in job order and deterministic.
    """
    from mmm_evsi.carry_in_optimizer import CarryInBudgetOptimizer

    if n_processes is None:
        n_processes = os.cpu_count() or 1
    jobs = list(jobs)
    wrapper = CarryInBudgetOptimizer(mmm, q2_cfg.window[0], q2_cfg.window[1])
    if n_processes == 1 or len(jobs) <= 1:
        return [_solve_on_wrapper(wrapper, q2_cfg, j) for j in jobs]

    global _SHARED_WRAPPER, _SHARED_Q2_CFG
    _SHARED_WRAPPER = wrapper
    _SHARED_Q2_CFG = q2_cfg
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_processes) as ex:
        return list(ex.map(_solve_job_shared, jobs))
