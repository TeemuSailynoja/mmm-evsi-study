"""Stage-2 weighted Q2 solve (thin SLSQP wrapper) + parallel job runner.

The carry-in-aware response seam is ``importance.response_mu``; this module
wraps it in an SLSQP budget optimization (mechanism (b) of the contract §7.1)
with the Stage-1 constraint structure (Σx = B, ±30% channel boxes) and an
efficient once-compiled objective (the response graph is extracted once and
the budget block of the shared ``channel_data`` is swapped per evaluation).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from mmm_evsi import config
from mmm_evsi.baseline import solve_baseline
from mmm_evsi.budgets import QuarterBudget
from mmm_evsi.importance import _extract_mu_graph
from pymc_marketing.mmm.budget_optimizer import MinimizeException

# ── Allocation cache: key="Q1" or "Q2" ──────────────────────────────────
# Saves compiled allocations so we skip the ~30 s optimizer compile on
# repeated runs.  Stored as plain JSON with channel names as keys.
_ALLOCATION_CACHE: dict[str, dict] = {}


def _allocation_cache_path(quarter: str) -> str:
    return f"data/fit/baseline_{quarter.lower()}_allocation.json"


def _load_cached_allocation(quarter: str) -> xr.DataArray | None:
    """Read a saved allocation from disk, or return None."""
    global _ALLOCATION_CACHE
    if quarter in _ALLOCATION_CACHE:
        return _ALLOCATION_CACHE[quarter]
    path = _allocation_cache_path(quarter)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    channels = d["coords"]["channel"]["data"]
    vals = d["data"]
    alloc = xr.DataArray(
        np.array(vals, dtype=float),
        dims=["channel"],
        coords={"channel": channels},
    )
    _ALLOCATION_CACHE[quarter] = alloc
    return alloc


def _save_allocation(quarter: str, alloc: xr.DataArray) -> None:
    """Persist allocation to disk and cache it."""
    global _ALLOCATION_CACHE
    d = alloc.to_dict()
    _ALLOCATION_CACHE[quarter] = alloc
    with open(_allocation_cache_path(quarter), "w") as f:
        json.dump(d, f, default=str)
    print(f"Saved {quarter} allocation to {_allocation_cache_path(quarter)}")


def get_baseline_allocation(
    mmm,
    q_cfg: QuarterBudget,
    quarter: str = "Q1",
) -> xr.DataArray:
    """Return the baseline allocation, loading from disk if available."""
    cached = _load_cached_allocation(quarter)
    if cached is not None:
        print(f"Loaded cached {quarter} allocation")
        return cached
    print(f"Optimizing {quarter} allocation...")
    window = config.Q1_WINDOW if quarter == "Q1" else config.Q2_WINDOW
    result = solve_baseline(mmm, window, q_cfg)
    _save_allocation(quarter, result.budgets)
    return result.budgets


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
    q2_cfg: QuarterBudget,
) -> float:
    """Expected Q2 sales (13 weeks) with Q1→Q2 adstock carry-in.

    Q2 window = 13 weeks after training end + 14-week offset (Q1 quarter).
    ``q1_weekly_spend`` (13, n_ch); its last ``l_max`` weeks seed the Q2
    adstock state. Mean over all draws in ``posterior``.

    Q2 weekly spend preserves the flighting pattern from ``q2_cfg.weekly_spend``:
    ``weekly[:, j] = baseline[:, j] * (budgets[j] / q2_cfg.planned[j])``.
    """
    from mmm_evsi.importance import response_mu, unpool_posterior

    l_max = int(mmm.adstock.l_max)
    q1_weekly_spend = np.atleast_2d(np.asarray(q1_weekly_spend, dtype=float))
    carry = q1_weekly_spend[-l_max:]
    budgets_vals = np.asarray(budgets.values, dtype=float)
    q2_baseline = np.asarray(q2_cfg.weekly_spend, dtype=float)
    channels = list(budgets.coords["channel"].values)
    scales = budgets_vals / np.array([q2_cfg.planned[c] for c in channels])
    weekly = q2_baseline * scales[np.newaxis, :]
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
    posterior, then allocate. No compilation here.

    SLSQP can fail with "Positive directional derivative for linesearch"
    when the objective is degenerate (e.g. resampled posterior has few
    unique draws or extreme weights). We retry with progressively looser
    tolerances and fallback initial guesses.
    """
    wrapper.set_q1_carry_in(np.asarray(job.q1_weekly_spend, dtype=float))
    wrapper.set_posterior(job.posterior)

    # Try multiple configurations until one succeeds
    attempts = [
        {"x0": job.x0, "ftol": 1e-6},
        {"x0": job.x0, "ftol": 1e-4},
        {"x0": None, "ftol": 1e-4},
        {"x0": None, "ftol": 1e-2},
        {"x0": None, "ftol": 1e-1},
    ]
    last_exc = None
    for attempt in attempts:
        try:
            res = wrapper.allocate_budget(
                total_budget=q2_cfg.total,
                budget_bounds=q2_cfg.boxes,
                x0=attempt["x0"],
                minimize_kwargs={"options": {"ftol": attempt["ftol"]}},
            )
            if res.scipy_result.success:
                break
            # Not successful but didn't raise — continue to next attempt
        except MinimizeException as e:
            last_exc = e
            continue
    else:
        # All attempts failed
        raise RuntimeError(
            f"Q2 weighted solve failed after all retries: {last_exc}"
        )

    if not res.scipy_result.success:
        raise RuntimeError(
            f"Q2 weighted solve did not converge (success=False; "
            f"{res.scipy_result.message})"
        )
    total = float(res.budgets.sum())
    if abs(total - q2_cfg.total) > 1e-2:
        raise RuntimeError(
            f"Q2 weighted solve |sum(x) - B| = {abs(total - q2_cfg.total)} > 1e-2"
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


# Global cache: key = (window_start, window_end)
# Within a single process, the optimizer is compiled once per window.
_q2_optimizer_cache: dict = {}


def _get_q2_optimizer(mmm, window_start, window_end):
    """Get or create a cached ``CarryInBudgetOptimizer``."""
    from mmm_evsi.carry_in_optimizer import CarryInBudgetOptimizer
    
    key = (window_start, window_end)
    if key not in _q2_optimizer_cache:
        _q2_optimizer_cache[key] = CarryInBudgetOptimizer(
            mmm, window_start, window_end
        )
    return _q2_optimizer_cache[key]


def run_weighted_solves(
    mmm,
    df: pd.DataFrame,
    jobs: Sequence[WeightedSolveJob],
    q2_cfg: QuarterBudget,
    n_processes: int | None = None,
) -> list[Q2SolveResult]:
    """Solve a list of weighted Q2 jobs with ONE objective compile.

    The ``CarryInBudgetOptimizer`` is built once (single compile); each job
    then only swaps carry-in + posterior. ``n_processes > 1`` uses forked
    workers that inherit the compiled wrapper copy-on-write (no per-worker
    compile). Results are in job order and deterministic.

    Falls back to serial if ``fork`` start method is unavailable (e.g., on
    Windows or when the process manager uses ``spawn``).
    """
    if n_processes is None:
        n_processes = os.cpu_count() or 1
    jobs = list(jobs)
    wrapper = _get_q2_optimizer(mmm, q2_cfg.window[0], q2_cfg.window[1])
    if n_processes == 1 or len(jobs) <= 1:
        return [_solve_on_wrapper(wrapper, q2_cfg, j) for j in jobs]

    # Try fork context (copy-on-write, no per-worker compile).
    # spawn/forkserver won't work because the compiled PyTensor graph
    # can't be pickled.
    import multiprocessing
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        logger.warning(
            "fork start method unavailable; running %d jobs serially",
            len(jobs),
        )
        return [_solve_on_wrapper(wrapper, q2_cfg, j) for j in jobs]

    global _SHARED_WRAPPER, _SHARED_Q2_CFG
    _SHARED_WRAPPER = wrapper
    _SHARED_Q2_CFG = q2_cfg

    with ctx.Pool(processes=n_processes) as pool:
        return pool.map(_solve_job_shared, jobs)
