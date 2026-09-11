"""Baseline (optimal allocation within planned +/-30% boxes) solves."""
from __future__ import annotations

from dataclasses import dataclass

import xarray as xr
from pymc_marketing.mmm import MMM
from scipy.optimize import OptimizeResult

from mmm_evsi import config
from mmm_evsi.budgets import QuarterBudget


@dataclass(frozen=True)
class BaselineResult:
    """Outcome of one quarter's pinned budget-optimizer solve."""

    quarter: str  # "Q1" | "Q2"
    window: tuple[str, str]
    budgets: xr.DataArray  # dims ("channel",), coords = raw mdsp_* names
    objective_value: float  # expected sales = -float(scipy_result.fun)
    scipy_result: OptimizeResult


def solve_baseline(
    mmm: MMM, window: tuple[str, str], budget_cfg: QuarterBudget
) -> BaselineResult:
    """Solve ``mmm.budget_optimizer`` for one quarter and validate the result.

    Exact pinned call: ``allocate_budget(total_budget=B, budget_bounds=boxes,
    x0=None, minimize_kwargs={"options": {"ftol": 1e-6}})`` with ``x0=None``
    = uniform start. The ``ftol=1e-6`` override is REQUIRED: the pinned
    optimizer defaults SLSQP to ``ftol=1e-9``, which fails the Q1 solve with
    ``MinimizeException: Positive directional derivative for linesearch`` on
    an objective of scale ~5e9 (deterministic across seeds/draw counts;
    verified 2026-09-11, see docs/LOG.md). ``ftol=1e-6`` (scipy default)
    converges Q1 and Q2 to the identical optimum.
    Post-solve validation raises ``RuntimeError`` (naming the violated
    constraint) unless the optimizer reports success,
    ``|sum(budgets) - B| <= 1e-6``, and every channel sits in its box +/-1e-6.
    ``objective_value`` is the maximized expected sales over the 13-week
    window: ``-float(scipy_result.fun)``.
    """
    if window == config.Q1_WINDOW:
        quarter = "Q1"
    elif window == config.Q2_WINDOW:
        quarter = "Q2"
    else:
        raise ValueError(
            f"unsupported baseline window {window}; expected "
            f"config.Q1_WINDOW {config.Q1_WINDOW} or "
            f"config.Q2_WINDOW {config.Q2_WINDOW}"
        )

    opt = mmm.budget_optimizer(window[0], window[1])
    res = opt.allocate_budget(
        total_budget=budget_cfg.total,
        budget_bounds=budget_cfg.boxes,
        x0=None,  # x0=None = uniform start
        minimize_kwargs={"options": {"ftol": 1e-6}},
    )

    if not res.scipy_result.success:
        raise RuntimeError(
            f"baseline {quarter} {window}: optimizer did not converge "
            f"(success=False; {res.scipy_result.message})"
        )

    total = float(res.budgets.sum())
    if abs(total - budget_cfg.total) > 1e-6:
        raise RuntimeError(
            f"baseline {quarter} {window}: |sum(budgets) - B| = "
            f"{abs(total - budget_cfg.total)} exceeds 1e-6 (total={total}, "
            f"B={budget_cfg.total})"
        )

    for channel, (lo, hi) in budget_cfg.boxes.items():
        x = float(res.budgets.sel(channel=channel))
        if not (lo - 1e-6 <= x <= hi + 1e-6):
            raise RuntimeError(
                f"baseline {quarter} {window}: channel {channel} budget {x} "
                f"outside box ({lo}, {hi}) +/- 1e-6"
            )

    return BaselineResult(
        quarter=quarter,
        window=window,
        budgets=res.budgets,
        objective_value=-float(res.scipy_result.fun),
        scipy_result=res.scipy_result,
    )