"""G0.4 — optimizer capability probe on a 7-channel toy MMM (Option A)."""
from __future__ import annotations

import numpy as np
import xarray as xr

CHANNELS = [f"ch{i}" for i in range(1, 8)]
START_DATE = "2020-08-09"
END_DATE = "2020-11-01"  # inclusive; 13 weekly Sundays, num_periods=13
TOTAL_BUDGET = 1000.0
PLANNED = {
    "ch1": 100.0,
    "ch2": 200.0,
    "ch3": 150.0,
    "ch4": 120.0,
    "ch5": 180.0,
    "ch6": 140.0,
    "ch7": 110.0,
}  # sum == 1000.0
ALL_TRUE_MASK = xr.DataArray(
    np.ones(len(CHANNELS), dtype=bool),
    dims=("channel",),
    coords={"channel": CHANNELS},
)


def _wide_bounds():
    """``{channel: (0, total_budget)}`` for every channel."""
    return {c: (0.0, TOTAL_BUDGET) for c in CHANNELS}


def test_seven_channel_level_budgets(toy_mmm):
    """(i) Exactly 7 channel-level decision variables; total budget wired."""
    opt = toy_mmm.budget_optimizer(START_DATE, END_DATE)

    assert opt._budget_dims == ["channel"]
    assert opt._budget_shape == (7,)
    assert opt.budgets_to_optimize.dims == ("channel",)
    assert opt.budgets_to_optimize.dims == ALL_TRUE_MASK.dims
    assert bool(np.array_equal(opt.budgets_to_optimize.values, ALL_TRUE_MASK.values))
    assert bool(opt.budgets_to_optimize.values.all())  # all 7 optimized
    assert opt.num_periods == 13

    res = opt.allocate_budget(total_budget=TOTAL_BUDGET, budget_bounds=_wide_bounds())

    assert res.scipy_result.success
    assert res.budgets.dims == ("channel",)
    assert res.budgets.size == 7
    assert abs(float(res.budgets.sum()) - TOTAL_BUDGET) <= 1e-6


def test_channel_level_percent_bounds(toy_mmm):
    """(ii) Per-channel ±30% box bounds via dict form."""
    opt = toy_mmm.budget_optimizer(START_DATE, END_DATE)

    lo = {c: 0.7 * PLANNED[c] for c in CHANNELS}
    hi = {c: 1.3 * PLANNED[c] for c in CHANNELS}
    res = opt.allocate_budget(
        total_budget=TOTAL_BUDGET,
        budget_bounds={c: (lo[c], hi[c]) for c in CHANNELS},
    )

    assert res.scipy_result.success
    x = res.budgets.sel(channel=CHANNELS)
    assert float(x.min()) >= min(lo.values()) - 1e-6  # low bound respected
    assert float(x.max()) <= max(hi.values()) + 1e-6  # high bound respected
    for c in CHANNELS:  # per-channel box
        assert x.sel(channel=c).item() >= lo[c] - 1e-6
        assert x.sel(channel=c).item() <= hi[c] + 1e-6
    assert abs(float(res.budgets.sum()) - TOTAL_BUDGET) <= 1e-6


def test_mask_fixes_excluded_channel(toy_mmm):
    """(iii) A False mask cell is fixed at 0 while the others optimize."""
    mask = xr.DataArray(
        np.array([True, True, True, True, True, True, False]),
        dims=("channel",),
        coords={"channel": CHANNELS},
    )
    opt = toy_mmm.budget_optimizer(START_DATE, END_DATE, budgets_to_optimize=mask)

    res = opt.allocate_budget(total_budget=TOTAL_BUDGET, budget_bounds=_wide_bounds())

    assert res.scipy_result.success
    assert int(opt.budgets_to_optimize.values.sum()) == 6
    assert res.scipy_result.x.size == 6  # decision vector = True cells
    assert np.isclose(float(res.budgets.sel(channel="ch7").item()), 0.0, atol=1e-9)
    others = res.budgets.drop_sel(channel="ch7")
    assert bool(np.all(np.isfinite(others.values)))
    assert float(others.min()) >= 0.0
    assert abs(float(others.sum()) - TOTAL_BUDGET) <= 1e-6  # optimized cells take total


def test_hot_start_x0_accepted(toy_mmm):
    """(iv) A warm start reaches SLSQP (labelled and flat forms)."""
    opt = toy_mmm.budget_optimizer(START_DATE, END_DATE)
    mk = {"method": "SLSQP", "options": {"maxiter": 1000}}

    cold = opt.allocate_budget(
        total_budget=TOTAL_BUDGET, budget_bounds=_wide_bounds(), minimize_kwargs=mk
    )
    assert cold.scipy_result.success

    warm = opt.allocate_budget(
        total_budget=TOTAL_BUDGET,
        budget_bounds=_wide_bounds(),
        x0=cold.budgets,
        minimize_kwargs=mk,
    )
    assert warm.scipy_result.success
    assert bool(np.all(np.isfinite(warm.budgets.values)))
    assert abs(float(warm.budgets.sum()) - TOTAL_BUDGET) <= 1e-6
    assert warm.scipy_result.nit <= cold.scipy_result.nit

    # A flat ndarray x0 sized to the decision vector is also accepted.
    x0_flat = np.full(7, TOTAL_BUDGET / 7.0)
    hot2 = opt.allocate_budget(
        total_budget=TOTAL_BUDGET,
        budget_bounds=_wide_bounds(),
        x0=x0_flat,
        minimize_kwargs=mk,
    )
    assert hot2.scipy_result.success
    assert bool(np.all(np.isfinite(hot2.budgets.values)))

    print(
        f"(iv) nit cold={cold.scipy_result.nit} warm={warm.scipy_result.nit}"
        f" hot2={hot2.scipy_result.nit}"
    )
