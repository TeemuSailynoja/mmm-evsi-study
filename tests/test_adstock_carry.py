"""G2.3b — the Q1→Q2 carry-in seam on the toy MMM (runs now).

``q2_expected_response`` is the single carry-in-aware response seam (§2.3):
with the posterior and Q2 budgets fixed, changing the Q1 weekly-spend
history must change the expected Q2 response. The gate uses the 50-draw
session-scoped ``toy_mmm`` fixture from ``tests/conftest.py`` with two
feasible Q1 allocations (planned vs 1.25 × planned, inside the Stage-0
[0.7, 1.3] boxes; the contract suggests "planned vs 1.3 × planned", and the
strictly-interior 1.25 × variant is used here).

``toy_mmm`` was fit on a deterministic synthetic DataFrame; the fixture's
data construction is replicated exactly here (same seed, same operations) so
the response path sees the data the model was actually fit on.

W1 implements ``mmm_evsi.optimize_slsqp`` in parallel with this file; the
module skips cleanly until it lands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

slsqp = pytest.importorskip("mmm_evsi.optimize_slsqp")

TOY_CHANNELS = [f"ch{i}" for i in range(1, 8)]
TOY_N_WEEKS = 120
TOY_Q1_WINDOW = ("2021-01-03", "2021-03-28")  # 13 weekly Sundays (Stage-0)
TOY_Q2_WINDOW = ("2021-04-04", "2021-06-27")  # the next 13 weeks
TOY_N_WEEKS_Q = 13
# Stage-0 toy plan (tests/test_baseline_solves.py): sum == 1000.0.
TOY_PLANNED = {
    "ch1": 100.0,
    "ch2": 200.0,
    "ch3": 150.0,
    "ch4": 120.0,
    "ch5": 180.0,
    "ch6": 140.0,
    "ch7": 110.0,
}
# Strictly-interior alternative Q1 allocation inside the [0.7, 1.3] boxes.
TOY_PLANNED_WIDE = {c: 1.25 * v for c, v in TOY_PLANNED.items()}


def _toy_df() -> pd.DataFrame:
    """Replicate tests/conftest.py ``toy_mmm``'s deterministic synthetic data."""
    dates = pd.date_range("2020-01-05", periods=TOY_N_WEEKS, freq="7D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": dates})
    coef = rng.uniform(0.1, 1.0, len(TOY_CHANNELS))
    for c in TOY_CHANNELS:
        df[c] = rng.uniform(10.0, 100.0, TOY_N_WEEKS)
    df["y"] = (
        5.0
        + (df[TOY_CHANNELS].to_numpy() * coef).sum(axis=1)
        + rng.normal(0.0, 2.0, TOY_N_WEEKS)
    )
    return df


def _q2_budgets(planned) -> xr.DataArray:
    """Q2 channel budgets as a ("channel",) DataArray in toy-channel order."""
    return xr.DataArray(
        np.array([planned[c] for c in TOY_CHANNELS], dtype=float),
        dims="channel",
        coords={"channel": TOY_CHANNELS},
    )


def _q1_weekly_spend(planned) -> np.ndarray:
    """Constant weekly spend per channel over the 13 Q1 weeks: (13, 7)."""
    rates = np.array([planned[c] for c in TOY_CHANNELS], dtype=float) / TOY_N_WEEKS_Q
    return np.tile(rates, (TOY_N_WEEKS_Q, 1))


def test_g23b_expected_q2_response_depends_on_q1_spend_history(toy_mmm):
    """Changing the Q1 weekly spend changes the expected Q2 response."""
    df = _toy_df()
    posterior = toy_mmm.idata["posterior"]  # fitted toy MMM (50 draws, 1 chain)
    budgets = _q2_budgets(TOY_PLANNED)

    spend_a = _q1_weekly_spend(TOY_PLANNED)
    spend_b = _q1_weekly_spend(TOY_PLANNED_WIDE)
    assert np.allclose(spend_b.sum(axis=0), 1.25 * spend_a.sum(axis=0))

    resp_a = slsqp.q2_expected_response(
        toy_mmm, posterior, df, spend_a, budgets
    )
    resp_b = slsqp.q2_expected_response(
        toy_mmm, posterior, df, spend_b, budgets
    )
    assert np.isfinite(resp_a) and np.isfinite(resp_b)

    scale = max(1.0, abs(resp_a), abs(resp_b))
    assert abs(resp_a - resp_b) > 1e-6 * scale, (
        f"expected Q2 response must depend on the Q1 weekly spend history: "
        f"resp(planned)={resp_a:.6g}, resp(1.25x planned)={resp_b:.6g}, "
        f"diff={abs(resp_a - resp_b):.6g} <= 1e-6 * {scale:.6g}"
    )