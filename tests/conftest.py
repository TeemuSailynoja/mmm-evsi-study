"""Shared fixtures for the mmm-evsi Stage 0 test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation


# ---------------------------------------------------------------------------
# Stage 3 — BO fixtures (artifact-gated)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mmm():
    """Fitted case-study MMM loaded from zarr (session-scoped).

    Skips the entire test session if the model artifact is missing.
    """
    try:
        from mmm_evsi.load_mmm import load_mmm, SanityCheckError

        mmm, _ = load_mmm()
    except FileNotFoundError:
        pytest.skip(
            "Stage 1 artifacts missing — run scripts/fit_case_study.py first"
        )
    except SanityCheckError as exc:
        pytest.skip(f"Stage 1 artifact sanity check failed: {exc}")
    return mmm


@pytest.fixture(scope="session")
def idata(mmm):
    """Posterior idata from the fitted MMM (``mmm.idata``)."""
    return mmm.idata


@pytest.fixture(scope="session")
def df():
    """Raw case-study DataFrame (weekly spend + target)."""
    from mmm_evsi.load_mmm import load_case_study_data

    return load_case_study_data()


@pytest.fixture(scope="session")
def budgets(df):
    """Q1/Q2 budget plan derived from the raw data."""
    from mmm_evsi.load_mmm import load_budgets

    return load_budgets(df)


@pytest.fixture(scope="session")
def q1_cfg(budgets):
    """Q1 budget config (window, planned, total, boxes)."""
    return budgets.q1


@pytest.fixture(scope="session")
def q2_cfg(budgets):
    """Q2 budget config (window, planned, total, boxes)."""
    return budgets.q2


@pytest.fixture(scope="session")
def baseline_allocation(mmm, q1_cfg):
    """Baseline Q1 allocation from ``solve_baseline`` (7,)."""
    from mmm_evsi.baseline import solve_baseline
    from mmm_evsi import config

    result = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg)
    return result.budgets


@pytest.fixture(scope="session")
def V_Q1_baseline(mmm, idata, df, baseline_allocation, q1_cfg):
    """Expected Q1 sales under baseline (n_draws=100, fast)."""
    from mmm_evsi.bo_design import compute_v_q1

    return compute_v_q1(
        mmm, idata["posterior"].to_dataset(), df,
        baseline_allocation, q1_cfg, n_draws=100,
    )


@pytest.fixture(scope="session")
def toy_mmm():
    """A fitted 7-channel weekly toy MMM for the Stage 0 capability probe.

    Seven channels ``ch1`` … ``ch7`` on 120 weekly Sundays, deterministic
    synthetic spend/target data. The fixture is the base for the **Option A**
    capability probe: all 7 channels are optimizable and weekly budget
    adjustment is deferred (the pinned optimizer is channel-level only).

    The small nutpie fit (draws=50, tune=50, chains=1) takes ~5–15 s and is
    shared session-wide via ``scope="session"``.
    """
    channels = [f"ch{i}" for i in range(1, 8)]
    n = 120
    dates = pd.date_range("2020-01-05", periods=n, freq="7D")
    rng = np.random.default_rng(0)

    df = pd.DataFrame({"date": dates})
    coef = rng.uniform(0.1, 1.0, len(channels))
    for c in channels:
        df[c] = rng.uniform(10.0, 100.0, n)
    df["y"] = 5.0 + (df[channels].to_numpy() * coef).sum(axis=1) + rng.normal(0.0, 2.0, n)

    mmm = MMM(
        date_column="date",
        channel_columns=channels,
        target_column="y",
        adstock=GeometricAdstock(l_max=6),
        saturation=LogisticSaturation(),
        yearly_seasonality=5,
    )
    mmm.fit(
        df.drop(columns=["y"]),
        y=df["y"],
        draws=50,
        tune=50,
        chains=1,
        nuts_sampler="nutpie",
        progressbar=False,
        random_seed=0,
    )
    return mmm
