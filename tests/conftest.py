"""Shared fixtures for the mmm-evsi Stage 0 test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation


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
