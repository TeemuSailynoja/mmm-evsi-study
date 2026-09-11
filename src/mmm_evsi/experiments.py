"""Q1-outcome simulation under an allocation (Stage 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from mmm_evsi import config


def allocation_to_weekly_spend(
    allocation: xr.DataArray, n_weeks: int = 13
) -> np.ndarray:
    """Constant weekly rate per channel: (n_weeks, n_channels)."""
    if list(allocation.coords.get("channel", []).values) != config.CHANNEL_COLUMNS:
        raise ValueError("allocation channel coords must equal config.CHANNEL_COLUMNS")
    vals = np.asarray(allocation.values, dtype=float)
    if np.any(vals < 0):
        raise ValueError("allocation must be non-negative")
    return np.tile(vals / n_weeks, (n_weeks, 1))


def simulate_quarter(
    mmm,
    idata: xr.DataTree,
    df: pd.DataFrame,
    window: tuple[str, str],
    allocation: xr.DataArray,
    seed: int | None = None,
) -> np.ndarray:
    """(13,) simulated Q1 sales under allocation ``a``.

    Draws one posterior index s uniformly (``rng``), then samples
    ``y*_t ~ Normal(mu_t(theta_s, a), sigma_s)`` via the shared MMM response
    path (``importance._window_response``).
    """
    from mmm_evsi.importance import response_mu

    weekly = allocation_to_weekly_spend(allocation, n_weeks=13)
    l_max = int(mmm.adstock.l_max)
    train_tail_dates = pd.date_range(
        end=pd.Timestamp(window[0]) - pd.Timedelta(days=7), periods=l_max, freq="7D"
    )
    dfd = df.set_index(mmm.date_column)
    carry = np.column_stack(
        [dfd.loc[train_tail_dates, c].to_numpy() for c in mmm.channel_columns]
    )
    mu_orig, sigma_orig = response_mu(
        mmm, idata["posterior"].to_dataset(), df, window[0], weekly, carry_weekly=carry
    )
    rng = np.random.default_rng(seed)
    s = int(rng.integers(0, mu_orig.shape[0]))
    return rng.normal(mu_orig[s], sigma_orig[s])  # (13,)
