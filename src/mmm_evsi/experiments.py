"""Q1-outcome simulation under an allocation (Stage 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from mmm_evsi import config


def allocation_to_weekly_spend(
    allocation: xr.DataArray,
    baseline_weekly_spend: np.ndarray,  # (n_weeks, n_channels)
    baseline_quarterly: dict[str, float],
    n_weeks: int = 13,
) -> np.ndarray:
    """Weekly spend preserving flighting pattern: (n_weeks, n_channels).

    For each channel j: ``weekly[:, j] = baseline_weekly[:, j] * (allocation[j] / baseline_quarterly[j])``.

    The sum of weekly spend per channel equals ``allocation[j]``.
    """
    channels = list(allocation.coords.get("channel", []).values)
    if set(channels) != set(baseline_quarterly.keys()):
        raise ValueError(
            f"allocation channels {channels} must match baseline_quarterly keys {list(baseline_quarterly.keys())}"
        )
    vals = np.asarray(allocation.values, dtype=float)
    if np.any(vals < 0):
        raise ValueError("allocation must be non-negative")
    baseline = np.asarray(baseline_weekly_spend, dtype=float)
    if baseline.shape != (n_weeks, len(vals)):
        raise ValueError(
            f"baseline_weekly_spend must be ({n_weeks}, {len(vals)}), got {baseline.shape}"
        )
    scales = vals / np.array([baseline_quarterly[ch] for ch in channels])
    return baseline * scales[np.newaxis, :]


def simulate_quarter(
    mmm,
    idata: xr.DataTree,
    df: pd.DataFrame,
    window: tuple[str, str],
    allocation: xr.DataArray,
    q1_cfg,
    seed: int | None = None,
) -> np.ndarray:
    """(13,) simulated Q1 sales under allocation ``a``.

    Draws one posterior index s uniformly (``rng``), then samples
    ``y*_t ~ Normal(mu_t(theta_s, a), sigma_s)`` via the shared MMM response
    path (``importance._window_response``).
    """
    from mmm_evsi.importance import response_mu

    weekly = allocation_to_weekly_spend(
        allocation, q1_cfg.weekly_spend, q1_cfg.planned, n_weeks=13
    )
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
