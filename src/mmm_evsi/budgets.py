"""Planned quarterly budgets and +/-box constraints, computed from raw weekly data.

Budgets stay out of ``config.py`` (a constants module); spend computations from
the raw data live here. Every dict/box is keyed by the **raw** ``mdsp_*``
channel names (``config.CHANNEL_COLUMNS``) -- never human names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from mmm_evsi import config


def planned_weekly_spend(
    df: pd.DataFrame, window: tuple[str, str]
) -> np.ndarray:
    """Weekly spend per channel over ``window`` (inclusive).

    Returns an array of shape ``(n_weeks, n_channels)`` sorted by date.
    """
    dates = pd.to_datetime(df[config.DATE_COLUMN])
    mask = (dates >= pd.Timestamp(window[0])) & (dates <= pd.Timestamp(window[1]))
    df_window = df.loc[mask].sort_values(config.DATE_COLUMN).reset_index(drop=True)
    return df_window[config.CHANNEL_COLUMNS].to_numpy(dtype=float)


def planned_budget_per_channel(
    df: pd.DataFrame, window: tuple[str, str]
) -> dict[str, float]:
    """Sum each kept channel's weekly spend over ``window`` (inclusive).

    Keyed by raw ``mdsp_*`` names (``config.CHANNEL_COLUMNS``).
    """
    dates = pd.to_datetime(df[config.DATE_COLUMN])
    mask = (dates >= pd.Timestamp(window[0])) & (dates <= pd.Timestamp(window[1]))
    df_window = df.loc[mask]
    return {channel: float(df_window[channel].sum()) for channel in config.CHANNEL_COLUMNS}


def budget_boxes(
    planned: Mapping[str, float], box_pct: float = config.BOX_PCT
) -> dict[str, tuple[float, float]]:
    """Per-channel ``(1 - box_pct) * p`` … ``(1 + box_pct) * p`` boxes.

    Raises ``ValueError`` if ``planned`` names a channel outside
    ``config.CHANNEL_COLUMNS`` or misses one of them.
    """
    missing = set(config.CHANNEL_COLUMNS) - set(planned)
    unknown = set(planned) - set(config.CHANNEL_COLUMNS)
    if missing or unknown:
        raise ValueError(
            "planned budgets must be keyed by exactly config.CHANNEL_COLUMNS "
            f"(raw mdsp_* names); missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    return {
        channel: ((1 - box_pct) * planned[channel], (1 + box_pct) * planned[channel])
        for channel in planned
    }


@dataclass(frozen=True)
class QuarterBudget:
    """Planned spend of one 13-week quarter over the 7 kept channels."""

    window: tuple[str, str]
    planned: dict[str, float]  # {raw channel: spend in window}
    total: float  # B = sum(planned.values())
    boxes: dict[str, tuple[float, float]]  # (0.7*p, 1.3*p) per channel
    weekly_spend: np.ndarray  # shape (13, n_channels), weekly spend per channel


@dataclass(frozen=True)
class BudgetPlan:
    """Q1 (observation/experiment) and Q2 (decision) planned budgets."""

    q1: QuarterBudget
    q2: QuarterBudget