"""G1.5 — planned quarterly budgets and ±30% boxes from the raw data.

Runs now (no fit artifacts needed): sums weekly spend per raw ``mdsp_*``
channel over each quarter window from the CSV and checks the ±30% boxes.
Raw channel names are the mandatory keys — human names from
``config.CHANNEL_MAPPING`` are reporting labels only and would raise a
``KeyError`` in the pinned optimizer.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from mmm_evsi import config

budgets = pytest.importorskip("mmm_evsi.budgets")

BOX_PCT = getattr(config, "BOX_PCT", 0.30)
QUARTERS = (config.Q1_WINDOW, config.Q2_WINDOW)


def _planned_by_quarter():
    df = pd.read_csv(config.RAW_CSV)
    return {
        window: budgets.planned_budget_per_channel(df, window)
        for window in QUARTERS
    }


def test_planned_keys_are_the_seven_raw_channels():
    """Planned budgets are keyed by the 7 raw ``mdsp_*`` channel names."""
    for planned in _planned_by_quarter().values():
        assert set(planned.keys()) == set(config.CHANNEL_COLUMNS)


def test_planned_matches_manual_sums():
    """Planned per-channel sums and quarter totals equal manual CSV sums."""
    df = pd.read_csv(config.RAW_CSV)
    dates = pd.to_datetime(df[config.DATE_COLUMN])
    for window, planned in _planned_by_quarter().items():
        mask = dates.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))
        window_df = df.loc[mask]

        for c in config.CHANNEL_COLUMNS:
            assert math.isclose(
                planned[c], float(window_df[c].sum()), rel_tol=1e-9, abs_tol=1e-6
            )

        total = sum(planned.values())
        manual_total = float(window_df[config.CHANNEL_COLUMNS].to_numpy().sum())
        assert math.isclose(total, manual_total, rel_tol=1e-9, abs_tol=1e-6)
        assert abs(total - manual_total) <= 1e-6  # G1.5 gate tolerance


def test_boxes_are_plus_minus_box_pct():
    """``budget_boxes`` = ``(0.7 * p, 1.3 * p)`` per channel, straddling p."""
    for planned in _planned_by_quarter().values():
        boxes = budgets.budget_boxes(planned)
        assert set(boxes.keys()) == set(config.CHANNEL_COLUMNS)
        for c in config.CHANNEL_COLUMNS:
            lo, hi = boxes[c]
            assert math.isclose(
                lo, (1 - BOX_PCT) * planned[c], rel_tol=1e-9, abs_tol=1e-6
            )
            assert math.isclose(
                hi, (1 + BOX_PCT) * planned[c], rel_tol=1e-9, abs_tol=1e-6
            )
            assert lo < planned[c] < hi