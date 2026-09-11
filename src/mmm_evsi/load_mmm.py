"""Load case-study artifacts (raw CSV, fitted MMM zarr) with pinned sanity checks."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM

from mmm_evsi import config
from mmm_evsi.budgets import (
    BudgetPlan,
    QuarterBudget,
    budget_boxes,
    planned_budget_per_channel,
)


class SanityCheckError(ValueError):
    """Raised when a loaded artifact violates a pinned sanity check."""


def load_case_study_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the full 209-row case-study dataset, downloading it if absent.

    ``path=None`` → ``config.RAW_CSV``. ``config.DATE_COLUMN`` is always parsed
    to ``datetime64`` before returning (all windows included).
    """
    csv_path = Path(path) if path is not None else config.RAW_CSV
    if not csv_path.exists():
        try:
            urllib.request.urlretrieve(config.DATA_URL, csv_path)
        except Exception as err:
            raise RuntimeError(
                f"Failed to download case-study data from {config.DATA_URL} "
                f"to {csv_path}: {err}"
            ) from err
    df = pd.read_csv(csv_path)
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    return df


def select_window(df: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    """Rows of ``df`` with dates in ``[window[0], window[1]]`` inclusive.

    Raises ``ValueError`` if ``window[0] > window[1]`` or nothing is selected.
    """
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    if start > end:
        raise ValueError(
            f"window start {window[0]!r} is after window end {window[1]!r}"
        )
    dates = pd.to_datetime(df[config.DATE_COLUMN])
    selected = df.loc[(dates >= start) & (dates <= end)]
    if selected.empty:
        raise ValueError(f"no rows selected for window {window}")
    return selected


def load_mmm(model_file: str | Path | None = None) -> tuple[MMM, xr.DataTree]:
    """Load the fitted case-study MMM from its zarr store and sanity-check it.

    ``None`` → ``config.MODEL_FILE``. Returns ``(mmm, idata)`` with
    ``idata is mmm.idata``.
    """
    store = Path(model_file) if model_file is not None else config.MODEL_FILE
    if not store.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {store}. Run scripts/fit_case_study.py "
            "first (see docs/contracts/stage1-fit.md)."
        )
    mmm = MMM.load(str(store))
    idata = mmm.idata

    # Sanity check 1: channel coords == config.CHANNEL_COLUMNS (raw mdsp_* names).
    channels = list(
        idata["posterior"]["channel_contribution"].coords["channel"].values
    )
    if channels != config.CHANNEL_COLUMNS:
        raise SanityCheckError(
            f"channel coords mismatch: got {channels}, "
            f"expected {config.CHANNEL_COLUMNS}"
        )

    # Sanity check 2: target column.
    if mmm.target_column != config.SALES_COLUMN:
        raise SanityCheckError(
            f"target_column mismatch: got {mmm.target_column!r}, "
            f"expected {config.SALES_COLUMN!r}"
        )

    # Sanity check 3: observed dates within the train window (fit on train only).
    obs_dates = idata["observed_data"]["date"]
    if (
        obs_dates.min() < pd.Timestamp(config.TRAIN_WINDOW[0])
        or obs_dates.max() > pd.Timestamp(config.TRAIN_WINDOW[1])
    ):
        raise SanityCheckError(
            "observed_data date range "
            f"[{obs_dates.min()}, {obs_dates.max()}] is not contained in "
            f"TRAIN_WINDOW {config.TRAIN_WINDOW}"
        )

    # Sanity check 4: non-empty controls, all hldy_* prefixed.
    if not mmm.control_columns or not all(
        col.startswith(config.CONTROL_COLUMNS_PREFIX)
        for col in mmm.control_columns
    ):
        raise SanityCheckError(
            "control_columns must be non-empty and every entry must start with "
            f"{config.CONTROL_COLUMNS_PREFIX!r}; got {mmm.control_columns}"
        )

    return mmm, idata


def _quarter_budget(df: pd.DataFrame, window: tuple[str, str]) -> QuarterBudget:
    planned = planned_budget_per_channel(df, window)
    total = float(sum(planned.values()))
    boxes = budget_boxes(planned)
    # Pinned QuarterBudget invariant (contract §2.2).
    assert planned.keys() == set(config.CHANNEL_COLUMNS), planned.keys()
    assert total == sum(planned.values()), (total, sum(planned.values()))
    assert boxes == budget_boxes(planned), boxes
    return QuarterBudget(window=window, planned=planned, total=total, boxes=boxes)


def load_budgets(df: pd.DataFrame | None = None) -> BudgetPlan:
    """Planned Q1/Q2 budgets from the raw weekly data (or the auto-loaded CSV).

    A ``DataTree``/idata is rejected with ``TypeError``: the posterior tree
    contains no Q1/Q2 spend (only training-window channels + target).
    """
    if isinstance(df, xr.DataTree):
        raise TypeError(
            "load_budgets requires the raw weekly DataFrame (or None for the "
            "auto-loaded CSV), not a DataTree/idata: the posterior tree has no "
            "Q1/Q2 planned spend."
        )
    if df is None:
        df = load_case_study_data()
    return BudgetPlan(
        q1=_quarter_budget(df, config.Q1_WINDOW),
        q2=_quarter_budget(df, config.Q2_WINDOW),
    )