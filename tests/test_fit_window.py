"""G1.2 — fit window constants vs. the raw dataset; select_window; observed span.

Part (a) is fully self-contained (``config`` constants + the raw CSV only, no
``mmm_evsi`` module import), so it runs before any ``mmm_evsi`` module exists.
Part (b) exercises ``mmm_evsi.load_mmm.select_window`` behind
``pytest.importorskip``. Part (c) checks the fitted ``observed_data`` date
span against the train window and skips until the Stage 1 artifacts exist.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 1 artifacts missing — run scripts/fit_case_study.py; "
    "see docs/contracts/stage1-fit.md"
)

WINDOWS = (config.TRAIN_WINDOW, config.Q1_WINDOW, config.Q2_WINDOW)
EXPECTED_WEEKS = (183, 13, 13)


def test_windows_disjoint_ordered_contiguous_and_cover_dataset():
    """(a) Stored windows: disjoint, in order, +7 d contiguous, equal the span."""
    dates = pd.to_datetime(pd.read_csv(config.RAW_CSV)[config.DATE_COLUMN])
    starts = [pd.Timestamp(w[0]) for w in WINDOWS]
    ends = [pd.Timestamp(w[1]) for w in WINDOWS]

    # The dataset is a weekly Sunday series (the +7 d joins rely on it).
    assert dates.is_monotonic_increasing
    assert (dates.diff().dropna() == pd.Timedelta(days=7)).all()

    # Disjoint, in order, contiguous (Q1 start == train end + 7 d, etc.).
    assert starts[0] < ends[0] < starts[1] < ends[1] < starts[2] < ends[2]
    assert starts[1] == ends[0] + pd.Timedelta(days=7)
    assert starts[2] == ends[1] + pd.Timedelta(days=7)

    # The union of the three windows equals the dataset span (209 rows).
    assert dates.min() == starts[0]
    assert dates.max() == ends[2]
    assert len(dates) == sum(EXPECTED_WEEKS)

    in_train = dates.between(starts[0], ends[0])
    in_q1 = dates.between(starts[1], ends[1])
    in_q2 = dates.between(starts[2], ends[2])
    assert bool((in_train | in_q1 | in_q2).all())
    assert int(in_train.sum()) + int(in_q1.sum()) + int(in_q2.sum()) == len(dates)

    # Week counts per stored window: 183 / 13 / 13.
    for window, expected in zip(WINDOWS, EXPECTED_WEEKS):
        mask = dates.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))
        assert int(mask.sum()) == expected


def test_select_window_returns_183_train_rows():
    """(b) ``select_window(df, TRAIN_WINDOW)`` selects exactly the 183 train rows."""
    load_mmm = pytest.importorskip("mmm_evsi.load_mmm")
    df = pd.read_csv(config.RAW_CSV)
    train = load_mmm.select_window(df, config.TRAIN_WINDOW)
    assert len(train) == EXPECTED_WEEKS[0]


def test_observed_data_within_train_window():
    """(c) Fitted ``observed_data`` date range is contained in TRAIN_WINDOW."""
    model_file = getattr(config, "MODEL_FILE", None)
    idata_file = getattr(config, "IDATA_FILE", None)
    if not (model_file and model_file.is_dir() and idata_file and idata_file.is_dir()):
        pytest.skip(SKIP_MSG)

    import xarray as xr

    idata = xr.open_datatree(str(idata_file), engine="zarr")
    dates = idata["observed_data"]["date"].values
    assert pd.Timestamp(dates.min()) >= pd.Timestamp(config.TRAIN_WINDOW[0])
    assert pd.Timestamp(dates.max()) <= pd.Timestamp(config.TRAIN_WINDOW[1])