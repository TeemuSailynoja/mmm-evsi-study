"""G1.1 — fit artifacts exist and round-trip to the pinned content contract.

The two ``data/fit/`` zarr artifacts are written by
``scripts/fit_case_study.py``; until they exist this whole module skips.
Raw (``mdsp_*``) channel names are the mandatory keys everywhere; human names
from ``config.CHANNEL_MAPPING`` are reporting labels only.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 1 artifacts missing — run scripts/fit_case_study.py; "
    "see docs/contracts/stage1-fit.md"
)

_model_file = getattr(config, "MODEL_FILE", None)
_idata_file = getattr(config, "IDATA_FILE", None)
if not (_model_file and _model_file.is_dir() and _idata_file and _idata_file.is_dir()):
    pytest.skip(SKIP_MSG, allow_module_level=True)


def test_mmm_load_round_trip():
    """``MMM.load`` returns an MMM whose posterior matches the pinned contract."""
    from pymc_marketing.mmm import MMM

    mmm = MMM.load(str(_model_file))
    assert isinstance(mmm, MMM)
    idata = mmm.idata

    assert idata["posterior"].sizes["chain"] == config.CHAINS
    assert idata["posterior"].sizes["draw"] == config.DRAWS

    channels = list(
        idata["posterior"]["channel_contribution"].coords["channel"].values
    )
    assert channels == config.CHANNEL_COLUMNS

    assert "diverging" in idata["sample_stats"]


def test_idata_snapshot_matches_loaded_model():
    """The standalone idata snapshot has identical posterior sizes and draws."""
    import xarray as xr

    from pymc_marketing.mmm import MMM

    mmm = MMM.load(str(_model_file))
    snap = xr.open_datatree(str(_idata_file), engine="zarr")

    assert snap["posterior"].sizes == mmm.idata["posterior"].sizes
    assert snap["posterior"].sizes["chain"] == config.CHAINS
    assert snap["posterior"].sizes["draw"] == config.DRAWS
    assert np.allclose(
        snap["posterior"]["adstock_alpha"].values,
        mmm.idata["posterior"]["adstock_alpha"].values,
    )