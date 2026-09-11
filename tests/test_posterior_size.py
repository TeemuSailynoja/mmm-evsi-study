"""G1.3 — posterior size targets: script defaults (always) + fitted artifacts.

The pure-config test always runs: the fit script's defaults
(4 x 8,000 = 32,000 pooled draws) must meet the 2x case-study target. The
artifact test opens the fitted idata snapshot and skips until the Stage 1
artifacts exist.
"""
from __future__ import annotations

import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 1 artifacts missing — run scripts/fit_case_study.py; "
    "see docs/contracts/stage1-fit.md"
)

TARGET_POOLED_DRAWS = 4 * 8_000  # 32,000 >= 2x the case study's 4 x 4,000


def test_script_defaults_meet_posterior_target():
    """Pure config: ``CHAINS * DRAWS`` meets the 2x target by default."""
    assert config.CHAINS * config.DRAWS >= TARGET_POOLED_DRAWS


def test_fitted_posterior_pooled_draws():
    """Fitted artifact: chain x draw meets the same target (skip if absent)."""
    idata_file = getattr(config, "IDATA_FILE", None)
    if not (idata_file and idata_file.is_dir()):
        pytest.skip(SKIP_MSG)

    import xarray as xr

    idata = xr.open_datatree(str(idata_file), engine="zarr")
    pooled = idata["posterior"].sizes["chain"] * idata["posterior"].sizes["draw"]
    assert pooled >= TARGET_POOLED_DRAWS