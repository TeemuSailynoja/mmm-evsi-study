"""G1.4 — sampler health: pooled ESS, R-hat, divergences (fit artifacts).

Opens the standalone ``IDATA_FILE`` snapshot and derives the gated
diagnostics exactly as ``mmm_evsi.diagnostics.sampler_diagnostics`` does
(Section 2.3 of docs/contracts/stage1-fit.md): pooled ``az.summary``
(chain+draw combined), NaNs dropped from ``r_hat`` before the max, and the
sum of the ``diverging`` indicator. Skips until the Stage 1 artifacts exist;
also cross-checks the numbers recorded in ``fit_summary.json``.
"""
from __future__ import annotations

import json

import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 1 artifacts missing — run scripts/fit_case_study.py; "
    "see docs/contracts/stage1-fit.md"
)

_idata_file = getattr(config, "IDATA_FILE", None)
_model_file = getattr(config, "MODEL_FILE", None)
if not (
    _idata_file and _idata_file.is_dir() and _model_file and _model_file.is_dir()
):
    pytest.skip(SKIP_MSG, allow_module_level=True)


def _summary():
    """Return (idata, pooled wide-format az.summary over FREE RVs).

    The summary is limited to the model's free random variables
    (``[var.name for var in mmm.model.free_RVs]``) so the deterministic
    ``*_contribution`` variables (zero-spend weeks -> NaN r-hat/ESS) are
    excluded, matching the fit script's ``sampler_diagnostics`` derivation.
    """
    import arviz as az
    import xarray as xr
    from pymc_marketing.mmm import MMM

    idata = xr.open_datatree(str(_idata_file), engine="zarr")
    mmm = MMM.load(str(_model_file))
    var_names = [var.name for var in mmm.model.free_RVs]
    return idata, az.summary(idata, var_names=var_names, fmt="wide")


def test_pooled_ess_thresholds():
    """Pooled ``ess_bulk``/``ess_tail`` (chains+draws) are >= 2000."""
    _, summ = _summary()
    assert float(summ["ess_bulk"].min()) >= 2_000
    assert float(summ["ess_tail"].min()) >= 2_000


def test_rhat_threshold():
    """``r_hat`` (NaNs dropped) is < 1.01."""
    _, summ = _summary()
    rhat = summ["r_hat"].dropna()
    assert float(rhat.max()) < 1.01


def test_no_divergences():
    """No divergent transition recorded in ``sample_stats``."""
    idata, _ = _summary()
    assert int(idata["sample_stats"]["diverging"].values.sum()) == 0


def test_fit_summary_diagnostics_match_idata():
    """Section 6.2: ``fit_summary.json`` diagnostics equal the idata's numbers."""
    summary_file = getattr(config, "SUMMARY_FILE", None)
    if not (summary_file and summary_file.is_file()):
        pytest.skip(SKIP_MSG)

    idata, summ = _summary()
    rhat = summ["r_hat"].dropna()
    expected = {
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "max_rhat": float(rhat.max()),
        "n_rhat_nan": int(summ["r_hat"].isna().sum()),
        "n_divergences": int(idata["sample_stats"]["diverging"].values.sum()),
    }
    with open(summary_file) as fh:
        diag = json.load(fh)["diagnostics"]
    for key, value in expected.items():
        assert diag[key] == pytest.approx(value, rel=1e-9)