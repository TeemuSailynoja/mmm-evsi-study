"""G2.6 — hot-startable Q2 weighted solve (artifact-gated).

The same weighted problem solved from a warm ``x0`` (a prior solve's
budgets) must converge to the IDENTICAL optimum as the cold ``x0=None``
solve: budgets allclose rtol 1e-4 (hard gate). The iteration ordering
``nit(warm) < nit(cold)`` is asserted but path-dependent per Stage-0 §2.4
(iv); if it flickers the test converts that check to an informational log
message while keeping the optimum-equality gate hard (contract §4 G2.6).
Both iteration counts are logged. Artifact-gated (Stage-1 artifacts +
``mmm_evsi.optimize_slsqp``).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 2 modules/artifacts missing — run scripts/fit_case_study.py and "
    "check docs/contracts/stage2-weighted.md"
)

_model_file = getattr(config, "MODEL_FILE", None)
_idata_file = getattr(config, "IDATA_FILE", None)
if not (
    _model_file and _model_file.is_dir() and _idata_file and _idata_file.is_dir()
):
    pytest.skip(SKIP_MSG, allow_module_level=True)

from mmm_evsi.baseline import solve_baseline  # noqa: E402
from mmm_evsi.load_mmm import load_budgets, load_case_study_data, load_mmm  # noqa: E402

slsqp = pytest.importorskip("mmm_evsi.optimize_slsqp")

_LOGGER = logging.getLogger("mmm_evsi.tests")


def test_g26_hot_start_converges_to_cold_optimum():
    """Warm x0 == cold optimum (hard); nit ordering informational (soft)."""
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2

    # Q1 weekly spend from the Stage-1 baseline Q1 allocation (carry-in set).
    q1_budgets = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg).budgets
    q1_rates = np.array(
        [float(q1_budgets.sel(channel=c)) for c in config.CHANNEL_COLUMNS]
    ) / 13.0
    q1_weekly_spend = np.tile(q1_rates, (13, 1))

    kwargs = {"options": {"ftol": 1e-6}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # benign cold-start
        cold = slsqp.solve_q2_weighted(
            mmm, df, q2_cfg, q1_weekly_spend, x0=None, minimize_kwargs=kwargs
        )
        warm = slsqp.solve_q2_weighted(
            mmm, df, q2_cfg, q1_weekly_spend, x0=cold.budgets,
            minimize_kwargs=kwargs,
        )
    assert not cold.skipped and not warm.skipped

    # Hard gate: the identical optimum, warm or cold.
    assert np.allclose(
        warm.budgets.values, cold.budgets.values, rtol=1e-4, atol=0.0
    ), f"warm budgets {warm.budgets.values} vs cold {cold.budgets.values}"
    assert np.isclose(
        warm.objective_value, cold.objective_value, rtol=1e-6, atol=0.0
    ), (
        f"objective {warm.objective_value:.9g} vs {cold.objective_value:.9g}"
    )

    # Soft gate: iteration ordering (may be path-dependent; informational if
    # it flickers — recorded in docs/LOG.md per contract §4 G2.6).
    nit_cold = cold.iteration_count
    nit_warm = warm.iteration_count
    print(f"G2.6 nits: cold={nit_cold} warm={nit_warm}")
    _LOGGER.info("G2.6 SLSQP iterations: cold=%d warm=%d", nit_cold, nit_warm)
    try:
        assert nit_warm < nit_cold, f"warm nit {nit_warm} >= cold nit {nit_cold}"
    except AssertionError:
        _LOGGER.warning(
            "G2.6 nit ordering flickered (warm=%d >= cold=%d); informational "
            "only — the optimum-equality gate stays hard (contract §4 G2.6).",
            nit_warm,
            nit_cold,
        )