"""G2.9 — Stage-2 STOP checkpoint smoke (artifact-gated).

A full ``evaluate_allocation`` at the baseline Q1 allocation with the pinned
deterministic tuple ``(seed=0, n_outcomes=config.N_SIM_OUTCOMES=5,
n_processes=1)`` runs end-to-end, and a second identical run reproduces the
first bit-for-bit (allclose atol 1e-9 on ``utility``, ``khats``,
``weight_ess``). ``utility`` is finite, ``n_skipped >= 0`` is reported, and
every accepted solve succeeded (``evaluate_allocation`` raises ``RuntimeError``
naming the violated constraint if any solve fails, so the call returning
implies success). A smoke summary is printed; the orchestrator records it in
docs/LOG.md (PLAN G2.9). Artifact-gated.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 2 modules/artifacts missing — run scripts/fit_case_study.py and "
    "check docs/contracts/stage2-weighted.md"
)

N_SIM_OUTCOMES = getattr(config, "N_SIM_OUTCOMES", 5)
SEED = 0
N_PROCESSES = 1  # serial: single deterministic solve path

_model_file = getattr(config, "MODEL_FILE", None)
_idata_file = getattr(config, "IDATA_FILE", None)
if not (
    _model_file and _model_file.is_dir() and _idata_file and _idata_file.is_dir()
):
    pytest.skip(SKIP_MSG, allow_module_level=True)

from mmm_evsi.baseline import solve_baseline  # noqa: E402
from mmm_evsi.load_mmm import load_budgets, load_case_study_data, load_mmm  # noqa: E402

importance = pytest.importorskip("mmm_evsi.importance")


def _smoke_run(mmm, idata, df, allocation, q1_cfg, q2_cfg):
    return importance.evaluate_allocation(
        mmm, idata, df, allocation, q1_cfg, q2_cfg,
        n_outcomes=N_SIM_OUTCOMES, seed=SEED, n_processes=N_PROCESSES,
    )


def test_g29_stage2_smoke_end_to_end_and_reproducible():
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2
    allocation = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg).budgets

    run1 = _smoke_run(mmm, idata, df, allocation, q1_cfg, q2_cfg)
    run2 = _smoke_run(mmm, idata, df, allocation, q1_cfg, q2_cfg)

    # Determinism: the second identical run reproduces the first.
    assert np.isclose(run1.utility, run2.utility, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        run1.khats, run2.khats, rtol=0.0, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        run1.weight_ess, run2.weight_ess, rtol=0.0, atol=1e-9, equal_nan=True
    )

    # Smoke summary.
    assert run1.n_outcomes == N_SIM_OUTCOMES
    assert run1.n_skipped >= 0
    assert run1.n_skipped == len(run1.skipped_indices)
    assert np.isfinite(run1.utility), "mean utility over accepted quarters"
    assert np.all(np.isfinite(run1.utilities))
    # Every accepted solve succeeded: evaluate_allocation raises RuntimeError
    # (naming the violation: success | |sum(x)-B| <= 1e-6 | boxes +/-1e-6)
    # unless the post-solve validation passes, so returning implies success.

    print(
        "G2.9 smoke summary: "
        f"utility={run1.utility:.9g} "
        f"khats={np.round(run1.khats, 4).tolist()} "
        f"weight_ess={np.round(run1.weight_ess, 1).tolist()} "
        f"n_outcomes={run1.n_outcomes} n_skipped={run1.n_skipped}"
    )