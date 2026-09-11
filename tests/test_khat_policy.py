"""G2.3 (runs now) + G2.3c (artifact-gated) — the k-hat > 0.7 skip policy.

G2.3 — scoped to ``apply_khat_policy`` + its WARNING logging: a forced bad
quarter (degenerate dominant draw → non-finite/oversized k-hat) is skipped
and logged; a genuine toy quarter and the degenerate uniform case are
accepted. Cross-quarter ``n_skipped`` bookkeeping is G2.3c's job.
G2.3c — real-pipeline ``evaluate_allocation`` bookkeeping: ``n_skipped ==
len(skipped_indices)``, ``khats[skipped_indices]`` are exactly the ``nan``
entries of ``khats``, ``len(utilities) == n_outcomes - n_skipped``, and
skipped quarters are never solved. Artifact-gated (per-test skip; the
runs-now G2.3 tests must keep running).

W1 implements ``mmm_evsi.importance`` and ``toy.toy_gaussian`` in parallel
with this file; imports are guarded so collection never breaks.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mmm_evsi import config

importance = pytest.importorskip("mmm_evsi.importance")
toy = pytest.importorskip("toy.toy_gaussian")

K_HAT_THRESHOLD = getattr(config, "K_HAT_THRESHOLD", 0.7)

SKIP_MSG = (
    "Stage 2 modules/artifacts missing — run scripts/fit_case_study.py and "
    "check docs/contracts/stage2-weighted.md"
)

# Forced bad quarter: one dominant draw on an otherwise flat tail — the
# policy must skip it (k-hat > 0.7 on the §0.3 heavy-tail signal, or
# non-finite k-hat from the GPD flat-tail guard — both skip, §0.5/§5).
_ELL_DEGEN = np.full(50_000, -8.0)
_ELL_DEGEN[0] = 0.0


def test_g23_bad_quarter_skipped_and_logged(caplog):
    """G2.3 — forced bad quarter → skipped=True + WARNING log containing 'khat'."""
    psis = importance.psis_weights(_ELL_DEGEN)
    with caplog.at_level(logging.WARNING, logger="mmm_evsi.importance"):
        verdict = importance.apply_khat_policy(psis)
    assert verdict.skipped
    assert verdict.reason, "the verdict must carry a reason string"
    messages = [r.getMessage() for r in caplog.records]
    assert messages, "apply_khat_policy must log a WARNING"
    assert any("khat" in m for m in messages), messages


def test_g23_genuine_quarter_not_skipped():
    """G2.3 — a genuine toy quarter (fixed seed 0, a = 2.0) is accepted."""
    spec = toy.ToySpec()
    y_star = toy.simulate_quarter(spec, 2.0, np.random.default_rng(0))
    ell = toy.joint_log_likelihood(spec, 2.0, y_star)
    psis = importance.psis_weights(ell)
    assert np.isfinite(psis.khat) and psis.khat <= K_HAT_THRESHOLD, psis.khat
    assert not importance.apply_khat_policy(psis).skipped


def test_g23_degenerate_uniform_weights_not_skipped():
    """G2.3 — degenerate (exactly-flat ell) is intercepted: khat 0.0, accepted."""
    psis = importance.psis_weights(np.full(500, 1.0))
    assert psis.degenerate and psis.khat == 0.0
    assert not importance.apply_khat_policy(psis).skipped


def test_g23c_skipped_quarter_bookkeeping():
    """G2.3c — real-pipeline evaluate_allocation skip bookkeeping (gated)."""
    model_file = getattr(config, "MODEL_FILE", None)
    idata_file = getattr(config, "IDATA_FILE", None)
    if not (
        model_file and model_file.is_dir() and idata_file and idata_file.is_dir()
    ):
        pytest.skip(SKIP_MSG)

    from mmm_evsi.baseline import solve_baseline
    from mmm_evsi.load_mmm import load_budgets, load_case_study_data, load_mmm

    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2
    allocation = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg).budgets

    ev = importance.evaluate_allocation(
        mmm, idata, df, allocation, q1_cfg, q2_cfg,
        n_outcomes=3, seed=0, n_processes=1,
    )
    assert ev.n_outcomes == 3
    assert ev.n_skipped == len(ev.skipped_indices)
    assert ev.n_skipped >= 0

    # khats[skipped_indices] are exactly the nan entries of khats.
    nan_idx = list(np.where(np.isnan(ev.khats))[0])
    assert sorted(nan_idx) == sorted(int(i) for i in ev.skipped_indices)

    # Accepted quarters produce exactly one solved utility each; skipped
    # quarters are never solved (utilities holds only accepted outcomes).
    assert len(ev.utilities) == ev.n_outcomes - ev.n_skipped
    assert np.all(np.isfinite(ev.utilities))
    accepted = np.ones(ev.n_outcomes, dtype=bool)
    accepted[list(ev.skipped_indices)] = False
    assert np.all(np.isfinite(ev.khats[accepted]))
    assert ev.weight_ess.shape == (ev.n_outcomes,)  # recorded even when skipped