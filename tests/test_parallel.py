"""G2.7 — deterministic parallel weighted solves (artifact-gated).

Identical ``WeightedSolveJob`` lists solved serially (``n_processes=1``) and
through the process pool (``n_processes=2``) must agree bit-for-bit: solves
contain no RNG (resampling happens before the jobs are built, seeded), so
allocations are allclose (rtol 1e-8, atol 1e-6) and objectives allclose
(rtol 1e-12) regardless of pool size/process layout. Jobs are built from
evaluate_allocation's own steps 1–6; quarters skipped by the k-hat policy
never enter the pool ("already filtered by the caller").

Artifact-gated (Stage-1 artifacts + Stage-2 modules).
"""
from __future__ import annotations

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

importance = pytest.importorskip("mmm_evsi.importance")
experiments = pytest.importorskip("mmm_evsi.experiments")
slsqp = pytest.importorskip("mmm_evsi.optimize_slsqp")

N_JOBS = 2
SEED = 0


def _build_jobs(mmm, idata, df, allocation, q1_cfg, n_jobs, seed):
    """Steps 1–6 of evaluate_allocation: simulate, weight, resample, job."""
    pooled = importance.pool_posterior(idata["posterior"])
    jobs = []
    for i in range(n_jobs):
        y_star = experiments.simulate_quarter(
            mmm, idata, df, q1_cfg.window, allocation, q1_cfg, seed=seed + i
        )
        ell = importance.quarter_log_likelihood(
            mmm, idata, df, q1_cfg.window, allocation, y_star,
            baseline_weekly_spend=q1_cfg.weekly_spend,
            baseline_quarterly=q1_cfg.planned,
        )
        psis = importance.psis_weights(ell)
        if importance.apply_khat_policy(psis).skipped:
            continue  # skipped quarters never enter the pool
        posterior_r = importance.resample_posterior(
            pooled, np.exp(psis.smoothed_log_weights), seed=10_000 + i
        )
        q1_weekly = experiments.allocation_to_weekly_spend(allocation, q1_cfg.weekly_spend, q1_cfg.planned, 13)
        jobs.append(
            slsqp.WeightedSolveJob(
                allocation=allocation,
                outcome_index=i,
                posterior=posterior_r,
                q1_weekly_spend=q1_weekly,
                y_star=y_star,
                khat=psis.khat,
                weight_ess=psis.weight_ess,
                x0=None,
            )
        )
    return jobs


def test_g27_serial_and_pool_weighted_solves_agree():
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2
    allocation = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg).budgets

    jobs = _build_jobs(mmm, idata, df, allocation, q1_cfg, N_JOBS, SEED)
    assert len(jobs) >= 2, "expected at least two policy-accepted quarters"

    serial = slsqp.run_weighted_solves(mmm, df, jobs, q2_cfg, n_processes=1)
    pooled = slsqp.run_weighted_solves(mmm, df, jobs, q2_cfg, n_processes=2)
    assert len(serial) == len(pooled) == len(jobs)

    for s_res, p_res in zip(serial, pooled):
        assert s_res.skipped == p_res.skipped
        # allocations: allclose rtol 1e-8, atol 1e-6
        assert np.allclose(
            s_res.budgets.values, p_res.budgets.values, rtol=1e-8, atol=1e-6
        ), (
            f"job outcome {s_res.outcome_index}: serial budgets "
            f"{s_res.budgets.values} vs pool {p_res.budgets.values}"
        )
        # objectives: allclose rtol 1e-12
        assert np.allclose(
            s_res.objective_value, p_res.objective_value, rtol=1e-12, atol=0.0
        ), (
            f"job outcome {s_res.outcome_index}: serial objective "
            f"{s_res.objective_value:.15g} vs pool {p_res.objective_value:.15g}"
        )