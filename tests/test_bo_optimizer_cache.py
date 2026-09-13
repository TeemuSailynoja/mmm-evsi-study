"""G3.6 — CarryInBudgetOptimizer cache gate.

Verifies that the optimizer cache in `optimize_slsqp.py` works correctly:
  1. First call compiles the optimizer (slow)
  2. Subsequent calls reuse the cached optimizer (fast)
  3. The cached optimizer produces identical results
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from mmm_evsi import config


@pytest.fixture(autouse=True)
def reset_optimizer_cache():
    """Clear the optimizer cache before each test to ensure isolation."""
    from mmm_evsi import optimize_slsqp
    optimize_slsqp._q2_optimizer_cache.clear()
    yield


def test_g36_first_call_compiles(
    mmm, idata, df, q1_cfg, q2_cfg, baseline_allocation
):
    """G3.6 (real): First call to run_weighted_solves compiles the optimizer.

    Verifies that the first call to `run_weighted_solves()` takes > 5s
    (the compilation time for CarryInBudgetOptimizer).
    """
    from mmm_evsi.bo_design import generate_feasible_proposal, pool_posterior
    from mmm_evsi.experiments import allocation_to_weekly_spend, simulate_quarter
    from mmm_evsi.importance import quarter_log_likelihood, psis_weights, apply_khat_policy, resample_posterior
    from mmm_evsi.optimize_slsqp import WeightedSolveJob, run_weighted_solves
    from mmm_evsi import config

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    V_Q1_baseline = 16_000_000

    # Generate one proposal
    alloc = generate_feasible_proposal(
        baseline_allocation, q1_cfg, rng,
        E_max, V_Q1_baseline, n_iter=10,
    )
    q1_weekly = allocation_to_weekly_spend(alloc, q1_cfg.weekly_spend, q1_cfg.planned, 13)

    # Simulate one outcome
    y_star = simulate_quarter(
        mmm, idata, df, q1_cfg.window, alloc, q1_cfg, seed=42
    )

    # Compute log-likelihood and weights
    ell = quarter_log_likelihood(
        mmm, idata, df, q1_cfg.window, alloc, y_star,
        baseline_weekly_spend=q1_cfg.weekly_spend,
        baseline_quarterly=q1_cfg.planned,
    )
    psis = psis_weights(ell)
    verdict = apply_khat_policy(psis)

    if verdict.skipped:
        pytest.skip("k-hat policy skipped this outcome")

    # Resample posterior
    pooled = pool_posterior(idata["posterior"].to_dataset())
    posterior_r = resample_posterior(
        pooled,
        np.exp(psis.smoothed_log_weights),
        n=config.RESAMPLE_DRAWS,
        seed=42,
    )

    # Create one job
    job = WeightedSolveJob(
        allocation=alloc,
        outcome_index=0,
        posterior=posterior_r,
        q1_weekly_spend=q1_weekly,
        y_star=y_star,
        khat=psis.khat,
        weight_ess=psis.weight_ess,
        x0=None,
    )

    # Time the first call (should include compilation)
    t0 = time.time()
    results = run_weighted_solves(
        mmm, df, [job], q2_cfg, n_processes=1
    )
    t1 = time.time()
    elapsed = t1 - t0

    print(f"G3.6 first call: {elapsed:.1f}s")
    assert elapsed > 5.0, (
        f"First call took only {elapsed:.1f}s — optimizer may not have compiled"
    )
    assert len(results) == 1
    assert not results[0].skipped


def test_g36_second_call_reuses(
    mmm, idata, df, q1_cfg, q2_cfg, baseline_allocation
):
    """G3.6 (real): Second call reuses the cached optimizer (fast).

    Verifies that the second call to `run_weighted_solves()` takes < 5s
    (no recompilation, just data swap + solve).
    """
    from mmm_evsi.bo_design import generate_feasible_proposal, pool_posterior
    from mmm_evsi.experiments import allocation_to_weekly_spend, simulate_quarter
    from mmm_evsi.importance import quarter_log_likelihood, psis_weights, apply_khat_policy, resample_posterior
    from mmm_evsi.optimize_slsqp import WeightedSolveJob, run_weighted_solves

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    V_Q1_baseline = 16_000_000

    # Generate one proposal
    alloc = generate_feasible_proposal(
        baseline_allocation, q1_cfg, rng,
        E_max, V_Q1_baseline, n_iter=10,
    )
    q1_weekly = allocation_to_weekly_spend(alloc, q1_cfg.weekly_spend, q1_cfg.planned, 13)

    # Simulate one outcome
    y_star = simulate_quarter(
        mmm, idata, df, q1_cfg.window, alloc, q1_cfg, seed=42
    )

    # Compute log-likelihood and weights
    ell = quarter_log_likelihood(
        mmm, idata, df, q1_cfg.window, alloc, y_star,
        baseline_weekly_spend=q1_cfg.weekly_spend,
        baseline_quarterly=q1_cfg.planned,
    )
    psis = psis_weights(ell)
    verdict = apply_khat_policy(psis)

    if verdict.skipped:
        pytest.skip("k-hat policy skipped this outcome")

    # Resample posterior
    pooled = pool_posterior(idata["posterior"].to_dataset())
    posterior_r = resample_posterior(
        pooled,
        np.exp(psis.smoothed_log_weights),
        n=config.RESAMPLE_DRAWS,
        seed=42,
    )

    # Create one job
    job = WeightedSolveJob(
        allocation=alloc,
        outcome_index=0,
        posterior=posterior_r,
        q1_weekly_spend=q1_weekly,
        y_star=y_star,
        khat=psis.khat,
        weight_ess=psis.weight_ess,
        x0=None,
    )

    # First call (compilation)
    results1 = run_weighted_solves(
        mmm, df, [job], q2_cfg, n_processes=1
    )

    # Second call (should reuse cache)
    t0 = time.time()
    results2 = run_weighted_solves(
        mmm, df, [job], q2_cfg, n_processes=1
    )
    t1 = time.time()
    elapsed = t1 - t0

    print(f"G3.6 second call: {elapsed:.1f}s")
    assert elapsed < 5.0, (
        f"Second call took {elapsed:.1f}s — optimizer may have recompiled"
    )

    # Results should be identical (same input, same cached optimizer)
    np.testing.assert_allclose(
        results1[0].budgets.values,
        results2[0].budgets.values,
        rtol=1e-6,
        err_msg="Cached optimizer produced different results"
    )
    print(f"G3.6 cache verified: results identical, second call {elapsed:.1f}s")


def test_g36_multiple_jobs(
    mmm, idata, df, q1_cfg, q2_cfg, baseline_allocation
):
    """G3.6 (real): Multiple jobs in one call all use the same cached optimizer.

    Verifies that when multiple jobs are passed to `run_weighted_solves()`,
    they all use the same compiled optimizer (only one compilation).
    """
    from mmm_evsi.bo_design import generate_feasible_proposal, pool_posterior
    from mmm_evsi.experiments import allocation_to_weekly_spend, simulate_quarter
    from mmm_evsi.importance import quarter_log_likelihood, psis_weights, apply_khat_policy, resample_posterior
    from mmm_evsi.optimize_slsqp import WeightedSolveJob, run_weighted_solves

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    V_Q1_baseline = 16_000_000

    # Generate one proposal
    alloc = generate_feasible_proposal(
        baseline_allocation, q1_cfg, rng,
        E_max, V_Q1_baseline, n_iter=10,
    )
    q1_weekly = allocation_to_weekly_spend(alloc, q1_cfg.weekly_spend, q1_cfg.planned, 13)

    # Simulate one outcome
    y_star = simulate_quarter(
        mmm, idata, df, q1_cfg.window, alloc, q1_cfg, seed=42
    )

    # Compute log-likelihood and weights
    ell = quarter_log_likelihood(
        mmm, idata, df, q1_cfg.window, alloc, y_star,
        baseline_weekly_spend=q1_cfg.weekly_spend,
        baseline_quarterly=q1_cfg.planned,
    )
    psis = psis_weights(ell)
    verdict = apply_khat_policy(psis)

    if verdict.skipped:
        pytest.skip("k-hat policy skipped this outcome")

    # Resample posterior
    pooled = pool_posterior(idata["posterior"].to_dataset())
    posterior_r = resample_posterior(
        pooled,
        np.exp(psis.smoothed_log_weights),
        n=config.RESAMPLE_DRAWS,
        seed=42,
    )

    # Create 3 jobs (same posterior, different outcome indices)
    jobs = [
        WeightedSolveJob(
            allocation=alloc,
            outcome_index=i,
            posterior=posterior_r,
            q1_weekly_spend=q1_weekly,
            y_star=y_star,
            khat=psis.khat,
            weight_ess=psis.weight_ess,
            x0=None,
        )
        for i in range(3)
    ]

    # Time the call with 3 jobs (should still be ~1 compilation + 3 solves)
    t0 = time.time()
    results = run_weighted_solves(
        mmm, df, jobs, q2_cfg, n_processes=1
    )
    t1 = time.time()
    elapsed = t1 - t0

    print(f"G3.6 3 jobs: {elapsed:.1f}s")
    assert len(results) == 3
    assert not any(r.skipped for r in results)
    # Should be faster than 3 separate calls (only 1 compilation)
    # Each solve takes ~1s, compilation ~10-20s, so 3 jobs should be ~15-25s
    # vs 30-60s for 3 separate compilations
    assert elapsed < 40.0, (
        f"3 jobs took {elapsed:.1f}s — may have compiled multiple times"
    )
    print(f"G3.6 3 jobs verified: {elapsed:.1f}s < 40s")
