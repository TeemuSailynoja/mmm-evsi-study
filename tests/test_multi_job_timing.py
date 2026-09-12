"""Timing test: 5 proposals × 10 outcomes with the optimizer cache."""
import time
import numpy as np
import pytest

def test_multi_job_timing_5x10(mmm, idata, df, q1_cfg, q2_cfg, baseline_allocation, V_Q1_baseline):
    """Evaluate 5 proposals × 10 outcomes and measure total time.
    
    With the caches: ~14s per proposal × 5 = ~70s total.
    Without the caches: ~34s per proposal × 5 = ~170s total.
    
    This demonstrates the cache benefit for the BO loop scenario.
    """
    from mmm_evsi.bo_design import generate_feasible_proposal, pool_posterior
    from mmm_evsi.experiments import allocation_to_weekly_spend, simulate_quarter
    from mmm_evsi.importance import quarter_log_likelihood, psis_weights, apply_khat_policy, resample_posterior
    from mmm_evsi.optimize_slsqp import WeightedSolveJob, run_weighted_solves
    from mmm_evsi import config

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    n_proposals = 5
    n_outcomes = 10  # 10 outcomes per proposal (like BO_N_OUTCOMES)
    
    all_jobs = []
    t_start = time.time()
    
    for p in range(n_proposals):
        alloc = generate_feasible_proposal(
            baseline_allocation, q1_cfg, rng,
            E_max, V_Q1_baseline, n_iter=10,
        )
        q1_weekly = allocation_to_weekly_spend(alloc, 13)
        
        for o in range(n_outcomes):
            y_star = simulate_quarter(
                mmm, idata, df, q1_cfg.window, alloc, seed=42 + p * 10 + o
            )
            ell = quarter_log_likelihood(
                mmm, idata, df, q1_cfg.window, alloc, y_star
            )
            psis = psis_weights(ell)
            verdict = apply_khat_policy(psis)
            
            if verdict.skipped:
                continue
                
            pooled = pool_posterior(idata["posterior"].to_dataset())
            posterior_r = resample_posterior(
                pooled, np.exp(psis.smoothed_log_weights),
                n=config.RESAMPLE_DRAWS, seed=42 + p * 10 + o,
            )
            
            all_jobs.append(WeightedSolveJob(
                allocation=alloc, outcome_index=o, posterior=posterior_r,
                q1_weekly_spend=q1_weekly, y_star=y_star,
                khat=psis.khat, weight_ess=psis.weight_ess, x0=None,
            ))
    
    t_build = time.time() - t_start
    print(f"Built {len(all_jobs)} jobs in {t_build:.1f}s")
    
    # Now run all jobs at once (uses cache)
    t0 = time.time()
    results = run_weighted_solves(mmm, df, all_jobs, q2_cfg, n_processes=1)
    t_solve = time.time() - t0
    
    print(f"Solved {len(results)} jobs in {t_solve:.1f}s")
    print(f"Total time: {t_solve + t_build:.1f}s")
    
    # With caches: should be < 200s
    assert t_solve < 200, f"Solve took {t_solve:.1f}s — caches may not be working"
    assert len(results) == len(all_jobs)
    assert not any(r.skipped for r in results)
