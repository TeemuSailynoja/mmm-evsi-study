"""Profile the multi-job pipeline: 5 proposals x 2 outcomes."""
import cProfile
import pstats
import io
import numpy as np
import sys

sys.path.insert(0, "/home/teemu/repos/mmm-evsi-study/src")

from mmm_evsi.load_mmm import load_mmm, load_case_study_data, load_budgets
from mmm_evsi.bo_design import generate_feasible_proposal, pool_posterior, compute_v_q1
from mmm_evsi.experiments import allocation_to_weekly_spend, simulate_quarter
from mmm_evsi.importance import (
    quarter_log_likelihood,
    psis_weights,
    apply_khat_policy,
    resample_posterior,
)
from mmm_evsi.optimize_slsqp import WeightedSolveJob, run_weighted_solves
from mmm_evsi.baseline import solve_baseline
from mmm_evsi import config


def main():
    print("Loading model...")
    mmm, _ = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets(df)
    q1_cfg = budgets.q1
    q2_cfg = budgets.q2

    print("Solving baseline...")
    result = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg)
    baseline_allocation = result.budgets

    print("Computing V_Q1_baseline...")
    V_Q1_baseline = compute_v_q1(
        mmm, mmm.idata["posterior"].to_dataset(), df,
        baseline_allocation, q1_cfg, n_draws=100,
    )

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    n_proposals = 5
    n_outcomes = 2

    all_jobs = []

    # Phase 1: Build jobs
    for p in range(n_proposals):
        alloc = generate_feasible_proposal(
            baseline_allocation, q1_cfg, rng,
            E_max, V_Q1_baseline, n_iter=10,
        )
        q1_weekly = allocation_to_weekly_spend(alloc, 13)

        for o in range(n_outcomes):
            y_star = simulate_quarter(
                mmm, mmm.idata, df, q1_cfg.window, alloc, seed=42 + p * 10 + o
            )
            ell = quarter_log_likelihood(
                mmm, mmm.idata, df, q1_cfg.window, alloc, y_star
            )
            psis = psis_weights(ell)
            verdict = apply_khat_policy(psis)

            if verdict.skipped:
                continue

            pooled = pool_posterior(mmm.idata["posterior"].to_dataset())
            posterior_r = resample_posterior(
                pooled, np.exp(psis.smoothed_log_weights),
                n=config.RESAMPLE_DRAWS, seed=42 + p * 10 + o,
            )

            all_jobs.append(WeightedSolveJob(
                allocation=alloc, outcome_index=o, posterior=posterior_r,
                q1_weekly_spend=q1_weekly, y_star=y_star,
                khat=psis.khat, weight_ess=psis.weight_ess, x0=None,
            ))

    print(f"Built {len(all_jobs)} jobs")

    # Phase 2: Solve
    results = run_weighted_solves(mmm, df, all_jobs, q2_cfg, n_processes=1)
    print(f"Solved {len(results)} jobs")


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print(s.getvalue())
