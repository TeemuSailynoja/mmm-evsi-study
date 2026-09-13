"""Stage 3: Bayesian optimization over feasible Q1 budget perturbations.

Runs the full BO loop:
1. Load model, posterior, data, budgets
2. Solve baseline allocation
3. Compute V_Q1_baseline
4. Run BO loop (LHS initial design + BO iterations)
5. Re-evaluate winner with CRN
6. Save results

Usage:
    python scripts/run_bo.py --n-evaluations 100 --n-outcomes 10 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
import logging

logging.getLogger("sklearn").setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from mmm_evsi import config
from mmm_evsi.bo_design import (
    bayesian_optimization,
    re_evaluate_winner_paired,
    decompose_utility,
)
from mmm_evsi.importance import evaluate_allocation, pool_posterior
from mmm_evsi.load_mmm import load_budgets, load_case_study_data, load_mmm
from mmm_evsi.optimize_slsqp import get_baseline_allocation, q2_expected_response

RESULTS_PATH = "data/fit/bo_result.json"
TRACE_PATH = "data/fit/bo_trace.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-evaluations", type=int, default=config.BO_N_EVALUATIONS)
    ap.add_argument("--n-initial", type=int, default=config.BO_N_INITIAL)
    ap.add_argument("--n-outcomes", type=int, default=config.BO_N_OUTCOMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--tol", type=float, default=config.BO_TOL)
    ap.add_argument("--patience", type=int, default=config.BO_PATIENCE)
    ap.add_argument("--e-max-fraction", type=float, default=config.E_MAX_FRACTION)
    ap.add_argument("--winner-outcomes", type=int, default=100,
                    help="Number of CRN outcomes for winner re-evaluation")
    ap.add_argument("--n-processes", type=int, default=None,
                    help="Number of parallel processes (default: half of CPU cores)")
    args = ap.parse_args()

    print("=" * 70, flush=True)
    print("Stage 3: Bayesian Optimization", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()

    # --- Load model, posterior, data ---
    print(f"[{time.time() - t0:.0f}s] Loading model and data...", flush=True)
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2

    # --- Solve baseline (load from cache if available) ---
    print(f"[{time.time() - t0:.0f}s] Loading/optimizing baseline allocation...", flush=True)
    a0 = get_baseline_allocation(mmm, q1_cfg, "Q1")
    print(f"[{time.time() - t0:.0f}s] Baseline: {dict(a0.to_dict())}", flush=True)
    
    # Save Q2 baseline allocation too (used for hot starts)
    q2_alloc = get_baseline_allocation(mmm, q2_cfg, "Q2")
    print(f"[{time.time() - t0:.0f}s] Q2 Baseline: {dict(q2_alloc.to_dict())}", flush=True)

    # --- Compute V_Q1_baseline ---
    print(f"[{time.time() - t0:.0f}s] Computing V_Q1_baseline...", flush=True)
    pooled = pool_posterior(idata["posterior"].to_dataset())
    a0_weekly = np.asarray(a0.values, dtype=float)  # (7,)
    V_Q1_baseline = evaluate_allocation(
        mmm=mmm, idata=idata, df=df,
        allocation=a0, q1_cfg=q1_cfg, q2_cfg=q2_cfg,
        n_outcomes=1, seed=0, n_processes=None,
    ).utility  # Quick evaluation for baseline

    # Actually, V_Q1_baseline is just expected Q1 sales under baseline
    # Use compute_v_q1 directly
    from mmm_evsi.bo_design import compute_v_q1
    V_Q1_baseline = compute_v_q1(mmm, pooled, df, a0, q1_cfg, n_draws=config.BO_N_DRAW_VQ1)
    print(f"[{time.time() - t0:.0f}s] V_Q1_baseline = {V_Q1_baseline:.6g}", flush=True)

    E_max = args.e_max_fraction * q1_cfg.total
    print(f"[{time.time() - t0:.0f}s] E_max = {E_max:.6g} ({args.e_max_fraction*100:.0f}% of Q1 total)", flush=True)
    
    # Determine number of parallel processes
    if args.n_processes is None:
        n_proc = max(1, (os.cpu_count() or 2) // 2)
    else:
        n_proc = args.n_processes
    print(f"[{time.time() - t0:.0f}s] BO config: n_eval={args.n_evaluations}, n_initial={args.n_initial}, "
          f"n_outcomes={args.n_outcomes}, seed={args.seed}, n_processes={n_proc}", flush=True)

    # --- Run BO loop ---
    print(f"\n{'='*70}", flush=True)
    print(f"Starting BO loop...", flush=True)
    print(f"{'='*70}", flush=True)

    result = bayesian_optimization(
        mmm=mmm,
        idata=idata,
        df=df,
        q1_cfg=q1_cfg,
        q2_cfg=q2_cfg,
        baseline_allocation=a0,
        V_Q1_baseline=V_Q1_baseline,
        n_evaluations=args.n_evaluations,
        n_initial=args.n_initial,
        n_outcomes=args.n_outcomes,
        E_max_fraction=args.e_max_fraction,
        seed=args.seed,
        tol=args.tol,
        patience=args.patience,
        x0_warm=None,
        n_processes=n_proc,
    )

    print(f"\n{'='*70}", flush=True)
    print(f"BO loop complete!", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Best utility: {result.best_utility:.6g}", flush=True)
    print(f"Best V_Q1:    {result.best_v_q1:.6g}", flush=True)
    print(f"Best Q1 loss: {result.best_q1_loss:.6g}", flush=True)
    print(f"Best Q2 gain: {result.best_q2_gain:.6g}", flush=True)
    print(f"N evaluations: {result.n_evaluations}", flush=True)
    print(f"Early stop:    {result.n_stopped_early}", flush=True)
    print(f"Reason:        {result.reason}", flush=True)

    # --- Winner re-evaluation with CRN ---
    print(f"\n{'='*70}", flush=True)
    print(f"Re-evaluating winner with CRN (n={args.winner_outcomes})...", flush=True)
    print(f"{'='*70}", flush=True)

    winner_re_eval = re_evaluate_winner_paired(
        winner_allocation=result.best_allocation,
        baseline_allocation=a0,
        mmm=mmm,
        idata=idata,
        df=df,
        q1_cfg=q1_cfg,
        q2_cfg=q2_cfg,
        V_Q1_baseline=V_Q1_baseline,
        n_outcomes=args.winner_outcomes,
        seed=args.seed,
        n_processes=n_proc,
    )

    print(f"Winner utility: {winner_re_eval['winner_utility']:.6g}", flush=True)
    print(f"Baseline utility: {winner_re_eval['baseline_utility']:.6g}", flush=True)
    print(f"Delta (winner - baseline): {winner_re_eval['delta']:.6g} +/- {winner_re_eval['delta_se']:.3g}", flush=True)
    print(f"95% CI: [{winner_re_eval['delta_ci_lower']:.6g}, {winner_re_eval['delta_ci_upper']:.6g}]", flush=True)

    # --- Utility decomposition ---
    # Use prior utility from baseline-arm EVSI if available
    baseline_utility = 0.0  # Placeholder; replace with actual baseline-arm EVSI if needed
    decomposition = decompose_utility(result, baseline_utility, winner_re_eval)

    # --- Save results ---
    print(f"\nSaving results...", flush=True)

    # Save trace
    if result.trace:
        trace_df = pd.DataFrame([
            {
                "iteration": t.iteration,
                "allocation": str(t.allocation.tolist()),
                "v_q1": t.v_q1,
                "q1_loss": t.q1_loss,
                "q2_gain": t.q2_gain,
                "q2_gains": str(t.q2_gains.tolist()),
                "utility": t.utility,
                "gp_mean": t.gp_mean,
                "gp_std": t.gp_std,
                "acquisition_value": t.acquisition_value,
            }
            for t in result.trace
        ])
        trace_df.to_csv(TRACE_PATH, index=False)
        print(f"Saved trace to {TRACE_PATH}", flush=True)

    # Save result JSON
    result_dict = {
        "best_allocation": result.best_allocation.to_dict(),
        "best_utility": result.best_utility,
        "best_v_q1": result.best_v_q1,
        "best_q1_loss": result.best_q1_loss,
        "best_q2_gain": result.best_q2_gain,
        "n_evaluations": result.n_evaluations,
        "n_stopped_early": result.n_stopped_early,
        "reason": result.reason,
        "winner_re_eval": winner_re_eval,
        "decomposition": decomposition,
        "config": {
            "n_evaluations": args.n_evaluations,
            "n_initial": args.n_initial,
            "n_outcomes": args.n_outcomes,
            "seed": args.seed,
            "e_max_fraction": args.e_max_fraction,
            "tol": args.tol,
            "patience": args.patience,
        },
        "wall_time": time.time() - t0,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"Saved result to {RESULTS_PATH}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Total wall time: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
