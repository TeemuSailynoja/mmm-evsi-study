"""Phase-by-phase profiling of BO script (serial vs parallel)."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmm_evsi import config
from mmm_evsi.baseline import solve_baseline
from mmm_evsi.bo_design import (
    bayesian_optimization,
    re_evaluate_winner_paired,
    decompose_utility,
    generate_feasible_proposal,
    lhs_feasible_design,
    compute_v_q1,
)
from mmm_evsi.importance import evaluate_allocation, pool_posterior
from mmm_evsi.load_mmm import load_budgets, load_case_study_data, load_mmm
from mmm_evsi.optimize_slsqp import get_baseline_allocation, q2_expected_response


def run_phase_bench(n_processes: int, tag: str) -> dict:
    """Run BO with phase timing. Returns dict of phase durations."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {tag} (n_processes={n_processes})")
    print(f"{'='*70}", flush=True)

    phases = {}
    t0 = time.time()

    # Phase 1: Model loading
    t1 = time.time()
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2
    phases["model_load"] = time.time() - t1

    # Phase 2: Baseline allocation (Q1 + Q2)
    t2 = time.time()
    a0 = get_baseline_allocation(mmm, q1_cfg, "Q1")
    q2_alloc = get_baseline_allocation(mmm, q2_cfg, "Q2")
    phases["baseline_alloc"] = time.time() - t2

    # Phase 3: V_Q1_baseline
    t3 = time.time()
    pooled = pool_posterior(idata["posterior"].to_dataset())
    V_Q1_baseline = compute_v_q1(mmm, pooled, df, a0, q1_cfg, n_draws=100)
    phases["v_q1_baseline"] = time.time() - t3

    # Phase 4: BO loop
    t4 = time.time()
    result = bayesian_optimization(
        mmm=mmm,
        idata=idata,
        df=df,
        q1_cfg=q1_cfg,
        q2_cfg=q2_cfg,
        baseline_allocation=a0,
        V_Q1_baseline=V_Q1_baseline,
        n_evaluations=5,
        n_initial=3,
        n_outcomes=3,
        E_max_fraction=E_MAX_FRACTION,
        seed=0,
        tol=1e-4,
        patience=10,
        n_processes=n_processes,
    )
    phases["bo_loop"] = time.time() - t4

    # Phase 5: Winner re-evaluation
    t5 = time.time()
    winner_result = re_evaluate_winner_paired(
        winner_allocation=result.best_allocation,
        baseline_allocation=a0,
        mmm=mmm,
        idata=idata,
        df=df,
        q1_cfg=q1_cfg,
        q2_cfg=q2_cfg,
        V_Q1_baseline=V_Q1_baseline,
        n_outcomes=3,
        seed=0,
        n_processes=n_processes,
    )
    phases["winner_eval"] = time.time() - t5

    total = time.time() - t0
    phases["total"] = total
    return phases


if __name__ == "__main__":
    # Need to define E_MAX_FRACTION
    E_MAX_FRACTION = 0.10

    results = {}
    for tag, n_procs in [("serial", 1), ("parallel", 8)]:
        results[tag] = run_phase_bench(n_procs, tag)

    # Print summary
    print(f"\n{'='*70}")
    print("PHASE-BY-PHASE TIMING BREAKDOWN")
    print(f"{'='*70}")
    print(f"{'Phase':<25} {'Serial (s)':>12} {'Parallel (s)':>14} {'Speedup':>10}")
    print(f"{'-'*65}")
    for phase in ["model_load", "baseline_alloc", "v_q1_baseline", "bo_loop", "winner_eval", "total"]:
        s = results["serial"][phase]
        p = results["parallel"][phase]
        speedup = s / p if p > 0 else 0
        print(f"{phase:<25} {s:>12.1f} {p:>14.1f} {speedup:>10.2f}x")

    # Resource contention analysis
    print(f"\n{'='*70}")
    print("RESOURCE CONTENTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Serial bo_loop:  {results['serial']['bo_loop']:.1f}s")
    print(f"Parallel bo_loop: {results['parallel']['bo_loop']:.1f}s")
    print(f"Parallel overhead: {results['parallel']['bo_loop'] - results['serial']['bo_loop']:.1f}s")
    
    if results["parallel"]["bo_loop"] > results["serial"]["bo_loop"] * 1.1:
        print("⚠️  PARALLELISM CAUSES CONTENTION - parallel is SLOWER")
        print("   Likely cause: zarr I/O contention among 8 processes")
    elif results["parallel"]["bo_loop"] < results["serial"]["bo_loop"] * 0.7:
        print("✅ PARALLELISM HELPFUL - parallel is faster")
    else:
        print("⚠️  NEUTRAL - parallel neither helps nor hurts much")

    # Write report
    report_path = "toy/profile_bo_results.txt"
    with open(report_path, "w") as f:
        f.write("BO PROFILING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("PHASE-BY-PHASE TIMING\n")
        f.write("=" * 70 + "\n\n")
        for tag, data in results.items():
            f.write(f"\n{tag.upper()} (n_processes={1 if tag=='serial' else 8})\n")
            for phase, dur in data.items():
                f.write(f"  {phase}: {dur:.1f}s\n")

    print(f"\nReport saved to {report_path}")
