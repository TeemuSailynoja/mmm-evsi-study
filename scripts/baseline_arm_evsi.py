"""G2.9 STOP checkpoint at real scale: baseline-arm information value, with a
running (incremental) precision trace.

EVSI(a0) = E_{y*|a0}[OptQ2(y*, a0)] - OptQ2(prior; a0 carry-in)

Both arms share the same Q1 carry-in (a0's weekly spend), so the difference
isolates the INFORMATION value of observing Q1 under the baseline allocation.
Utilities are in correct sales units via ``q2_expected_response``.

The outcomes are processed sequentially (deterministic seeds; one wrapper
compile serves all solves). After each outcome the cumulative estimate is
appended to ``data/fit/evsi_trace.csv``:

    n_accepted, evsi, mc_se, mean_util, std_util, khat, skipped

so the sample-size/precision trade-off can be read off the trace. Final
summary -> ``data/fit/baseline_arm_evsi.npz`` + ``data/fit/evsi_trace.png``.

    python scripts/baseline_arm_evsi.py --n-outcomes 200 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from mmm_evsi import config
from mmm_evsi.carry_in_optimizer import CarryInBudgetOptimizer
from mmm_evsi.experiments import allocation_to_weekly_spend, simulate_quarter
from mmm_evsi.importance import (
    apply_khat_policy,
    evaluate_allocation,
    pool_posterior,
    psis_weights,
    quarter_log_likelihood,
    resample_posterior,
)
from mmm_evsi.load_mmm import load_budgets, load_case_study_data, load_mmm
from mmm_evsi.optimize_slsqp import (
    WeightedSolveJob,
    _solve_on_wrapper,
    get_baseline_allocation,
    q2_expected_response,
)

TRACE_PATH = "data/fit/evsi_trace.csv"
FINAL_PATH = "data/fit/baseline_arm_evsi.npz"
PLOT_PATH = "data/fit/evsi_trace.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-outcomes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true",
                    help="continue appending to an existing trace (skips prior arm)")
    args = ap.parse_args()

    t0 = time.time()
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2
    a0 = get_baseline_allocation(mmm, q1_cfg, "Q1")
    a0_weekly = allocation_to_weekly_spend(a0, 13)
    pooled = pool_posterior(idata["posterior"].to_dataset())
    print(f"[{time.time() - t0:.0f}s] load + baseline Q1", flush=True)

    # One compiled wrapper serves the prior arm AND every weighted solve.
    t1 = time.time()
    w = CarryInBudgetOptimizer(mmm, config.Q2_WINDOW[0], config.Q2_WINDOW[1])
    print(f"[{time.time() - t1:.0f}s] wrapper compile", flush=True)

    # OptQ2(prior; a0 carry-in): full 32,000-draw posterior, same Q1 carry-in.
    if args.resume and os.path.exists(TRACE_PATH):
        with open(TRACE_PATH) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit("empty trace; cannot resume")
        prior_util = float(rows[0]["prior_utility"])
        print(f"resuming: prior utility {prior_util:.6g} (from trace)", flush=True)
    else:
        t1 = time.time()
        w.set_q1_carry_in(np.asarray(a0_weekly, dtype=float))
        prior_res = w.allocate_budget(
            total_budget=q2_cfg.total,
            budget_bounds=q2_cfg.boxes,
            minimize_kwargs={"options": {"ftol": 1e-6}},
        )
        prior_util = q2_expected_response(
            mmm, pooled, df, a0_weekly, prior_res.budgets
        )
        print(
            f"[{time.time() - t1:.0f}s] OptQ2(prior) util={prior_util:.6g}",
            flush=True,
        )

    print(f"[{time.time()-t0:.0f}s] Starting outcome loop (n={args.n_outcomes}, seed={args.seed})", flush=True)
    # Incremental trace: one row per outcome (cumulative stats over accepted).
    header = ["n_accepted", "n_processed", "evsi", "mc_se", "mean_util",
              "std_util", "khat", "skipped", "prior_utility"]
    write_header = not (args.resume and os.path.exists(TRACE_PATH))
    utils: list[float] = []
    n_processed = 0
    with open(TRACE_PATH, "a", newline="") as fh:
        out = csv.writer(fh)
        if write_header:
            out.writerow(header)
        for i in range(args.n_outcomes):
            t_i = time.time()
            y_star = simulate_quarter(
                mmm, idata, df, q1_cfg.window, a0, seed=args.seed + i
            )
            print(f"[{time.time()-t0:.0f}s] outcome {i}: simulate={time.time()-t_i:.1f}s", flush=True)
            t_i = time.time()
            ell = quarter_log_likelihood(mmm, idata, df, q1_cfg.window, a0, y_star)
            print(f"[{time.time()-t0:.0f}s] outcome {i}: loglik={time.time()-t_i:.1f}s", flush=True)
            t_i = time.time()
            psis = psis_weights(ell)
            print(f"[{time.time()-t0:.0f}s] outcome {i}: psis={time.time()-t_i:.1f}s", flush=True)
            verdict = apply_khat_policy(psis)
            n_processed += 1
            skipped = 1 if verdict.skipped else 0
            khat = psis.khat
            if not verdict.skipped:
                t_i = time.time()
                posterior_r = resample_posterior(
                    pooled, np.exp(psis.smoothed_log_weights),
                    n=config.RESAMPLE_DRAWS, seed=args.seed + i,
                )
                print(f"[{time.time()-t0:.0f}s] outcome {i}: resample={time.time()-t_i:.1f}s", flush=True)
                t_i = time.time()
                job = WeightedSolveJob(
                    allocation=a0, outcome_index=i, posterior=posterior_r,
                    q1_weekly_spend=a0_weekly, y_star=y_star, khat=psis.khat,
                    weight_ess=psis.weight_ess, x0=None,
                )
                try:
                    r = _solve_on_wrapper(w, q2_cfg, job)
                    print(f"[{time.time()-t0:.0f}s] outcome {i}: solve={time.time()-t_i:.1f}s", flush=True)
                    t_i = time.time()
                    u = q2_expected_response(mmm, posterior_r, df, a0_weekly, r.budgets)
                    print(f"[{time.time()-t0:.0f}s] outcome {i}: utility={time.time()-t_i:.1f}s", flush=True)
                    utils.append(float(u))
                except Exception as e:
                    print(f"[{time.time()-t0:.0f}s] outcome {i}: SOLVE FAILED ({e}), skipping", flush=True)
            n = len(utils)
            mean_u = float(np.mean(utils)) if n else float("nan")
            std_u = float(np.std(utils, ddof=1)) if n > 1 else float("nan")
            evsi = mean_u - prior_util
            se = std_u / np.sqrt(n) if n > 1 else float("nan")
            out.writerow([n, n_processed, evsi, se, mean_u, std_u, khat,
                          skipped, prior_util])
            fh.flush()
            if (i + 1) % args.log_every == 0 or i + 1 == args.n_outcomes:
                print(
                    f"[{time.time() - t0:5.0f}s] n={n:4d} (of {n_processed} processed) "
                    f"EVSI={evsi:.6g} +/- {se:.3g} khat={khat:.3f}",
                    flush=True,
                )

    utils_arr = np.asarray(utils)
    evsi = float(utils_arr.mean() - prior_util)
    se = float(utils_arr.std(ddof=1) / np.sqrt(len(utils_arr)))
    print(f"FINAL EVSI(a0) = {evsi:.6g} +/- {se:.3g}  (n={len(utils_arr)}, "
          f"skipped={n_processed - len(utils_arr)})")
    np.savez(FINAL_PATH, n_outcomes=args.n_outcomes, seed=args.seed,
             utilities=utils_arr, prior_utility=prior_util, evsi=evsi, mc_se=se,
             allocation=a0.values)
    print(f"saved {FINAL_PATH} and {TRACE_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = np.genfromtxt(TRACE_PATH, delimiter=",", names=True)
        n = rows["n_accepted"]; ev = rows["evsi"]; se = rows["mc_se"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(n, ev, ".-"); axes[0].set_xlabel("accepted outcomes")
        axes[0].set_ylabel("EVSI(a0)"); axes[0].set_title("running estimate")
        axes[1].plot(n, se, ".-"); axes[1].set_xlabel("accepted outcomes")
        axes[1].set_ylabel("MC SE"); axes[1].set_title("running precision")
        fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=110)
        print(f"saved {PLOT_PATH}")
    except Exception as e:  # plotting is best-effort
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
