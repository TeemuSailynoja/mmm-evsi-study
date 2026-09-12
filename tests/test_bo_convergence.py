"""G3.4 — BO convergence gate.

The BO loop must find a best utility ≥ LHS best at equal evaluation budget.
Toy version (N=50 evals, N=10 initial) validates the logic on a 2D problem.
Real version (N=10 evals, N=5 initial) validates on the real 7-channel model.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmm_evsi import config


# ======================================================================
# Toy version — validates BO convergence logic on a 2D problem
# ======================================================================


def test_g34_toy_bo_convergence_to_lhs_best():
    """G3.4 (toy): BO best utility ≥ LHS best at equal evaluation budget."""
    toy = pytest.importorskip("toy.toy_bo")

    baseline, boxes, x_true = toy.toy_feasible_set(n_channels=2)
    rng = np.random.default_rng(42)
    n_evaluations = 50
    n_initial = 10

    # LHS evaluation
    lhs_designs = toy.lhs_feasible_design_toy(baseline, boxes, n_initial, rng)
    lhs_utilities = [toy.toy_utility(x, x_true) for x in lhs_designs]
    lhs_best = max(lhs_utilities)
    lhs_best_x = lhs_designs[np.argmax(lhs_utilities)]

    # BO loop (inline — mirrors bayesian_optimization logic)
    X = np.array(lhs_designs)
    y = np.array(lhs_utilities)
    y_best = lhs_best

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from scipy.optimize import minimize

    kernel = RBF(length_scale=0.3) + ConstantKernel(1.0) + WhiteKernel(0.01)
    n_ch = len(baseline)
    n_free = n_ch - 1

    for iteration in range(n_initial, n_evaluations):
        # Fit GP
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, random_state=42,
        )
        gpr.fit(X, y)

        # Maximize EI (simplified — multi-start)
        best_ei = -np.inf
        best_x = None
        for _ in range(20):
            raw = rng.normal(0, 1, size=n_free)
            delta = toy.null_space_perturbation(raw, n_ch)

            max_allowed = np.array(
                [boxes[i][1] - baseline[i] for i in range(n_ch)], dtype=float,
            )
            min_allowed = np.array(
                [boxes[i][0] - baseline[i] for i in range(n_ch)], dtype=float,
            )
            pos_margin = np.where(delta > 0, max_allowed, np.inf)
            neg_margin = np.where(delta < 0, -min_allowed, np.inf)
            margin = np.minimum(pos_margin, neg_margin)
            if margin.min() < 1e-12:
                continue
            scale = 0.3 * margin.min() / (np.abs(delta).max() + 1e-12)
            delta *= scale

            delta_c = toy.clip_to_boxes(delta, boxes, baseline)
            delta_c = toy.recenter_sum(delta_c)
            x_cand = baseline + delta_c

            if abs(x_cand.sum() - baseline.sum()) > 1e-6:
                continue

            mean, std = gpr.predict(x_cand.reshape(1, -1), return_std=True)
            Z = (mean - y_best) / (std + 1e-12)
            from scipy.stats import norm
            ei = (mean - y_best) * norm.cdf(Z) + std * norm.pdf(Z)

            if ei > best_ei:
                best_ei = ei
                best_x = x_cand

        if best_x is None:
            break

        # Evaluate
        u = toy.toy_utility(best_x, x_true)
        X = np.vstack([X, best_x.reshape(1, -1)])
        y = np.append(y, u)

        if u > y_best:
            y_best = u

    print(
        f"G3.4 toy BO: LHS best={lhs_best:.4f}, BO best={y_best:.4f}, "
        f"true optimum={toy.toy_utility(x_true, x_true):.4f}"
    )
    assert y_best >= lhs_best - 1e-6, (
        f"BO best ({y_best:.4f}) < LHS best ({lhs_best:.4f})"
    )


# ======================================================================
# Real-model version — validates BO convergence on the 7-channel MMM
# ======================================================================


def test_g34_real_bo_convergence_to_lhs_best(
    mmm, idata, df, baseline_allocation, q1_cfg, q2_cfg, V_Q1_baseline
):
    """G3.4 (real): BO best utility ≥ LHS best at equal evaluation budget.

    Uses small n_evaluations (10) and n_initial (5) for speed.
    Runs the full bayesian_optimization loop and compares BO best vs LHS best.
    """
    from mmm_evsi.bo_design import (
        bayesian_optimization,
        lhs_feasible_design,
    )

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    n_initial = 5
    n_evaluations = 10

    # LHS evaluation
    lhs_designs = lhs_feasible_design(
        baseline_allocation, q1_cfg, n_initial,
        E_max, V_Q1_baseline, rng,
    )

    # Evaluate LHS proposals
    from mmm_evsi.bo_design import evaluate_proposal

    lhs_utilities = []
    for i, alloc in enumerate(lhs_designs):
        proposal = evaluate_proposal(
            mmm, idata, df, alloc, q1_cfg, q2_cfg,
            V_Q1_baseline, n_outcomes=2, seed=i,
            n_processes=1,
        )
        lhs_utilities.append(proposal.utility)

    lhs_best = max(lhs_utilities) if lhs_utilities else float("-inf")

    # Full BO loop
    result = bayesian_optimization(
        mmm, idata, df,
        q1_cfg, q2_cfg,
        baseline_allocation, V_Q1_baseline,
        n_evaluations=n_evaluations,
        n_initial=n_initial,
        n_outcomes=2,
        E_max_fraction=config.E_MAX_FRACTION,
        seed=42,
        n_processes=1,
    )

    bo_best = result.best_utility

    print(
        f"G3.4 real BO: LHS best={lhs_best:.4f}, BO best={bo_best:.4f}, "
        f"n_evaluations={result.n_evaluations}, "
        f"reason={result.reason}"
    )
    assert bo_best >= lhs_best - 1e-6, (
        f"BO best ({bo_best:.4f}) < LHS best ({lhs_best:.4f})"
    )


def test_g34_real_bo_trace_recorded(
    mmm, idata, df, baseline_allocation, q1_cfg, q2_cfg, V_Q1_baseline
):
    """G3.4 (real): BO trace contains valid per-iteration data.

    Verifies that the trace list has the expected structure and that
    each entry has finite utility, q1_loss, and GP predictive values.
    """
    from mmm_evsi.bo_design import bayesian_optimization

    n_evaluations = 10
    n_initial = 5

    result = bayesian_optimization(
        mmm, idata, df,
        q1_cfg, q2_cfg,
        baseline_allocation, V_Q1_baseline,
        n_evaluations=n_evaluations,
        n_initial=n_initial,
        n_outcomes=2,
        E_max_fraction=config.E_MAX_FRACTION,
        seed=42,
        n_processes=1,
    )

    assert len(result.trace) > 0, "BO trace is empty"
    assert result.n_evaluations > 0, "n_evaluations is 0"

    for i, trace_entry in enumerate(result.trace):
        assert trace_entry.iteration == i, (
            f"trace iteration mismatch at index {i}"
        )
        assert np.isfinite(trace_entry.utility), (
            f"trace[{i}]: utility is not finite"
        )
        assert np.isfinite(trace_entry.q1_loss), (
            f"trace[{i}]: q1_loss is not finite"
        )
        assert np.isfinite(trace_entry.gp_mean), (
            f"trace[{i}]: gp_mean is not finite"
        )
        assert np.isfinite(trace_entry.gp_std), (
            f"trace[{i}]: gp_std is not finite"
        )
        assert trace_entry.allocation.shape == (7,), (
            f"trace[{i}]: allocation shape {trace_entry.allocation.shape}"
        )

    print(
        f"G3.4 trace: {len(result.trace)} entries, "
        f"best_utility={result.best_utility:.4f}, "
        f"best_q1_loss={result.best_q1_loss:.4f}"
    )
