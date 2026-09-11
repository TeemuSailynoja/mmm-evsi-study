"""G2.4 + G2.5 — weighted Q2 solve equivalence and set_posterior rebind.

G2.4 — uniform weights (degenerate path, k-hat 0.0) + zero Q1 spend ⇒ the
weighted Q2 solve reproduces the Stage-1 baseline Q2 allocation (allclose
rtol 1e-4) and objective (rtol 1e-6). This is the arbiter equivalence gate
for W1's carry-in mechanism (b): in the zero-carry-in limit the weighted
solve must coincide with the stock ``BudgetOptimizer`` baseline.
G2.5 — real-pipeline ``set_posterior`` rebind (Stage-0 G0.2 pattern): the
objective value changes when the (resampled) draws change, and the second
rebind keeps ``opt._objective_and_grad`` object identity (no recompile).

Artifact-gated: the module skips unless the Stage-1 artifacts exist, and the
Stage-2 modules are imported lazily via ``importorskip``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 2 modules/artifacts missing — run scripts/fit_case_study.py and "
    "check docs/contracts/stage2-weighted.md"
)

K_HAT_THRESHOLD = getattr(config, "K_HAT_THRESHOLD", 0.7)

_model_file = getattr(config, "MODEL_FILE", None)
_idata_file = getattr(config, "IDATA_FILE", None)
if not (
    _model_file and _model_file.is_dir() and _idata_file and _idata_file.is_dir()
):
    pytest.skip(SKIP_MSG, allow_module_level=True)

from mmm_evsi.baseline import solve_baseline  # noqa: E402
from mmm_evsi.load_mmm import (  # noqa: E402
    load_budgets,
    load_case_study_data,
    load_mmm,
)

importance = pytest.importorskip("mmm_evsi.importance")
slsqp = pytest.importorskip("mmm_evsi.optimize_slsqp")
experiments = pytest.importorskip("mmm_evsi.experiments")


def test_g24_weighted_zero_spend_solve_equals_stage1_baseline():
    """G2.4 — uniform weights + zero Q1 spend ≡ stock Stage-1 Q2 baseline."""
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q2_cfg = budgets.q2

    baseline = solve_baseline(mmm, config.Q2_WINDOW, q2_cfg)
    assert baseline.scipy_result.success

    # §5 degenerate-path precondition: an exactly-constant ell is intercepted
    # by the wrapper (psislw itself raises "All tail values are the same")
    # into raw normalized weights, k-hat 0.0, degenerate=True.
    ell_const = np.full(config.CHAINS * config.DRAWS, 1.0)
    psis = importance.psis_weights(ell_const)
    assert psis.degenerate and psis.khat == 0.0
    assert not importance.apply_khat_policy(psis).skipped

    # Zero Q1 spend -> zero Q1->Q2 carry-in; uniform (full-posterior) weights
    # must reproduce the stock baseline optimum.
    q1_weekly_spend = np.zeros((13, len(config.CHANNEL_COLUMNS)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # benign cold-start
        weighted = slsqp.solve_q2_weighted(
            mmm,
            df,
            q2_cfg,
            q1_weekly_spend,
            minimize_kwargs={"options": {"ftol": 1e-6}},
        )
    assert not weighted.skipped
    assert np.allclose(
        weighted.budgets.values, baseline.budgets.values, rtol=1e-4, atol=0.0
    ), f"weighted budgets {weighted.budgets.values} vs baseline {baseline.budgets.values}"
    assert np.isclose(
        weighted.objective_value, baseline.objective_value, rtol=1e-6, atol=0.0
    ), (
        f"objective {weighted.objective_value:.9g} vs baseline "
        f"{baseline.objective_value:.9g}"
    )


def test_g25_set_posterior_rebind_changes_objective_keeps_identity():
    """G2.5 — first rebind recompiles; second rebind keeps object identity."""
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2
    allocation = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg).budgets

    # A real weight set: one simulated quarter's PSIS smoothed weights.
    y_star = experiments.simulate_quarter(
        mmm, idata, df, q1_cfg.window, allocation, seed=0
    )
    ell = importance.quarter_log_likelihood(
        mmm, idata, df, q1_cfg.window, allocation, y_star
    )
    psis = importance.psis_weights(ell)
    pooled = importance.pool_posterior(idata["posterior"])
    probs = np.exp(psis.smoothed_log_weights)
    posterior_r1 = importance.resample_posterior(pooled, probs, seed=0)
    posterior_r2 = importance.resample_posterior(pooled, probs, seed=1)

    opt = mmm.budget_optimizer(config.Q2_WINDOW[0], config.Q2_WINDOW[1])
    bounds = q2_cfg.boxes
    kwargs = {"options": {"ftol": 1e-6}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # benign cold-start

        # First call: the one-time bind (recompile).
        opt.set_posterior(posterior_r1)
        bound_fn = opt._objective_and_grad
        assert bound_fn is not None
        res1 = opt.allocate_budget(
            total_budget=q2_cfg.total, budget_bounds=bounds,
            minimize_kwargs=kwargs,
        )
        assert res1.scipy_result.success
        obj1 = float(res1.scipy_result.fun)

        # Second call: rebind in place — same compiled function object.
        opt.set_posterior(posterior_r2)
        assert opt._objective_and_grad is bound_fn
        res2 = opt.allocate_budget(
            total_budget=q2_cfg.total, budget_bounds=bounds,
            minimize_kwargs=kwargs,
        )
        assert res2.scipy_result.success
        obj2 = float(res2.scipy_result.fun)

    # New (resampled) draws change the objective value at the solution.
    assert not np.isclose(obj1, obj2, rtol=1e-6, atol=1e-9), (
        f"objective must change with the new resampled draws: obj1={obj1:.9g}, "
        f"obj2={obj2:.9g}"
    )