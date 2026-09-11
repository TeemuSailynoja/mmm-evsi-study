"""G1.6 — baseline budget solves: toy part runs now, real part after the fit.

The toy part reuses the Stage-0 toy plan (``total == 1000.0``) on a 13-week
window inside the toy training span and needs no ``mmm_evsi.baseline``
import. The real part (Q1/Q2 on the fitted case-study MMM) skips until the
Stage 1 artifacts exist and lazily imports ``load_mmm``/``load_budgets``/
``solve_baseline``. Raw ``mdsp_*`` channel names are the mandatory
``budget_bounds``/boxes keys — human names raise ``KeyError`` in the pinned
optimizer.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmm_evsi import config

SKIP_MSG = (
    "Stage 1 artifacts missing — run scripts/fit_case_study.py; "
    "see docs/contracts/stage1-fit.md"
)

TOY_TOTAL = 1000.0
TOY_WINDOW = ("2021-01-03", "2021-03-28")  # 13 weekly Sundays in the toy span
TOY_PLANNED = {
    "ch1": 100.0,
    "ch2": 200.0,
    "ch3": 150.0,
    "ch4": 120.0,
    "ch5": 180.0,
    "ch6": 140.0,
    "ch7": 110.0,
}  # sum == 1000.0


def _feasible_perturbations(opt, planned, boxes, seed=0):
    """Section 4.2: planned + 4 jittered feasible points (inside boxes, Σx = B).

    Each perturbation picks a distinct channel pair (channels 0..6 cyclically)
    and moves ``dx = min(hi_i - p_i, p_j - lo_j) * u`` (``u ~ Uniform(0, 1)``)
    from the lower side to the higher side, keeping every point inside the
    boxes and the sum at exactly ``B``. Points are returned in raw ``channel``
    coordinate order, i.e. already in ``_objective_and_grad``'s flat layout.
    """
    channels = list(opt.budgets_to_optimize.coords["channel"].values)
    p = np.array([planned[c] for c in channels])
    lo = np.array([boxes[c][0] for c in channels])
    hi = np.array([boxes[c][1] for c in channels])
    rng = np.random.default_rng(seed)
    points = [p.copy()]
    for k in range(4):
        i = k % len(channels)
        j = (k + 1) % len(channels)
        u = rng.uniform(0.0, 1.0)
        dx = min(hi[i] - p[i], p[j] - lo[j]) * u
        x = p.copy()
        x[i] += dx
        x[j] -= dx
        points.append(x)
    return points


def _assert_optimality_smoke(opt, planned, boxes, minimized_fun):
    """Every feasible point's objective is >= the optimizer's minimum - 1e-6."""
    fn = opt._objective_and_grad
    assert fn is not None
    for x in _feasible_perturbations(opt, planned, boxes):
        assert float(fn(x)[0]) >= minimized_fun - 1e-6


def test_toy_baseline_solves(toy_mmm):
    """Toy G1.6: optimizer converges; Σx = B; boxes ± 1e-6; optimality smoke."""
    opt = toy_mmm.budget_optimizer(TOY_WINDOW[0], TOY_WINDOW[1])
    budget_bounds = {
        c: (0.7 * TOY_PLANNED[c], 1.3 * TOY_PLANNED[c]) for c in TOY_PLANNED
    }
    res = opt.allocate_budget(total_budget=TOY_TOTAL, budget_bounds=budget_bounds)

    assert res.scipy_result.success
    assert res.scipy_result.nit >= 1
    assert abs(float(res.budgets.sum()) - TOY_TOTAL) <= 1e-6  # Σx = B
    for c in TOY_PLANNED:  # per-channel box ± 1e-6
        lo, hi = budget_bounds[c]
        assert lo - 1e-6 <= float(res.budgets.sel(channel=c)) <= hi + 1e-6

    _assert_optimality_smoke(
        opt, TOY_PLANNED, budget_bounds, float(res.scipy_result.fun)
    )


def test_real_baseline_solves():
    """Real G1.6 (user-run): Q1/Q2 solves via ``solve_baseline`` + smoke."""
    model_file = getattr(config, "MODEL_FILE", None)
    idata_file = getattr(config, "IDATA_FILE", None)
    if not (model_file and model_file.is_dir() and idata_file and idata_file.is_dir()):
        pytest.skip(SKIP_MSG)

    import warnings

    from mmm_evsi.load_mmm import load_mmm, load_budgets  # lazy (W1)
    from mmm_evsi.baseline import solve_baseline  # lazy (W1)

    mmm, idata = load_mmm()
    budgets = load_budgets()
    for quarter, window, cfg in (
        ("Q1", config.Q1_WINDOW, budgets.q1),
        ("Q2", config.Q2_WINDOW, budgets.q2),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = solve_baseline(mmm, window, cfg)

        # Q1 is contiguous with training -> must NOT emit the cold-start
        # carry-in warning. Q2 cold-starts (training ends 2018-01-28, Q2
        # starts 2018-05-06) -> the benign UserWarning is expected.
        cold_start_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and ("cold start" in str(w.message) or "not contiguous" in str(w.message))
        ]
        if quarter == "Q1":
            assert not cold_start_warnings, (
                f"Q1 (contiguous carry-in) must not emit a cold-start warning; "
                f"got {[str(w.message) for w in cold_start_warnings]}"
            )
        else:
            assert cold_start_warnings, "Q2 should emit a cold-start warning"

        assert res.quarter == quarter
        assert res.window == window
        assert res.scipy_result.success
        assert abs(float(res.budgets.sum()) - cfg.total) <= 1e-6  # Σx = B
        for c in config.CHANNEL_COLUMNS:  # per-channel box ± 1e-6
            lo, hi = cfg.boxes[c]
            assert lo - 1e-6 <= float(res.budgets.sel(channel=c)) <= hi + 1e-6
        assert res.objective_value == -float(res.scipy_result.fun)

        opt = mmm.budget_optimizer(window[0], window[1])
        _assert_optimality_smoke(
            opt, cfg.planned, cfg.boxes, float(res.scipy_result.fun)
        )