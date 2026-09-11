"""G2.8 — resampling RNG noise: split-half stability (artifact-gated).

The multinomial resample is unbiased by construction, so its only effect on
the per-allocation utility is RNG noise across ``resample_seeds``. With
``resample_seeds`` given, ``evaluate_allocation`` uses exactly ONE simulated
outcome (the k-hat policy/ell are identical across seeds — only the resample
RNG changes) and the loop iterates over the seeds. The gate requires ≥ 20
seeds total, split into two DISJOINT halves of ≥ 10 seeds each, and asserts
the no-bias split-half stability bound

    |mean_A - mean_B| <= 3 * sd_pooled / sqrt(n_half),   n_half >= 10.

Per-seed utilities are reported (min/max/mean/sd); the user produces the
notebook plot. Artifact-gated.
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

N_SEEDS_TOTAL = 20  # >= 20 seeds, two disjoint halves of >= 10 each
N_HALF = 10
SEEDS = list(range(N_SEEDS_TOTAL))
SEED = 0  # drives the ONE simulated outcome (seed + 0)


def test_g28_resampling_seed_sweep_split_half_stability():
    mmm, idata = load_mmm()
    df = load_case_study_data()
    budgets = load_budgets()
    q1_cfg, q2_cfg = budgets.q1, budgets.q2
    allocation = solve_baseline(mmm, config.Q1_WINDOW, q1_cfg).budgets

    ev = importance.evaluate_allocation(
        mmm, idata, df, allocation, q1_cfg, q2_cfg,
        n_outcomes=1, seed=SEED, n_processes=2, resample_seeds=SEEDS,
    )
    utilities = np.asarray(ev.utilities, dtype=float)
    assert utilities.shape == (N_SEEDS_TOTAL,), (
        "expected one utility per resample seed (the single outcome is "
        "either accepted for every seed or skipped for every seed)"
    )
    assert np.all(np.isfinite(utilities))
    assert ev.n_skipped == 0  # identical policy across seeds -> all or none

    group_a, group_b = utilities[:N_HALF], utilities[N_HALF:]
    assert len(group_a) >= 10 and len(group_b) >= 10  # disjoint halves

    mean_a, mean_b = float(group_a.mean()), float(group_b.mean())
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)
    n_pool = len(group_a) + len(group_b)
    sd_pooled = np.sqrt(
        ((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b)
        / (n_pool - 2)
    )
    bound = 3.0 * sd_pooled / np.sqrt(N_HALF)
    assert abs(mean_a - mean_b) <= bound, (
        f"split-half mean drift |{mean_a:.9g} - {mean_b:.9g}| = "
        f"{abs(mean_a - mean_b):.9g} exceeds 3*sd_pooled/sqrt(n_half) = "
        f"{bound:.9g} (n_half={N_HALF})"
    )

    # Per-seed utility summary (reported; the notebook plot is user-side).
    print(
        "G2.8 per-seed utilities: "
        f"min={utilities.min():.9g} max={utilities.max():.9g} "
        f"mean={utilities.mean():.9g} sd={utilities.std(ddof=1):.9g} "
        f"mean_A={mean_a:.9g} mean_B={mean_b:.9g} "
        f"bound={bound:.9g}"
    )