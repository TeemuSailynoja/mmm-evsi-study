"""G2.0 — analytic toy EVSI end-to-end, CRN-mandated VoE gate (runs now).

The scalar conjugate Gaussian-Gaussian toy (``toy.toy_gaussian``, §3 of
docs/contracts/stage2-weighted.md) drives the real ``importance.py`` code
paths. ``run_toy_evsi`` estimates the value of exploration
``VoE(a) = U(a) - U(a0)`` over N=5,000 simulated Q1 quarters; the gate
compares the MC estimate to the exact closed form with the pinned relative
tolerance ``|voE_mc - voE_analytic| <= 0.10 * voE_analytic``.

Common random numbers are MANDATORY (§2.4/§3): both arms (``a`` vs ``a0``)
share the same ``theta_s`` prior draws and the same standardized Gaussian
noise, so the MC SE of the VoE difference is ~2-3% at N=5,000 and the 0.10
tolerance is calibrated for CRN. The a == a0 arm is therefore asserted to
produce an (essentially) exact zero VoE — an independent-draws pairing would
differ by the MC SE (~0.01-0.05), separating CRN from independent seeding.

W1 implements ``toy.toy_gaussian`` in parallel with this file; the module
skips cleanly until it lands.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmm_evsi import config

toy = pytest.importorskip("toy.toy_gaussian")

MIN_POOLED_DRAWS = getattr(config, "MIN_POOLED_DRAWS", 25)

N_OUTCOMES = 5_000  # pinned; §7.5 allows 2,000 if this proves slow (recorded)
SEED = 0  # deterministic seeds, single fixed-seed run is stable
A0 = 0.0


def test_g20_toy_voE_matches_analytic_with_crn():
    """G2.0 (a ∈ {0, 1, 2}): MC VoE within 0.10 relative of the closed form."""
    spec = toy.ToySpec()
    assert spec.n_prior_draws >= MIN_POOLED_DRAWS, (
        "the toy's prior draws must satisfy the psislw tail-fit floor "
        f"(config.MIN_POOLED_DRAWS = {MIN_POOLED_DRAWS})"
    )

    # Sanity-pin the §3 closed forms (the table's decimals are sanity checks).
    assert np.isclose(
        toy.value_of_exploration_analytic(spec, 1.0),
        416 / 113 - 208 / 61,
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.isclose(
        toy.value_of_exploration_analytic(spec, 2.0),
        208 / 55 - 208 / 61,
        rtol=1e-9,
        atol=1e-9,
    )

    for a in (0.0, 1.0, 2.0):
        res = toy.run_toy_evsi(
            spec, a=a, a0=A0, n_outcomes=N_OUTCOMES, seed=SEED
        )
        assert np.isfinite(res.utility_mc) and np.isfinite(res.utility_analytic)
        assert res.n_skipped >= 0  # skip rate is only reported, not gated

        voe_analytic = toy.value_of_exploration_analytic(spec, a, a0=A0)
        if a == A0:
            # CRN pin: identical arms (same draws + same noise) -> VoE ~ 0,
            # NOT the MC SE of two independent arms.
            assert abs(res.value_of_exploration_mc) <= 1e-6, (
                "the a == a0 arm must be ~0 under CRN (shared theta_s and "
                "standardized noise); an independent-draws pairing would "
                "differ at the MC-SE scale (~1e-2)"
            )
        else:
            assert (
                abs(res.value_of_exploration_mc - voe_analytic)
                <= 0.10 * abs(voe_analytic)
            ), (
                f"a={a}: |voE_mc - voE_analytic| = "
                f"{abs(res.value_of_exploration_mc - voe_analytic):.4f} exceeds "
                f"0.10 * voE_analytic = {0.10 * abs(voe_analytic):.4f} "
                f"(CRN calibrated; deterministic seed {SEED})"
            )