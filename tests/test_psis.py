"""G2.1, G2.1b, G2.2 — PSIS wrapper behavior on the analytic toy (runs now).

G2.1 — one fixed simulated quarter (seed 0, a = 2.0): raw normalized weights
equal the manual formula, weighted/resampled moments reproduce the analytic
(μ*, v), smoothed weights are normalized and k-hat finite.
G2.1b — the psislw convention is pinned on ``(-ell).azstats.psislw``
(``.stats.psislw`` is NOT registered in arviz-stats 1.3.2; PLAN.md and
docs/methods.md are outdated on this point). A single-dominant-draw heavy
right tail gives ``khat(-ell) > 0.7`` with the dominant draw's smoothed
weight BELOW its raw weight; ``khat(+ell)`` is NOT computed on the flat-tail
examples — the pinned implementation raises ``ValueError``.
G2.2 — one k-hat per simulated quarter, smoothed weights on a single pooled
``("sample",)`` dim, chain × draw pooling pinned.

W1 implements ``toy.toy_gaussian`` and ``mmm_evsi.importance`` in parallel
with this file; the module skips cleanly until they land.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import arviz as az  # noqa: F401  (registers the DataArray .azstats accessor)
from scipy.special import logsumexp

from mmm_evsi import config

toy = pytest.importorskip("toy.toy_gaussian")
importance = pytest.importorskip("mmm_evsi.importance")

K_HAT_THRESHOLD = getattr(config, "K_HAT_THRESHOLD", 0.7)
MIN_POOLED_DRAWS = getattr(config, "MIN_POOLED_DRAWS", 25)

_A = 2.0  # pinned toy allocation knob (higher a = more precise observation)
_SEED = 0  # one fixed simulated quarter (seed 0, a = 2.0)

# The toy's posterior variable name for theta (drawn from the prior).
_THETA_VAR = "theta"


def _fixed_quarter():
    """One fixed simulated quarter: seed 0, a = 2.0, full toy prior draws."""
    spec = toy.ToySpec()
    draws = toy.prior_draws(spec)  # (S,)
    y_star = toy.simulate_quarter(spec, _A, np.random.default_rng(_SEED))
    ell = toy.joint_log_likelihood(spec, _A, y_star)  # (S,) joint over 13 weeks
    mu_star, v = toy.posterior_sufficient(spec, _A, y_star)
    return spec, draws, y_star, ell, mu_star, v


def test_g21_normalized_and_weighted_moments_match_closed_forms():
    """G2.1 (a)-(d): manual weights, weighted/resampled moments, smoothing."""
    spec, draws, y_star, ell, mu_star, v = _fixed_quarter()
    S = ell.shape[0]
    assert ell.ndim == 1 and S == spec.n_prior_draws
    assert S >= MIN_POOLED_DRAWS

    # (a) raw normalized weights equal the manual formula exactly.
    raw = importance.normalized_log_weights(ell)
    manual = ell - logsumexp(ell)
    np.testing.assert_allclose(raw, manual, rtol=0.0, atol=1e-12)
    assert abs(float(np.exp(raw).sum()) - 1.0) <= 1e-12

    # (b) raw-weights mean of theta_s reproduces mu* within 3 * SE_w with
    #     SE_w = sqrt(var_w / ESS_w), ESS_w from the (Kish) weights.
    p = np.exp(raw)
    theta_bar = float(np.sum(p * draws))
    var_w = float(np.sum(p * (draws - theta_bar) ** 2))
    ess_w = 1.0 / float(np.sum(p**2))
    se_w = np.sqrt(var_w / ess_w)
    assert abs(theta_bar - mu_star) <= 3.0 * se_w, (
        f"weighted mean {theta_bar:.6f} vs mu* {mu_star:.6f} "
        f"(3*SE_w = {3.0 * se_w:.6f})"
    )

    # (c) resampled draws reproduce mu* within 3*sqrt(v/M) and v within
    #     rtol 0.10.
    posterior = toy.posterior_dataset(spec, draws)  # (chain=1, draw=S)
    pooled = importance.pool_posterior(posterior)
    assert "sample" in pooled.dims
    resampled = importance.resample_posterior(pooled, p, n=S, seed=_SEED)
    theta_r = np.asarray(resampled[_THETA_VAR].values).reshape(-1)
    assert theta_r.shape == (S,)
    mean_r = float(np.mean(theta_r))
    var_r = float(np.var(theta_r, ddof=1))
    assert abs(mean_r - mu_star) <= 3.0 * np.sqrt(v / S), (
        f"resampled mean {mean_r:.6f} vs mu* {mu_star:.6f} "
        f"(3*sqrt(v/M) = {3.0 * np.sqrt(v / S):.6f})"
    )
    assert np.isclose(var_r, v, rtol=0.10), (
        f"resampled variance {var_r:.6f} vs v {v:.6f}"
    )

    # (d) smoothed weights: normalized exp-sum == 1 (atol 1e-6), khat finite.
    psis = importance.psis_weights(ell)
    smoothed = psis.smoothed_log_weights
    assert smoothed.shape == (S,)
    assert abs(float(np.exp(smoothed).sum()) - 1.0) <= 1e-6
    assert np.isfinite(psis.khat)


def test_g21b_pins_psislw_convention_on_pinned_degenerate_example():
    """G2.1b (pin): the exact pinned call + normalization + khat(+ell) raise.

    Pinned call: ``(-ell).azstats.psislw(dim="sample")`` — NOT
    ``.stats.psislw`` (the ``stats`` accessor is not registered in
    arviz-stats 1.3.2). Output 0 is the normalized smoothed POSITIVE
    log-weights in the ``ell`` space (``exp``-sum == 1); output 1 is the
    fitted k-hat.
    """
    S = 50_000  # toy-default prior-draw count; pinned degenerate construction
    ell_degen = np.full(S, -8.0)
    ell_degen[0] = 0.0  # one dominant draw

    neg_lw = xr.DataArray(-ell_degen, dims="sample")
    smoothed, khat = neg_lw.azstats.psislw(dim="sample")
    assert abs(float(np.exp(smoothed).sum()) - 1.0) <= 1e-6
    assert smoothed.shape == (S,)

    # arviz-stats 1.3.2 details (verified during gate implementation): the
    # input is negated internally and the GPD is fit on the right tail of
    # ``ell``; on the exactly-flat example the tail after negating is flat
    # (+8), so ``khat(+ell)`` raises ValueError("All tail values are the
    # same") — this is why G2.1b does NOT evaluate khat(+ell).
    with pytest.raises(ValueError):
        xr.DataArray(ell_degen, dims="sample").azstats.psislw(dim="sample")

    # The flat example also trips the GPD flat-tail guard on the -ell side:
    # khat(-ell) is non-finite (nan) in arviz-stats 1.3.2 rather than the
    # §0.3 value 1.39 (that claim was verified on a heavy-right-tail shape;
    # see test_g21b_dominant_draw_khat_and_smoothing below). The §0.5 policy
    # treats any non-finite k-hat as a skip exactly like k-hat > 0.7.
    assert not np.isfinite(float(khat))


def test_g21b_dominant_draw_khat_and_smoothing():
    """G2.1b (signal): khat(-ell) > 0.7, exp-sum == 1, smoothed < raw.

    On the §0.3 shape — a single dominant draw on a 500-draw heavy right
    tail (flat floor far below), which also satisfies the contract's
    "khat(+ell) not computed — flat tail raises ValueError" clause — the
    convention call returns:
      * khat(-ell) > 0.7 (the influential-draw signal the policy skips on);
      * exp(smoothed) summing to 1 (atol 1e-6);
      * a smoothed dominant-draw weight BELOW its raw normalized weight
        (the heavy tail is smoothed down).
    """
    ell = np.r_[
        0.0,  # single dominant draw
        -8.0 - np.linspace(0.0, 400.0, 400),  # heavy right tail (shoulder)
        np.full(99, -5_000.0),  # flat floor -> khat(+ell) tail is flat
    ]
    dom = int(np.argmax(ell))

    neg_lw = xr.DataArray(-ell, dims="sample")
    smoothed, khat = neg_lw.azstats.psislw(dim="sample")

    assert float(khat) > K_HAT_THRESHOLD, f"khat(-ell) = {float(khat)}"
    assert abs(float(np.exp(smoothed).sum()) - 1.0) <= 1e-6

    p_smoothed = np.exp(np.asarray(smoothed))
    p_raw = np.exp(ell - logsumexp(ell))
    assert p_smoothed[dom] < p_raw[dom], (
        f"dominant draw's smoothed weight {p_smoothed[dom]:.6f} must be "
        f"below its raw normalized weight {p_raw[dom]:.6f} (tail smoothed down)"
    )

    # khat(+ell) is not computable on this example either: the negated right
    # tail is the flat floor, all values identical -> ValueError.
    with pytest.raises(ValueError):
        xr.DataArray(ell, dims="sample").azstats.psislw(dim="sample")


def test_g22_pooling_gives_one_khat_per_quarter_on_sample_dim():
    """G2.2 — one k-hat per quarter; single pooled ("sample",) dim."""
    spec, draws, y_star, ell, mu_star, v = _fixed_quarter()
    S = ell.shape[0]
    assert ell.ndim == 1

    psis = importance.psis_weights(ell)
    # one k-hat per simulated quarter: a scalar float, not a per-week vector
    assert isinstance(psis.khat, float) and np.ndim(psis.khat) == 0
    # smoothed weights live on the single pooled ("sample",) dim
    assert psis.smoothed_log_weights.shape == (S,)

    # chain × draw pooled input (any chain layout) → identical k-hat and
    # smoothed weights (chain-major pooling, §0.4)
    ell_da = xr.DataArray(ell.reshape(1, -1), dims=("chain", "draw"))
    psis2 = importance.psis_weights(ell_da)
    assert psis2.khat == psis.khat
    np.testing.assert_array_equal(
        psis2.smoothed_log_weights, psis.smoothed_log_weights
    )

    # The toy's quarter-ell joins all 13 weeks: replacing it with any single
    # week's slice changes the result semantics (per-week, not per-quarter).
    week_ell = toy.joint_log_likelihood(spec, _A, y_star[:1])
    assert week_ell.shape == ell.shape
    assert not np.allclose(week_ell, ell)
    assert importance.psis_weights(week_ell).khat != psis.khat