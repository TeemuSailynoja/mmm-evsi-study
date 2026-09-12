"""G3.3 — GP surrogate gate.

The GP surrogate must:
  1. Fit sweep data (X, utility)
  2. Achieve held-out rank correlation > 0.5

Toy version (N=50) validates the logic on a 2D problem.
Real version (N=20) validates on the real 7-channel model.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from mmm_evsi import config


# ======================================================================
# Toy version — validates GP surrogate logic on a 2D problem
# ======================================================================


def test_g33_toy_gp_surrogate_rank_correlation():
    """G3.3 (toy): GP surrogate fits sweep data, rank corr > 0.5."""
    toy = pytest.importorskip("toy.toy_bo")

    baseline, boxes, x_true = toy.toy_feasible_set(n_channels=2)
    rng = np.random.default_rng(42)
    n_samples = 50

    # Generate LHS samples
    designs = toy.lhs_feasible_design_toy(baseline, boxes, n_samples, rng)

    # Evaluate true utility
    X = np.array(designs)
    y = np.array([toy.toy_utility(x, x_true) for x in X])

    # Split into train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Fit GP
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kernel = RBF(length_scale=0.3) + ConstantKernel(1.0) + WhiteKernel(0.01)
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, random_state=42,
    )
    gpr.fit(X_train, y_train)

    # Predict on held-out set
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Rank correlation
    corr, pval = spearmanr(y_test, y_pred)

    print(
        f"G3.3 toy GP: rank corr={corr:.3f} (p={pval:.3g}), "
        f"train={n_train}, test={len(X_test)}"
    )
    assert corr > 0.5, (
        f"GP rank correlation {corr:.3f} ≤ 0.5 on held-out set"
    )


# ======================================================================
# Real-model version — validates GP surrogate on the 7-channel MMM
# ======================================================================


def test_g33_real_gp_surrogate_rank_correlation(
    mmm, idata, df, baseline_allocation, q1_cfg, q2_cfg, V_Q1_baseline
):
    """G3.3 (real): GP surrogate fits real sweep data, rank corr > 0.5.

    Uses N=20 for speed. Generates proposals, evaluates them, fits GP,
    and checks held-out rank correlation.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    from mmm_evsi.bo_design import (
        _baseline_to_array,
        evaluate_proposal,
        generate_feasible_proposal,
    )

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    n_samples = 20

    # Generate proposals and evaluate
    X_list = []
    y_list = []
    for i in range(n_samples):
        alloc = generate_feasible_proposal(
            baseline_allocation, q1_cfg, rng,
            E_max, V_Q1_baseline, n_iter=10,
        )
        proposal = evaluate_proposal(
            mmm, idata, df, alloc, q1_cfg, q2_cfg,
            V_Q1_baseline, n_outcomes=2, seed=i,
            n_processes=1,
        )
        X_list.append(_baseline_to_array(proposal.allocation))
        y_list.append(proposal.utility)

    X = np.array(X_list)
    y = np.array(y_list)

    # Split into train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Fit GP
    kernel = RBF(length_scale=np.ones(7)) + ConstantKernel(1.0) + WhiteKernel(0.1)
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, random_state=42,
    )
    gpr.fit(X_train, y_train)

    # Predict on held-out set
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Rank correlation
    corr, pval = spearmanr(y_test, y_pred)

    print(
        f"G3.3 real GP: rank corr={corr:.3f} (p={pval:.3g}), "
        f"train={n_train}, test={len(X_test)}, "
        f"y_train_range=[{y_train.min():.2f}, {y_train.max():.2f}]"
    )
    assert corr > 0.5, (
        f"GP rank correlation {corr:.3f} ≤ 0.5 on held-out set "
        f"(train={n_train}, test={len(X_test)})"
    )


def test_g33_real_gp_surrogate_backend_protocol(
    mmm, idata, df, baseline_allocation, q1_cfg, q2_cfg, V_Q1_baseline
):
    """G3.3 (real): GP backend protocol works (fit → predict → acquire).

    Verifies that the _SklearnGP backend implements the abstract protocol
    (fit, predict, acquisition) correctly on real data.
    """
    from mmm_evsi.bo_design import (
        _SklearnGP,
        _baseline_to_array,
        evaluate_proposal,
        generate_feasible_proposal,
    )

    rng = np.random.default_rng(42)
    E_max = config.E_MAX_FRACTION * q1_cfg.total
    n_samples = 10

    # Generate and evaluate
    X_list = []
    y_list = []
    for i in range(n_samples):
        alloc = generate_feasible_proposal(
            baseline_allocation, q1_cfg, rng,
            E_max, V_Q1_baseline, n_iter=10,
        )
        proposal = evaluate_proposal(
            mmm, idata, df, alloc, q1_cfg, q2_cfg,
            V_Q1_baseline, n_outcomes=2, seed=i,
            n_processes=1,
        )
        X_list.append(_baseline_to_array(proposal.allocation))
        y_list.append(proposal.utility)

    X = np.array(X_list)
    y = np.array(y_list)

    # Fit GP backend
    backend = _SklearnGP()
    backend.fit(X, y)

    # Predict
    y_pred, y_std = backend.predict(X)
    assert y_pred.shape == y.shape, (
        f"predict output shape {y_pred.shape} != input shape {y.shape}"
    )
    assert np.all(np.isfinite(y_pred)), "GP predictions contain non-finite values"
    assert np.all(np.isfinite(y_std)), "GP std contains non-finite values"

    # Acquisition (EI)
    y_best = float(y.max())
    ei = backend.acquisition(X, y_best)
    assert ei.shape == y.shape, (
        f"acquisition output shape {ei.shape} != input shape {y.shape}"
    )
    assert np.all(np.isfinite(ei)), "EI values contain non-finite values"

    # EI should be non-negative (Expected Improvement ≥ 0)
    assert np.all(ei >= -1e-10), (
        f"EI values should be ≥ 0, got min={ei.min():.6g}"
    )

    print(
        f"G3.3 GP backend: fit={len(X)} points, "
        f"y_best={y_best:.4f}, EI range=[{ei.min():.4f}, {ei.max():.4f}]"
    )
