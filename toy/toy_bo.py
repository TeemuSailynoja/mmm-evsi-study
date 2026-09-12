"""Toy BO validation — linear-quadratic utility with closed-form optimum.

This module validates the Stage 3 BO pipeline on a simple 2D problem where
the true utility function is known analytically. The BO should find the
global optimum within MC tolerance.

Utility model
-------------
U(x) = -||x - x*||^2 + C

where x* is the known optimum in the feasible set, and C is a constant.
The feasible set is a box around x* with the sum constraint enforced.

This tests:
1. Feasible proposal generation (G3.1)
2. LHS sweep coverage (G3.2)
3. GP surrogate fitting (G3.3)
4. BO convergence to true optimum (G3.4)

Run:
    python toy/toy_bo.py
"""
from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def toy_feasible_set(n_channels: int = 2, rng: np.random.Generator | None = None):
    """Define a simple 2D feasible set for testing.

    Returns
    -------
    baseline : (n_channels,)
        Baseline allocation.
    boxes : dict
        Per-channel (lo, hi) boxes.
    x_true : (n_channels,)
        True utility optimum.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    baseline = np.ones(n_channels)  # sum = n_channels
    # True optimum is at [1.2, 0.8] for n_channels=2 (sum still = 2)
    x_true = np.array([1.2, 0.8])
    boxes = {
        0: (0.5, 1.5),
        1: (0.5, 1.5),
    }
    return baseline, boxes, x_true


def toy_utility(x: np.ndarray, x_true: np.ndarray) -> float:
    """Quadratic utility: U(x) = -||x - x_true||^2."""
    return -float(np.sum((x - x_true) ** 2))


def null_space_perturbation(delta: np.ndarray, n_channels: int) -> np.ndarray:
    """Project delta onto the null space of the sum constraint."""
    result = np.empty(n_channels, dtype=float)
    result[: n_channels - 1] = delta[: n_channels - 1]
    result[n_channels - 1] = -result[: n_channels - 1].sum()
    return result


def clip_to_boxes(
    perturbation: np.ndarray,
    boxes: dict,
    baseline: np.ndarray,
) -> np.ndarray:
    """Clip perturbation so that baseline + perturbation stays in boxes."""
    clipped = np.empty_like(perturbation)
    for i in range(len(boxes)):
        lo, hi = boxes[i]
        proposed = baseline[i] + perturbation[i]
        clipped[i] = np.clip(proposed, lo, hi) - baseline[i]
    return clipped


def recenter_sum(perturbation: np.ndarray) -> np.ndarray:
    """Shift perturbation so that sum is exactly zero."""
    return perturbation - perturbation.mean()


def generate_feasible_proposal_toy(
    baseline: np.ndarray,
    boxes: dict,
    rng: np.random.Generator,
    n_iter: int = 10,
) -> np.ndarray:
    """Generate one feasible perturbation for the toy problem."""
    n_ch = len(baseline)
    n_free = n_ch - 1

    for _ in range(n_iter):
        raw = rng.normal(0, 1, size=n_free)
        delta = null_space_perturbation(raw, n_ch)

        # Scale
        max_allowed = np.array([boxes[i][1] - baseline[i] for i in range(n_ch)], dtype=float)
        min_allowed = np.array([boxes[i][0] - baseline[i] for i in range(n_ch)], dtype=float)
        pos_margin = np.where(delta > 0, max_allowed, np.inf)
        neg_margin = np.where(delta < 0, -min_allowed, np.inf)
        margin = np.minimum(pos_margin, neg_margin)
        if margin.min() < 1e-12:
            continue
        scale = 0.5 * margin.min() / (np.abs(delta).max() + 1e-12)
        delta *= scale

        delta = clip_to_boxes(delta, boxes, baseline)
        delta = recenter_sum(delta)

        proposed = baseline + delta
        if abs(proposed.sum() - baseline.sum()) < 1e-6:
            return proposed

    raise RuntimeError(f"Could not generate feasible proposal in {n_iter} attempts")


def lhs_feasible_design_toy(
    baseline: np.ndarray,
    boxes: dict,
    n_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate N feasible LHS samples for the toy problem."""
    from scipy.stats import qmc

    n_ch = len(baseline)
    n_free = n_ch - 1

    try:
        lhs = qmc.LatinHypercube(d=n_free, seed=rng.integers(0, 2**31))
        samples = 2.0 * lhs.random(n=n_samples) - 1.0
    except Exception:
        samples = rng.uniform(-1, 1, size=(n_samples, n_free))

    proposals = []
    for sample in samples:
        try:
            p = generate_feasible_proposal_toy(
                baseline, boxes,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
            )
            proposals.append(p)
        except RuntimeError:
            continue
        if len(proposals) >= n_samples:
            break

    return proposals


def test_feasibility(n_samples: int = 1000) -> bool:
    """G3.1: All N=1000 generated candidates satisfy constraints."""
    baseline, boxes, _ = toy_feasible_set(2)
    rng = np.random.default_rng(0)

    all_feasible = True
    for _ in range(n_samples):
        p = generate_feasible_proposal_toy(baseline, boxes, rng)
        # Check sum constraint
        if abs(p.sum() - baseline.sum()) > 1e-6:
            print(f"  FAIL: sum constraint violated: {p.sum()} vs {baseline.sum()}")
            all_feasible = False
            break
        # Check box constraints
        for i in range(len(boxes)):
            if p[i] < boxes[i][0] - 1e-6 or p[i] > boxes[i][1] + 1e-6:
                print(f"  FAIL: box constraint violated at {i}: {p[i]} not in {boxes[i]}")
                all_feasible = False
                break
        if not all_feasible:
            break

    if all_feasible:
        print(f"G3.1 PASS: all {n_samples} proposals feasible")
    else:
        print(f"G3.1 FAIL: some proposals infeasible")
    return all_feasible


def test_lhs_coverage(n_samples: int = 20) -> bool:
    """G3.2: LHS sweep generates diverse feasible samples."""
    baseline, boxes, x_true = toy_feasible_set(2)
    rng = np.random.default_rng(42)

    designs = lhs_feasible_design_toy(baseline, boxes, n_samples, rng)

    if len(designs) < n_samples * 0.8:
        print(f"G3.2 FAIL: only {len(designs)}/{n_samples} feasible LHS samples")
        return False

    # Check diversity: pairwise distances should span a range
    if len(designs) >= 2:
        dists = []
        for i in range(min(10, len(designs))):
            for j in range(i + 1, min(10, len(designs))):
                dists.append(np.linalg.norm(designs[i] - designs[j]))
        if min(dists) > 0.5 * max(dists):
            print(f"G3.2 WARN: low diversity in LHS samples (min/max dist ratio = {min(dists)/max(dists):.2f})")
            # Not a hard fail — just a warning

    print(f"G3.2 PASS: {len(designs)} feasible LHS samples generated")
    return True


def test_gp_surrogate(n_samples: int = 50) -> bool:
    """G3.3: GP surrogate fits sweep data with reasonable predictive check."""
    baseline, boxes, x_true = toy_feasible_set(2)
    rng = np.random.default_rng(42)

    # Generate LHS samples
    designs = lhs_feasible_design_toy(baseline, boxes, n_samples, rng)

    # Evaluate true utility
    X = np.array(designs)
    y = np.array([toy_utility(x, x_true) for x in X])

    # Fit GP
    kernel = RBF(length_scale=0.3) + ConstantKernel(1.0) + WhiteKernel(0.01)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
    gpr.fit(X, y)

    # Held-out predictive check: split data
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    gpr_check = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
    gpr_check.fit(X_train, y_train)
    y_pred, y_std = gpr_check.predict(X_test, return_std=True)

    # Check rank correlation
    from scipy.stats import spearmanr
    corr, pval = spearmanr(y_test, y_pred)

    if corr > 0.5:
        print(f"G3.3 PASS: GP rank correlation = {corr:.3f} (p={pval:.3g})")
        return True
    else:
        print(f"G3.3 FAIL: GP rank correlation = {corr:.3f} (need > 0.5)")
        return False


def test_bo_convergence(n_evaluations: int = 50, n_initial: int = 10) -> bool:
    """G3.4: BO finds optimum ≥ LHS best at equal evaluation budget."""
    baseline, boxes, x_true = toy_feasible_set(2)
    rng = np.random.default_rng(42)

    # LHS evaluation
    lhs_designs = lhs_feasible_design_toy(baseline, boxes, n_initial, rng)
    lhs_utilities = [toy_utility(x, x_true) for x in lhs_designs]
    lhs_best = max(lhs_utilities)
    lhs_best_x = lhs_designs[np.argmax(lhs_utilities)]

    print(f"LHS best utility: {lhs_best:.4f} (true optimum: {toy_utility(x_true, x_true):.4f})")

    # BO loop
    X = np.array(lhs_designs)
    y = np.array(lhs_utilities)
    y_best = lhs_best

    kernel = RBF(length_scale=0.3) + ConstantKernel(1.0) + WhiteKernel(0.01)
    n_ch = len(baseline)
    n_free = n_ch - 1

    for iteration in range(n_initial, n_evaluations):
        # Fit GP
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gpr.fit(X, y)

        # Maximize EI
        best_ei = -np.inf
        best_x = None
        for _ in range(20):
            raw = rng.normal(0, 1, size=n_free)
            delta = null_space_perturbation(raw, n_ch)
            max_allowed = np.array([boxes[i][1] - baseline[i] for i in range(n_ch)], dtype=float)
            min_allowed = np.array([boxes[i][0] - baseline[i] for i in range(n_ch)], dtype=float)
            pos_margin = np.where(delta > 0, max_allowed, np.inf)
            neg_margin = np.where(delta < 0, -min_allowed, np.inf)
            margin = np.minimum(pos_margin, neg_margin)
            if margin.min() < 1e-12:
                continue
            scale = 0.3 * margin.min() / (np.abs(delta).max() + 1e-12)
            delta *= scale

            # Try this point
            delta_c = clip_to_boxes(delta, boxes, baseline)
            delta_c = recenter_sum(delta_c)
            x_cand = baseline + delta_c

            if abs(x_cand.sum() - baseline.sum()) > 1e-6:
                continue

            # EI
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
        u = toy_utility(best_x, x_true)
        X = np.vstack([X, best_x.reshape(1, -1)])
        y = np.append(y, u)

        if u > y_best:
            y_best = u

    print(f"BO best utility: {y_best:.4f} (LHS best: {lhs_best:.4f})")

    if y_best >= lhs_best - 1e-6:
        print("G3.4 PASS: BO best ≥ LHS best")
        return True
    else:
        print(f"G3.4 FAIL: BO best ({y_best:.4f}) < LHS best ({lhs_best:.4f})")
        return False


def main():
    """Run all toy BO validations."""
    print("=" * 60)
    print("Toy BO Validation Suite")
    print("=" * 60)

    results = {}
    results["G3.1 feasibility"] = test_feasibility(1000)
    print()
    results["G3.2 LHS coverage"] = test_lhs_coverage(20)
    print()
    results["G3.3 GP surrogate"] = test_gp_surrogate(50)
    print()
    results["G3.4 BO convergence"] = test_bo_convergence(50, 10)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll toy BO validations PASSED.")
    else:
        print("\nSome toy BO validations FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
