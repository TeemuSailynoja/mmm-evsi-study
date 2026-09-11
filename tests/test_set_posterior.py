"""G0.2 — set_posterior rebinds draws without recompiling."""
from __future__ import annotations

import numpy as np

CHANNELS = [f"ch{i}" for i in range(1, 8)]


def _perturb_idata(idata, factor=1.05):
    """Return the posterior Dataset with every float variable scaled by ``factor``.

    dims/coords are preserved via DataArray algebra. Do NOT mutate through a
    DataTree node with tuple assignment (``tree_node[name] = (dims, vals)``) —
    that drops dims/coords and makes ``set_posterior`` raise. ``set_posterior``
    accepts a Dataset directly.
    """
    posterior = idata["posterior"].to_dataset().copy()  # bracket access only
    for name in list(posterior.data_vars):
        da = posterior[name]
        if da.dtype.kind == "f":
            posterior[name] = da * factor
    return posterior


def test_set_posterior_rebinds_without_recompile(toy_mmm):
    """The second ``set_posterior`` call swaps draws in place (no recompile).

    The implementation docstring states the first call costs one recompile
    (draws move into shared variables), and later calls are ``set_value`` only.
    Asserting identity after a single call would be wrong and fail, so the
    no-recompile property is asserted across the **second** call.
    """
    opt = toy_mmm.budget_optimizer("2020-08-09", "2020-11-01")

    fn0 = opt._objective_and_grad
    assert fn0 is not None

    bounds = {c: (0.0, 1000.0) for c in CHANNELS}

    # First call: one-time bind that recompiles the objective.
    opt.set_posterior(_perturb_idata(opt.idata, 1.05))
    bound_fn = opt._objective_and_grad
    assert bound_fn is not None

    res1 = opt.allocate_budget(total_budget=1000.0, budget_bounds=bounds)
    assert res1.scipy_result.success
    obj1 = float(res1.scipy_result.fun)

    # Second call: rebind in place — same compiled function object (no recompile).
    opt.set_posterior(_perturb_idata(opt.idata, 0.90))
    assert opt._objective_and_grad is bound_fn

    res2 = opt.allocate_budget(total_budget=1000.0, budget_bounds=bounds)
    assert res2.scipy_result.success
    obj2 = float(res2.scipy_result.fun)

    # Different draws change the objective value at the solution.
    assert not np.isclose(obj1, obj2, rtol=1e-6, atol=1e-9)
