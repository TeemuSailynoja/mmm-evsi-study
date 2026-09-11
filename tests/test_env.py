"""G0.1 — PR-pinned install exposes the no-recompile rebind API."""
from __future__ import annotations


def test_shared_posterior_importable():
    from pymc_marketing.pytensor_utils import SharedPosterior

    assert SharedPosterior is not None
    assert isinstance(SharedPosterior, type)


def test_budget_optimizer_set_posterior():
    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    assert hasattr(BudgetOptimizer, "set_posterior")
    assert callable(BudgetOptimizer.set_posterior)


def test_extract_accepts_shared_posterior_kwarg():
    import inspect

    from pymc_marketing.pytensor_utils import extract_response_distribution

    sig = inspect.signature(extract_response_distribution)
    assert "shared_posterior" in sig.parameters
