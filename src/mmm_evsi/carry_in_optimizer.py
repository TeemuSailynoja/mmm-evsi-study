"""Stage 2b — shared Q1→Q2 carry-in optimizer (single compile).

Wraps the pinned ``BudgetOptimizer`` so the Q1→Q2 adstock carry-in enters the
Q2 objective through ONE shared pytensor variable (``q1_carry_in``) instead
of numpy constants baked at build time. Per solve we only swap (i) draws via
``set_posterior`` (Stage-0 semantics) and (ii) the carry-in spend via
``set_q1_carry_in`` — no recompilation. See
docs/contracts/stage2b-carry-in.md.
"""
from __future__ import annotations

import numpy as np
import pytensor
from pymc import do
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc_marketing.mmm.budget_optimizer import (
    MediaVariable,
    OptimizationVariables,
)


class CarryInBudgetOptimizer:
    """A pinned ``BudgetOptimizer`` with a shared Q1→Q2 carry-in variable.

    ``__init__`` builds the stock optimizer for the window and then REBUILDS
    its step-7 substitution so ``carry_in_values`` is a shared variable
    instead of the baked numpy constants, compiling the objective+gradient
    once. The wrapped optimizer stays the single owner of ``set_posterior``
    (first call recompiles, later calls rebind in place) and
    ``allocate_budget``; this class only adds the carry-in seam.

    Parameters
    ----------
    mmm : pymc_marketing.mmm.MMM
        Fitted MMM; ``mmm.budget_optimizer(start_date, end_date)`` is the
        inner optimizer this wrapper is built from.
    start_date, end_date : str or pd.Timestamp
        Decision window (e.g. the Q2 window). Must admit carry-in periods
        (``carry_in_periods > 0``), guaranteed for the Q2 window.

    Raises
    ------
    ValueError
        If the window/model has no carry-in periods.
    """

    def __init__(self, mmm, start_date, end_date):
        inner = mmm.budget_optimizer(start_date, end_date)
        self._inner = inner
        self._n_channels = int(np.prod(inner._budget_shape))
        self._q1_carry_in: pytensor.compile.sharedvalue.SharedVariable | None = None
        self._variables: OptimizationVariables | None = None
        self._prepare_carry_in_periods()
        self._rebuild_shared_carry_in()

    def _prepare_carry_in_periods(self) -> None:
        """Resolve the carry-in length, resizing the model when needed.

        The pinned ``create_zero_dataset`` only prepends carry-in rows when
        the window is contiguous with training data; for a window like Q2
        (first decision week 14 weeks after train end) it cold-starts with
        ZERO carry-in rows, so ``inner.carry_in_periods == 0`` while
        ``adstock_periods == l_max == 6``. The Q1→Q2 spend still seeds the
        adstock state over ``l_max`` weeks, so when the model has no
        carry-in rows we PREPEND ``adstock_periods`` empty rows to the
        channel-data node — exactly the block a contiguous window would have
        — and raise the model's ``carry_in_periods`` to match. Zero carry-in
        then reproduces the stock objective identically (the prepended zero
        rows contribute nothing to the media contribution).

        Raises
        ------
        ValueError
            If neither the model nor the adstock admits carry-in
            (``carry_in_periods == adstock_periods == 0``).
        """
        inner = self._inner
        if inner.carry_in_periods > 0:
            self._carry_in_periods = int(inner.carry_in_periods)
            return
        extra = int(inner.adstock_periods)
        if extra <= 0:
            raise ValueError(
                "CarryInBudgetOptimizer requires carry-in periods > 0; got "
                f"carry_in_periods=0 and adstock_periods={int(inner.adstock_periods)} "
                f"for window ({self._inner_dates()}). A model with no adstock "
                "warm-up cannot be seeded by pre-window spend."
            )
        node = inner.model[inner.channel_data_var]
        old = np.asarray(node.get_value())
        new = np.zeros((extra + old.shape[0],) + tuple(old.shape[1:]), dtype=old.dtype)
        new[extra:] = old
        node.set_value(new)
        inner.carry_in_periods = extra
        self._carry_in_periods = extra

    def _inner_dates(self) -> str:
        dims = self._inner.model.named_vars_to_dims.get(
            self._inner.channel_data_var, ()
        )
        date_dim = self._inner.date_dim
        if date_dim in dims:
            coord = self._inner.model.coords.get(date_dim)
            if coord is not None:
                try:
                    return f"{coord[0]} .. {coord[-1]}"
                except Exception:
                    pass
        return f"carry_in={self._inner.carry_in_periods}, " \
               f"adstock={self._inner.adstock_periods}"

    # ------------------------------------------------------------------
    # Rebuild (one-time, in __init__)
    # ------------------------------------------------------------------

    def _rebuild_shared_carry_in(self) -> None:
        """Replicate ``BudgetOptimizer.model_post_init`` step 7 with a SHARED
        carry-in, then compile the objective+gradient once.

        Mirrors the pinned step 7 field-for-field: ``MediaVariable`` with the
        existing mask/scales/dtype/date dim, spends and levers passed through
        exactly as the original (``[]`` for the 7-channel case), one ``do()``
        over the joint substitutions, then the pinned compile of objective +
        gradient and the constraint recompile against the new flat vector.
        """
        inner = self._inner
        channel_data_var = inner.channel_data_var
        dtype = inner.model[channel_data_var].dtype

        # The shared carry-in the objective reads on every evaluation. Zeros
        # reproduce the stock cold-start Q2 path bit-for-bit (carry-in only
        # becomes non-zero through set_q1_carry_in).
        self._q1_carry_in = pytensor.shared(
            np.zeros((inner.carry_in_periods, self._n_channels), dtype=dtype),
            name="q1_carry_in",
        )

        variables = list(inner._variables.variables)
        if not variables or variables[0].name != channel_data_var:
            raise ValueError(
                "expected the channel-data MediaVariable to be the first "
                f"optimization variable; first variable is "
                f"{variables[0].name if variables else None!r}"
            )
        media_variable: MediaVariable = variables[0]
        # The constructor already validated the carry-in shape against the
        # mask; swap the constant array for the shared variable of the same
        # (carry_in_periods, n_channels) layout.
        media_variable.carry_in_values = self._q1_carry_in

        self._variables = OptimizationVariables(variables)
        inner._variables = self._variables
        inner._budgets_flat = self._variables.flat
        inner._budgets = media_variable.scattered(
            self._variables.variable_slice(channel_data_var)
        )
        inner._pymc_model = do(
            freeze_dims_and_data(inner.model, data=[]),
            self._variables.substitutions(),
        )

        # Compile the objective+gradient ONCE against the shared carry-in,
        # then recompile the constraints against the new flat decision vector
        # (the same Constraint objects as model_post_init step 10).
        inner._compile_objective_and_grad()
        inner.set_constraints(constraints=list(inner._constraints.values()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_q1_carry_in(self, q1_weekly_spend: np.ndarray) -> None:
        """Set the shared Q1→Q2 carry-in to the last ``carry_in_periods`` rows
        of ``q1_weekly_spend``. MUST NOT recompile.

        ``q1_weekly_spend`` holds the full Q1 weekly spend, shape
        ``(>= carry_in_periods, n_channels)`` in raw spend units; only the
        last ``carry_in_periods`` rows seed the Q2 adstock state. The
        compiled ``objective_and_grad`` object identity is unchanged.

        Raises
        ------
        ValueError
            If ``q1_weekly_spend`` is not shape ``(>= carry_in_periods,
            n_channels)``.

        >>> opt.set_q1_carry_in(q1_weekly_spend)  # 13 x n_ch Q1 weeks
        """
        arr = np.asarray(q1_weekly_spend, dtype=float)
        if (
            arr.ndim != 2
            or arr.shape[0] < self.carry_in_periods
            or arr.shape[1] != self._n_channels
        ):
            raise ValueError(
                "q1_weekly_spend must have shape "
                f"(>= {self.carry_in_periods}, {self._n_channels}) "
                f"(carry_in_periods, n_channels); got {arr.shape}"
            )
        carry = arr[-self.carry_in_periods :]
        self._q1_carry_in.set_value(
            carry.astype(self._q1_carry_in.type.dtype), borrow=True
        )

    def set_posterior(self, posterior) -> None:
        """Delegate: point the compiled objective at a new posterior.

        Stage-0 semantics of the pinned optimizer: the first call binds the
        draws through shared variables (one recompile of objective +
        constraints), every later call rebinds in place.
        """
        self._inner.set_posterior(posterior)

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds=None,
        x0=None,
        minimize_kwargs=None,
    ):
        """Delegate to the pinned optimizer (identical result semantics:
        ``BudgetOptimizationResult`` with ``.budgets`` / ``.scipy_result``)."""
        return self._inner.allocate_budget(
            total_budget=total_budget,
            budget_bounds=budget_bounds,
            x0=x0,
            minimize_kwargs=minimize_kwargs,
        )

    @property
    def objective_and_grad(self):
        """The compiled objective+gradient callable. Identity-stable across
        ``set_q1_carry_in`` calls and later posterior rebinds."""
        return self._inner._objective_and_grad

    @property
    def carry_in_periods(self) -> int:
        """Number of leading periods seeded by the Q1 carry-in (6 for Q2)."""
        return self._inner.carry_in_periods