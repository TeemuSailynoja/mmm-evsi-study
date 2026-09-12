# Stage 2b — Shared Q1→Q2 carry-in optimizer (single compile): contract

Goal: extend the pinned `BudgetOptimizer` so the **Q1 carry-in spend is a
shared variable**, giving ONE Q2 compile for the whole study. Per solve we
then only swap (i) draws via `set_posterior` and (ii) carry-in via
`set_value` — no recompilation. Replaces the data-extension plan.

## 0. Pinned facts (verified by the orchestrator against the pinned install —
   do NOT re-derive)

1. `BudgetOptimizer.model_post_init` step 7 bakes the carry-in as **numpy
   constants** at build time:
   ```python
   def carry_in_for(name):
       if not self.carry_in_periods: return None
       return np.asarray(self.model[name].get_value())[: self.carry_in_periods]
   carry_in_values = carry_in_for(self.channel_data_var)
   media_variable = MediaVariable(
       name=self.channel_data_var, mask=self.budgets_to_optimize,
       num_periods=self.num_periods, adstock_periods=self.adstock_periods,
       carry_in_values=carry_in_values,
       channel_scales=self.channel_scales,
       dtype=self.model[self.channel_data_var].dtype, date_dim=self.date_dim,
       budget_distribution_over_period_tensor=self._budget_distribution_over_period_tensor,
       cost_per_unit_tensor=self._cost_per_unit_tensor)
   self._variables = OptimizationVariables([media_variable, *spends, *levers])
   self._budgets_flat = self._variables.flat
   self._budgets = media_variable.scattered(self._variables.variable_slice(self.channel_data_var))
   self._pymc_model = do(...)   # substitutes channel_data := [carry_in | budget | carry-over]
   ```
   (`MediaVariable`, `OptimizationVariables` live in
   `pymc_marketing.mmm.budget_optimizer`; `do` in `pytensor.graph.basic`.)
2. `_compile_objective_and_grad` compiles `function([budgets_flat], [objective, objective_grad])`
   from `self.extract_response_distribution(self.response_variable)` on
   `self._pymc_model`. Draws enter via `SharedPosterior` after the first
   `set_posterior` (Stage-0 pinned: first call recompiles, later calls rebind).
3. **Verified empirically**: mutating the optimization model's `channel_data`
   shared var after build changes NOTHING (delta = 0.0) — the carry-in is
   baked constants, so post-hoc `set_value` on `channel_data` cannot work.
4. Q2 window facts: `carry_in_periods == 6`, `num_periods == 13`,
   optimization `channel_data` shape `(19, 7)` = [6 carry-in | 13 window];
   cold-start zeros for Q2 (training ends 2018-01-28, Q2 starts 2018-05-06).
5. `q2_expected_response(mmm, posterior, df, q1_weekly_spend, budgets)` in
   `mmm_evsi.optimize_slsqp` already computes the carry-in-aware Q2 response
   (validated by G2.3b) — it is the arbiter reference for G-CI-3.
6. Repo conventions: raw `mdsp_*` channel keys; DataTree bracket access +
   `.to_dataset()` where a Dataset is needed; `set_posterior` accepts a
   Dataset (chain=1, draw=M resampled, from `importance.resample_posterior`).

## 1. Module API — `src/mmm_evsi/carry_in_optimizer.py` (W1, new file)

```python
class CarryInBudgetOptimizer:
    def __init__(self, mmm, start_date, end_date):
        """Build the inner optimizer and recompile it with a SHARED carry-in."""
    def set_q1_carry_in(self, q1_weekly_spend: np.ndarray) -> None
    def set_posterior(self, posterior: xr.Dataset) -> None
    def allocate_budget(self, total_budget, budget_bounds=None, x0=None,
                        minimize_kwargs=None) -> BudgetOptimizationResult
    @property
    def objective_and_grad(self) -> object   # the compiled callable
    @property
    def carry_in_periods(self) -> int
```

- `__init__`: `inner = mmm.budget_optimizer(start_date, end_date)`, then
  REBUILD `inner`'s substitution so `carry_in_values` is
  `pytensor.shared(np.zeros((inner.carry_in_periods, n_channels)),
  name="q1_carry_in", dtype=<channel_data dtype>)` instead of constants, and
  `inner._objective_and_grad` is recompiled ONCE. Implementation route:
  replicate `model_post_init` step 7 on the same `inner` fields (reuse
  `MediaVariable`, `OptimizationVariables`, `do`; the existing `spends` /
  `levers` lists are `[]` for this 7-channel case — the wrap must still pass
  them through exactly as the original does). The rebuild may be implemented
  as a subclass or a patching function — W1's choice; the observable API above
  is what is tested.
- `set_q1_carry_in(q1_weekly_spend)`: sets the shared var to
  `q1_weekly_spend[-carry_in_periods:]` (shape (6, n_ch), raw spend units).
  MUST NOT recompile; `objective_and_grad` object identity unchanged.
- `set_posterior`: delegate to the inner optimizer (Stage-0 semantics).
- `allocate_budget`: delegate; identical result semantics to Stage 1/2
  (`BudgetOptimizationResult` with `.budgets`, `.scipy_result`).
- Raises `ValueError` if `q1_weekly_spend` shape != (>=carry_in_periods, n_ch).

## 2. Integration (W1, edit `src/mmm_evsi/optimize_slsqp.py`)

`solve_q2_weighted` (and `_solve_q2_core` when a resampled `posterior` is
given) switches from the stock cold-start path to:

```python
opt = CarryInBudgetOptimizer(mmm, q2_cfg.window[0], q2_cfg.window[1])
opt.set_q1_carry_in(np.asarray(q1_weekly_spend))
if posterior is not None: opt.set_posterior(posterior)
res = opt.allocate_budget(total_budget=q2_cfg.total, budget_bounds=q2_cfg.boxes,
                          x0=x0, minimize_kwargs={...ftol 1e-6...})
```

Zero-carry-in must reproduce the Stage-1 stock baseline exactly (G2.4 stays
green; G-CI-2 makes this explicit). Post-solve validation (success, Σx=B,
boxes) unchanged.

## 3. Gate tests — `tests/test_carry_in_optimizer.py` (W2, new file only)

Artifact-gated like Stage 2 (skip via getattr(config,"MODEL_FILE"/"IDATA_FILE",None)
+ the Stage-2 SKIP_MSG pattern; imports of `mmm_evsi.carry_in_optimizer`
guarded with `pytest.importorskip`).

| Gate | Behavior / tolerance |
|---|---|
| G-CI-1 no-recompile | Build wrapper; grab `fn = wrapper.objective_and_grad`; call `set_q1_carry_in(zeros)` and `set_q1_carry_in(q1_baseline)`; assert `wrapper.objective_and_grad is fn` after BOTH; evaluate `fn(x0)` under zeros vs q1_baseline carry-in and assert the values differ (`abs(Δ) > 1e-6 * max(1,|f|)`). |
| G-CI-2 zero-carry-in ≡ stock | `set_q1_carry_in(zeros)` + `allocate_budget(total=B_Q2, bounds=boxes)` reproduces `solve_baseline(mmm, Q2, q2_cfg)`: budgets allclose rtol 1e-4, objective isclose rtol 1e-6. |
| G-CI-3 carry-in-aware objective | Let `q1 = solve_baseline(mmm, Q1, q1_cfg).budgets`; `q1_weekly = tile(q1/13)`; `res = wrapper.allocate_budget(...)` with that carry-in; then `ref = q2_expected_response(mmm, idata-posterior-as-Dataset, df, q1_weekly, res.budgets)`; assert `res.objective_value ≈ ref` (isclose rtol 1e-3, atol 1e-3 · |ref| — both are the same model response path, so tight). |
| G-CI-4 set_posterior rebind | With the same carry-in: `r1 = resample_posterior(pooled, probs, seed=0)`, `r2 = ... seed=1` (2000 draws via `config.RESAMPLE_DRAWS`); `set_posterior(r1)`, solve, record obj1; `set_posterior(r2)` — assert `objective_and_grad` identity unchanged — solve, record obj2; assert `not isclose(obj1, obj2, rtol=1e-6)`. |

All tests run with `PYTHONPATH=src` in the venv; each solve ~1–2 min; keep the
suite to these 4 tests. No new dependencies.

## 4. Error behavior

- `set_q1_carry_in` with wrong shape → `ValueError` naming the expected shape.
- Wrapper construction on a model without carry-in periods → `ValueError`
  (carry_in_periods must be > 0; Q2 window guarantees 6).
- Everything else delegates to the pinned optimizer's errors unchanged.

## 5. Verification (orchestrator, after both land)

```
cd /home/teemu/repos/mmm-evsi-study
.venv/bin/python -m pytest tests/test_carry_in_optimizer.py -q            # G-CI-1..4
.venv/bin/python -m pytest tests/test_weighted_solve.py tests/test_hot_start.py \
  tests/test_parallel.py tests/test_resampling_noise.py tests/test_stage2_smoke.py -q  # Stage-2 regression (carry-in now wired)
```

Stage-2 regression must stay green (G2.4 zero-carry-in equivalence, G2.6/2.7
consistency, G2.9 smoke reproducibility).

## 6. File partition (parallel phase)

- W1 writes: `src/mmm_evsi/carry_in_optimizer.py` (new) + edits
  `src/mmm_evsi/optimize_slsqp.py` only.
- W2 writes: `tests/test_carry_in_optimizer.py` only.
- No shared-file edits. Orchestrator owns docs/LOG.md.
