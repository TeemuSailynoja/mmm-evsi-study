# Stage 0 Test Suite & Optimizer Capability Probe — Interface/API Contract

Status: **agreed contract** (W1). Reviewer (R) critiques; W2 implements the
tests **exactly** against this document. This file covers the **quality gates
G0.1, G0.2 and G0.4** only.

- G0.1: `SharedPosterior` and `BudgetOptimizer.set_posterior` import from the
  PR-pinned install → `tests/test_env.py`.
- G0.2: toy test that `set_posterior` rebinds different draws into the
  compiled objective **without** a new compile → `tests/test_set_posterior.py`.
- G0.4: optimizer capability probe on a toy 7-channel MMM → `tests/test_optimizer_capability.py`.
- G0.3 (README / docs / LOG.md) is a **separate docs review** and is out of
  scope here.

All facts below were verified against the pinned install (pymc-marketing
1.1.0, PR #3002 head 5743b310) and are normative.

---

## 1. Pinned API facts this contract relies on (do not re-derive)

- `pymc_marketing.mmm.transformers` is **broken** (ImportError) in the pinned
  install. Import transformers from the package top level:
  `from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation`.
- `MMM.fit(X, y, ...)` requires `X` **without** the target column. Always call

  ```python
  mmm.fit(df.drop(columns=["y"]), y=df["y"], draws=50, tune=50, chains=1,
          nuts_sampler="nutpie", progressbar=False, random_seed=0)
  ```

  (`nuts_sampler="nutpie"` is mandatory in the gates.)
- ArviZ 1.0 DataTree: always bracket-access groups, e.g. `idata["posterior"]`,
  **never** `idata.posterior`.
- For our 7-channel weekly model the optimizer has exactly one budget dim:
  `opt._budget_dims == ["channel"]`, `opt._budget_shape == (7,)`.
  `opt.budgets_to_optimize` is a boolean `xr.DataArray` over `dims=("channel",)`.
  With no explicit mask, auto-detection yields **all 7 cells True** on the toy
  fit (non-zero mean `channel_contribution`).
- `opt = mmm.budget_optimizer(start_date, end_date)` with
  `start_date="2020-08-09"`, `end_date="2020-11-01"` produces a 13-week window:
  `opt.num_periods == 13`. The `date` dim is **not** a decision variable.
- `BudgetOptimizer.allocate_budget(total_budget, budget_bounds=None, x0=None,
  minimize_kwargs=None, return_if_fail=False, callback=False)`:
  - `budget_bounds` may be `dict {channel: (low, high)}` (valid because there is
    exactly one budget dim) or an `xr.DataArray` with dims `("channel", "bound")`.
  - `x0` may be a labelled `xr.DataArray` over `("channel",)`, a flat
    `np.ndarray` sized to the decision vector, or `None` (uniform spread).
  - Returns `BudgetOptimizationResult` with `.budgets` (DataArray over
    `("channel",)`), `.scipy_result` (`OptimizeResult`; objective value =
    `scipy_result.fun`, `.nit` = iteration count, `.x` = flat decision vector),
    `.spend_var_allocations`, `.callback_info`.
  - `minimize_kwargs` defaults to SLSQP (`ftol=1e-9`, `maxiter=1000`); user keys
    override defaults after merging.
  - Un-optimized (mask `False`) cells are **fixed at 0** — the decision vector
    only contains the `True` cells.
- `BudgetOptimizer.set_posterior(idata)`:
  - The **first** call moves the draws into shared variables and **recompiles**
    the objective (one-time cost). Every **later** call swaps draws in place
    with `set_value` only — `opt._objective_and_grad` is then the **same object**
    before and after the call.
  - Accepts a posterior `xr.Dataset` (it is wrapped into a DataTree internally)
    or a DataTree / InferenceData with a `posterior` group.
  - Raises `ValueError` when the new posterior's channel coordinates differ from
    the construction posterior (verified: "budgets_to_optimize has coordinates
    the model does not have …"). The optimizer state is unchanged in that case.
- `BudgetOptimizer` and `BudgetOptimizationResult` import from
  `pymc_marketing.mmm.budget_optimizer`; `SharedPosterior` and
  `extract_response_distribution` import from `pymc_marketing.pytensor_utils`.

---

## 2. Files & module API

### 2.1 `tests/conftest.py`

Session-scoped fixture `toy_mmm`. W2 **rewrites** the current file (its
transformers import and `fit(df, ...)` call are broken per Section 1).

```python
@pytest.fixture(scope="session")
def toy_mmm():
```

- **7 channels** named `"ch1"` … `"ch7"` (list order = ch1…ch7).
- Weekly synthetic data: `n = 120` rows, `dates = pd.date_range("2020-01-05",
  periods=120, freq="7D")` (Sundays; first 2020-01-05, last 2022-04-17).
  Deterministic RNG: `np.random.default_rng(0)`.
- Columns: `date`, one spend column per channel (`rng.uniform(10.0, 100.0, n)`),
  and target `y = 5.0 + (spend · coef) + noise` (coef `uniform(0.1, 1.0, 7)`).
- Model: `MMM(date_column="date", channel_columns=channels,
  target_column="y", adstock=GeometricAdstock(l_max=6),
  saturation=LogisticSaturation(), yearly_seasonality=5)`.
- **Small, fast, deterministic fit** (exact call, per Section 1):

  ```python
  mmm.fit(df.drop(columns=["y"]), y=df["y"], draws=50, tune=50, chains=1,
          nuts_sampler="nutpie", progressbar=False, random_seed=0)
  ```

  (draws ≤ 50, tune ≤ 50, chains = 1, `random_seed=0` fixed.)
- Docstring states: 7-channel toy for the **Option A** capability probe (all 7
  channels optimizable; weekly adjustment deferred). The fit takes ~5–15 s.
- Return the fitted `mmm` object (tests call `mmm.budget_optimizer(...)`).

### 2.2 `tests/test_env.py` — G0.1 (already conforming; keep)

Three tests, no fixture needed (pure import/signature checks):

1. `test_shared_posterior_importable`
   `from pymc_marketing.pytensor_utils import SharedPosterior`; assert
   `SharedPosterior is not None` **and** `isinstance(SharedPosterior, type)`.
2. `test_budget_optimizer_set_posterior`
   `from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer`; assert
   `hasattr(BudgetOptimizer, "set_posterior")` and `callable(...)`.
3. `test_extract_accepts_shared_posterior_kwarg`
   `inspect.signature(extract_response_distribution)` from
   `pymc_marketing.pytensor_utils`; assert `"shared_posterior" in
   sig.parameters`.

### 2.3 `tests/test_set_posterior.py` — G0.2

**One** test: `test_set_posterior_rebinds_without_recompile(toy_mmm)`.

Helper (module-private, rewritten from the current broken version):

```python
def _perturb_idata(idata, factor=1.05):
    """Posterior Dataset with every float var scaled by ``factor``.

    dims/coords are preserved via DataArray algebra. Do NOT mutate through a
    DataTree node with tuple assignment (``tree_node[name] = (dims, vals)``) —
    that drops dims/coords and makes set_posterior raise. set_posterior accepts
    a Dataset directly.
    """
    posterior = idata["posterior"].to_dataset().copy()   # bracket access only
    for name in list(posterior.data_vars):
        da = posterior[name]
        if da.dtype.kind == "f":
            posterior[name] = da * factor
    return posterior
```

Test body (exact sequencing — the first call is a one-time bind that
recompiles; the no-recompile property is asserted across the **second** call):

1. `opt = toy_mmm.budget_optimizer("2020-08-09", "2020-11-01")`.
2. `fn0 = opt._objective_and_grad`; assert `fn0 is not None`.
3. **First** `opt.set_posterior(_perturb_idata(opt.idata, 1.05))` — binds draws
   into shared variables (implementation recompiles here).
4. `bound_fn = opt._objective_and_grad`; assert `bound_fn is not None`.
5. `res1 = opt.allocate_budget(total_budget=1000.0, budget_bounds={c: (0.0, 1000.0)
   for c in ["ch1","ch2","ch3","ch4","ch5","ch6","ch7"]})`; assert
   `res1.scipy_result.success`; `obj1 = float(res1.scipy_result.fun)`.
6. **Second** `opt.set_posterior(_perturb_idata(opt.idata, 0.90))` — rebind in
   place. **Assert `opt._objective_and_grad is bound_fn`** (same compiled
   function object, i.e. no recompile).
7. `res2 = opt.allocate_budget(total_budget=1000.0, budget_bounds=<same>)`;
   assert `res2.scipy_result.success`; `obj2 = float(res2.scipy_result.fun)`.
8. Assert `not np.isclose(obj1, obj2, rtol=1e-6, atol=1e-9)` — different draws
   change the objective value at the solution.

Rationale pinned in a comment: the implementation docstring states the first
call costs one recompile, later calls are `set_value` only; asserting identity
after a single call would be wrong and fail.

### 2.4 `tests/test_optimizer_capability.py` — G0.4 (new file)

Module constants:

```python
CHANNELS = [f"ch{i}" for i in range(1, 8)]
START_DATE = "2020-08-09"
END_DATE = "2020-11-01"            # inclusive; 13 weekly Sundays, num_periods=13
TOTAL_BUDGET = 1000.0
PLANNED = {"ch1": 100.0, "ch2": 200.0, "ch3": 150.0, "ch4": 120.0,
           "ch5": 180.0, "ch6": 140.0, "ch7": 110.0}      # sum == 1000.0
ALL_TRUE_MASK  # xr.DataArray True(7), dims=("channel",), coords={"channel": CHANNELS}
```

Four tests, all built on `toy_mmm` via
`mmm.budget_optimizer(START_DATE, END_DATE, ...)`:

**(i) `test_seven_channel_level_budgets`** — 7 independent channel-level
decision variables, constant weekly rate over the 13-week window:

```python
opt = toy_mmm.budget_optimizer(START_DATE, END_DATE)
assert opt._budget_dims == ["channel"]
assert opt._budget_shape == (7,)
assert opt.budgets_to_optimize.dims == ("channel",)
assert bool(opt.budgets_to_optimize.values.all())            # all 7 optimized
assert opt.num_periods == 13
res = opt.allocate_budget(total_budget=TOTAL_BUDGET,
                          budget_bounds={c: (0.0, TOTAL_BUDGET) for c in CHANNELS})
assert res.scipy_result.success
assert res.budgets.dims == ("channel",) and res.budgets.size == 7
assert abs(float(res.budgets.sum()) - TOTAL_BUDGET) <= 1e-6  # wired sum constraint
```

**(ii) `test_channel_level_percent_bounds`** — ±30% box per channel:

```python
opt = toy_mmm.budget_optimizer(START_DATE, END_DATE)
lo = {c: 0.7 * PLANNED[c] for c in CHANNELS}
hi = {c: 1.3 * PLANNED[c] for c in CHANNELS}
res = opt.allocate_budget(total_budget=TOTAL_BUDGET,
                          budget_bounds={c: (lo[c], hi[c]) for c in CHANNELS})
assert res.scipy_result.success
x = res.budgets.sel(channel=CHANNELS)
assert float(x.min()) >= min(lo.values()) - 1e-6            # low bound respected
assert float(x.max()) <= max(hi.values()) + 1e-6            # high bound respected
for c in CHANNELS:                                          # per-channel box
    assert x.sel(channel=c).item() >= lo[c] - 1e-6
    assert x.sel(channel=c).item() <= hi[c] + 1e-6
assert abs(float(res.budgets.sum()) - TOTAL_BUDGET) <= 1e-6
```

(Measured: solution lies strictly inside every box, max violation ≈ −3e-12.)
The `xr.DataArray` form of the bounds (dims `("channel", "bound")`, coords
`bound=["low","high"]`) is a valid alternative; the dict form is the one
exercised here.

**(iii) `test_mask_fixes_excluded_channel`** — `budgets_to_optimize` fixes one
channel at 0 while the others optimize:

```python
mask = xr.DataArray(np.array([True, True, True, True, True, True, False]),
                    dims=("channel",), coords={"channel": CHANNELS})
opt = toy_mmm.budget_optimizer(START_DATE, END_DATE, budgets_to_optimize=mask)
res = opt.allocate_budget(total_budget=TOTAL_BUDGET,
                          budget_bounds={c: (0.0, TOTAL_BUDGET) for c in CHANNELS})
assert res.scipy_result.success
assert int(opt.budgets_to_optimize.values.sum()) == 6
assert res.scipy_result.x.size == 6                         # decision vector = True cells
assert np.isclose(float(res.budgets.sel(channel="ch7").item()), 0.0, atol=1e-9)  # fixed at 0
others = res.budgets.drop_sel(channel="ch7")
assert bool(np.all(np.isfinite(others.values))) and float(others.min()) >= 0.0
assert abs(float(others.sum()) - TOTAL_BUDGET) <= 1e-6      # optimized cells take the total
```

**(iv) `test_hot_start_x0_accepted`** — `x0` reaches SLSQP; both labelled and
flat forms are accepted and return valid results:

```python
opt = toy_mmm.budget_optimizer(START_DATE, END_DATE)
mk = {"method": "SLSQP", "options": {"maxiter": 1000}}
# labelled DataArray x0 (e.g. a prior solution scaled 0.95)
cold = opt.allocate_budget(total_budget=TOTAL_BUDGET,
                           budget_bounds={c: (0.0, TOTAL_BUDGET) for c in CHANNELS},
                           minimize_kwargs=mk)
x0_da = (cold.budgets * 0.95)
hot = opt.allocate_budget(total_budget=TOTAL_BUDGET,
                          budget_bounds={c: (0.0, TOTAL_BUDGET) for c in CHANNELS},
                          x0=x0_da, minimize_kwargs=mk)
assert hot.scipy_result.success
assert bool(np.all(np.isfinite(hot.budgets.values)))
assert abs(float(hot.budgets.sum()) - TOTAL_BUDGET) <= 1e-6
assert isinstance(hot.scipy_result.nit, int) and hot.scipy_result.nit >= 1
# flat ndarray x0 sized to the decision vector is also accepted
x0_flat = np.full(7, TOTAL_BUDGET / 7.0)
hot2 = opt.allocate_budget(total_budget=TOTAL_BUDGET,
                           budget_bounds={c: (0.0, TOTAL_BUDGET) for c in CHANNELS},
                           x0=x0_flat, minimize_kwargs=mk)
assert hot2.scipy_result.success
assert bool(np.all(np.isfinite(hot2.budgets.values)))
# informational only — do NOT gate on nit ordering (can be flaky path-dependent)
print(f"(iv) nit cold={cold.scipy_result.nit} hot={hot.scipy_result.nit}"
      f" hot2={hot2.scipy_result.nit}")
```

Optional (W2's discretion): if `nit(hot) < nit(cold)` holds stably in the
pinned solver, add an informational assertion; if it ever flickers, log the
comparison instead. The hard gate is: x0 accepted (both forms), solve
succeeds, sums to budget, finite budgets.

---

## 3. Data shapes & conventions

| Item | Value |
|---|---|
| Channel names | `ch1` … `ch7` (7 channels) |
| Toy data | 120 weekly Sundays, 2020-01-05 → 2022-04-17, `date` column + 7 spend cols + `y` |
| Fit | draws=50, tune=50, chains=1, `nuts_sampler="nutpie"`, `random_seed=0` |
| Optimization window | `START_DATE="2020-08-09"`, `END_DATE="2020-11-01"` → **13 weeks**, `num_periods=13`; window lies inside the training span (fine for the probe) |
| Budget dims | `_budget_dims == ["channel"]`, `_budget_shape == (7,)`; `date` is not a decision dim |
| `budgets_to_optimize` | boolean DataArray, `dims=("channel",)`; auto-detected all-True for the toy fit; mask `False` ⇒ fixed at 0 |
| `budgets` result | DataArray `dims=("channel",)`, size 7, monetary units, sum = total_budget (|Σx − B| ≤ 1e-6) |
| Decision vector | `scipy_result.x` size = number of True cells (7 unmasked / 6 with one excluded) |
| Posterior access | **bracket access only**: `idata["posterior"]`, `out["posterior"]`, `idata["posterior"].to_dataset()` |
| Determinism | `np.random.default_rng(0)` for data; `random_seed=0` for fit — same draws every session |

---

## 4. Error behavior & edge cases

The following are **documented and pinned**; coverage is required only where
marked.

- **First `set_posterior` recompiles; later calls do not** (core G0.2
  semantics; see 2.3). Required coverage: identity is asserted across the
  second call; the objective value must change on a perturbed posterior.
- **`set_posterior` with channel-label mismatch raises `ValueError`** and
  leaves the optimizer unchanged (verified message: "budgets_to_optimize has
  coordinates the model does not have …"). **Optional coverage** — W2 may add
  `pytest.raises(ValueError)` in a test that renames the `channel` coords of a
  posterior Dataset copy.
- **Auto-detected mask change on `set_posterior` raises `ValueError`** (per the
  implementation docstring). Not exercised by the gates (scaling preserves the
  mask); do **not** add coverage.
- **`budget_bounds` dict is invalid for multi-dim budgets** — `allocate_budget`
  raises `ValueError` ("Dict approach to budget_bounds is not supported for
  multi-dimensional budgets"). **Note, do not require coverage**: the pinned
  version rejects `(date, channel)` masks at optimizer construction (pydantic
  `ValidationError`: "('channel',) must be a permuted list of ('date',
  'channel')"), so a multi-budget-dim optimizer cannot be constructed through
  `MMM.budget_optimizer` today. Weekly/per-period adjustment is deferred anyway
  (Decision A).
- `budget_bounds=None` emits `UserWarning` ("No budget bounds provided. Using
  default bounds (0, total_budget)…"). Tests that do not test bounds pass
  explicit bounds to keep CI output clean.
- nutpie emits "Only 50 samples per chain. Reliable r-hat and ESS diagnostics
  require longer chains…" — benign; ignore (optional `filterwarnings`).
- `allocate_budget` raises `MinimizeException` on solver failure (default
  `return_if_fail=False`). Gates assert `scipy_result.success`, so this path is
  not triggered.

---

## 5. Verification (how W2 runs the gates)

From the repository root:

```bash
cd /home/teemu/repos/mmm-evsi-study
.venv/bin/python -m pytest tests/test_env.py tests/test_set_posterior.py tests/test_optimizer_capability.py -x -q
```

Expected result: **all tests pass** (3 in `test_env.py`, 1 in
`test_set_posterior.py`, 4 in `test_optimizer_capability.py`; total 8). The
session-scoped `toy_mmm` fit (~5–15 s with nutpie) runs once. W2 reports
per-gate pass/fail mapped to G0.1 / G0.2 / G0.4. G0.3 is reviewed separately.