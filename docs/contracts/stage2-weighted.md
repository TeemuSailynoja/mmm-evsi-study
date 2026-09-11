# Stage 2 — Importance-Weighted Q2 Optimization: Interface/API Contract

Status: **contract draft (W1)**. Reviewer (R) critiques; W2 implements the
quality-gate tests **exactly** against this document. W1 later implements the
modules below against the same document. This file covers **quality gates
G2.0–G2.9** (PLAN.md Stage 2).

Scope pinned up front: this stage implements the **resampling-based weighted
optimization only** (PLAN Decision 4a / Decision 5 / Decision 8). The outer BO
loop (Stage 3), the λ·Q1-loss trade-off term and `E_max` (Stage 3), the exact
`shared_weights` tensor and JAX solver (Stage 5) are **out of scope here**;
these parameters are not consumed by any Stage-2 module.

---

## 0. Pinned API facts (trusted, do not re-derive)

Facts marked **[O]** were verified by the orchestrator against the installed
arviz-stats 1.3.2 and are normative. Facts marked **[W1]** were re-confirmed
empirically by W1 while writing this contract (July-of-this-study, pure
numpy / no MCMC) and are equally normative.

0.1 **[O] PSIS accessor is `da.azstats.psislw`, NOT `da.stats.psislw`.**
    PLAN.md (Decision 4) and `docs/methods.md` (PSIS section) say
    `.stats.psislw(dim="sample")` — **outdated**. In the installed
    arviz 1.3.0 / arviz-stats 1.3.2 the registered DataArray accessor is
    `azstats`; `.stats` is **not registered** (AttributeError, verified by
    W1). **Correction: the pinned call is
    `(-ell).azstats.psislw(dim="sample")`.** `docs/methods.md` and PLAN.md
    will be corrected when the corresponding docs task runs.

0.2 **[O] Sign convention (G2.1b).** Input to `psislw` is the **negated**
    joint quarter log-likelihoods, `-ell_s` with
    `ell_s = log p(y* | θ_s, a)` (joint over the whole 13-week simulated
    quarter, one value per posterior draw). Returns a tuple
    `(smoothed, khat)`.

0.3 **[W1] Exact return semantics (verified on a 500-draw example, heavy
    right tail).** For `neg_lw = -ell` (DataArray, dims `("sample",)`) and
    `out = neg_lw.azstats.psislw(dim="sample")`:
    - `out[0]` is a DataArray with the same dims/shape as the input and
      `float(np.exp(out[0]).sum()) == 1.0` (atol `1e-6`): it is the
      **normalized smoothed positive log-weights** (large = more weight),
      already in the `ell` space (`np.corrcoef(out[0], ell) ≈ +0.8` on the
      dominant-draw example). Per-draw **correction**
      `correction_s := ell_s - out[0]_s`, i.e. `out[0] == ell - correction`
      — this is the identity PLAN.md refers to.
    - `out[1]` (`khat`) is a 0-d DataArray; `float(out[1])` is the GPD shape
      fitted to the **right tail of `ell`** (the largest log-likelihoods,
      i.e. the most influential / dominant draws). Empirically a single
      dominant draw gives `khat(-ell) = 1.39 > 0.7` while `khat(+ell) = 0.69
      < 0.7` — consistent with the orchestrator's finding. **Resampling uses
      `p_s = np.exp(out[0])` directly** (sums to 1; dominant draw received
      probability 0.41 in the verification run).
    - The wrapper in §2.2 pins this convention in one place; tests may not
      call `psislw` directly except to pin the convention (G2.1b).

0.4 **[O] Chains are pooled before `psislw`.** Stack `chain` × `draw` into a
    single `sample` dim: one weight per posterior draw and **one k-hat per
    simulated quarter** (G2.2). Pool order pinned: chain-major, i.e.
    `posterior.stack(sample=("chain", "draw"))` (index `s = chain*n_draws +
    draw`).

0.5 **[O] k-hat > 0.7 policy (G2.3).** Skip that simulated quarter's
    outcome (do not solve; log it). Later remedy (Stage 5+):
    `arviz_stats.loo_moment_match` (module-level function — **not** a
    DataArray accessor in 1.3.2, verified by W1) or refit. Non-finite k-hat
    is treated as a skip too (§5).

0.6 **[O] Weight target (API-independent).** `ell_s = log p(y* | θ_s, a)`
    joint over the 13-week quarter;
    `log w_s = ell_s - logsumexp(ell)`. `q(y*)` IS the posterior predictive,
    so **no outer proposal correction** — weights are pure posterior
    reweighting.

0.7 **[O] Resampling.** Multinomial draw of posterior indices
    `∝ exp(smoothed log-weights)` → new (resampled) idata →
    `BudgetOptimizer.set_posterior` (first call recompiles, later calls
    rebind — see §0.9 and the Stage-0 contract §2.3).

0.8 **[O] Artifacts.** A 4×8,000 fit is complete at `data/fit/`
    (`case_study_mmm.zarr`, `case_study_idata.zarr`, `fit_summary.json`;
    see the Stage-1 contract). Real-pipeline gates G2.4–G2.9 gate on these
    artifacts with the **same SKIP_MSG pattern as Stage 1**
    (§6).

0.9 **[O] Pinned Stage-0/1 optimizer facts reused unchanged.** Channel-level
    budgets only (`_budget_dims == ["channel"]`, `_budget_shape == (7,)`);
    `allocate_budget(total_budget, budget_bounds=None, x0=None,
    minimize_kwargs=None, return_if_fail=False, callback=False)` with dict
    bounds keyed by **raw `mdsp_*` names**; `scipy_result.fun` is the
    minimized objective (`objective = -expected sales`), `.nit` iteration
    count, `.x` flat decision vector; `x0` accepts a labelled DataArray or a
    flat ndarray; SLSQP default `ftol=1e-9` — every Stage-2 solve MUST pass
    `minimize_kwargs={"options": {"ftol": 1e-6}}` (Stage-1 §2.4 fix,
    normative); `set_posterior` first-call recompiles / later-calls rebind
    (`opt._objective_and_grad` object identity); warm/cold-start `Q2` Q2
    window emits the benign cold-start `UserWarning` (tolerate in tests).

0.10 **[O] DataTree access.** Bracket access only: `idata["posterior"]`,
     `idata["sample_stats"]`, never `idata.posterior`.

0.11 **[O] `psislw` tail-fit requires ≥ 5 tail draws.** `_get_ps_tails`
     picks `floor(3*sqrt(n))` tail draws for `n*r_eff > 225`, else
     `floor(n/5)`. The floor-of-5 applies **only in the `tail=="both"`
     branch**; `psislw` fits the `tail=="right"` tail, where fewer than ~5
     tail draws **raises** (no floored fallback). Stage-2 wrapper therefore
     REQUIRES ≥ 25 pooled draws (asserted; `config.MIN_POOLED_DRAWS = 25`).
     `r_eff=1` (pooled, effectively independent draws) is pinned.

---

## 1. Scope & artifact layout

### 1.1 What Stage 2 builds

- `src/mmm_evsi/importance.py` — joint-quarter log-likelihoods, chain
  pooling, the single `psislw` wrapper (convention pinned in §0.3), the
  k-hat policy, multinomial resampling, raw-normalized weights, PSIS-ESS.
- `src/mmm_evsi/experiments.py` — simulate Q1 outcomes `y*` under an
  allocation (posterior predictive) + allocation→weekly-spend helper.
- `src/mmm_evsi/optimize_slsqp.py` — carry-in-aware expected Q2 response,
  hot-startable SLSQP Q2 solve, process pool over independent solves,
  allocation-level evaluation loop.
- `toy/toy_gaussian.py` (+ `toy/__init__.py`) — analytic conjugate model
  with exact closed forms (§3); drives the `importance.py` machinery.
- `src/mmm_evsi/config.py` — additive constants only (§1.3).

No new artifacts are **written** by Stage 2 modules: all outputs are
in-memory objects + log records. The Stage-2 smoke summary (G2.9) is
recorded in `docs/LOG.md` by the orchestrator, not by the pipeline.

### 1.2 File partition (parallel W1/W2, disjoint)

- **W1 writes**: `src/mmm_evsi/importance.py`, `src/mmm_evsi/experiments.py`,
  `src/mmm_evsi/optimize_slsqp.py`, `toy/__init__.py`,
  `toy/toy_gaussian.py`; additive edits to `src/mmm_evsi/config.py`.
- **W2 writes**: `tests/test_toy_evsi.py`, `tests/test_psis.py`,
  `tests/test_khat_policy.py`, `tests/test_adstock_carry.py`,
  `tests/test_weighted_solve.py`, `tests/test_hot_start.py`,
  `tests/test_parallel.py`, `tests/test_resampling_noise.py`,
  `tests/test_stage2_smoke.py`.
- No shared files this stage (the `config.py` additions below are tiny;
  the orchestrator lands them before the parallel phase, as in Stage 1).
  `toy/` is currently an empty directory; W1's `toy/__init__.py` makes it an
  importable package. Tests import `toy.toy_gaussian` via the repo root on
  `sys.path` (pytest `rootdir` convention; `pythonpath=["src"]` already set
  in pyproject, and `toy/` is importable because pytest adds the rootdir to
  `sys.path` for rootless-package imports — if the pinned pytest version
  needs it, add `toy` to `[tool.pytest.ini_options].pythonpath`).

### 1.3 `config.py` additions (additive only; nothing existing changes)

```python
# Stage 2 — importance sampling parameters
K_HAT_THRESHOLD = 0.7      # k-hat > threshold => skip the simulated quarter (G2.3)
N_SIM_OUTCOMES = 5         # default simulated quarters per allocation (G2.9 smoke);
                           # gates/sweeps pass larger values explicitly
RESAMPLE_SEED = 0          # default seeded RNG for the multinomial resample
MIN_POOLED_DRAWS = 25      # psislw tail-fit floor (n_draws_tail >= 5); asserted
```

### 1.4 Stage-2 module imports

All real-pipeline functions take the raw weekly DataFrame `df`
(`load_case_study_data()` result) so they can build Q1/Q2 design matrices
(dates, `hldy_*` controls, train-tail spend) that the fitted model's
posterior alone cannot supply (Stage-1 §2.3 pinned: the posterior tree has
no Q1/Q2 spend). The toy never touches `df`/`mmm` (analytic).

---

## 2. Module APIs (exact signatures & shapes)

### 2.1 `src/mmm_evsi/experiments.py`

```python
def allocation_to_weekly_spend(
    allocation: xr.DataArray, n_weeks: int = 13
) -> np.ndarray
```

- `allocation`: DataArray, dims `("channel",)`, coords = `config.CHANNEL_COLUMNS`
  (raw `mdsp_*` names), monetary units (sum = `B_Q1`).
- Returns `np.ndarray` shape `(n_weeks, len(CHANNEL_COLUMNS))`: constant
  weekly rate per channel, `allocation[c] / n_weeks`, columns in
  `config.CHANNEL_COLUMNS` order. Raises `ValueError` if the channel coords
  do not equal `config.CHANNEL_COLUMNS` exactly or if any value < 0.

```python
def simulate_quarter(
    mmm: MMM, idata: xr.DataTree, df: pd.DataFrame,
    window: tuple[str, str], allocation: xr.DataArray,
    seed: int | None = None,
) -> np.ndarray
```

- Pooled posterior predictive draw of one Q1 quarter: draw one index
  `s` uniformly from the pooled posterior (`rng = np.random.default_rng(seed)`),
  take that draw's parameters, and simulate
  `y*_t ~ Normal(mu_t(θ_s, a), y_sigma_s)` for the 13 weeks of `window`
  (independently). The `mu_t(θ_s, a)` computation is the **same response
  path** used by `quarter_log_likelihood` (§2.2) — one definition, one code
  path, no duplicated adstock/saturation math.
- Returns `np.ndarray` shape `(13,)` (weekly sales). Deterministic given
  `(seed, posterior, allocation)`.

### 2.2 `src/mmm_evsi/importance.py`

```python
class WeightError(ValueError):
    """Raised when a weight computation violates a pinned invariant."""
```

```python
def pool_posterior(posterior: xr.Dataset) -> xr.Dataset
```

- Stacks `("chain", "draw")` → `("sample",)` via
  `posterior.stack(sample=("chain", "draw"))`; returns the stacked Dataset.
  All down-stream pooled arrays (ℓ, weights) are aligned to this order
  (chain-major, §0.4). Raises `ValueError` if `sample` already exists.

```python
def normalized_log_weights(ell: np.ndarray) -> np.ndarray
```

- `ell - logsumexp(ell)` over axis -1; `ell` shape `(S,)`.
- This is the **G2.1 manual-computation target** for the raw weights
  (§0.6). Pure numpy `logsumexp` (stable).

```python
@dataclass(frozen=True)
class PsisResult:
    smoothed_log_weights: np.ndarray   # (S,) exp-sum == 1 (atol 1e-6);
                                       # degenerate case: raw normalized weights
    raw_log_weights: np.ndarray        # (S,) = normalized_log_weights(ell)
    khat: float                        # one value per simulated quarter (G2.2)
    weight_ess: float                  # 1 / sum(exp(smoothed)^2); degenerate -> S
    degenerate: bool                   # True when the tail was exactly flat
```

```python
def psis_weights(
    ell: np.ndarray | xr.DataArray, r_eff: float = 1.0,
    min_draws: int = config.MIN_POOLED_DRAWS,
) -> PsisResult
```

- `ell` accepted as (a) `np.ndarray` shape `(S,)` (already pooled) or
  (b) `xr.DataArray` with dims `("chain", "draw")` — pooled chain-major
  internally.
- Asserts `S >= min_draws` (else `WeightError`).
- **Pinned call (single place; G2.1b):**

  ```python
  neg_lw = xr.DataArray(-ell, dims="sample")
  smoothed, khat = neg_lw.azstats.psislw(dim="sample")   # NOT .stats.*
  ```

  `smoothed` → `PsisResult.smoothed_log_weights` (`np.asarray`),
  `khat = float(khat)`. `raw_log_weights = normalized_log_weights(ell)`.
  `weight_ess = 1.0 / np.sum(np.exp(smoothed) ** 2)` (in `[1, S]`).
- **Degenerate-tail interception (§5):** if `np.ptp(ell) == 0.0` (or
  `np.ptp(ell) <= 1e-12 * max(1.0, abs(np.mean(ell)))`), the pinned
  implementation raises `ValueError("All tail values are the same")`; the
  wrapper instead returns `smoothed == raw_log_weights`, `khat = 0.0`,
  `degenerate = True` (uniform weights need no smoothing; G2.4 exercises
  this path).

```python
@dataclass(frozen=True)
class KhatVerdict:
    skipped: bool
    khat: float
    reason: str   # "khat<=0.7: accept" | "khat>0.7: skip simulated quarter"
                  # | "non-finite khat: skip simulated quarter"
```

```python
def apply_khat_policy(
    psis: PsisResult, threshold: float = config.K_HAT_THRESHOLD
) -> KhatVerdict
```

- `skipped = not (0.0 <= khat <= threshold)` — i.e. skip when `khat > 0.7`
  **or** k-hat is non-finite. Accepts degenerate (`khat == 0.0`).
- Logs at `logging.getLogger("mmm_evsi.importance")` WARNING level, message
  includes the reason string and the k-hat value (G2.3 asserts the log).

```python
def resample_posterior(
    posterior: xr.Dataset,
    probabilities: np.ndarray,
    n: int | None = None,
    seed: int = config.RESAMPLE_SEED,
) -> xr.Dataset
```

- `posterior`: the **pooled** posterior (dims `("sample",)` from
  `pool_posterior`). `probabilities`: shape `(S,)`, non-negative,
  `sum == 1` (asserted with atol `1e-6`). `n`: resample size, default `S`.
- `indices = rng.choice(S, size=n, replace=True, p=probabilities)` with
  `rng = np.random.default_rng(seed)`.
- Returns a **new** Dataset `posterior.isel(sample=indices)` renamed to
  dims `("chain": 1, "draw": n)` (i.e. `.expand_dims(chain=[0])` after
  stacking), so it is directly acceptable as a posterior for
  `BudgetOptimizer.set_posterior` (Stage-0 §4: a 1-chain posterior Dataset
  is a pinned-valid input). Attaches `attrs["resampled_indices"] = indices`
  (np.ndarray) for determinism/debugging. `seed` is the ONLY RNG knob of
  this function.

```python
def quarter_log_likelihood(
    mmm: MMM, idata: xr.DataTree, df: pd.DataFrame,
    window: tuple[str, str], allocation: xr.DataArray, y_star: np.ndarray,
) -> np.ndarray
```

- Computes `ell_s = log p(y* | θ_s, a)`, the **joint** log-likelihood of the
  whole simulated quarter (`len(window)` weeks, 13 for Q1), one value per
  posterior draw, pooled chain-major → shape `(S,)` with
  `S = chains * draws`.
- Numerical contract: `ell_s = Σ_t log Normal(y*_t | mu_t(θ_s, a),
  y_sigma_s)` where `mu_t(θ_s, a)` is the fitted model's expected-response
  path (intercept + Fourier seasonality + `hldy_*` controls + per-channel
  `GeometricAdstock(l_max=6)` → `LogisticSaturation` contributions)
  evaluated at the Q1 dates with the Q1 spend = constant weekly rate
  `allocation[c]/13`, and with the per-draw adstock state carried in from
  the **last 6 training weeks** of `df` (Q1 is contiguous with the train
  window — Stage-1 §3.2). `y_sigma_s` is the per-draw noise sd from the
  posterior.
- Implementation directive: reuse the model's own transformation building
  blocks pinned by Stage 0 (`from pymc_marketing.mmm import
  GeometricAdstock, LogisticSaturation` — the `mmm.transformers` import is
  broken) and the posterior variables (`adstock_alpha`, `saturation_*`,
  `intercept_contribution`, `gamma_fourier`, control coefficients,
  `y_sigma`); do **not** re-implement the MMM math by hand.
- **W1 self-check (development time, not a gate):** the same code path
  evaluated on the training window must reproduce
  `pm.compute_log_likelihood(idata, model=mmm.model)` per-draw values
  (rtol `1e-3`) for the model's own observed data. W1 records the result in
  `docs/LOG.md` before running the real pipeline.
- Raises `WeightError` if `y_star.shape != (n_weeks,)` or `S <
  MIN_POOLED_DRAWS`.

### 2.3 `src/mmm_evsi/optimize_slsqp.py`

```python
def q2_expected_response(
    mmm: MMM, posterior: xr.Dataset, df: pd.DataFrame,
    q1_weekly_spend: np.ndarray, budgets: xr.DataArray,
) -> float
```

- Expected Q2 sales (13 weeks) over `posterior` for Q2 channel budgets
  `budgets` (DataArray, dims `("channel",)`, raw `mdsp_*` names), with the
  per-draw adstock state carried across the Q1→Q2 boundary **from the
  proposed Q1 spend**: the adstock state at the end of `q1_weekly_spend`
  (np.ndarray `(13, 7)`, `CHANNEL_COLUMNS` order) feeds the first Q2 week
  (PLAN Decision 8; the train-tail contribution from ≥ 19 weeks prior is
  decayed out and ignored). Same response path as `quarter_log_likelihood`
  (§2.2) — means over all draws in `posterior`.
- This is the single **carry-in-aware response seam**; G2.3b (runs now on
  the 50-draw toy MMM) and the real Q2 solves both consume it.

```python
@dataclass(frozen=True)
class Q2SolveResult:
    outcome_index: int          # simulated-quarter ordinal
    y_star: np.ndarray          # (13,) simulated Q1 sales (for records)
    allocation: xr.DataArray    # the Q1 allocation a ("channel",)
    khat: float                 # this quarter's k-hat
    weight_ess: float           # PSIS-ESS of the weighted posterior used
    budgets: xr.DataArray       # optimal Q2 budgets ("channel",), raw names
    objective_value: float      # expected Q2 sales = -minimized objective
    iteration_count: int        # scipy .nit
    scipy_result: OptimizeResult
    skipped: bool               # True when the k-hat policy skipped the solve

@dataclass(frozen=True)
class WeightedSolveJob:
    allocation: xr.DataArray     # Q1 allocation a ("channel",)
    outcome_index: int
    posterior: xr.Dataset        # resampled posterior (chain=1, draw=M)
    q1_weekly_spend: np.ndarray  # (13, 7)
    y_star: np.ndarray           # (13,)
    khat: float
    weight_ess: float
    x0: xr.DataArray | None      # Q2 warm start ("channel",) or None (uniform)
```

```python
def solve_q2_weighted(
    mmm: MMM, df: pd.DataFrame,
    q2_cfg: QuarterBudget, q1_weekly_spend: np.ndarray,
    x0: xr.DataArray | None = None,
    minimize_kwargs: dict | None = None,
) -> Q2SolveResult
```

- Solves `max_budgets q2_expected_response(...)` subject to the Stage-1
  constraint structure: `Σx = B_Q2` (equality), per-channel boxes
  `q2_cfg.boxes`, `x0` warm start, SLSQP with
  `minimize_kwargs={"options": {"ftol": 1e-6}}` for both uniform-weight and
  weighted solves.
- **Carry-in mechanism obligation (W1; see §7.1):** the solve MUST include
  the Q1→Q2 carry-in. W1's two sanctioned mechanisms, chosen during
  implementation and recorded in `docs/LOG.md`:
  (a) *data-extension*: feed the pinned `BudgetOptimizer` (`set_posterior`
  + `allocate_budget`) a model/data view whose adstock carry-in includes the
  proposed Q1 spend (the pinned optimizer exposes `carry_in_periods` +
  `channel_data` shared values — verified present in the pinned install);
  (b) *thin SLSQP wrapper*: `scipy.optimize.minimize(method="SLSQP")` over
  `q2_expected_response` with the linear equality + box constraints, using
  the same objective/derivative path. Mechanism (b) MUST additionally pass
  G2.4's equivalence gate (uniform weights + zero Q1 spend ≡ stock
  Stage-1 baseline, rtol `1e-4`). Both mechanisms must pass G2.3b, G2.5,
  G2.6, G2.7.
- Post-solve validation (mirrors `solve_baseline`, Stage-1 §2.4): raises
  `RuntimeError` naming the violation unless `success`, `|Σx - B_Q2| <= 1e-6`,
  and every channel inside its box ± `1e-6`.
- `objective_value = -float(scipy_result.fun)`. `iteration_count =
  int(scipy_result.nit)`.

```python
def run_weighted_solves(
    mmm: MMM, df: pd.DataFrame, jobs: Sequence[WeightedSolveJob],
    q2_cfg: QuarterBudget, n_processes: int | None = None,
) -> list[Q2SolveResult]
```

- Independent solves are embarrassingly parallel (PLAN Decision 5): each
  job solves one (allocation × simulated outcome). `n_processes=None` →
  `os.cpu_count()`; `n_processes=1` → in-process serial path.
- Pool: `concurrent.futures.ProcessPoolExecutor`; each **worker loads the
  MMM from `config.MODEL_FILE` once per process** (`load_mmm()`) instead of
  pickling it; jobs are plain picklable data (frozen dataclass + `xr.Dataset`
  + arrays).
- **Determinism contract (G2.7):** solves contain no RNG — identical job
  lists give bit-identical results regardless of pool size or process
  layout. `n_processes=1` vs pool on the same jobs must agree allclose
  (allocations rtol `1e-8` atol `1e-6`; objectives allclose rtol `1e-12`).
- Returns results in job order; `skipped` jobs (already filtered by the
  caller) never enter the pool.

```python
@dataclass(frozen=True)
class AllocationEvaluation:
    allocation: xr.DataArray
    utilities: np.ndarray        # (n_accepted,) OptQ2 per accepted quarter
    khats: np.ndarray            # (n_outcomes,), float("nan") for skipped
    weight_ess: np.ndarray       # (n_outcomes,)
    skipped_indices: np.ndarray  # outcome ordinals skipped by the k-hat policy
    utility: float               # mean over accepted; NaN if all skipped
    n_outcomes: int
    n_skipped: int

def evaluate_allocation(
    mmm: MMM, idata: xr.DataTree, df: pd.DataFrame,
    allocation: xr.DataArray, q1_cfg: QuarterBudget, q2_cfg: QuarterBudget,
    n_outcomes: int = config.N_SIM_OUTCOMES,
    seed: int = config.RESAMPLE_SEED,
    x0_warm: xr.DataArray | None = None,
    n_processes: int | None = None,
    resample_seeds: Sequence[int] | None = None,
) -> AllocationEvaluation
```

- The Stage-2 loop for one allocation: for each outcome `i` in
  `range(n_outcomes)`:
  1. `y_star = simulate_quarter(mmm, idata, df, q1_cfg.window, allocation, seed=seed + i)`
     (deterministic per `i`).
  2. `psis = psis_weights(quarter_log_likelihood(...))` (pooled, §0.4).
  3. `verdict = apply_khat_policy(psis)`; if `skipped`: record
     `(i, nan, psis.weight_ess)`, continue (no solve).
  4. `posterior_r = resample_posterior(pooled_posterior, exp(psis.smoothed_log_weights),
     seed=(resample_seeds[i] if resample_seeds is not None else seed + i))`.
  5. `q1_weekly = allocation_to_weekly_spend(allocation, 13)`.
  6. `res = solve_q2_weighted(mmm, df, q2_cfg, q1_weekly, x0=x0_warm)`
     — pooled via `run_weighted_solves` when `n_processes != 1` (jobs built
     from steps 1–6).
- `resample_seeds` (G2.8): list of ≥ 20 RNG seeds **total**, split into
  two **disjoint halves of ≥ 10 seeds each** for the two compared groups;
  when given, exactly ONE simulated outcome is used (`n_outcomes=1`) and
  the loop iterates over the seeds (k-hat policy/ℓ identical across seeds —
  only the resample RNG changes).
- `utility = float(np.mean(utilities))` over accepted quarters. Raises
  `WeightError` if `n_skipped == n_outcomes` ("all simulated quarters
  skipped by the k-hat policy").
- Determinism: a fixed `(seed, n_outcomes, n_processes)` tuple yields
  identical evaluations across runs (G2.9 asserts this).

### 2.4 `toy/` — analytic conjugate model (see §3 for the math)

```python
# toy/__init__.py  (empty, marks toy/ a package)
# toy/toy_gaussian.py

@dataclass(frozen=True)
class ToySpec:
    tau: float = 2.0          # prior sd
    sigma: float = 3.0        # observation noise at a=0
    t1: int = 13              # quarter length (weeks), == Q1 length
    n_prior_draws: int = 50_000
    seed: int = 0

def noise_sd(spec: ToySpec, a: float) -> float
def prior_draws(spec: ToySpec) -> np.ndarray                    # (S,)
def simulate_quarter(spec: ToySpec, a: float, rng) -> np.ndarray  # (t1,)
def joint_log_likelihood(spec: ToySpec, a: float, y_star: np.ndarray) -> np.ndarray  # (S,)
def posterior_sufficient(spec: ToySpec, a: float, y_star: np.ndarray) -> tuple[float, float]
def opt_q2_analytic(spec: ToySpec, a: float, y_star: np.ndarray) -> float  # (mu*)^2
def value_of_exploration_analytic(spec: ToySpec, a: float, a0: float = 0.0) -> float
def posterior_dataset(spec: ToySpec, draws: np.ndarray) -> xr.Dataset   # (chain=1, draw=S)
def run_toy_evsi(spec: ToySpec, a: float, a0: float = 0.0,
                 n_outcomes: int = 5_000, seed: int = 0,
                 resample_size: int | None = None) -> ToyEvsiResult
```

- `posterior_dataset` wraps the analytic prior draws as the pooled-posterior
  input for `resample_posterior` (dims `("chain": 1, "draw": S)`), so G2.0/
  G2.1 drive the **real** `importance.py` code paths.
- `run_toy_evsi` is the toy "pipeline": per outcome `simulate_quarter` →
  `joint_log_likelihood` → `psis_weights` (importance.py) →
  `resample_posterior` (importance.py, `seed+i`) → weighted Q2 value =
  (resampled θ mean)² → averaged over outcomes.
- **COMMON RANDOM NUMBERS (CRN), mandated (G2.0):** the two utility
  estimates — `utility_mc(a)` and the baseline arm `utility_mc(a0)` — are
  computed on the **same** `θ_s` prior draws and the **same** standardized
  Gaussian noise draws (only the arm-dependent observation sd
  `σ/√(1 + a)` differs), so the MC SE of their difference (the VoE) is
  ~2–3% at N=5,000. The single `seed` drives **both** arms; there is no
  per-arm RNG knob. The G2.0 tolerance is calibrated **for CRN** (§3).

```python
@dataclass(frozen=True)
class ToyEvsiResult:
    a: float
    utility_mc: float            # mean over outcomes of (resampled-mean)^2
    utility_analytic: float
    value_of_exploration_mc: float   # utility_mc(a) mean minus (mean over mc at a0=0)
    value_of_exploration_analytic: float
    khats: np.ndarray            # (n_outcomes,)
    weight_ess: np.ndarray       # (n_outcomes,)
    n_skipped: int
```

---

## 3. Toy specification (exact closed forms — normative for G2.0–G2.3)

Scalar conjugate Gaussian-Gaussian (no MCMC, all analytic; draws `θ_s` are
drawn from the prior):

**Model.** Prior `θ ~ N(0, τ²)`. Q1 likelihood under allocation `a`:
`y*_t | θ, a ~ N(θ, σ²/(1 + a))` for `t = 1..t1` independent. Higher `a` =
more precise observation (the allocation knob controls **information**, which
is what the real Q1 spend buys). Q2 utility
`u2(d, θ) = 2·d·θ − d²` (linear-quadratic; maximizer `d* = θ`, realized
value `θ²`).

**Closed forms** (all exact; `M := t1·(1+a)/σ²`, `v := (1/τ² + M)^{-1}`):

```
ℓ_s(a, y*) = Σ_t log N(y*_t | θ_s, σ²/(1 + a))                 (1)
θ | y*, a ~ N(μ*, v),  μ* = v·M·ȳ*,   ȳ* = (1/t1) Σ_t y*_t      (2)
OptQ2(y*, a) = max_d E[ u2(d, θ) | y*, a ] = (μ*)²              (3)
U(a) = E_{y*|a}[ (μ*)² ] = τ² · z_a / (1 + z_a),  z_a = t1(1+a)·τ²/σ²   (4)
VoE(a) = U(a) − U(0) = τ² · (z_a − z_0) / ((1 + z_a)(1 + z_0))   (5)
```

`(3)` holds because `u2` is concave quadratic in `d` ⇒ `d* = E[θ|y*] = μ*`.
`(4)` uses `E[μ*] = 0` and `Var(μ*) = τ² z/(1+z)` under the prior
predictive (marginal `ȳ* ~ N(0, τ² + 1/M)`, with `μ* = v·M·ȳ*`).

**Pinned constants** (`ToySpec` defaults): `τ = 2`, `σ = 3`, `t1 = 13`,
`S = 50_000` prior draws, `seed 0`. Then `z_a = (52/9)(1+a)` and:

| a | z_a | U(a) = 4·z/(1+z) | VoE(a) = U(a) − 208/61 |
|---|---|---|---|
| 0 (baseline) | 52/9 ≈ 5.7778 | 208/61 ≈ 3.4098 | 0 |
| 1 | 104/9 ≈ 11.5556 | 4·104/113 ≈ 3.6814 | 416/113 − 208/61 ≈ 0.2716 |
| 2 | 52/3 ≈ 17.3333 | 208/55 ≈ 3.7818 | 208/55 − 208/61 ≈ 0.3720 |

(W2 computes from the formulas — the decimals above are sanity checks, not
authoritative.)

**G2.0** — `run_toy_evsi(spec, a=2.0, n_outcomes=5_000)` vs
`value_of_exploration_analytic(2.0)`: assert
`|voE_mc − voE_analytic| <= 0.10 · voE_analytic`. **CRN is mandatory
(§2.4):** both arms (`a` vs `a0`) run on shared `θ_s` prior draws and shared
standardized noise, keeping the MC SE of the VoE difference ≈ 2–3% at
N=5,000 (≈ ≤ ~5 SE against the 0.10 tolerance). The 0.10 tolerance is
**calibrated for CRN** — it does NOT hold if the two arms use independent
draws. Deterministic seeds, so the single fixed-seed run is stable. Sanity
binds chosen so the toy remains a few minutes of CPU (numpy vectorized ℓ
over `(13, S)` per outcome × 5,000; no MCMC anywhere).

**G2.1** — one fixed simulated quarter (seed 0, `a = 2.0`):
(a) `normalized_log_weights(ℓ)` equals the manual formula `ℓ − logsumexp(ℓ)`
exactly (`np.allclose` atol 1e-12) *and* `sum(exp(ℓ − logsumexp(ℓ))) == 1`;
(b) the raw-weights mean of `θ_s` reproduces `μ*` within `3·SE_w` with
`SE_w = sqrt(v̂_w / ESS_w)` (ESS from the weights); (c) the **resampled**
draws (via `resample_posterior`) reproduce `μ*` within `3·sqrt(v/M)` and the
resampled sample variance reproduces `v` within rtol `0.10`; (d) smoothed
weights: `sum(exp(smoothed)) == 1` (atol 1e-6), `khat` finite.

**G2.1b** — convention pin: a degenerate ℓ with one dominant draw
(`ell_degen = np.full(S, -8.0); ell_degen[0] = 0.0`). Assert
`khat(-ell) > 0.7`, `sum(exp(smoothed)) == 1` (atol 1e-6; normalized), and
resampling weights `p ∝ exp(smoothed)` put the dominant draw's weight
**below** its raw normalized weight (the heavy tail is smoothed down). Do
NOT evaluate `khat(+ell)` on this example: its tail draws are flat (+8), so
`psislw` raises `ValueError("All tail values are the same")` (§0.11, §5).

**G2.2** — pooling: `ℓ` for a full quarter is a single `(S,)` vector;
`psis_weights` returns exactly **one** `khat` float per simulated quarter
(not per week) and the smoothed weights live on a single pooled
`("sample",)` dim; the toy's quarter-ℓ joins all 13 weeks (replacing ℓ with
any single week's slice changes the result shape/semantics).

**G2.3** — k-hat policy (runs now; scoped to `apply_khat_policy` +
logging): `apply_khat_policy(psis_weights(ell_degen))` → `skipped=True`,
logged (WARNING contains `"khat"`); a normal quarter → `skipped=False`. The
cross-quarter `n_skipped` bookkeeping is NOT part of this runs-now gate — it
needs the real `mdsp_*` channels and is asserted by the artifact-gated
**G2.3c** below.

**G2.3b** — carry-in (uses the 50-draw toy MMM from `tests/conftest.py` +
`q2_expected_response`): with the posterior and Q2 budgets fixed, changing
the Q1 weekly spend history changes the expected Q2 response:
`|resp(spend_a) − resp(spend_b)| > 1e-6 · max(1, |resp|)` for two feasible Q1
allocations (e.g. planned vs `1.3 × planned` inside the Stage-0 PLANNED
boxes). Runs now, CPU-only.

**G2.3c** — skipped-quarter bookkeeping (artifact-gated; needs the real
`mdsp_*` channels): one `evaluate_allocation` run must count `n_skipped`
consistently across quarters — `n_skipped == len(skipped_indices)`,
`khats[skipped_indices]` are exactly the `nan` entries of `khats`,
`len(utilities) == n_outcomes − n_skipped`, and skipped quarters are never
solved (no Q2 solve emitted for them).

---

## 4. Gate-to-test mapping (G2.0–G2.9)

Rules inherited from the Stage-1 contract §4: tests that run *now* use only
pinned library APIs + `mmm_evsi.config` (exists) + the toy; artifact-gated
tests import `mmm_evsi.*` modules **lazily, inside the artifact-present
branch** (`pytest.importorskip("mmm_evsi.importance")` AND an artifact-missing
`pytest.skip` with the SKIP_MSG) so collection never breaks while modules are
absent. The Stage-0 session-scoped `toy_mmm` fixture is reused by G2.3b.

| Gate | Test file | Behavior / tolerance |
|---|---|---|
| G2.0 | `tests/test_toy_evsi.py` | **Runs now.** Analytic toy end-to-end `VoE`: `|voE_mc − voE_analytic| ≤ 0.10·voE_analytic` (a ∈ {0, 1, 2}, a0=0; N=5,000, seeds fixed). **CRN mandated (§2.4/§3):** both arms share `θ_s` draws + standardized noise; the 0.10 tolerance assumes CRN. |
| G2.1 | `tests/test_psis.py` | **Runs now.** Raw normalized weights == manual (atol 1e-12); weighted/resampled mean & covariance ≈ analytic `(μ*, v)` (3-SE / rtol 0.10); smoothed exp-sum == 1 (atol 1e-6); khat finite. |
| G2.1b | `tests/test_psis.py` | **Runs now.** Pins `(-ell).azstats.psislw(dim="sample")` (with a comment noting PLAN/methods `.stats` is outdated): degenerate dominant draw → `khat(-l) > 0.7`, exp-sum == 1 (atol 1e-6), and `p ∝ exp(smoothed)` puts the dominant draw's weight **below** its raw dominant weight (heavy tail smoothed down). `khat(+ell)` NOT computed — flat +8 tail raises `ValueError("All tail values are the same")`. |
| G2.2 | `tests/test_psis.py` | **Runs now.** One weight per draw, one k-hat per quarter; pooled single `("sample",)` dim before `psislw`. |
| G2.3 | `tests/test_khat_policy.py` | **Runs now.** Scoped to `apply_khat_policy` + logging: forced bad quarter (khat > 0.7, e.g. `ell_degen`) → `skipped=True`, logged (caplog); genuine quarter → `skipped=False`. Cross-quarter `n_skipped` counting moved to G2.3c (artifact-gated). |
| G2.3b | `tests/test_adstock_carry.py` | **Runs now** (toy MMM). `q2_expected_response` depends on the Q1 spend history with posterior/budgets fixed (diff > 1e-6·max(1,|resp|)). |
| G2.3c | `tests/test_khat_policy.py` | **Artifact-gated.** Skipped-quarter bookkeeping on real `mdsp_*` channels: `n_skipped` counted across quarters == `len(skipped_indices)` == number of `nan` `khats`; `len(utilities) == n_outcomes − n_skipped`; skipped quarters never solved. |
| G2.4 | `tests/test_weighted_solve.py` | **Artifact-gated.** Uniform weights (degenerate path, khat 0.0) + zero Q1 spend ⇒ Q2 solve reproduces the Stage-1 baseline Q2 allocation (allclose rtol `1e-4`) and objective (rtol `1e-6`). |
| G2.5 | `tests/test_weighted_solve.py` | **Artifact-gated.** Real-pipeline `set_posterior` rebind: objective value changes with new (resampled) draws; second rebind keeps `opt._objective_and_grad` object identity (Stage-0 G0.2 pattern). |
| G2.6 | `tests/test_hot_start.py` | **Artifact-gated.** Same weighted problem: warm `x0` (prior solve's budgets) converges to the **identical** optimum as cold `x0=None` (hard gate: allclose rtol `1e-4`); `nit(warm) < nit(cold)` asserted — Stage-0 §2.4(iv) found `nit` ordering can be path-dependent, so if it flickers W2 converts the ordering check to an informational log and keeps the optimum-equality gate hard (recorded in LOG.md). Both nits logged. |
| G2.7 | `tests/test_parallel.py` | **Artifact-gated.** Identical weighted-solve job lists: serial (`n_processes=1`) ≡ pool (`n_processes=2`) → allocations allclose (rtol `1e-8`, atol `1e-6`), objectives allclose (rtol `1e-12`). |
| G2.8 | `tests/test_resampling_noise.py` | **Artifact-gated.** `resample_seeds` sweep ≥ 20 seeds **total**, ONE simulated outcome: per-seed utilities recorded/reported (min/max/mean/sd); no-bias signal = split-half stability: the ≥ 20 seeds split into two **disjoint halves of ≥ 10 seeds each** satisfy `|mean_A − mean_B| ≤ 3·sd_pooled/√n_half` with `n_half = min(|A|, |B|) ≥ 10` (the exact-weighted reference is a Stage-5 upgrade; resampling is unbiased by construction, so stability across seeds is the pinned Stage-2 check). Notebook plot produced by the user. |
| G2.9 | `tests/test_stage2_smoke.py` | **Artifact-gated.** STOP checkpoint: `evaluate_allocation` at the baseline Q1 allocation (deterministic `(seed, n_outcomes=config.N_SIM_OUTCOMES=5)`): runs end-to-end; second identical run reproduces the first (allclose atol `1e-9` on `utility`, `khats`, `weight_ess`); `utility` finite, `n_skipped ≥ 0` reported, every accepted solve `success`. Test prints a smoke summary; the orchestrator records it in `docs/LOG.md` (PLAN G2.9). |

`q1_cfg` for all real-pipeline tests = `load_budgets().q1`; `q2_cfg` =
`load_budgets().q2`; allocations for Stage-2 tests = the Stage-1 baseline Q1
budgets (`solve_baseline(...).budgets`) or variants inside the boxes.

---

## 5. Error behavior & edge cases

| Case | Behavior |
|---|---|
| `psislw` on exactly-constant ℓ (uniform weights) | Pinned impl raises `ValueError("All tail values are the same")`. **Wrapper intercepts** (degenerate detection, §2.2) → raw normalized weights, `khat=0.0`, `degenerate=True`. G2.4 relies on this. |
| Pooled draws `S < 25` | `WeightError` from `psis_weights` (tail-fit floor, §0.11). |
| k-hat `> 0.7` (or non-finite) | Quarter skipped + WARNING log (khat value + reason); excluded from `utilities`; `n_skipped` counted. Never solved silently. |
| All outcomes skipped in `evaluate_allocation` | `WeightError("all simulated quarters skipped by the k-hat policy")`. |
| `y_star` wrong length / `allocation` coords ≠ `CHANNEL_COLUMNS` / negative budgets | `ValueError`/`WeightError` with the violating condition named. |
| Q2 solve infeasible / solver failure | `RuntimeError` naming the violated constraint (success | Σx−B ≤ 1e-6 | boxes ±1e-6), mirroring `solve_baseline`; `MinimizeException` from the stock path propagates. |
| Q2 window cold-start `UserWarning` | Expected when a data-extension carry-in path is *not* used (Stage-1 §0); tests tolerate with `filterwarnings`/"pytest.warns". |
| `set_posterior` channel-coord mismatch | `ValueError` propagates unchanged (Stage-0 §4 pinned). |
| Artifact/module absent | Artifact-gated tests: `pytest.skip("Stage 2 modules/artifacts missing — run scripts/fit_case_study.py and check docs/contracts/stage2-weighted.md")` + `pytest.importorskip(...)` (never a failure). |
| Resampled posterior `(chain=1, draw=M)` rejected by `set_posterior` | Documented fallback: pass a `("draw",)`-only Dataset; W1 records the outcome in LOG.md (G2.5 exercises). |
| `weights` sum ≠ 1 in `resample_posterior` | `ValueError` (atol `1e-6`). |

---

## 6. Verification

### 6.1 Toy gates (runs now; CPU; after W1's modules land)

```bash
cd /home/teemu/repos/mmm-evsi-study
.venv/bin/python -m pytest tests/test_toy_evsi.py tests/test_psis.py \
  tests/test_khat_policy.py tests/test_adstock_carry.py -q
```

Expected: **G2.0, G2.1, G2.1b, G2.2, G2.3, G2.3b pass** (toy MMM fit ~5–15 s
once, session-scoped; toy Gaussian is pure numpy, no MCMC). W2 reports
per-gate pass/fail mapped to these gates.

### 6.2 Full suite (user/worker run; artifacts + Stage-2 modules present)

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all Stage-2 gates execute (the 4×8000 fit artifacts already exist at
`data/fit/`), Stage-0/1 regression green. G2.4–G2.9 unskip the moment the
Stage-2 modules exist; before W1 lands them they skip cleanly (§5).

---

## 7. Open questions / recorded decision points

1. **Carry-in mechanism (the one real seam).** Two sanctioned paths (§2.3):
   data-extension of the pinned `BudgetOptimizer` vs a thin SLSQP wrapper
   over `q2_expected_response`. W1 picks during implementation, documents
   the choice and the validation in `docs/LOG.md`, and the arbiter gates are
   **G2.3b** (carry-in is real), **G2.4** (wrapper route must equal the
   stock baseline in the zero-carry-in limit), **G2.5** (rebind still
   verifiable). Not a silent choice — logged, reviewable.
2. **Skipped-quarter exclusion** from the utility mean is a known
   (documented) selection effect until the Stage-5 remedy (moment matching /
   refit); the skip rate is always reported.
3. **λ / E_max / V_Q1 trade-off terms are Stage-3 inputs, not consumed in
   Stage 2** (only `U(a) = E[OptQ2(y*, a)]` and the baseline arm are
   computed here; G2.9 is exactly PLAN's "easier version").
4. **`r_eff=1` assumed** for pooled independent draws; revisit only if the
   sampler summary ever shows autocorrelation that matters.
5. **Toy runtime**: 5,000 outcomes × 50,000 draws × 13 weeks (≈ a few
   minutes numpy). If it proves slow on the machine, W2 may reduce
   `n_outcomes` to 2,000 while keeping the rtol `0.10` gate (document the
   change in LOG.md).
6. **`nit` ordering** in G2.6 is asserted but may be converted to an
   informational log if the pinned SLSQP is path-dependent (Stage-0 §2.4
   precedent); the optimum-equality gate is hard regardless.
7. **Doc corrections owned by the docs task**: PLAN.md Decision 4/Reuse and
   `docs/methods.md` say `.stats.psislw` — the pinned accessor is
   `.azstats.psislw` (§0.1). `docs/methods.md`'s "(smoothed log-weights,
   khat)" framing stands, augmented by §0.3's exact return semantics.
8. **Revision history (R's review → W1, six fixes):** (P1-1) G2.1b no
   longer computes `khat(+ell)` on the degenerate example — kept assertions
   are `khat(-ell) > 0.7`, normalized exp-sum == 1, and the smoothed
   dominant-draw weight below the raw one; (P1-2) CRN is now mandated for
   the two `run_toy_evsi` arms and the G2.0 `0.10·|voE|` tolerance is
   calibrated for CRN (§2.4/§3); (P2-3) G2.8 seed requirement is ≥ 20 total
   split into two disjoint halves of ≥ 10 each (§2.3/§4); (P2-4) §3 table
   a=1 VoE fraction corrected to `416/113 − 208/61`; (P2-5) §0.11
   corrected — floor-of-5 applies only in the `tail=="both"` branch,
   `tail=="right"` raises below ~5 tail draws, `MIN_POOLED_DRAWS = 25`
   unchanged; (P2-6) runs-now G2.3 scoped to `apply_khat_policy` +
   logging, with the `n_skipped` cross-quarter bookkeeping moved to
   artifact-gated G2.3c (§3/§4).