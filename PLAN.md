# MMM-EVSI Study: Importance-Weighted Budget Optimization

## Context

**Nested structure** (clarified with user):

A model fitted on the pymc-marketing **MMM case study** data (Stan MMM weekly
sales dataset: 7 media channels + holiday controls, `GeometricAdstock(l_max=6)`,
`LogisticSaturation`, `yearly_seasonality=5`). We fit on **the full time period
except the last two quarters** (>3 years of training data, so Q1 observations
shift the posterior only modestly); the held-out last two quarters are our
**Q1 (observation/experimentation)** and **Q2 (decision)** planning quarters.

1. **Baseline optimization (Q1)**: optimize Q1 channel budgets subject to
   total-budget + per-channel box constraints → baseline allocation + value.
2. **Exploration (outer problem)**: high-dimensional **Bayesian optimization**
   over *perturbations* of the baseline Q1 allocation. Every proposed
   allocation must lie in the feasible set (same total-budget + box
   constraints) — infeasible suggestions waste compute, so feasibility is
   built into the proposal mechanism (reparameterization/projection), not
   checked after the fact.
3. **Proposal utility (expensive, nested)**:
   a. Simulate observed Q1 outcomes y* under the proposed allocation a
      (posterior predictive; q(y*) **is** the posterior predictive, so no
      outer proposal correction is needed — see Decision 4).
   b. Importance-weight the *existing* posterior draws with the **joint**
      likelihood of the whole simulated quarter — no refitting.
   c. **Inner optimization (Q2)**: budget-allocate Q2 under the
      importance-weighted posterior, same constraint structure. With
      geometric adstock, the Q2 response depends **directly on the proposed
      Q1 spend history**, so the inner objective is **OptQ2(y\*, a)** — a
      function of both the simulated outcome and the allocation itself
      (adstock state carries across the Q1→Q2 boundary).
   d. Utility = optimized Q2 value (decision after learning).
4. **Baseline value of exploration** (utility of the *unmodified* baseline Q1
   allocation): same three-step procedure but with the baseline allocation —
   simulate Q1 outcomes under it, importance-weight the posterior, run the Q2
   optimization, average over simulated outcomes. **Both** arms learn from Q1;
   the only difference is which Q1 allocation produced the information.
5. **Trade-off term + experimentation budget (user requirement)**: deviation
   from the optimal Q1 allocation costs revenue *during* Q1. Planning-time
   expected Q1 revenue V_Q1(a) (current posterior, no reweighting — known
   before running) enters the utility:
   **U(a) = E_{y*|a}[ OptQ2(y*, a) ] − λ · ( V_Q1(baseline) − V_Q1(a) )**,
   with **λ = 1** initially ($1 lost in Q1 must be compensated by $1 expected
   Q2 gain). Additionally, an **experimentation budget E_max caps the Q1
   loss**: proposals with λ·(V_Q1(baseline) − V_Q1(a)) > E_max are
   infeasible. λ and E_max become tunable, reported sensitivity parameters.
6. **Value of exploration** = max_a U(a) − U(baseline) (where the baseline's
   Q1 loss term is 0 by construction). An EVPI-style quantity with an
   intermediate decision, evaluated via importance sampling (no refitting).
   Because OptQ2(y*, a) depends on a through **both** learning (weights) and
   **physical adstock carryover**, we report the benefit **two ways**:
   (i) *total benefit* (carryover included) and (ii) *isolated information
   value* (adstock history fixed at baseline, only the weights vary) — so
   the reported number is interpretable. Effects extending beyond Q2 are
   truncated at the Q2 horizon; the discarded tail is estimated and reported
   as a sensitivity note.

**Hot starts (user suggestion)**: the initial cold run (baseline Q1 solve +
Q2 solve under the unweighted posterior) provides warm-start points for all
subsequent weighted solves. Q1 observations should shift the optimum only
modestly, so starting the inner solves near the previous solution should cut
iterations sharply. SLSQP accepts x0; the JAX solver likewise.

Why this is non-trivial:
- Stock pymc-marketing 1.1.0 `extract_response_distribution` bakes draws into
  the graph as constants → recompile per posterior. PR #3002 (open, unmerged)
  adds `SharedPosterior` + `BudgetOptimizer.set_posterior()` to rebind draws
  without recompiling — but it has **no slot for importance weights**.
- User wants a **highly parallelizable** optimization routine; stock
  `BudgetOptimizer` uses serial scipy SLSQP.

## Decisions (agreed with user)

1. **Data / model**: reproduce the **pymc-marketing MMM case study**
   (https://www.pymc-marketing.io/en/0.15.1/notebooks/mmm/mmm_case_study.html)
   in a **separate fitting script** (`scripts/fit_case_study.py`) with the
   case-study settings, modified to: fit on all data **except the last two
   quarters**, and draw **at least 2× the case-study posterior size**
   (4 chains × 8,000 draws) for high ESS in both bulk and tails — robust
   importance sampling. Save the fitted model + idata for loading.
2. **Allocation shape & budget bookkeeping**: the case study has 7 channels;
   **Decision (Option A): optimize channel-level quarterly budgets over the
   FULL 7 channels** — 7 decision variables (one per channel), a constant
   weekly rate per channel across the 13-week quarter (matches the case
   study and the pinned optimizer's `budget_dims=['channel']`). **Weekly
   (per-period) budget adjustment is deferred to a later phase.** No fixed
   channels, so the total-budget equality is **Σx = B** over all 7 channels
   (6 free dimensions). Outer BO dimensionality = **7**. BO flavor: **GP**
   (GPyTorch on GPU) or trust-region / particle swarm acceptable; start with
   random/Latin hypercube sweeps for validation, then the surrogate.
3. **pymc-marketing**: **install from PR #3002** (pin the PR head SHA),
   upgrade to the merged release later. This gives `SharedPosterior` +
   `BudgetOptimizer.set_posterior()` for no-recompile draw swapping.
4. **Weights mechanism (revised)**: **start with (a) importance-weighted
   resampling** — resample draws ∝ weights → new idata → `set_posterior` →
   solve with the **stock `BudgetOptimizer`**, no custom graph work.
   **Weight target, defined independently of any API convention**: for
   existing draws θ_s ~ p(θ|D) and a simulated quarter y* under allocation a,
   **ℓ_s = log p(y*|θ_s, a)** is the **joint log-likelihood of the entire
   simulated quarter** (one weight per posterior draw, one diagnostic per
   simulated quarter — *not* per week), and
   **log w_s = ℓ_s − logsumexp(ℓ)**.
   Since y* is drawn from the posterior predictive itself, q(y*) equals the
   mixture of p(y*|θ_s) over draws and **no outer proposal correction is
   needed** — the weights above are pure posterior reweighting; this is
   documented separately from any future q(y*)-correction.
   Smoothing uses **PSIS** via `arviz_stats`'s `psislw` (accessor
   `.stats.psislw(dim="sample")`) with **chains pooled** (stack chain+draw
   into one `sample` dim) so weights and **k-hat** are joint. **The input
   sign convention of `psislw` must be verified against the pinned
   implementation in a test** (do not assume; keep log-likelihoods,
   log-weights, and negated inputs clearly distinct in code and docs).
   **k-hat policy**: if k-hat > 0.7 for a simulated quarter, **skip that
   posterior predictive draw initially** (log it); later upgrade the remedy
   to **moment matching** (`loo_moment_match`) or **refitting**.
   **(b) exact shared-weights tensor** deferred to the final upgrade stage
   (Stage 5), only if resampling noise/cost proves limiting. Resampling
   noise should **not bias** the utility, but it **increases its
   uncertainty** — quantified via a resample-seed sweep.
5. **Inner optimizer**: stock **SLSQP via `BudgetOptimizer`** first (ground
   truth, hot-startable via `minimize_kwargs={'x0': ...}`). SLSQP is serial
   *within* a solve but the workload is embarrassingly parallel *across*
   solves → process pool (independent solves per (proposal × simulated
   outcome)). JAX/`jaxopt` vmap-on-GPU stage later for scale, cross-validated
   against SLSQP.
6. **Constraints (concrete)**: in each optimization round, **total budget =
   the currently planned budget for that quarter** (sum of planned spend in
   the dataset); the 7 channel budgets satisfy **Σx = B**. Per-channel **box
   = ±30% of the planned spend** for that channel/quarter (planned $1,000 →
   [700, 1,300]), enforced as channel-level SLSQP bounds. BO proposals are
   generated inside this feasible set (reparameterization / projection),
   never infeasible. The experimentation budget E_max (item 7) is an
   additional feasibility gate on proposals. Soft penalties only in the
   later JAX stage (tunable λ + sweep). **Weekly granularity: deferred.**
7. **Utility & economic units**: U(a) = E_{y*|a}[OptQ2(y*, a)] −
   λ·(V_Q1(baseline) − V_Q1(a)), λ = 1 initially. **All quantities are in
   expected sales units** (the MMM's target); sales, revenue, and profit are
   *not* used interchangeably — if profit matters later, an explicit
   margin/conversion step is added and documented. **E_max caps the
   λ-weighted Q1 loss** (identical to the actual loss while λ = 1; if λ is
   changed, the cap semantics are revisited explicitly).
8. **Adstock carryover**: the inner objective is OptQ2(y*, a) — adstock
   state carries across the train→Q1 and Q1→Q2 boundaries, so a proposed Q1
   spend history feeds the Q2 response directly. Reported benefit is split
   into *total benefit* (carryover included) and *isolated information
   value* (adstock fixed at baseline, only weights vary); beyond-Q2 effects
   are truncated at the Q2 horizon with the tail estimated and reported.
9. **Hot starts**: initial cold run (Q1 + Q2, unweighted) seeds warm starts
   for all weighted solves (Q1 observations should shift optima only
   modestly); record iteration counts to confirm the speedup.
10. **Optional research thread — exploratory (global) optimizers**: at some
   point assess plausibility of particle-based / swarm-style optimization
   (e.g. particle swarm) for the budget allocation. Motivation: the loss is
   fast to evaluate with analytic gradients, so plain gradient methods work
   well *within* a basin; but constraints plus the non-convexity of the
   expected-response landscape may create multiple attractive basins, where
   multi-region exploration could win. Method: benchmark on the *same*
   (draws, weights) landscape — multi-start gradient (SLSQP/JAX) vs.
   particle/swarm global search — compare best-found optima and sensitivity
   to start points. Decision: adopt whichever is more reliable, or hybrid
   (particles locate basins, gradients polish). Non-blocking; runs after the
   main pipeline is validated.

## Proposed Approach

### Stage 0: Documentation, skeleton & environment
- Install **pymc-marketing from PR #3002** first (pin the PR head SHA via uv;
  upgrade to the merged release later) — Stage 1 already uses it. Verify
  `SharedPosterior` + `BudgetOptimizer.set_posterior` import and rebind
  without recompiling (toy test).
- **Verify channel-level optimization capability of the pinned optimizer** (a
  capability probe, not just rebinding): prove it supports (i) 7
  independently adjustable **channel-level** budgets (one decision var per
  channel, constant weekly rate over 13 weeks), (ii) **channel-level ±30%
  bounds**, (iii) the `budgets_to_optimize` mask can fix/exclude a channel
  (general capability), and (iv) the hot-start API
  (`minimize_kwargs={'x0': ...}` reaches SLSQP). If any capability is
  missing, adapt or wrap before Stage 1.
- `README.md` — project purpose, EVSI definition, pipeline diagram.
- `docs/` — method notes (importance sampling for EVSI, **PSIS + k-hat
  diagnostics and the k-hat > 0.7 policy**: skip draw → moment matching →
  refit) and `docs/LOG.md` — running log of problems found, attempts,
  resolutions (the "no drift" reference).
- Marimo notebook(s) per stage.

### Stage 1: Fit the case-study MMM + baselines
- `scripts/fit_case_study.py`: reproduce the case study (7 channels, holiday
  controls, GeometricAdstock(l_max=6), LogisticSaturation,
  yearly_seasonality=5, target_accept=0.9) with **2× draws** (4×8,000),
  fitting on all data **except the last two quarters**; save model + idata
  (+ sampler summary: ESS, R-hat, divergences) to disk.
- `load_mmm.py` loader + sanity checks (dims, contribution var, date range).
- Define **explicit, disjoint date windows** (train / Q1 / Q2) as concrete
  start–end dates stored in one config module — calendar quarters vs.
  13-week blocks are decided here, not via date arithmetic at check time.
  Extract **planned budgets** per channel from the data → total budget B
  (all 7 channels) and ±30% channel boxes.
- Stock `BudgetOptimizer`: baseline Q1 solve (B, boxes) and Q2 solve
  under the unweighted posterior → V_Q1(baseline), V_Q2(unweighted),
  hot-start seeds.

### Stage 2: Resampling-based weighted optimization (first working pipeline)
- **Analytic end-to-end toy validation FIRST** (`toy/`): a conjugate toy
  model (e.g. Gaussian likelihood / Gaussian posterior, linear-quadratic Q2
  utility with closed-form OptQ2) runs the *entire* loop — simulate y* →
  weights → PSIS → resample → optimize → aggregate — and is compared against
  the analytic value of exploration. This must pass **before** the expensive
  MMM pipeline runs (cheap, CPU-only).
- `importance.py` (reusable): simulate Q1 outcomes y* under an allocation
  (posterior predictive), compute the **joint** log-likelihood ℓ_s of the
  whole simulated quarter per draw, log w_s = ℓ_s − logsumexp(ℓ), pool
  chains, **PSIS** via `.stats.psislw` (input sign convention verified
  against the pinned implementation) → smoothed log weights + **one joint
  k-hat per simulated quarter**; apply the k-hat > 0.7 policy (skip draw,
  log); multinomial resample draws ∝ smoothed weights → resampled idata.
- `optimize_slsqp.py`: per (allocation × simulated outcome): `set_posterior`
  (no recompile) → SLSQP Q2 solve for **OptQ2(y*, a)** (adstock state from
  the proposed Q1 spend carried into Q2) with total budget B and ±30%
  channel boxes, **hot-started** from the previous/cold optimum via
  `minimize_kwargs`; independent solves run in a **process pool**.
- Baseline arm: same procedure at a = baseline allocation → U(baseline).
- Resampling-noise experiment: repeat with several resample seeds; noise
  should **not bias** utility but **inflate its uncertainty** — compare seed
  spread vs. (later) exact-weighted mean; report both.
- **⏸ STOP — Stage 2 checkpoint (smoke test)**: with everything above we can
  already evaluate *the utility gained by the Q2 optimization after observing
  Q1 under the baseline allocation* — the easier version of the full
  workflow. All Stage 2 gates must be green and the smoke-test result
  recorded in LOG.md before proceeding.

### Stage 3: Bayesian-optimization outer loop over Q1 perturbations
- **7-dim** (7 channel budgets) proposals, feasible by construction:
  total = B, per channel within ±30% of planned spend, and **Q1 revenue loss
  ≤ E_max** (experimentation budget) — reparameterization / projection,
  never infeasible queries.
- Start: random / Latin-hypercube sweep (validation, parallel batch via
  process pool). Then: **GP surrogate on GPU (GPyTorch)**; if the GP
  misbehaves at 7 dims, fall back to trust-region or particle swarm.
- Surrogate over the full utility U(a); batch candidate evaluation in
  parallel.
- **Winner evaluation (anti winner's-curse)**: the BO-selected best proposal
  is re-evaluated with **fresh simulations**, **paired** with baseline-arm
  simulations (common random numbers), so the reported exploration benefit
  is not inflated by selecting the maximum of noisy utility estimates.
- Track: best proposal, expected value of exploration, posterior over the
  utility landscape.

### Stage 4: Aggregation & value of exploration
- Per-proposal U(a) = E_{y*}[OptQ2(y*, a)] (weighted) − λ·(V_Q1(baseline) −
  V_Q1(a)); U(baseline) has a zero Q1-loss term.
  Value of exploration = best U(a) − U(baseline), computed on the **fresh
  paired simulations** of the winner (Stage 3), not on the BO search
  estimates.
- **Explicit uncertainty procedure** (replaces vague "ESS-adjusted
  intervals"): decompose and report variance contributions from
  (i) outcome-simulation MC error (number of y* draws),
  (ii) posterior-approximation error (PSIS ESS, k-hat),
  (iii) resampling error (seed sweep spread),
  (iv) optimization error (solver gap/tolerance).
- Report benefit **two ways**: total benefit (adstock carryover included)
  and isolated information value (carryover fixed at baseline); plus the
  beyond-Q2 tail estimate.
- Sensitivity: number of draws, proposal q, λ (revenue trade-off weight),
  E_max, box-penalty λ, BO budget, hot-start vs. cold-start solve cost.

### Stage 5: Exactness & scale upgrades (only if earlier stages show the need)
- **Shared-weights tensor** (`weighted_extract.py`): extend the extraction
  with a `shared_weights` node in the utility graph (exact weighted mean,
  no resampling noise); compile once, swap draws + weights per solve.
  Cross-validate against Stage 2 resampled allocations (agreement within
  resampling noise; bias check).
- **JAX solver** (`optimize_jax.py`): same objective in JAX (or graph
  translation), batch (proposals × candidate budgets), vmap on GPU;
  total budget by parameterization (`total × simplex`), box constraints
  soft (log-barrier/hinge, tunable λ). Solver: `jaxopt` (L-BFGS/Adam).
  Cross-validate against SLSQP on identical (draws, weights).

## Files to create (planned)

- `README.md` (rewrite)
- `docs/` (method notes, `LOG.md`)
- `scripts/fit_case_study.py` — fit + save the case-study MMM (2× draws,
  fit window = all but last two quarters)
- `src/mmm_evsi/` (reusable modules):
  - `config.py` — explicit date windows (train/Q1/Q2), channel selection,
    budget/box/E_max/λ parameters
  - `load_mmm.py` — loader + sanity checks
  - `importance.py` — joint-quarter log-likelihood weights, PSIS (verified
    sign convention, pooled chains, per-quarter k-hat policy), resampling,
    ESS diagnostics — reusable across stages
  - `experiments.py` — simulate Q1 outcomes under an allocation
  - `weighted_extract.py` — shared-weights exact utility (Stage 5 upgrade)
  - `optimize_slsqp.py` — hot-startable per-solve SLSQP path + process pool
  - `optimize_jax.py` — vmapped JAX solver (Stage 5)
  - `bo_design.py` — Bayesian-optimization outer loop over feasible
      7-dim perturbations (±30% channel boxes, Σx = B, E_max gate) + fresh
      paired winner evaluation
  - `evsi.py` — value-of-exploration aggregation + uncertainty decomposition
- `toy/` — analytic conjugate end-to-end validation (Stage 2, runs first)
- `tests/` — pytest suite implementing the quality gates below
- `notebooks/` — marimo notebooks per stage

## Reuse (found)

- `pymc_marketing.mmm.budget_optimizer.BudgetOptimizer` — model handling,
  budget dims/mask, scipy SLSQP loop, result xarray builders.
- `pymc_marketing.pytensor_utils.extract_response_distribution` — graph
  extraction logic (clone/replace/vectorize) to adapt for shared draws+weights.
- PR #3002's `SharedPosterior` (github.com/pymc-labs/pymc-marketing/pull/3002)
  — installed via PR pin; `set_posterior` rebinds resampled draws without
  recompiling.
- `pymc_marketing.mmm.utility` (`average_response` etc.) — utility functions.
- `arviz_stats` PSIS: DataArray accessor `.stats.psislw(dim=...)` (input =
  **negated** log-weights; returns smoothed log-weights + k-hat),
  `.stats.pareto_khat`, and `loo_moment_match` for the k-hat > 0.7 remedy
  path. Verified present in the installed arviz-stats 1.3.2.
- Case-study data: https://raw.githubusercontent.com/sibylhe/mmm_stan/main/data.csv
  (weekly sales, 7 media channels, holiday controls) + the notebook's
  model/sampler settings.

## Execution workflow & quality gates

**Agent orchestration workflow** (enforced by the **main orchestrator agent**
for every stage; documented in README, tracked in `docs/LOG.md`).

Agent roles (mapped to the pi-subagents builtin definitions; note the
builtin `reviewer` is **read-only** — no bash/write — so anything that must
write or execute tests is a `worker` instance):

- **W1 — contract & implementation worker** (`worker`, fresh context per
  task, **model: `deepseek-v4-flash [opencode-go]`**): plans the interface / API
  contract, later implements the solution.
- **R — contract & code reviewer** (`reviewer`, read-only, **model:
  `deepseek-v4-pro [opencode-go]`**): critiques the contract before
  implementation; after implementation reviews code quality and
  documentation completeness.
- **W2 — gate worker** (`worker`, *separate* instance, fresh context,
  **model: `deepseek-v4-flash [opencode-go]`**): implements the quality-gate
  tests against the agreed contract, runs them, and reports per-gate
  pass/fail. Independence from W1 keeps the gates honest.
- **O — oracle** (`oracle`, **model: `deepseek-v4-pro [opencode-go]`**):
  arbitrates if W1 and R/W2 cannot agree on the contract; its ruling is
  logged.

Per-task protocol:

1. Orchestrator picks the next task item from the stage checklist below.
2. **W1** plans the **interface / API contract** for the task: module
   signatures, data shapes, file artifacts, error behavior — written down
   (e.g. `docs/contracts/<task>.md`) before any implementation.
3. **R** reviews the contract; **W2** then **implements the task's quality
   gates** (pytest tests / checks) against the agreed contract. If the
   contract has problems, W1, R and W2 iterate until they reach a
   **compromise that works for all** (escalate to **O** if stuck).
4. **Only after the gates are implemented** does **W1** implement its
   solution.
5. **W2** runs the gates and hands back per-gate pass/fail results;
   **R** reviews the code and ensures the **documentation is updated**
   accordingly (README / docs / notebook).
6. The orchestrator **logs every final decision and its relevant reasoning**
   in `docs/LOG.md`, then advances to the next task/stage. A stage is only
   complete when all its gates are green.

**Execution constraints:**
- **No parallel sub-agents** — everything runs on a single GPU; tasks are
  executed sequentially.
- **GPU-dependent tests are run by the user.** Nothing that depends on such
  a test's result may advance until the user hands back the result. While
  waiting, the orchestrator may schedule **unrelated** tasks.
- Gates are yes/no questions, each backed by a named test or check
  (implemented by W2 in step 3, run by W2 in step 5).

### Stage 0 gates
- [ ] G0.1 `SharedPosterior` and `BudgetOptimizer.set_posterior` import from
      the PR-pinned install (`test_env.py`).
- [ ] G0.2 Toy test: `set_posterior` with different draws changes the
      compiled objective's value **without** a new compile (compile counter
      / function identity unchanged) (`test_set_posterior.py`).
- [ ] G0.3 README, docs/ method notes (incl. PSIS k-hat policy), and
      docs/LOG.md exist and describe the pipeline (sub-agent review).
- [ ] G0.4 Optimizer capability probe (toy model): the pinned
      `BudgetOptimizer` supports (i) 7 independently adjustable
      **channel-level** budgets, (ii) **channel-level ±30% bounds**, (iii)
      the `budgets_to_optimize` mask can fix/exclude a channel (general
      capability), (iv) `minimize_kwargs={'x0': ...}` actually reaching SLSQP
      (hot start accepted / changes iteration count)
      (`test_optimizer_capability.py`).

### Stage 1 gates
- [ ] G1.1 Fit script runs end-to-end and saves model + idata; files reload
      (`test_fit_artifacts.py`).
- [ ] G1.2 Date windows: train / Q1 / Q2 windows are explicit stored date
      ranges, mutually **disjoint**, in order, and together cover the
      dataset; the fit uses exactly the train window (`test_fit_window.py`).
- [ ] G1.3 Posterior size ≥ 2× case study: chains × draws ≥ 4 × 8,000
      (assert on idata dims).
- [ ] G1.4 Sampler health: min bulk & tail ESS ≥ 2,000 (pooled), max R-hat
      < 1.01, zero divergences (`test_sampler_health.py`).
- [ ] G1.5 Planned budgets extracted: B = sum of planned spend per quarter
      (all 7 channels); boxes = ±30% of planned per channel
      (`test_budgets.py`).
- [ ] G1.6 Baseline Q1/Q2 solves feasible: |Σx − B| ≤ 1e-6 over the 7
      channels, all x within channel boxes; and optimal ≥ random feasible
      points' objective (optimality smoke) (`test_baseline_solves.py`).

### Stage 2 gates
- [ ] G2.0 **Analytic toy end-to-end**: the conjugate toy pipeline's value
      of exploration matches the closed-form analytic result within MC
      tolerance; runs before any MMM-stage compute (`test_toy_evsi.py`).
- [ ] G2.1 Weight correctness on an **analytic conjugate case** (e.g.
      Gaussian-Gaussian): reweighted/resampled draws reproduce the analytic
      updated posterior mean/cov within tolerance. **Raw normalized weights
      tested separately**: log w_s = ℓ_s − logsumexp(ℓ) matches manual
      computation exactly; PSIS-smoothed weights are *not* required to equal
      unsmoothed weights exactly (monotone-ish, finite k-hat)
      (`test_psis.py`).
- [ ] G2.1b `psislw` input sign convention verified against the **pinned**
      implementation (documented in code + docs; test pins the convention)
      (`test_psis.py`).
- [ ] G2.2 Joint quarter likelihood: one weight per posterior draw, **one
      k-hat per simulated quarter** (not per week); chains pooled before
      PSIS (single `sample` dim) (`test_psis.py`).
- [ ] G2.3 k-hat policy: a forced bad quarter (k-hat > 0.7) is skipped and
      logged, not silently used (`test_khat_policy.py`).
- [ ] G2.3b Adstock dependence: with y* fixed, changing the Q1 allocation a
      changes the Q2 objective (OptQ2(y*, a) genuinely depends on a);
      adstock state crosses the Q1→Q2 boundary (`test_adstock_carry.py`).
- [ ] G2.4 Uniform weights reproduce the unweighted baseline allocation
      (allclose, rtol=1e-4) (`test_weighted_solve.py`).
- [ ] G2.5 `set_posterior` rebind in the real pipeline: objective changes
      with new draws, compile count unchanged (`test_weighted_solve.py`).
- [ ] G2.6 Hot start: iterations(x0=warm) < iterations(cold) on the same
      weighted problem; both logged (`test_hot_start.py`).
- [ ] G2.7 Process pool ≡ serial: identical seeds → identical allocations &
      utilities (allclose) (`test_parallel.py`).
- [ ] G2.8 Resampling-noise sweep (≥20 seeds, one simulated outcome):
      utility spread reported; mean stable across seeds (no bias signal)
      (`test_resampling_noise.py`, notebook plot).
- [ ] **G2.9 STOP checkpoint**: baseline-arm utility (Q2 gain after observing
      Q1 under baseline allocation) computed end-to-end; smoke-test summary
      recorded in LOG.md; all Stage 2 gates green.

### Stage 3 gates
- [ ] G3.1 Proposal feasibility property test: N=1,000 generated candidates
      all satisfy |Σx − B| ≤ 1e-6, ±30% channel boxes, Q1 loss ≤ E_max
      (`test_bo_feasibility.py`).
- [ ] G3.2 LHS sweep runs in parallel; wall-clock and per-eval timings logged
      (notebook).
- [ ] G3.3 GP surrogate fits sweep data: held-out predictive check passes
      (e.g. rank correlation > 0.5 or calibrated intervals)
      (`test_gp_surrogate.py`) — *GPU test: user-run.*
- [ ] G3.4 BO best ≥ LHS best at equal evaluation budget (monotone
      improvement check, logged).
- [ ] G3.5 Winner re-evaluated on **fresh, paired** simulations (common
      random numbers with the baseline arm); reported benefit uses these,
      not the BO search estimates (`test_winner_eval.py`).

### Stage 4 gates
- [ ] G4.1 Single-proposal (baseline only) → value of exploration = 0
      (allclose to 0 within MC tolerance) (`test_aggregation.py`).
- [ ] G4.2 Value of exploration ≥ −MC error (sanity bound, logged).
- [ ] G4.3 Uncertainty decomposition reported: separate variance
      contributions from outcome simulation, posterior approximation
      (PSIS ESS/k-hat), resampling (seed sweep), and optimization error
      (solver gap) (`test_uncertainty.py`, notebook).
- [ ] G4.4 Benefit reported both ways: total (carryover included) and
      isolated information value; beyond-Q2 tail estimate included
      (notebook check).
- [ ] G4.5 Sensitivity outputs (λ, E_max, draws, BO budget) exist as
      tables/plots (notebook check).

### Stage 5 gates (conditional)
- [ ] G5.1 Exact shared-weights utility = manual weighted mean of per-draw
      responses (allclose, toy) (`test_weighted_extract.py`).
- [ ] G5.2 Exact vs. resampled allocations agree within resampling noise;
      bias check across seeds (`test_weighted_extract.py`).
- [ ] G5.3 JAX vs. SLSQP allocations agree (rtol) on identical inputs;
      batched speedup logged (`test_optimize_jax.py`) — *GPU test:
      user-run; dependent work blocked until results return.*
