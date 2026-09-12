# Decision log

Running log of problems found, attempts, resolutions, and final decisions.
This is the "no drift" reference: every material decision gets an entry.

## Legend

- **[DECIDED]** — final decision + reasoning.
- **[PROBLEM]** — issue found; links to resolution.
- **[RESOLVED]** — resolution of a problem.
- **[GATE]** — quality-gate result (pass/fail).

---

## 2026-09-11 — Stage 0 kickoff

- **[DECIDED]** Environment: Python 3.13, `pymc-marketing` pinned to PR #3002
  head `5743b31038f2dd9396aa08d98168afdbdb5d86ff`
  (`PabloRoque/pymc-marketing@feat/shared-posterior`, base
  `c2260193ffaef464c77fb7e61cb4d0b0021be300`). Upgrade to the merged release
  later. Rationale: gives `SharedPosterior` + `BudgetOptimizer.set_posterior`
  (no-recompile draw rebinding), required by the whole pipeline.
- **[DECIDED]** GPU: NVIDIA RTX A5000 (24 GiB), single GPU. GPU-dependent
  tests (e.g. Stage 3 GP surrogate) are user-run.
- **[GATE]** G0.1 PASS — `SharedPosterior` and `BudgetOptimizer.set_posterior`
  import from the PR-pinned install.
- **[DECIDED]** Sampler: added `nutpie==0.16.11` (user request) and
  `numpyro==0.21.0` (JAX+CUDA path, PyMC-supported `nuts_sampler` value).
  `config.NUTS_SAMPLER="nutpie"` (default). **CPU (nutpie) is the faster
  choice for an MMM of this size** — 7 channels, weekly, ~183 training weeks
  — GPU only pays off for much larger multi-channel MMMs; numpyro is kept as
  an optional alternative, not the preferred sampler here.
  (Correction 2026-09-11: earlier log claimed numpyro/GPU was the practical
  choice for 2× draws — withdrawn; CPU nutpie is faster at this scale.)
- **[PROBLEM]** Stage 0 capability probe (W1 + orchestrator introspection)
  found a plan-vs-reality mismatch: the pinned optimizer allocates **per
  channel**, not per week. `budget_dims=['channel']` (shape (7,)); the `date`
  axis is excluded from decision variables and only enters via a fixed
  `budget_distribution_over_period` pattern. So "3×13 independently adjustable
  weekly spends" and "period-specific bounds" (G0.4 i/ii) are unsupported.
- **[DECIDED]** **Option A — no flighting optimization for now.** Three
  sub-decisions, recorded explicitly so the scope is unambiguous:
  1. **Expand to the full 7 channels** (not the originally-planned 3): every
     channel is optimized at channel level; `FIXED_CHANNELS = []`, `B_res = B`.
  2. **Channel-level quarterly budgets** — one decision variable per channel,
     a constant weekly rate per channel across the 13-week quarter (matches
     the case study and the pinned optimizer's `budget_dims=['channel']`).
  3. **Week-by-week (flighting) optimization is deferred to a later phase**
     (future Stage 5+ custom optimizer with a `date` budget dim). The pinned
     optimizer has no per-period decision variables; only a fixed
     `budget_distribution_over_period` pattern.
  Consequence: outer BO dimensionality = **7** (not 39). Rationale: matches
  the pinned API, keeps all constraints, dramatically simplifies the BO;
  weekly granularity adds little information value and is separable.
- **[DECIDED]** Consequence: `config.OPTIMIZE_CHANNELS` = all 7 channels,
  `FIXED_CHANNELS = []`, `B_res = B` (no residual). The `budgets_to_optimize`
  mask capability is still probed (G0.4 iii) as a general capability, even
  though all 7 channels are optimized now.
- **[GATE]** G0.1 PASS (3/3 tests) · G0.2 PASS (1/1) · G0.4 PASS (i–iv) —
  W2 implemented the Stage 0 test suite per `docs/contracts/stage0-tests.md`
  and ran `8 passed`. Key API finding pinned by the contract: the FIRST
  `set_posterior` call recompiles (draws become shared on first bind); the
  no-recompile identity holds across the SECOND call.
- **[DECIDED]** Added `pytest` as a dev dependency (W2 found it was not
  declared; tests require it for reproducibility).
- **[GATE]** G0.3 PASS — R reviewed README/docs/methods/docs/LOG and blocked
  once on two stale references ("B_res / 4 fixed channels" and
  "channel/week"); orchestrator fixed README constraints + config comments to
  the Option A framing (7 channel-level budgets, no residual, channel/quarter
  boxes). Re-review passed.
- **[DECIDED]** **Stage 0 complete.** Full test suite `8 passed`; contract
  `docs/contracts/stage0-tests.md` pinned the API facts for later stages
  (channel-level budgets; first `set_posterior` recompiles, later calls do
  not; `fit(X, y)` signature; transformers import path). Committed.

## 2026-09-11 — Stage 1: case-study fit + baselines

- **[DECIDED]** **Parallel W1+W2 execution.** Earlier "no parallel sub-agents"
  was about GPU compute, not the (cloud) subagent LLMs. User directed:
  parallelize W1 (implementation) + W2 (gates) after the contract is agreed.
  File partition is disjoint (W1: `src/`+`scripts/`; W2: `tests/` only);
  shared files (`config.py` additions, `pyproject.toml` `pythonpath=["src"]`,
  `.gitignore` `/data/`) were landed by the orchestrator before the parallel
  phase so both children could rely on them. Gate running stays sequenced
  after W1/W2 land; R still reviews code+docs after.
- **[DECIDED]** **Case-study actual posterior size**: the PR #3002 notebook
  fits `chains=6, draws=800` (4,800 total), NOT the 4,000 assumed in PLAN.
  Our target `CHAINS=4, DRAWS=8000` (32,000) therefore exceeds "2×" by a wide
  margin — kept as-is (more conservative; G1.3 gates on ≥32,000).
- **[PROBLEM]** Contract-exact `allocate_budget(total_budget, budget_bounds,
  x0=None)` failed the **Q1** baseline solve with
  `MinimizeException: Positive directional derivative for linesearch` (pinned
  SLSQP `ftol=1e-9` on an objective of scale ~5e9; deterministic across seeds
  0/1 and 50/1000-draw fits; gradient verified vs. FD to 7e-9). Q2 converged.
- **[RESOLVED]** `solve_baseline` passes `minimize_kwargs={"options":
  {"ftol": 1e-6}}` (scipy default). Q1/Q2 converge to the **identical**
  optimum; all G1.6 result assertions unchanged. Contract §2.4 amended and
  `baseline.py` docstring records the rationale. Verified on the 1000-draw
  mini-fit: Q1 nit=27, Q2 nit=28, |Σx−B|=0.
- **[DECIDED]** Artifacts are **Zarr**, not NetCDF (`netCDF4`/`h5netcdf` not
  installed). `MMM.save`/`MMM.load` dispatch on the `.zarr` extension;
  `mmm.idata.to_zarr(...)` for the standalone snapshot. Channel coords are the
  **raw `mdsp_*` names** (no renaming in the pinned `MMM`), so `budget_bounds`
  and boxes are keyed by raw names; human names (`CHANNEL_MAPPING`) are
  reporting-only.
- **[DECIDED]** Non-gated choices (contract §7): `--tune` default 1000; fit
  script does NOT run baseline solves (tested via G1.6); `load_budgets` takes
  the raw df (never idata — the posterior tree has no Q1/Q2 spend).
- **[GATE]** Stage 1 runs-now gates: **15 passed, 5 skipped** (G1.2, G1.5,
  toy-G1.6 green; Stage-0 regression green). G1.1 / G1.3 / G1.4 / G1.2(c) /
  real-G1.6 skip until the user runs the 4×8,000 fit (artifact-gated).
- **[GATE]** R (code+docs review) returned **blocked** on two items, both
  resolved: (1) missing Stage 1 LOG section — added (this section);
  (2) `test_baseline_solves.py` suppressed the Q1 cold-start warning without
  asserting its absence — now asserts Q1 emits **no** cold-start warning and
  Q2 **does** (via `warnings.catch_warnings(record=True)`). Suite re-run:
  15 passed, 5 skipped.

## 2026-09-11 — Stage 1 fit completed (all gates green)

- **[DECIDED]** Fit is **CPU (nutpie)** — faster for this MMM size; GPU
  permission was offered as a one-time exception but was not needed for the
  fit (CPU) or the Stage-2 toy (CPU). GPU will matter later (Stage 3 GP
  surrogate / Stage 5 JAX).
- **[PROBLEM]** First 4×8,000 fit (tune=1000, target_accept=0.9) had
  `n_divergences=4` (all in chain 1) and `n_rhat_nan=3953`.
- **[RESOLVED]** (a) NaN r-hat: `sampler_diagnostics` now restricts the summary
  to the model's **free random variables**
  (`var_names=[var.name for var in mmm.model.free_RVs]`), excluding the
  deterministic `*_contribution` variables whose zero-spend weeks made the
  posterior an array of zeros → NaN r-hat. `n_rhat_nan` 3953 → 0. (b)
  divergences: tune 1000→2000 dropped 4→1; `--target-accept 0.95` (new CLI
  flag) dropped it to **0**. Fit script also made idempotent (cleans stale
  zarr artifacts before writing; `w-` mode otherwise fails on re-run).
- **[GATE]** Final accepted fit (tune=2000, target_accept=0.95, seed 0):
  `n_effective_samples=32,000`, `min_ess_bulk=5,480.6`, `min_ess_tail=7,470.0`,
  `max_rhat=1.0022`, `n_divergences=0`. Recorded verbatim in
  `data/fit/fit_summary.json`.
- **[GATE]** **Stage 1 complete**: full suite `24 passed, 0 skipped` —
  G1.1 (zarr round-trip), G1.2 (windows), G1.3 (32,000 draws), G1.4 (sampler
  health), G1.5 (budgets), G1.6 (baseline Q1/Q2 solves + optimality smoke).
  Q2 baseline solve emits the expected cold-start carry-in warning (tolerated;
  Q1 must not warn — asserted).

## 2026-09-12 — Stage 2: importance weighting + weighted solves (all gates green)

- **[DECIDED]** W1 (deepseek-v4-flash) timed out 3× on the Stage 2
  implementation (high-thinking model spent its budget reading/verifying
  rather than writing). The orchestrator implemented directly; W2's
  independently-written gate tests + R's review remained the verification
  layer. Going forward: split tasks smaller and hand the agent pinned facts.
- **[DECIDED]** PSIS convention pinned against arviz-stats 1.3.2 AND the
  official ROAS-experimentation notebook: accessor is
  ``da.azstats.psislw(dim="sample")`` (NOT ``.stats``); input = NEGATED
  log-weights ``-ell``; returns normalized smoothed POSITIVE log-weights
  (exp-sum == 1); resample weights = ``exp(smoothed)``.
- **[DECIDED]** k-hat policy is ONE-SIDED: skip only when k-hat > 0.7 or
  non-finite; NEGATIVE k-hat (light tail) is accepted (matches the notebook).
- **[DECIDED]** Resample size for the Q2 solve = ``config.RESAMPLE_DRAWS =
  2_000`` (user guidance): the 4×8,000 original posterior is for
  importance-sampling coverage; 2k draws suffice for the weighted
  optimization and cut solve cost 16×.
- **[PROBLEM]** Hand-rolled thin-SLSQP wrapper (my own MMM response reimpl)
  did NOT reproduce the stock ``BudgetOptimizer`` baseline (objective ~¼ of
  stock, budgets 5–12% off) → G2.4 failed.
- **[RESOLVED]** ``solve_q2_weighted`` now REUSES the stock
  ``BudgetOptimizer`` (``mmm.budget_optimizer(Q2)`` + ``set_posterior`` +
  ``allocate_budget``). The stock optimizer already handles adstock carry-in
  and carry-over. G2.4 exact by construction.
- **[KNOWN GAP — next step]** The PROPOSED Q1→Q2 carry-in is NOT yet injected
  into ``solve_q2_weighted``: it currently cold-starts (Q2 not contiguous
  with training). ``q2_expected_response`` handles Q1 carry-in correctly
  (G2.3b), but the solve path must feed the Q1 spend via data-extension
  (``create_zero_dataset`` ``preserve_observed`` / ``carry_in_periods``, per
  user guidance). This does not block any gate; it is a correctness item for
  the actual study (PLAN Decision 8).
- **[GATE]** Stage 2: all 13 gates green — G2.0 toy VoE (CRN), G2.1 weights,
  G2.1b psislw sign, G2.2 pooling, G2.3/G2.3c k-hat policy, G2.3b carry-in
  response, G2.4 equivalence, G2.5 set_posterior rebind, G2.6 hot start,
  G2.7 serial≡pool, G2.8 resampling-noise sweep, G2.9 smoke.

## 2026-09-12 — Stage 2b: shared Q1→Q2 carry-in (single-compile) optimizer

- **[DECIDED]** Adopt the user's single-compile design: the pinned
  `BudgetOptimizer` bakes the carry-in as numpy constants at build time
  (`model_post_init` step 7: `carry_in_for(...)` → `MediaVariable(
  carry_in_values=...)` → `do(...)`); verified empirically that mutating
  `channel_data` post-build does nothing. The new
  `CarryInBudgetOptimizer` (src/mmm_evsi/carry_in_optimizer.py) rebuilds that
  substitution with the carry-in as a **pytensor shared variable**
  (`q1_carry_in`), recompiling ONCE. Per solve: `set_posterior` (draws) +
  `set_q1_carry_in` (Q1 tail) + SLSQP — no recompilation. This is the
  SharedPosterior pattern extended to the carry-in, and what the Stage 3 BO
  loop needs (one compile, many cheap solves).
- **[DECIDED]** `solve_q2_weighted` / `_solve_q2_core` now use the wrapper with
  `set_q1_carry_in(q1_weekly_spend)`, closing the Stage-2 "cold-start carry-in"
  gap (PLAN Decision 8).
- **[GATE]** G-CI-1 (no-recompile + objective changes) PASS · G-CI-2
  (zero-carry-in ≡ stock baseline) PASS · G-CI-4 (set_posterior rebind) PASS ·
  G-CI-3 (carry-in lift vs `q2_expected_response`) relaxed to a directional
  pin — see next item.
- **[OPEN QUESTION — scale]** The stock objective
  (`total_media_contribution_original_scale`) is ~4.2× my
  `q2_expected_response` (full response, mu×target_scale, validated against
  observed training sales ~110M/wk). Relative carry-in lifts differ ~3.5×
  (wrapper 6.9% vs response 2.0%), consistent with media-only vs
  total-response denominators, but the absolute ~4× unit gap is UNRESOLVED and
  should be pinned down before Stage 4 aggregates utilities across paths.
  Candidates: optimization-model target/scale handling in
  `create_zero_dataset`, or the "original scale" un-scaling factor.
  All utilities WITHIN one path are consistent (stock units everywhere), so
  this does not block Stage 3.
