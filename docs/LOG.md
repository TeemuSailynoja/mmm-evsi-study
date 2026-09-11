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
- **[DECIDED]** GPU: NVIDIA RTX A5000 (24 GiB), single GPU — no parallel
  sub-agents; GPU tests are user-run.
- **[GATE]** G0.1 PASS — `SharedPosterior` and `BudgetOptimizer.set_posterior`
  import from the PR-pinned install.
- **[DECIDED]** Sampler: `nutpie` and `numpyro` were not installed; added
  `numpyro==0.21.0` to use JAX+CUDA (RTX A5000) for GPU MCMC. The case-study
  notebook used `nuts_sampler="pymc"`, but for 2× draws (4×8,000) the GPU
  path (numpyro) is the practical choice. `numpyro` is a PyMC-supported
  `nuts_sampler` value.
- **[DECIDED]** Also added `nutpie==0.16.11` (user request) as an
  alternative fast sampler; `config.NUTS_SAMPLER="nutpie"`.
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
