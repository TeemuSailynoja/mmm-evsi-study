# mmm-evsi-study

Value-of-exploration / EVSI study for PyMC-Marketing budget optimization.

## Purpose

Given a fitted Marketing Mix Model (MMM), we want to choose **next quarter's
(Q1) channel budget allocation** not only to maximize immediate expected
return, but also to **maximize the value of the information it produces** for
the *following* quarter's (Q2) budget optimization.

This is a nested *design-to-decide* problem:

1. **Baseline Q1 optimization** — optimize Q1 channel budgets under a total
   budget constraint + per-channel box constraints.
2. **Exploration (outer loop)** — Bayesian optimization over perturbations of
   the baseline Q1 allocation, every proposal kept inside the feasible set.
3. **Proposal utility (expensive, nested)** — for a proposed Q1 allocation:
   a. simulate observed Q1 outcomes (posterior predictive);
   b. importance-weight the existing posterior draws with the joint
      likelihood of the simulated quarter (no refitting);
   c. run a **Q2 budget optimization** under the weighted posterior (with
      geometric-adstock carryover from the proposed Q1 spend);
   d. utility = optimized Q2 value.
4. **Value of exploration** = best proposal's utility − baseline-allocation
   utility, where *both* arms learn from Q1 (the only difference is which Q1
   allocation produced the information).

## Key quantities

For existing posterior draws θ_s ~ p(θ | D) and a simulated quarter y* under
allocation a (with a ~ posterior-predictive q(y*), so no outer proposal
correction is needed):

```
ℓ_s  = log p(y* | θ_s, a)          # joint log-likelihood of the whole quarter
log w_s = ℓ_s − logsumexp(ℓ)        # importance weights (one per draw)
```

```
U(a) = E_{y*|a}[ OptQ2(y*, a) ] − λ · ( V_Q1(baseline) − V_Q1(a) )
```

- `OptQ2(y*, a)` — optimized Q2 value under the weighted posterior; depends on
  `a` through **both** learning (weights) and **physical adstock carryover**.
- `V_Q1(a)` — planning-time expected Q1 revenue (current posterior, no
  reweighting).
- `λ` — cost of Q1 revenue deviation (initially 1: $1 Q1 loss = $1 expected
  Q2 gain).
- `E_max` — experimentation budget capping the λ-weighted Q1 loss.

Benefit is reported two ways: **total benefit** (carryover included) and
**isolated information value** (adstock history fixed at baseline, only
weights vary).

## Constraints

- **Total budget** = currently planned budget for the quarter (sum of planned
  spend over all 7 channels — no residual; `FIXED_CHANNELS = []`).
- **Per-channel box** = ±30% of the planned spend for that channel/quarter.
  (Channel-level quarterly budgets over the full 7 channels; **no flighting
  optimization for now** — week-by-week scheduling is deferred to a later
  phase.)
- **Experimentation budget** `E_max` caps Q1 revenue loss.

All quantities are in expected **sales** units (the MMM target); sales,
revenue and profit are not used interchangeably.

## Pipeline (see PLAN.md for full detail)

- **Stage 0** — docs, skeleton, pinned `pymc-marketing` (PR #3002) install,
  optimizer capability probe. ✅ COMPLETE
- **Stage 1** — fit the case-study MMM (2× draws, all-but-last-two-quarters),
  baseline Q1/Q2 solves. ✅ COMPLETE (24 gates green)
- **Stage 2** — analytic toy validation, then resampling-based weighted
  optimization (PSIS weights, `set_posterior`, hot-started SLSQP, process
  pool). ✅ COMPLETE (13 gates green + 2b carry-in)
- **⏸ STOP checkpoint** — baseline-arm EVSI computed: 4.72% ± 0.40% of prior
  utility (200 outcomes, k-hat=0.590). See `data/fit/baseline_arm_evsi.npz`.
- **Stage 3** — Bayesian optimization over feasible Q1 perturbations.
- **Stage 4** — aggregation, uncertainty decomposition, value of exploration.
- **Stage 5** — exact shared-weights tensor + JAX solver (conditional
  upgrades).

## Development workflow

Agent-orchestrated, contract-first. See PLAN.md "Execution workflow & quality
gates" and `docs/LOG.md` for the decision record. Each stage has yes/no
quality gates implemented as pytest tests in `tests/`.

## Layout

```
docs/        method notes + decision log (LOG.md) + interface contracts
scripts/     fit_case_study.py (fit + save the MMM), baseline_arm_evsi.py
src/mmm_evsi/ reusable pipeline modules
toy/         analytic conjugate end-to-end validation
tests/       pytest quality gates
notebooks/   marimo notebooks per stage
```

## First Results

The baseline-arm EVSI has been computed (200 outcomes):

- **EVSI = 4.72% ± 0.40%** of prior Q2 utility
- Standard error of the mean: 5.4M (8.3% relative)
- 95% CI: [3.95%, 5.49%]
- k-hat = 0.590 (well below 0.7 skip threshold)
- Total computation: ~400s (CompiledResponseEvaluator, 24x speedup)

See `data/fit/baseline_arm_evsi.npz` and `docs/LOG.md` for details.

## Performance

Key benchmarks (after CompiledResponseEvaluator optimization):

| Operation | Time |
|-----------|------|
| Graph compile (one-time) | ~35s |
| simulate_quarter | ~0.16s |
| quarter_log_likelihood | ~0.16s |
| PSIS weights | ~0.00s |
| resample_posterior | ~0.35s |
| **Per outcome** | **~0.66s** |
| 200 outcomes | **~400s** |
| Without caching | ~4000s |

The CompiledResponseEvaluator caches the PyTensor graph and swaps shared
variables (posterior draws + spend data) without recompilation, achieving
a ~24x speedup over the naive approach.
