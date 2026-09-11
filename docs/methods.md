# Method notes

## Importance sampling for the Q1→Q2 posterior update

We never refit the model. Given posterior draws θ_s ~ p(θ | D) (s = 1..S)
and a simulated quarter y* under Q1 allocation a, the reweighted posterior is

```
ℓ_s    = log p(y* | θ_s, a)      # joint log-likelihood of the whole quarter
log w_s = ℓ_s − logsumexp_s(ℓ)    # normalized log-weights, one per draw
```

Because y* is simulated from the posterior predictive q(y*) = (1/S) Σ_s
p(y* | θ_s), the proposal distribution q(y*) coincides with the denominator
of the reweighting, so **no outer q(y*)-correction is needed**. The weights
above are pure posterior reweighting. (If a different proposal were ever
used, a separate q(y*) term would have to be subtracted — kept distinct in
code and docs.)

**Jointness**: ℓ_s sums over the whole simulated quarter, giving **one weight
per draw and one diagnostic per simulated quarter** (not per week).

## Pareto-smoothed importance sampling (PSIS)

- Chains are **pooled** first: stack `chain` × `draw` into a single `sample`
  dimension, so weights and k-hat are joint across chains.
- Smoothing uses `arviz_stats`'s `psislw` via the DataArray accessor
  `.stats.psislw(dim="sample")`, returning `(smoothed log-weights, khat)`.
- **Sign convention**: the pinned `arviz_stats` `_psislw` internally negates
  its input before tail-fitting, so the input is the **negated** log-weights.
  This must be **verified against the pinned implementation in a test** and
  kept distinct in code: log-likelihoods, log-weights, and negated inputs are
  three different things.
- **k-hat policy** (validity gate):
  - k-hat ≤ 0.7: weights usable.
  - k-hat > 0.7: **skip that simulated quarter** (log it) initially.
  - Later remedy (Stage 5+): moment matching (`arviz_stats.loo_moment_match`)
    or refit.
- PSIS-smoothed weights are **not required to equal** unsmoothed weights
  exactly; they only need finite k-hat and to reproduce the target
  expectation within Monte-Carlo tolerance.

## Resampling vs. exact weighting

- **Stage 2 (default)**: multinomial-resample draws ∝ smoothed weights → new
  idata → `BudgetOptimizer.set_posterior` (no recompile) → solve with the
  stock SLSQP `BudgetOptimizer`. Resampling noise should **not bias** utility
  but **inflates its uncertainty** (quantified with a seed sweep).
- **Stage 5 (upgrade)**: an exact `shared_weights` tensor wired into the
  weighted-mean utility graph — no resampling noise, compiled once.

## Adstock carryover

Geometric adstock means Q2 response depends directly on the proposed Q1 spend
history. The inner objective is therefore `OptQ2(y*, a)`, and adstock state
crosses the train→Q1 and Q1→Q2 boundaries. The reported benefit is split:

- **total benefit** — carryover included;
- **isolated information value** — adstock history fixed at baseline, only
  the weights vary.

Effects extending beyond Q2 are truncated at the Q2 horizon; the discarded
tail is estimated and reported.

## Uncertainty decomposition (Stage 4)

The final value of exploration decomposes variance into four independent
sources:

1. outcome-simulation MC error (number of y* draws),
2. posterior-approximation error (PSIS ESS, k-hat),
3. resampling error (seed-sweep spread),
4. optimization error (solver gap/tolerance).

## Economic units

All quantities are expected **sales** (the MMM target). `E_max` caps the
λ-weighted Q1 loss (identical to actual loss while λ = 1). If profit matters
later, add an explicit margin/conversion and re-document.
