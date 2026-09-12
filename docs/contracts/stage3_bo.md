# Stage 3 Contract: Bayesian Optimization Outer Loop

**Date:** 2024
**Status:** Draft → awaiting R review

## Overview

Bayesian optimization over feasible 7-dim Q1 budget perturbations, with
GP surrogate, EI acquisition, and CRN paired winner evaluation.

## Interface Contract

### Data Structures

```python
@dataclass
class BOProposal:
    """A single Q1 allocation proposal."""
    channels: list[str]
    values: np.ndarray  # shape (7,) — one budget per channel
    total: float  # Σx = B (budget constraint)

@dataclass
class BOTrace:
    """One iteration of the BO loop (flat: no nested BOProposal)."""
    iteration: int
    allocation: xr.DataArray  # shape (7,) with channel coords
    utility: float  # U(a) = E[OptQ2] - λ·Q1_loss
    q1_loss: float  # V_Q1(baseline) - V_Q1(proposal)
    gp_mean: float  # GP posterior mean at proposal
    gp_std: float   # GP posterior std at proposal
    elapsed: float  # wall-clock for this evaluation

@dataclass
class BOResult:
    """Final BO output."""
    best_proposal: BOProposal
    best_utility: float
    trace: list[BOTrace]
    n_evaluations: int
    early_stopped: bool
    reason: str  # "converged" | "budget_exhausted" | "early_stop"
```

### Core Functions

```python
def generate_feasible_proposal(
    baseline_allocation: xr.DataArray,
    q1_cfg: BudgetConfig,
    rng: np.random.Generator,
    E_max: float,
    V_Q1_baseline: float,
    n_iter: int = 10,
) -> xr.DataArray:
    """Generate one feasible perturbation.

    Two-phase feasibility:
    Phase 1 (structural): null-space reparameterization (6 free dims +
    sum constraint) + box clipping (±30%) — ensures Σx = B and
    per-channel bounds by construction. Infeasible points w.r.t. these
    constraints are never proposed.
    Phase 2 (E_max): perturbation magnitude is bounded as a proxy for
    Q1 loss. Full Q1 loss check happens at evaluation time in
    evaluate_proposal → compute_v_q1 → reject if q1_loss > E_max.

    Returns: xr.DataArray with channel coords, values summing to B.
    Raises RuntimeError if no feasible proposal found within n_iter attempts.
    """

def lhs_feasible_design(
    baseline_allocation: xr.DataArray,
    q1_cfg: BudgetConfig,
    n_samples: int,
    E_max: float,
    V_Q1_baseline: float,
    rng: np.random.Generator,
) -> list[xr.DataArray]:
    """Generate N feasible LHS samples.

    Uses scipy LHS on 6 free dims, projects onto sum constraint,
    clips to boxes, gates on E_max.

    Returns: list of feasible proposals (may be < n_samples if some fail).
    """

def compute_v_q1(
    mmm, posterior, df, allocation, q1_cfg, n_draws: int = 100,
) -> float:
    """Fast Q1 expected sales under allocation.

    Uses n_draws=100 for BO-loop speed. Full evaluation reserved for winner.

    Returns: expected Q1 sales in the same units as the MMM target.
    """

class GPSurrogate(ABC):
    """Abstract GP backend — swap sklearn → GPyTorch later."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to observed (X, y) pairs."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) at query points."""

    @abstractmethod
    def acquire(self, acquisition: str = "ei") -> np.ndarray:
        """Return the next candidate point via acquisition maximization."""

def bayesian_optimization(
    mmm, idata, df,
    q1_cfg: BudgetConfig, q2_cfg: BudgetConfig,
    baseline_allocation: xr.DataArray,
    V_Q1_baseline: float,
    n_evaluations: int = 100,
    n_initial: int = 20,
    n_outcomes: int = 10,
    E_max_fraction: float = 0.10,
    lambda_: float = 1.0,
    gp_backend: type[GPSurrogate] | None = None,
    seed: int = 42,
) -> BOResult:
    """Full BO loop.

    1. Generate LHS initial design (n_initial feasible samples).
    2. Evaluate each with evaluate_proposal (n_outcomes draws).
    3. Loop: fit GP → maximize EI → evaluate new proposal → update trace.
    4. Early stop: relative tolerance on best utility over patience steps.
    5. Return BOResult with best proposal, trace, and metadata.

    Utility = E[OptQ2] - λ·Q1_loss (λ = lambda_, default 1.0).
    """

def re_evaluate_winner_paired(
    winner_allocation: xr.DataArray,
    baseline_allocation: xr.DataArray,
    mmm, idata, df,
    q1_cfg: BudgetConfig, q2_cfg: BudgetConfig,
    V_Q1_baseline: float,
    n_outcomes: int = 100,
    seed: int = 42,
) -> dict:
    """CRN paired re-evaluation of BO winner vs baseline.

    Uses common random numbers for both allocations.
    Returns a dict with keys: `delta`, `delta_se`, `delta_ci_lower`,
    `delta_ci_upper`, `winner_utility`, `baseline_utility`, `q1_loss`.

    This is the anti-winner's-curse step: fresh simulations, paired,
    not the BO search estimates.
    """

def decompose_utility(
    re_eval_result: dict,
    baseline_utility_unweighted: float,
) -> dict:
    """Decompose winner utility into total benefit and isolated information value.

    total_benefit = winner_utility - baseline_utility (includes adstock carryover)
    isolated_information_value = benefit with adstock fixed at baseline (weights only)

    Returns dict with: `total_benefit`, `isolated_information_value`, `carryover_effect`.
    """
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `E_MAX_FRACTION` | 0.10 | E_max as fraction of Q1 total budget |
| `BO_N_EVALUATIONS` | 100 | Max BO evaluations |
| `BO_N_INITIAL` | 20 | LHS initial design size |
| `BO_N_OUTCOMES` | 10 | Predictive samples per proposal |
| `BO_TOL` | 1e-4 | Early stop relative tolerance |
| `BO_PATIENCE` | 10 | Early stop patience (iterations) |
| `BO_N_DRAW_VQ1` | 100 | n_draws for fast Q1 evaluation in BO loop |
| `LAMBDA` | 1.0 | Revenue trade-off weight in U(a) = E[OptQ2] − λ·Q1_loss |

## Quality Gates (to be implemented by W2)

- **G3.1** `test_bo_feasibility.py`: N=1,000 proposals all satisfy
  |Σx − B| ≤ 1e-6, ±30% boxes, Q1 loss ≤ E_max
- **G3.2** `test_bo_lhs.py`: LHS sweep runs in parallel, wall-clock logged
- **G3.3** `test_gp_surrogate.py`: GP surrogate fits sweep data,
  held-out rank correlation > 0.5 (GPU test: user-run)
- **G3.4** `test_bo_convergence.py`: BO best ≥ LHS best at equal budget
- **G3.5** `test_winner_eval.py`: Winner re-evaluated with CRN,
  paired with baseline, delta CI valid

## Notes

- Start with CPU-only GP (`sklearn.gaussian_process`), modular interface
  for GPyTorch swap later.
- Feasibility built-in via reparameterization (6 free + sum constraint,
  box clipping, E_max gate). Never propose infeasible points.
- Utility in sales units (MMM target). λ=1 initially.
