# Flighting-Pattern-Aware Spend Plan

## Problem
`allocation_to_weekly_spend()` in `experiments.py` flattens all weekly spend to constant rates, discarding the actual flighting shapes from the data. This means the BO optimization ignores how spend timing affects adstock carry-in from Q1→Q2.

## Goal
Modify the BO pipeline to preserve actual flighting patterns: the BO proposes 7 quarterly scalars, each scaling the corresponding channel's weekly pattern from the data.

## Implementation Phases

### Phase 1: Data Layer (`budgets.py`)
**Task**: Add `weekly_spend: np.ndarray` (shape 13,7) field to `QuarterBudget` dataclass. Create `planned_weekly_spend(df, window)` function that returns the (13,7) array of weekly spend per channel. Update `BudgetPlan` construction to include weekly spend.

**Files**: `src/mmm_evsi/budgets.py`

**Acceptance Criteria**:
- `QuarterBudget` has `weekly_spend` field
- `planned_weekly_spend()` returns correct (13,7) array
- All existing callers still work (or are updated)

### Phase 2: Core Conversion (`experiments.py`)
**Task**: Rewrite `allocation_to_weekly_spend()` to accept baseline weekly spend pattern and scale it: `weekly = baseline * (proposal_quarterly / baseline_quarterly)`. Update `simulate_quarter()` to pass the baseline pattern.

**Files**: `src/mmm_evsi/experiments.py`

**Acceptance Criteria**:
- Given baseline pattern B and allocation a, output is `B * (a / B_quarterly)`
- Sum of output per channel equals allocation value
- Channel shapes are preserved from baseline

### Phase 3: BO & Importance Layer (`bo_design.py`, `importance.py`)
**Task**: Update `compute_v_q1()` in `bo_design.py`, `quarter_log_likelihood()` and `evaluate_allocation()` in `importance.py` to use flighting-aware weekly spend.

**Files**: `src/mmm_evsi/bo_design.py`, `src/mmm_evsi/importance.py`

**Acceptance Criteria**:
- All calls to `allocation_to_weekly_spend()` pass the baseline weekly spend
- No regressions in existing tests

### Phase 4: Q2 Optimization (`optimize_slsqp.py`)
**Task**: Replace flat spend creation in `q2_expected_response()` with flighting-aware scaling using Q2 baseline pattern.

**Files**: `src/mmm_evsi/optimize_slsqp.py`

**Acceptance Criteria**:
- Q2 optimizer uses flighting-aware spend derived from Q2 baseline pattern
- Sum of Q2 weekly spend per channel equals Q2 allocation

### Phase 5: Q2 Optimizer Core (`carry_in_optimizer.py`)
**Task**: Add `set_q2_weekly_spend()` method to `CarryInBudgetOptimizer` so the Q2 optimizer's objective function uses flighting-aware spend in its shared variables.

**Files**: `src/mmm_evsi/carry_in_optimizer.py`

**Acceptance Criteria**:
- `set_q2_weekly_spend()` method exists and updates shared variables
- Q2 optimization uses flighting-aware spend in the objective

### Phase 6: Tests & Contracts
**Task**: 
- Update G2.3b (adstock_carry) to use actual weekly spend
- Add G2.3c `flighting_preserved` — verify shapes are preserved
- Add G2.3d `q2_flighting` — verify Q2 optimizer uses flighting-aware spend
- Add G3.6 `flighting_bo` — verify BO result differs from flat-spend version
- Update `docs/contracts/stage3_bo.md` with new signatures

**Files**: `tests/test_flighting_preserved.py`, `tests/test_q2_flighting.py`, `tests/test_flighting_bo.py`, `tests/test_adstock_carry.py`, `docs/contracts/stage3_bo.md`

**Acceptance Criteria**:
- All new gates pass
- All existing gates still pass
