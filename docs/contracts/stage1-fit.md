# Stage 1 — Case-Study MMM Fit + Baselines: Interface/API Contract

Status: **agreed contract** (W1). Reviewer (R) critiques; W2 implements the
quality-gate tests **exactly** against this document. W1 later implements the
modules below against the same document. This file covers **quality gates
G1.1–G1.6** (PLAN.md Stage 1).

Scope decisions that differ from the Stage-0 environment are pinned in
Section 0 — read it first. All facts were **verified against the pinned
install** (pymc-marketing 1.1.0, PR #3002 head `5743b310`, Python 3.13,
ArviZ 1.3.0) and are normative for W2 and W1.

---

## 0. Pinned API facts (verified; do not re-derive)

- **Artifacts must be Zarr, not NetCDF.** `netCDF4`/`h5netcdf` are **not
  installed**; `xr.DataTree.to_netcdf` raises
  `ValueError: cannot write NetCDF files with format='NETCDF4' because none of
  the suitable backend libraries (netCDF4, h5netcdf) are installed`.
  - `MMM.save(fname)` dispatches on the extension: `.zarr` (or a directory) →
    `idata_to_zarr`; anything else → `to_netcdf` (fails). `MMM.load` loads
    `.zarr`/directories via `idata_from_zarr`.
  - Verified round-trip: `mmm.save("m.zarr")` → `MMM.load("m.zarr")` preserves
    the posterior; `mmm.idata.to_zarr("i.zarr")` →
    `xr.open_datatree("i.zarr", engine="zarr")` preserves the posterior.
    Both emit benign `ZarrUserWarning` messages (consolidated metadata) —
    ignore them.
- **The DataTree is `mmm.idata`; `mmm.fit(X, y, ...)` returns it.**
  `mmm.fit_result` is a **posterior `xr.Dataset` view, not a DataTree** —
  do not use it for the gates. Always bracket-access: `idata["posterior"]`,
  `idata["sample_stats"]`, never `idata.posterior`.
- **Channel coordinates are the raw `mdsp_*` names.** The pinned `MMM` has no
  channel-renaming parameter; `channel_contribution` coords and
  `budgets_to_optimize` coords equal the `channel_columns` list passed at
  construction (`["mdsp_audtr", "mdsp_dm", "mdsp_inst", "mdsp_nsp",
  "mdsp_on", "mdsp_so", "mdsp_vidtr"]` for
  `channel_columns=config.CHANNEL_COLUMNS`). **Consequence: `budget_bounds`
  dicts, budget dicts, and boxes are keyed by raw `mdsp_*` names, never human
  names** — a human-keyed `budget_bounds` raises
  `KeyError("mdsp_dm")` (verified). Human names (`config.CHANNEL_MAPPING`)
  are for reporting only.
- **Posterior var names** (fit with controls): `adstock_alpha`, `gamma_fourier`,
  `intercept_contribution`, `channel_contribution`, `saturation_*`,
  `y_sigma`, `*_contribution`, plus control vars. The observed-data node is
  internally named **`y`** regardless of `target_column`; trust
  `mmm.target_column` for the target name.
- **`observed_data` dates = the training span** (with `date` coords as
  `datetime64`); the model never sees Q1/Q2 rows. `idata["fit_data"]` holds
  only training channels + target — it cannot supply Q1/Q2 planned spend.
- **`random_seed=<int>` with multiple chains seeds chains independently**
  (verified: chains are not identical). Deterministic per seed.
- **Diagnostics:** `az.summary(idata, fmt="wide")` returns columns
  `mean, sd, eti89_lb, eti89_ub, ess_bulk, ess_tail, r_hat, mcse_mean,
  mcse_sd` — **pooled** (chains+draws combined). `r_hat` is `NaN` with
  `chains == 1` (drop NaNs before the max). Nutpie writes
  `sample_stats` including **`diverging`**.
- **`BudgetOptimizer`**: `opt = mmm.budget_optimizer(start, end)` gives
  `_budget_dims == ["channel"]`, `_budget_shape == (7,)`, all-True
  `budgets_to_optimize` (dims `("channel",)`). `allocate_budget(total_budget,
  budget_bounds=None, x0=None, minimize_kwargs=None, return_if_fail=False,
  callback=False)` returns a `BudgetOptimizationResult` with `.budgets`
  (DataArray over `("channel",)`), `.scipy_result` (`OptimizeResult`; `.fun`
  is the **minimized** objective = −expected response, `.x` flat decision
  vector, `.nit`, `.success`), and `.spend_var_allocations` (empty dict here —
  no spend vars). `x0=None` = uniform spread.
- **`opt._objective_and_grad(x_flat)`** returns a **list `[fun, grad]`** (`fun`
  = −expected response; grad shape `(7,)`), where `x_flat` is the
  full 7-length decision vector (all cells optimized). Already pinned by the
  Stage-0 contract (G0.2); reused by the G1.6 optimality smoke.
- **Out-of-training optimization windows work.** Windows joining the training
  end date + 7 days are contiguous → automatic 6-week adstock carry-in from
  training spend (verified: no warning). Windows separated by a gap warn
  `Requested 6 carry-in periods, but the training data ending … is not
  contiguous with the window starting … Falling back to a cold start (zeros)`
  — **expected for Q2** (training ends 2018-01-28, Q2 starts 2018-05-06).
  Tolerate the warning in tests.
- **`mmm_evsi` is not installed in the venv** — `import mmm_evsi` fails
  without the repo `src/` on `sys.path`. Pytest gets it via
  `[tool.pytest.ini_options] pythonpath = ["src"]` (Section 2.7); the fit
  script bootstraps itself (Section 2.6).

---

## 1. Scope & artifact layout

### 1.1 Single artifacts directory + exact filenames (config.py)

New constants in `src/mmm_evsi/config.py` (repo-root-absolute, CWD-agnostic):

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "fit"
MODEL_FILE = ARTIFACTS_DIR / "case_study_mmm.zarr"     # directory (zarr store)
IDATA_FILE = ARTIFACTS_DIR / "case_study_idata.zarr"   # directory (zarr store)
SUMMARY_FILE = ARTIFACTS_DIR / "fit_summary.json"
```

`ARTIFACTS_DIR` is the **single constant** every module defaults to; the
script's `--outdir` overrides it per run. All three artifact paths must be
derived from `ARTIFACTS_DIR` exactly as above.

### 1.2 Artifact contents

| File | Written by | Contents |
|---|---|---|
| `data/fit/case_study_mmm.zarr` | `mmm.save(str(MODEL_FILE))` | Full MMM DataTree: posterior + sample_stats + observed_data (train window only) + model config attrs. `MMM.load` round-trips it. |
| `data/fit/case_study_idata.zarr` | `mmm.idata.to_zarr(str(IDATA_FILE))` | Standalone idata snapshot (same groups) for cheap stats-only access (no model graph rebuild). |
| `data/fit/fit_summary.json` | `json.dump` | Fit settings + sampler diagnostics (schema in 1.3). |

### 1.3 `fit_summary.json` schema (normative)

```json
{
  "created_at": "ISO-8601 UTC",
  "data": {"source": "/tmp/mmm_stan.csv", "rows": 209,
           "date_min": "2014-08-03", "date_max": "2018-07-29"},
  "fit": {"window": ["2014-08-03", "2018-01-28"], "n_weeks": 183,
          "chains": 4, "draws": 8000, "tune": 1000,
          "sampler": "nutpie", "target_accept": 0.9, "random_seed": 0,
          "n_effective_samples": 32000,
          "adstock": "GeometricAdstock(l_max=6)",
          "saturation": "LogisticSaturation",
          "yearly_seasonality": 5,
          "channels": ["mdsp_audtr", "mdsp_dm", "mdsp_inst", "mdsp_nsp",
                       "mdsp_on", "mdsp_so", "mdsp_vidtr"],
          "controls": ["hldy_*"]},
  "diagnostics": {"min_ess_bulk": 0.0, "min_ess_tail": 0.0,
                  "max_rhat": 0.0, "n_rhat_nan": 0, "n_divergences": 0},
  "artifacts": {"model": "data/fit/case_study_mmm.zarr",
                "idata": "data/fit/case_study_idata.zarr",
                "summary": "data/fit/fit_summary.json"}
}
```

`diagnostics` values come from `sampler_diagnostics(idata)` (Section 2.5) —
identical derivation to the G1.4 test. The `fit_summary.json` `diagnostics`
object stores ONLY the 5 gated fields (`min_ess_bulk`, `min_ess_tail`,
`max_rhat`, `n_rhat_nan`, `n_divergences`); `chains`, `draws`,
`n_effective_samples` are recorded under `fit` (not duplicated under
`diagnostics`, even though `sampler_diagnostics` returns all 8).
`"controls": ["hldy_*"]` means "all `hldy_*` columns", recorded as the actual
list of column names.

### 1.4 Gitignore

`.gitignore` gains one entry (implementation task):

```
# Stage 1 model artifacts (large, user-generated)
/data/
```

(`/tmp/mmm_stan.csv` is outside the repo and needs no entry.)

---

## 2. Module APIs

### 2.1 `src/mmm_evsi/load_mmm.py` (new)

```python
class SanityCheckError(ValueError):
    """Raised when a loaded artifact violates a pinned sanity check."""

def load_case_study_data(path: str | Path | None = None) -> pd.DataFrame
def select_window(df: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame
def load_mmm(model_file: str | Path | None = None) -> tuple[MMM, xr.DataTree]
def load_budgets(df: pd.DataFrame | None = None) -> BudgetPlan
```

- `load_case_study_data(path=None)`: `path=None` → `config.RAW_CSV`
  (`/tmp/mmm_stan.csv`). If the file is missing, **download** it to `path`
  from `config.DATA_URL` (standard-library `urllib.request.urlretrieve`; no
  new dependency). Always parse `config.DATE_COLUMN` to `datetime64` before
  returning. Returns the full 209-row dataset (all windows included).
- `select_window(df, window)`: rows with
  `df[config.DATE_COLUMN]` in `[pd.Timestamp(window[0]), pd.Timestamp(window[1])]`
  inclusive. Raises `ValueError` if `window[0] > window[1]` or nothing is
  selected.
- `load_mmm(model_file=None)`: `None` → `config.MODEL_FILE`. Raises
  `FileNotFoundError` (with a message pointing at
  `scripts/fit_case_study.py` and this contract) when the artifact is absent.
  Loads `mmm = MMM.load(str(model_file))`, sets `idata = mmm.idata`, then
  **sanity checks** (each failure raises `SanityCheckError` with the failing
  condition named):
  1. Channel coords:
     `list(idata["posterior"]["channel_contribution"].coords["channel"].values)
     == config.CHANNEL_COLUMNS` (7 raw `mdsp_*` names, exact list).
  2. Target: `mmm.target_column == config.SALES_COLUMN` (`"sales"`).
  3. Date range: `idata["observed_data"]["date"].min() >=
     Timestamp(TRAIN_WINDOW[0])` and `.max() <= Timestamp(TRAIN_WINDOW[1])`
     (the model was fit on the train window only).
  4. Controls: `mmm.control_columns` non-empty and every entry
     `startswith(config.CONTROL_COLUMNS_PREFIX)`.
  Returns `(mmm, idata)` (`idata is mmm.idata`).
- `load_budgets(df=None)`: `None` → `load_case_study_data()` (downloads if
  absent). Computes planned Q1/Q2 spend from **the raw weekly data** and
  returns a `BudgetPlan` (Section 2.3). A `DataTree`/idata cannot be a source
  — it contains no Q1/Q2 spend (Section 0); passing one raises `TypeError`
  with that explanation.

### 2.2 `src/mmm_evsi/budgets.py` (new — budgets stay out of config.py)

Rationale (design choice, per the brief): `config.py` remains a constants
module; spend computations from raw data live in a dedicated module.

```python
def planned_budget_per_channel(df: pd.DataFrame,
                               window: tuple[str, str]) -> dict[str, float]
def budget_boxes(planned: Mapping[str, float],
                 box_pct: float = config.BOX_PCT) -> dict[str, tuple[float, float]]
```

- `planned_budget_per_channel`: for each `config.CHANNEL_COLUMNS` channel,
  `float(df_window[channel].sum())` (sum of weekly spend in `window`).
  Keyed by raw `mdsp_*` names.
- `budget_boxes`: `{(c): (0.7 * p_c, 1.3 * p_c)}` — i.e.
  `((1 - box_pct) * p, (1 + box_pct) * p)` with `box_pct=0.30`.
  Raises `ValueError` on missing/unknown channel keys.

```python
@dataclass(frozen=True)
class QuarterBudget:
    window: tuple[str, str]
    planned: dict[str, float]                   # {raw channel: spend in window}
    total: float                                # B = sum(planned.values())
    boxes: dict[str, tuple[float, float]]       # (0.7*p, 1.3*p) per channel

@dataclass(frozen=True)
class BudgetPlan:
    q1: QuarterBudget
    q2: QuarterBudget
```

`QuarterBudget` invariant (asserted in `load_budgets`): `planned.keys() ==
set(config.CHANNEL_COLUMNS)`, `total == sum(planned.values())`, and
`boxes == budget_boxes(planned)`. (The toy fixture's `ch1…ch7` plan is a
test-local stand-in, never routed through `load_budgets`.)

### 2.3 `src/mmm_evsi/diagnostics.py` (new — shared G1.4 quantities)

```python
def sampler_diagnostics(
    idata: xr.DataTree, var_names: list[str] | None = None
) -> dict[str, float | int]
```

**`var_names` (REQUIRED for the case-study MMM, 2026-09-11 fix):** limit the
summary to the model's **free random variables**
(`[var.name for var in mmm.model.free_RVs]`). The posterior group also holds
deterministic `*_contribution` variables (`channel_contribution` etc.) with
zero-spend weeks that yield NaN r-hat/ESS when summarized — summarizing them
produced `n_rhat_nan == 3953` (fixed to 0 by restricting to free RVs). The
fit script passes the free-RV names; the G1.4 test derives them from the
loaded model identically.

Implementations (mandatory, matching the G1.4 test exactly):

```python
summ = az.summary(idata, var_names=var_names, fmt="wide")  # pooled
rhat = summ["r_hat"].dropna()
return {
    "min_ess_bulk":    float(summ["ess_bulk"].min()),
    "min_ess_tail":    float(summ["ess_tail"].min()),
    "max_rhat":        float(rhat.max()),
    "n_rhat_nan":      int(summ["r_hat"].isna().sum()),
    "n_divergences":   int(idata["sample_stats"]["diverging"].values.sum()),
    "chains":          int(idata["posterior"].sizes["chain"]),
    "draws":           int(idata["posterior"].sizes["draw"]),
    "n_effective_samples": int(idata["posterior"].sizes["chain"]
                               * idata["posterior"].sizes["draw"]),
}
```

### 2.4 `src/mmm_evsi/baseline.py` (new)

```python
@dataclass(frozen=True)
class BaselineResult:
    quarter: str                 # "Q1" | "Q2"
    window: tuple[str, str]
    budgets: xr.DataArray        # dims ("channel",), coords = raw mdsp_* names
    objective_value: float       # expected sales = -float(scipy_result.fun)
    scipy_result: scipy.optimize.OptimizeResult

def solve_baseline(mmm: MMM, window: tuple[str, str],
                   budget_cfg: QuarterBudget) -> BaselineResult
```

Exact call (mandatory):

```python
opt = mmm.budget_optimizer(window[0], window[1])
res = opt.allocate_budget(total_budget=budget_cfg.total,
                          budget_bounds=budget_cfg.boxes,
                          x0=None,           # x0=None = uniform start
                          minimize_kwargs={"options": {"ftol": 1e-6}})
```

> **Solver fix (2026-09-11, W1):** the pinned optimizer defaults SLSQP to
> `ftol=1e-9`, which fails the **Q1** solve with
> `MinimizeException: Positive directional derivative for linesearch` on an
> objective of scale ~5e9 (deterministic across seeds 0/1 and 50/1000-draw
> fits; gradient verified vs. FD to 7e-9). `ftol=1e-6` (scipy default)
> converges Q1 and Q2 to the **identical** optimum and leaves every G1.6
> result assertion unchanged. The `minimize_kwargs` override above is
> normative.

Post-solve validation inside `solve_baseline` (raises `RuntimeError` naming
the violated constraint):
- `res.scipy_result.success` is True;
- `abs(float(res.budgets.sum()) - budget_cfg.total) <= 1e-6` (Σx = B);
- every channel within its box ± `1e-6`.

`objective_value = -float(res.scipy_result.fun)` (maximized expected sales in
target units over the 13-week window). `quarter` is `"Q1"` when `window ==
config.Q1_WINDOW`, `"Q2"` when `window == config.Q2_WINDOW` (any other window
is a `ValueError`). Solve order in later stages:
`V_Q1(baseline) = solve_baseline(mmm, Q1_WINDOW, budgets.q1).objective_value`.

### 2.5 `src/mmm_evsi/config.py` (existing file, additive edits only)

New: `REPO_ROOT`, `DATA_DIR`, `ARTIFACTS_DIR`, `MODEL_FILE`, `IDATA_FILE`,
`SUMMARY_FILE` (Section 1.1). Existing constants (`TRAIN_WINDOW`, `Q1_WINDOW`,
`Q2_WINDOW`, `CHANNEL_MAPPING`, `CHANNEL_COLUMNS`, `CONTROL_COLUMNS_PREFIX`,
`ADSTOCK_L_MAX`, `YEARLY_SEASONALITY`, `TARGET_ACCEPT`, `CHAINS`, `DRAWS`,
`NUTS_SAMPLER`, `BOX_PCT`, `DATA_URL`, `RAW_CSV`, `DATE_COLUMN`,
`SALES_COLUMN`) are **unchanged**. No existing value is modified.

### 2.6 `scripts/fit_case_study.py` (new CLI; the user-run fit)

Usage (argparse, all options with defaults):

```
python scripts/fit_case_study.py [--data PATH] [--outdir PATH] [--chains N]
    [--draws N] [--tune N] [--sampler {nutpie,numpyro}] [--target-accept F]
    [--seed N]
```

Defaults: `--data config.RAW_CSV`, `--outdir config.ARTIFACTS_DIR`,
`--chains config.CHAINS (4)`, `--draws config.DRAWS (8000)`,
`--tune 1000`, `--sampler config.NUTS_SAMPLER ("nutpie")`,
`--target-accept config.TARGET_ACCEPT (0.9)`, `--seed 0`.

Behavior, in order:
1. Bootstrap `sys.path` (insert `<repo>/src`) so `import mmm_evsi…` works
   when invoked as a plain script.
2. `df = load_case_study_data(args.data)` (downloads if absent).
3. `train = select_window(df, config.TRAIN_WINDOW)` (183 weeks).
4. Build the case-study model:
   ```python
   controls = [c for c in train.columns
               if c.startswith(config.CONTROL_COLUMNS_PREFIX)]
   mmm = MMM(date_column=config.DATE_COLUMN,
             channel_columns=config.CHANNEL_COLUMNS,
             target_column=config.SALES_COLUMN,
             control_columns=controls,
             adstock=GeometricAdstock(l_max=config.ADSTOCK_L_MAX),
             saturation=LogisticSaturation(),
             yearly_seasonality=config.YEARLY_SEASONALITY)
   ```
5. Fit on the **train window only** (X without the target):
   ```python
   mmm.fit(train.drop(columns=[config.SALES_COLUMN]),
           y=train[config.SALES_COLUMN],
           chains=args.chains, draws=args.draws, tune=args.tune,
           target_accept=args.target_accept, nuts_sampler=args.sampler,
           random_seed=args.seed, progressbar=True)
   ```
   No `compute_log_likelihood` argument (PyMC 6.2 `.sample` has none; the
   default stores no `log_likelihood` — an adequate, lean artifact for Stage
   1; the Stage-2 weighting path computes its own joint quarter likelihoods).
6. `outdir.mkdir(parents=True, exist_ok=True)`; write
   `model_file = outdir / MODEL_FILE.name`, `idata_file = outdir /
   IDATA_FILE.name`:
   `mmm.save(str(model_file))`; `mmm.idata.to_zarr(str(idata_file))`.
7. Write `fit_summary.json` at `outdir / SUMMARY_FILE.name` using
   `sampler_diagnostics(mmm.idata, var_names=[var.name for var in
   mmm.model.free_RVs])` and the schema of Section 1.3 (with paths
   rebased to `outdir`).
8. Print: artifact paths, `n_effective_samples`, `min_ess_bulk`,
   `min_ess_tail`, `max_rhat`, `n_divergences`.

`--draws/--chains` honor the G1.3 target by default (4 × 8,000 = 32,000
≥ 2× the case study's 4,800). `--tune` is not gated (burn-in only).

### 2.7 `pyproject.toml` (edit) + `.gitignore` (edit)

- `pyproject.toml`: add exactly
  ```toml
  [tool.pytest.ini_options]
  pythonpath = ["src"]
  ```
  so the gate tests can `import mmm_evsi.*` in the uninstalled venv.
- `.gitignore`: add `/data/` (Section 1.4).

> **Parallel-phase note (already landed by the orchestrator before the
> parallel W1/W2 phase):** the config.py additions (Section 2.5),
> `pyproject.toml` `pythonpath`, and `.gitignore` `/data/` are committed
> ahead of the parallel phase, so W2 may rely on `import mmm_evsi.config`
> and the new path constants from the start. W2 writes only `tests/*.py`;
> W1 writes only `src/mmm_evsi/{load_mmm,budgets,diagnostics,baseline}.py`
> + `scripts/fit_case_study.py`. The file partition is disjoint.

---

## 3. Data shapes & conventions

### 3.1 Channels

- 7 case-study channels, **keyed by raw `mdsp_*` column names** everywhere in
  code (dicts, boxes, `budget_bounds`, `budgets` DataArray coords):
  `["mdsp_audtr", "mdsp_dm", "mdsp_inst", "mdsp_nsp", "mdsp_on", "mdsp_so",
  "mdsp_vidtr"]` = `config.CHANNEL_COLUMNS` = `sorted(CHANNEL_MAPPING)`.
  Excluded (not in scope): `mdsp_auddig`, `mdsp_viddig`, `mdsp_sem`.
- Human names (`config.CHANNEL_MAPPING`) are **reporting labels only**:
  Direct Mail, Insert, Newspaper, Radio, TV, Social Media, Online Display.
- `budgets` result DataArray: `dims == ("channel",)`, size 7, monetary units.

### 3.2 Date windows (all in `config.py`, concrete stored dates, no check-time
arithmetic)

| Window | Dates | Weeks |
|---|---|---|
| TRAIN | 2014-08-03 … 2018-01-28 | 183 |
| Q1 | 2018-02-04 … 2018-04-29 | 13 |
| Q2 | 2018-05-06 … 2018-07-29 | 13 |

Disjoint, in order, contiguous (Q1 start = train end + 7 d; Q2 start = Q1 end
+ 7 d), and their union equals the dataset span (209 = 183 + 13 + 13 rows;
dataset 2014-08-03 … 2018-07-29, weekly Sundays).

### 3.3 Budgets, boxes, tolerances

- `B_Q1/`B_Q2` = sum over the **7 kept channels** of planned (summed weekly)
  spend in that window. Verified magnitudes (for tests/smoke size sanity):
  `B_Q1 = 16,201,032.73`, `B_Q2 = 14,036,915.49` — tests must not hard-code
  these; compute from the CSV.
- Boxes: `boxes[c] = (0.7 * planned[c], 1.3 * planned[c])` (`BOX_PCT = 0.30`).
- **Feasibility tolerances (normative):**
  - `abs(Σx − B) <= 1e-6` (sum constraint, G1.6);
  - every `x[c] in [lo[c] − 1e-6, hi[c] + 1e-6]` (box constraint, G1.6);
  - planned totals vs. manual sums: `math.isclose(..., rel_tol=1e-9,
    abs_tol=1e-6)` (G1.5).
- `scipy_result.x` sizes 7 (all channels optimized, no fixed channels).

---

## 4. Gate-to-test mapping

All gate tests are **W2-implemented**, live under `tests/`, and follow the
**self-containment rule**: tests that must pass *now* use only the pinned
library API + `mmm_evsi.config` (exists) + the raw CSV. Tests gated on the
fit artifacts import future `mmm_evsi` modules **lazily, inside the
artifact-present branch**, so collection never breaks while they are absent.

| Gate | Test file | Behavior |
|---|---|---|
| G1.1 | `tests/test_fit_artifacts.py` | Skip if `MODEL_FILE`/`IDATA_FILE` missing. Else round-trip: `MMM.load(str(MODEL_FILE))` is an `MMM`; `mmm.idata["posterior"].sizes["chain"] == CHAINS`; `sizes["draw"] == DRAWS`; channel coords == `CHANNEL_COLUMNS`; `xr.open_datatree(IDATA_FILE, engine="zarr")` posterior sizes identical and `adstock_alpha` values `np.allclose` to the loaded model's; `"diverging" in sample_stats`. |
| G1.2 | `tests/test_fit_window.py` | **Runs now.** (a) Stored windows: disjoint, in order, contiguous (test computes the +7 d joins), equal the dataset span, week counts 183/13/13 — computed from `config` constants + `pd.read_csv(config.RAW_CSV)` only (NO `mmm_evsi.load_mmm` import; self-contained). (b) `select_window(df, TRAIN_WINDOW)` selects exactly 183 rows — guarded by `pytest.importorskip("mmm_evsi.load_mmm")` (skips cleanly if the module is not yet present, runs otherwise). (c) Artifact part: `observed_data` date range ⊆ `TRAIN_WINDOW` — **skip if artifacts absent** (guard via `getattr(config, "MODEL_FILE", None)`). |
| G1.3 | `tests/test_posterior_size.py` | (new file; the brief left the name open) Skip if artifacts absent. Else `chain * draw >= 4 * 8000` from `idata["posterior"].sizes`. Also asserts the script defaults satisfy it: `CHAINS * DRAWS >= 32_000` (always runs, pure config). |
| G1.4 | `tests/test_sampler_health.py` | Skip if artifacts absent. Else load `MODEL_FILE` (for `mmm.model.free_RVs`) + `IDATA_FILE`, compute `az.summary(idata, var_names=[var.name for var in mmm.model.free_RVs], fmt="wide")`: `ess_bulk.min() >= 2000`, `ess_tail.min() >= 2000` (pooled), `r_hat.dropna().max() < 1.01`, `n_divergences == 0`. |
| G1.5 | `tests/test_budgets.py` | **Runs now.** From the raw CSV + config: planned per channel per quarter; `planned.keys() == set(CHANNEL_COLUMNS)`; `B_Q1/B_Q2` == sum of planned (tol 1e-6); boxes == `(0.7p, 1.3p)` (tol 1e-6). Imports `mmm_evsi.budgets` (exists at gate-run time — see 6.1 ordering). |
| G1.6 | `tests/test_baseline_solves.py` | **Toy part runs now** (self-contained on `toy_mmm`, no `mmm_evsi.baseline` import). **Real part** skips if artifacts absent; else lazily imports `load_mmm`, `load_budgets`, `solve_baseline` and solves Q1 and Q2. |

### 4.1 `test_baseline_solves.py` details (toy part, runs now)

Reuse the Stage-0 toy plan (`total == 1000.0`):
`PLANNED = {"ch1": 100.0, "ch2": 200.0, "ch3": 150.0, "ch4": 120.0, "ch5":
180.0, "ch6": 140.0, "ch7": 110.0}`, window `("2021-01-03", "2021-03-28")`
(13 Sundays, inside the toy training span 2020-01-05 … 2022-04-17). The toy
part needs no `QuarterBudget` and imports nothing from `mmm_evsi.budgets`
(that module does not exist until W1 implements it, and the toy part must
pass before then) — it calls the pinned optimizer directly with a
test-local plan:

```python
opt = toy_mmm.budget_optimizer("2021-01-03", "2021-03-28")
res = opt.allocate_budget(total_budget=1000.0,
                          budget_bounds={c: (0.7 * PLANNED[c], 1.3 * PLANNED[c])
                                         for c in PLANNED})
```

Assertions:
1. `res.scipy_result.success` and `res.scipy_result.nit >= 1`.
2. `abs(float(res.budgets.sum()) - 1000.0) <= 1e-6`.
3. Per channel `lo[c] - 1e-6 <= x[c] <= hi[c] + 1e-6`.
4. **Optimality smoke**: let `fn = opt._objective_and_grad` (list `[fun, grad]`).
   For the planned allocation and 4 jittered feasible points (see 4.2),
   `float(fn(x_flat)[0]) >= res.scipy_result.fun - 1e-6` (the optimizer's
   minimized value is ≤ any feasible point's).

### 4.2 Feasible perturbation recipe (shared by toy + real smoke)

Start from the planned allocation `p` (inside boxes, sums to `B`). For each
of 4 perturbations pick distinct channel pairs `(i, j)` (channels 0..6
cyclically), draw `u ~ Uniform(0, 1)`,
`dx = min(hi_i - p_i, p_j - lo_j) * u`, set `x_i = p_i + dx`,
`x_j = p_j - dx`, others = `p`. Every `x` stays inside the boxes and sums to
`B` exactly. Evaluate via `opt._objective_and_grad(x_flat)` with
`x_flat = np.array([x[c] for c in opt.budgets_to_optimize.coords["channel"]])`
(raw-name order).

### 4.3 Real part (user-run, after the fit)

```python
def test_real_baseline_solves():
    if not (config.MODEL_FILE.is_dir() and config.IDATA_FILE.is_dir()):
        pytest.skip("Stage 1 artifacts missing — run scripts/fit_case_study.py; see docs/contracts/stage1-fit.md")
    from mmm_evsi.load_mmm import load_mmm, load_budgets      # lazy
    from mmm_evsi.baseline import solve_baseline              # lazy
    mmm, idata = load_mmm()
    budgets = load_budgets()
    for quarter, window, cfg in (("Q1", config.Q1_WINDOW, budgets.q1),
                                 ("Q2", config.Q2_WINDOW, budgets.q2)):
        res = solve_baseline(mmm, window, cfg)
        # success; |sum x − B| <= 1e-6; per-channel boxes ± 1e-6
        # optimality smoke vs. planned + 4 jittered feasible points (4.2)
        # objective_value == -scipy_result.fun
```

---

## 5. Error behavior & edge cases

| Case | Behavior |
|---|---|
| `--data` file absent | `load_case_study_data` downloads from `config.DATA_URL` to that path; `urlretrieve` errors propagate with the URL in the message. |
| Artifact absent (tests) | `pytest.skip("Stage 1 artifacts missing — run scripts/fit_case_study.py; see docs/contracts/stage1-fit.md")` — clean skip, never a failure. |
| Artifact absent (`load_mmm`) | `FileNotFoundError` naming `MODEL_FILE` and pointing at the fit script. |
| `budget_bounds` keyed by human names | `KeyError(msg)` from the pinned optimizer (verified: `KeyError('mdsp_dm')`) — do not "fix"; document in the test docstring that raw keys are mandatory. |
| Window misalignment | `select_window` with `start > end` or empty selection → `ValueError`; G1.2 asserts the constant windows are disjoint/ordered/covering, so a bad edit fails the gate. |
| Non-contiguous window run (real Q2) | Benign `UserWarning` "…not contiguous… cold start (zeros)". Expected; tests use `pytest.warns`/`filterwarnings` to tolerate it. Q1 (contiguous) must NOT warn. |
| `allocate_budget` solver failure | Raises `MinimizeException` (default `return_if_fail=False`). Gates assert `success`, so the raise-path is not exercised. |
| `r_hat` NaN (deterministic vars / zero-spend weeks) | Fixed by restricting to free RVs (`var_names=[var.name for var in mmm.model.free_RVs]`); `n_rhat_nan` is 0 for the free-RV summary. `sampler_diagnostics` still drops NaNs before `max` defensively. |
| DataTree access | Bracket access only (`idata["posterior"]`, `idata["sample_stats"]`). `fit_result` (Dataset) must not be used by tests. |
| Human names requested | `config.CHANNEL_MAPPING` only; any code path treating human names as optimizer keys raises (see budget_bounds row). |

---

## 6. Verification

### 6.1 Gate run (must pass; no fit artifacts required)

**Ordering (parallel-aware):** W1 implements `src/mmm_evsi/*` +
`scripts/fit_case_study.py` while W2 writes `tests/*.py` in parallel. The
pytest commands below are executed **after W1's implementation lands** (so
`mmm_evsi.budgets`, `mmm_evsi.load_mmm`, `mmm_evsi.baseline` import). The
G1.2/G1.5 self-containment rules in Section 4 still hold so the suite never
crashes on a partially-present module (importorskip/getattr guards).

Toy data exists, CSV at `/tmp/mmm_stan.csv`, `mmm_evsi.config` importable:

```bash
cd /home/teemu/repos/mmm-evsi-study
.venv/bin/python -m pytest tests/test_fit_window.py tests/test_budgets.py tests/test_baseline_solves.py -q
```

Expected result: **pass** for G1.2 (config-window + `select_window` checks;
artifact part skips 1), G1.5 (budgets/boxes from data), and the **toy G1.6
part**; the real-G1.6 test **skips** (1 skip). The session-scoped `toy_mmm`
fit (~5–15 s nutpie) runs once. Report per-gate pass/fail for G1.2, G1.5,
toy-G1.6 and the two skips.

### 6.2 USER runs after the fit (unskips G1.1/G1.3/G1.4/real-G1.6)

```bash
cd /home/teemu/repos/mmm-evsi-study
.venv/bin/python scripts/fit_case_study.py          # defaults: 4×8000, nutpie, seed 0 → data/fit/
.venv/bin/python -m pytest tests/ -q                # full suite incl. Stage 0
```

Second command passes only when all `data/fit/` artifacts exist; G1.1/G1.3/
G1.4/real-G1.6 then execute (no skips), G1.4 additionally validates the
`fit_summary.json` numbers (informational: summary `diagnostics` must equal
what `az.summary` yields on the idata).

---

## 7. Open questions / non-gated choices (recorded for the LOG)

- `--tune` default **1000** (burn-in only; not gated by any Stage-1 gate).
- `--sampler` accepts `nutpie|numpyro`; default `nutpie` (CPU — faster for an
  MMM of this size; `--sampler numpyro` uses the JAX/CUDA GPU path, useful
  only for much larger multi-channel MMMs; `target_accept=0.9` is passed for
  both).
- The fit script does **not** run baseline solves (separate step; the user
  triggers them via the G1.6 real test after the fit). No baseline script is
  planned; `solve_baseline` is tested directly.
- Baseline Q2 objective under the unweighted posterior uses the optimizer's
  **cold-start adstock** (Q1 spend is not yet in the optimizer's history) —
  an accepted Stage-1 baseline property; the adstock-aware `OptQ2(y*, a)`
  arrives in Stage 2/3 (PLAN Decision 8).
- `load_budgets` intentionally takes the raw `df` **or** `None` (auto-load),
  **not** idata: the posterior tree contains no Q1/Q2 spend (Section 0).