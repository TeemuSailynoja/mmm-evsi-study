"""Central configuration for the mmm-evsi study.

All date windows, channel selections, and budget parameters live here so the
rest of the pipeline has a single source of truth (see PLAN.md Stage 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/sibylhe/mmm_stan/main/data.csv"
RAW_CSV = Path("/tmp/mmm_stan.csv")

# ---------------------------------------------------------------------------
# Repo paths & artifact layout (single source of truth; CWD-agnostic)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "fit"
MODEL_FILE = ARTIFACTS_DIR / "case_study_mmm.zarr"  # directory (zarr store)
IDATA_FILE = ARTIFACTS_DIR / "case_study_idata.zarr"  # directory (zarr store)
SUMMARY_FILE = ARTIFACTS_DIR / "fit_summary.json"

DATE_COLUMN = "wk_strt_dt"
SALES_COLUMN = "sales"

# 7 media channels (case study mapping mdsp_* -> human name)
CHANNEL_MAPPING = {
    "mdsp_dm": "Direct Mail",
    "mdsp_inst": "Insert",
    "mdsp_nsp": "Newspaper",
    "mdsp_audtr": "Radio",
    "mdsp_vidtr": "TV",
    "mdsp_so": "Social Media",
    "mdsp_on": "Online Display",
}
CHANNEL_COLUMNS = sorted(CHANNEL_MAPPING)
CONTROL_COLUMNS_PREFIX = "hldy_"

# ---------------------------------------------------------------------------
# Explicit, disjoint date windows (weekly, Sundays)
# ---------------------------------------------------------------------------
TRAIN_WINDOW = ("2014-08-03", "2018-01-28")  # 183 weeks
Q1_WINDOW = ("2018-02-04", "2018-04-29")  # 13 weeks: observation/experiment
Q2_WINDOW = ("2018-05-06", "2018-07-29")  # 13 weeks: decision

# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------
# Channel-level quarterly budgets over the FULL 7-channel list (Decision:
# Option A). Weekly (per-period) adjustment is deferred to a later phase.
OPTIMIZE_CHANNELS = sorted(CHANNEL_MAPPING.values())
FIXED_CHANNELS: list[str] = []

# ---------------------------------------------------------------------------
# Budget constraints
# ---------------------------------------------------------------------------
BOX_PCT = 0.30  # per channel/quarter: [1-BOX_PCT, 1+BOX_PCT] * planned spend
LAMBDA = 1.0  # $1 Q1 revenue loss <=> $1 expected Q2 gain
# Experimentation budget: cap on the lambda-weighted Q1 revenue loss.
# Expressed as a fraction of the planned quarterly budget.
E_MAX_FRACTION = 0.10  # 10% of total quarterly budget

# ---------------------------------------------------------------------------
# Fitting (case study settings, 2x draws)
# ---------------------------------------------------------------------------
ADSTOCK_L_MAX = 6
YEARLY_SEASONALITY = 5
TARGET_ACCEPT = 0.9
CHAINS = 4
DRAWS = 8_000  # 2x the case study's 4,000
NUTS_SAMPLER = "nutpie"  # CPU: fastest for an MMM of this size (7 channels,
                         # ~183 weekly rows); GPU (numpyro) only helps at
                         # much larger multi-channel scales

# ---------------------------------------------------------------------------
# Stage 2 — importance sampling parameters (additive; see
# docs/contracts/stage2-weighted.md)
# ---------------------------------------------------------------------------
LOAD_POSTERIOR_INTO_MEMORY = True  # if True, load posterior into memory at startup; avoids zarr I/O bottleneck
K_HAT_THRESHOLD = 0.7  # k-hat > threshold => skip the simulated quarter (G2.3)
N_SIM_OUTCOMES = 5  # default simulated quarters per allocation (G2.9 smoke);
                    # gates/sweeps pass larger values explicitly
RESAMPLE_SEED = 0  # default seeded RNG for the multinomial resample
MIN_POOLED_DRAWS = 25  # psislw tail-fit floor (n_draws_tail >= 5); asserted
RESAMPLE_DRAWS = 2_000  # resample size for the Q2 solve (the 32k original
                        # posterior is for IS coverage; 2k is enough for the
                        # weighted optimization)

# ---------------------------------------------------------------------------
# Stage 3 — Bayesian optimization parameters
# ---------------------------------------------------------------------------
BO_N_EVALUATIONS = 100  # total function evaluations (LHS + BO iterations)
BO_N_INITIAL = 20  # LHS initial design size
BO_N_OUTCOMES = 10  # simulated outcomes per proposal evaluation
BO_TOL = 1e-4  # relative improvement threshold for early stopping
BO_PATIENCE = 10  # consecutive non-improvements to trigger early stop
BO_N_DRAW_VQ1 = 100  # posterior draws for fast V_Q1 computation in BO loop


@dataclass(frozen=True)
class BudgetConfig:
    """Budget bookkeeping computed from planned spend (filled by load_mmm)."""

    total: float  # planned spend of the full quarter (7 channels)
    residual: float  # B_res = total - sum(fixed) == total (no fixed channels now)
    planned: dict[str, float] = field(default_factory=dict)  # channel -> spend
