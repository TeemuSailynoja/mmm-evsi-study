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
# Placeholder until V_Q1(baseline) is known (Stage 1/4); expressed as a
# fraction of the planned quarterly budget for now.
E_MAX_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Fitting (case study settings, 2x draws)
# ---------------------------------------------------------------------------
ADSTOCK_L_MAX = 6
YEARLY_SEASONALITY = 5
TARGET_ACCEPT = 0.9
CHAINS = 4
DRAWS = 8_000  # 2x the case study's 4,000
NUTS_SAMPLER = "nutpie"


@dataclass(frozen=True)
class BudgetConfig:
    """Budget bookkeeping computed from planned spend (filled by load_mmm)."""

    total: float  # planned spend of the full quarter (7 channels)
    residual: float  # B_res = total - sum(fixed) == total (no fixed channels now)
    planned: dict[str, float] = field(default_factory=dict)  # channel -> spend
