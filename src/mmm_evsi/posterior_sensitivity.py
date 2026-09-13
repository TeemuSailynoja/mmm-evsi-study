"""Posterior sensitivity to channel spend scaling.

For each of the top N spenders, scale the quarterly budget by factors
(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0) while preserving the weekly
spend shape. Then measure how the posterior shifts under importance
reweighting.

Key metrics:
- KL divergence (Monte Carlo estimate via PSIS smoothed weights)
- Posterior contraction (ratio of posterior SD to prior SD)
- PSIS k-hat and ESS
- Q2 optimal allocation shift
- Q2 value shift

Designed as a marimo-backed module: the core computation lives in pure
Python functions so it can be called from tests or a notebook.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import spearmanr

from mmm_evsi import config, importance
from mmm_evsi.importance import (
    CompiledResponseEvaluator,
    pool_posterior,
    psis_weights,
    resample_posterior,
    response_mu,
)
from mmm_evsi.experiments import allocation_to_weekly_spend
from mmm_evsi.load_mmm import BudgetPlan, load_budgets, load_case_study_data, load_mmm
from mmm_evsi.optimize_slsqp import q2_expected_response, WeightedSolveJob, run_weighted_solves

logger = logging.getLogger("mmm_evsi.posterior_sensitivity")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PosteriorShift:
    """KL divergence and contraction for one (channel, scale_factor) pair.

    KL is computed via PSIS-smoothed importance weights. The KL at
    scale_factor=1.0 is the "noise floor" (simulation noise only).
    Excess KL (KL - KL_1.0x) represents true posterior shift beyond
    simulation noise; computed on-demand in SensitivityStudy methods.
    """

    channel: str
    scale_factor: float
    khat: float
    weight_ess: float
    degenerate: bool
    # KL divergence estimates (Monte Carlo, via smoothed weights)
    kl_adstock_alpha: float  # mean over channels
    kl_saturation_beta: float
    kl_saturation_lam: float
    kl_y_sigma: float
    # Posterior contraction: posterior SD / prior SD (per parameter)
    # Values < 1 indicate contraction (sharper posterior)
    contraction_adstock_alpha: float
    contraction_saturation_beta: float
    contraction_saturation_lam: float
    contraction_y_sigma: float
    # Per-channel KL for adstock_alpha (7 values)
    kl_adstock_alpha_per_channel: np.ndarray  # (7,)


@dataclass(frozen=True)
class Q2Shift:
    """Q2 optimal allocation and value shift under reweighted posterior."""

    channel: str
    scale_factor: float
    baseline_q2_value: float
    reweighted_q2_value: float
    q2_value_delta: float
    # Per-channel Q2 allocation shift (baseline - reweighted)
    q2_allocation_delta: np.ndarray  # (7,)
    q2_allocation_max_delta: float
    # Optimization diagnostics
    n_accepted: int
    n_skipped: int
    khat: float
    weight_ess: float


@dataclass(frozen=True)
class SensitivityResult:
    """Complete result for one (channel, scale_factor) pair."""

    channel: str
    scale_factor: float
    posterior: PosteriorShift
    q2: Q2Shift | None = None


@dataclass
class SensitivityStudy:
    """Accumulator for the full study."""

    channels: list[str]
    scale_factors: list[float]
    results: list[SensitivityResult] = field(default_factory=list)

    def _excess_kl(self, r: SensitivityResult) -> dict:
        """Compute excess KL for a result by subtracting the 1.0x baseline."""
        baseline = next(
            (rr for rr in self.results
             if rr.channel == r.channel and rr.scale_factor == 1.0),
            None,
        )
        if baseline is None:
            return {
                "excess_kl_adstock_alpha": float("nan"),
                "excess_kl_saturation_beta": float("nan"),
                "excess_kl_saturation_lam": float("nan"),
                "excess_kl_y_sigma": float("nan"),
            }
        return {
            "excess_kl_adstock_alpha": r.posterior.kl_adstock_alpha - baseline.posterior.kl_adstock_alpha,
            "excess_kl_saturation_beta": r.posterior.kl_saturation_beta - baseline.posterior.kl_saturation_beta,
            "excess_kl_saturation_lam": r.posterior.kl_saturation_lam - baseline.posterior.kl_saturation_lam,
            "excess_kl_y_sigma": r.posterior.kl_y_sigma - baseline.posterior.kl_y_sigma,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten results into a DataFrame for plotting."""
        rows = []
        for r in self.results:
            row = {
                "channel": r.channel,
                "scale_factor": r.scale_factor,
                "khat": r.posterior.khat,
                "weight_ess": r.posterior.weight_ess,
                "degenerate": r.posterior.degenerate,
                "kl_adstock_alpha": r.posterior.kl_adstock_alpha,
                "kl_saturation_beta": r.posterior.kl_saturation_beta,
                "kl_saturation_lam": r.posterior.kl_saturation_lam,
                "kl_y_sigma": r.posterior.kl_y_sigma,
                "contraction_adstock_alpha": r.posterior.contraction_adstock_alpha,
                "contraction_saturation_beta": r.posterior.contraction_saturation_beta,
                "contraction_saturation_lam": r.posterior.contraction_saturation_lam,
                "contraction_y_sigma": r.posterior.contraction_y_sigma,
            }
            row.update(self._excess_kl(r))
            if r.q2 is not None:
                row.update({
                    "q2_value_delta": r.q2.q2_value_delta,
                    "q2_allocation_max_delta": r.q2.q2_allocation_max_delta,
                    "n_accepted": r.q2.n_accepted,
                    "n_skipped": r.q2.n_skipped,
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Posterior Sensitivity Study: {len(self.results)} configurations"]
        lines.append(f"Channels: {', '.join(self.channels)}")
        lines.append(f"Scale factors: {self.scale_factors}")
        lines.append("")

        # Per-channel summary
        for ch in self.channels:
            ch_results = [r for r in self.results if r.channel == ch]
            lines.append(f"--- {ch} ---")
            for r in ch_results:
                sf = r.scale_factor
                kl = r.posterior.kl_adstock_alpha
                excess_kl = self._excess_kl(r)["excess_kl_adstock_alpha"]
                ess = r.posterior.weight_ess
                khat = r.posterior.khat
                lines.append(
                    f"  {sf:5.2f}x: KL={kl:.4f}, excess_KL={excess_kl:+.4f}, "
                    f"ESS={ess:.0f}, khat={khat:.3f}"
                )
                if r.q2 is not None:
                    lines.append(
                        f"         Q2 delta={r.q2.q2_value_delta:.2f}, "
                        f"max_alloc_delta={r.q2.q2_allocation_max_delta:.2f}"
                    )
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_kl_divergence(
    smoothed_weights: np.ndarray,
    posterior: xr.Dataset,
    param_names: list[str],
) -> dict[str, float]:
    """Monte Carlo KL divergence estimate via PSIS smoothed weights.

    KL(q || p) ≈ sum_i w_i * log(w_i * N) where w_i are the smoothed
    importance weights and N is the number of draws. This is the
    cross-entropy estimate from the smoothed weights.

    Returns dict of {param_name: mean_KL}.
    """
    N = len(smoothed_weights)
    # KL(q || p) = E_q[log(q/p)] ≈ sum_i w_i * log(w_i * N)
    # where w_i are the normalized importance weights
    log_w = np.log(smoothed_weights)  # already normalized
    log_weights_scaled = log_w + np.log(N)
    kl = float(np.sum(smoothed_weights * log_weights_scaled))

    # Per-parameter KL: use the same weights but compute on each parameter
    # The weights are global, but we report per-parameter KL for interpretability
    per_param_kl = {}
    for name in param_names:
        if name in posterior.data_vars:
            # For multi-dimensional params, report mean KL across dimensions
            # (the weights are the same, so KL is the same for all dims)
            per_param_kl[name] = kl
    return per_param_kl


def _compute_kl_per_channel(
    smoothed_weights: np.ndarray,
    posterior: xr.Dataset,
    param_name: str,
) -> np.ndarray:
    """KL divergence per channel dimension for a multi-dimensional parameter.

    Uses the same importance weights (they're global), but reports the
    per-channel KL for interpretability. Since the weights are the same
    for all dimensions of the same parameter, the KL is identical per
    channel. We report it once per channel for consistency.
    """
    N = len(smoothed_weights)
    log_w = np.log(smoothed_weights)
    log_weights_scaled = log_w + np.log(N)
    kl = float(np.sum(smoothed_weights * log_weights_scaled))
    # Return per-channel (same KL for all channels since weights are global)
    if param_name in posterior.data_vars:
        n_ch = posterior[param_name].sizes.get("channel", 1)
    else:
        n_ch = 1
    return np.full(n_ch, kl)


def _compute_posterior_contraction(
    posterior: xr.Dataset,
    prior: xr.Dataset | None,
    param_names: list[str],
) -> dict[str, float]:
    """Posterior contraction: posterior SD / prior SD.

    Values < 1 indicate the posterior is more concentrated than the prior
    (information gained). Values > 1 indicate the posterior is more diffuse
    (rare, usually indicates model misspecification).

    If no prior is available, returns NaN.
    """
    contraction = {}
    if prior is None:
        for name in param_names:
            contraction[name] = float("nan")
        return contraction

    for name in param_names:
        if name in posterior.data_vars and name in prior.data_vars:
            post_sd = float(np.std(posterior[name].values))
            prior_sd = float(np.std(prior[name].values))
            if prior_sd > 0:
                contraction[name] = post_sd / prior_sd
            else:
                contraction[name] = float("nan")
        else:
            contraction[name] = float("nan")
    return contraction


def _scale_allocation(
    allocation: xr.DataArray,
    channel: str,
    scale_factor: float,
) -> xr.DataArray:
    """Scale one channel's allocation by a factor, preserving shape.

    The allocation is a quarterly budget per channel. Scaling the total
    for one channel while keeping the weekly shape constant means scaling
    the quarterly budget directly.
    """
    vals = allocation.values.copy()
    ch_idx = list(allocation.coords["channel"].values).index(channel)
    vals[ch_idx] *= scale_factor
    return xr.DataArray(vals, coords=allocation.coords, dims=allocation.dims)


def _compute_reweighted_posterior(
    mmm,
    idata: xr.DataTree,
    df: pd.DataFrame,
    q1_cfg,
    allocation: xr.DataArray,
    n_outcomes: int = 3,
    seed: int = 42,
) -> tuple[xr.Dataset, importance.PsisResult, xr.Dataset]:
    """Compute the reweighted posterior for a given allocation.

    Returns (reweighted_posterior, psis_result, original_pooled).
    """
    pooled = pool_posterior(idata["posterior"].to_dataset())
    original_pooled = pooled.copy()

    # Simulate outcomes and compute importance weights
    khats = []
    weight_ess = []
    all_smoothed = []
    all_raw = []

    for i in range(n_outcomes):
        from mmm_evsi.experiments import simulate_quarter

        y_star = simulate_quarter(
            mmm, idata, df, q1_cfg.window, allocation, seed=seed + i
        )
        ell = importance.quarter_log_likelihood(
            mmm, idata, df, q1_cfg.window, allocation, y_star
        )
        psis = psis_weights(ell)
        verdict = importance.apply_khat_policy(psis)

        if not verdict.skipped:
            khats.append(psis.khat)
            weight_ess.append(psis.weight_ess)
            all_smoothed.append(psis.smoothed_log_weights)
            all_raw.append(psis.raw_log_weights)

    if not all_smoothed:
        logger.warning("All outcomes skipped by k-hat policy for %s", allocation)
        avg_khat = float("nan")
        avg_ess = 0.0
        avg_smoothed = np.ones(len(pooled.sample)) / len(pooled.sample)
    else:
        avg_khat = float(np.mean(khats))
        avg_ess = float(np.mean(weight_ess))
        # Average smoothed weights across outcomes
        avg_smoothed = np.exp(logsumexp(np.log(np.exp(all_smoothed)) - np.log(len(all_smoothed)), axis=0))

    # Resample with averaged weights
    reweighted = resample_posterior(
        pooled,
        avg_smoothed,
        n=config.RESAMPLE_DRAWS,
        seed=seed,
    )

    return reweighted, importance.PsisResult(
        smoothed_log_weights=avg_smoothed,
        raw_log_weights=np.mean(all_raw, axis=0) if all_raw else np.zeros(len(pooled.sample)),
        khat=avg_khat,
        weight_ess=avg_ess,
        degenerate=False,
    ), original_pooled


def _run_q2_optimization(
    mmm,
    df: pd.DataFrame,
    reweighted_posterior: xr.Dataset,
    q1_weekly_spend: np.ndarray,
    q2_cfg,
    x0: xr.DataArray | None = None,
) -> tuple[float, np.ndarray]:
    """Run Q2 optimization under reweighted posterior.

    Returns (q2_value, optimal_allocation).
    """
    # Create a WeightedSolveJob and run it
    job = WeightedSolveJob(
        allocation=x0 if x0 is not None else xr.DataArray(
            np.zeros(len(config.CHANNEL_COLUMNS)),
            coords={"channel": config.CHANNEL_COLUMNS},
            dims=["channel"],
        ),
        outcome_index=0,
        posterior=reweighted_posterior,
        q1_weekly_spend=q1_weekly_spend,
        y_star=None,
        khat=0.0,
        weight_ess=1.0,
        x0=x0,
    )
    results = run_weighted_solves(mmm, df, [job], q2_cfg, n_processes=1)
    q2_value = q2_expected_response(
        mmm, reweighted_posterior, df, q1_weekly_spend, results[0].budgets
    )
    return q2_value, results[0].allocation


def run_sensitivity_configuration(
    mmm,
    idata: xr.DataTree,
    df: pd.DataFrame,
    q1_cfg,
    q2_cfg,
    budget_plan: BudgetPlan,
    channel: str,
    scale_factor: float,
    n_outcomes: int = 3,
    seed: int = 42,
    include_q2: bool = True,
) -> SensitivityResult:
    """Run sensitivity analysis for one (channel, scale_factor) configuration.

    Parameters
    ----------
    mmm : pymc_marketing.mmm.BaseMMM
        Fitted MMM model.
    idata : xr.DataTree
        Inference data with posterior.
    df : pd.DataFrame
        Full dataset.
    q1_cfg : QuarterBudget
        Q1 budget configuration.
    q2_cfg : QuarterBudget
        Q2 budget configuration.
    budget_plan : BudgetPlan
        Full budget plan (for baseline allocation).
    channel : str
        Channel to scale (e.g., 'mdsp_dm').
    scale_factor : float
        Scale factor (e.g., 0.5, 1.5).
    n_outcomes : int
        Number of simulated outcomes for importance weighting.
    seed : int
        Random seed.
    include_q2 : bool
        Whether to run Q2 optimization (expensive).

    Returns
    -------
    SensitivityResult
        Complete sensitivity analysis result.
    """
    # 1. Compute baseline allocation (original spend)
    baseline_allocation = xr.DataArray(
        list(q1_cfg.planned.values()),
        coords={"channel": list(q1_cfg.planned.keys())},
        dims=["channel"],
    )

    # 2. Scale the allocation for the target channel
    scaled_allocation = _scale_allocation(baseline_allocation, channel, scale_factor)

    # 3. Compute reweighted posterior
    reweighted, psis, original_pooled = _compute_reweighted_posterior(
        mmm, idata, df, q1_cfg, scaled_allocation, n_outcomes=n_outcomes, seed=seed
    )

    # 4. Compute KL divergence
    param_names = ["adstock_alpha", "saturation_beta", "saturation_lam", "y_sigma"]
    kl_per_param = _compute_kl_divergence(
        psis.smoothed_log_weights, reweighted, param_names
    )

    # 5. Compute per-channel KL for adstock_alpha
    kl_adstock_per_channel = _compute_kl_per_channel(
        psis.smoothed_log_weights, reweighted, "adstock_alpha"
    )

    # 6. Compute posterior contraction
    # Need prior for contraction - use original posterior as proxy if prior unavailable
    prior_ds = None
    try:
        prior_ds = idata["prior"].to_dataset()
    except KeyError:
        logger.debug("No prior group found; contraction will be NaN")

    contraction = _compute_posterior_contraction(
        reweighted, prior_ds, param_names
    )

    # 7. Build PosteriorShift
    posterior_shift = PosteriorShift(
        channel=channel,
        scale_factor=scale_factor,
        khat=psis.khat,
        weight_ess=psis.weight_ess,
        degenerate=psis.degenerate,
        kl_adstock_alpha=kl_per_param.get("adstock_alpha", 0.0),
        kl_saturation_beta=kl_per_param.get("saturation_beta", 0.0),
        kl_saturation_lam=kl_per_param.get("saturation_lam", 0.0),
        kl_y_sigma=kl_per_param.get("y_sigma", 0.0),
        contraction_adstock_alpha=contraction.get("adstock_alpha", float("nan")),
        contraction_saturation_beta=contraction.get("saturation_beta", float("nan")),
        contraction_saturation_lam=contraction.get("saturation_lam", float("nan")),
        contraction_y_sigma=contraction.get("y_sigma", float("nan")),
        kl_adstock_alpha_per_channel=kl_adstock_per_channel,
    )

    # 8. Q2 optimization (optional, expensive)
    q2_shift = None
    if include_q2:
        q1_weekly = allocation_to_weekly_spend(scaled_allocation, 13)
        try:
            q2_value, q2_alloc = _run_q2_optimization(
                mmm, df, reweighted, q1_weekly, q2_cfg, x0=baseline_allocation
            )
            baseline_q2_value = q2_expected_response(
                mmm, original_pooled, df, q1_weekly, q2_cfg
            )
            q2_shift = Q2Shift(
                channel=channel,
                scale_factor=scale_factor,
                baseline_q2_value=baseline_q2_value,
                reweighted_q2_value=q2_value,
                q2_value_delta=q2_value - baseline_q2_value,
                q2_allocation_delta=np.asarray(q2_alloc.values) - np.asarray(baseline_allocation.values),
                q2_allocation_max_delta=float(np.max(np.abs(q2_alloc.values - baseline_allocation.values))),
                n_accepted=1,
                n_skipped=0,
                khat=psis.khat,
                weight_ess=psis.weight_ess,
            )
        except Exception as e:
            logger.warning("Q2 optimization failed for %s @ %.2fx: %s", channel, scale_factor, e)

    return SensitivityResult(
        channel=channel,
        scale_factor=scale_factor,
        posterior=posterior_shift,
        q2=q2_shift,
    )


def run_sensitivity_study(
    mmm,
    idata: xr.DataTree,
    df: pd.DataFrame,
    budget_plan: BudgetPlan,
    channels: list[str] | None = None,
    scale_factors: list[float] | None = None,
    n_outcomes: int = 10,
    seed: int = 42,
    include_q2: bool = False,
) -> SensitivityStudy:
    """Run the full posterior sensitivity study.

    Parameters
    ----------
    mmm : pymc_marketing.mmm.BaseMMM
        Fitted MMM model.
    idata : xr.DataTree
        Inference data with posterior.
    df : pd.DataFrame
        Full dataset.
    budget_plan : BudgetPlan
        Full budget plan.
    channels : list[str] | None
        Channels to study. Defaults to top 3 Q1 spenders.
    scale_factors : list[float] | None
        Scale factors. Defaults to [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0].
    n_outcomes : int
        Number of simulated outcomes per configuration. Higher = more stable
        KL estimates. Default 10 for study; use 2-3 for quick tests.
    seed : int
        Random seed.
    include_q2 : bool
        Whether to run Q2 optimization (expensive, ~30s per config).
        Default False for quick KL-only analysis.

    Returns
    -------
    SensitivityStudy
        Complete study results with excess KL computed.
    """
    if scale_factors is None:
        scale_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    if channels is None:
        # Top 3 Q1 spenders
        top3 = sorted(budget_plan.q1.planned.items(), key=lambda x: x[1], reverse=True)[:3]
        channels = [ch for ch, _ in top3]

    study = SensitivityStudy(channels=channels, scale_factors=scale_factors)

    for ch in channels:
        for sf in scale_factors:
            logger.info("Running %s @ %.2fx", ch, sf)
            result = run_sensitivity_configuration(
                mmm=mmm,
                idata=idata,
                df=df,
                q1_cfg=budget_plan.q1,
                q2_cfg=budget_plan.q2,
                budget_plan=budget_plan,
                channel=ch,
                scale_factor=sf,
                n_outcomes=n_outcomes,
                seed=seed,
                include_q2=include_q2,
            )
            study.results.append(result)
            logger.info(
                "  KL=%.4f, ESS=%.0f, khat=%.3f",
                result.posterior.kl_adstock_alpha,
                result.posterior.weight_ess,
                result.posterior.khat,
            )

    return study
