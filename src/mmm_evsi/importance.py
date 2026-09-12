"""Importance-sampling machinery for the EVSI pipeline (Stage 2).

Pure-numpy/xarray weighting + PSIS + resampling, plus the MMM response path
(``quarter_log_likelihood``) shared with ``experiments.simulate_quarter`` and
``optimize_slsqp.q2_expected_response``.

Pinned PSIS convention (verified against arviz-stats 1.3.2):
``da.azstats.psislw(dim="sample")`` takes NEGATED log-weights ``-ell`` and
returns ``(smoothed_log_weights, khat)`` where ``smoothed_log_weights`` are
ALREADY normalized (exp-sum == 1) and in the positive orientation (larger =
more weight). Resampling weights are ``exp(smoothed_log_weights)``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import arviz as az  # noqa: F401  (registers the DataArray ``.azstats`` accessor)
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp

from mmm_evsi import config

logger = logging.getLogger("mmm_evsi.importance")


class WeightError(ValueError):
    """Raised when a weight computation violates a pinned invariant."""


# ---------------------------------------------------------------------------
# Pure pooling / weighting
# ---------------------------------------------------------------------------


def pool_posterior(posterior: xr.Dataset | xr.DataTree) -> xr.Dataset:
    """Stack (chain, draw) -> (sample,) (chain-major). Accepts a Dataset or
    a DataTree posterior node (``idata["posterior"]``)."""
    if isinstance(posterior, xr.DataTree):
        posterior = posterior.to_dataset()
    if "sample" in posterior.dims:
        raise ValueError("'sample' dim already present; posterior is already pooled")
    return posterior.stack(sample=("chain", "draw"))


def normalized_log_weights(ell: np.ndarray) -> np.ndarray:
    """ell - logsumexp(ell) over the last axis; ell shape (S,)."""
    ell = np.asarray(ell, dtype=float)
    return ell - logsumexp(ell)


@dataclass(frozen=True)
class PsisResult:
    smoothed_log_weights: np.ndarray  # (S,) exp-sum == 1 (atol 1e-6)
    raw_log_weights: np.ndarray  # (S,) = normalized_log_weights(ell)
    khat: float  # one value per simulated quarter
    weight_ess: float  # 1 / sum(exp(smoothed)^2)
    degenerate: bool  # True when the tail was exactly flat


def psis_weights(
    ell: np.ndarray | xr.DataArray,
    r_eff: float = 1.0,
    min_draws: int = config.MIN_POOLED_DRAWS,
) -> PsisResult:
    """PSIS-smoothed log-weights + k-hat for one simulated quarter.

    ``ell``: (S,) pooled array, or DataArray with dims ("chain", "draw").
    """
    if isinstance(ell, xr.DataArray):
        if "sample" in ell.dims:
            ell = ell.values
        else:
            ell = ell.stack(sample=("chain", "draw")).values
    ell = np.asarray(ell, dtype=float).reshape(-1)
    s = ell.shape[0]
    if s < min_draws:
        raise WeightError(
            f"need >= {min_draws} pooled draws for PSIS, got {s}"
        )

    raw = normalized_log_weights(ell)
    degenerate = bool(np.ptp(ell) <= 1e-12 * max(1.0, abs(np.mean(ell))))

    if degenerate:
        smoothed = raw
        khat = 0.0
    else:
        neg = xr.DataArray(-ell, dims="sample")
        smoothed_da, khat_da = neg.azstats.psislw(dim="sample")
        smoothed = np.asarray(smoothed_da).reshape(s)
        khat = float(np.asarray(khat_da).ravel()[0])

    ess = s if degenerate else 1.0 / float(np.sum(np.exp(smoothed) ** 2))
    return PsisResult(
        smoothed_log_weights=smoothed,
        raw_log_weights=raw,
        khat=khat,
        weight_ess=ess,
        degenerate=degenerate,
    )


@dataclass(frozen=True)
class KhatVerdict:
    skipped: bool
    khat: float
    reason: str


def apply_khat_policy(
    psis: PsisResult, threshold: float = config.K_HAT_THRESHOLD
) -> KhatVerdict:
    """Skip only when k-hat is non-finite or exceeds the threshold.

    A NEGATIVE k-hat (light tail) is well-behaved and is accepted; the
    heavy-tail cutoff is one-sided (k-hat > 0.7), matching the official
    ROAS-experimentation notebook.
    """
    khat = psis.khat
    if not np.isfinite(khat) or khat > threshold:
        reason = (
            "non-finite khat: skip simulated quarter"
            if not np.isfinite(khat)
            else f"khat>{threshold}: skip simulated quarter"
        )
        verdict = KhatVerdict(skipped=True, khat=khat, reason=reason)
        logger.warning("khat=%s -> %s", khat, reason)
        return verdict
    return KhatVerdict(
        skipped=False, khat=khat, reason=f"khat<={threshold}: accept"
    )


def resample_posterior(
    posterior: xr.Dataset,
    probabilities: np.ndarray,
    n: int | None = None,
    seed: int = config.RESAMPLE_SEED,
) -> xr.Dataset:
    """Multinomial-resample a pooled posterior (dims ("sample",)) -> (chain=1,
    draw=n) Dataset suitable for ``BudgetOptimizer.set_posterior``."""
    if "sample" not in posterior.dims:
        # Accept an unpooled (chain, draw) posterior: flatten chain-major.
        posterior = posterior.stack(sample=("chain", "draw"))
    s = posterior.sizes["sample"]
    n = s if n is None else n
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.shape[0] != s:
        raise WeightError(f"probabilities must be shape ({s},), got {p.shape}")
    if np.any(p < 0) or abs(p.sum() - 1.0) > 1e-6:
        raise WeightError("probabilities must be non-negative and sum to 1")
    rng = np.random.default_rng(seed)
    indices = rng.choice(s, size=n, replace=True, p=p)
    out = posterior.isel(sample=indices)
    if "sample" in out.coords:  # drop the stacked MultiIndex (and its levels)
        out = out.drop_vars(["sample", "chain", "draw"], errors="ignore")
    out = out.rename({"sample": "draw"})
    out = out.assign_coords(draw=np.arange(n))
    extra = [d for d in out.dims if d not in ("chain", "draw")]
    out = out.expand_dims(chain=[0], axis=0).transpose("chain", "draw", *extra)
    out.attrs["resampled_indices"] = indices
    return out


# ---------------------------------------------------------------------------
# MMM response path (shared with experiments / optimize_slsqp)
# ---------------------------------------------------------------------------


def _extract_mu_graph(mmm, posterior, df, window_start, weekly_spend, carry_weekly):
    """Set the window's shared data and return (mu_graph, l_max, target_scale).

    Shared helper for ``response_mu`` (evaluates) and
    ``optimize_slsqp._compile_q2_objective`` (compiles). The returned graph is
    vectorized over the posterior draws (sample dim first).
    """
    from pymc_marketing.pytensor_utils import extract_response_distribution

    channels = list(mmm.channel_columns)
    date_col = mmm.date_column
    control_cols = list(mmm.control_columns or [])
    l_max = int(mmm.adstock.l_max)
    n_ch = len(channels)

    weekly_spend = np.atleast_2d(np.asarray(weekly_spend, dtype=float))
    if weekly_spend.shape != (13, n_ch):
        raise WeightError(f"weekly_spend must be (13, {n_ch}), got {weekly_spend.shape}")
    if carry_weekly is None:
        carry_weekly = np.zeros((l_max, n_ch))
    carry_weekly = np.atleast_2d(np.asarray(carry_weekly, dtype=float))
    if carry_weekly.shape != (l_max, n_ch):
        raise WeightError(
            f"carry_weekly must be ({l_max}, {n_ch}), got {carry_weekly.shape}"
        )

    window_dates = pd.date_range(window_start, periods=13, freq="7D")
    carry_dates = pd.date_range(
        end=pd.Timestamp(window_start) - pd.Timedelta(days=7),
        periods=l_max,
        freq="7D",
    )
    all_dates = carry_dates.append(window_dates)

    dfd = df.set_index(date_col)
    spend = np.vstack([carry_weekly, weekly_spend])  # (l_max+13, n_ch)
    rows: dict = {date_col: all_dates}
    for j, ch in enumerate(channels):
        rows[ch] = spend[:, j]
    for c in control_cols:
        if c in dfd.columns and all_dates.isin(dfd.index).all():
            rows[c] = dfd.loc[all_dates, c].to_numpy()
        else:
            rows[c] = np.zeros(len(all_dates))
    X = pd.DataFrame(rows)

    ds = mmm._posterior_predictive_data_transformation(
        X, include_last_observations=False
    )
    mmm._set_xarray_data(ds, model=mmm.model, clone_model=False)

    mu_node = mmm.model.named_vars["y"].owner.inputs[1]
    idata = xr.DataTree.from_dict({"posterior": posterior})
    mu_graph = extract_response_distribution(mmm.model, idata, mu_node)
    target_scale = float(np.asarray(mmm.model.named_vars["target_scale"].get_value()))
    return mu_graph, l_max, target_scale


def response_mu(
    mmm,
    posterior: xr.Dataset,
    df: pd.DataFrame,
    window_start: str | pd.Timestamp,
    weekly_spend: np.ndarray,
    carry_weekly: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-draw original-scale response for a 13-week window.

    Returns ``(mu_orig, sigma_orig)``: ``mu_orig`` (S, 13), ``sigma_orig``
    (S,).
    """
    mu_graph, l_max, target_scale = _extract_mu_graph(
        mmm, posterior, df, window_start, weekly_spend, carry_weekly
    )
    mu_scaled = np.asarray(mu_graph.eval())  # (S, l_max+13)
    mu_orig = mu_scaled[:, l_max:] * target_scale  # (S, 13)

    sigma_scaled = (
        posterior["y_sigma"].stack(sample=("chain", "draw")).values
        if "chain" in posterior.dims
        else posterior["y_sigma"].values
    )
    sigma_orig = sigma_scaled * target_scale  # (S,)
    return mu_orig, sigma_orig


def quarter_log_likelihood(
    mmm,
    idata: xr.DataTree,
    df: pd.DataFrame,
    window: tuple[str, str],
    allocation: xr.DataArray,
    y_star: np.ndarray,
) -> np.ndarray:
    """Joint log-likelihood of one simulated quarter: (S,) ``ell_s``.

    ``ell_s = sum_t log Normal(y*_t | mu_t(theta_s, a), sigma_s)`` with the
    adstock state carried in from the ``l_max`` training weeks before
    ``window`` (Q1 is contiguous with the train window).
    """
    y_star = np.asarray(y_star, dtype=float)
    if y_star.shape != (13,):
        raise WeightError(f"y_star must have shape (13,), got {y_star.shape}")

    from mmm_evsi.experiments import allocation_to_weekly_spend

    weekly = allocation_to_weekly_spend(allocation, n_weeks=13)
    l_max = int(mmm.adstock.l_max)
    train_tail_dates = pd.date_range(
        end=pd.Timestamp(window[0]) - pd.Timedelta(days=7), periods=l_max, freq="7D"
    )
    dfd = df.set_index(mmm.date_column)
    carry = np.column_stack(
        [dfd.loc[train_tail_dates, c].to_numpy() for c in mmm.channel_columns]
    )  # (l_max, n_ch)

    posterior = idata["posterior"].to_dataset()
    mu_orig, sigma_orig = response_mu(
        mmm, posterior, df, window[0], weekly, carry_weekly=carry
    )
    sigma2 = sigma_orig**2  # (S,)
    diff = y_star[None, :] - mu_orig  # (S, 13)
    ell = -0.5 * 13 * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(
        diff**2, axis=1
    ) / sigma2
    return ell


# ---------------------------------------------------------------------------
# One-allocation evaluation loop (G2.8 / G2.9 / G2.3c)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationEvaluation:
    allocation: xr.DataArray
    utilities: np.ndarray  # (n_accepted,) OptQ2 per accepted quarter/seed
    khats: np.ndarray  # (n_outcomes,) float("nan") for skipped
    weight_ess: np.ndarray  # (n_outcomes,)
    skipped_indices: np.ndarray  # outcome ordinals skipped by the k-hat policy
    utility: float  # mean over accepted; NaN if all skipped
    n_outcomes: int
    n_skipped: int


def evaluate_allocation(
    mmm,
    idata: xr.DataTree,
    df: pd.DataFrame,
    allocation: xr.DataArray,
    q1_cfg,
    q2_cfg,
    n_outcomes: int = config.N_SIM_OUTCOMES,
    seed: int = config.RESAMPLE_SEED,
    x0_warm: xr.DataArray | None = None,
    n_processes: int | None = None,
    resample_seeds: Sequence[int] | None = None,
) -> AllocationEvaluation:
    """Stage-2 loop for one allocation: simulate → weight → resample → solve.

    With ``resample_seeds`` given, exactly ONE simulated outcome is used and
    the loop iterates over the resample seeds (same ℓ/k-hat, different
    resample RNG). Otherwise one outcome per ``n_outcomes``.
    """
    from mmm_evsi.experiments import allocation_to_weekly_spend, simulate_quarter
    from mmm_evsi.optimize_slsqp import WeightedSolveJob, run_weighted_solves

    pooled = pool_posterior(idata["posterior"].to_dataset())
    q1_weekly = allocation_to_weekly_spend(allocation, 13)

    jobs = []
    khats: list[float] = []
    weight_ess: list[float] = []
    skipped_indices: list[int] = []

    def _process(y_star, idx, rseed):
        ell = quarter_log_likelihood(
            mmm, idata, df, q1_cfg.window, allocation, y_star
        )
        psis = psis_weights(ell)
        verdict = apply_khat_policy(psis)
        if verdict.skipped:
            khats.append(float("nan"))
            weight_ess.append(psis.weight_ess)
            skipped_indices.append(idx)
            return
        khats.append(psis.khat)
        weight_ess.append(psis.weight_ess)
        posterior_r = resample_posterior(
            pooled,
            np.exp(psis.smoothed_log_weights),
            n=config.RESAMPLE_DRAWS,
            seed=rseed,
        )
        jobs.append(
            WeightedSolveJob(
                allocation=allocation,
                outcome_index=idx,
                posterior=posterior_r,
                q1_weekly_spend=q1_weekly,
                y_star=y_star,
                khat=psis.khat,
                weight_ess=psis.weight_ess,
                x0=x0_warm,
            )
        )

    if resample_seeds is not None:
        y_star = simulate_quarter(
            mmm, idata, df, q1_cfg.window, allocation, seed=seed
        )
        for k, rseed in enumerate(resample_seeds):
            _process(y_star, k, rseed)
    else:
        for i in range(n_outcomes):
            y_star = simulate_quarter(
                mmm, idata, df, q1_cfg.window, allocation, seed=seed + i
            )
            _process(y_star, i, seed + i)

    results = run_weighted_solves(mmm, df, jobs, q2_cfg, n_processes=n_processes)
    # Utilities are recorded in CORRECT sales units via the validated response
    # path (the stock optimizer's -scipy_result.fun is ~6.2-6.4x inflated by an
    # optimization-model un-scaling quirk; see docs/LOG.md). Deterministic.
    from mmm_evsi.optimize_slsqp import q2_expected_response

    utilities = np.array(
        [
            q2_expected_response(
                mmm, job.posterior, df, job.q1_weekly_spend, r.budgets
            )
            for job, r in zip(jobs, results)
        ],
        dtype=float,
    )
    if utilities.size == 0:
        raise WeightError("all simulated quarters skipped by the k-hat policy")

    return AllocationEvaluation(
        allocation=allocation,
        utilities=utilities,
        khats=np.array(khats, dtype=float),
        weight_ess=np.array(weight_ess, dtype=float),
        skipped_indices=np.array(skipped_indices, dtype=int),
        utility=float(utilities.mean()),
        n_outcomes=n_outcomes,
        n_skipped=len(skipped_indices),
    )
