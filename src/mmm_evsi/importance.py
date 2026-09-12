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

# Cache the pooled posterior after the first call.  Since the posterior never
# changes during the BO/importance loop, this eliminates the zarr I/O that
# would otherwise occur on every call (10 outcomes × 6 dimensions × ~5 s).
_pooled_posterior_cache: xr.Dataset | None = None

# Cache the unpooled (chain, draw) posterior after the first
# ``unpool_posterior()`` call.  Each ``resample_posterior()`` invocation
# calls ``unpool_posterior()`` which iterates over data_vars and calls
# ``.values`` on zarr-backed arrays.  Since the posterior never changes
# during the BO/importance loop, we can safely cache the result.
_unpooled_posterior_cache: xr.Dataset | None = None


def pool_posterior(posterior: xr.Dataset | xr.DataTree) -> xr.Dataset:
    """Stack (chain, draw) -> (sample,) (chain-major). Accepts a Dataset or
    a DataTree posterior node (``idata["posterior"]``). The stacked
    MultiIndex is dropped so ``sample`` is a PLAIN positional dim, and the
    original chain/draw counts are stashed in attrs so ``resample_posterior``
    can reconstruct the (chain, draw) layout.

    The result is cached after the first call since the posterior never
    changes during the BO loop.
    """
    global _pooled_posterior_cache
    if _pooled_posterior_cache is None:
        _pooled_posterior_cache = _pool_posterior_impl(posterior)
    return _pooled_posterior_cache


def _pool_posterior_impl(posterior: xr.Dataset | xr.DataTree) -> xr.Dataset:
    """Implementation of pool_posterior without caching."""
    if isinstance(posterior, xr.DataTree):
        posterior = posterior.to_dataset()
    if "sample" in posterior.dims:
        raise ValueError("'sample' dim already present; posterior is already pooled")
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    stacked = posterior.stack(sample=("chain", "draw"))
    stacked = stacked.drop_vars(["sample", "chain", "draw"], errors="ignore")
    stacked = stacked.assign_coords(sample=np.arange(stacked.sizes["sample"]))
    stacked.attrs["pooled_n_chains"] = n_chains
    stacked.attrs["pooled_n_draws"] = n_draws
    return stacked


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


def unpool_posterior(pooled: xr.Dataset) -> xr.Dataset:
    """Reconstruct the (chain, draw) layout from a ``pool_posterior``
    output (its attrs carry the original chain/draw counts). The flat
    sample axis is chain-major.

    The result is cached after the first call since the posterior never
    changes during the BO loop.
    """
    global _unpooled_posterior_cache
    if _unpooled_posterior_cache is None:
        _unpooled_posterior_cache = _unpool_posterior_impl(pooled)
    return _unpooled_posterior_cache


def _unpool_posterior_impl(pooled: xr.Dataset) -> xr.Dataset:
    """Implementation of unpool_posterior without caching."""
    n_chains = pooled.attrs.get("pooled_n_chains")
    n_draws = pooled.attrs.get("pooled_n_draws")
    if not n_chains:
        raise WeightError(
            "pooled posterior lacks pooled_n_chains/pooled_n_draws attrs; "
            "produce it via pool_posterior"
        )
    ds = xr.Dataset()
    for name, da in pooled.data_vars.items():
        # Move the sample axis to the front, then split chain-major into
        # (chain, draw) — 'sample' may sit at any position.
        axis = da.dims.index("sample")
        arr = np.moveaxis(da.values, axis, 0)  # (S, ...)
        rest = [d for d in da.dims if d != "sample"]
        shape2 = (n_chains, n_draws) + arr.shape[1:]
        ds[name] = xr.DataArray(
            arr.reshape(shape2),
            dims=("chain", "draw", *rest),
            coords={d: da.coords[d].values for d in rest if d in da.coords},
        )
    return ds


def resample_posterior(
    posterior: xr.Dataset,
    probabilities: np.ndarray,
    n: int | None = None,
    seed: int = config.RESAMPLE_SEED,
) -> xr.Dataset:
    """Weighted resample — the official ROAS-experimentation notebook's
    function (chain_idx/draw_idx indexing), with one change: the resampled
    draws are RESHAPED to ``(n_chains, n // n_chains)`` so the optimizer
    receives a multi-chain posterior (RESAMPLE_DRAWS=2000 -> 4 x 500).

    Accepts either the unpooled ``(chain, draw)`` posterior or a pooled
    ``(sample,)`` one from ``pool_posterior`` (its attrs carry the original
    chain/draw counts; the flat sample axis is chain-major, matching the
    flattened ``probabilities``).
    """
    rng = np.random.default_rng(seed)
    if "sample" in posterior.dims:
        posterior = unpool_posterior(posterior)
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    s = n_chains * n_draws

    n = s if n is None else n
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.shape[0] != s:
        raise WeightError(f"probabilities must have {s} entries, got {p.shape[0]}")
    if np.any(p < 0) or abs(p.sum() - 1.0) > 1e-6:
        raise WeightError("probabilities must be non-negative and sum to 1")
    if n % n_chains != 0:
        raise WeightError(
            f"num_samples {n} must be a multiple of n_chains {n_chains}"
        )

    idx = rng.choice(s, size=n, replace=True, p=p)
    chain_idx = idx // n_draws
    draw_idx = idx % n_draws

    resampled_vars = {}
    for var in posterior.data_vars:
        da = posterior[var]
        extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
        indexed = da.values[chain_idx, draw_idx]
        reshaped = indexed.reshape((n_chains, n // n_chains, *indexed.shape[1:]))
        coords = {"chain": range(n_chains), "draw": range(n // n_chains)}
        for d in extra_dims:
            coords[d] = da.coords[d].values
        resampled_vars[var] = xr.DataArray(
            reshaped, dims=["chain", "draw", *extra_dims], coords=coords
        )
    out = xr.Dataset(resampled_vars)
    out.attrs["resampled_indices"] = idx
    return out


# ---------------------------------------------------------------------------
# Compiled response evaluator (cache graph, swap data)
# ---------------------------------------------------------------------------

import pytensor
import pytensor.tensor as pt


class CompiledResponseEvaluator:
    """Compile the MMM response graph ONCE, then swap posterior + spend data.

    This is the key speedup: ``extract_response_distribution`` recompiles the
    full PyTensor graph (~20 s) when called with a new posterior.  By binding
    posterior draws through a ``SharedPosterior`` and spend data through
    shared PyTensor variables, we can re-evaluate the compiled graph in
    milliseconds for every subsequent call.

    Parameters
    ----------
    mmm : pymc_marketing.mmm.BaseMMM
        Fitted MMM instance.
    df : pd.DataFrame
        Full dataset (used to look up control variables by date).
    window_start : str | pd.Timestamp
        First date of the 13-week window.
    l_max : int
        Adstock memory length.
    n_channels : int
        Number of media channels.
    """

    def __init__(
        self,
        mmm,
        df: pd.DataFrame,
        window_start: str | pd.Timestamp,
        l_max: int,
        n_channels: int,
    ):
        self.mmm = mmm
        self.df = df
        self.window_start = pd.Timestamp(window_start)
        self.l_max = l_max
        self.n_channels = n_channels
        self.date_col = mmm.date_column
        self.channels = list(mmm.channel_columns)
        self.control_cols = list(mmm.control_columns or [])
        self.target_scale = float(
            np.asarray(mmm.model.named_vars["target_scale"].get_value())
        )

        # Build the date index for the window (carry + 13 weeks)
        carry_dates = pd.date_range(
            end=self.window_start - pd.Timedelta(days=7),
            periods=l_max,
            freq="7D",
        )
        self.window_dates = pd.date_range(
            self.window_start, periods=13, freq="7D"
        )
        self.all_dates = carry_dates.append(self.window_dates)

        # Shared variables for spend data: (l_max + 13, n_channels)
        self._spend_shared = pytensor.shared(
            np.zeros((l_max + 13, n_channels), dtype=float),
            name="evsi_spend",
        )

        # Shared variables for control data: (l_max + 13, n_controls)
        n_ctrl = len(self.control_cols)
        self._ctrl_shared = pytensor.shared(
            np.zeros((l_max + 13, n_ctrl), dtype=float),
            name="evsi_controls",
        ) if n_ctrl > 0 else None

        # Build the design matrix from spend + controls
        self._build_design_matrix()

        # Extract and compile the response graph ONCE
        # Use initial zero arrays for DataFrame construction (shared vars can't go in DF)
        channels = self.channels
        spend_array = self._spend_shared.get_value()  # (l_max+13, n_ch)
        rows = {self.date_col: self.all_dates}
        for j, ch in enumerate(channels):
            rows[ch] = spend_array[:, j]
        if self._ctrl_shared is not None:
            ctrl_array = self._ctrl_shared.get_value()  # (l_max+13, n_ctrl)
            for k, c in enumerate(self.control_cols):
                rows[c] = ctrl_array[:, k]
        X = pd.DataFrame(rows)

        ds = mmm._posterior_predictive_data_transformation(
            X, include_last_observations=False
        )
        mmm._set_xarray_data(ds, model=mmm.model, clone_model=False)

        # CRITICAL: Replace the model's channel_data pm.Data (which is a
        # pytensor shared variable) with our shared variable so that
        # set_spend() updates the data that the compiled graph reads.
        channel_data_name = 'channel_data'
        old_channel_data = mmm.model.named_vars.get(channel_data_name)
        if old_channel_data is not None:
            # Set initial value on our shared variable
            self._spend_shared.set_value(old_channel_data.get_value())
            # Replace in model's named_vars so the compiled graph reads from ours
            mmm.model.named_vars[channel_data_name] = self._spend_shared

        # SharedPosterior will be set up on first call to set_posterior()
        self._shared_posterior = None
        self._mu_graph = None
        self._sigma_graph = None
        self._compiled = False

        # Store the idata for initial compilation (same pattern as BudgetOptimizer)
        # We'll use the actual posterior from the model's idata
        if hasattr(mmm, 'idata') and mmm.idata is not None:
            self._idata = mmm.idata
        else:
            # Fallback: create minimal idata from model
            sample_posterior = {var: xr.DataArray(np.zeros((1, 1)), dims=["chain", "draw"])
                               for var in mmm.model.named_vars}
            dummy_ds = xr.Dataset(sample_posterior)
            self._idata = xr.DataTree.from_dict({"posterior": dummy_ds})

    def _build_design_matrix(self):
        """Build the design matrix for control variables by date."""
        if not self.control_cols:
            return
        dfd = self.df.set_index(self.date_col)
        ctrl_values = []
        for c in self.control_cols:
            if c in dfd.columns and self.all_dates.isin(dfd.index).all():
                values = dfd.loc[self.all_dates, c].to_numpy()
                ctrl_values.append(values.reshape(-1, 1) if len(values.shape) == 1 else values)
        if ctrl_values:
            self._ctrl_shared.set_value(np.hstack(ctrl_values))

    def set_spend(self, carry_weekly: np.ndarray, weekly_spend: np.ndarray):
        """Swap spend data without recompiling.

        Parameters
        ----------
        carry_weekly : (l_max, n_channels)
            Adstock carry-in from previous weeks.
        weekly_spend : (13, n_channels)
            Weekly spend for the 13-week window.
        """
        carry_weekly = np.atleast_2d(np.asarray(carry_weekly, dtype=float))
        weekly_spend = np.atleast_2d(np.asarray(weekly_spend, dtype=float))
        spend = np.vstack([carry_weekly, weekly_spend])  # (l_max+13, n_ch)
        self._spend_shared.set_value(spend)

    def set_posterior(self, posterior: xr.Dataset):
        """Swap posterior draws without recompiling.
        
        First call compiles the graph (one-time cost ~20s); later calls rebind.
        """
        from pymc_marketing.pytensor_utils import SharedPosterior, extract_response_distribution, _posterior_sample_major
        from pytensor.graph.replace import clone_replace
        from pytensor.graph.traversal import ancestors
        
        if not self._compiled:
            # First call: create SharedPosterior and compile
            self._shared_posterior = SharedPosterior()
            
            # Extract mu (linear predictor) - this is the deterministic part
            mu_node = self.mmm.model.named_vars["y"].owner.inputs[1]
            mu_var = extract_response_distribution(self.mmm.model, self._idata, mu_node,
                                                   shared_posterior=self._shared_posterior)
            # CRITICAL: Replace channel_data in the extracted graph with our shared variable
            # extract_response_distribution doesn't replace channel_data (it's not a free_RV),
            # so we need to do it manually. The channel_data in the extracted graph is a
            # clone of the model's channel_data, so we need to find it in the ancestors.
            # Note: channel_data is an XTensorSharedVariable (not TensorSharedVariable),
            # so we check for SharedVariable base class.
            mu_ancestors = list(ancestors([mu_var]))
            channel_data_to_replace = None
            for v in mu_ancestors:
                if (isinstance(v, pytensor.compile.sharedvalue.SharedVariable) and
                    hasattr(v, 'name') and v.name == 'channel_data'):
                    channel_data_to_replace = v
                    break
            if channel_data_to_replace is not None:
                mu_var = clone_replace(mu_var, replace={channel_data_to_replace: self._spend_shared})
            self._mu_graph = pytensor.function([], mu_var)
            
            # Also compile sigma graph (y_sigma is a Deterministic in the model)
            sigma_var = extract_response_distribution(self.mmm.model, self._idata, "y_sigma",
                                                      shared_posterior=self._shared_posterior)
            sigma_ancestors = list(ancestors([sigma_var]))
            channel_data_to_replace = None
            for v in sigma_ancestors:
                if (isinstance(v, pytensor.compile.sharedvalue.SharedVariable) and
                    hasattr(v, 'name') and v.name == 'channel_data'):
                    channel_data_to_replace = v
                    break
            if channel_data_to_replace is not None:
                sigma_var = clone_replace(sigma_var, replace={channel_data_to_replace: self._spend_shared})
            self._sigma_graph = pytensor.function([], sigma_var)
            self._compiled = True
        
        # Swap posterior data (no recompilation)
        # Use SharedPosterior for the initial setup, but for subsequent swaps,
        # manually set the values to avoid the overhead of _aligned_values
        # Stack chain/draw into sample dimension (like _posterior_sample_major but faster)
        # Avoid xarray.stack (22s for 32k draws) by using numpy reshape directly
        posterior_ds = posterior if isinstance(posterior, xr.Dataset) else posterior.to_dataset()
        for name in self._shared_posterior._variables:
            da = posterior_ds[name]
            # Get the dims that SharedPosterior expects (includes 'sample')
            dims = self._shared_posterior._dims[name]
            # Extract values - if dims include 'sample' but da has 'chain'/'draw',
            # we need to reshape (chain, draw, ...) -> (chain*draw, ...)
            values = np.asarray(da)
            if "chain" in da.dims and "draw" in da.dims and "sample" in dims:
                # Get the position of chain and draw in da.dims
                chain_idx = da.dims.index("chain")
                draw_idx = da.dims.index("draw")
                # Transpose to put chain and draw first
                other_dims = [d for d in da.dims if d not in ("chain", "draw")]
                transposed_dims = ("chain", "draw") + tuple(other_dims)
                values = np.asarray(da.transpose(*transposed_dims))
                # Reshape (chain, draw, ...) -> (chain*draw, ...)
                new_shape = (values.shape[0] * values.shape[1],) + values.shape[2:]
                values = values.reshape(new_shape)
            elif "sample" in dims:
                # dims include 'sample' but da might have 'sample' already
                sample_idx = dims.index("sample")
                if "sample" in da.dims:
                    # Transpose to put sample first
                    other_dims = [d for d in da.dims if d != "sample"]
                    transposed_dims = ("sample",) + tuple(other_dims)
                    values = np.asarray(da.transpose(*transposed_dims))
                # else: values already has sample first from the initial setup
            var = self._shared_posterior._variables[name]
            var.set_value(values.astype(var.type.dtype), borrow=True)

    def evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the compiled graph. Returns (mu_orig, sigma_orig)."""
        # Evaluate both graphs
        mu_scaled = np.asarray(self._mu_graph())  # (S, l_max+13)
        sigma_scaled = np.asarray(self._sigma_graph())  # (S,)
        
        mu_orig = mu_scaled[:, self.l_max:] * self.target_scale  # (S, 13)
        sigma_orig = sigma_scaled * self.target_scale  # (S,)
        return mu_orig, sigma_orig


# Global cache: key = (window_start, l_max, n_channels)
# Within a single process, the graph is compiled once per window.
_response_evaluator_cache: dict = {}


def _get_response_evaluator(
    mmm, df, window_start, l_max, n_channels
) -> CompiledResponseEvaluator:
    """Get or create a cached CompiledResponseEvaluator."""
    key = (str(window_start), l_max, n_channels)
    if key not in _response_evaluator_cache:
        _response_evaluator_cache[key] = CompiledResponseEvaluator(
            mmm, df, window_start, l_max, n_channels
        )
    return _response_evaluator_cache[key]


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

    Uses a cached compiled evaluator so that only data is swapped between
    calls (no recompilation).  First call compiles the graph (~20 s);
    subsequent calls are near-instant (~0.1 s).

    Returns ``(mu_orig, sigma_orig)``: ``mu_orig`` (S, 13), ``sigma_orig``
    (S,).
    """
    l_max = int(mmm.adstock.l_max)
    n_ch = len(list(mmm.channel_columns))
    evaluator = _get_response_evaluator(mmm, df, window_start, l_max, n_ch)
    if carry_weekly is None:
        carry_weekly = np.zeros((l_max, n_ch))
    evaluator.set_spend(carry_weekly, weekly_spend)
    evaluator.set_posterior(posterior)
    return evaluator.evaluate()


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
