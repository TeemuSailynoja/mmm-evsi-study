"""Shared sampler-health quantities (G1.4) for the case-study fit."""
from __future__ import annotations

import arviz as az
import xarray as xr


def sampler_diagnostics(idata: xr.DataTree) -> dict[str, float | int]:
    """Pooled ESS/rhat/divergence summary of a fitted MMM idata (8 keys).

    Derivation mirrors the G1.4 test exactly: ``az.summary(idata, fmt="wide")``
    is pooled (chains+draws combined), NaNs are dropped from ``r_hat`` before
    the max, and the five gated fields are the first five keys -- the remaining
    three (``chains``/``draws``/``n_effective_samples``) are recorded under the
    ``fit`` block of ``fit_summary.json``, not under ``diagnostics``.
    """
    summ = az.summary(idata, fmt="wide")  # pooled (chain+draw)
    rhat = summ["r_hat"].dropna()
    return {
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "max_rhat": float(rhat.max()),
        "n_rhat_nan": int(summ["r_hat"].isna().sum()),
        "n_divergences": int(idata["sample_stats"]["diverging"].values.sum()),
        "chains": int(idata["posterior"].sizes["chain"]),
        "draws": int(idata["posterior"].sizes["draw"]),
        "n_effective_samples": int(
            idata["posterior"].sizes["chain"] * idata["posterior"].sizes["draw"]
        ),
    }