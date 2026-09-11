#!/usr/bin/env python
"""Stage 1 case-study MMM fit (user-run; the 4x8000 default fit).

Fits the 7-channel case-study MMM on the train window (2014-08-03 …
2018-01-28, 183 weeks) and writes the Zarr artifacts + ``fit_summary.json``
per docs/contracts/stage1-fit.md §2.6.

Usage:
    python scripts/fit_case_study.py [--data PATH] [--outdir PATH]
        [--chains N] [--draws N] [--tune N] [--sampler {nutpie,numpyro}]
        [--seed N]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# 1. Bootstrap sys.path so `import mmm_evsi...` works as a plain script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from mmm_evsi import config
from mmm_evsi.diagnostics import sampler_diagnostics
from mmm_evsi.load_mmm import load_case_study_data, select_window

TUNE_DEFAULT = 1000  # burn-in only; not gated by any Stage-1 gate


def _artifact_path(path: Path) -> str:
    """Repo-root-relative artifact path when inside the repo, else absolute."""
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the Stage 1 case-study MMM on the train window and write "
            "Zarr artifacts + fit_summary.json."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=config.RAW_CSV,
        help=f"raw case-study CSV (default: {config.RAW_CSV})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help=f"artifact output directory (default: {config.ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=config.CHAINS,
        help=f"MCMC chains (default: {config.CHAINS})",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=config.DRAWS,
        help=f"posterior draws per chain (default: {config.DRAWS})",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=TUNE_DEFAULT,
        help=f"burn-in draws per chain (default: {TUNE_DEFAULT})",
    )
    parser.add_argument(
        "--sampler",
        choices=["nutpie", "numpyro"],
        default=config.NUTS_SAMPLER,
        help=f"sampler backend (default: {config.NUTS_SAMPLER})",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=config.TARGET_ACCEPT,
        help=f"NUTS target_accept (default: {config.TARGET_ACCEPT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducibility (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 2. Load the full case-study dataset (downloads if absent).
    df = load_case_study_data(args.data)

    # 3. Train window only (183 weeks).
    train = select_window(df, config.TRAIN_WINDOW)

    # 4. Build the case-study model (human-name-free; raw mdsp_* channels).
    controls = [
        col for col in train.columns
        if col.startswith(config.CONTROL_COLUMNS_PREFIX)
    ]
    mmm = MMM(
        date_column=config.DATE_COLUMN,
        channel_columns=config.CHANNEL_COLUMNS,
        target_column=config.SALES_COLUMN,
        control_columns=controls,
        adstock=GeometricAdstock(l_max=config.ADSTOCK_L_MAX),
        saturation=LogisticSaturation(),
        yearly_seasonality=config.YEARLY_SEASONALITY,
    )

    # 5. Fit on the train window only (X without the target). No
    #    compute_log_likelihood: PyMC 6.2 .sample has no such argument and the
    #    default leaves log_likelihood out of the lean Stage-1 artifact.
    mmm.fit(
        train.drop(columns=[config.SALES_COLUMN]),
        y=train[config.SALES_COLUMN],
        chains=args.chains,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        nuts_sampler=args.sampler,
        random_seed=args.seed,
        progressbar=True,
    )

    # 6. Write Zarr artifacts (not NetCDF). Remove stale artifacts first so
    #    re-runs are idempotent (zarr 'w-' mode fails on existing stores).
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    model_file = outdir / config.MODEL_FILE.name
    idata_file = outdir / config.IDATA_FILE.name
    summary_file = outdir / config.SUMMARY_FILE.name
    import shutil

    for stale in (model_file, idata_file, summary_file):
        if stale.exists():
            shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    mmm.save(str(model_file))
    mmm.idata.to_zarr(str(idata_file))

    # 7. fit_summary.json (schema: contract §1.3; diagnostics = the 5 gated
    #    fields; chain/draw/n_effective_samples live under "fit"). Diagnostics
    #    are limited to the model's FREE random variables (excludes the
    #    deterministic *_contribution variables whose zero-spend weeks yield
    #    NaN r-hat/ESS when summarized).
    var_names = [var.name for var in mmm.model.free_RVs]
    diag = sampler_diagnostics(mmm.idata, var_names=var_names)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "source": str(args.data),
            "rows": int(len(df)),
            "date_min": df[config.DATE_COLUMN].min().date().isoformat(),
            "date_max": df[config.DATE_COLUMN].max().date().isoformat(),
        },
        "fit": {
            "window": list(config.TRAIN_WINDOW),
            "n_weeks": int(len(train)),
            "chains": args.chains,
            "draws": args.draws,
            "tune": args.tune,
            "sampler": args.sampler,
            "target_accept": args.target_accept,
            "random_seed": args.seed,
            "n_effective_samples": diag["n_effective_samples"],
            "adstock": f"GeometricAdstock(l_max={config.ADSTOCK_L_MAX})",
            "saturation": "LogisticSaturation",
            "yearly_seasonality": config.YEARLY_SEASONALITY,
            "channels": list(config.CHANNEL_COLUMNS),
            "controls": list(controls),
        },
        "diagnostics": {
            key: diag[key]
            for key in (
                "min_ess_bulk",
                "min_ess_tail",
                "max_rhat",
                "n_rhat_nan",
                "n_divergences",
            )
        },
        "artifacts": {
            "model": _artifact_path(model_file),
            "idata": _artifact_path(idata_file),
            "summary": _artifact_path(summary_file),
        },
    }
    with open(summary_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # 8. Report.
    print(f"model:            {model_file}")
    print(f"idata:            {idata_file}")
    print(f"fit_summary.json: {summary_file}")
    print(f"n_effective_samples: {diag['n_effective_samples']}")
    print(f"min_ess_bulk:        {diag['min_ess_bulk']}")
    print(f"min_ess_tail:        {diag['min_ess_tail']}")
    print(f"max_rhat:            {diag['max_rhat']}")
    print(f"n_divergences:       {diag['n_divergences']}")


if __name__ == "__main__":
    main()