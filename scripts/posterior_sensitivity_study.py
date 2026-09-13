#!/usr/bin/env python
"""Posterior sensitivity study: how the posterior shifts when channel spend scales.

Usage:
    python scripts/posterior_sensitivity_study.py          # default (top 3, no Q2)
    python scripts/posterior_sensitivity_study.py --q2     # include Q2 optimization (~15 min)

Outputs:
    data/fit/posterior_sensitivity.parquet  - full results DataFrame
    data/fit/posterior_sensitivity_summary.txt  - human-readable summary
"""
from __future__ import annotations

import argparse
import sys
import time

import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Posterior sensitivity to channel spend scaling"
    )
    parser.add_argument(
        "--q2", action="store_true", help="Include Q2 optimization (slower)"
    )
    parser.add_argument(
        "--n-outcomes", type=int, default=10, help="Simulated outcomes per config"
    )
    parser.add_argument(
        "--channels", nargs="+", default=None, help="Channels to study (default: top 3)"
    )
    args = parser.parse_args()

    from mmm_evsi.load_mmm import load_case_study_data, load_budgets, load_mmm
    from mmm_evsi.posterior_sensitivity import run_sensitivity_study
    from mmm_evsi import config

    t0 = time.time()

    print("=" * 70)
    print("Posterior Sensitivity Study")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_case_study_data()
    budget_plan = load_budgets(df)
    mmm, idata = load_mmm()
    print(f"  Q1 total: {budget_plan.q1.total:,.0f}")
    print(f"  Posterior: {idata['posterior'].to_dataset()['adstock_alpha'].sizes}")

    # Determine channels
    if args.channels:
        channels = args.channels
    else:
        top3 = sorted(budget_plan.q1.planned.items(), key=lambda x: x[1], reverse=True)[:3]
        channels = [ch for ch, _ in top3]
    print(f"  Study channels: {channels}")

    # Run study
    print("\n[2/4] Running sensitivity study...")
    study = run_sensitivity_study(
        mmm=mmm,
        idata=idata,
        df=df,
        budget_plan=budget_plan,
        channels=channels,
        scale_factors=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        n_outcomes=args.n_outcomes,
        seed=42,
        include_q2=args.q2,
    )

    # Save results
    print("\n[3/4] Saving results...")
    df_results = study.to_dataframe()
    output_dir = config.ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(output_dir / "posterior_sensitivity.parquet", index=False)
    print(f"  Saved: {output_dir / 'posterior_sensitivity.parquet'}")

    # Print summary
    summary = study.to_summary()
    print("\n[4/4] Summary:")
    print(summary)

    with open(output_dir / "posterior_sensitivity_summary.txt", "w") as f:
        f.write(summary)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({len(study.results)} configurations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
