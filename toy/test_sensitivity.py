"""Quick smoke test for posterior_sensitivity module."""
from __future__ import annotations

import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from mmm_evsi.load_mmm import load_case_study_data, load_budgets, load_mmm
from mmm_evsi.posterior_sensitivity import run_sensitivity_study


def main():
    print("=" * 70)
    print("Posterior Sensitivity Study — Smoke Test")
    print("=" * 70)
    t0 = time.time()

    # Load data
    print("\n[1/3] Loading data...")
    df = load_case_study_data()
    budget_plan = load_budgets(df)
    mmm, idata = load_mmm()
    print(f"  Q1 total: {budget_plan.q1.total:,.0f}")
    print(f"  Channels: {list(budget_plan.q1.planned.keys())}")
    post = idata["posterior"].to_dataset()
    print(f"  Posterior draws: {post['adstock_alpha'].sizes}")

    # Top 3 spenders
    top3 = sorted(budget_plan.q1.planned.items(), key=lambda x: x[1], reverse=True)[:3]
    channels = [ch for ch, _ in top3]
    print(f"  Study channels: {channels}")

    # Full scale factors
    scale_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    print(f"  Scale factors: {scale_factors}")

    # Run study (KL + contraction only, no Q2 for speed)
    print("\n[2/3] Running sensitivity study (KL + contraction only, no Q2)...")
    study = run_sensitivity_study(
        mmm=mmm,
        idata=idata,
        df=df,
        budget_plan=budget_plan,
        channels=channels,
        scale_factors=scale_factors,
        n_outcomes=10,
        seed=42,
        include_q2=False,
    )

    elapsed = time.time() - t0
    print(f"\n[3/3] Completed in {elapsed:.1f}s")
    print()
    print(study.to_summary())

    # Verify results
    assert len(study.results) == len(channels) * len(scale_factors), \
        f"Expected {len(channels) * len(scale_factors)} results, got {len(study.results)}"
    for r in study.results:
        assert np.isfinite(r.posterior.kl_adstock_alpha) or r.posterior.degenerate, \
            f"KL not finite for {r.channel} @ {r.scale_factor}x"
        assert np.isfinite(r.posterior.weight_ess) and r.posterior.weight_ess > 0, \
            f"ESS not valid for {r.channel} @ {r.scale_factor}x"

    print("\n✅ All smoke tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
