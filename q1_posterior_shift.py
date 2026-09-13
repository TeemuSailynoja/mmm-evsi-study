"""
Bayesian sensitivity: How much does Q1 observation shift the posterior?

For each scaling factor:
1. Scale Q1 spend flighting shapes by factor
2. Sample Q1 observations from posterior predictive (given scaled spend)
3. Compute importance weights: p(Q1_sim | theta_i, spend_scaled)
4. Measure posterior shift via KL divergence and parameter changes
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import time

from mmm_evsi.load_mmm import load_mmm, load_case_study_data
from mmm_evsi.importance import pool_posterior, CompiledResponseEvaluator
from mmm_evsi import config

# Load data
print("Loading data...")
mmm, idata = load_mmm()
df = load_case_study_data()
print("Done loading.")

# Q1 data
q1_mask = (df['wk_strt_dt'] >= pd.Timestamp(config.Q1_WINDOW[0])) & \
          (df['wk_strt_dt'] <= pd.Timestamp(config.Q1_WINDOW[1]))
q1_df = df[q1_mask].reset_index(drop=True)
q1_sales = q1_df['sales'].values.astype(float)
q1_weeks = len(q1_sales)
print(f"Q1 periods: {q1_weeks} weeks")
print(f"Q1 sales mean: {q1_sales.mean():.0f}")
print()

# Pool posterior
pooled = pool_posterior(idata["posterior"].to_dataset())
n_samples = pooled["adstock_alpha"].shape[0]
print(f"Posterior samples: {n_samples}")
print()

# Get Q1 covariates (spend + controls) - FILTER to model channels only
spend_cols = [c for c in df.columns if 'mdsp_' in c]
model_channels = mmm.channel_columns
q1_spend = q1_df[[c for c in spend_cols if c in model_channels]].values.astype(float)
q1_controls = q1_df[[c for c in df.columns if 'mdip_' in c or 'hldy_' in c or 'seas_' in c or 'me_' in c or 'mrkdn_' in c or 'va_pub_' in c or 'st_ct' in c]].values.astype(float)

print(f"Q1 spend shape: {q1_spend.shape}")
print(f"Q1 controls shape: {q1_controls.shape}")
print()

# Get carry-in from training data
training_spend = df[[c for c in spend_cols if c in model_channels]].values.astype(float)
l_max = int(mmm.adstock.l_max)
n_channels = len(model_channels)
carry_weekly = training_spend[-l_max:]  # last l_max weeks from training
print(f"Carry-in shape: {carry_weekly.shape}")
print(f"l_max: {l_max}, n_channels: {n_channels}")
print()

# Scaling factors
scaling_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

print("=" * 80)
print("Q1 POSTERIOR SHIFT ANALYSIS")
print("=" * 80)
print()

results = []

for scale in scaling_factors:
    print(f"Scaling: {scale:.2f}x")
    print("-" * 60)
    
    # Scale Q1 spend
    scaled_spend = q1_spend * scale
    print(f"  Scaled spend total: {scaled_spend.sum():.0f} (original: {q1_spend.sum():.0f})")
    
    # Initialize evaluator
    evaluator = CompiledResponseEvaluator(
        mmm=mmm,
        df=df,
        window_start=config.Q1_WINDOW[0],
        l_max=l_max,
        n_channels=n_channels,
    )
    evaluator.set_spend(carry_weekly, scaled_spend)
    evaluator.set_posterior(pooled)
    
    # Sample Q1 observations from posterior predictive
    print("  Evaluating posterior predictive...")
    t0 = time.time()
    mu_orig, sigma_orig = evaluator.evaluate()  # (n_samples, 13), (n_samples,)
    print(f"  Evaluation took {time.time() - t0:.1f}s")
    print(f"  mu_orig shape: {mu_orig.shape}")
    print(f"  sigma_orig shape: {sigma_orig.shape}")
    
    # Use only first 1000 samples for efficiency
    n_use = min(1000, mu_orig.shape[0])
    mu_use = mu_orig[:n_use]
    sigma_use = sigma_orig[:n_use]
    
    y_sim = np.random.normal(mu_use, sigma_use[:, None])  # (n_use, 13)
    
    # Compute log likelihood of simulated Q1 under each posterior draw
    log_lik = -0.5 * np.log(2 * np.pi) - np.log(sigma_use[:, None]) - 0.5 * (y_sim - mu_use) ** 2 / sigma_use[:, None] ** 2
    log_lik = log_lik.sum(axis=1)  # sum over weeks
    
    # Normalize weights (importance weights)
    log_weights = log_lik - logsumexp(log_lik)
    weights = np.exp(log_weights)
    
    # Effective sample size
    n_eff = 1.0 / np.sum(weights ** 2)
    
    # KL divergence between weighted and original posterior
    kl = float(np.sum(weights * log_weights) + np.log(n_use))
    
    # Per-parameter shifts - use only first n_use samples
    shifts = {}
    for param in ['adstock_alpha', 'saturation_beta', 'saturation_lam', 'y_sigma']:
        orig_vals = pooled[param].values  # Shape: (n_channels, n_samples) or (n_samples,)
        # Slice to first n_use samples from the last dimension
        if orig_vals.ndim > 1:
            orig_vals_sliced = orig_vals[:, :n_use]  # (n_channels, n_use)
        else:
            orig_vals_sliced = orig_vals[:n_use]  # (n_use,)
        
        # Transpose to (n_use, n_channels) for weighted averaging
        if orig_vals_sliced.ndim > 1:
            orig_vals_T = orig_vals_sliced.T  # (n_use, n_channels)
        else:
            orig_vals_T = orig_vals_sliced[:, None]  # (n_use, 1)
        
        # Compute weighted mean and std across the sample dimension
        weighted_mean = np.sum(weights[:, None] * orig_vals_T, axis=0)  # (n_channels,)
        weighted_std = np.sqrt(np.sum(weights[:, None] * (orig_vals_T - weighted_mean) ** 2, axis=0))
        
        orig_mean = orig_vals_sliced.mean()
        orig_std = orig_vals_sliced.std()
        
        delta = weighted_mean - orig_mean
        contraction = weighted_std / orig_std if orig_std > 0 else 1.0
        
        # For multi-dimensional params, report mean across dimensions
        if isinstance(delta, np.ndarray):
            delta = delta.mean()
            contraction = contraction.mean()
        
        shifts[param] = {
            'delta': delta,
            'contraction': contraction,
            'delta_pct': delta / orig_mean * 100 if orig_mean != 0 else 0,
        }
    
    print(f"  ESS: {n_eff:.0f} / {n_use} ({n_eff/n_use:.2%})")
    print(f"  KL: {kl:.4f}")
    print(f"  y_sigma contraction: {shifts['y_sigma']['contraction']:.4f}")
    print(f"  saturation_beta avg contraction: {shifts['saturation_beta']['contraction']:.4f}")
    print()
    
    results.append({
        'scale': scale,
        'ess': n_eff,
        'ess_ratio': n_eff / n_use,
        'kl': kl,
        'shifts': shifts,
    })

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"{'Scale':>6s} {'ESS':>8s} {'ESS%':>8s} {'KL':>8s} {'y_sig Δ':>10s} {'y_sig ctr':>10s} {'sat β Δ':>10s} {'sat β ctr':>10s}")
print("-" * 80)

for r in results:
    scale = r['scale']
    ess = r['ess']
    ess_pct = r['ess_ratio'] * 100
    kl = r['kl']
    y_sig = r['shifts']['y_sigma']
    sat_beta = r['shifts']['saturation_beta']
    
    print(f"{scale:6.2f} {ess:8.0f} {ess_pct:7.2f}% {kl:8.4f} {y_sig['delta']:10.4f} {y_sig['contraction']:10.4f} {sat_beta['delta']:10.4f} {sat_beta['contraction']:10.4f}")

print()
print("Interpretation:")
print("  ESS << n_samples → Q1 observations strongly shift the posterior")
print("  ESS ≈ n_samples → Q1 observations barely shift the posterior")
print("  KL > 0.5 → substantial shift")
print("  KL < 0.1 → minimal shift")
print("  Contraction < 1 → posterior tighter (more informed)")
print("  Contraction > 1 → posterior wider (less informed)")
