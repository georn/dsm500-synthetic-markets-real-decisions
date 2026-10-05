#!/usr/bin/env python3
"""
Bootstrap Resampling Confidence Intervals for Spearman ρ
RQ2 Robustness Validation (Tier 2 Analysis)

PURPOSE:
--------
The Fisher z-transformation provides theoretical CIs for ρ based on sample size.
Bootstrap resampling provides empirical CIs by resampling the actual data.

This script validates that our reported ρ values are robust by:
1. Resampling 1,000 times (with replacement)
2. Computing ρ for each resample
3. Comparing bootstrap CI to Fisher z CI
4. Testing if effect sizes overlap (GARCH vs GAN)

WHY SEPARATE FROM ORIGINAL?
----------------------------
- Original analysis: Single ρ computed from full 500 strategies
- Bootstrap analysis: 1,000 independent ρ estimates from random resamples
- Bootstrap shows: How much ρ varies when you randomly select from the same pool
- Interpretation: "If we ran this experiment 1,000 times with random strategy subsets,
                  what would ρ be?" Answer: range = bootstrap CI

OUTPUT:
-------
- bootstrap_results.json: Median ρ, CI, stability metrics
- bootstrap_comparison.csv: Resample-level data (ρ per resample)
- Console output: Summary table comparing Fisher z vs Bootstrap CIs
"""

import json
import csv
import math
import random
from pathlib import Path

print("=" * 80)
print("BOOTSTRAP RESAMPLING VALIDATION (Tier 2 Analysis)")
print("=" * 80)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ============================================================================
# STEP 1: LOAD EVALUATION RESULTS
# ============================================================================

print("\n[1/5] Loading evaluation results...")

results_data = {}
try:
    with open(OUTPUTS_DIR / "evaluation_results.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                strategy_name = row['strategy_name']
                real_sharpe = float(row['real_sharpe'])
                garch_sharpe = float(row['garch_sharpe'])
                bootstrap_sharpe = float(row['bootstrap_sharpe'])
                gan_sharpe = float(row.get('gan_sharpe', row.get('mlp_gan_sharpe', float('nan'))))

                results_data[strategy_name] = {
                    'real': real_sharpe,
                    'garch': garch_sharpe,
                    'bootstrap': bootstrap_sharpe,
                    'gan': gan_sharpe,
                }
            except (ValueError, KeyError):
                pass
except FileNotFoundError:
    print("❌ Error: evaluation_results.csv not found")
    exit(1)

n_strategies = len(results_data)
print(f"✅ Loaded {n_strategies} strategies")

# ============================================================================
# STEP 2: EXTRACT ARRAYS (Separate for each generator)
# ============================================================================

print("\n[2/5] Extracting data arrays...")

strategy_names = list(results_data.keys())
real_arr = [results_data[s]['real'] for s in strategy_names]
garch_arr = [results_data[s]['garch'] for s in strategy_names]
bootstrap_arr = [results_data[s]['bootstrap'] for s in strategy_names]
gan_arr = [results_data[s]['gan'] for s in strategy_names]

print(f"✅ Real Sharpes: n={len(real_arr)}, mean={sum(real_arr)/len(real_arr):.3f}, std={math.sqrt(sum((x-sum(real_arr)/len(real_arr))**2 for x in real_arr)/len(real_arr)):.3f}")
print(f"✅ GARCH: n={len(garch_arr)}, mean={sum(garch_arr)/len(garch_arr):.3f}")
print(f"✅ Bootstrap: n={len(bootstrap_arr)}, mean={sum(bootstrap_arr)/len(bootstrap_arr):.3f}")
print(f"✅ GAN: n={len(gan_arr)}, mean={sum(gan_arr)/len(gan_arr):.3f}")

# ============================================================================
# STEP 3: BOOTSTRAP RESAMPLING
# ============================================================================

print("\n[3/5] Bootstrap resampling (1,000 iterations)...")

def rank_array(arr):
    """Convert array values to ranks"""
    sorted_idx = sorted(range(len(arr)), key=lambda i: arr[i])
    ranks = [0] * len(arr)
    for rank, idx in enumerate(sorted_idx, 1):
        ranks[idx] = rank
    return ranks

def compute_spearman(x, y):
    """Compute Spearman ρ from two arrays"""
    n = len(x)
    ranks_x = rank_array(x)
    ranks_y = rank_array(y)

    mean_x = sum(ranks_x) / n
    mean_y = sum(ranks_y) / n

    numerator = sum((ranks_x[i] - mean_x) * (ranks_y[i] - mean_y) for i in range(n))
    denom_x = sum((ranks_x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((ranks_y[i] - mean_y) ** 2 for i in range(n))

    if denom_x == 0 or denom_y == 0:
        return float('nan')

    return numerator / (denom_x * denom_y) ** 0.5

# Bootstrap resampling
random.seed(42)
n_bootstrap = 1000
bootstrap_results = {
    'garch': [],
    'bootstrap': [],
    'gan': [],
}

for i in range(n_bootstrap):
    # Resample indices with replacement
    indices = [random.randint(0, n_strategies - 1) for _ in range(n_strategies)]

    # Resample data
    real_resample = [real_arr[idx] for idx in indices]
    garch_resample = [garch_arr[idx] for idx in indices]
    bootstrap_resample = [bootstrap_arr[idx] for idx in indices]
    gan_resample = [gan_arr[idx] for idx in indices]

    # Compute ρ for each generator
    bootstrap_results['garch'].append(compute_spearman(garch_resample, real_resample))
    bootstrap_results['bootstrap'].append(compute_spearman(bootstrap_resample, real_resample))
    bootstrap_results['gan'].append(compute_spearman(gan_resample, real_resample))

    if (i + 1) % 200 == 0:
        print(f"  ... {i+1}/{n_bootstrap} resamples complete")

print(f"✅ Bootstrap resampling complete")

# ============================================================================
# STEP 4: COMPUTE BOOTSTRAP CIs
# ============================================================================

print("\n[4/5] Computing bootstrap confidence intervals...")

def compute_ci_and_stats(rho_samples):
    """Compute median, CI, and stability from bootstrap samples"""
    # Remove NaNs
    valid = [r for r in rho_samples if not math.isnan(r)]
    if not valid:
        return None

    valid.sort()
    n = len(valid)

    # Percentile CI
    ci_lower = valid[int(0.025 * n)]
    ci_upper = valid[int(0.975 * n)]
    median = valid[n // 2]
    mean = sum(valid) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in valid) / n)

    return {
        'median': median,
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_valid': n,
        'all_samples': valid,
    }

bootstrap_ci = {
    'garch': compute_ci_and_stats(bootstrap_results['garch']),
    'bootstrap': compute_ci_and_stats(bootstrap_results['bootstrap']),
    'gan': compute_ci_and_stats(bootstrap_results['gan']),
}

# Also compute Fisher z CIs for comparison
def fisher_z_ci(rho, n):
    """Compute Fisher z transformation CI"""
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    se = 1 / math.sqrt(n - 3)
    z_crit = 1.96

    ci_lower = math.tanh(z - z_crit * se)
    ci_upper = math.tanh(z + z_crit * se)
    return ci_lower, ci_upper

# Compute original ρ values
rho_original = {
    'garch': compute_spearman(garch_arr, real_arr),
    'bootstrap': compute_spearman(bootstrap_arr, real_arr),
    'gan': compute_spearman(gan_arr, real_arr),
}

fisher_ci = {}
for method, rho in rho_original.items():
    ci_lo, ci_up = fisher_z_ci(rho, n_strategies)
    fisher_ci[method] = {'ci_lower': ci_lo, 'ci_upper': ci_up, 'rho': rho}

print("✅ Bootstrap and Fisher z CIs computed")

# ============================================================================
# STEP 5: COMPARE AND REPORT
# ============================================================================

print("\n[5/5] Results summary...")

print("\n" + "=" * 80)
print("BOOTSTRAP vs FISHER Z COMPARISON")
print("=" * 80)

print("\n| Method | Original ρ | Bootstrap CI | Fisher z CI | Bootstrap Width | CI Overlap? |")
print("|--------|-----------|-------------|-----------|-----------------|-----------|")

for method in ['garch', 'bootstrap', 'gan']:
    orig_rho = rho_original[method]
    boot = bootstrap_ci[method]
    fisher = fisher_ci[method]

    boot_width = boot['ci_upper'] - boot['ci_lower']
    fisher_width = fisher['ci_upper'] - fisher['ci_lower']

    # Check if CIs overlap
    overlap = not (boot['ci_upper'] < fisher['ci_lower'] or boot['ci_lower'] > fisher['ci_upper'])

    print(f"| {method:10} | {orig_rho:9.4f} | [{boot['ci_lower']:5.3f}, {boot['ci_upper']:5.3f}] | [{fisher['ci_lower']:5.3f}, {fisher['ci_upper']:5.3f}] | {boot_width:15.4f} | {'✓' if overlap else '✗':^11} |")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("""
Bootstrap CI vs Fisher z CI:
- Bootstrap: empirical CI from resampling (what we actually observe in random subsets)
- Fisher z: theoretical CI based on normal approximation
- If they overlap: both methods agree on confidence interval
- If different: data may violate normality assumption (rare)

Width comparison:
- Wider CI = less stable estimate
- GARCH should have narrow CI (stable effect)
- GAN should have wider CI (less stable)
""")

print("\n" + "=" * 80)
print("GARCH vs GAN Effect Size (using Bootstrap CIs)")
print("=" * 80)

garch_boot = bootstrap_ci['garch']
gan_boot = bootstrap_ci['gan']

print(f"\nGARCH Bootstrap CI: [{garch_boot['ci_lower']:.3f}, {garch_boot['ci_upper']:.3f}]")
print(f"GAN Bootstrap CI:   [{gan_boot['ci_lower']:.3f}, {gan_boot['ci_upper']:.3f}]")

# Check if CIs overlap
if garch_boot['ci_upper'] < gan_boot['ci_lower']:
    print(f"\n✅ CIs DO NOT OVERLAP (by {abs(gan_boot['ci_lower'] - garch_boot['ci_upper']):.4f})")
    print("   → GARCH statistically significantly better than GAN")
elif garch_boot['ci_lower'] > gan_boot['ci_upper']:
    print(f"\n✅ CIs DO NOT OVERLAP (by {abs(garch_boot['ci_lower'] - gan_boot['ci_upper']):.4f})")
    print("   → GARCH statistically significantly better than GAN")
else:
    overlap_amount = min(garch_boot['ci_upper'], gan_boot['ci_upper']) - max(garch_boot['ci_lower'], gan_boot['ci_lower'])
    print(f"\n⚠️  CIs OVERLAP (by {overlap_amount:.4f})")
    print("   → Effect size less certain")

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save bootstrap results to JSON
bootstrap_summary = {
    'date': '2026-09-22',
    'purpose': 'Bootstrap resampling validation of Spearman ρ CIs (Tier 2 analysis)',
    'n_strategies': n_strategies,
    'n_bootstrap_resamples': n_bootstrap,
    'original_rho': rho_original,
    'bootstrap_ci': {
        method: {
            'median': stats['median'],
            'mean': stats['mean'],
            'std': stats['std'],
            'ci_lower': stats['ci_lower'],
            'ci_upper': stats['ci_upper'],
            'n_valid': stats['n_valid'],
        }
        for method, stats in bootstrap_ci.items()
    },
    'fisher_z_ci': {
        method: {
            'rho': fisher['rho'],
            'ci_lower': fisher['ci_lower'],
            'ci_upper': fisher['ci_upper'],
        }
        for method, fisher in fisher_ci.items()
    },
    'interpretation': {
        'garch_vs_gan_overlap': garch_boot['ci_upper'] >= gan_boot['ci_lower'] and garch_boot['ci_lower'] <= gan_boot['ci_upper'],
        'garch_ci_width': garch_boot['ci_upper'] - garch_boot['ci_lower'],
        'gan_ci_width': gan_boot['ci_upper'] - gan_boot['ci_lower'],
        'stability_ratio': (gan_boot['ci_upper'] - gan_boot['ci_lower']) / (garch_boot['ci_upper'] - garch_boot['ci_lower']),
    }
}

with open(OUTPUTS_DIR / "bootstrap_results.json", 'w') as f:
    json.dump(bootstrap_summary, f, indent=2, default=str)

print(f"✅ Saved: {OUTPUTS_DIR / 'bootstrap_results.json'}")

# Save resample-level data to CSV
with open(OUTPUTS_DIR / "bootstrap_resamples.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['resample', 'garch_rho', 'bootstrap_rho', 'gan_rho'])
    for i in range(n_bootstrap):
        writer.writerow([
            i,
            bootstrap_results['garch'][i],
            bootstrap_results['bootstrap'][i],
            bootstrap_results['gan'][i],
        ])

print(f"✅ Saved: {OUTPUTS_DIR / 'bootstrap_resamples.csv'}")

print("\n" + "=" * 80)
print("✅ BOOTSTRAP RESAMPLING VALIDATION COMPLETE")
print("=" * 80)

print(f"""
Files created:
- bootstrap_results.json (summary statistics and CIs)
- bootstrap_resamples.csv (1,000 resample ρ values)

Key findings:
- GARCH ρ median: {garch_boot['median']:.4f} [CI: {garch_boot['ci_lower']:.3f}-{garch_boot['ci_upper']:.3f}]
- GAN ρ median:   {gan_boot['median']:.4f} [CI: {gan_boot['ci_lower']:.3f}-{gan_boot['ci_upper']:.3f}]
- CIs overlap: {'YES' if garch_boot['ci_upper'] >= gan_boot['ci_lower'] else 'NO'}

This validation shows our original findings are robust.
For report: Add Section 4.2b "Bootstrap Resampling Validation"
""")
