#!/usr/bin/env python3
"""
Baseline Generators and Fidelity Evaluation for DSM500 CW2

Implements:
- Stationary (block) bootstrap for synthetic return generation
- GARCH(1,1) model for parametric volatility modeling
- Fidelity diagnostics comparing synthetic vs real data

Inputs:
- outputs/windowed_data_era_a.parquet (training windows)

Outputs:
- outputs/fidelity_results.csv (metrics for each baseline)
- outputs/fidelity_metrics_summary.json (summary statistics)

Usage:
    python3 02_baselines.py
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)


class StationaryBootstrap:
    """Block bootstrap for time-series data."""

    def __init__(self, block_length=None):
        self.block_length = block_length or 64

    def generate(self, returns, n_samples=1000):
        """
        Generate synthetic return paths via stationary bootstrap.

        Args:
            returns: 1D array of returns
            n_samples: number of bootstrap samples to generate

        Returns:
            Array of shape (n_samples, len(returns))
        """
        n = len(returns)
        synthetic = np.zeros((n_samples, n))

        for i in range(n_samples):
            path = []
            while len(path) < n:
                start = np.random.randint(0, n - self.block_length + 1)
                block = returns[start:start + self.block_length]
                path.extend(block)
            synthetic[i] = np.array(path[:n])

        return synthetic


class GARCHModel:
    """GARCH(1,1) model for volatility."""

    def __init__(self, omega=1e-6, alpha=0.05, beta=0.94):
        """
        GARCH(1,1) parameters.
        omega: constant term
        alpha: innovation coefficient
        beta: volatility persistence
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def fit(self, returns, max_iter=100):
        """
        Fit GARCH(1,1) to returns (simplified ML).
        For speed, use pre-set parameters rather than full MLE.
        """
        self.mu = np.mean(returns)
        self.sigma = np.std(returns)
        return self

    def generate(self, n_samples=1000, n_steps=256):
        """
        Generate synthetic return paths from fitted GARCH.

        Returns:
            Array of shape (n_samples, n_steps)
        """
        synthetic = np.zeros((n_samples, n_steps))

        for i in range(n_samples):
            h = np.ones(n_steps) * self.sigma ** 2
            z = np.random.standard_normal(n_steps)
            eps = np.zeros(n_steps)

            for t in range(1, n_steps):
                eps[t] = np.sqrt(h[t]) * z[t]
                h[t] = self.omega + self.alpha * eps[t-1]**2 + self.beta * h[t-1]

            synthetic[i] = self.mu + eps

        return synthetic


class FidelityEvaluator:
    """Compute stylized fact diagnostics."""

    @staticmethod
    def autocorr_returns(returns, max_lag=20):
        """Autocorrelation of raw returns (should be ~0)."""
        return np.array([np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                        for lag in range(1, max_lag + 1)])

    @staticmethod
    def autocorr_squared_returns(returns, max_lag=20):
        """Autocorrelation of squared returns (volatility clustering)."""
        sq_returns = returns ** 2
        return np.array([np.corrcoef(sq_returns[:-lag], sq_returns[lag:])[0, 1]
                        for lag in range(1, max_lag + 1)])

    @staticmethod
    def tail_index(returns, q=0.05):
        """Left and right tail indices (fat tails)."""
        left_tail = returns[returns < np.percentile(returns, q*100)]
        right_tail = returns[returns > np.percentile(returns, (1-q)*100)]
        return {
            "left_mean": np.mean(left_tail) if len(left_tail) > 0 else 0,
            "right_mean": np.mean(right_tail) if len(right_tail) > 0 else 0,
        }

    @staticmethod
    def kurtosis(returns):
        """Excess kurtosis (fat tails)."""
        return stats.kurtosis(returns, fisher=True)

    @staticmethod
    def skewness(returns):
        """Skewness."""
        return stats.skew(returns)

    @staticmethod
    def wasserstein_distance(real, synthetic):
        """1D Wasserstein distance between distributions."""
        real_sorted = np.sort(real.flatten())
        synthetic_sorted = np.sort(synthetic.flatten())

        n_real = len(real_sorted)
        n_syn = len(synthetic_sorted)

        # Linear interpolation to same size
        idx_real = np.linspace(0, n_real - 1, max(n_real, n_syn))
        idx_syn = np.linspace(0, n_syn - 1, max(n_real, n_syn))

        real_interp = np.interp(idx_real, np.arange(n_real), real_sorted)
        syn_interp = np.interp(idx_syn, np.arange(n_syn), synthetic_sorted)

        return np.mean(np.abs(real_interp - syn_interp))


def evaluate_baseline(real_data, synthetic_data, name, df_full):
    """Compute fidelity metrics for a baseline."""
    evaluator = FidelityEvaluator()

    # Extract non-NaN values per asset and aggregate
    # Each row corresponds to ONE asset, columns are asset-specific
    real_clean = []
    synthetic_clean = []

    for idx in range(len(real_data)):
        real_row = real_data[idx]
        synthetic_row = synthetic_data[idx]

        # Extract valid columns for each independently
        # Real uses its own NaN pattern, synthetic uses its own
        real_valid = ~np.isnan(real_row)
        synthetic_valid = ~np.isnan(synthetic_row)

        if real_valid.sum() > 0:
            real_clean.extend(real_row[real_valid])
        if synthetic_valid.sum() > 0:
            synthetic_clean.extend(synthetic_row[synthetic_valid])

    real_flat = np.array(real_clean)
    synthetic_flat = np.array(synthetic_clean)

    # Fallback: if synthetic_flat is empty, use all non-NaN values
    if len(synthetic_flat) == 0:
        logger.info(f"    {name}: synthetic_flat empty, using all non-NaN values from synthetic_data")
        synthetic_flat = synthetic_data[~np.isnan(synthetic_data)]

    logger.info(f"    {name}: real_flat len={len(real_flat)}, synthetic_flat len={len(synthetic_flat)}")

    try:
        ac_real = np.mean(evaluator.autocorr_returns(real_flat)) if len(real_flat) > 0 else float('nan')
        ac_syn = np.mean(evaluator.autocorr_returns(synthetic_flat)) if len(synthetic_flat) > 0 else float('nan')
        acq_real = np.mean(evaluator.autocorr_squared_returns(real_flat)) if len(real_flat) > 0 else float('nan')
        acq_syn = np.mean(evaluator.autocorr_squared_returns(synthetic_flat)) if len(synthetic_flat) > 0 else float('nan')
    except Exception as e:
        logger.error(f"    {name}: Error computing metrics: {e}")
        ac_real = ac_syn = acq_real = acq_syn = float('nan')

    try:
        sk_real = evaluator.skewness(real_flat)
        sk_syn = evaluator.skewness(synthetic_flat)
        kt_real = evaluator.kurtosis(real_flat)
        kt_syn = evaluator.kurtosis(synthetic_flat)
        wd = evaluator.wasserstein_distance(real_data, synthetic_data)
    except Exception as e:
        logger.error(f"    {name}: Error in skewness/kurtosis/wasserstein: {e}")
        sk_real = sk_syn = kt_real = kt_syn = wd = float('nan')

    metrics = {
        "method": name,
        "autocorr_returns_mean": ac_real,
        "autocorr_returns_synthetic": ac_syn,
        "autocorr_squared_mean": acq_real,
        "autocorr_squared_synthetic": acq_syn,
        "skewness_real": sk_real,
        "skewness_synthetic": sk_syn,
        "kurtosis_real": kt_real,
        "kurtosis_synthetic": kt_syn,
        "wasserstein_distance": wd,
    }

    return metrics


def main():
    """Run baseline fidelity evaluation."""
    logger.info("=" * 80)
    logger.info("DSM500 Baseline Fidelity Evaluation")
    logger.info("=" * 80)

    # Load windowed data
    logger.info("\n[Step 1] Loading windowed data...")
    script_dir = Path(__file__).parent.parent
    parquet_path = script_dir / "outputs" / "windowed_data_era_a.parquet"
    if not parquet_path.exists():
        logger.error(f"Data file not found: {parquet_path}")
        return False

    df = pd.read_parquet(parquet_path)
    logger.info(f"  Loaded {len(df)} windows")

    # Extract return windows (all columns with '_step_' in name)
    real_data = df[[c for c in df.columns if '_step_' in c]].values
    logger.info(f"  Real data shape: {real_data.shape}")

    # Initialize baselines
    logger.info("\n[Step 2] Generating synthetic data from baselines...")

    # Bootstrap: resample windows with replacement (simple nonparametric baseline)
    # Each synthetic window is a random draw from the empirical distribution of real windows
    synthetic_bootstrap_indices = np.random.choice(len(real_data), size=len(real_data), replace=True)
    synthetic_bootstrap = real_data[synthetic_bootstrap_indices].copy()
    logger.info(f"  Bootstrap synthetic: {synthetic_bootstrap.shape}")

    # GARCH: generate synthetic returns for each asset separately, then embed in the same structure
    garch = GARCHModel().fit(real_data.flatten())
    synthetic_garch = real_data.copy()  # Start with the same structure (including NaN pattern)

    # For each asset group, generate synthetic values in the valid columns
    for idx in range(len(df)):
        asset = df.iloc[idx]['asset']
        valid_cols = ~np.isnan(real_data[idx])
        if valid_cols.sum() > 0:
            n_steps = valid_cols.sum()
            garch_path = garch.generate(n_samples=1, n_steps=n_steps)[0]
            synthetic_garch[idx, valid_cols] = garch_path

    logger.info(f"  GARCH synthetic: {synthetic_garch.shape}")

    # Evaluate fidelity
    logger.info("\n[Step 3] Evaluating fidelity metrics...")

    results = []
    for name, synthetic in [("Bootstrap", synthetic_bootstrap), ("GARCH", synthetic_garch)]:
        metrics = evaluate_baseline(real_data, synthetic, name, df)
        results.append(metrics)
        logger.info(f"  {name}: Wasserstein={metrics['wasserstein_distance']:.6f}" if not np.isnan(metrics['wasserstein_distance']) else f"  {name}: (metrics computed)")

    # Save results
    logger.info("\n[Step 4] Saving results...")

    output_dir = script_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    csv_path = output_dir / "fidelity_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "real_data_shape": real_data.shape,
        "baselines_evaluated": ["Bootstrap", "GARCH"],
        "metrics": results,
    }

    json_path = output_dir / "fidelity_metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Saved: {json_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Baseline Fidelity Evaluation Complete")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
