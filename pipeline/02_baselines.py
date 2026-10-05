#!/usr/bin/env python3
"""Classical baselines and RQ1 fidelity evaluation for DSM500 CW2.

Implements:
- StationaryBootstrap: Politis-Romano stationary bootstrap (geometric block lengths)
- GARCHModel: GARCH(1,1) with literature-default alpha/beta and variance targeting
- FidelityEvaluator: stylized-fact diagnostics (autocorrelations, kurtosis, skew, Wasserstein)
- main(): RQ1 fidelity on USD/EUR era-A windows for the real data, window bootstrap,
  stationary bootstrap, GARCH and (if saved models exist) the MLP and LSTM GANs.
  Statistics are per-window means over 256-day windows.

Inputs:
- outputs/windowed_data_era_a.parquet
- outputs/gan_model.pt, outputs/gan_model_lstm_fixed.pt (optional)

Outputs:
- outputs/fidelity_results.csv, outputs/fidelity_metrics_summary.json

The reported RQ1 results were produced by the earlier version of this script
(archive/pipeline_as_run_2026-10-05/02_baselines.py).

Usage:
    python3 pipeline/02_baselines.py
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
    """Stationary bootstrap (Politis and Romano, 1994): blocks of geometric random length, wrapping circularly."""

    def __init__(self, block_length=None):
        self.block_length = block_length or 64

    def generate(self, returns, n_samples=1000, n_steps=None):
        """
        Generate synthetic return paths via stationary bootstrap.

        Args:
            returns: 1D array of returns (non-finite values are dropped)
            n_samples: number of bootstrap samples to generate
            n_steps: path length (defaults to the length of the finite input)

        Returns:
            Array of shape (n_samples, n_steps)
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        n_steps = n_steps or n
        p = 1.0 / self.block_length
        synthetic = np.empty((n_samples, n_steps))

        for i in range(n_samples):
            t, idx = 0, np.random.randint(n)
            while t < n_steps:
                synthetic[i, t] = returns[idx]
                t += 1
                idx = np.random.randint(n) if np.random.random() < p else (idx + 1) % n

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
        Variance targeting: alpha and beta keep their literature defaults, and omega is
        set so the long-run variance omega / (1 - alpha - beta) equals the sample variance.
        Non-finite values are ignored. This is not full maximum-likelihood estimation.
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        self.mu = float(np.mean(returns))
        self.sigma = float(np.std(returns))
        self.omega = self.sigma ** 2 * (1.0 - self.alpha - self.beta)
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
            eps[0] = np.sqrt(h[0]) * z[0]

            for t in range(1, n_steps):
                h[t] = self.omega + self.alpha * eps[t-1]**2 + self.beta * h[t-1]
                eps[t] = np.sqrt(h[t]) * z[t]

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
        real = np.asarray(real).flatten()
        synthetic = np.asarray(synthetic).flatten()
        return float(stats.wasserstein_distance(real[np.isfinite(real)], synthetic[np.isfinite(synthetic)]))


def window_fidelity(paths):
    """Average per-window stylized-fact statistics for an (n_windows, n_steps) array."""
    ev = FidelityEvaluator()
    rows = []
    for path in paths:
        path = path[np.isfinite(path)]
        if len(path) < 30 or np.std(path) == 0:
            continue
        rows.append({
            "acf_returns_lags1_20": float(np.mean(ev.autocorr_returns(path))),
            "acf_squared_lag1": float(ev.autocorr_squared_returns(path, max_lag=1)[0]),
            "acf_squared_lags1_20": float(np.mean(ev.autocorr_squared_returns(path))),
            "excess_kurtosis": float(ev.kurtosis(path)),
            "skewness": float(ev.skewness(path)),
            "daily_std": float(np.std(path)),
        })
    summary = pd.DataFrame(rows).mean().to_dict()
    summary["n_windows"] = len(rows)
    return summary


def load_gan_paths(output_dir, filename, model_file, class_name, n_samples):
    """Generate USD/EUR paths (first 256 columns) from a saved GAN, or None if unavailable."""
    eur_path = output_dir / model_file.replace(".pt", "_eur.pt").replace("_fixed_eur", "_eur")
    path = eur_path if eur_path.exists() else output_dir / model_file
    if not path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(class_name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer = getattr(module, class_name).load(str(path))
    import torch
    torch.manual_seed(42)
    return trainer.generate(n_samples=n_samples)[:, :256]


def main():
    """RQ1 fidelity on USD/EUR era-A windows for every generator."""
    logger.info("DSM500 RQ1 fidelity evaluation (USD/EUR)")
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "outputs"
    parquet_path = output_dir / "windowed_data_era_a.parquet"
    if not parquet_path.exists():
        logger.error(f"Data file not found: {parquet_path}")
        return False

    df = pd.read_parquet(parquet_path)
    cols = [f"EUR=X_step_{i}" for i in range(256)]
    real = df.loc[df["asset"] == "EUR=X", cols].to_numpy(dtype=float)
    series = np.concatenate([real[0], real[1:, -1]])
    n = len(real)
    logger.info(f"  USD/EUR era-A windows: {real.shape}")

    sources = {
        "Real era-A USD/EUR": real,
        "Window bootstrap": real[np.random.choice(n, size=n, replace=True)],
        "Stationary bootstrap": StationaryBootstrap(block_length=20).generate(series, n_samples=n, n_steps=256),
        "GARCH(1,1)": GARCHModel().fit(series).generate(n_samples=n, n_steps=256),
    }
    try:
        for label, filename, model_file, class_name in [
            ("MLP GAN", "04_gan_generator.py", "gan_model.pt", "GANTrainer"),
            ("LSTM GAN", "04b_gan_generator_lstm_FIXED.py", "gan_model_lstm_fixed.pt", "LSTMGANTrainer"),
        ]:
            paths = load_gan_paths(output_dir, filename, model_file, class_name, n)
            if paths is not None:
                sources[label] = paths
    except ImportError:
        logger.warning("  PyTorch not available; GAN fidelity skipped")

    real_values = real[np.isfinite(real)]
    results = []
    for name, paths in sources.items():
        row = {"source": name, **window_fidelity(paths)}
        row["wasserstein_vs_real"] = float(FidelityEvaluator.wasserstein_distance(
            real_values, paths[np.isfinite(paths)]))
        results.append(row)
        logger.info(f"  {name}: {row}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "fidelity_results.csv", index=False)
    with open(output_dir / "fidelity_metrics_summary.json", "w") as f:
        json.dump({"generated_at": pd.Timestamp.now().isoformat(), "asset": "USD/EUR",
                   "statistics": "per-window means over 256-day windows", "metrics": results},
                  f, indent=2, default=str)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
