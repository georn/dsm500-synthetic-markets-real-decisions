#!/usr/bin/env python3
"""Task-based utility evaluation (RQ2) for DSM500 CW2.

For every trading rule, compare its mean Sharpe ratio on synthetic paths from each
generator with its mean Sharpe ratio on the real era-B USD/EUR windows, and report
the Spearman rank correlation between the two.

Pipeline:
1. Load USD/EUR era-A (training) and era-B (test) windows (asset rows only, no zero-filling)
2. Build synthetic paths: window bootstrap (resampled era-A windows), GARCH(1,1)
   fitted to the reconstructed era-A series, and the saved MLP GAN
3. Backtest every rule on every real and synthetic window (StrategyEvaluator)
4. Rank-correlate synthetic and real mean Sharpe ratios per generator

Backtest convention (fixed 2026-10-05): long-only, one position at a time; enter at
the close of the signal day and hold for the following days until the exit; the
Sharpe ratio is mean / std of the daily strategy returns within a window (not
annualised). Results are keyed by rule name, so names must be unique.

Outputs:
- outputs/evaluation_results.csv (per-rule Sharpe ratios)
- outputs/evaluation_summary.json (Spearman rho and p-values)

Usage:
    python3 pipeline/05_evaluation.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)


def asset_windows(df: pd.DataFrame, ticker: str = "EUR=X") -> np.ndarray:
    """Return the (n_windows, 256) return windows that belong to one asset."""
    cols = [f"{ticker}_step_{i}" for i in range(256)]
    return df.loc[df["asset"] == ticker, cols].to_numpy(dtype=float)


def reconstruct_series(windows: np.ndarray) -> np.ndarray:
    """Undo stride-1 windowing: first window plus the last value of each later window."""
    return np.concatenate([windows[0], windows[1:, -1]])


class StrategyEvaluator:
    """Evaluate strategy performance on real vs synthetic data."""

    def __init__(self, window_length: int = 256):
        """Initialize evaluator."""
        self.window_length = window_length

    def apply_strategy(self, returns: np.ndarray, entry_params: Dict, exit_params: Dict) -> float:
        """
        Apply trading strategy and compute Sharpe ratio.

        Args:
            returns: 1D array of returns (window of 256 steps)
            entry_params: Entry signal parameters
            exit_params: Exit signal parameters

        Returns:
            Daily Sharpe ratio of the strategy return stream (position x return)
        """
        if len(returns) < 10:
            return np.nan

        # Simple entry signal implementation
        entry_type = entry_params.get('type', 'ma_crossover')

        if entry_type == 'ma_crossover':
            ma_fast = entry_params.get('ma_fast', 10)
            ma_slow = entry_params.get('ma_slow', 50)

            # Ensure windows exist in data
            if ma_slow >= len(returns):
                return np.nan

            log_price = pd.Series(np.cumsum(returns))
            fast_ma = log_price.rolling(window=ma_fast).mean()
            slow_ma = log_price.rolling(window=ma_slow).mean()

            # Entry: fast MA crosses above slow MA
            crossovers = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            entry_points = np.where(crossovers)[0]

        elif entry_type == 'momentum':
            threshold = entry_params.get('threshold', 0.01)
            entry_points = np.where(returns > threshold)[0]

        elif entry_type == 'rsi':
            threshold = entry_params.get('threshold', 30)
            # RSI-style rule: enter when today's return is below the given percentile of
            # the returns seen so far (past only, at least 20 observations), so no look-ahead.
            past_quantile = pd.Series(returns).expanding(min_periods=20).quantile(threshold / 100).shift(1)
            entry_points = np.where(returns < past_quantile.to_numpy())[0]

        elif entry_type == 'bollinger':
            window = entry_params.get('window', 20)
            std_dev = entry_params.get('std_dev', 2.0)

            if window >= len(returns):
                return np.nan

            log_price = pd.Series(np.cumsum(returns))
            mean = log_price.rolling(window=window).mean()
            std = log_price.rolling(window=window).std()
            lower_band = mean - std_dev * std

            entry_points = np.where(log_price < lower_band)[0]

        else:
            return np.nan

        if len(entry_points) == 0:
            return np.nan

        # Exit signal implementation
        exit_type = exit_params.get('type', 'time_based')
        hold_bars = exit_params.get('hold_bars', 5)
        profit_target = exit_params.get('profit_target', 0.02)
        stop_loss = exit_params.get('stop_loss', -0.01)

        # Long-only, one position at a time: enter at the close of entry_idx and hold
        # for days entry_idx+1 .. exit_idx; signals while already in a position are ignored.
        position = np.zeros(len(returns))
        busy_until = -1
        for entry_idx in entry_points:
            if entry_idx + 1 >= len(returns):
                break
            if entry_idx < busy_until:
                continue

            exit_idx = entry_idx + 1

            # Determine exit point
            if exit_type == 'time_based':
                exit_idx = min(entry_idx + hold_bars, len(returns) - 1)

            elif exit_type == 'profit_target' or exit_type == 'combined':
                # Check for profit target or stop loss before hold time
                for i in range(entry_idx + 1, min(entry_idx + hold_bars, len(returns))):
                    path_return = returns[entry_idx + 1:i + 1].sum()
                    if path_return >= profit_target or path_return <= stop_loss:
                        exit_idx = i
                        break
                else:
                    exit_idx = min(entry_idx + hold_bars, len(returns) - 1)

            position[entry_idx + 1:exit_idx + 1] = 1.0
            busy_until = exit_idx

        # Daily Sharpe ratio of the strategy's return stream over the window (not annualised)
        daily = position * returns
        if not position.any() or np.std(daily) == 0:
            return np.nan
        return float(np.mean(daily) / np.std(daily))

    def evaluate_on_real(self, real_data: np.ndarray, strategies: List, asset_idx: int = 0) -> Dict[str, float]:
        """
        Evaluate all strategies on real windows and average the Sharpe ratio per rule.

        Args:
            real_data: (n_windows, 256) single-asset windows, or (n_windows, 768)
                three-asset rows from which asset_idx selects the columns
            strategies: List of Strategy objects (names must be unique)
            asset_idx: Asset block for 768-column input (0=USD/EUR, 1=SPY, 2=FTSE)

        Returns:
            Dict mapping strategy name to mean Sharpe ratio over windows with a defined value
        """
        results = {}

        # Extract asset-specific returns (256 steps per asset)
        asset_start = asset_idx * 256
        asset_end = asset_start + 256

        for window in real_data:
            asset_returns = window[asset_start:asset_end]

            # Skip if all NaN or all zero
            valid = ~np.isnan(asset_returns)
            if valid.sum() < 10:
                continue

            # Evaluate strategies on this window
            for strategy in strategies:
                if strategy.name not in results:
                    results[strategy.name] = []

                sharpe = self.apply_strategy(
                    asset_returns,
                    entry_params=self._extract_entry_params(strategy),
                    exit_params=self._extract_exit_params(strategy),
                )

                if not np.isnan(sharpe):
                    results[strategy.name].append(sharpe)

        # Aggregate across windows (mean Sharpe)
        aggregated = {}
        for name, sharpes in results.items():
            if len(sharpes) > 0:
                aggregated[name] = np.mean(sharpes)
            else:
                aggregated[name] = np.nan

        return aggregated

    def evaluate_on_synthetic(
        self, synthetic_data: np.ndarray, strategies: List, asset_idx: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate all strategies on synthetic data.

        Args:
            synthetic_data: Synthetic windowed data (n_windows, 768 or 256)
            strategies: List of Strategy objects
            asset_idx: Which asset

        Returns:
            Dict mapping strategy_name → Sharpe ratio
        """
        results = {}

        # Handle both 2D (768) and 1D (256) synthetic data
        if synthetic_data.ndim == 2 and synthetic_data.shape[1] == 768:
            asset_start = asset_idx * 256
            asset_end = asset_start + 256
        else:
            asset_start = 0
            asset_end = min(256, synthetic_data.shape[1] if synthetic_data.ndim > 1 else len(synthetic_data))

        for window in synthetic_data:
            if window.ndim == 1:
                asset_returns = window[asset_start:asset_end]
            else:
                asset_returns = window

            # Skip if all NaN or all zero
            valid = ~np.isnan(asset_returns)
            if valid.sum() < 10:
                continue

            for strategy in strategies:
                if strategy.name not in results:
                    results[strategy.name] = []

                sharpe = self.apply_strategy(
                    asset_returns,
                    entry_params=self._extract_entry_params(strategy),
                    exit_params=self._extract_exit_params(strategy),
                )

                if not np.isnan(sharpe):
                    results[strategy.name].append(sharpe)

        # Aggregate
        aggregated = {}
        for name, sharpes in results.items():
            if len(sharpes) > 0:
                aggregated[name] = np.mean(sharpes)
            else:
                aggregated[name] = np.nan

        return aggregated

    @staticmethod
    def _extract_entry_params(strategy) -> Dict:
        """Extract entry parameters from strategy."""
        return {
            'type': strategy.entry_type,
            'ma_fast': strategy.parameters.get('ma_fast'),
            'ma_slow': strategy.parameters.get('ma_slow'),
            'threshold': strategy.parameters.get('threshold'),
            'window': strategy.parameters.get('window'),
            'std_dev': strategy.parameters.get('std_dev'),
        }

    @staticmethod
    def _extract_exit_params(strategy) -> Dict:
        """Extract exit parameters from strategy."""
        return {
            'type': strategy.exit_type,
            'hold_bars': strategy.parameters.get('hold_bars'),
            'profit_target': strategy.parameters.get('profit_target', strategy.parameters.get('target')),
            'stop_loss': strategy.parameters.get('stop_loss', strategy.parameters.get('loss')),
        }


def compute_rank_correlation(real_sharpes: np.ndarray, synthetic_sharpes: np.ndarray) -> Dict:
    """
    Compute rank correlation between real and synthetic performance.

    Args:
        real_sharpes: Array of Sharpe ratios from real data
        synthetic_sharpes: Array of Sharpe ratios from synthetic data

    Returns:
        Dict with correlation metrics
    """
    # Remove NaN pairs
    valid = ~(np.isnan(real_sharpes) | np.isnan(synthetic_sharpes))
    real_clean = real_sharpes[valid]
    syn_clean = synthetic_sharpes[valid]

    if len(real_clean) < 3:
        return {
            'spearman_rho': np.nan,
            'spearman_pval': np.nan,
            'kendall_tau': np.nan,
            'kendall_pval': np.nan,
            'pearson_r': np.nan,
            'pearson_pval': np.nan,
            'n_strategies': len(real_clean),
        }

    spearman_rho, spearman_pval = stats.spearmanr(real_clean, syn_clean)
    kendall_tau, kendall_pval = stats.kendalltau(real_clean, syn_clean)
    pearson_r, pearson_pval = stats.pearsonr(real_clean, syn_clean)

    return {
        'spearman_rho': spearman_rho,
        'spearman_pval': spearman_pval,
        'kendall_tau': kendall_tau,
        'kendall_pval': kendall_pval,
        'pearson_r': pearson_r,
        'pearson_pval': pearson_pval,
        'n_strategies': len(real_clean),
    }


def main():
    """Run full evaluation pipeline."""
    logger.info("=" * 80)
    logger.info("DSM500 Task-Based Utility Evaluation (RQ2)")
    logger.info("=" * 80)

    # Setup paths
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ===== Step 1: Load data =====
    logger.info("\n[Step 1] Loading data...")

    parquet_path_a = output_dir / "windowed_data_era_a.parquet"
    parquet_path_b = output_dir / "windowed_data_era_b.parquet"

    if not parquet_path_a.exists() or not parquet_path_b.exists():
        logger.error("Data files not found. Run 01_data_pipeline.py first.")
        return False

    df_a = pd.read_parquet(parquet_path_a)
    df_b = pd.read_parquet(parquet_path_b)

    real_data_a = asset_windows(df_a, "EUR=X")
    real_data_b = asset_windows(df_b, "EUR=X")

    logger.info(f"  Era-A: {real_data_a.shape}")
    logger.info(f"  Era-B: {real_data_b.shape}")

    # ===== Step 2: Load strategies =====
    logger.info("\n[Step 2] Loading strategies...")

    import importlib.util
    strategies_spec = importlib.util.spec_from_file_location(
        "strategies", Path(__file__).parent / "03_strategies.py"
    )
    strategies_mod = importlib.util.module_from_spec(strategies_spec)
    strategies_spec.loader.exec_module(strategies_mod)
    generate_strategies = strategies_mod.generate_strategies

    strategies = generate_strategies(n=500)
    assert len({s.name for s in strategies}) == len(strategies), "results are keyed by strategy name"
    logger.info(f"  Generated {len(strategies)} strategies")

    # ===== Step 3: Load/Generate synthetic data =====
    logger.info("\n[Step 3] Loading synthetic data...")

    try:
        import torch
        gan_spec = importlib.util.spec_from_file_location(
            "gan_generator", Path(__file__).parent / "04_gan_generator.py"
        )
        gan_mod = importlib.util.module_from_spec(gan_spec)
        gan_spec.loader.exec_module(gan_mod)
        GANTrainer = gan_mod.GANTrainer

        # Prefer the USD/EUR-only model (04_gan_generator.py --eur-only); fall back to the
        # as-run model trained on the pooled three-asset rows.
        gan_path = output_dir / "gan_model_eur.pt"
        if not gan_path.exists():
            gan_path = output_dir / "gan_model.pt"
        if gan_path.exists():
            logger.info(f"  Loading trained GAN: {gan_path.name}")
            gan_trainer = GANTrainer.load(str(gan_path))
        else:
            logger.info("  Training GAN on USD/EUR era-A windows...")
            gan_trainer = GANTrainer(real_data_a)
            gan_trainer.train(epochs=50)
            gan_trainer.save(str(output_dir / "gan_model_eur.pt"))

        torch.manual_seed(42)
        synthetic_gan = gan_trainer.generate(n_samples=1000)
        logger.info(f"  GAN synthetic: {synthetic_gan.shape}")
    except ImportError:
        logger.warning("  PyTorch not available, skipping GAN")
        synthetic_gan = None

    # Bootstrap
    baselines_spec = importlib.util.spec_from_file_location(
        "baselines", Path(__file__).parent / "02_baselines.py"
    )
    baselines_mod = importlib.util.module_from_spec(baselines_spec)
    baselines_spec.loader.exec_module(baselines_mod)
    GARCHModel = baselines_mod.GARCHModel

    indices = np.random.choice(len(real_data_a), size=1000, replace=True)
    synthetic_bootstrap = real_data_a[indices]
    logger.info(f"  Bootstrap synthetic: {synthetic_bootstrap.shape}")

    # GARCH
    garch = GARCHModel()
    garch.fit(reconstruct_series(real_data_a))
    parameters = {"method": "variance targeting; alpha and beta fixed", "asset": "USD/EUR",
                  "omega": float(garch.omega), "alpha": float(garch.alpha), "beta": float(garch.beta),
                  "mean_daily_return": float(garch.mu), "daily_std": float(garch.sigma),
                  "persistence": float(garch.alpha + garch.beta),
                  "half_life_days": float(np.log(0.5) / np.log(garch.alpha + garch.beta))}
    (output_dir / "garch_parameters.json").write_text(json.dumps(parameters, indent=2))
    synthetic_garch = garch.generate(n_samples=1000, n_steps=256)
    logger.info(f"  GARCH synthetic: {synthetic_garch.shape}")

    # ===== Step 4: Evaluate strategies =====
    logger.info("\n[Step 4] Evaluating strategies on real and synthetic data...")

    evaluator = StrategyEvaluator()

    # Use first asset (USD/EUR, indices 0-255)
    asset_idx = 0

    # Evaluate on real era-B data
    logger.info("  Evaluating on real era-B data...")
    real_sharpes = evaluator.evaluate_on_real(real_data_b, strategies, asset_idx=asset_idx)

    # Evaluate on synthetic data
    logger.info("  Evaluating on synthetic Bootstrap data...")
    synthetic_bootstrap_sharpes = evaluator.evaluate_on_synthetic(
        synthetic_bootstrap, strategies, asset_idx=asset_idx
    )

    logger.info("  Evaluating on synthetic GARCH data...")
    synthetic_garch_sharpes = evaluator.evaluate_on_synthetic(synthetic_garch, strategies, asset_idx=asset_idx)

    if synthetic_gan is not None:
        logger.info("  Evaluating on synthetic GAN data...")
        synthetic_gan_sharpes = evaluator.evaluate_on_synthetic(synthetic_gan, strategies, asset_idx=asset_idx)
    else:
        synthetic_gan_sharpes = None

    # ===== Step 5: Compute rank correlations (RQ2 answer) =====
    logger.info("\n[Step 5] Computing rank correlations (RQ2)...")

    real_arr = np.array([real_sharpes.get(s.name, np.nan) for s in strategies])
    bootstrap_arr = np.array([synthetic_bootstrap_sharpes.get(s.name, np.nan) for s in strategies])
    garch_arr = np.array([synthetic_garch_sharpes.get(s.name, np.nan) for s in strategies])

    corr_bootstrap = compute_rank_correlation(real_arr, bootstrap_arr)
    corr_garch = compute_rank_correlation(real_arr, garch_arr)

    if synthetic_gan_sharpes is not None:
        gan_arr = np.array([synthetic_gan_sharpes.get(s.name, np.nan) for s in strategies])
        corr_gan = compute_rank_correlation(real_arr, gan_arr)
    else:
        corr_gan = None

    logger.info(f"\n  Bootstrap Spearman ρ: {corr_bootstrap['spearman_rho']:.4f} (p={corr_bootstrap['spearman_pval']:.4f})")
    logger.info(f"  GARCH Spearman ρ: {corr_garch['spearman_rho']:.4f} (p={corr_garch['spearman_pval']:.4f})")
    if corr_gan:
        logger.info(f"  GAN Spearman ρ: {corr_gan['spearman_rho']:.4f} (p={corr_gan['spearman_pval']:.4f})")

    # ===== Step 6: Save results =====
    logger.info("\n[Step 6] Saving results...")

    # Save per-strategy results
    results_df = pd.DataFrame({
        'strategy_name': [s.name for s in strategies],
        'real_sharpe': real_arr,
        'bootstrap_sharpe': bootstrap_arr,
        'garch_sharpe': garch_arr,
        'gan_sharpe': gan_arr if synthetic_gan_sharpes is not None else [np.nan] * len(strategies),
    })

    csv_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    # Save summary (RQ2 answer)
    summary = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'rq2_question': 'To what extent does synthetic validation improve strategy performance estimation?',
        'rq2_answer': {
            'bootstrap': {
                'spearman_rho': float(corr_bootstrap['spearman_rho']),
                'spearman_pval': float(corr_bootstrap['spearman_pval']),
                'interpretation': 'Signed validation-to-real rank agreement; interpret with confidence intervals and paired comparisons',
            },
            'garch': {
                'spearman_rho': float(corr_garch['spearman_rho']),
                'spearman_pval': float(corr_garch['spearman_pval']),
                'interpretation': 'Signed validation-to-real rank agreement; interpret with confidence intervals and paired comparisons',
            },
        },
        'data_summary': {
            'real_data_shape': real_data_b.shape,
            'synthetic_bootstrap_shape': synthetic_bootstrap.shape,
            'synthetic_garch_shape': synthetic_garch.shape,
            'n_strategies': len(strategies),
            'asset_tested': 'USD/EUR (EUR=X rows only)',
        },
    }

    if corr_gan:
        summary['rq2_answer']['gan'] = {
            'spearman_rho': float(corr_gan['spearman_rho']),
            'spearman_pval': float(corr_gan['spearman_pval']),
            'interpretation': 'Signed validation-to-real rank agreement; interpret with confidence intervals and paired comparisons',
        }

    json_path = output_dir / "evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Saved: {json_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete (RQ2 answered)")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
