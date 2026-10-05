#!/usr/bin/env python3
"""
LSTM GAN Comparison Evaluation for RQ2

Parallel to 05_evaluation.py. Evaluates LSTM GAN using identical protocol:
- 500 strategies × 1,522 real era-B windows × 1,000 LSTM synthetic paths
- Computes Spearman rank-correlation vs real Sharpe ratios
- Compares directly to MLP results

Follows protocol: drafts/draft_gan_comparison_protocol.md

Output:
- outputs/evaluation_results_lstm.csv (per-strategy Sharpe ratios)
- outputs/evaluation_summary_lstm.json (RQ2 answer: Spearman ρ for LSTM)

Usage:
    python3 05b_evaluation_lstm_comparison.py

Comparison workflow:
    1. Run 05_evaluation.py (MLP) → evaluation_results.csv + evaluation_summary.json
    2. Run 05b_evaluation_lstm_comparison.py (LSTM) → evaluation_results_lstm.csv + evaluation_summary_lstm.json
    3. Compare both JSON files for ρ_LSTM vs ρ_MLP
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)


class StrategyEvaluator:
    """Evaluate strategy performance on real vs synthetic data (shared with 05_evaluation.py)."""

    def __init__(self, window_length: int = 256):
        """Initialize evaluator."""
        self.window_length = window_length

    def apply_strategy(self, returns: np.ndarray, entry_params: Dict, exit_params: Dict) -> float:
        """
        Apply trading strategy and compute Sharpe ratio.
        (Identical implementation to 05_evaluation.py)
        """
        if len(returns) < 10:
            return np.nan

        trades = []
        entry_type = entry_params.get('type', 'ma_crossover')

        if entry_type == 'ma_crossover':
            ma_fast = entry_params.get('ma_fast', 10)
            ma_slow = entry_params.get('ma_slow', 50)
            if ma_slow >= len(returns):
                return np.nan
            fast_ma = pd.Series(returns).rolling(window=ma_fast).mean()
            slow_ma = pd.Series(returns).rolling(window=ma_slow).mean()
            crossovers = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            entry_points = np.where(crossovers)[0]
        elif entry_type == 'momentum':
            threshold = entry_params.get('threshold', 0.01)
            entry_points = np.where(returns > threshold)[0]
        elif entry_type == 'rsi':
            threshold = entry_params.get('threshold', 30)
            entry_points = np.where(returns < np.percentile(returns, threshold))[0]
        elif entry_type == 'bollinger':
            window = entry_params.get('window', 20)
            std_dev = entry_params.get('std_dev', 2.0)
            if window >= len(returns):
                return np.nan
            mean = pd.Series(returns).rolling(window=window).mean()
            std = pd.Series(returns).rolling(window=window).std()
            lower_band = mean - std_dev * std
            entry_points = np.where(returns < lower_band)[0]
        else:
            return np.nan

        if len(entry_points) == 0:
            return np.nan

        exit_type = exit_params.get('type', 'time_based')
        hold_bars = exit_params.get('hold_bars', 5)
        profit_target = exit_params.get('profit_target', 0.02)
        stop_loss = exit_params.get('stop_loss', -0.01)

        for entry_idx in entry_points:
            if entry_idx + 1 >= len(returns):
                break
            exit_idx = entry_idx + 1
            if exit_type == 'time_based':
                exit_idx = min(entry_idx + hold_bars, len(returns) - 1)
            elif exit_type == 'profit_target' or exit_type == 'combined':
                for i in range(entry_idx + 1, min(entry_idx + hold_bars, len(returns))):
                    path_return = returns[i] - returns[entry_idx]
                    if path_return >= profit_target or path_return <= stop_loss:
                        exit_idx = i
                        break
                else:
                    exit_idx = min(entry_idx + hold_bars, len(returns) - 1)
            trade_return = returns[exit_idx] - returns[entry_idx]
            trades.append(trade_return)

        if len(trades) == 0:
            return np.nan
        trades = np.array(trades)
        mean_return = np.mean(trades)
        std_return = np.std(trades)
        if std_return == 0:
            return np.nan
        sharpe = mean_return / std_return
        return sharpe

    def evaluate_on_real(self, real_data: np.ndarray, strategies, asset_idx: int = 0) -> Dict[str, float]:
        """Evaluate strategies on real era-B data."""
        results = {}
        asset_start = asset_idx * 256
        asset_end = asset_start + 256
        for window in real_data:
            asset_returns = window[asset_start:asset_end]
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
        aggregated = {}
        for name, sharpes in results.items():
            if len(sharpes) > 0:
                aggregated[name] = np.mean(sharpes)
            else:
                aggregated[name] = np.nan
        return aggregated

    def evaluate_on_synthetic(self, synthetic_data: np.ndarray, strategies, asset_idx: int = 0) -> Dict[str, float]:
        """Evaluate strategies on synthetic data."""
        results = {}
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
        aggregated = {}
        for name, sharpes in results.items():
            if len(sharpes) > 0:
                aggregated[name] = np.mean(sharpes)
            else:
                aggregated[name] = np.nan
        return aggregated

    @staticmethod
    def _extract_entry_params(strategy) -> Dict:
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
        return {
            'type': strategy.exit_type,
            'hold_bars': strategy.parameters.get('hold_bars'),
            'profit_target': strategy.parameters.get('profit_target', strategy.parameters.get('target')),
            'stop_loss': strategy.parameters.get('stop_loss', strategy.parameters.get('loss')),
        }


def compute_rank_correlation(real_sharpes: np.ndarray, synthetic_sharpes: np.ndarray) -> Dict:
    """Compute rank correlation between real and synthetic performance."""
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
    """Run LSTM GAN evaluation pipeline (RQ2)."""
    logger.info("=" * 80)
    logger.info("DSM500 LSTM GAN Evaluation (RQ2) - Comparison Protocol")
    logger.info("=" * 80)

    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "outputs"

    # ===== Load data =====
    logger.info("\n[Step 1] Loading data...")
    parquet_path_b = output_dir / "windowed_data_era_b.parquet"
    if not parquet_path_b.exists():
        logger.error("Data file not found.")
        return False

    df_b = pd.read_parquet(parquet_path_b)
    step_cols = [c for c in df_b.columns if '_step_' in c]
    real_data_b = df_b[step_cols].values
    logger.info(f"  Era-B: {real_data_b.shape}")

    # ===== Load strategies =====
    logger.info("\n[Step 2] Loading strategies...")
    import importlib.util
    strategies_spec = importlib.util.spec_from_file_location(
        "strategies", Path(__file__).parent / "03_strategies.py"
    )
    strategies_mod = importlib.util.module_from_spec(strategies_spec)
    strategies_spec.loader.exec_module(strategies_mod)
    generate_strategies = strategies_mod.generate_strategies
    strategies = generate_strategies(n=500)
    logger.info(f"  Generated {len(strategies)} strategies")

    # ===== Load LSTM GAN =====
    logger.info("\n[Step 3] Loading LSTM GAN...")
    gan_spec = importlib.util.spec_from_file_location(
        "gan_generator_lstm", Path(__file__).parent / "04b_gan_generator_lstm_FIXED.py"
    )
    gan_mod = importlib.util.module_from_spec(gan_spec)
    gan_spec.loader.exec_module(gan_mod)
    LSTMGANTrainer = gan_mod.LSTMGANTrainer
    generate_synthetic_lstm = gan_mod.generate_synthetic_lstm

    # Try new fixed model first, fall back to old if not found
    gan_path_fixed = output_dir / "gan_model_lstm_fixed.pt"
    gan_path_old = output_dir / "gan_model_lstm.pt"
    gan_path = gan_path_fixed if gan_path_fixed.exists() else gan_path_old

    if gan_path.exists():
        logger.info("  Loading trained LSTM GAN...")
        gan_trainer = LSTMGANTrainer.load(str(gan_path))
    else:
        logger.error(f"  LSTM model not found")
        logger.info("  No pre-trained model found. Training new LSTM GAN...")
        # Train the LSTM GAN if not found
        train_gan_lstm = gan_mod.train_gan_lstm
        gan_trainer = train_gan_lstm(real_data_a, output_dir=str(output_dir), epochs=100)
        logger.info("  LSTM GAN training complete")

    logger.info("  Generating LSTM synthetic data...")
    synthetic_lstm = gan_trainer.generate(n_samples=1000)
    logger.info(f"  LSTM synthetic: {synthetic_lstm.shape}")

    # ===== Evaluate strategies =====
    logger.info("\n[Step 4] Evaluating strategies on real and LSTM synthetic data...")
    evaluator = StrategyEvaluator()
    asset_idx = 0

    logger.info("  Evaluating on real era-B data...")
    real_sharpes = evaluator.evaluate_on_real(real_data_b, strategies, asset_idx=asset_idx)

    logger.info("  Evaluating on LSTM synthetic data...")
    synthetic_lstm_sharpes = evaluator.evaluate_on_synthetic(synthetic_lstm, strategies, asset_idx=asset_idx)

    # ===== Compute rank correlation (RQ2 answer) =====
    logger.info("\n[Step 5] Computing rank correlation (RQ2 - LSTM)...")
    real_arr = np.array([real_sharpes.get(s.name, np.nan) for s in strategies])
    lstm_arr = np.array([synthetic_lstm_sharpes.get(s.name, np.nan) for s in strategies])
    corr_lstm = compute_rank_correlation(real_arr, lstm_arr)

    logger.info(f"\n  LSTM Spearman ρ: {corr_lstm['spearman_rho']:.4f} (p={corr_lstm['spearman_pval']:.4f})")

    # ===== Save results =====
    logger.info("\n[Step 6] Saving results...")
    results_df = pd.DataFrame({
        'strategy_name': [s.name for s in strategies],
        'real_sharpe': real_arr,
        'lstm_sharpe': lstm_arr,
    })
    csv_path = output_dir / "evaluation_results_lstm.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    summary = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'model_type': 'LSTM GAN',
        'rq2_question': 'RQ2 - LSTM: Spearman rank-correlation between LSTM synthetic and real Sharpe ratios',
        'rq2_answer_lstm': {
            'spearman_rho': float(corr_lstm['spearman_rho']),
            'spearman_pval': float(corr_lstm['spearman_pval']),
            'kendall_tau': float(corr_lstm['kendall_tau']),
            'n_strategies': corr_lstm['n_strategies'],
        },
        'comparison_note': 'Compare this ρ value to evaluation_summary.json (MLP ρ) per protocol in drafts/draft_gan_comparison_protocol.md',
    }
    json_path = output_dir / "evaluation_summary_lstm.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Saved: {json_path}")

    logger.info("\n" + "=" * 80)
    logger.info("LSTM Evaluation Complete (RQ2 answered)")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
