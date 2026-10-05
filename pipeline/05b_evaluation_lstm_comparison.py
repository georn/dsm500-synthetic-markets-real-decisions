#!/usr/bin/env python3
"""LSTM GAN evaluation for RQ2, using the same protocol as 05_evaluation.py.

Imports StrategyEvaluator, compute_rank_correlation and asset_windows from
05_evaluation.py so that both evaluations share one backtester. Evaluates the saved
LSTM GAN (outputs/gan_model_lstm_fixed.pt) on 1,000 synthetic paths against the real
era-B USD/EUR windows.

Outputs:
- outputs/evaluation_results_lstm.csv (per-rule Sharpe ratios)
- outputs/evaluation_summary_lstm.json (Spearman rho for the LSTM GAN)

Usage:
    python3 pipeline/05b_evaluation_lstm_comparison.py
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


import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("evaluation", Path(__file__).parent / "05_evaluation.py")
_evaluation = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)
StrategyEvaluator = _evaluation.StrategyEvaluator
compute_rank_correlation = _evaluation.compute_rank_correlation
asset_windows = _evaluation.asset_windows


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
    real_data_b = asset_windows(df_b, "EUR=X")
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
    assert len({s.name for s in strategies}) == len(strategies), "results are keyed by strategy name"
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

    # Prefer the USD/EUR-only model (04b_gan_generator_lstm_FIXED.py --eur-only), then the
    # as-run model trained on the pooled three-asset rows.
    candidates = [output_dir / "gan_model_lstm_eur.pt", output_dir / "gan_model_lstm_fixed.pt"]
    gan_path = next((p for p in candidates if p.exists()), candidates[-1])

    if gan_path.exists():
        logger.info("  Loading trained LSTM GAN...")
        gan_trainer = LSTMGANTrainer.load(str(gan_path))
    else:
        logger.error("  LSTM model not found; run 04b_gan_generator_lstm_FIXED.py first")
        return False

    logger.info("  Generating LSTM synthetic data...")
    import torch
    torch.manual_seed(42)
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
