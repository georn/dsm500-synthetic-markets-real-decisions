#!/usr/bin/env python3
"""
Walk-forward baseline for RQ2.

Each rule is backtested on the real era-A USD/EUR windows (the single history a
practitioner would have) and on the real era-B windows. The Spearman correlation
between era-A and era-B mean Sharpe ratios is the ranking accuracy of plain
historical validation, the benchmark the synthetic generators must beat.

Replaces an earlier version that simulated era-A predictions by adding noise to the
era-B results (archived in archive/pipeline_as_run_2026-10-05/).

Outputs:
- outputs/walkforward_results.json
- outputs/walkforward_comparison.csv

Usage:
    python3 pipeline/07_walkforward_baseline.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).parent
OUT = PIPELINE.parent / "outputs"


def load(name, filename):
    """Import a numbered pipeline script as a module."""
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Backtest every rule on real era-A and era-B USD/EUR windows and rank-correlate the two."""
    evaluation = load("evaluation", "05_evaluation.py")
    strategies = load("strategies", "03_strategies.py").generate_strategies(n=500)

    era_a = evaluation.asset_windows(pd.read_parquet(OUT / "windowed_data_era_a.parquet"), "EUR=X")
    era_b = evaluation.asset_windows(pd.read_parquet(OUT / "windowed_data_era_b.parquet"), "EUR=X")

    evaluator = evaluation.StrategyEvaluator()
    sharpe_a = evaluator.evaluate_on_real(era_a, strategies)
    sharpe_b = evaluator.evaluate_on_real(era_b, strategies)

    names = [s.name for s in strategies]
    a = np.array([sharpe_a.get(n, np.nan) for n in names])
    b = np.array([sharpe_b.get(n, np.nan) for n in names])
    corr = evaluation.compute_rank_correlation(b, a)

    pd.DataFrame({"strategy_name": names, "era_a_sharpe": a, "era_b_sharpe": b}).to_csv(
        OUT / "walkforward_comparison.csv", index=False)
    result = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "method": "Spearman rho between era-A (in-sample) and era-B (out-of-sample) mean Sharpe, USD/EUR",
        "era_a_windows": int(len(era_a)),
        "era_b_windows": int(len(era_b)),
        **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in corr.items()},
    }
    (OUT / "walkforward_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
