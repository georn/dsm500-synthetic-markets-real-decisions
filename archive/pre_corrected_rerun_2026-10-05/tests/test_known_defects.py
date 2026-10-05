"""Regression tests for defects found and fixed during the 2026-10-05 review."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "evaluation", Path(__file__).parent.parent / "pipeline" / "05_evaluation.py"
)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def test_sharpe_uses_holding_period_daily_returns():
    returns = np.zeros(30)
    returns[[0, 1, 2, 3]] = [0.05, 0.01, 0.02, 0.01]
    returns[[10, 11, 12, 13]] = [0.05, -0.01, 0.00, -0.01]
    position = np.zeros(30)
    position[1:4] = position[11:14] = 1.0
    daily = position * returns
    expected_sharpe = daily.mean() / daily.std()

    sharpe = evaluation.StrategyEvaluator().apply_strategy(
        returns,
        entry_params={"type": "momentum", "threshold": 0.03},
        exit_params={"type": "time_based", "hold_bars": 3},
    )

    assert sharpe == pytest.approx(expected_sharpe)


def test_garch_long_run_volatility_matches_sample():
    spec = importlib.util.spec_from_file_location("baselines", Path(__file__).parent.parent / "pipeline" / "02_baselines.py")
    baselines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines)
    rng = np.random.default_rng(0)
    sample = np.r_[rng.normal(0, 0.005, 2000), [np.nan] * 10]
    model = baselines.GARCHModel().fit(sample)
    long_run_var = model.omega / (1 - model.alpha - model.beta)
    assert np.sqrt(long_run_var) == pytest.approx(np.nanstd(sample), rel=1e-6)
    np.random.seed(0)
    assert np.isfinite(model.generate(n_samples=5, n_steps=256)).all()


def test_rsi_style_entries_ignore_future_returns():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.005, 120)
    altered = returns.copy()
    altered[80:] = rng.normal(0, 0.05, 40)  # very different future
    params = {"type": "rsi", "threshold": 30}
    exit_params = {"type": "time_based", "hold_bars": 1}
    evaluator = evaluation.StrategyEvaluator()
    # A rule that only sees the first 80 days must give the same result however the future looks.
    assert evaluator.apply_strategy(returns[:80], params, exit_params) == pytest.approx(
        evaluator.apply_strategy(altered[:80], params, exit_params))
    past_quantile = pd.Series(returns).expanding(min_periods=20).quantile(0.30).shift(1)
    altered_quantile = pd.Series(altered).expanding(min_periods=20).quantile(0.30).shift(1)
    assert np.allclose(past_quantile[:81].dropna(), altered_quantile[:81].dropna())
