#!/usr/bin/env python3
"""
Technical Trading Rule Generator for DSM500 CW2

Generates a universe of ~500 technical trading rules by combining:
- Entry signals: MA crossover, momentum, RSI, Bollinger Bands
- Exit signals: time-based hold, profit-taking, stop-loss
- Parameter combinations: sweeps across window lengths, thresholds, etc.

Output: List of Strategy objects with entry/exit logic and parameters.
Used by 05_evaluation.py to validate strategies on synthetic vs real data.

Usage:
    from strategies import generate_strategies
    rules = generate_strategies(n=500)
    for rule in rules:
        print(f"{rule.name}: {rule.entry_type} → {rule.exit_type}")
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Represents a single trading rule."""
    name: str
    entry_type: str
    exit_type: str
    parameters: Dict[str, float]

    def __repr__(self):
        return f"Strategy({self.name}, params={len(self.parameters)})"


class EntrySignals:
    """Entry signal functions for trading strategies."""

    @staticmethod
    def ma_crossover(window_fast: int, window_slow: int) -> Dict[str, Any]:
        """Fast MA crosses above slow MA (bullish signal)."""
        return {
            "type": "ma_crossover",
            "ma_fast": window_fast,
            "ma_slow": window_slow,
        }

    @staticmethod
    def momentum(threshold: float) -> Dict[str, Any]:
        """Return exceeds momentum threshold."""
        return {
            "type": "momentum",
            "threshold": threshold,
        }

    @staticmethod
    def rsi(threshold: float) -> Dict[str, Any]:
        """RSI falls below threshold (oversold condition)."""
        return {
            "type": "rsi",
            "threshold": threshold,
        }

    @staticmethod
    def bollinger_break(window: int, std_dev: float) -> Dict[str, Any]:
        """Price breaks below lower Bollinger Band."""
        return {
            "type": "bollinger",
            "window": window,
            "std_dev": std_dev,
        }


class ExitSignals:
    """Exit signal functions for trading strategies."""

    @staticmethod
    def time_based(hold_bars: int) -> Dict[str, Any]:
        """Exit after holding for N bars."""
        return {
            "type": "time_based",
            "hold_bars": hold_bars,
        }

    @staticmethod
    def profit_target(target: float) -> Dict[str, Any]:
        """Exit when profit reaches target (e.g., +2%)."""
        return {
            "type": "profit_target",
            "target": target,
        }

    @staticmethod
    def stop_loss(loss: float) -> Dict[str, Any]:
        """Exit when loss reaches threshold (e.g., -1%)."""
        return {
            "type": "stop_loss",
            "loss": loss,
        }

    @staticmethod
    def combined(hold_bars: int, profit_target: float, stop_loss: float) -> Dict[str, Any]:
        """Exit on time, profit, or loss (whichever comes first)."""
        return {
            "type": "combined",
            "hold_bars": hold_bars,
            "profit_target": profit_target,
            "stop_loss": stop_loss,
        }


class StrategyGenerator:
    """Generate a universe of trading strategies via parameter sweep."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)

    def generate(self, n_rules: int = 500) -> List[Strategy]:
        """
        Generate n_rules strategies by sweeping parameter combinations.

        Args:
            n_rules: Target number of rules to generate

        Returns:
            List of Strategy objects
        """
        strategies = []
        strategy_id = 0

        # Entry signal parameter ranges
        ma_fast_range = [10, 15, 20, 25, 30]
        ma_slow_range = [50, 75, 100, 150]
        momentum_range = [0.01, 0.02, 0.03, 0.05]
        rsi_threshold_range = [25, 30, 35, 40]
        bollinger_window_range = [20, 30, 40]
        bollinger_std_range = [1.5, 2.0, 2.5]

        # Exit signal parameter ranges
        hold_bars_range = [5, 10, 15, 20]
        profit_target_range = [0.01, 0.02, 0.03, 0.05, 0.10]
        stop_loss_range = [-0.01, -0.02, -0.03, -0.05]

        # Generate MA Crossover strategies
        for ma_fast in ma_fast_range:
            for ma_slow in ma_slow_range:
                if ma_fast >= ma_slow:
                    continue
                for hold in hold_bars_range:
                    entry = EntrySignals.ma_crossover(ma_fast, ma_slow)
                    exit_sig = ExitSignals.time_based(hold)
                    strategy = Strategy(
                        name=f"MA_{ma_fast}_{ma_slow}_Hold{hold}",
                        entry_type=entry["type"],
                        exit_type=exit_sig["type"],
                        parameters={**entry, **exit_sig},
                    )
                    strategies.append(strategy)
                    strategy_id += 1

                    if len(strategies) >= n_rules:
                        return strategies[:n_rules]

        # Generate MA + Profit Target strategies
        for ma_fast in ma_fast_range:
            for ma_slow in ma_slow_range:
                if ma_fast >= ma_slow:
                    continue
                for profit in profit_target_range:
                    for stop in stop_loss_range:
                        entry = EntrySignals.ma_crossover(ma_fast, ma_slow)
                        exit_sig = ExitSignals.combined(15, profit, stop)
                        strategy = Strategy(
                            name=f"MA_{ma_fast}_{ma_slow}_Profit{profit:.2f}_Stop{abs(stop):.2f}",
                            entry_type=entry["type"],
                            exit_type=exit_sig["type"],
                            parameters={**entry, **exit_sig},
                        )
                        strategies.append(strategy)
                        strategy_id += 1

                        if len(strategies) >= n_rules:
                            return strategies[:n_rules]

        # Generate Momentum strategies
        for momentum in momentum_range:
            for hold in hold_bars_range:
                entry = EntrySignals.momentum(momentum)
                exit_sig = ExitSignals.time_based(hold)
                strategy = Strategy(
                    name=f"Momentum_{momentum:.3f}_Hold{hold}",
                    entry_type=entry["type"],
                    exit_type=exit_sig["type"],
                    parameters={**entry, **exit_sig},
                )
                strategies.append(strategy)
                strategy_id += 1

                if len(strategies) >= n_rules:
                    return strategies[:n_rules]

        # Generate RSI strategies
        for rsi_threshold in rsi_threshold_range:
            for profit in profit_target_range:
                for stop in stop_loss_range:
                    entry = EntrySignals.rsi(rsi_threshold)
                    exit_sig = ExitSignals.combined(10, profit, stop)
                    strategy = Strategy(
                        name=f"RSI_{rsi_threshold}_Profit{profit:.2f}_Stop{abs(stop):.2f}",
                        entry_type=entry["type"],
                        exit_type=exit_sig["type"],
                        parameters={**entry, **exit_sig},
                    )
                    strategies.append(strategy)
                    strategy_id += 1

                    if len(strategies) >= n_rules:
                        return strategies[:n_rules]

        # Generate Bollinger Band strategies
        for bb_window in bollinger_window_range:
            for bb_std in bollinger_std_range:
                for hold in hold_bars_range:
                    entry = EntrySignals.bollinger_break(bb_window, bb_std)
                    exit_sig = ExitSignals.time_based(hold)
                    strategy = Strategy(
                        name=f"BB_{bb_window}_{bb_std:.1f}_Hold{hold}",
                        entry_type=entry["type"],
                        exit_type=exit_sig["type"],
                        parameters={**entry, **exit_sig},
                    )
                    strategies.append(strategy)
                    strategy_id += 1

                    if len(strategies) >= n_rules:
                        return strategies[:n_rules]

        logger.info(f"Generated {len(strategies)} strategies")
        return strategies


def _assert_unique_names(strategies: List[Strategy]) -> List[Strategy]:
    """Raise if two strategies share a name; downstream results are stored by name."""
    names = [s.name for s in strategies]
    duplicates = len(names) - len(set(names))
    if duplicates:
        raise ValueError(f"{duplicates} duplicate strategy names; results are keyed by name")
    return strategies


def generate_strategies(n: int = 500) -> List[Strategy]:
    """
    Convenience function to generate a strategy universe.

    Args:
        n: Number of strategies to generate

    Returns:
        List of Strategy objects
    """
    generator = StrategyGenerator()
    return _assert_unique_names(generator.generate(n_rules=n))


if __name__ == "__main__":
    strategies = generate_strategies(500)
    logger.info(f"Generated {len(strategies)} strategies")
    logger.info(f"Entry types: {set(s.entry_type for s in strategies)}")
    logger.info(f"Exit types: {set(s.exit_type for s in strategies)}")

    # Print a few examples
    for strategy in strategies[:5]:
        logger.info(f"  {strategy.name}: {strategy.parameters}")
