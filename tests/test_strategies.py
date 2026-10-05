"""
Lean smoke tests for strategy generator.

Tests validate:
- Strategy objects are generated correctly
- Parameters are valid (not NaN, within reasonable bounds)
- Strategy universe has diversity (multiple entry/exit combinations)
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

try:
    from strategies import generate_strategies, Strategy
except ImportError:
    pytest.skip("Strategies module not available", allow_module_level=True)


class TestStrategyGeneration:
    """Test that strategy generation produces valid rules."""

    def test_strategies_generate_expected_count(self):
        """Test that requested number of strategies are generated."""
        for n in [10, 50, 100, 500]:
            strategies = generate_strategies(n=n)
            assert len(strategies) == n, f"Expected {n} strategies, got {len(strategies)}"

    def test_strategy_objects_have_required_fields(self):
        """Test that each strategy has required fields."""
        strategies = generate_strategies(n=20)

        for strategy in strategies:
            assert isinstance(strategy, Strategy)
            assert hasattr(strategy, 'name')
            assert hasattr(strategy, 'entry_type')
            assert hasattr(strategy, 'exit_type')
            assert hasattr(strategy, 'parameters')
            assert isinstance(strategy.name, str)
            assert isinstance(strategy.parameters, dict)

    def test_strategy_parameters_are_finite(self):
        """Test that all parameters are finite (not NaN or Inf)."""
        strategies = generate_strategies(n=50)

        for strategy in strategies:
            for key, value in strategy.parameters.items():
                if isinstance(value, str):
                    continue
                assert not np.isnan(value), \
                    f"Strategy {strategy.name} has NaN parameter: {key}={value}"
                assert not np.isinf(value), \
                    f"Strategy {strategy.name} has Inf parameter: {key}={value}"
                assert isinstance(value, (int, float)), \
                    f"Strategy {strategy.name} parameter {key} has non-numeric type: {type(value)}"

    def test_strategy_universe_has_diversity(self):
        """Test that strategy universe includes multiple entry/exit types."""
        strategies = generate_strategies(n=500)

        entry_types = set(s.entry_type for s in strategies)
        exit_types = set(s.exit_type for s in strategies)

        # Should have at least 3 different entry types
        assert len(entry_types) >= 3, \
            f"Expected >= 3 entry types, got {len(entry_types)}: {entry_types}"

        # Should have at least 2 different exit types
        assert len(exit_types) >= 2, \
            f"Expected >= 2 exit types, got {len(exit_types)}: {exit_types}"

    def test_strategy_names_are_unique(self):
        """Test that strategy names are unique (or nearly so)."""
        strategies = generate_strategies(n=100)
        names = [s.name for s in strategies]

        # Allow up to 10% duplication (due to parameter overlap)
        unique_names = len(set(names))
        expected_min = int(0.9 * len(strategies))

        assert unique_names >= expected_min, \
            f"Only {unique_names}/{len(strategies)} unique names"

    def test_ma_strategies_have_valid_windows(self):
        """Test that MA strategies have fast < slow windows."""
        strategies = generate_strategies(n=100)
        ma_strategies = [s for s in strategies if s.entry_type == 'ma_crossover']

        assert len(ma_strategies) > 0, "No MA crossover strategies generated"

        for strategy in ma_strategies:
            ma_fast = strategy.parameters.get('ma_fast')
            ma_slow = strategy.parameters.get('ma_slow')

            if ma_fast is not None and ma_slow is not None:
                assert ma_fast < ma_slow, \
                    f"Strategy {strategy.name}: fast MA {ma_fast} >= slow MA {ma_slow}"

    def test_exit_signal_parameters_in_bounds(self):
        """Test that exit signal parameters are in realistic bounds."""
        strategies = generate_strategies(n=100)

        for strategy in strategies:
            # Check hold bars if present
            hold_bars = strategy.parameters.get('hold_bars')
            if hold_bars is not None:
                assert 1 < hold_bars < 100, \
                    f"Hold bars {hold_bars} out of bounds [1, 100]"

            # Check profit target if present
            profit_target = strategy.parameters.get('profit_target')
            if profit_target is not None:
                assert 0 < profit_target < 0.5, \
                    f"Profit target {profit_target} out of bounds (0, 0.5)"

            # Check stop loss if present
            stop_loss = strategy.parameters.get('stop_loss')
            if stop_loss is not None:
                assert -0.5 < stop_loss < 0, \
                    f"Stop loss {stop_loss} out of bounds (-0.5, 0)"

    def test_strategy_reproducibility(self):
        """Test that same seed produces same strategies."""
        from strategies import StrategyGenerator

        gen1 = StrategyGenerator(seed=42)
        gen2 = StrategyGenerator(seed=42)

        strategies1 = gen1.generate(n_rules=20)
        strategies2 = gen2.generate(n_rules=20)

        # Should have same names and parameters
        assert len(strategies1) == len(strategies2)
        for s1, s2 in zip(strategies1, strategies2):
            assert s1.name == s2.name
            assert s1.entry_type == s2.entry_type
            assert s1.exit_type == s2.exit_type
            assert s1.parameters == s2.parameters

    def test_strategies_are_json_serializable(self):
        """Test that strategies can be serialized (for saving/loading)."""
        import json
        strategies = generate_strategies(n=10)

        # Convert to JSON-friendly format
        strategies_dict = [
            {
                'name': s.name,
                'entry_type': s.entry_type,
                'exit_type': s.exit_type,
                'parameters': s.parameters,
            }
            for s in strategies
        ]

        # Should be JSON serializable
        json_str = json.dumps(strategies_dict)
        reloaded = json.loads(json_str)

        assert len(reloaded) == len(strategies)
        assert all('name' in r for r in reloaded)
