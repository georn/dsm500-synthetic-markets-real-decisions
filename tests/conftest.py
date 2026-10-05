"""Pytest fixtures for data pipeline tests."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


def _alias_numbered_modules():
    """Expose numbered pipeline scripts under the import names the tests use."""
    import importlib.util
    import types

    def load(alias, filename):
        if alias in sys.modules:
            return sys.modules[alias]
        spec = importlib.util.spec_from_file_location(alias, PIPELINE_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module

    load("baselines", "02_baselines.py")
    load("strategies", "03_strategies.py")
    load("gan_generator", "04_gan_generator.py")
    package = sys.modules.setdefault("pipeline", types.ModuleType("pipeline"))
    package.__path__ = [str(PIPELINE_DIR)]
    package.pipeline = load("pipeline.pipeline", "01_data_pipeline.py")


_alias_numbered_modules()


@pytest.fixture
def sample_hourly_data():
    """Generate sample hourly OHLCV data (10 days)."""
    dates = pd.date_range("2023-01-01", periods=240, freq="1h", tz="UTC")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(240) * 100)

    df = pd.DataFrame({
        "open": close + np.random.randn(240) * 50,
        "high": close + np.abs(np.random.randn(240) * 75),
        "low": close - np.abs(np.random.randn(240) * 75),
        "close": close,
        "volume": np.random.randint(100, 1000, 240),
    }, index=dates)
    return df


@pytest.fixture
def sample_daily_data():
    """Generate sample daily close price data (2 years)."""
    dates = pd.date_range("2022-01-01", periods=730, freq="D", tz="UTC")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(730) * 2)

    df = pd.DataFrame({
        "close": close,
    }, index=dates)
    return df


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for test artifacts."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
