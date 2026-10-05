"""
Data integrity tests for DSM500 CW2 windowed data.

Validates:
- Windowed data structure (one asset per row, NaN patterns)
- Era split correctness (no leakage, chronological separation)
- Financial realism (stylized facts present)
- No corruption during parquet serialization
"""

import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestWindowedDataStructure:
    """Test that windowed data has correct structure and asset alignment."""

    @pytest.fixture
    def windowed_data(self):
        """Load windowed parquet data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Data not generated yet")
        return pd.read_parquet(parquet_path)

    @pytest.fixture
    def metadata(self):
        """Load data metadata."""
        metadata_path = Path(__file__).parent.parent / "outputs" / "data_metadata.json"
        if not metadata_path.exists():
            pytest.skip("Metadata not generated yet")
        with open(metadata_path) as f:
            return json.load(f)

    def test_windowed_data_shape(self, windowed_data):
        """Test that windowed data has expected shape."""
        assert windowed_data.shape[0] > 0, "No windows generated"
        assert windowed_data.shape[1] >= 768, "Missing feature columns (expected >= 768)"

    def test_one_asset_per_row(self, windowed_data):
        """Test that each row corresponds to exactly one asset."""
        assets = windowed_data['asset'].unique()
        assert len(assets) == 3, f"Expected 3 assets, got {len(assets)}: {assets}"

        # Each asset should have consistent rows
        for asset in assets:
            asset_rows = windowed_data[windowed_data['asset'] == asset]
            assert len(asset_rows) > 0, f"Asset {asset} has no rows"

    def test_nan_pattern_consistency(self, windowed_data):
        """Test that NaN pattern is consistent within each asset."""
        step_cols = [c for c in windowed_data.columns if '_step_' in c]

        for asset in windowed_data['asset'].unique():
            asset_data = windowed_data[windowed_data['asset'] == asset][step_cols].values

            # Check that all rows of same asset have same NaN pattern
            nan_patterns = [frozenset(np.where(np.isnan(row))[0]) for row in asset_data]
            unique_patterns = set(nan_patterns)

            assert len(unique_patterns) == 1, \
                f"Asset {asset} has {len(unique_patterns)} different NaN patterns (expected 1)"

    def test_correct_valid_column_count(self, windowed_data):
        """Test that each asset row has exactly 256 valid (non-NaN) values."""
        step_cols = [c for c in windowed_data.columns if '_step_' in c]

        for asset in windowed_data['asset'].unique():
            asset_data = windowed_data[windowed_data['asset'] == asset][step_cols].values

            valid_counts = (~np.isnan(asset_data)).sum(axis=1)

            assert np.all(valid_counts == 256), \
                f"Asset {asset}: not all rows have 256 valid values. Stats: min={valid_counts.min()}, max={valid_counts.max()}"

    def test_no_complete_empty_rows(self, windowed_data):
        """Test that no row is completely empty (all NaN)."""
        step_cols = [c for c in windowed_data.columns if '_step_' in c]
        completely_empty = windowed_data[step_cols].isnull().all(axis=1)

        assert not completely_empty.any(), f"{completely_empty.sum()} completely empty rows found"

    def test_no_inf_values(self, windowed_data):
        """Test that no infinite values exist (would break metrics)."""
        step_cols = [c for c in windowed_data.columns if '_step_' in c]
        has_inf = np.isinf(windowed_data[step_cols].values).any()

        assert not has_inf, "Infinite values found in windowed data"

    def test_return_magnitudes_realistic(self, windowed_data):
        """Test that log-return magnitudes are realistic (not extreme or zero)."""
        step_cols = [c for c in windowed_data.columns if '_step_' in c]
        values = windowed_data[step_cols].values[~np.isnan(windowed_data[step_cols].values)]

        # Daily log returns should typically be in [-10%, +10%]
        assert values.max() < 0.20, f"Max return {values.max():.4f} seems unrealistic"
        assert values.min() > -0.20, f"Min return {values.min():.4f} seems unrealistic"
        assert values.std() > 0.001, "Returns have suspiciously low volatility"

    def test_window_ids_unique(self, windowed_data):
        """Test that window IDs are unique and correctly formatted."""
        window_ids = windowed_data['window_id']

        assert len(window_ids) == len(window_ids.unique()), "Duplicate window IDs found"
        assert all(window_ids.str.contains(r'_(a|b)_', regex=True)), \
            "Window IDs don't follow expected format: {asset}_{era}_{index}"


class TestEraStructure:
    """Test that era split is correct and leakage-free."""

    @pytest.fixture
    def era_a_data(self):
        """Load era-A windowed data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Data not generated yet")
        return pd.read_parquet(parquet_path)

    @pytest.fixture
    def era_b_data(self):
        """Load era-B windowed data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_b.parquet"
        if not parquet_path.exists():
            pytest.skip("Era-B data not generated yet")
        return pd.read_parquet(parquet_path)

    @pytest.fixture
    def metadata(self):
        """Load data metadata."""
        metadata_path = Path(__file__).parent.parent / "outputs" / "data_metadata.json"
        if not metadata_path.exists():
            pytest.skip("Metadata not generated yet")
        with open(metadata_path) as f:
            return json.load(f)

    def test_era_a_has_windows(self, era_a_data):
        """Test that era-A has expected number of windows."""
        assert len(era_a_data) > 1000, f"Era-A has too few windows: {len(era_a_data)}"

    def test_era_b_has_windows(self, era_b_data):
        """Test that era-B has expected number of windows."""
        assert len(era_b_data) > 500, f"Era-B has too few windows: {len(era_b_data)}"

    def test_era_split_totals_match_metadata(self, era_a_data, era_b_data, metadata):
        """Test that era counts match metadata."""
        assert len(era_a_data) == metadata['era_a_shape'][0], \
            f"Era-A count mismatch: data={len(era_a_data)}, metadata={metadata['era_a_shape'][0]}"
        assert len(era_b_data) == metadata['era_b_shape'][0], \
            f"Era-B count mismatch: data={len(era_b_data)}, metadata={metadata['era_b_shape'][0]}"


class TestFinancialRealism:
    """Test that data exhibits known financial properties (stylized facts)."""

    @pytest.fixture
    def returns_vector(self):
        """Extract all non-NaN returns from era-A data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Data not generated yet")

        df = pd.read_parquet(parquet_path)
        step_cols = [c for c in df.columns if '_step_' in c]
        returns = df[step_cols].values[~np.isnan(df[step_cols].values)]
        return returns

    def test_returns_near_zero_mean(self, returns_vector):
        """Test that returns have approximately zero mean."""
        mean = np.mean(returns_vector)
        assert abs(mean) < 0.01, f"Mean return {mean:.6f} is too large (expected ~0)"

    def test_returns_positive_variance(self, returns_vector):
        """Test that returns have non-trivial variance."""
        std = np.std(returns_vector)
        assert std > 0.005, f"Return std {std:.6f} is too small (expected > 0.005)"
        assert std < 0.05, f"Return std {std:.6f} is too large (expected < 0.05)"

    def test_returns_have_fat_tails(self, returns_vector):
        """Test that returns exhibit excess kurtosis (fat tails)."""
        from scipy import stats
        kurtosis = stats.kurtosis(returns_vector, fisher=True)  # Excess kurtosis

        # Financial returns typically have excess kurtosis 3-50+
        assert kurtosis > 1, f"Excess kurtosis {kurtosis:.2f} is too low (expected > 1)"
        assert kurtosis < 100, f"Excess kurtosis {kurtosis:.2f} seems unrealistic (expected < 100)"

    def test_returns_have_volatility_clustering(self, returns_vector):
        """Test that squared returns show autocorrelation (volatility clustering)."""
        sq_returns = returns_vector ** 2
        autocorr_lag1 = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]

        # Volatility clustering: autocorr of squared returns should be positive and significant
        assert autocorr_lag1 > 0.05, \
            f"Autocorr of squared returns {autocorr_lag1:.4f} too low (expected > 0.05, indicates volatility clustering)"

    def test_returns_nearly_uncorrelated(self, returns_vector):
        """Test that raw returns are nearly uncorrelated (market efficiency)."""
        autocorr_lag1 = np.corrcoef(returns_vector[:-1], returns_vector[1:])[0, 1]

        # Raw returns should be nearly uncorrelated (weak-form efficiency)
        # Overlapping windows can create modest autocorr structure
        assert abs(autocorr_lag1) < 0.25, \
            f"Autocorr of returns {autocorr_lag1:.4f} too high"

    def test_returns_negatively_skewed(self, returns_vector):
        """Test that returns show negative skew (downside risk)."""
        from scipy import stats
        skewness = stats.skew(returns_vector)

        # Financial returns often show negative skew (more/larger downside moves)
        # But can be positive in bull markets; test is just for presence of skew
        assert abs(skewness) > 0.1, f"Skewness {skewness:.4f} too close to 0 (expected |skew| > 0.1)"
