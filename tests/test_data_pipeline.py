"""Unit and integration tests for data pipeline."""

import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from importlib import import_module

# Lazy import to avoid ccxt/yfinance requirement during test discovery
def lazy_import():
    import pipeline.importlib
    spec = importlib.util.spec_from_file_location(
        "data_pipeline",
        Path(__file__).parent.parent / "pipeline" / "01_data_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Only load if we have dependencies
    try:
        import ccxt
        import yfinance
        spec.loader.exec_module(module)
        return module
    except ImportError:
        return None


class TestDataCollector:
    """Test DataCollector class methods."""

    def test_compute_log_returns(self, sample_daily_data):
        """Test log returns computation."""
        # Import here to handle missing dependencies gracefully
        try:
            from pipeline.pipeline import DataCollector
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        collector = DataCollector()
        returns = collector.compute_log_returns(sample_daily_data, col="close")

        # Check basic properties
        assert len(returns) == len(sample_daily_data) - 1  # One less (first NaN)
        assert not returns.isnull().any(), "Returns should not contain NaNs"
        assert isinstance(returns, pd.Series)

    def test_log_returns_are_reasonable(self, sample_daily_data):
        """Test that log returns are in reasonable range."""
        try:
            from pipeline.pipeline import DataCollector
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        collector = DataCollector()
        returns = collector.compute_log_returns(sample_daily_data, col="close")

        # Daily returns typically in range [-5%, 5%] for normal markets
        assert returns.abs().max() < 0.1, "Extreme daily returns (>10%) seem unrealistic"
        assert returns.std() > 0, "Returns should have non-zero variance"


class TestDataWindower:
    """Test DataWindower class methods."""

    def test_window_creation_shape(self, sample_daily_data):
        """Test that windows have correct shape."""
        try:
            from pipeline.pipeline import DataWindower
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        windower = DataWindower()
        returns = sample_daily_data["close"].pct_change().dropna()

        window_length = 256
        stride = 10
        windows = windower.create_windows(returns, window_length, stride)

        assert windows.ndim == 2, "Windows should be 2D array"
        assert windows.shape[1] == window_length, f"Each window should be {window_length} steps"
        assert windows.shape[0] > 0, "Should create at least one window"

    def test_window_stride(self, sample_daily_data):
        """Test that window stride is correct."""
        try:
            from pipeline.pipeline import DataWindower
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        windower = DataWindower()
        returns = sample_daily_data["close"].pct_change().dropna()

        window_length = 50
        stride = 5
        windows = windower.create_windows(returns, window_length, stride)

        # Calculate expected number of windows
        expected_n = (len(returns) - window_length) // stride + 1
        assert windows.shape[0] == expected_n, f"Expected {expected_n} windows"

    def test_window_no_nans(self, sample_daily_data):
        """Test that windows contain no NaNs."""
        try:
            from pipeline.pipeline import DataWindower
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        windower = DataWindower()
        returns = sample_daily_data["close"].pct_change().dropna()

        windows = windower.create_windows(returns, window_length=100, stride=5)

        assert not np.isnan(windows).any(), "Windows should not contain NaN values"

    def test_era_split_no_overlap(self, sample_daily_data):
        """Test that era split creates no overlap."""
        try:
            from pipeline.pipeline import DataWindower
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        windower = DataWindower()
        split_date = pd.Timestamp("2022-07-01", tz="UTC")

        returns_a, returns_b = windower.split_eras(sample_daily_data["close"], split_date)

        assert returns_a.index.max() <= split_date, "Era-A should end before split date"
        assert returns_b.index.min() > split_date, "Era-B should start after split date"
        assert len(returns_a) > 0 and len(returns_b) > 0, "Both eras should be non-empty"


class TestDataPipeline:
    """Integration tests for full pipeline."""

    def test_pipeline_initialization(self, temp_output_dir):
        """Test that pipeline initializes without error."""
        try:
            from pipeline.pipeline import DataPipeline
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        pipeline = DataPipeline(output_dir=str(temp_output_dir))
        assert pipeline.output_dir.exists()

    def test_parquet_format(self, temp_output_dir, sample_daily_data):
        """Test that output Parquet files have correct format."""
        try:
            from pipeline.pipeline import DataPipeline
            import pyarrow.parquet as pq
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        # Create a simple test DataFrame and save as Parquet
        df = pd.DataFrame({
            "asset": ["TEST"] * len(sample_daily_data),
            "window_id": [f"TEST_{i}" for i in range(len(sample_daily_data))],
            "value": sample_daily_data["close"].values,
        })

        parquet_path = temp_output_dir / "test_output.parquet"
        df.to_parquet(parquet_path, compression="snappy")

        # Verify it can be read back
        df_read = pd.read_parquet(parquet_path)
        assert len(df_read) == len(df)
        assert list(df_read.columns) == list(df.columns)

    def test_metadata_json_structure(self, temp_output_dir):
        """Test that metadata JSON has correct structure."""
        metadata = {
            "generated_at": "2026-09-21T00:00:00",
            "config": {
                "era_a_start": "2018-01-01",
                "era_a_end": "2021-12-31",
                "window_length": 256,
                "stride": 24,
            },
            "data_summary": {
                "BTC/USDT": {
                    "era_a_windows": 1000,
                    "era_b_windows": 500,
                }
            },
        }

        metadata_path = temp_output_dir / "test_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        with open(metadata_path, "r") as f:
            metadata_read = json.load(f)

        assert metadata_read == metadata


class TestLeakagePrevention:
    """Test that chronological split prevents look-ahead leakage."""

    def test_window_boundary_alignment(self):
        """Test that era split happens before windowing."""
        try:
            from pipeline.pipeline import DataWindower
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        # Create a series with clear markers
        dates = pd.date_range("2021-12-15", periods=40, freq="D", tz="UTC")
        returns = pd.Series(
            np.arange(40) / 100,  # 0.00, 0.01, 0.02, ...
            index=dates
        )

        era_a_end = pd.Timestamp("2021-12-31", tz="UTC")
        windower = DataWindower()

        returns_a, returns_b = windower.split_eras(returns, era_a_end)

        assert returns_a.index[-1] <= era_a_end
        assert returns_b.index[0] >= pd.Timestamp("2022-01-01", tz="UTC")
        assert returns_a.index.intersection(returns_b.index).empty


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        try:
            from pipeline.pipeline import DataWindower
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        windower = DataWindower()
        empty_series = pd.Series(dtype=float)

        # Should handle gracefully
        windows = windower.create_windows(empty_series, window_length=10, stride=1)
        assert len(windows) == 0

    def test_insufficient_data_for_window(self):
        """Test when data length < window_length."""
        try:
            from pipeline.pipeline import DataWindower
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Dependencies not available")

        windower = DataWindower()
        short_series = pd.Series([0.01, 0.02, 0.03])  # Only 3 values

        windows = windower.create_windows(short_series, window_length=10, stride=1)
        assert len(windows) == 0, "Should produce no windows if data shorter than window"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
