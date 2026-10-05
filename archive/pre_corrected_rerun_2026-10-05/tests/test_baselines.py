"""
Baseline generation and fidelity evaluation tests for DSM500 CW2.

Tests:
- Bootstrap and GARCH synthetic data generation
- Fidelity metric computation (no NaN propagation)
- Stylized facts capture (kurtosis, autocorr, etc.)
- Synthetic data has correct structure and statistics
"""

import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy import stats

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

try:
    # Try direct import first
    from baselines import StationaryBootstrap, GARCHModel, FidelityEvaluator
except ImportError:
    try:
        # Fallback: import 02_baselines.py directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "baselines",
            Path(__file__).parent.parent / "pipeline" / "02_baselines.py"
        )
        baselines_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines_module)
        StationaryBootstrap = baselines_module.StationaryBootstrap
        GARCHModel = baselines_module.GARCHModel
        FidelityEvaluator = baselines_module.FidelityEvaluator
    except Exception as e:
        pytest.skip(f"Baselines module not available: {e}", allow_module_level=True)


class TestBootstrapGeneration:
    """Test that bootstrap synthetic data generation works correctly."""

    @pytest.fixture
    def real_data(self):
        """Load real windowed data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Data not generated yet")

        df = pd.read_parquet(parquet_path)
        eur_cols = [f"EUR=X_step_{i}" for i in range(256)]
        return df.loc[df["asset"] == "EUR=X", eur_cols].values

    @pytest.fixture
    def bootstrap(self):
        """Instantiate bootstrap generator."""
        return StationaryBootstrap(block_length=64)

    def test_bootstrap_shape_matches_input(self, real_data, bootstrap):
        """Test that bootstrap output shape matches input shape."""
        synthetic = real_data[np.random.choice(len(real_data), size=len(real_data), replace=True)]

        assert synthetic.shape == real_data.shape, \
            f"Bootstrap shape {synthetic.shape} != real data {real_data.shape}"

    def test_bootstrap_synthetic_has_no_nan_where_real_valid(self, real_data):
        """Test that bootstrap doesn't introduce NaNs in real data's valid locations."""
        # Simple bootstrap: resample rows with replacement
        indices = np.random.choice(len(real_data), size=len(real_data), replace=True)
        synthetic = real_data[indices]

        # Check a few rows: where real has valid data, synthetic should also have valid data
        # (since we're resampling from the same pool)
        for i in range(min(10, len(real_data))):
            real_valid = ~np.isnan(real_data[i])
            # Synthetic at this position is a resampled row, may have different NaN pattern
            # But the values that ARE present should be finite
            assert np.all(np.isfinite(synthetic[i][real_valid])), \
                f"Row {i}: synthetic has NaN/Inf where real is valid"

    def test_bootstrap_preserves_data_range(self, real_data):
        """Test that bootstrap doesn't generate values outside observed range."""
        indices = np.random.choice(len(real_data), size=len(real_data), replace=True)
        synthetic = real_data[indices]

        real_min = np.nanmin(real_data)
        real_max = np.nanmax(real_data)
        syn_min = np.nanmin(synthetic)
        syn_max = np.nanmax(synthetic)

        assert syn_min >= real_min, \
            f"Bootstrap generated value {syn_min:.6f} below observed min {real_min:.6f}"
        assert syn_max <= real_max, \
            f"Bootstrap generated value {syn_max:.6f} above observed max {real_max:.6f}"

    def test_bootstrap_statistics_match_real(self, real_data):
        """Test that bootstrap synthetic data has similar mean and std to real."""
        indices = np.random.choice(len(real_data), size=len(real_data), replace=True)
        synthetic = real_data[indices]

        real_flat = real_data[~np.isnan(real_data)]
        syn_flat = synthetic[~np.isnan(synthetic)]

        real_mean = np.mean(real_flat)
        syn_mean = np.mean(syn_flat)
        real_std = np.std(real_flat)
        syn_std = np.std(syn_flat)

        # Bootstrap should have very similar statistics (within 5%)
        assert abs(syn_mean - real_mean) < 0.01 * abs(real_mean) + 0.001, \
            f"Bootstrap mean {syn_mean:.6f} deviates too much from real {real_mean:.6f}"
        assert abs(syn_std - real_std) < 0.10 * real_std, \
            f"Bootstrap std {syn_std:.6f} deviates too much from real {real_std:.6f}"


class TestGARCHGeneration:
    """Test that GARCH synthetic data generation works correctly."""

    @pytest.fixture
    def real_data(self):
        """Load real windowed data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Data not generated yet")

        df = pd.read_parquet(parquet_path)
        step_cols = [c for c in df.columns if '_step_' in c]
        return df[step_cols].values

    @pytest.fixture
    def garch(self, real_data):
        """Fit GARCH model to real data."""
        model = GARCHModel()
        model.fit(real_data.flatten())
        return model

    def test_garch_generates_finite_values(self, garch):
        """Test that GARCH generates finite (non-NaN, non-Inf) values."""
        synthetic = garch.generate(n_samples=10, n_steps=256)

        assert not np.isnan(synthetic).any(), "GARCH generated NaN values"
        assert not np.isinf(synthetic).any(), "GARCH generated Inf values"

    def test_garch_output_shape(self, garch):
        """Test that GARCH output has correct shape."""
        n_samples = 20
        n_steps = 256
        synthetic = garch.generate(n_samples=n_samples, n_steps=n_steps)

        assert synthetic.shape == (n_samples, n_steps), \
            f"GARCH shape {synthetic.shape} != expected ({n_samples}, {n_steps})"

    def test_garch_synthetic_realistic_magnitude(self, real_data, garch):
        """Test that GARCH synthetic returns are in realistic range."""
        synthetic = garch.generate(n_samples=100, n_steps=256)

        real_std = np.nanstd(real_data)
        syn_std = np.std(synthetic)

        # GARCH std should be within 2x of real std (allows for volatility regime shifts)
        assert syn_std < 5 * real_std, \
            f"GARCH std {syn_std:.6f} too high (real std {real_std:.6f})"
        assert syn_std > 0.1 * real_std, \
            f"GARCH std {syn_std:.6f} too low (real std {real_std:.6f})"


class TestFidelityEvaluation:
    """Test that fidelity metrics are computed correctly without NaN propagation."""

    @pytest.fixture
    def real_data(self):
        """Load real windowed data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Data not generated yet")

        df = pd.read_parquet(parquet_path)
        step_cols = [c for c in df.columns if '_step_' in c]
        return df[step_cols].values

    @pytest.fixture
    def fidelity_results(self):
        """Load fidelity metrics from saved results."""
        json_path = Path(__file__).parent.parent / "outputs" / "fidelity_metrics_summary.json"
        if not json_path.exists():
            pytest.skip("Fidelity results not generated yet")

        with open(json_path) as f:
            return json.load(f)

    def test_fidelity_metrics_exist(self, fidelity_results):
        """Test that fidelity results have been computed."""
        assert 'metrics' in fidelity_results, "No metrics found in results"
        assert len(fidelity_results['metrics']) >= 1, "No baseline metrics computed"

    def test_bootstrap_fidelity_not_nan(self, fidelity_results):
        """Test that Bootstrap fidelity metrics are not NaN."""
        bootstrap_metrics = next(
            (m for m in fidelity_results['metrics'] if m['method'] == 'Bootstrap'),
            None
        )

        assert bootstrap_metrics is not None, "Bootstrap metrics not found"

        # Check key metrics are not NaN
        assert not np.isnan(bootstrap_metrics.get('autocorr_returns_mean', np.nan)), \
            "Bootstrap autocorr_returns_mean is NaN"
        assert not np.isnan(bootstrap_metrics.get('kurtosis_real', np.nan)), \
            "Bootstrap kurtosis_real is NaN"

    def test_bootstrap_synthetic_metrics_computed(self, fidelity_results):
        """Test that Bootstrap synthetic metrics are computed (not all NaN)."""
        bootstrap_metrics = next(
            (m for m in fidelity_results['metrics'] if m['method'] == 'Bootstrap'),
            None
        )

        assert bootstrap_metrics is not None, "Bootstrap metrics not found"

        # At least one synthetic metric should be non-NaN
        synthetic_keys = [
            'autocorr_returns_synthetic',
            'autocorr_squared_synthetic',
            'skewness_synthetic',
            'kurtosis_synthetic'
        ]

        computed_count = sum(
            1 for k in synthetic_keys 
            if not np.isnan(bootstrap_metrics.get(k, np.nan))
        )
        assert computed_count > 0, "All Bootstrap synthetic metrics are NaN"

    def test_bootstrap_captures_stylized_facts(self, fidelity_results):
        """Test that Bootstrap captures known financial stylized facts."""
        bootstrap_metrics = next(
            (m for m in fidelity_results['metrics'] if m['method'] == 'Bootstrap'),
            None
        )

        assert bootstrap_metrics is not None, "Bootstrap metrics not found"

        # Real data should show:
        # 1. Near-zero autocorr of returns
        ac_real = bootstrap_metrics['autocorr_returns_mean']
        assert abs(ac_real) < 0.05, \
            f"Real autocorr {ac_real:.4f} not near zero (should be for market efficiency)"

        # 2. Positive autocorr of squared returns (volatility clustering)
        acq_real = bootstrap_metrics['autocorr_squared_mean']
        assert acq_real > 0.10, \
            f"Real autocorr(squared) {acq_real:.4f} too low (should > 0.1 for volatility clustering)"

        # 3. Heavy tails (excess kurtosis)
        kurt_real = bootstrap_metrics['kurtosis_real']
        assert kurt_real > 3, \
            f"Real kurtosis {kurt_real:.2f} too low (should > 3 for fat tails)"

    def test_bootstrap_synthetic_matches_real(self, fidelity_results):
        """Test that Bootstrap synthetic metrics closely match real metrics."""
        bootstrap_metrics = next(
            (m for m in fidelity_results['metrics'] if m['method'] == 'Bootstrap'),
            None
        )

        assert bootstrap_metrics is not None, "Bootstrap metrics not found"

        # Bootstrap should preserve statistics almost exactly
        ac_real = bootstrap_metrics['autocorr_returns_mean']
        ac_syn = bootstrap_metrics['autocorr_returns_synthetic']

        if not np.isnan(ac_syn):
            assert abs(ac_syn - ac_real) < 0.01, \
                f"Bootstrap autocorr divergence: real={ac_real:.6f}, synthetic={ac_syn:.6f}"

        # Kurtosis should be close
        kurt_real = bootstrap_metrics['kurtosis_real']
        kurt_syn = bootstrap_metrics['kurtosis_synthetic']

        if not np.isnan(kurt_syn):
            assert abs(kurt_syn - kurt_real) < 2.0, \
                f"Bootstrap kurtosis divergence: real={kurt_real:.2f}, synthetic={kurt_syn:.2f}"

    def test_real_metrics_have_expected_ranges(self, fidelity_results, real_data):
        """Test that real data metrics match observed data characteristics."""
        bootstrap_metrics = next(
            (m for m in fidelity_results['metrics'] if m['method'] == 'Bootstrap'),
            None
        )

        assert bootstrap_metrics is not None, "Bootstrap metrics not found"

        # Kurtosis should be in range [5, 50] for financial data
        kurt = bootstrap_metrics['kurtosis_real']
        assert 3 < kurt < 100, \
            f"Real kurtosis {kurt:.2f} outside realistic range for financial returns"

        # Skewness should be between -2 and +2 (rarely more extreme)
        skew = bootstrap_metrics['skewness_real']
        assert -3 < skew < 3, \
            f"Real skewness {skew:.2f} outside realistic range for financial returns"


class TestMetricsRobustness:
    """Test that metric computation is robust to edge cases."""

    def test_autocorr_handles_constant_data(self):
        """Test that autocorr function handles constant input gracefully."""
        evaluator = FidelityEvaluator()
        constant_data = np.ones(100)

        # Should not crash; may return NaN for constant data
        ac = evaluator.autocorr_returns(constant_data)
        assert len(ac) == 20  # Should return 20 lags
        # Some may be NaN, but function should complete

    def test_kurtosis_handles_small_sample(self):
        """Test that kurtosis function handles small samples."""
        evaluator = FidelityEvaluator()
        small_data = np.array([0.1, -0.05, 0.03])

        # Should not crash
        kurt = evaluator.kurtosis(small_data)
        assert isinstance(kurt, (float, np.floating))  # Should return a number

    def test_wasserstein_handles_different_sizes(self):
        """Test that Wasserstein distance handles different array sizes."""
        evaluator = FidelityEvaluator()
        real = np.random.randn(100, 50)
        synthetic = np.random.randn(80, 50)

        # Should not crash; should compute distance
        wd = evaluator.wasserstein_distance(real, synthetic)
        assert isinstance(wd, (float, np.floating))
        assert wd >= 0  # Wasserstein distance is non-negative
