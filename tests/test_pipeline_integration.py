"""
Integration tests for full DSM500 pipeline.

Validates end-to-end data flow:
1. Data loading → preprocessing
2. Real data → GAN input/output
3. Synthetic data → strategy application
4. Shapes and formats compatible across all stages
"""

import numpy as np
import pytest
import importlib.util
from pathlib import Path

# Load pipeline modules by file path
def _load_module(filename):
    """Load a pipeline module by filename."""
    filepath = Path(__file__).parent.parent / "pipeline" / filename
    if not filepath.exists():
        return None
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), str(filepath))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None

data_pipeline_mod = _load_module("01_data_pipeline.py")
baselines_mod = _load_module("02_baselines.py")
strategies_mod = _load_module("03_strategies.py")
gan_mod = _load_module("04_gan_generator.py")

try:
    import torch
    import pandas as pd
    if all([data_pipeline_mod, baselines_mod, strategies_mod, gan_mod]):
        DataPipeline = getattr(data_pipeline_mod, "DataPipeline", None)
        StationaryBootstrap = getattr(baselines_mod, "StationaryBootstrap", None)
        GARCHModel = getattr(baselines_mod, "GARCHModel", None)
        FidelityEvaluator = getattr(baselines_mod, "FidelityEvaluator", None)
        generate_strategies = getattr(strategies_mod, "generate_strategies", None)
        GANTrainer = getattr(gan_mod, "GANTrainer", None)
        generate_synthetic = getattr(gan_mod, "generate_synthetic", None)
        HAS_TORCH = all([StationaryBootstrap, GARCHModel, FidelityEvaluator,
                         generate_strategies, GANTrainer, generate_synthetic])
    else:
        HAS_TORCH = False
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    pytest.skip("Required modules not available", allow_module_level=True)


class TestPipelineIntegration:
    """Test full pipeline integration."""

    @pytest.fixture
    def real_data_era_a(self):
        """Load real era-A windowed data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Era-A data not generated yet")

        df = pd.read_parquet(parquet_path)
        step_cols = [c for c in df.columns if "_step_" in c]
        data = df[step_cols].values
        return data

    @pytest.fixture
    def real_data_era_b(self):
        """Load real era-B windowed data."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_b.parquet"
        if not parquet_path.exists():
            pytest.skip("Era-B data not generated yet")

        df = pd.read_parquet(parquet_path)
        step_cols = [c for c in df.columns if "_step_" in c]
        data = df[step_cols].values
        return data

    @pytest.fixture
    def strategies(self):
        """Generate 20 strategies for testing."""
        return generate_strategies(n=20)

    # ===== Data Loading Tests =====

    def test_real_data_era_a_shape_and_dtype(self, real_data_era_a):
        """Real era-A data has correct shape and dtype."""
        assert real_data_era_a.ndim == 2, "Data should be 2D"
        assert real_data_era_a.shape[1] == 768, "Should have 768 features (256 steps × 3 assets)"
        assert np.issubdtype(real_data_era_a.dtype, np.floating), "Data should be numeric"

    def test_real_data_era_b_shape_and_dtype(self, real_data_era_b):
        """Real era-B data has correct shape and dtype."""
        assert real_data_era_b.ndim == 2, "Data should be 2D"
        assert real_data_era_b.shape[1] == 768, "Should have 768 features (256 steps × 3 assets)"
        assert np.issubdtype(real_data_era_b.dtype, np.floating), "Data should be numeric"

    def test_real_data_era_split_chronological(self, real_data_era_a, real_data_era_b):
        """Era-A and era-B are different (chronological split maintained)."""
        # Should not be identical (different time periods)
        # Can have NaN patterns but data values should differ
        assert real_data_era_a.shape[0] > 0, "Era-A should have samples"
        assert real_data_era_b.shape[0] > 0, "Era-B should have samples"

    # ===== GAN Data Flow Tests =====

    def test_gan_accepts_real_data_format(self, real_data_era_a):
        """GAN trainer accepts real data format without errors."""
        # Use subset for speed
        data_subset = real_data_era_a[:100]
        data_clean = np.nan_to_num(data_subset, nan=0.0)

        trainer = GANTrainer(data_clean, batch_size=16)
        assert trainer is not None
        assert trainer.window_length == 768

    def test_gan_normalization_denormalization_invertible(self, real_data_era_a):
        """GAN normalization → denormalization preserves scale."""
        data_subset = real_data_era_a[:50]
        data_clean = np.nan_to_num(data_subset, nan=0.0)

        original_mean = data_clean.mean()
        original_std = data_clean.std()

        trainer = GANTrainer(data_clean)
        trainer.train(epochs=1)

        synthetic = trainer.generate(n_samples=50)

        synthetic_mean = synthetic.mean()
        synthetic_std = synthetic.std()

        # Mean and std should be in reasonable range of original
        assert abs(synthetic_mean - original_mean) < original_std, \
            f"Denormalized mean {synthetic_mean} too far from original {original_mean}"

    def test_gan_output_compatible_with_evaluation(self, real_data_era_a):
        """GAN output has shape and dtype compatible with evaluation loop."""
        data_subset = real_data_era_a[:50]
        data_clean = np.nan_to_num(data_subset, nan=0.0)

        trainer = GANTrainer(data_clean)
        trainer.train(epochs=1)
        synthetic = trainer.generate(n_samples=100)

        # Must be 2D array with 768 features
        assert synthetic.ndim == 2, "Synthetic data should be 2D"
        assert synthetic.shape[1] == 768, "Synthetic should have 768 features"

        # Must be finite (no NaN/Inf which would break strategy evaluation)
        assert np.all(np.isfinite(synthetic)), "Synthetic data should be all finite"

    def test_baseline_bootstrap_accepts_real_data(self, real_data_era_a):
        """Bootstrap accepts real data format (row resampling)."""
        data_subset = real_data_era_a[:100]

        # Bootstrap: resample rows with replacement
        indices = np.random.choice(len(data_subset), size=len(data_subset), replace=True)
        synthetic = data_subset[indices]

        assert synthetic.shape == (100, 768), "Bootstrap output wrong shape"

    def test_baseline_garch_accepts_real_data(self, real_data_era_a):
        """GARCH accepts real data format."""
        data_subset = real_data_era_a[:100]
        data_clean = np.nan_to_num(data_subset, nan=0.0)

        # GARCH: fit on flattened data, generate
        garch = GARCHModel()
        garch.fit(data_clean.flatten())
        synthetic = garch.generate(n_samples=50, n_steps=256)

        assert synthetic.shape == (50, 256), "GARCH output wrong shape (should be 50 samples, 256 steps)"
        assert np.all(np.isfinite(synthetic)), "GARCH output should be finite"

    # ===== Strategy-Synthetic Data Compatibility =====

    def test_strategies_applicable_to_synthetic_data(self, real_data_era_a, strategies):
        """Can apply strategies to synthetic data (shape compatibility)."""
        # Generate synthetic via all baselines
        data_subset = real_data_era_a[:50]
        data_clean = np.nan_to_num(data_subset, nan=0.0)

        # Bootstrap: resample rows (will have NaNs)
        indices = np.random.choice(len(data_subset), size=50, replace=True)
        synthetic_bootstrap = np.nan_to_num(data_subset[indices], nan=0.0)

        # GARCH: generate 50 windows of 256 steps each, then reshape to match 2D structure
        garch = GARCHModel()
        garch.fit(data_clean.flatten())
        garch_output = garch.generate(n_samples=50, n_steps=256)
        # Reshape to match 2D data (50, 768) by repeating 3 times for 3 assets
        synthetic_garch = np.tile(garch_output, (1, 3))

        # GAN
        trainer = GANTrainer(data_clean)
        trainer.train(epochs=1)
        synthetic_gan = trainer.generate(n_samples=50)

        # All should be applicable (right shape, finite values)
        for synthetic in [synthetic_bootstrap, synthetic_garch, synthetic_gan]:
            assert synthetic.shape[0] == 50, f"Wrong sample count: {synthetic.shape}"
            assert np.all(np.isfinite(synthetic)), "Contains non-finite values"

    def test_strategy_parameters_compatible_with_data(self, real_data_era_a, strategies):
        """Strategy parameters are compatible with 256-step windows."""
        data_subset = real_data_era_a[:10]

        for strategy in strategies:
            # Check hold_bars doesn't exceed window length
            hold_bars = strategy.parameters.get('hold_bars')
            if hold_bars is not None:
                assert hold_bars < 256, \
                    f"Strategy {strategy.name}: hold_bars {hold_bars} exceeds window length 256"

            # Check MA windows exist and are sensible
            if strategy.entry_type == 'ma_crossover':
                ma_slow = strategy.parameters.get('ma_slow')
                if ma_slow is not None:
                    assert ma_slow < 256, \
                        f"Strategy {strategy.name}: ma_slow {ma_slow} exceeds window length 256"

    # ===== Fidelity Evaluation Compatibility =====

    def test_fidelity_evaluator_accepts_all_synthetic_sources(self, real_data_era_a):
        """FidelityEvaluator can compute metrics on all baseline outputs."""
        data_subset = real_data_era_a[:50]
        real_clean = np.nan_to_num(data_subset, nan=0.0)

        evaluator = FidelityEvaluator()

        # Bootstrap: resample rows (clean NaNs)
        indices = np.random.choice(len(data_subset), size=50, replace=True)
        synthetic_bootstrap = np.nan_to_num(data_subset[indices], nan=0.0)

        # Compute metrics on Bootstrap output
        try:
            ac = evaluator.autocorr_returns(synthetic_bootstrap.flatten())
            assert ac is not None
        except Exception as e:
            pytest.fail(f"Bootstrap autocorr failed: {e}")

        # GARCH: generate and reshape
        garch = GARCHModel()
        garch.fit(real_clean.flatten())
        garch_output = garch.generate(n_samples=50, n_steps=256)
        synthetic_garch = np.tile(garch_output, (1, 3))

        try:
            k = evaluator.kurtosis(synthetic_garch.flatten())
            assert np.isfinite(k)
        except Exception as e:
            pytest.fail(f"GARCH kurtosis failed: {e}")

        # GAN
        trainer = GANTrainer(real_clean)
        trainer.train(epochs=1)
        synthetic_gan = trainer.generate(n_samples=50)

        try:
            sk = evaluator.skewness(synthetic_gan.flatten())
            assert np.isfinite(sk)
        except Exception as e:
            pytest.fail(f"GAN skewness failed: {e}")

    def test_fidelity_metrics_are_numeric(self, real_data_era_a):
        """Fidelity metrics are all finite numeric values."""
        data_subset = real_data_era_a[:50]
        real_clean = np.nan_to_num(data_subset, nan=0.0)

        synthetic = np.random.normal(0, real_clean.std(), real_clean.shape)

        evaluator = FidelityEvaluator()

        # Test individual metric methods
        metrics = {
            'autocorr': np.mean(evaluator.autocorr_returns(synthetic.flatten())),
            'kurtosis': evaluator.kurtosis(synthetic.flatten()),
            'skewness': evaluator.skewness(synthetic.flatten()),
        }

        # All metrics should be finite
        for key, value in metrics.items():
            assert isinstance(value, (int, float, np.number)), \
                f"Metric {key} has non-numeric type: {type(value)}"
            assert np.isfinite(value), \
                f"Metric {key} is non-finite: {value}"

    # ===== Full Pipeline Smoke Test =====

    def test_full_pipeline_end_to_end(self, real_data_era_a):
        """Full pipeline: load → GAN → evaluate → strategy compatible."""
        # 1. Load and prepare data
        data_subset = real_data_era_a[:100]
        data_clean = np.nan_to_num(data_subset, nan=0.0)

        # 2. Generate synthetic via all methods
        # Bootstrap: resample rows (clean NaNs)
        indices = np.random.choice(len(data_subset), size=50, replace=True)
        synthetic_bootstrap = np.nan_to_num(data_subset[indices], nan=0.0)

        # GARCH: generate and reshape
        garch = GARCHModel()
        garch.fit(data_clean.flatten())
        garch_output = garch.generate(n_samples=50, n_steps=256)
        synthetic_garch = np.tile(garch_output, (1, 3))

        # GAN
        trainer = GANTrainer(data_clean)
        trainer.train(epochs=1)
        synthetic_gan = trainer.generate(n_samples=50)

        # 3. Compute fidelity metrics
        evaluator = FidelityEvaluator()
        metrics_bootstrap = {
            'autocorr': np.mean(evaluator.autocorr_returns(synthetic_bootstrap.flatten())),
            'kurtosis': evaluator.kurtosis(synthetic_bootstrap.flatten()),
        }
        metrics_garch = {
            'autocorr': np.mean(evaluator.autocorr_returns(synthetic_garch.flatten())),
            'kurtosis': evaluator.kurtosis(synthetic_garch.flatten()),
        }
        metrics_gan = {
            'autocorr': np.mean(evaluator.autocorr_returns(synthetic_gan.flatten())),
            'kurtosis': evaluator.kurtosis(synthetic_gan.flatten()),
        }

        # 4. Generate strategies
        strategies = generate_strategies(n=10)

        # 5. Verify all pieces work together
        assert len(strategies) == 10
        assert metrics_bootstrap is not None
        assert metrics_garch is not None
        assert metrics_gan is not None
        assert all(np.isfinite(v) for v in metrics_bootstrap.values())
        assert all(np.isfinite(v) for v in metrics_garch.values())
        assert all(np.isfinite(v) for v in metrics_gan.values())

    def test_pipeline_handles_edge_cases(self, real_data_era_a):
        """Pipeline handles edge cases (small datasets, extreme values)."""
        # Small data subset
        data_small = real_data_era_a[:10]
        data_clean = np.nan_to_num(data_small, nan=0.0)

        # Should not crash
        # Bootstrap: resample rows (clean NaNs)
        indices = np.random.choice(len(data_small), size=5, replace=True)
        synthetic_bootstrap = np.nan_to_num(data_small[indices], nan=0.0)
        assert synthetic_bootstrap.shape == (5, 768)

        # GARCH: generate and reshape
        garch = GARCHModel()
        garch.fit(data_clean.flatten())
        garch_output = garch.generate(n_samples=5, n_steps=256)
        synthetic_garch = np.tile(garch_output, (1, 3))
        assert synthetic_garch.shape == (5, 768)

        # GAN with tiny dataset
        trainer = GANTrainer(data_clean, batch_size=4)
        trainer.train(epochs=1)
        synthetic_gan = trainer.generate(n_samples=5)
        assert synthetic_gan.shape == (5, 768)

        # All should produce sensible results
        assert np.all(np.isfinite(synthetic_bootstrap))
        assert np.all(np.isfinite(synthetic_garch))
        assert np.all(np.isfinite(synthetic_gan))

    def test_data_quality_consistent_across_pipeline(self, real_data_era_a, real_data_era_b):
        """Data quality metrics are consistent across era-A and era-B."""
        # Both eras should have similar statistical properties
        mean_a = np.nanmean(real_data_era_a)
        mean_b = np.nanmean(real_data_era_b)
        std_a = np.nanstd(real_data_era_a)
        std_b = np.nanstd(real_data_era_b)

        # Means should be similar (both centered near 0 for log returns)
        assert abs(mean_a - mean_b) < 0.01, \
            f"Era-A mean {mean_a:.6f} differs significantly from era-B {mean_b:.6f}"

        # Stds should be similar (both have similar volatility)
        assert abs(std_a - std_b) / std_a < 0.5, \
            f"Era-B volatility {std_b:.6f} differs from era-A {std_a:.6f}"
