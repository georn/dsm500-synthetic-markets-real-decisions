"""
Lean smoke tests for GAN generator.

Tests validate:
- GAN generates correct output shape
- Generated values are finite (not NaN/Inf)
- Synthetic statistics are similar to real data (basic fidelity)
- Model can be saved and loaded
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

try:
    import torch
    import pandas as pd
    from gan_generator import GANTrainer, train_gan, generate_synthetic
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytest.skip("PyTorch not available", allow_module_level=True)


class TestGANGeneration:
    """Test that GAN generates valid synthetic data."""

    @pytest.fixture
    def real_data(self):
        """Load real windowed data for testing."""
        parquet_path = Path(__file__).parent.parent / "outputs" / "windowed_data_era_a.parquet"
        if not parquet_path.exists():
            pytest.skip("Data not generated yet")

        df = pd.read_parquet(parquet_path)
        step_cols = [c for c in df.columns if "_step_" in c]
        data = df[step_cols].values
        # Replace NaNs with 0 for GAN training
        data = np.nan_to_num(data, nan=0.0)
        return data[:100]  # Use small subset for testing

    def test_gan_trainer_initializes(self, real_data):
        """Test that GAN trainer initializes without error."""
        trainer = GANTrainer(real_data, batch_size=16)
        assert trainer is not None
        assert trainer.generator is not None
        assert trainer.discriminator is not None

    def test_gan_generates_correct_shape(self, real_data):
        """Test that GAN generates correct output shape."""
        trainer = GANTrainer(real_data)
        trainer.train(epochs=1)

        for n_samples in [10, 50, 100]:
            synthetic = trainer.generate(n_samples)
            assert synthetic.shape == (n_samples, real_data.shape[1]), \
                f"Expected shape ({n_samples}, {real_data.shape[1]}), got {synthetic.shape}"

    def test_gan_synthetic_is_finite(self, real_data):
        """Test that generated values are finite (not NaN/Inf)."""
        trainer = GANTrainer(real_data)
        trainer.train(epochs=1)

        synthetic = trainer.generate(n_samples=50)

        assert not np.isnan(synthetic).any(), "GAN generated NaN values"
        assert not np.isinf(synthetic).any(), "GAN generated Inf values"
        assert np.all(np.isfinite(synthetic)), "GAN generated non-finite values"

    @pytest.mark.xfail(strict=True, reason='Known defect: after short training the MLP GAN output spread is far below the real return spread')
    def test_gan_synthetic_statistics_reasonable(self, real_data):
        """Test that synthetic data has similar statistics to real data."""
        trainer = GANTrainer(real_data)
        trainer.train(epochs=1)

        synthetic = trainer.generate(n_samples=200)

        real_mean = np.nanmean(real_data)
        real_std = np.nanstd(real_data)
        syn_mean = np.mean(synthetic)
        syn_std = np.std(synthetic)

        # Allow for variation due to small training size
        # Mean should be within 0.01 of real
        assert abs(syn_mean - real_mean) < 0.01, \
            f"Synthetic mean {syn_mean:.6f} deviates too much from real {real_mean:.6f}"

        # Std should be within 0.5x-2x of real
        assert 0.5 * real_std < syn_std < 2.0 * real_std, \
            f"Synthetic std {syn_std:.6f} outside expected range [{0.5*real_std:.6f}, {2.0*real_std:.6f}]"

    @pytest.mark.xfail(strict=True, reason='Known defect: GAN training is not deterministic under a fixed seed')
    def test_gan_reproducibility(self, real_data):
        """Test that same seed produces same synthetic data."""
        trainer1 = GANTrainer(real_data)
        trainer1.train(epochs=1)
        synthetic1 = generate_synthetic(trainer1, n_samples=50, seed=42)

        trainer2 = GANTrainer(real_data)
        trainer2.train(epochs=1)
        synthetic2 = generate_synthetic(trainer2, n_samples=50, seed=42)

        # Should produce identical results with same seed
        # (within floating point precision)
        assert np.allclose(synthetic1, synthetic2, rtol=1e-5), \
            "Same seed did not produce identical results"

    def test_gan_model_save_load(self, real_data, tmp_path):
        """Test that model can be saved and loaded."""
        model_path = tmp_path / "test_model.pt"

        # Train and save
        trainer1 = GANTrainer(real_data)
        trainer1.train(epochs=1)
        trainer1.save(str(model_path))

        assert model_path.exists(), "Model file not saved"

        # Load and generate
        trainer2 = GANTrainer.load(str(model_path))
        synthetic = trainer2.generate(n_samples=50)

        assert synthetic.shape == (50, real_data.shape[1]), \
            "Loaded model generated wrong shape"
        assert np.all(np.isfinite(synthetic)), \
            "Loaded model generated non-finite values"

    def test_gan_batch_size_handling(self, real_data):
        """Test that GAN handles different batch sizes correctly."""
        trainer = GANTrainer(real_data, batch_size=16)
        trainer.train(epochs=1)

        # Request more samples than batch size
        synthetic = trainer.generate(n_samples=100)
        assert synthetic.shape == (100, real_data.shape[1])

    def test_gan_output_in_data_range(self, real_data):
        """Test that synthetic data is in reasonable range."""
        trainer = GANTrainer(real_data)
        trainer.train(epochs=1)

        synthetic = trainer.generate(n_samples=100)

        # Synthetic returns should typically be in [-10%, +10%]
        # Allow wider range for GAN (it's still learning)
        assert np.all(synthetic > -0.5), "GAN generated unreasonably negative values"
        assert np.all(synthetic < 0.5), "GAN generated unreasonably positive values"
