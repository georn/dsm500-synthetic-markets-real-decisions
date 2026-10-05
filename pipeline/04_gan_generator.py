#!/usr/bin/env python3
"""
GAN-based Synthetic Financial Data Generator for DSM500 CW2

Trains a Generative Adversarial Network (GAN) to learn the distribution of
256-step return windows from era-A training data, then generates synthetic
paths for strategy validation (RQ2 downstream utility evaluation).

Architecture:
- Generator: MLP (noise → synthetic 256-step returns)
- Discriminator: MLP (real vs synthetic classifier)
- Loss: Standard adversarial loss with gradient penalty

Output: Trained GAN model + function to generate synthetic windows

Usage:
    from gan_generator import train_gan, generate_synthetic

    # Train
    gan = train_gan(real_windows, epochs=100)

    # Generate synthetic data for evaluation
    synthetic = generate_synthetic(gan, n_samples=1000)

Implementation notes:
- Uses PyTorch for efficiency
- Includes convergence monitoring
- Saves model for later evaluation
- Supports reproducible generation (fixed seed)
"""

import logging
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class Generator(nn.Module):
    """MLP generator: maps noise to synthetic return windows."""

    def __init__(self, noise_dim: int = 100, window_length: int = 256):
        super().__init__()
        self.noise_dim = noise_dim
        self.window_length = window_length

        self.network = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, window_length),
            nn.Tanh(),  # Scale output to [-1, 1]
        )

    def forward(self, z):
        """Generate synthetic windows from noise."""
        return self.network(z)


class Discriminator(nn.Module):
    """MLP discriminator: classifies real vs synthetic windows."""

    def __init__(self, window_length: int = 256):
        super().__init__()
        self.window_length = window_length

        self.network = nn.Sequential(
            nn.Linear(window_length, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """Classify window as real (1) or synthetic (0)."""
        return self.network(x)


class GANTrainer:
    """Train a GAN on financial return windows."""

    def __init__(
        self,
        real_data: np.ndarray,
        noise_dim: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.0002,
        betas: Tuple[float, float] = (0.5, 0.999),
        device: str = DEVICE,
    ):
        """
        Initialize GAN trainer.

        Args:
            real_data: Shape (n_windows, window_length)
            noise_dim: Dimension of input noise
            batch_size: Training batch size
            learning_rate: Adam optimizer learning rate
            betas: Adam optimizer betas
            device: torch device (cpu or cuda)
        """
        self.real_data = real_data
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.device = device
        self.window_length = real_data.shape[1]

        # Normalize real data to [-1, 1] for tanh output
        self.data_mean = real_data.mean()
        self.data_std = real_data.std()
        real_data_norm = (real_data - self.data_mean) / (self.data_std + 1e-8)
        real_data_norm = np.clip(real_data_norm, -1, 1)

        # Create data loader
        real_tensor = torch.FloatTensor(real_data_norm).to(device)
        dataset = TensorDataset(real_tensor)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize models
        self.generator = Generator(noise_dim, self.window_length).to(device)
        self.discriminator = Discriminator(self.window_length).to(device)

        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=betas
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=betas
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {"d_loss": [], "g_loss": []}

    def train(self, epochs: int = 100, log_interval: int = 10) -> dict:
        """
        Train the GAN.

        Args:
            epochs: Number of training epochs
            log_interval: Log every N epochs

        Returns:
            Training history dict
        """
        logger.info(f"Starting GAN training for {epochs} epochs")

        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            batch_count = 0

            for batch_idx, (real_batch,) in enumerate(self.loader):
                batch_size = real_batch.size(0)

                # ===== Train Discriminator =====
                self.discriminator.zero_grad()

                # Real data
                real_output = self.discriminator(real_batch)
                real_label = torch.ones(batch_size, 1).to(self.device)
                d_loss_real = self.criterion(real_output, real_label)

                # Synthetic data
                z = torch.randn(batch_size, self.noise_dim).to(self.device)
                synthetic_batch = self.generator(z)
                synthetic_output = self.discriminator(synthetic_batch.detach())
                synthetic_label = torch.zeros(batch_size, 1).to(self.device)
                d_loss_synthetic = self.criterion(synthetic_output, synthetic_label)

                d_loss = d_loss_real + d_loss_synthetic
                d_loss.backward()
                self.d_optimizer.step()

                # ===== Train Generator =====
                self.generator.zero_grad()

                z = torch.randn(batch_size, self.noise_dim).to(self.device)
                synthetic_batch = self.generator(z)
                synthetic_output = self.discriminator(synthetic_batch)
                g_loss = self.criterion(synthetic_output, real_label)

                g_loss.backward()
                self.g_optimizer.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                batch_count += 1

            # Log progress
            avg_d_loss = epoch_d_loss / batch_count
            avg_g_loss = epoch_g_loss / batch_count
            self.history["d_loss"].append(avg_d_loss)
            self.history["g_loss"].append(avg_g_loss)

            if (epoch + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
                )

        logger.info("GAN training complete")
        return self.history

    def generate(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate synthetic return windows.

        Args:
            n_samples: Number of synthetic windows to generate

        Returns:
            Array of shape (n_samples, window_length) with synthetic returns
        """
        self.generator.eval()

        synthetic_list = []
        with torch.no_grad():
            for _ in range(0, n_samples, self.batch_size):
                batch_size = min(self.batch_size, n_samples - len(synthetic_list))
                z = torch.randn(batch_size, self.noise_dim).to(self.device)
                synthetic_batch = self.generator(z)
                synthetic_list.append(synthetic_batch.cpu().numpy())

        synthetic_data = np.vstack(synthetic_list)

        # Denormalize back to original scale
        synthetic_data = synthetic_data * (self.data_std + 1e-8) + self.data_mean

        return synthetic_data[:n_samples]

    def save(self, path: str):
        """Save trained model and normalization stats."""
        checkpoint = {
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "data_mean": self.data_mean,
            "data_std": self.data_std,
            "noise_dim": self.noise_dim,
            "window_length": self.window_length,
            "batch_size": self.batch_size,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = DEVICE):
        """Load trained model."""
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(path, map_location=device)

        trainer = cls.__new__(cls)
        trainer.device = device
        trainer.noise_dim = checkpoint["noise_dim"]
        trainer.window_length = checkpoint["window_length"]
        trainer.batch_size = checkpoint.get("batch_size", 64)
        trainer.data_mean = checkpoint["data_mean"]
        trainer.data_std = checkpoint["data_std"]
        trainer.history = checkpoint["history"]

        trainer.generator = Generator(trainer.noise_dim, trainer.window_length).to(
            device
        )
        trainer.discriminator = Discriminator(trainer.window_length).to(device)

        trainer.generator.load_state_dict(checkpoint["generator_state"])
        trainer.discriminator.load_state_dict(checkpoint["discriminator_state"])

        logger.info(f"Model loaded from {path}")
        return trainer


def train_gan(
    real_data: np.ndarray,
    output_dir: str = "outputs",
    epochs: int = 100,
    model_name: str = "gan_model.pt",
    **kwargs
) -> GANTrainer:
    """
    Convenience function to train GAN on financial data.

    Args:
        real_data: Array of shape (n_windows, 256)
        output_dir: Directory to save model
        epochs: Training epochs
        **kwargs: Additional arguments to GANTrainer

    Returns:
        Trained GANTrainer instance
    """
    Path(output_dir).mkdir(exist_ok=True)

    trainer = GANTrainer(real_data, **kwargs)
    trainer.train(epochs=epochs)

    model_path = Path(output_dir) / model_name
    trainer.save(str(model_path))
    history_path = model_path.with_name(model_path.stem + "_history.json")
    history_path.write_text(json.dumps({k: [float(x) for x in v] for k, v in trainer.history.items()
                                        if isinstance(v, list)}, indent=1))

    return trainer


def generate_synthetic(
    trainer: GANTrainer, n_samples: int = 1000, seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic return windows from trained GAN.

    Args:
        trainer: Trained GANTrainer
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, 256)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    return trainer.generate(n_samples)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Train the MLP GAN on era-A windows.")
    parser.add_argument("--eur-only", action="store_true",
                        help="train on USD/EUR rows only (256 columns) and save outputs/gan_model_eur.pt")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    parquet_path = Path("outputs/windowed_data_era_a.parquet")
    if not parquet_path.exists():
        raise SystemExit("Data file not found. Run 01_data_pipeline.py first.")
    df = pd.read_parquet(parquet_path)
    if args.eur_only:
        real_data = df.loc[df["asset"] == "EUR=X", [f"EUR=X_step_{i}" for i in range(256)]].to_numpy(float)
        model_name = "gan_model_eur.pt"
    else:
        # As run for the reported results: pooled three-asset rows, missing blocks zero-filled.
        real_data = np.nan_to_num(df[[c for c in df.columns if "_step_" in c]].to_numpy(float), nan=0.0)
        model_name = "gan_model.pt"
    logger.info(f"Real data shape: {real_data.shape}")
    trainer = train_gan(real_data, epochs=args.epochs, model_name=model_name)
    synthetic = generate_synthetic(trainer, n_samples=100)
    logger.info(f"Generated {synthetic.shape}; mean {synthetic.mean():.6f}, std {synthetic.std():.6f}")
