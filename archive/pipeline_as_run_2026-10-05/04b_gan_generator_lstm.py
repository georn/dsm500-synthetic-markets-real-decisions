#!/usr/bin/env python3
"""
LSTM-based Synthetic Financial Data Generator for DSM500 CW2

Parallel to 04_gan_generator.py (MLP version). Tests hypothesis that
temporal structure (LSTM) outperforms static representation (MLP) for
strategy-validation downstream utility (RQ2).

Architecture:
- Generator: Bidirectional LSTM → linear projection
- Discriminator: LSTM → binary classifier
- Loss: Standard adversarial loss

Comparison protocol: See drafts/draft_gan_comparison_protocol.md

Usage:
    from gan_generator_lstm import train_gan_lstm, generate_synthetic_lstm
    gan = train_gan_lstm(real_windows, epochs=50)
    synthetic = generate_synthetic_lstm(gan, n_samples=1000)

Output: gan_model_lstm.pt (checkpoint with both LSTM architecture and weights)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class LSTMGenerator(nn.Module):
    """LSTM generator: maps noise to synthetic return windows."""

    def __init__(self, noise_dim: int = 100, window_length: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.noise_dim = noise_dim
        self.window_length = window_length
        self.hidden_dim = hidden_dim

        # Projection from noise to LSTM input size
        self.noise_proj = nn.Linear(noise_dim, hidden_dim)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        """
        Generate synthetic windows from noise.

        Args:
            z: (batch_size, noise_dim)

        Returns:
            (batch_size, window_length) with values in [-1, 1]
        """
        batch_size = z.size(0)

        # Project noise to hidden dimension
        x = self.noise_proj(z)  # (batch_size, hidden_dim)

        # Repeat to create sequence (same input for each timestep, let LSTM create variation)
        x = x.unsqueeze(1).expand(-1, self.window_length, -1)  # (batch_size, window_length, hidden_dim)

        # Pass through LSTM layers
        x, _ = self.lstm1(x)  # (batch_size, window_length, hidden_dim*2)
        x, _ = self.lstm2(x)  # (batch_size, window_length, hidden_dim*2)

        # Project to single value per timestep
        x = self.output_proj(x)  # (batch_size, window_length, 1)
        x = x.squeeze(-1)  # (batch_size, window_length)

        # Scale to [-1, 1]
        x = self.tanh(x)

        return x


class LSTMDiscriminator(nn.Module):
    """LSTM discriminator: classifies real vs synthetic windows."""

    def __init__(self, window_length: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.window_length = window_length
        self.hidden_dim = hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)

        # Classification head
        self.dense1 = nn.Linear(hidden_dim * 2, 128)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Classify window as real (1) or synthetic (0).

        Args:
            x: (batch_size, window_length)

        Returns:
            (batch_size, 1) logits
        """
        # Reshape for LSTM: (batch_size, window_length, 1)
        x = x.unsqueeze(-1)

        # LSTM
        x, _ = self.lstm(x)  # (batch_size, window_length, hidden_dim*2)

        # Use last hidden state
        x = x[:, -1, :]  # (batch_size, hidden_dim*2)

        # Classification head
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x


class LSTMGANTrainer:
    """Train an LSTM-based GAN on financial return windows."""

    def __init__(
        self,
        real_data: np.ndarray,
        noise_dim: int = 100,
        hidden_dim: int = 256,
        batch_size: int = 64,
        learning_rate: float = 0.0002,
        betas: Tuple[float, float] = (0.5, 0.999),
        device: str = DEVICE,
    ):
        """
        Initialize LSTM GAN trainer.

        Args:
            real_data: Shape (n_windows, window_length)
            noise_dim: Dimension of input noise
            hidden_dim: Hidden dimension for LSTM layers
            batch_size: Training batch size
            learning_rate: Adam optimizer learning rate
            betas: Adam optimizer betas
            device: torch device (cpu or cuda)
        """
        self.real_data = real_data
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
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
        self.generator = LSTMGenerator(noise_dim, self.window_length, hidden_dim).to(device)
        self.discriminator = LSTMDiscriminator(self.window_length, hidden_dim).to(device)

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
        Train the LSTM GAN.

        Args:
            epochs: Number of training epochs
            log_interval: Log every N epochs

        Returns:
            Training history dict
        """
        logger.info(f"Starting LSTM GAN training for {epochs} epochs")

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

        logger.info("LSTM GAN training complete")
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
            "hidden_dim": self.hidden_dim,
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
            checkpoint = torch.load(path, map_location=device)

        trainer = cls.__new__(cls)
        trainer.device = device
        trainer.noise_dim = checkpoint["noise_dim"]
        trainer.hidden_dim = checkpoint["hidden_dim"]
        trainer.window_length = checkpoint["window_length"]
        trainer.batch_size = checkpoint.get("batch_size", 64)
        trainer.data_mean = checkpoint["data_mean"]
        trainer.data_std = checkpoint["data_std"]
        trainer.history = checkpoint["history"]

        trainer.generator = LSTMGenerator(trainer.noise_dim, trainer.window_length, trainer.hidden_dim).to(device)
        trainer.discriminator = LSTMDiscriminator(trainer.window_length, trainer.hidden_dim).to(device)

        trainer.generator.load_state_dict(checkpoint["generator_state"])
        trainer.discriminator.load_state_dict(checkpoint["discriminator_state"])

        logger.info(f"Model loaded from {path}")
        return trainer


def train_gan_lstm(
    real_data: np.ndarray,
    output_dir: str = "outputs",
    epochs: int = 50,
    **kwargs
) -> LSTMGANTrainer:
    """
    Convenience function to train LSTM GAN on financial data.

    Args:
        real_data: Array of shape (n_windows, 256)
        output_dir: Directory to save model
        epochs: Training epochs
        **kwargs: Additional arguments to LSTMGANTrainer

    Returns:
        Trained LSTMGANTrainer instance
    """
    Path(output_dir).mkdir(exist_ok=True)

    trainer = LSTMGANTrainer(real_data, **kwargs)
    trainer.train(epochs=epochs)

    model_path = Path(output_dir) / "gan_model_lstm.pt"
    trainer.save(str(model_path))

    return trainer


def generate_synthetic_lstm(
    trainer: LSTMGANTrainer, n_samples: int = 1000, seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic return windows from trained LSTM GAN.

    Args:
        trainer: Trained LSTMGANTrainer
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, 256)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    return trainer.generate(n_samples)


if __name__ == "__main__":
    # Example: train on sample data
    import pandas as pd

    parquet_path = Path("outputs/windowed_data_era_a.parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        step_cols = [c for c in df.columns if "_step_" in c]
        real_data = df[step_cols].values

        real_data = np.nan_to_num(real_data, nan=0.0)
        logger.info(f"Real data shape: {real_data.shape}")

        # Train LSTM GAN
        trainer = train_gan_lstm(real_data, epochs=50)

        # Generate synthetic data
        synthetic = generate_synthetic_lstm(trainer, n_samples=100)
        logger.info(f"Generated LSTM synthetic shape: {synthetic.shape}")
        logger.info(f"Synthetic mean: {synthetic.mean():.6f}, std: {synthetic.std():.6f}")
    else:
        logger.warning("Data file not found. Please run 01_data_pipeline.py first.")
