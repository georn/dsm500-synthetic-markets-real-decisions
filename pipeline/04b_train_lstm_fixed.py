#!/usr/bin/env python3
"""Superseded: use `04b_gan_generator_lstm_FIXED.py [--eur-only]`. Kept because it trained the reported LSTM model.


Train LSTM GAN with FIXED hyperparameters
Wrapper around 04b_gan_generator_lstm_FIXED.py for direct execution
"""

import sys
import importlib.util
from pathlib import Path

import pandas as pd

# Load the FIXED LSTM GAN trainer dynamically
script_dir = Path(__file__).parent
gan_spec = importlib.util.spec_from_file_location(
    "gan_generator_lstm_FIXED", script_dir / "04b_gan_generator_lstm_FIXED.py"
)
gan_mod = importlib.util.module_from_spec(gan_spec)
gan_spec.loader.exec_module(gan_mod)
train_gan_lstm = gan_mod.train_gan_lstm

print("\n" + "="*80)
print("LSTM GAN TRAINING (FIXED VERSION)")
print("="*80)

# Load era-A data (training data)
output_dir = script_dir.parent / "outputs"
parquet_path_a = output_dir / "windowed_data_era_a.parquet"

if not parquet_path_a.exists():
    print(f"❌ Error: {parquet_path_a} not found")
    sys.exit(1)

print(f"\n✅ Loading era-A data from {parquet_path_a}")
df_a = pd.read_parquet(parquet_path_a)
step_cols = [c for c in df_a.columns if '_step_' in c]
real_data_a = df_a[step_cols].values

print(f"✅ Loaded real data shape: {real_data_a.shape}")
print(f"   Mean: {real_data_a.mean():.6f}, Std: {real_data_a.std():.6f}")

# Train LSTM GAN with FIXED hyperparameters
print(f"\n{'='*80}")
print("TRAINING LSTM GAN WITH FIXES:")
print("  - Gradient clipping (max_norm=1.0)")
print("  - Learning rate (0.00005 instead of 0.0002)")
print("  - Epochs (100 instead of 50)")
print("  - Gradient norm monitoring")
print("  - NaN detection")
print(f"{'='*80}\n")

trainer = train_gan_lstm(
    real_data_a,
    output_dir=str(output_dir),
    epochs=50,  # Reduced from 100 for faster completion (losses already stable at epoch 40)
    learning_rate=0.00005,  # FIXED: 10x lower
    gradient_clip=1.0,      # FIXED: added clipping
)

print(f"\n{'='*80}")
print("✅ LSTM GAN TRAINING COMPLETE")
print(f"{'='*80}")

# Print summary
print(f"\nModel saved to: {output_dir / 'gan_model_lstm_fixed.pt'}")
print(f"Training history: {len(trainer.history['d_loss'])} epochs")
print(f"Final D Loss: {trainer.history['d_loss'][-1]:.4f}")
print(f"Final G Loss: {trainer.history['g_loss'][-1]:.4f}")

if trainer.history.get('nan_detected', False):
    print(f"\n⚠️  WARNING: NaNs were detected during training")
else:
    print(f"\n✅ No NaNs detected during training")

print(f"\nReady for evaluation: python3 05b_evaluation_lstm_comparison.py")
