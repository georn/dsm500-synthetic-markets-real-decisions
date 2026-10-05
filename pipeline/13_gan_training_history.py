#!/usr/bin/env python3
"""
Export the training-loss history stored inside each saved GAN checkpoint.

The checkpoints written by GANTrainer.save / LSTMGANTrainer.save include the per-epoch
discriminator and generator losses. This script reads them directly, replacing the
hand-written figures in outputs/gan_training_summary.json, which no script produced.

Inputs:  outputs/gan_model*.pt (any that exist)
Outputs: outputs/gan_training_history.json

Usage:
    python3 pipeline/13_gan_training_history.py
"""

import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "outputs"
MODELS = ["gan_model.pt", "gan_model_lstm_fixed.pt", "gan_model_eur.pt", "gan_model_lstm_eur.pt"]


def summarise(path):
    """Return epochs, window length and first/last losses recorded in one checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", {})
    summary = {"window_length": checkpoint.get("window_length"), "epochs": len(history.get("d_loss", []))}
    for key in ("d_loss", "g_loss"):
        values = [float(v) for v in history.get(key, [])]
        if values:
            summary[key] = {"first": values[0], "last": values[-1], "min": min(values), "max": max(values),
                            "per_epoch": values}
    return summary


def main():
    """Summarise every checkpoint that exists and write the JSON."""
    result = {name: summarise(OUT / name) for name in MODELS if (OUT / name).exists()}
    (OUT / "gan_training_history.json").write_text(json.dumps(result, indent=1))
    for name, s in result.items():
        print(name, {k: (v if not isinstance(v, dict) else {x: round(v[x], 4) for x in ("first", "last")})
                     for k, v in s.items()})


if __name__ == "__main__":
    main()
