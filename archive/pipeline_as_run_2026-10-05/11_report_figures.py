#!/usr/bin/env python3
"""
Report figures that are built from existing outputs (no retraining, no new results).

Outputs (outputs/figures/):
- fig_workflow.png         conceptual workflow of the evaluation
- fig_rank_scatter.png     real vs synthetic Sharpe ranks per generator, distinct rules
- fig_example_paths.png    cumulative returns of five draws per source (real EUR/USD and each generator)

Usage:
    python3 pipeline/11_report_figures.py
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
INK, MUTED, GRID, ACCENT = "#222222", "#666666", "#e5e5e5", "#2a5d8f"
GENERATORS = [("garch", "GARCH(1,1)"), ("bootstrap", "Window bootstrap"),
              ("lstm", "LSTM GAN"), ("gan", "MLP GAN")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflow():
    steps = [
        ("Real data", "EUR/USD, SPY, FTSE 100\ndaily, 2018–2024"),
        ("Era split", "A: 2018–2021 (train)\nB: 2022–2024 (test)"),
        ("Generators", "GARCH · window bootstrap\nMLP GAN · LSTM GAN\n(built from era A)"),
        ("Backtest rules", "MA crossover, momentum,\nRSI-style rules →\nmean Sharpe per rule"),
        ("Compare", "Spearman ρ: synthetic\nvs real era-B Sharpe\n+ FDR, regret"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.4), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    width, gap = 1.7, 0.33
    for i, (title, body) in enumerate(steps):
        x = 0.1 + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, 0.35), width, 1.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#f4f7fa", edgecolor=ACCENT, linewidth=1.2))
        ax.text(x + width / 2, 1.66, title, ha="center", va="center", fontsize=10, weight="bold", color=INK)
        ax.text(x + width / 2, 1.0, body, ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.4)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.03, 1.2), (x + width + gap - 0.03, 1.2),
                                         arrowstyle="-|>", mutation_scale=12, color=ACCENT, linewidth=1.2))
    fig.tight_layout()
    fig.savefig(FIG / "fig_workflow.png")
    plt.close(fig)


def rank_scatter():
    main = pd.read_csv(OUT / "evaluation_results.csv")
    lstm = pd.read_csv(OUT / "evaluation_results_lstm.csv")
    df = main.merge(lstm[["strategy_name", "lstm_sharpe"]].drop_duplicates("strategy_name"),
                    on="strategy_name", how="left")
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates("strategy_name")
    stats = json.loads((OUT / "rq2_distinct_rules.json").read_text())["generators"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), dpi=200, sharey=True)
    for ax, (key, label) in zip(axes, GENERATORS):
        m = df[["real_sharpe", f"{key}_sharpe"]].dropna()
        real_rank = m["real_sharpe"].rank(pct=True)
        syn_rank = m[f"{key}_sharpe"].rank(pct=True)
        ax.scatter(syn_rank, real_rank, s=10, color=ACCENT, alpha=0.6, linewidths=0)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(f"{label}\nρ = {stats[key]['spearman_rho']:.2f}, n = {stats[key]['n']}", fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Synthetic Sharpe (rank)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Real era-B Sharpe (rank)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_rank_scatter.png")
    plt.close(fig)


def example_paths():
    import torch

    rng = np.random.default_rng(7)
    torch.manual_seed(7)
    era_a = pd.read_parquet(OUT / "windowed_data_era_a.parquet")
    era_b = pd.read_parquet(OUT / "windowed_data_era_b.parquet")
    eur_cols = [f"EUR=X_step_{i}" for i in range(256)]
    real_b = era_b[eur_cols].dropna().to_numpy()
    real_a = era_a[eur_cols].dropna().to_numpy()

    numeric_a = era_a.drop(columns=["asset", "window_id"]).to_numpy(dtype=float)
    garch_model = load_module("baselines", "02_baselines.py").GARCHModel()
    np.random.seed(7)
    garch_model.fit(np.nan_to_num(numeric_a, nan=0.0).flatten())

    mlp = load_module("gan_generator", "04_gan_generator.py").GANTrainer.load(str(OUT / "gan_model.pt"))
    lstm = load_module("gan_lstm", "04b_gan_generator_lstm_FIXED.py").LSTMGANTrainer.load(
        str(OUT / "gan_model_lstm_fixed.pt"))

    draws = 5
    mlp_paths = mlp.generate(n_samples=draws)[:, :256]
    lstm_paths = lstm.generate(n_samples=draws)[:, :256]
    np.random.seed(7)
    garch_paths = garch_model.generate(n_samples=draws, n_steps=256)
    panels = [
        ("Real era-B EUR/USD", real_b[rng.choice(len(real_b), draws, replace=False)]),
        ("Window bootstrap\n(era-A EUR/USD windows)", real_a[rng.choice(len(real_a), draws, replace=False)]),
        ("GARCH(1,1)", garch_paths),
        ("MLP GAN", mlp_paths),
        ("LSTM GAN", lstm_paths),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.8), dpi=200, sharey=True)
    for ax, (label, paths) in zip(axes, panels):
        for path in paths:
            ax.plot(np.cumsum(np.nan_to_num(path)), color=ACCENT, linewidth=1, alpha=0.75)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_title(f"{label}\nmedian daily sd {np.median(np.nanstd(paths, axis=1)):.4f}", fontsize=8.5, color=INK)
        ax.set_xlabel("Day in window")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Cumulative log return")
    fig.tight_layout()
    fig.savefig(FIG / "fig_example_paths.png")
    plt.close(fig)

    return {label.split(chr(10))[0]: float(np.median(np.nanstd(paths, axis=1))) for label, paths in panels}


if __name__ == "__main__":
    workflow()
    rank_scatter()
    print(json.dumps({"daily_return_std": example_paths()}, indent=1))
