#!/usr/bin/env python3
"""
Descriptive statistics and an overview figure for the three assets and two eras.

Each era's daily log-return series is rebuilt from its stride-1 windows (first window
plus the last value of every later window). The stored windows carry no dates, so the
figure's x axis counts trading days from the start of era A.

Inputs:  outputs/windowed_data_era_a.parquet, outputs/windowed_data_era_b.parquet
Outputs: outputs/data_description.json, outputs/figures/fig_data_overview.png

Usage:
    python3 pipeline/14_data_description.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "outputs"
ASSETS = [("EUR=X", "USD/EUR"), ("SPY", "SPY"), ("^FTSE", "FTSE 100")]
INK, MUTED, GRID, ACCENT = "#222222", "#666666", "#e5e5e5", "#2a5d8f"


def series(df, ticker):
    """Rebuild one asset's daily log-return series from its overlapping windows."""
    cols = [f"{ticker}_step_{i}" for i in range(256)]
    windows = df.loc[df["asset"] == ticker, cols].to_numpy(float)
    return np.concatenate([windows[0], windows[1:, -1]])


def describe(r):
    """Summary statistics for a daily log-return series."""
    cumulative = np.cumsum(r)
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    return {"n": int(len(r)), "mean_daily_pct": float(100 * r.mean()),
            "annualised_vol_pct": float(100 * r.std() * np.sqrt(252)),
            "skewness": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
            "max_drawdown_pct": float(100 * (np.exp(drawdown.min()) - 1)),
            "acf_squared_lag1": float(np.corrcoef(r[:-1] ** 2, r[1:] ** 2)[0, 1])}


def main():
    """Write per-asset, per-era statistics and draw the USD/EUR overview figure."""
    era_a = pd.read_parquet(OUT / "windowed_data_era_a.parquet")
    era_b = pd.read_parquet(OUT / "windowed_data_era_b.parquet")
    result = {label: {"era_a": describe(series(era_a, t)), "era_b": describe(series(era_b, t))}
              for t, label in ASSETS}
    (OUT / "data_description.json").write_text(json.dumps(result, indent=2))

    a, b = series(era_a, "EUR=X"), series(era_b, "EUR=X")
    r = np.concatenate([a, b])
    vol = pd.Series(r).rolling(60).std() * np.sqrt(252) * 100
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4.6), dpi=200, sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    top.plot(np.cumsum(r), color=ACCENT, linewidth=1.2)
    top.set_ylabel("Cumulative log return")
    bottom.plot(vol, color=INK, linewidth=1)
    bottom.set_ylabel("60-day volatility\n(annualised, %)")
    bottom.set_xlabel("Trading day (era A starts at 0)")
    for ax in (top, bottom):
        ax.axvline(len(a), color=MUTED, linestyle="--", linewidth=0.9)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top.text(len(a) * 0.5, top.get_ylim()[1] * 0.92, "Era A (2018–2021)", ha="center", color=MUTED, fontsize=8)
    top.text(len(a) + len(b) * 0.5, top.get_ylim()[1] * 0.92, "Era B (2022–2024)", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_data_overview.png")
    print(json.dumps({k: {e: {m: round(v, 3) for m, v in s.items()} for e, s in d.items()}
                      for k, d in result.items()}, indent=1))


if __name__ == "__main__":
    main()
