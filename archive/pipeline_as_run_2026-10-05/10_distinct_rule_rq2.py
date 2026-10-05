#!/usr/bin/env python3
"""
RQ2 statistics recomputed on distinct trading rules.

evaluation_results.csv holds 500 rows but only 197 distinct strategy names; rows
sharing a name carry identical Sharpe ratios for every generator, so they are the
same rule counted more than once. Duplicates inflate n and narrow the CIs, so this
script keeps one row per name before computing every RQ2 statistic.

Inputs:  outputs/evaluation_results.csv, outputs/evaluation_results_lstm.csv
Outputs: outputs/rq2_distinct_rules.json, outputs/figures/fig_rho_distinct.png

Usage:
    python3 pipeline/10_distinct_rule_rq2.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
THRESHOLD = 0.1
GENERATORS = [("garch", "GARCH(1,1)"), ("bootstrap", "Window bootstrap"),
              ("lstm", "LSTM GAN"), ("gan", "MLP GAN")]


def fisher_ci(rho, n):
    z, se = np.arctanh(rho), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def main():
    main_df = pd.read_csv(OUT / "evaluation_results.csv")
    lstm_df = pd.read_csv(OUT / "evaluation_results_lstm.csv")
    df = main_df.merge(lstm_df[["strategy_name", "lstm_sharpe"]].drop_duplicates("strategy_name"),
                       on="strategy_name", how="left")
    df = df.replace([np.inf, -np.inf], np.nan)
    distinct = df.drop_duplicates("strategy_name").copy()
    distinct["type"] = distinct["strategy_name"].str.split("_").str[0]

    results = {"rows_in_csv": len(df), "distinct_rules": len(distinct),
               "endorsement_threshold": THRESHOLD, "generators": {}}
    for key, _ in GENERATORS:
        col = f"{key}_sharpe"
        m = distinct[["real_sharpe", col, "type"]].dropna()
        rho, p = stats.spearmanr(m["real_sharpe"], m[col])
        lo, hi = fisher_ci(rho, len(m))
        ma = m[m["type"] == "MA"]
        endorsed = m[m[col] > THRESHOLD]
        n_work = int((endorsed["real_sharpe"] > THRESHOLD).sum())
        top_pick = float(m.loc[m[col].idxmax(), "real_sharpe"])
        oracle = float(m["real_sharpe"].max())
        optimism = m[col] - m["real_sharpe"]
        results["generators"][key] = {
            "n": len(m), "types": m["type"].value_counts().to_dict(),
            "spearman_rho": float(rho), "spearman_p": float(p), "fisher_ci": [lo, hi],
            "ma_only_rho": float(stats.spearmanr(ma["real_sharpe"], ma[col])[0]), "ma_only_n": len(ma),
            "n_endorsed": len(endorsed), "n_endorsed_work": n_work,
            "fdr": (1 - n_work / len(endorsed)) if len(endorsed) else None,
            "optimism_mean": float(optimism.mean()), "optimism_median": float(optimism.median()),
            "top_pick_real_sharpe": top_pick, "oracle_real_sharpe": oracle,
            "selection_regret": oracle - top_pick,
        }

    (OUT / "rq2_distinct_rules.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    labels = [label for _, label in GENERATORS]
    rows = [results["generators"][k] for k, _ in GENERATORS]
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        lo, hi = r["fisher_ci"]
        ax.plot([lo, hi], [yi, yi], color="#2a5d8f", linewidth=2, solid_capstyle="round")
        ax.plot(r["spearman_rho"], yi, "o", color="#2a5d8f", markersize=8,
                markeredgecolor="white", markeredgewidth=2)
        ax.text(hi + 0.015, yi, f"ρ = {r['spearman_rho']:.2f}  (n = {r['n']})",
                va="center", fontsize=9, color="#333333")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.9)
    ax.axvline(0, color="#999999", linewidth=1)
    ax.set_xlabel("Spearman ρ between synthetic and real era-B Sharpe ratios (95% Fisher-z CI)")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_rho_distinct.png")
    print(json.dumps({k: (v["n"], round(v["spearman_rho"], 3), [round(x, 3) for x in v["fisher_ci"]],
                          v["n_endorsed"], v["n_endorsed_work"], round(v["optimism_mean"], 3),
                          round(v["optimism_median"], 3), round(v["selection_regret"], 3),
                          round(v["ma_only_rho"], 3))
                      for k, v in results["generators"].items()}, indent=1))


if __name__ == "__main__":
    main()
