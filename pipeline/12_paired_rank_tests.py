#!/usr/bin/env python3
"""
Paired resampling tests for RQ2: is one generator's rank correlation higher than another's?

Comparing marginal confidence intervals is conservative when both correlations are
computed on the same rules. This script resamples the distinct rules common to all generators and the historical-validation baseline (with replacement, 1,000 times), recomputes every generator's Spearman rho on
each resample, and reports percentile intervals for each rho and for every pairwise
difference.

Caveat: rules are resampled as if independent. Many moving-average variants are strongly
correlated, so the effective number of rules is smaller and these intervals are likely
too narrow.

Inputs:  outputs/evaluation_results.csv, outputs/evaluation_results_lstm.csv
Outputs: outputs/paired_rank_tests.json

Usage:
    python3 pipeline/12_paired_rank_tests.py
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "outputs"
GENERATORS = ["garch", "bootstrap", "lstm", "gan", "wf"]
N_RESAMPLES = 1000
SEED = 42


def distinct_complete_rules():
    """One row per rule name with real and synthetic Sharpe ratios defined for every generator."""
    main = pd.read_csv(OUT / "evaluation_results.csv")
    lstm = pd.read_csv(OUT / "evaluation_results_lstm.csv")
    df = main.merge(lstm[["strategy_name", "lstm_sharpe"]].drop_duplicates("strategy_name"),
                    on="strategy_name", how="left")
    wf = pd.read_csv(OUT / "walkforward_comparison.csv").rename(columns={"era_a_sharpe": "wf_sharpe"})
    df = df.merge(wf[["strategy_name", "wf_sharpe"]], on="strategy_name", how="left", validate="one_to_one")
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates("strategy_name")
    return df.dropna(subset=["real_sharpe"] + [f"{g}_sharpe" for g in GENERATORS]).reset_index(drop=True)


def interval(values):
    """Mean and 95% percentile interval of a resampling distribution."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci": [float(lo), float(hi)]}


def main():
    """Run the paired resampling and write the summary JSON."""
    df = distinct_complete_rules()
    rng = np.random.default_rng(SEED)
    n = len(df)
    rhos = {g: np.empty(N_RESAMPLES) for g in GENERATORS}
    for b in range(N_RESAMPLES):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        for g in GENERATORS:
            rhos[g][b] = stats.spearmanr(sample[f"{g}_sharpe"], sample["real_sharpe"])[0]

    result = {
        "n_rules": n,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "point_rho": {g: float(stats.spearmanr(df[f"{g}_sharpe"], df["real_sharpe"])[0]) for g in GENERATORS},
        "rho": {g: interval(rhos[g]) for g in GENERATORS},
        "differences": {},
    }
    for a, b in itertools.combinations(GENERATORS, 2):
        diff = rhos[a] - rhos[b]
        result["differences"][f"{a}-{b}"] = {**interval(diff), "share_le_zero": float(np.mean(diff <= 0))}

    (OUT / "paired_rank_tests.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
