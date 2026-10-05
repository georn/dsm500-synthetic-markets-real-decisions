# Output provenance

The current report uses the completed corrected evaluation recorded in `corrected_run_manifest.json`.

Current inputs are the two `windowed_data_era_*.parquet` files and `gan_model.pt` / `gan_model_lstm_fixed.pt`. Current results are `evaluation_results*.csv`, `evaluation_summary*.json`, `walkforward_comparison.csv`, `walkforward_results.json`, `rq2_distinct_rules.json`, `paired_rank_tests.json`, the fidelity files, and `data_description.json`. Training history comes from the checkpoints via `gan_training_history.json`.

The report links five figures: `fig_workflow.png`, `fig_data_overview.png`, `fig_example_paths.png`, `fig_rho_distinct.png`, and `fig_rank_scatter.png`.

## Legacy artifacts

Other outputs remain byte-for-byte from the original project for provenance. They are not corrected report results: `bootstrap_results.json`, `bootstrap_resamples.csv`, `spearman_ci_results.json`, `sensitivity_analysis_*`, `rq2_secondary_metrics.*`, `rq3_regime_results.json`, and `walkforward_baseline.json`. The last is the invalid earlier placeholder; use `walkforward_results.json` for the actual historical comparison.

`gan_training_summary.json` is the earlier untraceable summary discussed in the report. Use checkpoint-derived history instead. Unused figures such as `fig_rq3_regime.png` are earlier exploratory artifacts, not a completed RQ3 experiment.

The earlier report/results are also preserved in `archive/pre_corrected_rerun_2026-10-05/`. Raw prices are not included. The corrected run reused saved checkpoints without retraining.
