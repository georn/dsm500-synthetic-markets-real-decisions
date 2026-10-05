# Reproducibility runbook

Run from the repository root after installing the README's environment:

```bash
export MPLBACKEND=Agg
```

The accepted report uses the completed corrected evaluation.

## Recompute statistics from the saved scores

This needs neither fresh downloads nor GAN generation/training:

```bash
mkdir -p rerun-reported/pipeline rerun-reported/outputs
cp -R outputs/. rerun-reported/outputs/
cp pipeline/10_distinct_rule_rq2.py pipeline/12_paired_rank_tests.py pipeline/15_refresh_report_tables.py rerun-reported/pipeline/
cp DSM500_CW2_Report_Final.ipynb rerun-reported/
.venv/bin/python rerun-reported/pipeline/10_distinct_rule_rq2.py
.venv/bin/python rerun-reported/pipeline/12_paired_rank_tests.py
.venv/bin/python rerun-reported/pipeline/15_refresh_report_tables.py
```

Outputs and refreshed tables stay inside `rerun-reported/`. The table refresher changes tables, not narrative.

## Rerun the corrected evaluation with frozen inputs

```bash
mkdir -p rerun-corrected
cp -R pipeline outputs rerun-corrected/
cp DSM500_CW2_Report_Final.ipynb rerun-corrected/
.venv/bin/python rerun-corrected/pipeline/02_baselines.py
.venv/bin/python rerun-corrected/pipeline/05_evaluation.py
.venv/bin/python rerun-corrected/pipeline/05b_evaluation_lstm_comparison.py
.venv/bin/python rerun-corrected/pipeline/07_walkforward_baseline.py
.venv/bin/python rerun-corrected/pipeline/10_distinct_rule_rq2.py
.venv/bin/python rerun-corrected/pipeline/12_paired_rank_tests.py
.venv/bin/python rerun-corrected/pipeline/13_gan_training_history.py
.venv/bin/python rerun-corrected/pipeline/14_data_description.py
.venv/bin/python rerun-corrected/pipeline/11_report_figures.py
.venv/bin/python rerun-corrected/pipeline/15_refresh_report_tables.py
```

This reuses the bundled pooled GAN checkpoints and seed 42. Numerical reproduction can depend on platform/library versions. The historical comparator is a fixed chronological comparison. The main evaluation took about six minutes on the original machine; LSTM evaluation and historical validation add computation.

## Fresh data or retraining

The accepted report uses the frozen files. For a new experiment, run the downloader before evaluation:

```bash
.venv/bin/python rerun-corrected/pipeline/01_data_pipeline.py
```

It attempts cryptocurrency collection and daily Yahoo Finance collection. The reported run used daily `EUR=X`, `SPY`, and `^FTSE` after cryptocurrency collection failed. Network access is required; new downloads overwrite staging windows and can differ from the frozen data.

For new single-asset models, train before evaluation:

```bash
.venv/bin/python rerun-corrected/pipeline/04_gan_generator.py --eur-only
.venv/bin/python rerun-corrected/pipeline/04b_gan_generator_lstm_FIXED.py --eur-only
```

The option uses the USD/EUR series `EUR=X`. LSTM training took about two hours on the original CPU. These new models are not those used in the accepted report.

## Archived pilot and RQ3

Archived scripts preserve the earlier pilot and known defects. They derive paths from their script location; reproducing that pilot requires putting them under a staging `pipeline/` and supplying the earlier archived outputs. Keep those results distinct from the corrected tables.

RQ3 remains deferred. `08_rq3_regime_testing.py` and its old output are superseded exploratory artifacts; they do not compare separate regimes or assets.
