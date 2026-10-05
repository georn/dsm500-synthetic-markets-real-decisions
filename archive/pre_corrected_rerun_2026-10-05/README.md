# DSM500 CW2: Synthetic Markets, Real Decisions

Evaluating the downstream utility of generative models for trading-strategy validation.

Report: `DSM500_CW2_Report_Final.ipynb` (exported to `.pdf` and `.html`).

## Environment

Python 3.11.7.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt                 # exact environment
# or: .venv/bin/pip install -r requirements-build.txt          # pinned direct dependencies + notebook tools
.venv/bin/python -m playwright install chromium               # only needed for the PDF export
```

`requirements.txt` pins the pipeline's direct dependencies, and `requirements-build.txt` adds Jupyter and Playwright. `requirements-lock.txt` is a full `pip freeze` of the environment used on 2026-10-05.

No `.env` is needed. `.env.example` holds IG demo credentials, used only by `pipeline/ig_test.py` and `pipeline/check_data_access.py`; no reported result uses IG.

## Data

`pipeline/01_data_pipeline.py` downloads daily closes for `EUR=X` (USD/EUR: euros per US dollar), `SPY` and `^FTSE` with yfinance (`start="2018-01-01"`, `end="2024-12-31"`, end date exclusive). `auto_adjust` is left at the library default, which is `True` in yfinance 1.7.0. Raw prices are not saved. The windowed returns are the frozen dataset and are tracked in git:

| File | SHA-256 |
|---|---|
| `outputs/windowed_data_era_a.parquet` | `5987ffb8cf0bcbf2b386fa3dd4098323cf33368543567e0c36ab9287af053adb` |
| `outputs/windowed_data_era_b.parquet` | `342c28b24dbc7d5f7bb7f566bd37672055bf84364bcf5f1c22b20e764a89c344` |

Re-downloading may not reproduce these files exactly, because Yahoo data can be revised.

## Pipeline

Run from `attempt-01/`. Seeds are fixed (`np.random.seed(42)`, `torch.manual_seed(42)`), but GAN training is not deterministic (see Tests).

**Code status (2026-10-05).** The results in `outputs/` and in the report were produced by the scripts archived in `archive/pipeline_as_run_2026-10-05/`. The pipeline has since been fixed but **not rerun**:
- `03_strategies.py`: unique rule names.
- `05_evaluation.py`: holding-period returns, moving averages on the log-price path, daily position-based Sharpe ratio, and USD/EUR rows only (no zero-filled bootstrap rows; GARCH fitted on the USD/EUR series).
- `05b_evaluation_lstm_comparison.py`: reuses the fixed evaluator.
- `02_baselines.py`: a true stationary bootstrap, a GARCH fit with variance targeting that ignores NaN, and per-window RQ1 fidelity on USD/EUR for every generator.
- `07_walkforward_baseline.py`: a genuine walk-forward baseline (era-A vs era-B backtests).
- RSI-style rule uses past returns only (no look-ahead); GAN sampling is seeded.
- `04_gan_generator.py` / `04b_gan_generator_lstm_FIXED.py --eur-only` train on USD/EUR windows and save `*_eur.pt` with a loss history; the evaluation prefers these models when present.

The saved GAN models were trained on the pooled, zero-filled matrix; a clean comparison needs retraining on USD/EUR.

```bash
.venv/bin/python pipeline/01_data_pipeline.py                  # download + windows (network)
.venv/bin/python pipeline/02_baselines.py                      # RQ1 fidelity on USD/EUR, all generators
.venv/bin/python pipeline/04_gan_generator.py --eur-only       # MLP GAN on USD/EUR (omit flag = as-run pooled training)
.venv/bin/python pipeline/04b_gan_generator_lstm_FIXED.py --eur-only  # LSTM GAN on USD/EUR (~2 h on CPU)
.venv/bin/python pipeline/05_evaluation.py                     # per-rule Sharpe, GARCH / bootstrap / MLP GAN
.venv/bin/python pipeline/05b_evaluation_lstm_comparison.py    # per-rule Sharpe, LSTM GAN
.venv/bin/python pipeline/12_paired_rank_tests.py              # paired comparison of generators (Tables 4.4-4.5)
.venv/bin/python pipeline/13_gan_training_history.py           # GAN loss histories from checkpoints
.venv/bin/python pipeline/14_data_description.py              # descriptive statistics, Figure 3.2
.venv/bin/python pipeline/07_walkforward_baseline.py          # genuine walk-forward baseline
.venv/bin/python pipeline/10_distinct_rule_rq2.py              # RQ2 statistics used in the report
.venv/bin/python pipeline/11_report_figures.py                 # report figures
./GENERATE_PDF.sh                                              # execute notebook, export PDF + HTML
```

Superseded scripts, kept for the record but not used in the report: `06_bootstrap_resampling_validation.py` (three generators, replaced by 12), `04b_train_lstm_fixed.py` (trained the reported LSTM), `06_rq2_secondary_metrics.py` (all rows, old LSTM run), `07_sensitivity_analysis.py` (all rows), `08_rq3_regime_testing.py` (RQ3 deferred), `09_generate_figures.py` (stale figures).

## Tests

```bash
.venv/bin/python -m pytest -q tests      # 78 passed, 2 xfailed on 2026-10-05
```

`tests/conftest.py` maps the numbered scripts to the module names the tests import. `tests/test_known_defects.py` holds regression tests for the defects fixed on 2026-10-05. The two expected failures (`xfail(strict=True)`) are GAN issues that need retraining: output scale far below real returns, and non-deterministic training under a fixed seed.

## Building the report

```bash
./GENERATE_PDF.sh
```

This runs the tests, refreshes the contents list (`tools/build_toc.py`), executes the notebook, exports self-contained HTML and an A4 PDF with page numbers (`tools/export_pdf.py`), and prints the body word count (`tools/wordcount_report.py`).
