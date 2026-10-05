# DSM500 CW2 reproducibility materials

**Synthetic Markets, Real Decisions: Evaluating the Downstream Utility of Generative Models for Trading-Strategy Validation**

The report now uses the completed corrected evaluation of 5 October 2026. Frozen data and saved GAN checkpoints were reused; no GAN retraining was performed.

## Start here

- `DSM500_CW2_Report_Final.ipynb`: report notebook and executable report cell.
- `DSM500_CW2_Report_Final.pdf` and `.html`: current review exports.
- `outputs/corrected_run_manifest.json`: completed-run configuration and input, source, and output hashes.
- `runbooks/01_data_pipeline_runbook.md`: execution order and reproducibility instructions.
- `SHA256SUMS`: integrity checksums for the packaged files.

## Contents

`pipeline/` contains the corrected code, including the GARCH recursion fix and `15_refresh_report_tables.py`. Report tables are generated from result files. Historical validation is one fixed chronological comparison, not repeated rolling refitting.

`outputs/` contains the Parquet return windows, original GAN checkpoints, corrected evaluation CSVs, statistical summaries, and report figures. The experiment specifies 500 uniquely named rules, with 488 defined scores for the classical methods and 484 for the GANs. GARCH and the GANs have correlations near zero; bootstrap and historical validation have negative correlations. RQ3 remains deferred.

`archive/pipeline_as_run_2026-10-05/` preserves the initial scripts. `archive/pre_corrected_rerun_2026-10-05/` preserves the earlier report and supporting results. Some superseded files remain in `outputs/` as in the original project; `outputs/README.md` identifies them. They are not current results.

`tests/`, `tools/`, and `GENERATE_PDF.sh` contain the updated tests and build helpers. `video/` contains the matching script, presentation builder, and presentation HTML. A recorded MP4 is not included.

Credentials, IG access-check scripts, virtual environments, caches, intermediate drafts, and downloaded literature PDFs are excluded.

## Environment

The recorded environment used Python 3.11.7:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m ipykernel install --user --name dsm500-cw2
```

The lock records the original environment. Alternatively, `requirements-build.txt` supplies pinned direct dependencies and notebook/export tools. Set `MPLBACKEND=Agg` for terminal plotting.

## Frozen data and checkpoints

The Parquet files contain return windows derived from Yahoo Finance daily `EUR=X` (USD/EUR: euros per dollar), `SPY`, and `^FTSE`. Raw prices were not retained. Fresh downloads may differ if the source revises historical data.

| File under outputs/ | Approximate size | SHA-256 |
|---|---:|---|
| `windowed_data_era_a.parquet` | 5.7 MiB | `5987ffb8cf0bcbf2b386fa3dd4098323cf33368543567e0c36ab9287af053adb` |
| `windowed_data_era_b.parquet` | 3.9 MiB | `342c28b24dbc7d5f7bb7f566bd37672055bf84364bcf5f1c22b20e764a89c344` |
| `gan_model.pt` | 4.0 MiB | `3791492449602508e4c69fd6dc8d04b49b1487ccc4a14ee8b12c173b443ece19` |
| `gan_model_lstm_fixed.pt` | 12.4 MiB | `14bf5969d62f4f3188c3cfa03f219db8c4d98289d057da92fb2dcdfe48ce1a24` |

The checkpoints retain pooled, zero-filled training. They are the models evaluated in the corrected report, not newly trained single-asset models. Reusing them avoids the original LSTM training cost of about two hours on CPU.

## Execute and export the report

To execute to a new file while preserving the supplied notebook:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=dsm500-cw2 --output DSM500_CW2_Report_Executed.ipynb DSM500_CW2_Report_Final.ipynb
.venv/bin/python -m playwright install chromium
.venv/bin/python tools/export_review_pdf.py DSM500_CW2_Report_Executed.ipynb
```

The packaged review exports use `tools/export_review_pdf.py`, which adjusts pagination for long Markdown cells. The original exporter is retained unchanged.

The copied `GENERATE_PDF.sh` runs tests, refreshes tables and contents, executes the original notebook in place, and exports PDF/HTML. Install the build dependencies and Chromium first.

## Tests and integrity

Run `.venv/bin/python -m pytest -q tests`. The completed-run manifest records 79 passed and two expected failures concerning GAN scale and training determinism.

Run `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux. The manifest excludes itself and Git metadata. The original source path in the completed-run manifest is provenance information; execution uses paths relative to this checkout.
