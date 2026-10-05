#!/bin/bash
# Build the report: run tests, refresh the contents list, execute the notebook,
# export self-contained HTML and an A4 PDF with page numbers, and print the word count.
# One-time setup: .venv/bin/pip install -r requirements-build.txt && .venv/bin/python -m playwright install chromium
set -euo pipefail
cd "$(dirname "$0")"
NB=DSM500_CW2_Report_Final.ipynb
.venv/bin/python -m pytest -q tests
.venv/bin/python pipeline/15_refresh_report_tables.py
.venv/bin/python tools/build_toc.py "$NB"
.venv/bin/jupyter nbconvert --to notebook --execute --inplace "$NB"
.venv/bin/python tools/export_pdf.py "$NB"
.venv/bin/python tools/wordcount_report.py "$NB"
