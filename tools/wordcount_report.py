#!/usr/bin/env python3
"""Body word count of the report notebook (Sections 1-7).

Excludes the title, abstract, contents, headings, tables, figures, quoted notes,
in-text (Author, year) citations, references and appendices. Check the brief for
the official counting rules before relying on this number.

Usage:
    python3 tools/wordcount_report.py DSM500_CW2_Report_Final.ipynb
"""

import json
import re
import sys


def body_words(path):
    """Count words in Sections 1-7, excluding headings, tables, captions and citations."""
    nb = json.load(open(path, encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    body = text.split("\n## 1. Introduction", 1)[1].split("\n## 8. References", 1)[0]
    body = re.sub(r"\([^)]*\d{4}[^)]*\)", "", body)
    body = "\n".join(line for line in body.split("\n")
                     if not line.lstrip().startswith(("#", "|", "---", "!", ">", "*Figure", "*Table")))
    return len(re.sub(r"[*_>`\[\]]", "", body).split())


if __name__ == "__main__":
    print(f"body words (Sections 1-7): {body_words(sys.argv[1] if len(sys.argv) > 1 else 'DSM500_CW2_Report_Final.ipynb')}")
