#!/usr/bin/env python3
"""Insert or refresh a 'Contents' cell built from the notebook's ## and ### headings.

Links use the heading ids nbconvert generates (spaces -> '-', non-ASCII percent-encoded).

Usage:
    python3 tools/build_toc.py DSM500_CW2_Report_Final.ipynb
"""

import json
import re
import sys
from urllib.parse import quote

TAG = "toc"
FIRST_SECTION = "Abstract"


def anchor(heading):
    """Return the nbconvert heading id for a heading (spaces to '-', non-ASCII percent-encoded)."""
    return quote(heading.strip().replace(" ", "-"), safe="!$&'()*+,;=:@-._~?")


def main(path):
    """Rebuild the tagged Contents cell from the notebook's ## and ### headings."""
    nb = json.load(open(path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if TAG not in c.get("metadata", {}).get("tags", [])]

    entries, started = [], False
    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).split("\n"):
            match = re.match(r"^(##|###) (.+?)\s*$", line)
            if not match:
                continue
            level, title = len(match.group(1)), match.group(2)
            started = started or title == FIRST_SECTION
            if started:
                entries.append((level, title))

    lines = ["## Contents", ""]
    for level, title in entries:
        indent = "" if level == 2 else "    "
        lines.append(f"{indent}- [{title}](#{anchor(title)})")
    toc = {"cell_type": "markdown", "metadata": {"tags": [TAG]}, "source": ["\n".join(lines) + "\n"]}

    cells.insert(1, toc)
    nb["cells"] = cells
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"contents: {len(entries)} headings")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DSM500_CW2_Report_Final.ipynb")
