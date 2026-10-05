#!/usr/bin/env python3
"""Export the report with pagination suitable for review.

The original export_pdf.py remains unchanged. This wrapper allows long Markdown
cells to split across pages while keeping tables, figures, and code blocks intact.
"""

import sys

import export_pdf


REVIEW_CSS = """
  .jp-Cell, .jp-MarkdownCell { break-inside: auto; }
  .jp-CodeCell, img, table, pre { break-inside: avoid; }
  h1, h2, h3, h4 { break-after: avoid; }
  p, li { orphans: 3; widows: 3; }
  h2[id="Contents"], h2[id="Abstract"],
  h2[id^="1."], h2[id^="2."], h2[id^="3."],
  h2[id^="4."], h2[id^="5."], h2[id^="6."],
  h2[id^="7."], h2[id^="8."], h2[id^="9."] { break-before: page; }
  p:has(img), p:has(+ table) { break-after: avoid; }
"""


def main(notebook):
    export_pdf.PRINT_CSS = export_pdf.PRINT_CSS.replace(
        "</style>", "@media print {" + REVIEW_CSS + "}\n</style>"
    )
    export_pdf.main(notebook)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DSM500_CW2_Report_Final.ipynb")
