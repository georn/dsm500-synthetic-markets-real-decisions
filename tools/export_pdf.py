#!/usr/bin/env python3
"""Export the report notebook to self-contained HTML and to an A4 PDF with page numbers.

nbconvert renders the HTML (images embedded, input prompts hidden); Playwright's
Chromium prints it. nbconvert's own webpdf exporter always prints US Letter and has
no page-number footer, hence this script.

Usage:
    python3 tools/export_pdf.py DSM500_CW2_Report_Final.ipynb
"""

import sys
from pathlib import Path

from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

PRINT_CSS = """
<style>
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
@media print {
  body { font-size: 10.5pt; }
  .jp-Cell, img, table { break-inside: avoid; }
  h2, h3 { break-after: avoid; }
}
</style>
"""

FOOTER = ('<div style="font-size:8px;width:100%;text-align:center;color:#666;">'
          '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')


def main(notebook):
    """Render the notebook to HTML with nbconvert and print it to an A4 PDF with Playwright."""
    notebook = Path(notebook).resolve()
    exporter = HTMLExporter(embed_images=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_filename(str(notebook))
    body = body.replace("</head>", PRINT_CSS + "</head>", 1)

    html_path = notebook.with_suffix(".html")
    html_path.write_text(body, encoding="utf-8")

    pdf_path = notebook.with_suffix(".pdf")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.emulate_media(media="print")
        page.pdf(path=str(pdf_path), format="A4", print_background=True, prefer_css_page_size=True,
                 display_header_footer=True, header_template="<div></div>", footer_template=FOOTER)
        browser.close()
    print(f"wrote {html_path.name} and {pdf_path.name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DSM500_CW2_Report_Final.ipynb")
