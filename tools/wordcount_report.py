#!/usr/bin/env python3
"""Count the report under DSM500 CW2 April 2026 section 4.2.

Only Sections 1–6 are counted. Cover, abstract, contents, ethics, references,
appendices, headings, tables, figures/captions, quoted blocks, fenced code and
in-text citations (including narrative author names) are excluded.

Counting convention: whitespace separates words; en/em dashes also separate
words. Hyphenated terms, contractions, numbers, URLs and inline code count as
one word per whitespace-delimited token. Punctuation-only tokens and Markdown
list markers do not count. Quotation marks used as scare quotes do not identify
an externally quoted passage; review the exclusion audit for actual quotations.

This is a reproducible counting convention, not a guarantee of agreement with
an examiner's word processor. --audit-dir writes every counted word with its
source location and every exclusion, so the result can be checked independently.
Requires only the Python standard library; never modifies the notebook.
"""
import argparse
import csv
import hashlib
import html
import json
import re
from pathlib import Path

MAIN_HEADING = re.compile(r'^##\s+([1-9])\.\s+(.+?)\s*$')
YEAR = r'(?:19|20)\d{2}[a-z]?'
# A capitalised surname, with optional lowercase surname particles.
SURNAME = r'(?:(?:de|van|von|der|la|di|del)\s+)*[A-ZÀ-ÖØ-Þ][\w’\'-]*(?:\s+(?:(?:de|van|von|der|la|di|del)\s+)*[A-ZÀ-ÖØ-Þ][\w’\'-]*)*'
AUTHORS = rf'{SURNAME}(?:(?:,\s*|\s+and\s+){SURNAME})*(?:,?\s+and\s+{SURNAME})?(?:\s+et\s+al\.)?'
NARRATIVE = re.compile(rf'(?<!\w){AUTHORS}\s*\({YEAR}(?:,\s*(?:ch\.|p\.|pp\.)\s*[^)]*)?\)')
PARENTHETICAL = re.compile(r'\([^()]*\)')


def citation_parenthesis(value):
    inner = value[1:-1].strip()
    # Do not mistake experimental eras, counts or numeric results for citations.
    if re.search(r'\d{4}\s*[–—-]\s*\d', inner):
        return False
    if re.fullmatch(YEAR, inner):
        return True
    parts = inner.split(';')
    return all(re.fullmatch(rf'{AUTHORS},?\s+(?:{YEAR}|n\.d\.)(?:,\s*(?:ch\.|p\.|pp\.)\s*.*)?', p.strip()) for p in parts)


def count_report(path):
    path = Path(path)
    raw = path.read_bytes()
    nb = json.loads(raw)
    words, excluded, included = [], [], []
    section = None
    seen = []
    fenced = False
    fence_marker = None

    def omit(cell, line, reason, text):
        excluded.append(dict(cell=cell, line=line, section=section, reason=reason, text=text))

    for cell_index, cell in enumerate(nb['cells'], 1):
        if cell['cell_type'] != 'markdown':
            continue
        source = cell['source']
        source = ''.join(source) if isinstance(source, list) else source
        for line_index, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            heading = MAIN_HEADING.fullmatch(stripped)
            if heading:
                number = int(heading.group(1))
                section = number
                if 1 <= number <= 6:
                    seen.append(number)
                omit(cell_index, line_index, 'heading', line)
                continue
            if section not in range(1, 7):
                if stripped:
                    omit(cell_index, line_index, 'outside main body (Sections 1–6)', line)
                continue
            if not stripped:
                continue
            if stripped.startswith(('```', '~~~')):
                marker = stripped[:3]
                if not fenced:
                    fenced, fence_marker = True, marker
                elif marker == fence_marker:
                    fenced, fence_marker = False, None
                omit(cell_index, line_index, 'code fence', line)
                continue
            if fenced:
                omit(cell_index, line_index, 'code block', line)
                continue
            reason = None
            if stripped.startswith('#'):
                reason = 'heading'
            elif stripped.startswith('|'):
                reason = 'table'
            elif stripped.startswith('>'):
                reason = 'quotation block'
            elif stripped.startswith('![') or stripped.startswith('<img'):
                reason = 'figure'
            elif re.match(r'^[*_]+(?:Figure|Table)\s+\d', stripped):
                reason = 'figure/table caption'
            elif re.fullmatch(r'[-*_]{3,}', stripped):
                reason = 'horizontal rule'
            if reason:
                omit(cell_index, line_index, reason, line)
                continue
            cleaned = re.sub(r'^\s*(?:[-+*]|\d+[.)])\s+', '', line)
            # Render Markdown link labels; destinations are not extra body words.
            cleaned = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', cleaned)
            cleaned = re.sub(r'[*_`]', '', cleaned)
            cleaned = html.unescape(cleaned)

            def remove_narrative(match):
                omit(cell_index, line_index, 'narrative citation', match.group())
                return ' '

            def remove_parenthesis(match):
                if citation_parenthesis(match.group()):
                    omit(cell_index, line_index, 'parenthetical citation', match.group())
                    return ' '
                return match.group()

            cleaned = NARRATIVE.sub(remove_narrative, cleaned)
            cleaned = PARENTHETICAL.sub(remove_parenthesis, cleaned)
            cleaned = re.sub(r'[–—]', ' ', cleaned)
            tokens = [token for token in cleaned.split() if any(char.isalnum() for char in token)]
            included.append(dict(cell=cell_index, line=line_index, section=section,
                                 text=' '.join(tokens), count=len(tokens)))
            for token in tokens:
                words.append(dict(index=len(words)+1, section=section, cell=cell_index,
                                  line=line_index, word=token))
    if seen != list(range(1, 7)):
        raise ValueError(f'Expected main Sections 1–6 once, in order; found {seen}')
    if fenced:
        raise ValueError('Unclosed code fence in counted body')
    counts = {str(n): sum(w['section'] == n for w in words) for n in range(1, 7)}
    stated = re.findall(r'Word count:\s*([\d,]+)', '\n'.join(
        ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        for c in nb['cells'] if c['cell_type'] == 'markdown'))
    return dict(notebook=str(path.resolve()), notebook_sha256=hashlib.sha256(raw).hexdigest(),
                brief='DSM500_CW2April2026, section 4.2', total=len(words),
                by_section=counts, stated_counts=[int(s.replace(',', '')) for s in stated],
                words=words, included_lines=included, exclusions=excluded)


def body_words(path):
    """Compatibility entry point: count the main body excluding ethics."""
    return count_report(path)['total']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('notebook', nargs='?', default='DSM500_CW2_Report_Final.ipynb')
    parser.add_argument('--audit-dir', type=Path, help='Write summary.json, words.tsv, included.tsv and exclusions.tsv')
    args = parser.parse_args()
    result = count_report(args.notebook)
    for section, count in result['by_section'].items():
        print(f'Section {section}: {count:,}')
    print(f'Total body words (Sections 1–6): {result["total"]:,}')
    print(f'Stated count(s): {result["stated_counts"]}')
    if result['stated_counts'] != [result['total']]:
        print('WARNING: the report statement does not match this count.')
    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
        summary = {k: v for k, v in result.items() if k not in ('words', 'included_lines', 'exclusions')}
        (args.audit_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        for name, records in [('words', result['words']), ('included', result['included_lines']), ('exclusions', result['exclusions'])]:
            with (args.audit_dir/f'{name}.tsv').open('w', newline='', encoding='utf-8') as output:
                writer = csv.DictWriter(output, fieldnames=list(records[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(records)
        print(f'Word-by-word audit: {args.audit_dir.resolve()}')


if __name__ == '__main__':
    main()
