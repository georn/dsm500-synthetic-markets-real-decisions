#!/usr/bin/env python3
"""Build video/presentation.html: self-contained slides with figures embedded and
speaker notes taken from video/VIDEO_SCRIPT.md.

Usage:
    python3 video/build_presentation.py
"""

import base64
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"
VIDEO = ROOT / "video"


def img(name, alt):
    """Return an <img> tag with the figure embedded as base64."""
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def notes():
    """Read the per-slide speaker notes from VIDEO_SCRIPT.md."""
    text = (VIDEO / "VIDEO_SCRIPT.md").read_text()
    sections = re.split(r"\n## ", text)[1:]
    out = []
    for section in sections:
        body = section.split("\n", 1)[1].split("\n---")[0].strip()
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        out.append("".join(
            f'<p class="presenter-cue">{html.escape(p[2:].strip())}</p>'
            if p.startswith("> ") else f"<p>{html.escape(p)}</p>"
            for p in paragraphs))
    return out


SLIDES = [
    ("title", """
        <p class="kicker">DSM500 Final Project · October 2026</p>
        <h1>Synthetic Markets, Real Decisions</h1>
        <p class="lead">Evaluating the downstream utility of generative models for trading-strategy validation</p>
        <p class="author">Guillermo Enrique Olmos Ranalli</p>"""),
    ("text", """
        <h2>The problem</h2>
        <ul class="big">
          <li>A backtest tests a strategy on <strong>one</strong> historical path, so tuning many strategies finds lucky ones (backtest overfitting).</li>
          <li>Synthetic data promises <strong>many</strong> alternative histories.</li>
          <li>But generators are judged mainly on <em>realism</em> (sometimes on forecasting), rarely on the decisions they lead to.</li>
        </ul>"""),
    ("figure", f"""
        <h2>Question and approach</h2>
        <p class="lead">Does a rule's rank on synthetic data predict its rank on <strong>later, unseen</strong> real data?</p>
        <div class="flow">
          <div><h3>Real data</h3><p>USD/EUR daily<br>2018–2024</p></div><span>→</span>
          <div><h3>Era split</h3><p>Build 2018–21<br>Test 2022–24</p></div><span>→</span>
          <div><h3>Generators</h3><p>GARCH<br>Bootstrap<br>MLP GAN<br>LSTM GAN</p></div><span>→</span>
          <div><h3>Backtest</h3><p>Every rule<br>on synthetic<br>and real data</p></div><span>→</span>
          <div><h3>Compare</h3><p>Rank agreement<br>(Spearman ρ)</p></div>
        </div>
        <p class="caption">Spearman rank correlation (ρ) between synthetic and real 2022–2024 Sharpe ratios.</p>"""),
    ("text", """
        <h2>Study design</h2>
        <div class="grid">
          <div><h3>Data</h3><p>USD/EUR daily<br>Build: 2018–2021<br>Test: 2022–2024</p></div>
          <div><h3>Generators</h3><p>GARCH(1,1)<br>Window bootstrap<br>MLP GAN<br>LSTM GAN<br>1,000 paths each</p></div>
          <div><h3>Rules</h3><p>500 unique rules<br>480 moving-average variants</p></div>
        </div>"""),
    ("figure", f"""
        <h2>What the paths show</h2>
        {img("fig_example_paths.png", "Five cumulative-return paths per source")}
        <p class="caption">Bootstrap matches history · GARCH calibrated · Saved GANs show fidelity limitations</p>"""),
    ("figure", f"""
        <h2>Main result</h2>
        {img("fig_rho_distinct.png", "Spearman rho with 95% confidence intervals by generator")}
        <p class="caption">GARCH ρ ≈ 0.04 · LSTM ≈ 0.03 · MLP ≈ −0.02 · Bootstrap ≈ −0.31 · Historical ≈ −0.29</p>"""),
    ("text", """
        <h2>Scope and checks</h2>
        <ul class="big">
          <li>484 rules shared by all five methods; 480 moving-average variants dominate the set.</li>
          <li>Paired resampling does not clearly separate GARCH from either GAN.</li>
          <li>Closely related rules may make the resampling intervals too narrow.</li>
          <li>Close agreement with training moments did not guarantee the best ranking.</li>
        </ul>"""),
    ("text", """
        <h2>Limitations</h2>
        <ul class="big">
          <li>Saved GANs retain pooled, zero-filled training; no new training in this rerun.</li>
          <li>Transaction costs and repeated training seeds are not evaluated.</li>
          <li>No synthetic ground-truth control; historical comparison uses one fixed split.</li>
          <li>One asset, one test period, mostly MA rules.</li>
        </ul>
        <p class="note">Corrected results regenerated: daily returns, unique rules and GARCH recursion fixed.</p>"""),
    ("text", """
        <h2>Conclusion</h2>
        <ul class="big">
          <li>No generator shows reliable positive ranking utility.</li>
          <li>Bootstrap and historical rankings are negatively associated with later performance.</li>
          <li>Close statistical fit did not guarantee the best ranking.</li>
          <li>Results apply to saved models and this USD/EUR experiment.</li>
        </ul>
        <p class="note">Next research: better trained models, broader rules and additional test periods.</p>"""),
]

CSS = """
:root { --ink:#1d2430; --muted:#5b6575; --accent:#2a5d8f; --surface:#ffffff; --panel:#f4f7fa; --rule:#d9e1ea; }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; background:#e9edf2; color:var(--ink);
  font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.deck { position:fixed; inset:0; display:grid; place-items:center; }
.slide { display:none; width:min(94vw, calc(88vh * 16 / 9)); aspect-ratio:16/9; background:var(--surface);
  border-radius:12px; box-shadow:0 10px 40px rgba(0,0,0,.12); padding:5% 6%; flex-direction:column; justify-content:center;
  font-size:min(2.1vw, 3.7vh); overflow:hidden; position:relative; }
.slide.active { display:flex; }
.slide::before { content:""; position:absolute; left:0; top:0; bottom:0; width:.5em; background:var(--accent); }
h1 { font-size:2.5em; line-height:1.1; margin:.2em 0 .4em; letter-spacing:-.01em; }
h2 { font-size:1.75em; margin:0 0 .6em; color:var(--accent); }
h3 { font-size:1em; margin:0 0 .4em; color:var(--accent); }
.kicker { text-transform:uppercase; letter-spacing:.12em; font-size:.7em; color:var(--muted); margin:0; }
.lead { font-size:1.1em; color:var(--muted); margin:0 0 .6em; }
.author { font-size:1em; margin-top:1.4em; font-weight:600; }
ul.big { margin:0; padding-left:1.1em; } ul.big li { margin:.45em 0; line-height:1.35; }
.grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:1em; }
.grid > div { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:.9em 1em; line-height:1.5; }
.grid p { margin:0; }
img { width:100%; max-height:62%; object-fit:contain; display:block; margin:0 auto; }
.flow { display:flex; align-items:center; gap:.4em; margin:.6em 0 .4em; }
.flow > div { flex:1; background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:.6em .5em; text-align:center; }
.flow h3 { font-size:.85em; margin:0 0 .3em; }
.flow p { margin:0; font-size:.72em; line-height:1.35; color:var(--muted); }
.flow > span { color:var(--accent); font-size:1.2em; }
.caption { text-align:center; color:var(--muted); font-size:.85em; margin:.6em 0 0; }
.note { margin-top:1em; padding:.6em .9em; background:var(--panel); border-left:3px solid var(--accent); font-size:.85em; color:var(--muted); }
.counter { position:absolute; right:1.2em; bottom:.8em; font-size:.6em; color:var(--muted); }
.progress { position:fixed; left:0; bottom:0; height:4px; background:var(--accent); transition:width .2s; }
.notes { display:none; position:fixed; left:2vw; right:2vw; bottom:2vh; max-height:34vh; overflow:auto; background:rgba(29,36,48,.94);
  color:#f2f4f7; border-radius:10px; padding:1em 1.4em; font-size:15px; line-height:1.5; }
.notes.show { display:block; }
.notes p { margin:.4em 0; }
.notes .presenter-cue { color:#b9dcff; font-weight:600; border-top:1px solid #6b8297; padding-top:.6em; margin-top:.8em; }
.help { position:fixed; bottom:10px; left:14px; font-size:12px; color:var(--muted); }
@media print { .notes, .help, .progress { display:none !important; } .slide { display:flex; page-break-after:always; box-shadow:none; } }
"""

JS = """
const slides = [...document.querySelectorAll('.slide')];
const notes = JSON.parse(document.getElementById('notes-data').textContent);
const panel = document.querySelector('.notes');
const bar = document.querySelector('.progress');
let i = Math.max(0, Math.min(slides.length - 1, (parseInt(location.hash.slice(1)) || 1) - 1));
function show(n) {
  i = Math.max(0, Math.min(slides.length - 1, n));
  slides.forEach((s, k) => s.classList.toggle('active', k === i));
  panel.innerHTML = notes[i] || '';
  bar.style.width = ((i + 1) / slides.length * 100) + '%';
  history.replaceState(null, '', '#' + (i + 1));
}
document.addEventListener('keydown', e => {
  if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(i + 1); }
  else if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) { e.preventDefault(); show(i - 1); }
  else if (e.key === 'Home') show(0);
  else if (e.key === 'End') show(slides.length - 1);
  else if (e.key.toLowerCase() === 'n') panel.classList.toggle('show');
  else if (e.key.toLowerCase() === 'f') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
});
document.addEventListener('click', e => { if (!panel.contains(e.target)) show(i + (e.clientX > innerWidth / 3 ? 1 : -1)); });
window.addEventListener('hashchange', () => show((parseInt(location.hash.slice(1)) || 1) - 1));
show(i);
"""


def build():
    """Write video/presentation.html from the slide definitions and notes."""
    speaker = notes()
    assert len(speaker) == len(SLIDES), (len(speaker), len(SLIDES))
    body = "\n".join(
        f'<section class="slide">{content}<span class="counter">{k + 1} / {len(SLIDES)}</span></section>'
        for k, (_, content) in enumerate(SLIDES))
    import json
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSM500 Video Slides</title><style>{CSS}</style></head>
<body>
<div class="help">→ / ← move · N notes · F fullscreen</div>
<main class="deck">{body}</main>
<div class="notes"></div><div class="progress"></div>
<script type="application/json" id="notes-data">{json.dumps(speaker)}</script>
<script>{JS}</script>
</body></html>"""
    (VIDEO / "presentation.html").write_text(page)
    print("wrote", VIDEO / "presentation.html", f"{len(page) / 1024:.0f} KB")


if __name__ == "__main__":
    build()
