"""Build the interpretability course: wrap each parts/<slug>.html body fragment in
the shared template (sidebar TOC, header, prev/next) -> course/<slug>.html.

Usage: python course/build.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (slug, nav title, header subtitle, part)
LESSONS = [
    ("index", "Overview", "What this course is and the question it answers", "Start"),
    ("01-what-is-interp", "1 · What is interpretability?",
     "Why open the black box, and the question we chose", "Foundations"),
    ("02-transformer", "2 · How a transformer works",
     "Tokens, the residual stream, attention, and MLPs — from scratch", "Foundations"),
    ("03-the-model", "3 · Our model and its task",
     "Sorting key–value pairs, the exact architecture, and why a decoder", "Foundations"),
    ("04-attention", "4 · Reading attention",
     "Attention heatmaps and finding an induction head", "Techniques"),
    ("05-causal", "5 · Causal methods",
     "Ablation, patching, logit attribution, and reading the weights", "Techniques"),
    ("06-probing-geometry", "6 · Probing & geometry",
     "Linear probes, PCA, effective dimensionality, embeddings", "Techniques"),
    ("07-the-algorithm", "7 · The algorithm it learned",
     "Assembling the circuit and cracking the comparison", "Synthesis"),
    ("08-universality-failures", "8 · Universality, failures & honesty",
     "Multiple seeds, where it breaks, and what we got wrong", "Synthesis"),
    ("09-conclusion", "9 · What you learned",
     "The big lessons and where to go next", "Synthesis"),
]

TShell = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Tiny Transformer Interpretability</title>
<link rel="stylesheet" href="course.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<button id="menu" aria-label="Toggle navigation">☰ Contents</button>
<div class="shell">
<nav class="side" id="side">
  <a class="brand" href="index.html">Tiny Transformer<span>Interpretability, from scratch</span></a>
  <ol class="toc">{toc}</ol>
  <p class="sidenote">A hands-on course built around one 51k-parameter model that
  learned to sort.</p>
</nav>
<main id="main">
<article>
<header class="lh">
  <p class="part">{part}</p>
  <h1>{h1}</h1>
  <p class="sub">{subtitle}</p>
</header>
{body}
<nav class="pn">{prevnext}</nav>
</article>
<footer class="foot">Part of the <a href="index.html">Tiny Transformer Interpretability</a>
course · reproducible from <code>research/sort_scaling/</code> ·
built by generating <code>course/build.py</code>.</footer>
</main>
</div>
<script>
document.getElementById('menu').addEventListener('click',()=>document.getElementById('side').classList.toggle('open'));
</script>
</body>
</html>
"""


def toc_html(current):
    out, part = [], None
    for i, (slug, nav, _sub, p) in enumerate(LESSONS):
        if p != part:
            out.append(f'<li class="grp">{p}</li>'); part = p
        cls = "cur" if slug == current else ""
        out.append(f'<li class="{cls}"><a href="{slug}.html">{nav}</a></li>')
    return "".join(out)


def prevnext(i):
    bits = []
    if i > 0:
        s, nav = LESSONS[i - 1][0], LESSONS[i - 1][1]
        bits.append(f'<a class="prev" href="{s}.html"><span>← Previous</span>{nav}</a>')
    else:
        bits.append('<span></span>')
    if i < len(LESSONS) - 1:
        s, nav = LESSONS[i + 1][0], LESSONS[i + 1][1]
        bits.append(f'<a class="next" href="{s}.html"><span>Next →</span>{nav}</a>')
    return "".join(bits)


def main():
    for i, (slug, nav, subtitle, part) in enumerate(LESSONS):
        body = (HERE / "parts" / f"{slug}.html").read_text()
        h1 = nav.split("·", 1)[-1].strip() if "·" in nav else nav
        page = TShell.format(title=nav.replace("·", "—"), toc=toc_html(slug),
                             part=part, h1=h1, subtitle=subtitle, body=body,
                             prevnext=prevnext(i))
        (HERE / f"{slug}.html").write_text(page)
        print("wrote course/" + slug + ".html")


if __name__ == "__main__":
    main()
