"""Build a self-contained interactive 3D embedding viewer (keys vs values).

Computes the top-3 PCA coords of the key and value embedding tables from the
checkpoint and injects them into a standalone HTML page with a canvas 3D scatter
(drag to rotate, id-ordered path). Also published as an Artifact.

Usage: python research/sort_scaling/build_emb3d.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research" / "figviz"))
sys.path.insert(0, str(HERE))
import causal_sort as cs  # noqa: E402
from deep_interp import load  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "embedding_3d.html")


def pca3(E):
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    coords = Ec @ Vt[:3].T
    coords = coords / np.abs(coords).max()
    var3 = float((S ** 2 / (S ** 2).sum())[:3].sum())
    return coords.round(4).tolist(), var3


model, _ = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
W = model.token_embedding.weight.detach().cpu().numpy()
ck, vk = pca3(W[:cs.KEY_ALPH])
cv, vv = pca3(W[cs.KEY_ALPH:cs.KEY_ALPH + cs.VAL_ALPH])
data = {"keys": {"coords": ck, "var3": vk}, "values": {"coords": cv, "var3": vv}}

TEMPLATE = r"""<title>Embedding geometry: keys vs values</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #10151b; --muted: #5b6774;
    --line: #e4e8ee; --accent: #2f6df0; --grid: #eef1f5;
    --key-tint: #2f6df0; --val-tint: #e07b39;
    --canvas-bg: #0f1720;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0b0f14; --panel: #121820; --ink: #e9edf2; --muted: #8b98a6;
      --line: #212a34; --accent: #6ea0ff; --grid: #1a2028;
      --canvas-bg: #0a1017;
    }
  }
  :root[data-theme="light"] {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #10151b; --muted: #5b6774;
    --line: #e4e8ee; --accent: #2f6df0; --grid: #eef1f5; --canvas-bg: #0f1720;
  }
  :root[data-theme="dark"] {
    --bg: #0b0f14; --panel: #121820; --ink: #e9edf2; --muted: #8b98a6;
    --line: #212a34; --accent: #6ea0ff; --grid: #1a2028; --canvas-bg: #0a1017;
  }
  * { box-sizing: border-box; }
  body { margin: 0; }
  .wrap {
    max-width: 1080px; margin: 0 auto; padding: 40px 24px 56px;
    color: var(--ink); background: var(--bg);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace; }
  .eyebrow {
    font-family: ui-monospace, Menlo, monospace; font-size: 11px; letter-spacing: .18em;
    text-transform: uppercase; color: var(--accent); margin: 0 0 14px;
  }
  h1 {
    font-size: clamp(26px, 4.2vw, 40px); line-height: 1.08; margin: 0 0 14px;
    font-weight: 640; letter-spacing: -0.02em; text-wrap: balance; max-width: 20ch;
  }
  .lede { margin: 0; max-width: 66ch; color: var(--muted); font-size: 16px; line-height: 1.6; }
  .lede b { color: var(--ink); font-weight: 600; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    margin: 30px 0 18px; padding-bottom: 18px; border-bottom: 1px solid var(--line);
  }
  button {
    font: inherit; font-size: 13px; color: var(--ink); background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 13px; cursor: pointer;
    display: inline-flex; align-items: center; gap: 8px; transition: border-color .15s, background .15s;
  }
  button:hover { border-color: var(--accent); }
  button[aria-pressed="true"] { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 12%, var(--panel)); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted); }
  button[aria-pressed="true"] .dot { background: var(--accent); }
  .hint { margin-left: auto; color: var(--muted); font-size: 12.5px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  figure { margin: 0; }
  .panel {
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 16px 16px 14px; display: flex; flex-direction: column; gap: 10px;
  }
  .phead { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
  .ptitle { font-size: 15px; font-weight: 620; letter-spacing: -0.01em; }
  .ptitle span { color: var(--muted); font-weight: 500; }
  .tag { font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--muted); }
  .cv { position: relative; width: 100%; aspect-ratio: 1 / 1; border-radius: 10px; overflow: hidden;
        background: var(--canvas-bg); cursor: grab; }
  .cv.drag { cursor: grabbing; }
  canvas { display: block; width: 100%; height: 100%; touch-action: none; }
  .verdict {
    font-family: ui-monospace, Menlo, monospace; font-size: 12px; line-height: 1.5;
    color: var(--muted); border-top: 1px solid var(--line); padding-top: 10px;
  }
  .verdict b { color: var(--ink); font-weight: 600; }
  .legend { display: flex; align-items: center; gap: 9px; }
  .legend .bar { height: 9px; flex: 1; border-radius: 5px;
    background: linear-gradient(90deg,#440154,#414487,#2a788e,#22a884,#7ad151,#fde725); }
  .legend .lab { font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums; }
  footer { margin-top: 30px; color: var(--muted); font-size: 13px; line-height: 1.6; max-width: 72ch; }
  footer code { font-family: ui-monospace, Menlo, monospace; font-size: 12px;
    background: var(--grid); padding: 1px 5px; border-radius: 4px; }
</style>

<div class="wrap">
  <p class="eyebrow">Sort decoder · token embeddings</p>
  <h1>Keys are ordered. Values are an arbitrary code.</h1>
  <p class="lede">
    The tiny sort transformer must <b>compare</b> keys but only <b>copy</b> values (by
    identity). So it learns to lay its 50 key embeddings out <b>by numeric value</b> —
    while the 20 value embeddings stay an unordered cluster. Each dot is one token's
    48-dim embedding, projected to its top 3 principal components and coloured by id.
    Turn on the <b>id path</b> (0→1→2→…): for keys it threads a coherent route; for
    values it tangles. Drag to rotate.
  </p>

  <div class="controls">
    <button id="path" aria-pressed="true"><span class="dot"></span>id path</button>
    <button id="spin" aria-pressed="true"><span class="dot"></span>auto-rotate</button>
    <button id="reset">reset view</button>
    <span class="hint">drag to rotate · scroll to zoom</span>
  </div>

  <div class="grid">
    <figure class="panel">
      <div class="phead">
        <div class="ptitle">Keys <span>· ids 0–49 · compared</span></div>
        <div class="tag">top-3 PCs = KEYVAR% var</div>
      </div>
      <div class="cv" id="cv-keys"><canvas></canvas></div>
      <div class="legend"><span class="lab">0</span><div class="bar"></div><span class="lab">49</span></div>
      <div class="verdict">interpolation id-decode r = <b>0.79</b> · a real order code</div>
    </figure>
    <figure class="panel">
      <div class="phead">
        <div class="ptitle">Values <span>· ids 0–19 · payload</span></div>
        <div class="tag">top-3 PCs = VALVAR% var</div>
      </div>
      <div class="cv" id="cv-values"><canvas></canvas></div>
      <div class="legend"><span class="lab">0</span><div class="bar"></div><span class="lab">19</span></div>
      <div class="verdict">interpolation id-decode r = <b>0.09</b> · no order</div>
    </figure>
  </div>

  <footer>
    Embeddings from the length-30 causal pair-sort decoder
    (<code>causal_pairs_L30.pt</code>, d&#95;model = 48). Coordinates are the top-3
    principal components of each alphabet's embedding table; keys spread across more
    directions (only ~20% of key variance lives in 3 PCs, vs ~41% for values), so the
    key cloud is genuinely higher-dimensional. See <code>reports/sort-embeddings.md</code>.
  </footer>
</div>

<script>
const DATA = /*DATA*/;

const VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
  [31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
function viridis(t){
  t = Math.max(0, Math.min(1, t));
  const x = t*(VIRIDIS.length-1), i = Math.floor(x), f = x-i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i+1, VIRIDIS.length-1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
}

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const state = { showPath: true, spin: !reduceMotion };

class Scatter {
  constructor(box, coords){
    this.box = box; this.canvas = box.querySelector('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.pts = coords; this.n = coords.length;
    this.ry = 0.6; this.rx = -0.35; this.zoom = 1;
    this.dragging = false; this.px = 0; this.py = 0;
    this.resize();
    addEventListener('resize', () => this.resize());
    box.addEventListener('pointerdown', e => { this.dragging = true; box.classList.add('drag');
      this.px = e.clientX; this.py = e.clientY; box.setPointerCapture(e.pointerId); });
    box.addEventListener('pointermove', e => {
      if(!this.dragging) return;
      this.ry += (e.clientX-this.px)*0.01; this.rx += (e.clientY-this.py)*0.01;
      this.rx = Math.max(-1.5, Math.min(1.5, this.rx));
      this.px = e.clientX; this.py = e.clientY; });
    const end = () => { this.dragging = false; box.classList.remove('drag'); };
    box.addEventListener('pointerup', end); box.addEventListener('pointercancel', end);
    box.addEventListener('wheel', e => { e.preventDefault();
      this.zoom = Math.max(0.5, Math.min(2.4, this.zoom * (e.deltaY<0?1.08:0.93))); }, {passive:false});
  }
  resize(){
    const dpr = Math.min(devicePixelRatio||1, 2), r = this.box.getBoundingClientRect();
    this.canvas.width = r.width*dpr; this.canvas.height = r.height*dpr;
    this.ctx.setTransform(dpr,0,0,dpr,0,0); this.w = r.width; this.h = r.height;
  }
  project(p){
    const cy=Math.cos(this.ry), sy=Math.sin(this.ry), cx=Math.cos(this.rx), sx=Math.sin(this.rx);
    let x=p[0]*cy - p[2]*sy, z=p[0]*sy + p[2]*cy, y=p[1];
    let y2=y*cx - z*sx, z2=y*sx + z*cx;
    const s = Math.min(this.w,this.h)*0.34*this.zoom;
    return { sx: this.w/2 + x*s, sy: this.h/2 - y2*s, d: z2 };
  }
  frame(){
    if(state.spin && !this.dragging) this.ry += 0.0032;
    const ctx=this.ctx; ctx.clearRect(0,0,this.w,this.h);
    const proj = this.pts.map((p,i)=>({...this.project(p), i}));
    if(state.showPath){
      for(let i=0;i<this.n-1;i++){
        const a=proj[i], b=proj[i+1], dd=(a.d+b.d)/2;
        ctx.strokeStyle = `rgba(150,160,175,${0.16 + 0.30*(dd+1)/2})`;
        ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.sx,a.sy); ctx.lineTo(b.sx,b.sy); ctx.stroke();
      }
    }
    proj.sort((a,b)=>a.d-b.d);
    for(const q of proj){
      const depth=(q.d+1)/2, rad=2.7 + 3.4*depth;
      ctx.globalAlpha = 0.45 + 0.55*depth;
      ctx.fillStyle = viridis(q.i/(this.n-1));
      ctx.beginPath(); ctx.arc(q.sx,q.sy,rad,0,7); ctx.fill();
      ctx.globalAlpha = 0.5+0.5*depth; ctx.lineWidth=0.8;
      ctx.strokeStyle='rgba(255,255,255,0.5)'; ctx.stroke();
    }
    ctx.globalAlpha=1;
  }
}

const scatters = [
  new Scatter(document.getElementById('cv-keys'), DATA.keys.coords),
  new Scatter(document.getElementById('cv-values'), DATA.values.coords),
];
function loop(){ for(const s of scatters) s.frame(); requestAnimationFrame(loop); }
loop();

const bPath=document.getElementById('path'), bSpin=document.getElementById('spin');
bPath.addEventListener('click', ()=>{ state.showPath=!state.showPath; bPath.setAttribute('aria-pressed', state.showPath); });
bSpin.addEventListener('click', ()=>{ state.spin=!state.spin; bSpin.setAttribute('aria-pressed', state.spin); });
bSpin.setAttribute('aria-pressed', state.spin);
document.getElementById('reset').addEventListener('click', ()=>{
  for(const s of scatters){ s.ry=0.6; s.rx=-0.35; s.zoom=1; }});
</script>
"""

html = (TEMPLATE
        .replace("/*DATA*/", json.dumps(data))
        .replace("KEYVAR", str(round(data["keys"]["var3"] * 100)))
        .replace("VALVAR", str(round(data["values"]["var3"] * 100))))
Path(OUT).write_text(html)
print("wrote", OUT, f"({len(html)} bytes)")
