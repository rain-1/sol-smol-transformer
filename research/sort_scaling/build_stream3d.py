"""Build an interactive 3D viewer of the residual (activation) stream.

Points are token states, coloured by role (input key/value, output key/value, SEP),
projected into ONE shared PCA-3D frame across all four stages (embed, block 0,
block 1, final) so a stage toggle shows the SAME points reorganising as the
computation proceeds. Self-contained HTML + published as an Artifact.

Usage: python research/sort_scaling/build_stream3d.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import causal_sort as cs  # noqa: E402
from deep_interp import load, run_capture, batch  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "stream_3d.html")
STAGES = ["embed", "block0", "block1", "final"]
STAGE_LABELS = ["embedding", "block 0", "block 1", "final"]
N_POINTS = 1600
P = 8


def roles(seq_np, p):
    """Role index per position: 0 in-key,1 in-val,2 SEP,3 out-key,4 out-val,5 other."""
    L = seq_np.shape[1]
    pos = np.arange(L)[None, :].repeat(seq_np.shape[0], 0)
    tok = seq_np
    sep = 2 * p
    r = np.full(seq_np.shape, 5, dtype=int)
    ink = tok < cs.KEY_ALPH
    inv = (tok >= cs.KEY_ALPH) & (tok < cs.KEY_ALPH + cs.VAL_ALPH)
    r[ink & (pos < sep)] = 0
    r[inv & (pos < sep)] = 1
    r[tok == cs.SEP] = 2
    r[ink & (pos > sep)] = 3
    r[inv & (pos > sep)] = 4
    return r


def main():
    torch.manual_seed(0)
    model, _ = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
    dev = next(model.parameters()).device
    seq, _, _ = batch(model, P, n=80)
    with torch.inference_mode():
        _, _, resid = run_capture(model, seq[:, :-1])
    seq_np = seq[:, :-1].cpu().numpy()
    role = roles(seq_np, P).reshape(-1)
    # stack stages -> [S, N, 48]
    S = np.stack([resid[s].reshape(-1, resid[s].shape[-1]).cpu().numpy() for s in STAGES])
    N = S.shape[1]
    # subsample
    rng = np.random.RandomState(0)
    idx = rng.choice(N, size=min(N_POINTS, N), replace=False)
    S = S[:, idx]; role = role[idx]
    # center each stage, fit ONE shared PCA on the pooled centered data
    Sc = S - S.mean(1, keepdims=True)
    pooled = Sc.reshape(-1, Sc.shape[-1])
    Vt = np.linalg.svd(pooled - pooled.mean(0), full_matrices=False)[2][:3]
    coords = Sc @ Vt.T                                   # [S, N, 3]
    coords = coords / np.abs(coords).max()               # global scale
    # per-point [stage][xyz]
    pts = np.transpose(coords, (1, 0, 2)).round(3).tolist()   # [N, S, 3]
    data = {"stages": STAGE_LABELS, "points": pts, "roles": role.tolist()}
    print("points:", len(pts), "role counts:", {i: int((role == i).sum()) for i in range(6)})
    html = TEMPLATE.replace("/*DATA*/", json.dumps(data))
    Path(OUT).write_text(html)
    print("wrote", OUT, f"({len(html)} bytes)")


TEMPLATE = r"""<title>The activation stream in 3D: structure emerging with depth</title>
<style>
  :root {
    --bg:#f6f7f9; --panel:#fff; --ink:#10151b; --muted:#5b6774; --line:#e4e8ee;
    --accent:#2f6df0; --grid:#eef1f5; --canvas-bg:#0f1720;
  }
  @media (prefers-color-scheme: dark){ :root{
    --bg:#0b0f14; --panel:#121820; --ink:#e9edf2; --muted:#8b98a6; --line:#212a34;
    --accent:#6ea0ff; --grid:#1a2028; --canvas-bg:#0a1017; } }
  :root[data-theme="light"]{ --bg:#f6f7f9; --panel:#fff; --ink:#10151b; --muted:#5b6774;
    --line:#e4e8ee; --accent:#2f6df0; --grid:#eef1f5; --canvas-bg:#0f1720; }
  :root[data-theme="dark"]{ --bg:#0b0f14; --panel:#121820; --ink:#e9edf2; --muted:#8b98a6;
    --line:#212a34; --accent:#6ea0ff; --grid:#1a2028; --canvas-bg:#0a1017; }
  *{box-sizing:border-box} body{margin:0}
  .wrap{max-width:940px; margin:0 auto; padding:40px 24px 56px; color:var(--ink);
    background:var(--bg); font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;}
  .eyebrow{font-family:ui-monospace,Menlo,monospace; font-size:11px; letter-spacing:.18em;
    text-transform:uppercase; color:var(--accent); margin:0 0 14px;}
  h1{font-size:clamp(25px,4vw,38px); line-height:1.1; margin:0 0 14px; font-weight:640;
    letter-spacing:-0.02em; text-wrap:balance; max-width:22ch;}
  .lede{margin:0; max-width:66ch; color:var(--muted); font-size:16px; line-height:1.6;}
  .lede b{color:var(--ink); font-weight:600;}
  .stages{display:flex; gap:8px; flex-wrap:wrap; margin:28px 0 14px;}
  .stages button{font:inherit; font-size:13px; color:var(--muted); background:var(--panel);
    border:1px solid var(--line); border-radius:8px; padding:8px 14px; cursor:pointer;
    font-variant-numeric:tabular-nums; transition:.15s;}
  .stages button:hover{border-color:var(--accent);}
  .stages button[aria-pressed="true"]{color:var(--ink); border-color:var(--accent);
    background:color-mix(in srgb,var(--accent) 12%,var(--panel));}
  .stages button:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
  .stagenum{font-family:ui-monospace,Menlo,monospace; font-size:11px; opacity:.6; margin-right:6px;}
  .stage .cv{position:relative; width:100%; aspect-ratio:16/11; border-radius:14px;
    overflow:hidden; background:var(--canvas-bg); cursor:grab; border:1px solid var(--line);}
  .cv.drag{cursor:grabbing} canvas{display:block; width:100%; height:100%; touch-action:none;}
  .barrow{display:flex; justify-content:space-between; align-items:center; gap:14px;
    margin-top:12px; flex-wrap:wrap;}
  .legend{display:flex; gap:14px; flex-wrap:wrap;}
  .legend .it{display:inline-flex; align-items:center; gap:7px; font-size:12.5px; color:var(--muted);}
  .legend .sw{width:11px; height:11px; border-radius:50%;}
  .hint{font-size:12px; color:var(--muted);}
  .caption{margin-top:14px; font-family:ui-monospace,Menlo,monospace; font-size:12px;
    color:var(--muted); line-height:1.6;}
  .caption b{color:var(--ink);}
  footer{margin-top:26px; color:var(--muted); font-size:13px; line-height:1.6; max-width:72ch;}
  footer code{font-family:ui-monospace,Menlo,monospace; font-size:12px; background:var(--grid);
    padding:1px 5px; border-radius:4px;}
</style>

<div class="wrap">
  <p class="eyebrow">Sort decoder · residual stream</p>
  <h1>Watch the activation stream organize itself</h1>
  <p class="lede">
    Each dot is one token's 48-dim residual vector, coloured by its <b>role</b>. All
    four stages share one PCA-3D frame, so stepping <b>embedding → block 0 → block 1 →
    final</b> shows the same points <b>moving</b> as the computation runs. At the
    embedding, keys and values already split by identity but input and output states
    overlap; by block 0 the five roles pull apart into distinct clusters. (Contrast
    the raw embedding <em>table</em>, which stays a blob.) Drag to rotate.
  </p>

  <div class="stages" id="stages"></div>
  <div class="stage"><div class="cv" id="cv"><canvas></canvas></div></div>

  <div class="barrow">
    <div class="legend" id="legend"></div>
    <span class="hint">drag to rotate · scroll to zoom</span>
  </div>
  <div class="caption" id="cap"></div>

  <footer>
    States from the length-30 causal pair-sort decoder
    (<code>causal_pairs_L30.pt</code>), 1600 token positions, one shared 3-PC frame
    fitted across all stages. See <code>reports/sort-residual-stream.md</code>.
  </footer>
</div>

<script>
const DATA = /*DATA*/;
const ROLES = [
  {name:"input key",   c:"#2a78d6"},
  {name:"input value", c:"#eb6834"},
  {name:"SEP",         c:"#e34948"},
  {name:"output key",  c:"#6a5cd8"},
  {name:"output value",c:"#1a9e4b"},
  {name:"other",       c:"#94a0ad"},
];
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const legend = document.getElementById('legend');
ROLES.slice(0,5).forEach(r=>{ const s=document.createElement('span'); s.className='it';
  s.innerHTML=`<span class="sw" style="background:${r.c}"></span>${r.name}`; legend.appendChild(s); });

const stagesBar = document.getElementById('stages');
DATA.stages.forEach((s,i)=>{ const b=document.createElement('button');
  b.innerHTML=`<span class="stagenum">${i}</span>${s}`; b.setAttribute('aria-pressed', i===0);
  b.addEventListener('click',()=>go(i)); stagesBar.appendChild(b); });

const box=document.getElementById('cv'), canvas=box.querySelector('canvas'), ctx=canvas.getContext('2d');
const N=DATA.points.length;
let W,H; function resize(){ const dpr=Math.min(devicePixelRatio||1,2), r=box.getBoundingClientRect();
  canvas.width=r.width*dpr; canvas.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); W=r.width; H=r.height; }
resize(); addEventListener('resize', resize);

let ry=0.6, rx=-0.35, zoom=1, dragging=false, px=0, py=0, spin=!reduceMotion;
box.addEventListener('pointerdown',e=>{dragging=true; box.classList.add('drag'); px=e.clientX; py=e.clientY; box.setPointerCapture(e.pointerId);});
box.addEventListener('pointermove',e=>{ if(!dragging) return; ry+=(e.clientX-px)*0.01; rx+=(e.clientY-py)*0.01;
  rx=Math.max(-1.5,Math.min(1.5,rx)); px=e.clientX; py=e.clientY; });
const end=()=>{dragging=false; box.classList.remove('drag');};
box.addEventListener('pointerup',end); box.addEventListener('pointercancel',end);
box.addEventListener('wheel',e=>{e.preventDefault(); zoom=Math.max(0.5,Math.min(2.4,zoom*(e.deltaY<0?1.08:0.93)));},{passive:false});

let cur=0, from=0, t=1, tStart=0;
const cap=document.getElementById('cap');
function setCap(){ cap.innerHTML = `stage <b>${DATA.stages[cur]}</b> — the same points, in one shared frame`; }
function go(i){ if(i===cur) return; from=cur; cur=i; t=0; tStart=performance.now();
  [...stagesBar.children].forEach((b,j)=>b.setAttribute('aria-pressed', j===i)); setCap(); }
setCap();

const ease=x=>x<.5?2*x*x:1-Math.pow(-2*x+2,2)/2;
function project(p){
  const cy=Math.cos(ry),sy=Math.sin(ry),cx=Math.cos(rx),sx=Math.sin(rx);
  let x=p[0]*cy-p[2]*sy, z=p[0]*sy+p[2]*cy, y=p[1];
  let y2=y*cx-z*sx, z2=y*sx+z*cx;
  const s=Math.min(W,H)*0.42*zoom;
  return {sx:W/2+x*s, sy:H/2-y2*s, d:z2};
}
function frame(now){
  if(t<1){ t=Math.min(1,(now-tStart)/800); }
  if(spin && !dragging) ry+=0.003;
  ctx.clearRect(0,0,W,H);
  const tt=ease(t);
  const proj=new Array(N);
  for(let i=0;i<N;i++){
    const a=DATA.points[i][from], b=DATA.points[i][cur];
    const p=[a[0]+(b[0]-a[0])*tt, a[1]+(b[1]-a[1])*tt, a[2]+(b[2]-a[2])*tt];
    const pr=project(p); pr.r=DATA.roles[i]; proj[i]=pr;
  }
  proj.sort((a,b)=>a.d-b.d);
  for(const q of proj){
    const depth=(q.d+1)/2;
    ctx.globalAlpha=0.35+0.55*depth;
    ctx.fillStyle=ROLES[q.r].c;
    ctx.beginPath(); ctx.arc(q.sx,q.sy,1.7+2.6*depth,0,7); ctx.fill();
  }
  ctx.globalAlpha=1;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
"""

if __name__ == "__main__":
    main()
