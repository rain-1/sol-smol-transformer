# The SEP "menu": what the present-key set representation is and how it supports a thresholded readout

Model: 2-layer causal pair-sort decoder, `causal_pairs_L30.pt`. Analysis length p=12,
n=2048. All numbers from `agent_riddle_menu_metrics.json` (produced by
`agent_riddle_menu.py`). Figures: `reports/figures/riddle_menu_{composition,presence,threshold,mlp}.png`.

*(This file was written by the coordinating session from the agent's committed
metrics after the agent hit a rerun loop; every number below is from that JSON.)*

## 1. Composition — SEP is ~a sum of the present keys' embeddings (CONFIRMED)

The SEP residual (block-0 output at the SEP position) is well predicted as a linear
function of *which keys are present*:

- regress SEP residual on the present-key multi-hot indicator: **R² = 0.836**
- model SEP as a sum of the present keys' token embeddings: **R² = 0.811**
- as a sum of the L0 OV images of the present keys: R² = 0.754
- mean cosine of each key's unembed row to its OV image: 0.76

So `SEP ≈ Σ_{k present} embed(k)` — a superposed bag of the present keys, not an
ordered or rank-structured object.

## 2. Presence code — decodable per key (CONFIRMED)

"Is key k present?" decodes from the SEP residual at mean **AUC 0.88 / acc 0.885**
(xout), 0.895 (xmid), roughly uniform across the 50 key ids. The set membership is
linearly readable but lossy (a superposition of ~12 keys in 48 dims).

## 3. Thresholded readout — linear-in-principle, lossy from SEP (NEW, nuanced)

A probe given `[SEP menu, θ]` and asked for "the smallest present key > θ":

| readout input | accuracy |
|---|---:|
| SEP menu + θ, **linear** | **0.62** |
| SEP menu + θ, MLP (nonlinear) | 0.65 |
| **clean present-set multi-hot + θ, linear (oracle)** | **1.00** |
| θ only, no menu | 0.27 |

Two conclusions:
- **The readout is fundamentally linear:** given a clean present-set representation a
  purely linear map + θ solves "smallest present key > θ" essentially perfectly
  (0.9997). The nonlinear probe adds almost nothing over the linear one (0.65 vs
  0.62), so no essential nonlinearity is required.
- **SEP is a lossy carrier:** reading the same rule off the *superposed* SEP menu
  drops to 0.62 — the compression of ~12 keys into one 48-dim vector loses enough to
  hurt a from-SEP-alone readout, hardest in the mid-θ range (many candidates). This
  is consistent with the causal finding (`riddle_causal.md`) that the candidate
  identities the model actually uses are **distributed** across the block-0 residual,
  not read solely from SEP — SEP holds a readable-but-lossy copy and is a necessary
  read-hub, not a sufficient standalone menu.

## 4. What the L0 MLP adds at SEP (UNCERTAIN / small)

The SEP representation is similar before vs after the L0 MLP (presence 0.895 xmid vs
0.885 xout; linear thresholded readout 0.656 vs 0.621), so at the SEP position itself
the L0 MLP only mildly sharpens the set/menu code. This does **not** contradict the
L0 MLP being causally essential for keys overall (ablation collapses key accuracy
1.00→0.20, see `walkthrough`): its necessity is for building the representation the
downstream readout consumes, not for making the set decodable at SEP.

## Status

- SEP ≈ sum of present-key embeddings: **CONFIRMED** (R² 0.84 multihot / 0.81 emb-sum).
- Per-key presence decodable ~0.88: **CONFIRMED**.
- "Smallest present key > θ" is a **linear** readout given a clean set (oracle 1.0),
  lossy from SEP alone (0.62): **NEW**.
- Candidate identities are distributed, SEP a lossy read-hub not a swappable menu:
  **consistent with `riddle_causal.md`**.
