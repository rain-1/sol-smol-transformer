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
- mean cosine of each key's fitted per-key code `u_k` to its L0 OV image: 0.76

So `SEP ≈ Σ_{k present} embed(k)` — a superposed bag of the present keys, not an
ordered or rank-structured object.

## 2. Presence code — decodable per key (CONFIRMED)

"Is key k present?" decodes from the SEP residual at mean **AUC 0.88 / acc 0.885**
(xout), 0.895 (xmid). It is **non-uniform**: keys 0–19 decode at AUC ~0.965 but
mid-range keys 20–29 fall to ~0.75 (most collided in the additive superposition of
~12 keys in 48 dims). The thresholded readout in §3 collapses in exactly that
mid-range, tracking the presence-code degradation.

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

## 4. The L0 MLP does NOT build the menu (REFUTED) — yet is causally essential (open)

Before vs after the L0 MLP at SEP, every menu metric is flat-to-slightly-**worse**:
composition R² 0.836→0.835, presence AUC 0.895→0.880, thresholded readout 0.618→0.597.
So the present-key set is assembled by the L0 **attention** (already present in
`x_mid`); the L0 MLP does not refine or build it.

This creates a genuine puzzle: ablating the L0 MLP collapses key accuracy 1.00→0.20
(`walkthrough`), so it is causally essential for key selection — but not by improving
the linearly-decodable set code at SEP. Either its output is a nonlinear transform
consumed by the downstream readout that a linear probe cannot see as added
set-information, or it acts at non-SEP positions. **Its precise key-role is not
localized** by this analysis — a residual open question. (Earlier framing that "the
L0 MLP prepares the menu" is withdrawn.)

## Status

- SEP ≈ sum of present-key embeddings: **CONFIRMED** (R² 0.84 multihot / 0.81 emb-sum).
- Per-key presence decodable ~0.88: **CONFIRMED**.
- "Smallest present key > θ" is a **linear** readout given a clean set (oracle 1.0),
  lossy from SEP alone (0.62): **NEW**.
- Candidate identities are distributed, SEP a lossy read-hub not a swappable menu:
  **consistent with `riddle_causal.md`**.
