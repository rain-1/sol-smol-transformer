# The threshold + readout half of the sort riddle

Model: 2-layer, 2-head causal decoder (d_model=48, d_ff=96), checkpoint
`causal_pairs_L30.pt` (gen_exact 0.99879). All numbers reproduced by
`research/sort_scaling/agent_riddle_threshold.py` at analysis length **p=12**
(VRAM discipline); metrics dumped to `agent_findings/riddle_threshold_metrics.json`.
Figures: `reports/figures/riddle_threshold_{direction,sweep,readout}.png`.

Established upstream (`sorthead.md`, `geometry.md`): next key = **smallest input key
strictly greater than the running max of the emitted keys**; L1H1 reads the emitted
keys to recover that threshold; SEP carries a static per-sequence menu of present
keys; the next key is decodable ~1.0 from `caps[1]['x_mid']` (L1 attn output, before
the L1 MLP). This report reverse-engineers **how theta is encoded** and **how it
gates the readout**.

Notation: at key step q the query sits at seq index `sep+2q`; the running-max
threshold is `theta = sk_{q-1}` (emitted keys are ascending, so their max is the last
one). Analysis covers q=1..p-1 (a threshold exists only after the first emit).

---

## One-paragraph answer

The threshold is carried into the query position by **L1H1's OV output** and lands in
`x_mid`. It is **linearly decodable** there (content R²≈0.54 after controlling for
position) but it is **not a low-dimensional additive axis** — it is a distributed,
multi-dimensional code (~5-8 effective dims). The readout is a **theta-gated step
function over the present-key menu**: for present keys `logit(k) ≈ 16.7·[k>theta] −
0.66·(k−theta)₊ − 0.20·(theta−k)₊`. The `[k>theta]` term is a ~8-logit **cliff** at
theta (keys ≤ theta are killed) and the mild `−0.66/key` decay above theta makes the
**nearest present key above theta win**. The gate is **~97% linear at `x_mid`**: the
model's own unembedding read off `x_mid` (before the L1 MLP) already selects the right
key 97.3% of the time; the L1 MLP only sharpens the last ~2.7%. The **decisive test**:
holding one static menu fixed and sweeping the injected threshold reproduces a clean
staircase — the same menu emits *every present key in order* purely as a function of
theta.

---

## 1. How theta is encoded at `x_mid` — CONFIRMED (L1H1, distributed, not 1-D)

Linear probe (Ridge, 5-fold CV) of `theta` from `x_mid` at key-emit positions.

**Raw decode is confounded by position.** `theta` decodes at R²=**0.913**
(RMSE **3.97** key-ids) from full `x_mid` — but `theta = sk_{q-1}` is monotone in the
rank-slot q, and q is fully positional, so this partly reads the position embedding
(the confound flagged in `geometry.md`).

**Position-controlled (partial out q) = the genuine content threshold:**

| x_mid variant | content-theta decode R² (q-controlled) |
|---|---|
| full `x_mid` | **0.535** |
| `x_mid` − L1H1 OV | **0.317**  ← biggest drop (−0.22) |
| `x_mid` − L1H0 OV (control) | 0.572  (unchanged) |
| L1H1 OV contribution **alone** | **0.582** |

So **L1H1 supplies the threshold content, roughly additively**: ablating only its OV
output halves the content signal, ablating the value-head L1H0 does nothing, and the
L1H1 contribution vector *by itself* reconstructs theta as well as the full residual.

**It is NOT low-dimensional.** Decoding content-theta from the top-k PCA dims of
`x_mid`: k=1 → R²0.01, k=3 → 0.02, k=10 → 0.08, all 48 → 0.55. The threshold lives in
**many low-variance directions**, not a clean 1-2-dim magnitude line — consistent with
`geometry.md`'s finding that the model has no projectable key-magnitude scalar.

## 2. THE DECISIVE EXPERIMENT — synthetic threshold sweep → staircase — CONFIRMED

Fix one menu (present keys `[6,9,12,13,23,28,31,36,39,45,47,48]`). At q=1 the threshold
*is* the single emitted-key token, so overwriting that token with theta=0..49 is a
maximally in-distribution threshold injection. Read the argmax next key.

- **Selected key vs theta is a clean staircase = "smallest present key > theta",
  match 1.000** on this menu; **0.995** averaged over 40 random menus. The same static
  menu emits every present key in order as theta sweeps (left panel of
  `riddle_threshold_sweep.png`).
- With the **L1 MLP ablated** the staircase still holds at **0.979** → the selection
  is essentially linear at the L1 attention output (see §4).

**Is the causal threshold code in `x_mid` a single additive axis? No.** (right panel)
- Transplanting the *real captured* `x_mid` at the query reproduces the token-patch
  selection **exactly (1.000)** → the readout is genuinely a function of `x_mid[qpos]`.
- A **naive 1-D injection** along the linear theta-probe direction **fails (0.0625)** —
  moving `x_mid` along a single decoded-theta axis does not move the selection.
- Reconstructing the real threshold trajectory from its top-k PCs and transplanting:
  k=1 → 0.31, k=3 → 0.71, k=5 → 0.90, k=8 → **1.0**. The causal threshold code the
  readout consumes is **~5-8 dimensional** (trajectory variance fractions
  0.23/0.14/0.13/0.12/0.08/0.06 — no dominant axis).

**Decodability ≠ steerability**: theta is linearly *readable* from `x_mid`, but it is
not a single additive coordinate you can push; the readout compares a multi-dimensional
threshold representation against the menu.

## 3. Readout functional form — CONFIRMED (theta-gated step over the menu)

At key steps, the model's key-logits split sharply by the menu and the threshold
(`riddle_threshold_readout.png`, left):

- Mean logit of **present keys above theta = 7.35** vs **present keys ≤ theta = −0.69**
  — an ~8-logit **cliff** exactly at k=theta. Winner beats its best present competitor
  by a **15.4-logit margin** (extremely decisive).
- Argmax over present keys = "smallest present > theta" at **0.9998** (true logits) →
  reconfirms the rule.

Fitting the proposed form `logit(k) ~ a·present(k) − b·penalty(k,theta)` on present
keys gives `logit(k) = 16.74·[k>theta] − 0.66·(k−theta)₊ − 0.20·(theta−k)₊`:
- `[k>theta]` coef **+16.7** = the cliff (keys ≤ theta are removed from contention);
- `(k−theta)₊` coef **−0.66/key** = mild decay above theta ⇒ **nearest above wins**.
- The **form's argmax matches the true selection at 0.9998 and the ideal rule at 1.0.**
- Present-only fit **R²=0.37** (all-keys R²=0.10 is polluted by dead absent keys): the
  simple linear form captures the *decision* (cliff + nearest-above) but not the fine
  logit magnitudes, which carry extra per-key structure.

So the readout is `present(k)` AND a hard step at theta with a gentle above-theta
slope — argmax = smallest present key just above theta, as hypothesised.

## 4. Linear or MLP? — CONFIRMED mostly linear at `x_mid`; L0 MLP does the set work

Model's own unembedding as a logit lens (next-key accuracy at key steps):

| readout point | next-key accuracy |
|---|---|
| `x_mid` (L1 attn out, **before** L1 MLP) | **0.973** |
| `x_out` (**after** L1 MLP) | 0.9998 |

- **~97% of the selection is already linearly present at `x_mid`**; the L1 MLP adds the
  final ~2.7% (a cleanup, not the mechanism). Ablating the L1 MLP entirely drops key
  accuracy only 0.9998 → 0.973.
- **Ablating the L0 MLP collapses key accuracy to 0.198** — the L0 MLP is essential
  (it builds the SEP menu / set aggregation upstream), whereas the threshold gate
  itself is a near-linear read at the L1 attention output.

---

## Status tags

- **CONFIRMED** — next key = smallest present key > running-max threshold; decisive
  staircase from a static menu (token patch: 1.0 one menu, 0.995 over 40 menus).
- **CONFIRMED** — theta is carried by L1H1's OV into `x_mid` (position-controlled decode
  0.535 full → 0.317 without L1H1; L1H1 contribution alone 0.582; L0H1-value head is
  irrelevant), roughly additively.
- **CONFIRMED** — readout = theta-gated step over the menu: cliff +16.7, above-theta
  slope −0.66/key; argmax reproduces the rule at 0.9998–1.0; ~8-logit cliff, 15.4-logit
  winner margin.
- **CONFIRMED** — gating is ~linear at `x_mid` (lens 0.973 before the L1 MLP; staircase
  survives L1-MLP ablation at 0.979). L1 MLP = minor cleanup; L0 MLP = essential set work.
- **NEW / partial REFUTE** — the threshold is **not** a low-dimensional additive axis.
  It is linearly decodable (content R²0.54) yet distributed across many low-variance
  dims; a 1-D injection fails (0.06) while the true causal code needs ~5-8 dims
  (top-k PC reconstruction 1→0.31, 5→0.90, 8→1.0). Decodability ≠ steerability.

## Controls / uncertainty / OOD caveats

- **Position control**: theta is monotone in the positional rank-slot q; every "content"
  number partials out q (Ridge R²0.54 here vs `geometry.md`'s r≈0.67 — same regime).
- **In-distribution knob**: the q=1 token patch is maximally in-distribution (threshold
  = the one emitted token, no earlier emits to contradict it). The 0.995/40-menu number
  is the headline.
- **On- vs off-manifold**: the *transplant* (1.0) and *top-k PC reconstruction* curves
  use REAL captured `x_mid` activations (on the data manifold) → trustworthy. The failed
  1-D probe-axis injection (0.06) is an off-manifold linear push; its failure is a
  geometric statement (theta is not one additive axis), not a numerical artifact.
- **Probes**: Ridge (alpha=10), 5-fold CV, standardized features; d_model=48 so the
  full-dim decode ceiling is modest. Analysis at p=12; the L30 checkpoint generalises
  across lengths (upstream), but all figures here are p=12.
- Present-only readout R²=0.37 means the *linear* form explains the decision, not the
  exact logit values; the fine structure (extra per-key variance) is unmodelled.

## Files
- Code: `research/sort_scaling/agent_riddle_threshold.py`
- Metrics: `research/sort_scaling/agent_findings/riddle_threshold_metrics.json`
- Figures: `reports/figures/riddle_threshold_direction.png` (theta encoding + dims),
  `riddle_threshold_sweep.png` (**decisive staircase** + causal dimensionality),
  `riddle_threshold_readout.png` (cliff + theta-gated form fit).
