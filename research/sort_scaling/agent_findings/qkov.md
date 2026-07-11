# Weight-level QK/OV analysis of the value-carrying induction circuit L0H0 -> L1H0

Checkpoint: `research/sort_scaling/checkpoints/causal_pairs_L30.pt`
Code: `research/sort_scaling/agent_qkov.py` (run to regenerate every number below).
Figures: `reports/figures/agent_qkov_{ov_copy,qk_identity,qk_composed}.png`.

## Weight extraction
`in_proj_weight` is `[3d,d] = concat(Wq,Wk,Wv)` along rows; head h uses rows
`h*24:(h+1)*24`. `out_proj.weight` is `[d,d]`; head h uses columns `h*24:(h+1)*24`.
Verified shapes L1H0: Wq/Wk/Wv = [24,48], Wo = [48,24]. d_model=48, per-head=24.

## LayerNorm approximation (stated explicitly)
Attention input is `ln1(resid)`; the unembed reads `final_norm(resid)`.
LayerNorm(x) = gamma*(x-mean(x))/std(x) + beta. I **keep** the mean-centering
(exact linear projection) and the diagonal gain `gamma`, and **drop** (a) the
per-token `1/std(x)` rescale (data-dependent, not a fixed weight — it only scales
attention temperature / logit magnitude, not the token-to-token argmax structure)
and (b) the additive `beta` (a constant shift, no token-selective structure).
So every embedding row e is folded as `e -> gamma * (e - mean(e))` before the QKV/OV
map. This is the standard "fold LN gain, ignore LN scale" approximation; it is exact
for argmax/ranking of a linear head up to a positive per-token scalar.

## (2) OV circuit of L1H0 — CONFIRMED: it is a clean COPY matrix
Effective map `W_U @ final_norm @ Wo1 @ Wv1 @ ln1 @ W_E` restricted to value tokens
(ids 50-69), giving a 20x20 "attend to value X -> logit of value Y" matrix.

- argmax-correct fraction = **1.000** (20/20; chance 0.050)
- mean diagonal = **52.18**, mean off-diagonal = **-2.15**
- mean diagonal z-score within row = **3.54**
- Control: the same OV restricted to KEY tokens (0-49) is NOT a key->key copy:
  argmax-correct = **0.020** (= chance). So the OV copy is specific to the value
  subspace it must transport.

Interpretation: whatever input value token L1H0 attends to, it writes that exact
value token to the output logits. This is the "value carry" half of induction,
now shown at the weight level (not just from ablation). Fig: `agent_qkov_ov_copy.png`.

## (3) QK circuit of L1H0 — the match is NOT a raw-embedding identity match
Direct token-token QK over key tokens (`ln1(E_q)@Wq1.T` dotted with `ln1(E_k)@Wk1.T`):

- argmax-on-identity = **0.000**, diagonal z = **-1.15** (REFUTED as a *direct* match)

This is the important nuance: a naive induction head would match query-key identity
directly through the embeddings. Here it does not, because L1H0's key-side input is
not the raw key embedding — the key identity is delivered to the value positions by
the L0H0 precursor. Positional contribution is small: content QK std / positional QK
std = **5.6** (content dominates), so the head is content-driven, just not via the
raw key embedding. Fig: `agent_qkov_qk_identity.png`.

## (3b) COMPOSED QK via L0H0 — CONFIRMED: induction match is L0H0-mediated (NEW)
Build the key-side vector as it actually arrives at an input VALUE position: that
position (via L0H0, which attends to its preceding key) carries `Wo0@Wv0@ln0(E_x')`.
Feeding that through L1H0's key map, `Kcomp(x') = Wk1 @ diag(g_ln1) @ Wo0 @ Wv0 @ ln0(E_x')`,
and dotting with the query on generated key token x:

- argmax-on-identity = **1.000** (50/50; chance 0.020)
- mean diagonal = **8.15**, off-diagonal = **-0.05**, diagonal z = **3.99**

So once the L0H0 OV path is inserted on the key side, L1H0's QK becomes a perfect
key-identity matcher. Direct path = 0%, composed path = 100%. This is the weight-level
proof that the induction match runs *through* L0H0, matching the ablation evidence
(ablating L0H0 collapses L1H0 induction 0.99->0.10). Fig: `agent_qkov_qk_composed.png`.

## (4) K-composition L0H0 -> L1H0 — CONFIRMED and head-specific
Composition operator `Wk1 @ diag(g_ln1) @ (Wo0 @ Wv0)`, scored as a normalized
alignment `||Wk1 g Wo0 Wv0|| / (||Wk1 g|| * ||Wo0 Wv0||)`:

- L0H0 (precursor):        normalized score = **0.207**, z vs random = **+12.7**
- L0H1 (control head):     normalized score = **0.149**, z vs random = **+1.0**
- Random OV baseline (200): **0.144 +/- 0.005**
- Raw Frobenius for reference: L0H0 = 7.89, L0H1 = 5.28

L0H0's OV writes into a subspace strongly read by L1H0's K map, ~13 sigma above
random, while the sibling head L0H1 is at baseline. The composition is specific to
the established precursor head.

## Summary of verdicts
- OV of L1H0 = value COPY matrix: **CONFIRMED** (100% argmax, diag z 3.5).
- L1H0 as a *direct* embedding-identity induction head: **REFUTED** (0% argmax).
- L1H0 induction match is realized by **K-composition through L0H0**: **CONFIRMED /
  NEW** (composed QK 100% argmax, diag z 4.0; composition 12.7 sigma, head-specific).
- Positional QK contribution is minor vs content (ratio 5.6): **CONFIRMED**.

All numbers reproducible via `python research/sort_scaling/agent_qkov.py`.
Caveats: LN scale (1/std) and beta dropped (see approximation note); scores are
weight-space (do not include MLP writes into the same subspaces, which could add to
the K-composition path but are not needed to demonstrate it).
