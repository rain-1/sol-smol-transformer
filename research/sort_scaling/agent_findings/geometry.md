# Representation geometry of the causal pair-sort decoder

Model: 2-layer, 2-head causal decoder (d_model=48, d_ff=96, vocab=73, seq_len=122),
checkpoint `causal_pairs_L30.pt` (tf_token 0.99994, gen_exact 0.99879).
All numbers reproduced by `research/sort_scaling/agent_geometry.py`
(metrics dumped to `agent_findings/geometry_metrics.json`). Probes are 4/5-fold
cross-validated; the frontier probes are additionally position-controlled.
Figures: `reports/figures/agent_geom_{embedding,positional,frontier,valuecopy}.png`.

Analysis length p=20 (frontier/value-copy/content), p=30 (positional).

---

## 1. Embedding & unembedding geometry

**CONFIRMED — keys and values occupy linearly separable subspaces.**
A logistic classifier separates key tokens (0-49) from value tokens (50-69) with
CV accuracy **1.000** (embedding) and **0.986** (unembedding). In the embedding PCA
scatter the two families split cleanly along PC1 (keys negative, values positive);
SEP and EOS sit apart from both.

**UNCERTAIN / weak — there is a numeric-order direction, but not a clean 1-D
"magnitude line".**
- Best single embedding PC correlates with key numeric id at **r=+0.56** (PC5);
  best single unembedding PC at **r=+0.56** (PC1). This is a genuine descriptive
  correlation over all 50 keys (not a fit).
- Decoding the full 48-dim embedding to the key id with a linear regression gives
  only **CV r=0.30** (embedding) / **0.33** (unembedding). (The in-sample r is
  ~1.0 but that is pure interpolation — 50 points, 48 dims — and is not reported.)
- Interpretation: keys are arranged *loosely* along an order axis, enough to bias
  comparison but far from a projectable magnitude coordinate. This is consistent
  with the established result that the model **sorts by generation, not by computing
  ranks** — the embedding does not hand the model a clean sortable scalar.

## 2. Positional embedding geometry

**CONFIRMED — position is highly structured and carries the "slot" grammar.**
- **Parity (key-slot vs value-slot)** is linearly decodable from the position
  embedding at CV acc **0.992** — position alone tells the model whether it is on a
  key slot or a value slot (the even/odd interleave).
- **Region (input vs output)** decodable at CV acc **0.843**.
- **Rank slot q** at output key positions is decodable at **r=1.000** (and input
  key slot at r=1.000) — position linearly encodes *which rank slot* is being
  emitted. Because sorted keys increase monotonically with q, this makes the raw
  rank-slot a strong *prior* on the key value (see §3).
- **Not a smooth low-frequency curve.** Mean cosine between adjacent position
  embeddings is **-0.267** (negative → alternating, not slowly varying). The
  dominant positional structure is the key/value parity alternation plus a
  region/slot code, rather than a smooth sinusoidal manifold.

## 3. The sort "frontier" (position-controlled probes at key-emit steps)

This is the headline result and required a control. **NEW.**

Raw probes at key-emit positions are near-ceiling for everything (next_key r≈0.97,
last_key r≈0.97, count r≈0.94-1.00) — but this is a **positional confound**: the
sorted key `sk_q` is monotone in the rank slot q (batch mean 1.4→47.5 across q),
and position encodes q, so an uncontrolled probe just reads the rank-slot prior.

After **partialling out q** (centering features and labels within each rank slot),
the genuine content signal is:

| stage  | next key `sk_q` (content) | last key `sk_{q-1}` (content) |
|--------|---------------------------|-------------------------------|
| embed  | -0.02                     | -0.03                         |
| block0 | 0.21                      | 0.27                          |
| block1 | **0.69**                  | **0.65**                      |
| final  | 0.69                      | 0.65                          |

- **A content-based frontier IS represented, and it is built by block1** (r≈0.67),
  not present at the input (r≈0). Both the just-emitted key and the key about to be
  emitted are linearly present at the key-emit position by block1 — exactly where
  the L1 heads operate. This is a genuine "current threshold / frontier" signal
  that survives the position control, in contrast to the established negative result
  that *global* key-rank is not decodable (~0.35 everywhere).
- "Count emitted so far" (q) is trivially and fully positional (r≈1.0 raw); it is a
  read-off of the position embedding, not a computed counter.

Takeaway: the model does not represent global ranks, but it *does* carry a local,
content-bearing frontier (last key emitted + next key) that emerges at block1.

## 4. Value-copy geometry (induction OV path)

**CONFIRMED — the value carry is a copy in representation space, written at block1.**
At value-emit positions (logit lens = final_norm then unembed on the block residual):
- block0 residual → value accuracy **0.05** (chance ≈ 1/20); cosine to correct
  value's unembed row **0.07**.
- block1 residual → value accuracy **1.00** (full vocab and value-vocab both);
  cosine to correct value's unembed row **0.773**, vs **0.179** to a random value.

So the induction OV output at block1 points the residual directly along the correct
value token's unembedding direction — a clean copy, and it is absent one block
earlier. Confirms L1H0 as the value-copy induction head in geometric terms.

## 5. Content vs position per head

Measured as the std over the batch of the argmax-attended source index at fixed
query positions (position held constant → variation = content driven):

| head | argmax-src std, value step | argmax-src std, key step |
|------|----------------------------|--------------------------|
| L0H0 | 7.58 | 0.58 |
| L0H1 | 7.07 | 0.65 |
| L1H0 | 6.90 | 0.86 |
| L1H1 | 7.65 | 0.26 |

**CONFIRMED — value-emit attention is content-addressed; key-emit attention is
position-stable.** At value steps every head's attention target moves a lot with
content (std ≈ 7 input positions) — the induction/precursor hunt for the matching
key is content-driven. At key steps the attended source barely moves (std < 1),
i.e. attention lands at a near-fixed relative position (the recently generated
output). Note this does not mean key *selection* is positional: the sort head L1H1
(lowest key-step std, 0.26) attends to a fixed relative output slot but the *token
content* it reads there carries the identity — content enters via what is read, not
where. This matches the established picture: L1H1 reads generated output to pick the
next key.

---

## Summary of status

- Key/value linear separability: **CONFIRMED** (CV acc 1.00 / 0.99).
- Clean 1-D key-magnitude axis for comparison-by-projection: **REFUTED** as strong
  form; only a weak order direction exists (best PC r=0.56, full CV decode r=0.30).
- Position encodes slot parity / region / rank slot: **CONFIRMED** (0.99 / 0.84 /
  1.00); positional manifold is alternating, **not** a smooth curve (adj cos -0.27).
- Global rank not represented but a **local content frontier IS** (next+last key,
  content r≈0.67 at block1, ≈0 at embed): **NEW**, position-controlled.
- Induction OV = geometric copy at block1 (cos 0.77, lens acc 1.00 vs 0.05 at
  block0): **CONFIRMED**.
- Value-step attention content-driven, key-step attention position-stable:
  **CONFIRMED**.
