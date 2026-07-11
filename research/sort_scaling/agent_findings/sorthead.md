# The sort mechanism: how the decoder chooses the next key

Model: 2-layer, 2-head causal decoder (d_model=48), checkpoint `causal_pairs_L30.pt`,
99.9% gen-exact. All numbers below produced by `research/sort_scaling/agent_sorthead.py`
(length p=16 for attention/probes, p=12 for the causal patch). Key query for sorted
key `sk_q` sits at seq index `sep+2q`; previously emitted sorted keys sit at `sep+1+2j`
(j=0..q-1), with `sk_{q-1}` the most recent.

## One-sentence answer
The model emits, at each key step, **the smallest input key strictly greater than the
running maximum of the keys already emitted**. It computes this by having sort head
**L1H1** read the recently-emitted keys (recency-weighted, dominated by the last one)
to recover the threshold, while layer-0 heads aggregate the set of input keys into the
SEP position which is routed forward for the comparison. The chosen next key is fully
linearly decodable from the residual by the end of block 1.

---

## 1. Where the key-emitting query attends inside the output — CONFIRMED (recency, not broad)
L1H1 attention mass at key steps (q>=1), split by class: **output region 0.94**, SEP 0.056,
input 0.004. Within the output it is **sharply recency-weighted** onto the emitted keys
(mean over q, mass by recency r, r=0 is `sk_{q-1}`):

| recency r | 0 | 1 | 2 | 3 | 4 | >=5 |
|---|---|---|---|---|---|---|
| L1H1 mass | **0.605** | 0.228 | 0.090 | 0.036 | 0.014 | 0.009 |

So the hypothesis "attend to the most-recently emitted key `sk_{q-1}`" is confirmed as the
dominant mode (0.61), but it is **not exclusively** the last one — a geometric tail reaches
back over several recent keys (0.83 in the two most recent). The other heads carry ~zero
output-key mass at key steps (L0H0 0.0, L0H1 0.06 diffuse, L1H0 0.007). L1H1 is the sole
sort head.

## 2. The next-key rule — CONFIRMED as "min input key > running max of emitted keys"
- Logit-lens / final logits at key-query positions predict the true next key with
  **accuracy 1.000** (p=16).
- **Causal patch** (teacher-force the true prefix to step q=6, overwrite the last emitted
  key `sk_{q-1}` token with each input key, read the argmax next key):
  - Monotone-preserving substitutions (sub >= all earlier emitted keys):
    `pred = min input key > substituted key` in **99.8%** (n=1400).
  - Non-monotone substitutions (sub < some earlier emitted key, i.e. OOD): the "min > last"
    rule matches only **0.4%**, but `pred = min input key > max(all emitted keys)` matches
    **99.6%** (n=800).

  The clean dissociation shows the threshold is the **running max** of emitted keys, which
  under normal monotone generation equals the last emitted key — consistent with the
  recency-weighted (soft-max-like) read in analysis 1. The "min greater than previous" rule
  is therefore CONFIRMED as the special case, with the true implemented rule being
  "min greater than max-so-far".

## 3. How input-key information reaches the key position via layer 0 — CONFIRMED (routed through SEP)
- At key steps L0 heads "park on SEP" (L0H0 0.84, L0H1 0.89 mass onto SEP), verifying the
  prior observation. But ablating them (zero their OV) still **halves key accuracy**:
  L0H0 key-acc 1.00->0.50 (drop 0.50), L0H1 1.00->0.63 (drop 0.37). So L0 is causally
  required for key selection despite not reading the input at the key position itself.
- Resolution: **at the SEP position** L0 reads the input keys directly — L0H0 puts **0.996**
  of its SEP-position attention on input-key tokens, L0H1 0.83. L0 thus aggregates the set
  of present input keys into the SEP residual. The SEP block-0 residual linearly predicts
  per-key presence at **0.81** vs 0.68 majority-baseline (lossy but real; 16 keys into 48
  dims). Higher layers read this SEP summary forward (L1 attends heavily to SEP), giving the
  sort head the candidate set to pick the min-greater key from.
- L1H0 is the value/induction head (ablation: value-acc 1.00->0.05, key-acc barely moved),
  orthogonal to key selection. This matches prior findings.

## 4. Threshold / next-key representation in the residual — CONFIRMED (formed in layer 1)
Linear probes at key-query positions, accuracy by stage:

| target | embed | block0 | block1 | final |
|---|---|---|---|---|
| last-emitted key value | 0.10 | 0.13 | **0.87** | 0.87 |
| next key value (to emit) | 0.10 | 0.10 | **1.00** | 1.00 |

Neither quantity is present at embed/block0 (near chance for the ~16 present keys). Both
appear at **block1**: the last-emitted key (the threshold) at 0.87 and, strikingly, the
**next key already at 1.00** — the answer is fully computed in the layer-1 residual at the
query position, before the unembedding. This localizes the entire sort computation to
layer 1 (the L1H1 read plus the block-1 MLP), consuming the input-key set that layer 0
deposited at SEP.

## Circuit summary
1. **Layer 0** (at input/SEP positions): L0H0/L0H1 read all input keys and write the
   present-key set into the SEP residual.
2. **Sort head L1H1** (at each key query): recency-weighted read of the emitted keys →
   recovers threshold = running max of emitted keys.
3. **Block-1 MLP + SEP-routed key set**: compute "smallest present input key > threshold";
   next key is linearly decodable at 1.00 by end of block 1.
4. **L1H0** independently handles value carry by induction (not part of key selection).

## Status tags
- CONFIRMED: L1H1 reads output recency-weighted (dominant last key, geometric tail).
- CONFIRMED: next-key rule = min input key > running max of emitted keys (causal, 99.6–99.8%).
- NEW: the operative threshold is the running **max**, not literally the last token
  (revealed by non-monotone causal patch: 0.4% vs 99.6%).
- NEW: input-key set is routed through the **SEP residual** (L0 reads keys at SEP; SEP
  encodes presence at 0.81); L0 ablation halves key accuracy though L0 parks on SEP.
- CONFIRMED: sort answer computed in layer 1 — next key decodable 1.00 at block1, 0.10 before.

## Controls / uncertainty
- Probes use held-out 25% split, standardized features, logistic regression; chance for
  present-key classification is ~1/16=0.06, so embed/block0 ~0.10 is at floor.
- Causal patch is partly OOD (substituting keys breaks the model's training distribution of
  monotone outputs); the monotone split (0.998) is in-distribution and the non-monotone
  split is the informative OOD probe, not a claim about normal operation.
- SEP presence-probe accuracy (0.81) is modest because 16 keys are compressed into 48 dims;
  it demonstrates the signal exists, not that it is loss-free.

## Figures (reports/figures/)
- `agent_sort_attn_profiles.png` — L1H1/L1H0 attention vs recency + L1H1 rank heatmap.
- `agent_sort_probes_rule.png` — threshold/next-key probes by depth + per-head class mass at key steps.
- `agent_sort_causal_rule.png` — causal last-key patch: last-rule vs max-rule, monotone vs non-monotone.
