# Circuit universality across seeds

**Question.** The seed-42 study found a specific circuit for the 2-layer / 2-head
causal pair-sort decoder (d_model=48, d_ff=96). Is that circuit universal, or an
artifact of one seed? I trained 5 fresh models (seeds 0,1,2,3,7) at max_pairs=16,
11k steps, and re-ran the seed-42 diagnostics on each, aligning heads by **behavior**
(not index) at analysis length p=12.

Code: `research/sort_scaling/agent_seeds.py` (reuses `causal_sort.train/make_batch`
and `deep_interp` analysis functions unchanged).
Per-seed metrics: `research/sort_scaling/agent_findings/seeds_metrics.json`.
Checkpoints: `research/sort_scaling/checkpoints/seed_{0,1,2,3,7}.pt` (NOT committed).
Figures: `reports/figures/agent_seeds_summary.png`, `agent_seeds_head_matrix.png`,
and per-seed `agent_seeds_precursor_s*.png` / `agent_seeds_circuit_s*.png`.

## Convergence

All 5 seeds reached high generation-exact accuracy at length 16:

| seed | tf_token | gen_exact |
|------|----------|-----------|
| 0 | 1.000 | 1.000 |
| 1 | 1.000 | 0.999 |
| 2 | 0.997 | 0.957 |
| 3 | 1.000 | 0.999 |
| 7 | 1.000 | 0.999 |

(Seed 2 is slightly lower but still solves the task.)

## Per-seed aligned circuit (analysis length p=12)

| seed | induction head | ind. score | argmax-on-match | value-drop | precursor | prec. drop (base≈0.9) | sort head (reads output) | out-read mass | SEP sink | key-rank probe |
|------|------|------|------|------|------|------|------|------|------|------|
| 0 | L1H0 | 0.92 | 1.00 | 0.95 | L0H0 | 0.84 | L1H1 | 0.27 | 0.93 | 0.36 |
| 1 | L1H0 | 0.88 | 1.00 | 0.94 | L0H1 | 0.80 | L0H0 | 0.87 | 0.92 | 0.35 |
| 2 | L1H1 | 0.96 | 1.00 | 0.96 | L0H1 | 0.87 | L1H0 | 0.61 | 0.92 | 0.31 |
| 3 | L1H1 | 0.95 | 1.00 | 0.93 | L0H0 | 0.86 | L1H0 | 0.69 | 0.92 | 0.35 |
| 7 | L1H1 | 0.94 | 1.00 | **0.29** | L0H1 | 0.85 | L1H0 | 0.68 | 0.92 | 0.34 |

Seed-42 reference (length 12): induction L1H0 (argmax-on-match 1.0, value-drop 0.95),
precursor L0H0 (drop 0.89 of 0.99), sort head L1H1 (reads output), key-rank 0.32.

Heads are permuted across seeds (e.g. the induction head is L1H0 in seeds 0,1,42 and
L1H1 in seeds 2,3,7), confirming the need to align by behavior.

## Verdict per sub-claim

### (a) Exactly one dedicated value/induction head — CONFIRMED (4/5), with one NEW redundant case
Every seed develops a **layer-1 induction head** that attends by key identity: induction
score 0.88–0.96 and **argmax-on-match = 1.00 in all 5 seeds** (it always lands its top
attention on the matching input value). In seeds 0,1,2,3 (and 42) this head is *the*
value carrier: ablating its value-output collapses value accuracy (drop 0.93–0.96).

- **NEW (seed 7):** value-carrying is **redundant across two L1 heads.** The primary
  induction head L1H1 has a strong induction score (0.94, argmax 1.0) but ablating it
  alone drops value accuracy only 0.29, because L1H0 is a second induction head (score
  0.67). Verified directly: ablate L1H1 → value 0.71; ablate L1H0 → 0.83; **ablate both
  → 0.04 (drop 0.96).** So the *function* is universal (L1 induction) but seed 7 splits
  it over two heads instead of one.

### (b) One dedicated key/sort head — REFUTED as a crisp single head; the sort *function* is universal but delocalized
Unlike seed 42 (where L1H1 alone dominated key-output ablation, drop 0.93 vs next 0.52),
in the new seeds **key-output ablation is diffuse** — every head shows a 0.5–0.9 key-drop,
so "max key-drop" does not isolate one head. The behavioral signature (a head reading the
*generated output* at key-emit steps, i.e. sorting by generation) is present in every seed
but lives in different heads (L1H1, L0H0, L1H0×3) and at very different strengths
(output-read mass 0.27 → 0.87). Seed 0's sort head reads generated output only weakly (0.27).
So: a sort-by-generation reader exists everywhere, but "exactly one dedicated sort head,
cleanly separable by ablation" is a seed-42 particularity, not universal.

### (c) A single precursor whose ablation collapses induction — CONFIRMED (5/5)
Every seed has **exactly one layer-0 precursor**. Ablating it drops the induction head's
score by 0.80–0.87 (from a baseline ≈0.9 down to ≈0.05–0.1); ablating any other head moves
it by ≤0.11. This L0→L1 precursor→induction dependency is the most robust motif of all.

### (d) Sort-by-generation / low key-rank decodability — CONFIRMED (5/5)
Key-rank is only weakly linearly decodable from the final residual at input-key positions:
0.31–0.36 across all seeds, essentially identical to seed 42 (0.32). The model does not
pre-compute a rank field; it sorts autoregressively by reading what it has already emitted.

### Bonus: SEP attention sink — CONFIRMED (5/5)
Every seed parks ~0.92–0.93 attention mass on SEP in at least one head at key steps.

## Bottom line

The circuit is **substantially universal in mechanism, with permuted labels and one
redundancy exception.** Robustly recurring (5/5): a layer-1 key-identity **induction head**
(argmax-on-match 1.0 everywhere), a **single layer-0 precursor** that gates it, **sort-by-
generation** (low key-rank decodability), and a **SEP sink**. Deviations from the tidy
seed-42 story: (1) **seed 7 splits value-carrying across two redundant L1 induction heads**
rather than one; (2) the **key/sort head is not a single ablation-clean head** in the new
seeds — the sort-by-generation *reading* is universal but delocalized and variable in
strength. Head indices are freely permuted, so any claim must be stated per role, not per
index.
