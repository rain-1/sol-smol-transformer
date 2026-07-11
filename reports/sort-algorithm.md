# The sort decoder as an algorithm

This report states, as annotated pseudocode, the algorithm the trained 2-layer
causal pair-sort decoder actually implements — with every step backed by a specific
interpretability result. It is the synthesis of
[`sort-scaling.md`](sort-scaling.md) (the circuit and its universality),
[`sort-circuit-walkthrough.md`](sort-circuit-walkthrough.md) (heads and MLPs on an
example), and the component analyses in `research/sort_scaling/`.

Roles are stated for the committed length-30 checkpoint. Head *indices* permute
across seeds (the value/induction head + single precursor + sort-by-generation
recur; the clean single sort head does not — see `sort-scaling.md` §7), so read the
names as roles.

## Data flow

```
 input pairs            SEP (scratchpad)           generated output (left→right)
 k v k v k v ... ─┬──▶  [ present-key MENU ]  ──┐
                  │      built by L0 heads       │  read each key step (3 heads)
                  │      + L0 MLP                 ▼
                  │                        ┌─ KEY STEP q ───────────────┐
                  │   L1H1 reads emitted ─▶│ threshold = running max    │─▶ emit sk_q
                  │   keys → threshold     │ next key = smallest        │
                  │                        │   present key > threshold  │
                  │                        └────────────┬───────────────┘
                  │                                     ▼
                  └── L1H0 induction ───────▶  VALUE STEP q: copy value ─▶ emit sv_q
                        (match sk_q to its input pair)
```

## The algorithm

```python
# INPUT tokens:  k0 v0 k1 v1 ... k_{n-1} v_{n-1}  SEP   (then the model generates)
# keys are distinct ids; values are a disjoint id set.

# ── Layer 0: gather & bind (the "setup") ──────────────────────────────────────
for i in range(n):                      # at each input VALUE position
    bind(value_slot_i, identity_of=key_i)      # [1] precursor stamps the value with its key

menu = compress({k0, ..., k_{n-1}})            # [2] L0 reads all keys AT SEP; L0 MLP
residual[SEP] = menu                           #     compresses the present-key SET there

# ── Generation: emit the sorted pairs left to right ───────────────────────────
threshold = -infinity
for q in range(n):
    # KEY STEP  (position that emits the q-th sorted key)
    threshold = running_max(keys_emitted_so_far)   # [3] L1H1 reads the emitted keys
    sk_q = smallest key in menu with key > threshold  # [4] threshold-gated readout of
    emit(sk_q)                                      #     (menu + threshold); NOT a scan

    # VALUE STEP  (next position)
    sv_q = value_of_the_input_pair_whose_key == sk_q  # [5] L1H0 induction: match & copy
    emit(sv_q)

# (the model never emits EOS — that token was outside the training loss mask)   # [6]
```

## Every step, and the evidence for it

| # | Step | Mechanism | Evidence | Confidence |
|---|------|-----------|----------|------------|
| 1 | bind value↔key | L0H0/L0H1 are previous-token / precursor heads: each input value slot is stamped with its key's identity | input-region diagonal attention; the induction match is a perfect key-identity match only once L0H0's OV is composed into L1H0's K (+12.7σ, head-specific) | **weight-level** (`agent_qkov`) |
| 2 | build the candidate set | L0 heads read all input keys **at the SEP position**; the L0 MLP builds a present-key representation. `SEP` holds a readable copy of the set (≈ a sum of key embeddings) but the causally-used identities are **distributed** across the block-0 residual, not a swappable `SEP` object | `SEP` residual predicted by the present-key multi-hot at R²≈0.84 (≈emb-sum R²0.81); ablating the **L0 MLP** collapses key accuracy 1.00→0.20; but transplanting `SEP` alone installs a new set only at 0.08 (full block-0 transplant 1.0) | **causal + probe** (`walkthrough`, `agent_sorthead`, `riddle_menu`, `riddle_causal`) |
| 3 | threshold = running max | L1H1 reads the already-emitted keys with recency weighting → the running maximum | L1H1 key-step attention: 0.94 on generated output, geometric recency (0.61 on the last key); causal patch: next key tracks the running **max** (99.6%), not the last key (0.4%) | **causal** (`agent_sorthead`) |
| 4 | pick smallest present key > threshold | A **threshold-gated linear readout**: the logit for a present key is a **cliff** (~+8 logits for `k > θ`) plus a **linear penalty** growing with `k − θ`, so the argmax is the smallest present key just above θ. Not a scan or pairwise comparison | swept-θ staircase matches the ideal rule at 1.000 (0.995 over 40 menus); measured cliff ≈8 logits (6.2 above θ vs −2.0 at/below), peak at the smallest `k > θ`; readout is linear at the L1-attn output (97.3% pre-MLP); decision lives in a ~12–16-dim subspace | **confirmed; readout characterized** (`compare`, `riddle_threshold`, `riddle_causal`) |
| 5 | copy the paired value | L1H0 induction head matches the just-emitted key `sk_q` back to its input occurrence and copies the following value | argmax on the matching input value 100% of the time; L1H0's OV is a diagonal value-**copy** matrix (100% argmax); value decodable/aligned to its unembedding at block 1 (cos 0.77) | **weight-level** (`agent_qkov`, `deep_interp`, `agent_geometry`) |
| 6 | (no termination) | EOS is outside the training loss mask, so it is never learned; the model keeps trying to sort | P(emit EOS at the final slot)=0.000 at every length; it emits the max key id instead | **confirmed** (`agent_failure`) |

## What is solid vs. still open

- **Solid, down to the weights:** the value channel (steps 1 and 5) — a key-identity
  induction head fed by a single precursor, with an OV that is a literal copy matrix.
  This recurs across all five seeds.
- **Solid, causally:** the sort *rule* (step 3, running-max threshold) and the roles
  of each component (SEP scratchpad, L0-MLP menu builder). Sort-by-generation (no
  global rank) is seed-universal.
- **Step 4, now resolved (the former open block):** the `>` is realised as a
  threshold-gated **linear readout** at the L1-attention output — a logit cliff at
  `k > θ` plus a linear penalty in `(k − θ)`, whose argmax is the smallest present
  key just above θ. A swept-threshold experiment traces the exact sorting staircase
  from a single fixed menu (match 1.000). Two honest residual caveats: the threshold
  is a distributed ~5–8-dim code (steerable only in that many dims, not one), and the
  candidate set is distributed across the block-0 residual rather than a swappable
  `SEP` object — so the decision occupies a ~12–16-dim linear subspace, not a handful
  of named directions. The *mechanism* is pinned; a fully minimal basis for it is not.

## Caveats

One architecture family on one synthetic task; ablations are zero-ablations
(off distribution); probes establish accessible information, not that the model uses
that exact form. The depth-ladder "counting sort" result for the *encoder* (see
`sort-scaling.md` §1) is behavioural inference, not a weight-level proof like the
decoder's induction circuit.
