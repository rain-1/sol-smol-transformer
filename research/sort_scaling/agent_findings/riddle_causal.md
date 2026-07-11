# Causal test of the two-factor (menu × threshold) sort mechanism

Model: 2-layer, 2-head causal pair-sort decoder (d_model=48, d_ff=96, vocab=73,
seq_len=122), checkpoint `causal_pairs_L30.pt` (gen_exact 0.9988). Rule: next key =
smallest input key strictly greater than the running max of the keys already emitted.

All numbers below are produced by `research/sort_scaling/agent_riddle_causal.py`
(metrics → `agent_findings/riddle_causal_metrics.json`; figures →
`reports/figures/riddle_causal_swaps.png`, `riddle_causal_das.png`). The forward pass
is re-implemented so patched tensors can be injected; the committed model is not modified.
Residual staging matches `walkthrough.full_capture`: block-0 output = `caps[0].x_out`,
`x_mid` = post-L1-attn / pre-MLP = `caps[1].x_mid`. **Every patch is off-distribution;
see caveats.**

Query geometry (fixed length p, sep=2p): sorted key `sk_q` is emitted at query position
`sep+2q`; previously emitted keys sit at `sep+1+2j`, j<q; the running-max threshold at
step q is `sk_{q-1}`.

---

## Headline

- **THRESHOLD factor — CONFIRMED, cleanly transplantable.** Replacing the L1-attention
  output at a key-emit query with the L1-attention output from a *different step* of the
  same sequence moves the cut to the other step's threshold in **99.98%** of cases
  (n=56 320), while the emitted key stays inside the sequence's own candidate set in
  **99.98%**. The threshold and the candidate set are causally separable factors.
- **MENU factor — the "SEP holds a swappable menu" localization is REFUTED.** Transplanting
  the SEP block-0 residual (or the whole input-region block-0 residual) from sequence B
  into A's run does **not** install B's candidate set (follows-B 0.08 / 0.00, at chance).
  A full block-0 transplant does yield B's answer (0.999, plumbing control), so the
  machinery works — the candidate identities simply are not carried in any single
  input-side residual location in a transplantable form. The menu is *distributed*.
- **Decision subspace — CONFIRMED, low-rank.** The next-key decision transfers between
  sequences through a **~12–16-dimensional** subspace of the 48-dim `x_mid` (rank 12 →
  0.89, rank 16 → 0.99 transfer), vs. a random subspace of the same rank at 0.04 / 0.13.
- **Sanity — CONFIRMED.** The whole `x_mid` vector at the query position deterministically
  sets the next key (whole-vector transplant 1.00), and the model's own next key is a
  near-linear readout of `x_mid` (probe agreement 0.999). The L1 MLP is a minor refinement.

---

## 1b. THRESHOLD swap — CONFIRMED (the clean dissociation)

Within one sequence the SEP menu is constant across steps, so transplanting the **entire
L1-attention output** from a later step q2 into an earlier step q1 changes only the
step-varying part (the running-max threshold read by sort head L1H1). Prediction:
the cut moves from `sk[q1]` to `min(keys > sk[q2-1]) = sk[q2]`, while the candidate set
stays the sequence's own keys. All (q1<q2) pairs, p=12, n=1024:

| quantity | rate |
|---|---|
| unpatched follows original cut `sk[q1]` | 0.9998 |
| patched follows original cut `sk[q1]` | 0.000 |
| **patched follows swapped cut `sk[q2]`** | **0.9998** |
| patched key ∈ sequence's own menu | 0.9998 |

This is a full two-factor dissociation in one intervention: the **cut point moves** to the
transplanted threshold, and the **candidate set is preserved** (the emitted key is still
one of the sequence's own input keys, never invented). It localizes the threshold to the
step-varying component of the L1-attention output, consistent with the descriptive finding
that L1H1 recency-reads the emitted keys.

## 1a. MENU swap — SEP localization REFUTED; menu is distributed

Teacher-forced true prefixes; at each key step A's threshold is `T = sk_A[q-1]` and A's
own answer is `min(A_keys > T)`. If B's menu is installed, the two-factor model predicts
`min(B_keys > T)`. Measured on the discriminative subset where those differ (both real
keys), p=10, n=1024, n_disc=7 782 (× steps):

| block-0 positions patched from B | follows B's menu | pred ∈ A keys | pred ∈ B keys |
|---|---|---|---|
| SEP only | 0.077 | 0.921 | 0.269 |
| input pairs (0…2p−1), SEP kept as A | 0.000 | 1.000 | 0.198 |
| input pairs + SEP | 0.077 | 0.921 | 0.269 |
| **FULL block-0 (all positions) — plumbing control** | — | — | **0.999 = B's own next key** |

- Unpatched, the model follows A's menu at 0.9999.
- Patching input-pair residuals alone is a **perfect no-op** (follows-A 1.000): at key
  steps L1 does not read the input positions (L1 attention onto input ≈ 0), so their
  residuals are causally inert for the sort.
- Patching the SEP residual perturbs slightly (12% of A-answers break) but installs B's
  candidate set only at chance (follows-B 0.077; "pred ∈ B keys" 0.27 ≈ overlap baseline).
- The **full block-0 transplant reproduces B's answer at 0.999**, proving the patch path is
  correct. Hence the candidate identities are not isolable to SEP or the input residuals —
  they only materialize as a usable menu once combined at `x_mid` (see §2).

Diagnostic — L1 attention onto SEP at key steps: L0H0 0.994, L0H1 0.907, **L1H0 0.997**,
L1H1 0.065. The value/induction head L1H0 "parks" on SEP at key steps, but transplanting
SEP does not transfer a candidate set — its SEP read at key steps carries little
sort-relevant signal. This **refutes** the prior proposal that SEP is a static,
read-forward menu that three heads consume; the presence-probe evidence (0.81, lossy) was
not causally sufficient.

## 2. DAS-lite — the decision lives in ~12–16 of 48 dims

At the key-emit query, `x_mid` fully determines the next key (everything downstream —
ln2, MLP, final-norm, unembed — is position-wise). Transfer test on same-step
cross-sequence pairs (T,S) with differing model predictions:
`patched = x_T + U Uᵀ(x_S − x_T)`; success if the resulting argmax equals S's next key.
U = top-r PCA of the same-step difference vectors `x_S − x_T` (position held fixed, so
differences are pure menu+threshold variation). p=12, n_pairs=4000:

| rank r | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| difference-PCA subspace | 0.01 | 0.03 | 0.08 | 0.13 | 0.31 | 0.52 | **0.89** | **0.99** | 1.00 | 1.00 |
| random subspace (control) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.04 | 0.13 | 0.37 | 1.00 |

No-patch baseline (target keeps its own key) = 0.00; full transplant = 1.00. Rank 8
already transfers half the decisions and rank 12–16 nearly all, while a random subspace
needs ~24 dims to reach even 0.37. The sort decision is therefore carried in **roughly a
dozen directions** of the L1-attention output — low-rank but not one- or two-dimensional
(it must encode both a threshold and which of ~50 key identities is the minimal one above
it). The difference-vector spectrum is correspondingly spread (rank-12 captures 0.80 of
its variance, rank-16 0.90).

## 3. Sanity — `x_mid` at the query is sufficient and near-linear

- **Whole-vector transplant** of `x_mid` at the query between sequences sets the next key
  to the source's key at **1.00** (n=11 264) — confirms patching `x_mid` (post-L1-attn,
  pre-MLP) is sufficient to set the next key, so the MLP is not where the decision is made.
- A logistic probe on `x_mid` predicts the model's own next key at **0.999**, i.e. the
  decision is an essentially linear readout of `x_mid`; the L1 MLP only refines it.

---

## Status tags

- CONFIRMED (causal): threshold is a transplantable factor — swapping the L1-attn output
  across steps moves the cut (0.9998) while preserving the candidate set (0.9998).
- CONFIRMED (causal): next-key decision is fully carried at `x_mid` (whole-vector
  transplant 1.00) and is a near-linear readout (probe 0.999); MLP minor.
- CONFIRMED (causal): the decision transfers through a low-rank ~12–16-dim subspace of
  `x_mid` (rank 12 → 0.89, rank 16 → 0.99) far above a random-subspace control.
- REFUTED: "the SEP block-0 residual is a static, swappable candidate menu." Transplanting
  it (or any input-side block-0 residual) does not install another sequence's key set
  (follows-B 0.08 / 0.00), though the full-block-0 plumbing control works (0.999).
- NEW: the candidate menu is **distributed**, not localized to a single read-forward
  residual; it is only usable once combined with the threshold at `x_mid`. The two factors
  are nonetheless functionally separable (the threshold swap holds the menu fixed).

## Controls / OOD caveats

- **All patches are off-distribution.** Combining B's menu with A's threshold, or step-q2's
  threshold with step-q1's position, is a state the model never sees in training. The
  threshold swap and DAS results are clean *despite* being OOD, which is the strong evidence;
  the menu-swap null could in principle be partly an OOD artefact, but the plumbing control
  (full block-0 → B at 0.999) and the exact no-op of the input-pair patch (follows-A 1.000)
  show the null is about *localization*, not a broken intervention.
- Discriminative subsets exclude ties and EOS cases so "follows A" vs "follows B" (menu) and
  "orig cut" vs "swapped cut" (threshold) are genuinely distinct targets.
- DAS pairs are matched by step (position held constant) and required to differ in the
  model's own prediction, so the difference vectors isolate decision-relevant variation and
  transfer is measured against the source's *model* output (not the gold label).
- Chance for "pred ∈ B keys" is the A/B key-set overlap (~p/50 per key); the observed 0.20–0.27
  is at that baseline, confirming the SEP patch installs no B-specific candidate.

## Figures

- `reports/figures/riddle_causal_swaps.png` — 1b threshold swap (clean dissociation) and
  1a menu-swap localization across block-0 regions + plumbing control.
- `reports/figures/riddle_causal_das.png` — decision-transfer rate vs subspace rank
  (difference-PCA vs random control) and the difference-vector spectrum.
