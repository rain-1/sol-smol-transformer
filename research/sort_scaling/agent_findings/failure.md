# Failure-mode characterization — INTERRUPTED (GPU stand-down)

**Status: incomplete.** The full analysis run (`research/sort_scaling/agent_failure.py`)
was intentionally killed by the coordinator because it allocated ~15.8 GB VRAM and was
throttling a shared GPU / concurrent training job. This was **not** a code crash. No
figures were produced (`reports/figures/agent_fail_*.png` were never written), and the
large error-localization / duplicate-key / SEP-sink sweeps did not complete.

Below are **only** the results I actually computed and verified with small, already-completed
runs before the interruption. Everything else in the brief (duplicate-key attention split,
SEP-sink causal ablation, count/remaining probe, length-vs-induction curves, figures) is
**NOT DONE** and should be treated as open. The analysis code implementing all of these is
written and left in `agent_failure.py` for a VRAM-frugal re-run (reduce `n` in
`fig_error_localization`, which is the memory/time hog).

## Verified findings

### 4. EOS / termination — NEW (strong, verified two independent ways)
The "flaky EOS" behavior is actually **total, and mechanistically explained: the EOS token
is never in the training loss, so the model is never trained to emit it.**

- Verified from the data pipeline (`cs.make_batch`, CPU, length p=6): the scored label
  positions are seq indices `2p .. 4p` (SEP through the last sorted value `sv_{p-1}`). The
  EOS marker sits at index `4p+1` and is **not** in the score mask. Concretely for p=6:
  scored indices `[12..24]`, EOS at index 25 has `score=False`. So loss never covers EOS.
- Verified behaviorally (teacher-forced argmax, n=512 per length, lengths 5/15/29/30):
  P(emit EOS at the final slot) = **0.000** at every length tested. Instead the model
  emits a **key token**, almost always id **49** (the maximum key id), i.e. it tries to
  continue the sort with a spurious "next key" rather than stopping. Top predicted token at
  the EOS slot: id 49 with a small tail on 47/48 (also high key ids).

Implication: `generate_exact` in `causal_sort.py` only decodes exactly `2p` output tokens
and never consults EOS, so the reported ~99.9% "gen-exact" was never gated on termination.
The model has **no learned stop signal**. Whether a "pairs-remaining / count-emitted" signal
nonetheless exists in the residual stream (brief item 4) was **not** measured — that probe
did not run.

### 1. Error localization — PARTIAL (small-n only; large sweep not done)
At n=512 per length, teacher-forced **and** true autoregressive greedy decoding both give
per-token key accuracy = 1.000 and value accuracy = 1.000 at lengths p = 5, 15, 30. So
errors are rarer than ~1/512 per token at these lengths; the intended large-n (n≈6k–20k)
localization by rank/length and the key-vs-value cascade decomposition were **not** completed.
No conclusion on where the ~0.1% errors concentrate.

## NOT computed (open — do not treat as findings)
- 1. Error rank/length localization and value-error cascade-vs-induction decomposition (large n).
- 2. Duplicate-key breakdown: induction attention split on tied keys; accuracy vs #ties.
- 3. SEP attention-sink causal test (block/renormalize attention to SEP; effect on key vs value).
- 4. Count / positions-remaining residual probe.
- 5. Length curves for accuracy and induction sharpness across 2..30, error concentration near max length.
- 6. Figures (`agent_fail_*.png`).

## Reproduction notes for a lean re-run
- Code: `research/sort_scaling/agent_failure.py` (implements all six items; imports only
  `causal_sort` and `deep_interp`, does not modify them).
- The VRAM/time hog is `fig_error_localization` (autoregressive decode, `n` batch × up to
  61 forwards × 29 lengths). Drop `n` to ~1000–2000 and it should be frugal; the other
  figures (`fig_duplicate_keys`, `fig_sep_sink`, `fig_eos_termination`) are much lighter.
- The EOS result above needs no GPU sweep — it is a property of the loss mask plus a tiny
  forward pass and is the highest-confidence finding here.
