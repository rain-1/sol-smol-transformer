# Causal interventions in the tiny symbolic transformers

## Summary

This study trained the four published minimal presets and intervened on their computations on the same 4,096 held-out mixed-length examples. The main result is a qualitative split: **copy can be implemented almost position-free, while reverse, sort, and rotate require learned position signals and attention**. The `=` delimiter is not merely punctuation for reverse and sort: replacing it with `0` while leaving the desired digit output unchanged reduces digit-exact accuracy from approximately 100% to 0.10%.

These are causal effects in one trained model per task, not evidence that every model trained from a different seed uses the same circuit.

## Method

Models used sequence width 9 and lengths 2–8, with right padding. Each was trained for 1,000 steps with seed 42 using the repository's normal sampler and optimizer.

| Task | d_model | Heads | Layers | d_ff | Parameters | Baseline exact |
|---|---:|---:|---:|---:|---:|---:|
| Copy | 8 | 1 | 1 | 8 | 756 | 100.00% |
| Reverse | 16 | 2 | 2 | 16 | 3,964 | 100.00% |
| Sort | 16 | 1 | 1 | 16 | 2,268 | 98.90% |
| Rotate left | 8 | 1 | 1 | 8 | 756 | 100.00% |

For a head ablation, I zeroed the columns of the attention output projection corresponding to that head. For an MLP-neuron ablation, I zeroed that neuron's column in the MLP output projection. Whole-component ablations zeroed the entire attention or MLP output projection. This removes only the component's additive residual-stream contribution and leaves all other parameters fixed. Effects below are baseline exact accuracy minus intervened exact accuracy, in percentage points (pp). Evaluation examples were frozen and paired across interventions.

Two input/representation interventions were also performed:

1. Replace the input `=` with digit `0`, but retain the original target. `digit exact` ignores the output delimiter position, avoiding the trivial failure caused by expecting `=` there.
2. Zero all learned position embeddings while retaining tokens and targets.

The complete machine-readable measurements and checkpoints are in `research/causal/`.

## Component necessity

| Task | Layer | Remove attention | Remove MLP |
|---|---:|---:|---:|
| Copy | 0 | 39.9 pp | 100.0 pp |
| Reverse | 0 | 99.4 pp | 91.2 pp |
| Reverse | 1 | 99.8 pp | **0.0 pp** |
| Sort | 0 | 98.7 pp | 98.7 pp |
| Rotate left | 0 | 99.7 pp | 44.6 pp |

Reverse's second-layer MLP is completely dispensable under this intervention, despite being present in the architecture. The useful nonlinear computation is concentrated in layer 0; both attention layers remain necessary. Copy shows the opposite of a simple “attention does the task” story: its MLP contribution is jointly essential, while eliminating attention still leaves 60.1% of sequences exactly correct.

## Attention heads

| Task | Head effects (layer.head: exact drop) |
|---|---|
| Copy | 0.0: 39.9 pp |
| Reverse | 0.0: 78.8; 0.1: 90.2; 1.0: 67.0; 1.1: 72.4 pp |
| Sort | 0.0: 98.7 pp |
| Rotate left | 0.0: 99.7 pp |

Every reverse head is individually important, but none alone accounts for the full layer effect. This is consistent with overlapping or cooperative subcircuits rather than a single privileged “reverse head.” Weight magnitude is not a reliable importance proxy: across the four reverse heads, output-projection norm is negatively correlated with exact-accuracy effect (Pearson r = -0.87, n = 4). The sample is tiny, but it is a useful warning against ranking heads by norm.

## MLP neurons and distributed computation

The strongest single-neuron effects occur in reverse layer 0: neurons 14, 6, 3, 0, and 13 cause 68.4, 64.8, 60.3, 55.3, and 51.0 pp exact drops respectively. Sort likewise has two dominant neurons (layer 0 neurons 4 and 5: 86.1 and 83.8 pp).

Copy is strikingly distributed: removing its entire MLP costs 100 pp, yet no individual neuron costs more than 0.1 pp. Rotate has a milder version: whole-MLP removal costs 44.6 pp, while its largest individual-neuron effect is 4.6 pp. These results demonstrate redundancy and interaction; single-neuron effects do not add linearly and cannot substitute for group ablations.

## Variable length and the `=` delimiter

| Task | Baseline digit exact | Digit exact after `=`→`0` | Exact after zeroing positions |
|---|---:|---:|---:|
| Copy | 100.00% | 59.03% | 100.00% |
| Reverse | 100.00% | **0.10%** | **0.63%** |
| Sort | 98.90% | **0.10%** | 1.86% |
| Rotate left | 100.00% | 23.68% | 5.69% |

The delimiter is causally involved in all four models, not only used to emit the final `=`. Its strongest role is in reverse and sort. Because sequences are padded to a fixed width, the model also receives length information from the padding mask; nevertheless, replacing only the delimiter destroys almost all correct digit sequences for these tasks. A likely interpretation is that `=` serves as an attended anchor for the variable endpoint, though attention-pattern visualization or activation patching is needed to localize that information flow.

Position embeddings are unnecessary for copy in this seed: zeroing them changes no prediction. This fits a token-wise identity solution. In contrast, reverse, sort, and rotate collapse. Note that sort is permutation-invariant as a function of input digits, but its *ordered output slots* still require positional identity.

Before intervention, exact accuracy by input length was 100% at every tested length for copy, reverse, and rotate. Sort ranged from 97.3% (length 8) to 100% (lengths 2–3), so its 98.9% aggregate baseline should not be treated as fully solved.

## Caveats and next experiments

- One seed per task is insufficient for universal circuit claims. Repeat across at least 5–10 seeds and report distributions.
- Zero ablation is off-distribution and changes downstream layer-normalization inputs. Mean/resample ablation and activation patching would test robustness.
- Head ablation at the output projection identifies causal contribution, but not which query/key/value pathway carries endpoint, token, or positional information.
- Exact accuracy has threshold behavior: a small logit change can cause a large sequence-level drop. Token accuracy is included in the raw data.
- The delimiter replacement uses a real digit and therefore changes the digit multiset. Patching the clean `=` activation into a corrupted run, layer by layer and position by position, is the most direct next causal-tracing experiment.
- Joint neuron/head ablations would distinguish redundancy from synergy more precisely, but the combinatorial search should use sparse screening or Shapley approximations.

## Reproduction

```bash
python research/causal/run_interventions.py --retrain
```

The script writes `research/causal/results.json` and task checkpoints under `research/causal/checkpoints/`. Omit `--retrain` to reuse those checkpoints and rerun only interventions.
