# Interpretability research

Three independent studies examine the tiny operation transformers from complementary angles. Each report links claims to machine-readable results and a reproducible experiment script.

## Reports

- [Attention circuits](attention-circuits.md) — aggregate routing metrics and causal head-output ablations on preset-sized models.
- [Causal interventions](causal-interventions.md) — head, layer, MLP-neuron, delimiter, and positional-embedding interventions on 4,096 paired examples.
- [MLP representations](mlp-representations.md) — probes, neuron selectivity, group ablations, and cross-task CKA over three seeds per task.
- [Multi-task routing](multitask-routing.md) — task-selective heads, shared components, and operation-token redirection.
- [Multi-task causal control](multitask-causal-control.md) — token swaps, residual patching, corruptions, and embedding interpolation.
- [Multi-task representation geometry](multitask-representations.md) — task probes, CKA, PCA, additive task vectors, and causal steering.
- [Translation and rotation manifolds](translation-and-rotation-manifolds.md) — alphabet alignment, translation involution, and approximate cyclic equivariance.
- [Scaling sort, and what forces a routing circuit](sort-scaling.md) — counting vs. routing, why the encoder can't carry values, and the causal decoder's value-carrying induction head at length 30.
- [Sort decoder: a component-by-component walkthrough](sort-circuit-walkthrough.md) — every attention head and both MLPs explained on one concrete example, with heatmaps.
- [The sort decoder as an algorithm](sort-algorithm.md) — the decoder written as annotated pseudocode, each step backed by a specific interpretability result.
- [Attention gallery: all heads, every seed](sort-seed-gallery.md) — every head of seed 42 (main) plus the five universality seeds on increasing / decreasing / random inputs, induction and sort heads highlighted.
- [The activation (residual) stream](sort-residual-stream.md) — effective dimensionality by depth, per-component writes, stream geometry, and direct logit attribution.

## Joint picture

The strongest convergent finding is a two-stage variable-length reverse circuit. Early computation uses the terminal `=` as an endpoint/length anchor and develops a position-rich representation; the final attention layer becomes an almost perfectly anti-diagonal router. Both attention layers are causally necessary, while the second-layer MLP can be removed without affecting accuracy in the causal study.

Copy is qualitatively different. Its residual stream already contains the input digit at the correct output position, it remains perfect when positional embeddings are removed, and its MLP representation is substantially less position-selective. Attention and the MLP still contribute boundary/content corrections, but the computation is distributed: whole-component removal is damaging while individual-neuron removal is not.

Sort uses broad attention and distributed aggregation rather than a clean routing map. Rotate-left learns a sharp offset-routing circuit and strongly represents position. Across tasks, observational neuron selectivity is not a dependable proxy for causal importance; targeted selective-neuron ablations were generally no worse than random matched ablations.

## Shared multi-task model

The shared model reveals conditional modularity rather than four completely separate circuits. Task identity begins localized at the operation token, is broadcast into digit-position residuals during the first block, and becomes perfectly linearly decodable there. Early heads show strong task specialization—for example, distinct heads are selectively essential for rotate-left and sort—while later heads are more shared and redundant.

Changing only the operation token fully reroutes the same digits to the destination algorithm. Residual patching localizes this control flow: patching the operation position works at the embedding stage, but after block 1 the relevant state has spread into digit positions. Mean task-vector steering partially changes algorithms, whereas embedding interpolation produces sharp transitions rather than smooth mixtures. Together these results suggest a compact early task-control signal followed by nonlinear, content-dependent execution.

Translation adds a useful warning about embedding geometry: corresponding raw token embeddings (`1` and `a`, for example) are not similar, but their contextual final states are almost perfectly linearly aligned. The learned map is behaviorally involutive. For rotate-left, final token states follow the moved token much more closely than their absolute position and form approximate cyclic orbits; measurable closure and circulant-fit error rule out claiming exact equivariance.

## Scope and caution

The attention and causal studies use one fixed seed on small preset models; the representation study uses three seeds but a larger common architecture. These are studies of learned solutions, not proofs that training must find the same circuits. Zero ablation is also an off-distribution intervention. The reports document these limitations and propose activation patching, resample ablation, and multi-seed circuit comparison as next steps.

## Reproduce

```bash
python research/attention/analyze.py --retrain
python research/causal/run_interventions.py --retrain
python research/mlp/analyze.py --retrain
python research/multitask_routing/analyze.py
python research/multitask_causal/analyze.py
python research/multitask_geometry/analyze.py
python research/manifolds/analyze.py
```

The checked-in research checkpoints make analysis reproducible without retraining where supported. Raw outputs live beside each script as `results.json`.

## Figures

Every chart embedded in these reports is regenerated from the committed
`results.json` files (attention heatmaps are captured live from the attention
checkpoints) by a single script:

```bash
python research/figviz/make_figures.py
```

It writes PNGs into `reports/figures/`. Colours follow the shared palette in
`research/figviz/style.py`: one fixed hue per task, a single-hue blue ramp for
magnitude heatmaps.
