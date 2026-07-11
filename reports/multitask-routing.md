# Operation routing in the shared multi-task transformer

## Summary

The 18,512-parameter shared checkpoint uses its operation prefix as a genuine control input. Changing only that token redirects the same digits to the requested transformation with 98.9–100% exact accuracy. Routing is established mainly in layer 1, where heads show pronounced task specialization; layer 2 combines more shared, partially redundant computations. The operation signal remains task-decodable in task-conditioned activation means even when later attention no longer reads the prefix directly.

These claims come from 1,024 paired, fixed-length (7 digit) examples per task. Baseline exact accuracy was 100% for copy, reverse, and rotate-left, and 98.93% for sort.

## The prefix causally selects the algorithm

The strongest intervention changed the operation token while holding every digit fixed. For every source sequence, substituting `<copy>`, `<reverse>`, or `<rotate_left>` made the model produce that operation's target with 100% exact accuracy. Substituting `<sort>` produced the sorted target with 98.93% exact accuracy. Accuracy against the original, now-unrequested transformation was essentially zero (the occasional 0.2–0.3% match is expected when two operations happen to have the same output).

This is stronger than a correlation between prefix and output: a one-token counterfactual redirects behavior while all content tokens remain unchanged.

## Where the operation token is read

Mean attention from non-prefix, non-padding queries to prefix position 0:

| Task | L1H0 | L1H1 | L1H2 | L1H3 | L2H0 | L2H1 | L2H2 | L2H3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Copy | .394 | .715 | .851 | .599 | .645 | .647 | .370 | .612 |
| Reverse | .406 | .345 | .164 | .054 | .00008 | .170 | .00001 | .00016 |
| Sort | .100 | .284 | .767 | .100 | .120 | .407 | .016 | .027 |
| Rotate left | .351 | .497 | .186 | .210 | .00050 | .005 | .002 | .004 |

Copy repeatedly consults the prefix in both layers. Sort particularly recruits L1H2 and L2H1. In contrast, nearly all layer-2 reverse and rotation heads ignore the operation position. Since those tasks still execute correctly, the natural interpretation is that layer 1 has already written the control state into content-position residuals before layer 2 performs routing. Attention weights are observational and should not themselves be interpreted as causal effects.

Task-conditioned residual means are separated after both layers. Normalized pairwise distances after layer 1 range from 0.291 (reverse versus rotate) to 1.370 (copy versus rotate); after layer 2 they remain 0.303–0.475. MLP means become especially distinct in layer 2 (up to 0.724), except reverse versus rotation (0.106). Thus reverse and rotate share a similar late MLP state despite implementing different positional routes.

## Causal division of labor among heads

The following values are drops in exact accuracy from ablating one head's output contribution:

| Head | Copy | Reverse | Sort | Rotate left |
|---|---:|---:|---:|---:|
| L1H0 | .000 | .589 | .354 | **.991** |
| L1H1 | **1.000** | .557 | .676 | .785 |
| L1H2 | .956 | .415 | **.989** | .001 |
| L1H3 | .798 | **.611** | .148 | .383 |
| L2H0 | .838 | .223 | .703 | .055 |
| L2H1 | .544 | .206 | **.987** | .080 |
| L2H2 | .102 | .290 | .474 | .233 |
| L2H3 | .086 | .074 | .476 | .037 |

Layer 1 has the clearest specialization:

- L1H0 is a rotation specialist: its removal costs 99.1 points on rotation but none on copy.
- L1H2 is a sorting/copy head and is dispensable for rotation.
- L1H3 has the largest single reverse effect, although it is also important to copy.
- L1H1 is broadly necessary rather than task-specific.

Layer-2 heads are more shared. Sorting depends heavily on L2H0 and L2H1, while reverse and rotation distribute their dependence across heads and tolerate any single late-head ablation better. Single-head effects are not additive because heads can compensate nonlinearly.

## Attention and MLP components

Zeroing either whole attention layer destroys essentially every task, so both attention stages form shared infrastructure. The layer-1 MLP is also required by all tasks (96.4–100 point exact-accuracy drops). The layer-2 MLP is different: its removal costs 69.8 points for copy and 96.9 for sort, but only 0.1 for reverse and 1.2 for rotation. Late MLP computation is therefore task-selective at the component level; positional permutations mostly rely on the attention/residual pathway, whereas copy and especially sorting require late nonlinear processing.

## Interpretation and limitations

The evidence supports a two-stage picture: early heads read the operation prefix and establish a task-conditioned residual state; later components execute a mixture of specialized and shared transformations. There is no clean one-head-per-task modularity. Instead, specialization is strongest in layer 1 and for sorting, overlaid on components required by several tasks.

All ablations set output projections to zero. This is a strong off-distribution intervention, so effect size establishes necessity under ablation, not the exact normal-run contribution. Results use one checkpoint, one seed, and one length. Activation distances compare means and do not prove linear decodability. Repeating across seeds and lengths, mean-ablation or activation patching, and training explicit task probes would strengthen the conclusions.

## Reproduction

```bash
python research/multitask_routing/analyze.py
```

The script loads `checkpoints/multi_task-20260711-103650.pt` without retraining. Raw measurements and intervention outcomes are in `research/multitask_routing/results.json`.
