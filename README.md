# Smol Transformer Lab

A local GPU-backed web lab for training tiny transformers on symbolic operations and looking inside their attention heads and MLP neurons.

## Run

```bash
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. PyTorch automatically uses CUDA when it is available. Training batches mix lengths from two digits through the configured maximum and use `=` as an end marker (`1234=` → `4321=` for reverse). Padding is masked from attention and loss. Completed or stopped runs are saved under `checkpoints/` and can be loaded from the page.

The training panel includes measured presets for the smallest copy, reverse, sort, and rotate-left models, plus a larger multi-head explorer preset. Selecting a preset fills the architecture fields; editing any architecture field switches back to Custom.

The **All tasks · shared** preset trains one model on batches mixed across tasks and lengths. Inputs are prefixed with `<copy>`, `<reverse>`, `<sort>`, or `<rotate_left>`; the browser's inspector exposes a task selector for counterfactual inspection of the same digit string.

`translate` is bidirectional: `123=` becomes `abc=`, while `abc=` becomes `123=`. Its focused preset and the shared model add `a`, `b`, `c`, and `<translate>` without changing IDs used by older checkpoints.

## Test

```bash
python -m pytest
```

The explorer displays the selected attention head as a token-to-token heatmap and the selected layer's MLP hidden activations as a neuron grid. The JSON API is documented at `/docs` while the server is running.

To reproduce an architecture search and write `pareto-results.json`:

```bash
python scripts/find_pareto.py --steps 1000
```

On an RTX 4080 with maximum length 9, seed 42, and 1,000 steps, the smallest tested models reaching at least 99% held-out exact-sequence accuracy were:

| Operation | Width / heads / layers / MLP | Parameters | Exact accuracy |
| --- | --- | ---: | ---: |
| copy | 8 / 1 / 1 / 8 | 756 | 100% |
| reverse | 16 / 2 / 2 / 16 | 3,964 | 100% |
| sort | 16 / 1 / 1 / 16 | 2,268 | 99.08% |
| rotate left | 8 / 1 / 1 / 8 | 756 | 100% |

These are empirical results for one seed and training budget, not lower-bound proofs. Full measurements and the multi-objective frontier are in `pareto-results.json`.

The webpage's **Where capability appears** report summarizes class-wise 99% thresholds, a width-by-depth accuracy heatmap, and a log-scale accuracy/parameter Pareto chart.

Mechanistic interpretability studies covering attention circuits, causal ablations, probes, MLP selectivity, and cross-task representation similarity are indexed in [`reports/README.md`](reports/README.md).
