"""Analyze operation routing in a trained multi-task tiny transformer.

Run from repository root:
    python research/multitask_routing/analyze.py
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.model import ModelConfig, OPERATION_IDS, PAD_ID, TinyTransformer
from app.training import OPERATIONS, transform


@contextmanager
def zero_slice(tensor, index):
    old = tensor[index].detach().clone()
    with torch.no_grad(): tensor[index] = 0
    try: yield
    finally:
        with torch.no_grad(): tensor[index] = old


def make_task_batch(task, n, seq_len, length, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    digits = torch.randint(0, 10, (n, length), generator=g, device=device)
    x = torch.full((n, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full_like(x, PAD_ID)
    x[:, 0] = OPERATION_IDS[task]
    x[:, 1:length + 1] = digits; x[:, length + 1] = 10
    y[:, 1:length + 1] = transform(digits, task); y[:, length + 1] = 10
    return x, y


@torch.inference_mode()
def metrics(model, batches):
    exact = correct = valid_n = n = 0
    for x, y in batches:
        pred = model(x)[0].argmax(-1); valid = y.ne(PAD_ID)
        exact += int(((pred == y) | ~valid).all(-1).sum()); n += len(x)
        correct += int(((pred == y) & valid).sum()); valid_n += int(valid.sum())
    return {"exact": exact / n, "token": correct / valid_n}


def delta(base, changed):
    return {k: base[k] - changed[k] for k in base}


@torch.inference_mode()
def observe(model, batches):
    # Attention paid to operation key (position 0), and task-conditioned residual means.
    attn = [[[] for _ in range(model.cfg.n_heads)] for _ in model.blocks]
    residual = [[] for _ in model.blocks]
    mlp = [[] for _ in model.blocks]
    for x, _ in batches:
        h = model.token_embedding(x) + model.position_embedding[:, :x.shape[1]]
        mask = x.eq(PAD_ID)
        for li, block in enumerate(model.blocks):
            h, data = block(h, capture=True, padding_mask=mask)
            valid_queries = ~mask
            # Exclude op query itself; measure how much content/output positions read op.
            qmask = valid_queries.clone(); qmask[:, 0] = False
            for hi in range(model.cfg.n_heads):
                vals = data["attention"][:, hi, :, 0]
                attn[li][hi].append(vals[qmask].mean().cpu())
            residual[li].append(h[:, 1:][(~mask[:, 1:])].mean(0).cpu())
            mlp[li].append(data["mlp"][:, 1:][(~mask[:, 1:])].mean(0).cpu())
    return {
        "attention_to_operation": [[float(torch.stack(v).mean()) for v in layer] for layer in attn],
        "mean_residual": [torch.stack(v).mean(0).tolist() for v in residual],
        "mean_mlp": [torch.stack(v).mean(0).tolist() for v in mlp],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/multi_task-20260711-103650.pt")
    p.add_argument("--output", default="research/multitask_routing/results.json")
    p.add_argument("--examples", type=int, default=1024)
    p.add_argument("--length", type=int, default=7)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyTransformer(ModelConfig(**payload["model_config"])).to(device).eval()
    model.load_state_dict(payload["state_dict"])
    batches = {t: [make_task_batch(t, args.examples, model.cfg.seq_len, args.length,
                                        device, 9100)] for t in OPERATIONS}
    baseline = {t: metrics(model, b) for t, b in batches.items()}
    observations = {t: observe(model, b) for t, b in batches.items()}

    heads = []
    for li, block in enumerate(model.blocks):
        width = model.cfg.d_model // model.cfg.n_heads
        for hi in range(model.cfg.n_heads):
            cols = slice(hi * width, (hi + 1) * width)
            with zero_slice(block.attn.out_proj.weight, (slice(None), cols)):
                changed = {t: metrics(model, b) for t, b in batches.items()}
            heads.append({"layer": li, "head": hi, "ablated": changed,
                          "exact_effect": {t: baseline[t]["exact"] - changed[t]["exact"] for t in OPERATIONS}})

    components = []
    for li, block in enumerate(model.blocks):
        for name, weight in (("attention", block.attn.out_proj.weight), ("mlp", block.mlp[2].weight)):
            with zero_slice(weight, (slice(None), slice(None))):
                changed = {t: metrics(model, b) for t, b in batches.items()}
            components.append({"layer": li, "component": name, "ablated": changed,
                               "exact_effect": {t: baseline[t]["exact"] - changed[t]["exact"] for t in OPERATIONS}})

    # Change only the operation token. Score resulting predictions against every task target.
    swaps = {}
    for source in OPERATIONS:
        x, _ = batches[source][0]
        swaps[source] = {}
        for prefix in OPERATIONS:
            z = x.clone(); z[:, 0] = OPERATION_IDS[prefix]
            scores = {}
            digits = x[:, 1:args.length + 1]
            for target in OPERATIONS:
                y = torch.full_like(x, PAD_ID); y[:, 1:args.length + 1] = transform(digits, target); y[:, args.length + 1] = 10
                scores[target] = metrics(model, [(z, y)])["exact"]
            swaps[source][prefix] = scores

    # Euclidean task separation of mean activations, normalized by sqrt(width).
    separations = {"residual": [], "mlp": []}
    for kind, key in (("residual", "mean_residual"), ("mlp", "mean_mlp")):
        for li in range(model.cfg.n_layers):
            layer = {}
            for i, a in enumerate(OPERATIONS):
                for b in OPERATIONS[i+1:]:
                    va = torch.tensor(observations[a][key][li]); vb = torch.tensor(observations[b][key][li])
                    layer[f"{a}__{b}"] = float((va-vb).norm() / va.numel() ** .5)
            separations[kind].append(layer)

    result = {"metadata": {"checkpoint": args.checkpoint, "device": str(device),
              "examples_per_task": args.examples, "fixed_digit_length": args.length,
              "parameters": sum(p.numel() for p in model.parameters())},
              "baseline": baseline, "observations": observations,
              "activation_separations": separations, "head_ablations": heads,
              "component_ablations": components, "operation_token_swaps": swaps}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__": main()
