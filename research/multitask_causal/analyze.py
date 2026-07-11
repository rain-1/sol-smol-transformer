"""Causal analysis of operation control in the trained multi-task transformer.

Uses a fixed paired set: every digit string is evaluated under every operation.
The script intentionally reimplements the short forward pass so residual-stream
patches and individual attention-head ablations are exact and reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model import (MULTITASK_VOCAB, OPERATION_IDS, PAD_ID, ModelConfig,
                       TinyTransformer, VOCAB)
from app.training import OPERATIONS, transform

DEFAULT_CHECKPOINT = ROOT / "checkpoints/multi_task-20260711-103650.pt"


def load_model(path: Path, device: torch.device):
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def dataset(n: int, seq_len: int, device: torch.device):
    rng = random.Random(73031)
    strings = []
    # Stratified lengths, with duplicates allowed only in the random generator's
    # vanishingly unlikely collisions. Same strings are paired across all tasks.
    for i in range(n):
        length = 2 + i % (seq_len - 3)  # 2..seq_len-2 digits (op and '=' use two slots)
        strings.append("".join(str(rng.randrange(10)) for _ in range(length)))
    x, ys, valid = {}, {}, {}
    for op in OPERATIONS:
        rows, targets, masks = [], [], []
        for s in strings:
            inp = [OPERATION_IDS[op]] + [VOCAB.index(c) for c in s + "="]
            out = {"copy": s, "reverse": s[::-1], "sort": "".join(sorted(s)),
                   "rotate_left": s[1:] + s[:1]}[op]
            target = [PAD_ID] + [VOCAB.index(c) for c in out + "="]
            pad = seq_len - len(inp)
            rows.append(inp + [PAD_ID] * pad)
            targets.append(target + [PAD_ID] * pad)
            masks.append([False] + [True] * (len(inp) - 1) + [False] * pad)
        x[op] = torch.tensor(rows, device=device)
        ys[op] = torch.tensor(targets, device=device)
        valid[op] = torch.tensor(masks, device=device)
    return strings, x, ys, valid


def manual_forward(model, tokens, patch=None, ablate_head=None, ablate_mlp=None):
    """patch maps residual stage (0=embedding, 1/2=block outputs) to tensor/mask."""
    mask = tokens.eq(PAD_ID)
    x = model.token_embedding(tokens) + model.position_embedding[:, :tokens.shape[1]]
    residuals = [x.clone()]
    if patch and 0 in patch:
        donor, positions = patch[0]
        x = torch.where(positions.unsqueeze(-1), donor, x)
    for layer, block in enumerate(model.blocks):
        qkv = F.linear(block.ln1(x), block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        b, t, d = q.shape
        h, dh = block.attn.num_heads, d // block.attn.num_heads
        q, k, v = [z.view(b, t, h, dh).transpose(1, 2) for z in (q, k, v)]
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)
        scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
        heads = scores.softmax(-1) @ v
        if ablate_head and ablate_head[0] == layer:
            heads[:, ablate_head[1]] = 0
        attn = heads.transpose(1, 2).reshape(b, t, d)
        attn = F.linear(attn, block.attn.out_proj.weight, block.attn.out_proj.bias)
        x = x + attn
        mlp = block.mlp[2](block.mlp[1](block.mlp[0](block.ln2(x))))
        if ablate_mlp == layer:
            mlp = torch.zeros_like(mlp)
        x = x + mlp
        residuals.append(x.clone())
        stage = layer + 1
        if patch and stage in patch:
            donor, positions = patch[stage]
            x = torch.where(positions.unsqueeze(-1), donor, x)
    return model.output(model.final_norm(x)), residuals


def metrics(logits, target, valid):
    pred = logits.argmax(-1)
    good = pred.eq(target)
    exact = (good | ~valid).all(-1).float().mean().item()
    token = good[valid].float().mean().item()
    return {"exact": round(exact, 6), "token": round(token, 6)}


@torch.inference_mode()
def run(model, n, device):
    strings, xs, ys, valid = dataset(n, model.cfg.seq_len, device)
    baseline, clean_residuals = {}, {}
    for op in OPERATIONS:
        logits, res = manual_forward(model, xs[op])
        baseline[op] = metrics(logits, ys[op], valid[op])
        clean_residuals[op] = res

    # Actual token replacements. Report correctness under original and swapped-to
    # operation: the latter distinguishes task rerouting from generic breakage.
    swaps = {}
    for src in OPERATIONS:
        swaps[src] = {}
        for dst in OPERATIONS:
            z = xs[src].clone(); z[:, 0] = OPERATION_IDS[dst]
            logits, _ = manual_forward(model, z)
            swaps[src][dst] = {
                "source_target": metrics(logits, ys[src], valid[src]),
                "destination_target": metrics(logits, ys[dst], valid[dst]),
            }

    corruptions = {}
    for op in OPERATIONS:
        corruptions[op] = {}
        for label, token in {"pad": PAD_ID, "equals": VOCAB.index("="), "digit_0": 0}.items():
            z = xs[op].clone(); z[:, 0] = token
            logits, _ = manual_forward(model, z)
            corruptions[op][label] = metrics(logits, ys[op], valid[op])

    # Residual activation patching from destination-task run into a source-task
    # run on identical strings. Patch op only, digits only, or all positions.
    patching = {}
    for src in OPERATIONS:
        patching[src] = {}
        for dst in OPERATIONS:
            if src == dst: continue
            patching[src][dst] = {}
            for stage in range(model.cfg.n_layers + 1):
                donor = clean_residuals[dst][stage]
                modes = {
                    "operation_only": torch.zeros_like(valid[src]).bool(),
                    "nonoperation": valid[src].clone(),
                    "all_real": valid[src].clone(),
                }
                modes["operation_only"][:, 0] = True
                modes["all_real"][:, 0] = True
                patching[src][dst][str(stage)] = {}
                for mode, pos in modes.items():
                    logits, _ = manual_forward(model, xs[src], patch={stage: (donor, pos)})
                    patching[src][dst][str(stage)][mode] = {
                        "source_target": metrics(logits, ys[src], valid[src]),
                        "destination_target": metrics(logits, ys[dst], valid[dst]),
                    }

    components = {op: {"heads": {}, "mlps": {}} for op in OPERATIONS}
    for op in OPERATIONS:
        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                logits, _ = manual_forward(model, xs[op], ablate_head=(layer, head))
                components[op]["heads"][f"L{layer}H{head}"] = metrics(logits, ys[op], valid[op])
            logits, _ = manual_forward(model, xs[op], ablate_mlp=layer)
            components[op]["mlps"][f"L{layer}"] = metrics(logits, ys[op], valid[op])

    # Continuous counterfactual: interpolate only the operation embedding between
    # two tasks. The winning task is measured by exact accuracy against all tasks.
    interpolation = {}
    emb = model.token_embedding.weight
    for src, dst in [(a, b) for i, a in enumerate(OPERATIONS) for b in OPERATIONS[i+1:]]:
        interpolation[f"{src}->{dst}"] = {}
        for alpha in [0, .25, .5, .75, 1]:
            base = clean_residuals[src][0].clone()
            mixed = ((1-alpha) * emb[OPERATION_IDS[src]] + alpha * emb[OPERATION_IDS[dst]]
                     + model.position_embedding[0, 0])
            base[:, 0] = mixed
            pos = torch.zeros_like(valid[src]); pos[:, 0] = True
            logits, _ = manual_forward(model, xs[src], patch={0: (base, pos)})
            interpolation[f"{src}->{dst}"][str(alpha)] = {
                op: metrics(logits, ys[op], valid[op])["exact"] for op in OPERATIONS
            }

    return {"metadata": {"examples": n, "seed": 73031,
                         "lengths": [2, model.cfg.seq_len - 2], "paired": True},
            "baseline": baseline, "token_swaps": swaps, "corruptions": corruptions,
            "residual_patching": patching, "component_ablations": components,
            "embedding_interpolation": interpolation}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--examples", type=int, default=512)
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, blob = load_model(args.checkpoint, device)
    result = run(model, args.examples, device)
    result["metadata"].update({"checkpoint": str(args.checkpoint), "device": str(device),
                               "model_config": blob["model_config"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["baseline"], indent=2))


if __name__ == "__main__":
    main()
