"""Reproducible causal interventions for the symbolic tiny transformers.

Run from the repository root:
    python research/causal/run_interventions.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.model import ModelConfig, PAD_ID, TinyTransformer, VOCAB
from app.training import make_batch


CONFIGS = {
    "copy": dict(d_model=8, n_heads=1, n_layers=1, d_ff=8),
    "reverse": dict(d_model=16, n_heads=2, n_layers=2, d_ff=16),
    "sort": dict(d_model=16, n_heads=1, n_layers=1, d_ff=16),
    "rotate_left": dict(d_model=8, n_heads=1, n_layers=1, d_ff=8),
}


@contextmanager
def zero_slice(tensor, index):
    old = tensor[index].detach().clone()
    with torch.no_grad():
        tensor[index] = 0
    try:
        yield
    finally:
        with torch.no_grad():
            tensor[index] = old


@torch.inference_mode()
def metrics(model, batches):
    exact_n = digit_exact_n = token_n = eq_n = total = correct = eq_correct = 0
    for x, y in batches:
        pred = model(x)[0].argmax(-1)
        valid = y.ne(PAD_ID)
        row_ok = ((pred == y) | ~valid).all(-1)
        digits = valid & y.ne(10)
        digit_row_ok = ((pred == y) | ~digits).all(-1)
        exact_n += int(row_ok.sum()); total += len(x)
        digit_exact_n += int(digit_row_ok.sum())
        correct += int(((pred == y) & valid).sum()); token_n += int(valid.sum())
        eq = y.eq(10); eq_correct += int(((pred == y) & eq).sum()); eq_n += int(eq.sum())
    return {"exact": exact_n / total, "digit_exact": digit_exact_n / total,
            "token": correct / token_n, "equals": eq_correct / eq_n}


def train(operation, cfg, device, steps, seed):
    random.seed(seed); torch.manual_seed(seed)
    model = TinyTransformer(ModelConfig(seq_len=9, **cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    model.train()
    for _ in range(steps):
        x, y = make_batch(256, 9, operation, device)
        logits, _ = model(x)
        loss = loss_fn(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval()


def delta(base, changed):
    return {k: base[k] - changed[k] for k in base}


def investigate(operation, cfg, args, device):
    ckpt = Path(args.output).parent / "checkpoints" / f"{operation}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists() and not args.retrain:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model = TinyTransformer(ModelConfig(seq_len=9, **cfg)).to(device)
        model.load_state_dict(state)
    else:
        model = train(operation, cfg, device, args.steps, args.seed)
        torch.save(model.state_dict(), ckpt)

    # Frozen evaluation batches make every causal comparison paired.
    torch.manual_seed(1000 + args.seed)
    batches = [make_batch(args.eval_batch, 9, operation, device) for _ in range(args.eval_batches)]
    base = metrics(model, batches)
    result = {"config": cfg, "parameters": sum(p.numel() for p in model.parameters()),
              "baseline": base, "layer_components": [], "heads": [], "neurons": []}

    for li, block in enumerate(model.blocks):
        with zero_slice(block.attn.out_proj.weight, (slice(None), slice(None))):
            m = metrics(model, batches)
        result["layer_components"].append({"layer": li, "component": "attention", "ablated": m, "effect": delta(base, m)})
        with zero_slice(block.mlp[2].weight, (slice(None), slice(None))):
            m = metrics(model, batches)
        result["layer_components"].append({"layer": li, "component": "mlp", "ablated": m, "effect": delta(base, m)})

        width = cfg["d_model"] // cfg["n_heads"]
        for hi in range(cfg["n_heads"]):
            cols = slice(hi * width, (hi + 1) * width)
            norm = float(block.attn.out_proj.weight[:, cols].norm())
            with zero_slice(block.attn.out_proj.weight, (slice(None), cols)):
                m = metrics(model, batches)
            result["heads"].append({"layer": li, "head": hi, "output_weight_norm": norm,
                                    "ablated": m, "effect": delta(base, m)})
        for ni in range(cfg["d_ff"]):
            norm = float(block.mlp[2].weight[:, ni].norm())
            with zero_slice(block.mlp[2].weight, (slice(None), ni)):
                m = metrics(model, batches)
            result["neurons"].append({"layer": li, "neuron": ni, "output_weight_norm": norm,
                                      "ablated": m, "effect": delta(base, m)})

    # Counterfactual delimiters: replace '=' by a digit while retaining the original target.
    cf_batches = []
    for x, y in batches:
        z = x.clone(); z[z.eq(10)] = 0; cf_batches.append((z, y))
    result["equals_to_zero"] = {"ablated": metrics(model, cf_batches)}
    result["equals_to_zero"]["effect"] = delta(base, result["equals_to_zero"]["ablated"])

    # Remove learned position signal without changing tokens.
    with zero_slice(model.position_embedding, (slice(None), slice(None), slice(None))):
        m = metrics(model, batches)
    result["zero_position_embedding"] = {"ablated": m, "effect": delta(base, m)}

    # Per-length exact accuracy exposes whether effects are concentrated at long inputs.
    by_length = {}
    for length in range(2, 9):
        torch.manual_seed(2000 + args.seed + length)
        x = torch.full((args.eval_batch, 9), PAD_ID, dtype=torch.long, device=device)
        digits = torch.randint(0, 10, (args.eval_batch, length), device=device)
        x[:, :length] = digits; x[:, length] = 10
        from app.training import transform
        y = torch.full_like(x, PAD_ID); y[:, :length] = transform(digits, operation); y[:, length] = 10
        by_length[str(length)] = metrics(model, [(x, y)])
    result["by_length"] = by_length
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--output", default="research/causal/results.json")
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"metadata": {"seed": args.seed, "steps": args.steps, "eval_examples": args.eval_batches * args.eval_batch,
                            "device": str(device), "intervention": "zero ablation of output contribution"}, "tasks": {}}
    for operation, cfg in CONFIGS.items():
        print(f"Investigating {operation} on {device}", flush=True)
        results["tasks"][operation] = investigate(operation, cfg, args, device)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
