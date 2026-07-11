"""Benchmark small architectures and print the accuracy/size/speed Pareto frontier."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import MULTITASK_VOCAB, PAD_ID, ModelConfig, TinyTransformer
from app.training import OPERATIONS, make_batch


def evaluate(model, operation, max_len, device, batches=20, batch_size=512):
    correct = tokens = exact = examples = 0
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            x, y = make_batch(batch_size, max_len, operation, device)
            pred = model(x)[0].argmax(-1); valid = y.ne(PAD_ID)
            correct += ((pred == y) & valid).sum().item(); tokens += valid.sum().item()
            exact += (((pred == y) | ~valid).all(-1)).sum().item(); examples += len(x)
    return correct / tokens, exact / examples


def benchmark(operation, cfg, args, device):
    torch.manual_seed(args.seed)
    vocab_size = len(MULTITASK_VOCAB) if operation == "translate" else 12
    model = TinyTransformer(ModelConfig(seq_len=args.max_len, vocab_size=vocab_size, **cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    start = time.perf_counter(); model.train()
    for _ in range(args.steps):
        x, y = make_batch(args.batch_size, args.max_len, operation, device)
        logits, _ = model(x); loss = loss_fn(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type == "cuda": torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    token_accuracy, exact_accuracy = evaluate(model, operation, args.max_len, device)
    return {"operation": operation, **cfg, "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": round(seconds, 3), "token_accuracy": round(token_accuracy, 5),
            "exact_accuracy": round(exact_accuracy, 5)}


def frontier(rows):
    # Maximise exact accuracy while minimising parameters and training seconds.
    return [a for a in rows if not any(
        b["parameters"] <= a["parameters"] and b["seconds"] <= a["seconds"] and
        b["exact_accuracy"] >= a["exact_accuracy"] and
        (b["parameters"] < a["parameters"] or b["seconds"] < a["seconds"] or
         b["exact_accuracy"] > a["exact_accuracy"])
        for b in rows)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-len", type=int, default=9); p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--output", default="pareto-results.json")
    p.add_argument("--resume", action="store_true", help="Keep completed matching runs from the output file")
    args = p.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [{"d_model": d, "n_heads": h, "n_layers": l, "d_ff": f} for d, h, l, f in (
        (8, 1, 1, 8), (8, 2, 1, 16), (8, 2, 2, 16),
        (12, 1, 1, 12), (12, 2, 1, 24), (12, 2, 2, 24),
        (16, 1, 1, 16), (16, 2, 1, 32), (16, 2, 2, 32),
        (12, 1, 2, 24), (12, 2, 2, 16), (12, 1, 3, 24),
        (16, 1, 2, 32), (16, 2, 2, 16), (16, 1, 3, 16),
        (24, 2, 1, 48), (24, 4, 1, 48), (24, 4, 2, 48),
        (32, 4, 1, 64), (32, 4, 2, 64),
    )]
    rows = []
    if args.resume and Path(args.output).exists():
        rows = json.loads(Path(args.output).read_text()).get("runs", [])
    completed = {(r["operation"], r["d_model"], r["n_heads"], r["n_layers"], r["d_ff"])
                 for r in rows}
    for operation in args.operations:
        for cfg in configs:
            key = (operation, cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["d_ff"])
            if key in completed:
                continue
            row = benchmark(operation, cfg, args, device); rows.append(row)
            print(json.dumps(row), flush=True)
    result = {"device": str(device), "settings": vars(args), "runs": rows,
              "frontier": {op: frontier([r for r in rows if r["operation"] == op]) for op in args.operations}}
    with open(args.output, "w") as f: json.dump(result, f, indent=2)
    print("\nPareto frontier\n" + json.dumps(result["frontier"], indent=2))


if __name__ == "__main__": main()
