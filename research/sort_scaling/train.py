"""Scale a sort-only transformer to long strings over a 20-symbol alphabet.

Self-contained: uses app.model.TinyTransformer but its own vocab and batcher so
nothing in app/ or the web UI is touched. Two tasks:

  values (--task values): sort a string of value tokens 0..A-1. '='=A, pad=A+1
    (vocab = A+2). A bounded alphabet lets the model win by *counting* rather
    than routing, so attention need not be a permutation.

  pairs  (--task pairs, default): sort (key,value) pairs by key with the value
    carried along. Input interleaves k0 v0 k1 v1 ... =, output is the same
    interleaving after a stable sort by key. Key ids are 0..K-1 and value ids are
    K..K+V-1 (disjoint), '='=K+V, pad=K+V+1. Because each carried value is
    independent of the keys, the model cannot histogram it -- it must gather the
    value from its source pair, forcing a genuine routing circuit.

Usage:
  python research/sort_scaling/train.py --search              # architecture sweep
  python research/sort_scaling/train.py --config 32 2 2 64 --steps 8000 --save
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.model import ModelConfig, TinyTransformer  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"


# Pair-sort: keys are sampled WITHOUT replacement from a 50-symbol alphabet so
# there are no tie ambiguities -- each carried value routes by a unique key
# identity (associative recall) rather than through rank/counting indirection,
# which a 20-symbol tied-key version failed to learn at any depth. Values come
# from a disjoint 20-symbol alphabet (ties among values are fine: they are
# distinguished by which pair carries them).
KEY_ALPH = 50
VAL_ALPH = 20


def vocab_size_for(task, alphabet):
    if task == "pairs":
        return KEY_ALPH + VAL_ALPH + 2  # keys, values, '=', pad
    return alphabet + 2                 # values, '=', pad


def seq_len_for(task, max_len):
    """max_len is the payload length (values) or pair count (pairs), plus '='."""
    return (2 * max_len + 1) if task == "pairs" else (max_len + 1)


def make_pairsort_batch(batch_size, max_pairs, device, fixed_len=None):
    """Interleaved (key,value) pairs, target = stable sort by key with value carried.

    Token layout per row: k0 v0 k1 v1 ... k_{p-1} v_{p-1} '=', right-padded.
    Value ids are offset by KEY_ALPH so keys and values are distinct tokens.
    """
    seq_len = 2 * max_pairs + 1
    eq_id, pad_id = KEY_ALPH + VAL_ALPH, KEY_ALPH + VAL_ALPH + 1
    x = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    y = torch.full_like(x, pad_id)
    if fixed_len is not None:
        counts = torch.full((batch_size,), fixed_len, device=device)
    else:
        counts = torch.randint(2, max_pairs + 1, (batch_size,), device=device)
    for p in range(2, max_pairs + 1):
        rows = counts.eq(p)
        c = int(rows.sum())
        if not c:
            continue
        # Distinct keys per row: rank random noise and take the first p columns.
        keys = torch.rand(c, KEY_ALPH, device=device).argsort(dim=1)[:, :p]
        vals = torch.randint(0, VAL_ALPH, (c, p), device=device) + KEY_ALPH
        order = keys.argsort(stable=True, dim=1)
        skeys, svals = keys.gather(1, order), vals.gather(1, order)
        L = 2 * p
        idx = torch.nonzero(rows, as_tuple=True)[0]
        x[idx[:, None], torch.arange(0, L, 2, device=device)] = keys
        x[idx[:, None], torch.arange(1, L, 2, device=device)] = vals
        x[idx, L] = eq_id
        y[idx[:, None], torch.arange(0, L, 2, device=device)] = skeys
        y[idx[:, None], torch.arange(1, L, 2, device=device)] = svals
        y[idx, L] = eq_id
    return x, y, pad_id


def make_batch(task, batch_size, max_len, alphabet, device, fixed_len=None):
    if task == "pairs":
        return make_pairsort_batch(batch_size, max_len, device, fixed_len)
    return make_sort_batch(batch_size, max_len + 1, alphabet, device, fixed_len)


def make_sort_batch(batch_size, max_len, alphabet, device, fixed_len=None):
    """Random value strings ending in '=' (id=alphabet), right-padded with id=alphabet+1.

    Target is the sorted payload (ascending), '=' preserved at its position.
    """
    eq_id, pad_id = alphabet, alphabet + 1
    x = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full_like(x, pad_id)
    if fixed_len is not None:
        lengths = torch.full((batch_size,), fixed_len, device=device)
    else:
        lengths = torch.randint(2, max_len, (batch_size,), device=device)
    for length in range(2, max_len):
        rows = lengths.eq(length)
        count = int(rows.sum())
        if not count:
            continue
        payload = torch.randint(0, alphabet, (count, length), device=device)
        x[rows, :length], x[rows, length] = payload, eq_id
        y[rows, :length], y[rows, length] = payload.sort(1).values, eq_id
    return x, y, pad_id


@torch.inference_mode()
def evaluate(model, task, max_len, alphabet, device, batches=20, batch_size=512):
    model.eval()
    correct = tokens = exact = examples = 0
    for _ in range(batches):
        x, y, pad_id = make_batch(task, batch_size, max_len, alphabet, device)
        pred = model(x)[0].argmax(-1)
        valid = y.ne(pad_id)
        correct += ((pred == y) & valid).sum().item()
        tokens += valid.sum().item()
        exact += (((pred == y) | ~valid).all(-1)).sum().item()
        examples += len(x)
    return correct / tokens, exact / examples


@torch.inference_mode()
def exact_by_length(model, task, max_len, alphabet, device, batch_size=512):
    model.eval()
    out = {}
    for n in range(2, max_len + 1):
        x, y, pad_id = make_batch(task, batch_size, max_len, alphabet, device, fixed_len=n)
        pred = model(x)[0].argmax(-1)
        valid = y.ne(pad_id)
        out[n] = (((pred == y) | ~valid).all(-1)).float().mean().item()
    return out


def train(cfg, *, task, max_len, alphabet, steps, batch_size, lr, seed, device, log_every=0):
    random.seed(seed)
    torch.manual_seed(seed)
    model_cfg = ModelConfig(seq_len=seq_len_for(task, max_len),
                            vocab_size=vocab_size_for(task, alphabet), **cfg)
    model = TinyTransformer(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model_cfg.vocab_size - 1)
    model.train()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y, _ = make_batch(task, batch_size, max_len, alphabet, device)
        logits, _ = model(x)
        loss = loss_fn(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            tok, ex = evaluate(model, task, max_len, alphabet, device, batches=4)
            print(f"  step {step:5d} loss {loss.item():.4f} token {tok:.4f} exact {ex:.4f}",
                  flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return model, seconds


SEARCH_CONFIGS = [
    dict(d_model=16, n_heads=1, n_layers=1, d_ff=32),
    dict(d_model=16, n_heads=2, n_layers=1, d_ff=32),
    dict(d_model=24, n_heads=2, n_layers=1, d_ff=48),
    dict(d_model=32, n_heads=1, n_layers=1, d_ff=64),
    dict(d_model=32, n_heads=2, n_layers=1, d_ff=64),
    dict(d_model=32, n_heads=4, n_layers=1, d_ff=64),
    dict(d_model=32, n_heads=2, n_layers=2, d_ff=64),
    dict(d_model=32, n_heads=4, n_layers=2, d_ff=64),
    dict(d_model=48, n_heads=4, n_layers=1, d_ff=96),
    dict(d_model=48, n_heads=4, n_layers=2, d_ff=96),
    dict(d_model=64, n_heads=4, n_layers=2, d_ff=128),
    # Depth probe: does length-30 sort keep improving with more layers?
    dict(d_model=32, n_heads=2, n_layers=3, d_ff=64),
    dict(d_model=32, n_heads=2, n_layers=4, d_ff=64),
    dict(d_model=32, n_heads=4, n_layers=3, d_ff=64),
    dict(d_model=48, n_heads=4, n_layers=3, d_ff=96),
    dict(d_model=48, n_heads=4, n_layers=4, d_ff=96),
]


# Pair-sort forces routing, so it is harder; this ladder centres on depth.
PAIR_CONFIGS = [
    dict(d_model=32, n_heads=2, n_layers=1, d_ff=64),
    dict(d_model=32, n_heads=2, n_layers=2, d_ff=64),
    dict(d_model=32, n_heads=2, n_layers=3, d_ff=64),
    dict(d_model=32, n_heads=2, n_layers=4, d_ff=64),
    dict(d_model=48, n_heads=2, n_layers=2, d_ff=96),
    dict(d_model=48, n_heads=2, n_layers=3, d_ff=96),
    dict(d_model=48, n_heads=2, n_layers=4, d_ff=96),
    dict(d_model=64, n_heads=4, n_layers=3, d_ff=128),
    dict(d_model=64, n_heads=4, n_layers=4, d_ff=128),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("pairs", "values"), default="pairs")
    p.add_argument("--search", action="store_true")
    p.add_argument("--config", nargs=4, type=int, metavar=("D", "H", "L", "FF"))
    p.add_argument("--max-len", type=int, default=30,
                   help="payload length (values) or number of pairs (pairs)")
    p.add_argument("--alphabet", type=int, default=20)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", action="store_true")
    p.add_argument("--out", default=str(HERE / "search.json"))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} task={a.task} max_len={a.max_len} alphabet={a.alphabet} "
          f"steps={a.steps}", flush=True)

    if a.search:
        rows = []
        if Path(a.out).exists():
            rows = json.loads(Path(a.out).read_text()).get("runs", [])
        done = {(r["d_model"], r["n_heads"], r["n_layers"], r["d_ff"]) for r in rows}
        config_list = PAIR_CONFIGS if a.task == "pairs" else SEARCH_CONFIGS
        for cfg in config_list:
            key = (cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["d_ff"])
            if key in done:
                continue
            model, seconds = train(cfg, task=a.task, max_len=a.max_len, alphabet=a.alphabet,
                                    steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                                    seed=a.seed, device=device)
            tok, ex = evaluate(model, a.task, a.max_len, a.alphabet, device)
            row = {**cfg, "parameters": sum(p.numel() for p in model.parameters()),
                   "seconds": round(seconds, 1), "token_accuracy": round(tok, 5),
                   "exact_accuracy": round(ex, 5)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(a.out).write_text(json.dumps(
                {"device": str(device), "task": a.task, "max_len": a.max_len,
                 "alphabet": a.alphabet, "steps": a.steps, "runs": rows}, indent=2))
        best = max(rows, key=lambda r: (r["exact_accuracy"], -r["parameters"]))
        print("\nbest:", json.dumps(best))
        return

    cfg = dict(d_model=a.config[0], n_heads=a.config[1], n_layers=a.config[2], d_ff=a.config[3])
    model, seconds = train(cfg, task=a.task, max_len=a.max_len, alphabet=a.alphabet,
                           steps=a.steps, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                           device=device, log_every=max(1, a.steps // 20))
    tok, ex = evaluate(model, a.task, a.max_len, a.alphabet, device, batches=40)
    bylen = exact_by_length(model, a.task, a.max_len, a.alphabet, device)
    print(f"\nfinal: token {tok:.5f} exact {ex:.5f} ({seconds:.1f}s)")
    print("exact by length:", {k: round(v, 3) for k, v in bylen.items()})
    if a.save:
        CKPT.mkdir(parents=True, exist_ok=True)
        path = CKPT / f"{a.task}_a{a.alphabet}_L{a.max_len}.pt"
        torch.save({"model_config": model.cfg.dict(), "state_dict": model.state_dict(),
                    "task": a.task, "alphabet": a.alphabet, "max_len": a.max_len,
                    "key_alph": KEY_ALPH, "val_alph": VAL_ALPH,
                    "token_accuracy": tok, "exact_accuracy": ex,
                    "exact_by_length": bylen}, path)
        print("saved", path)


if __name__ == "__main__":
    main()
