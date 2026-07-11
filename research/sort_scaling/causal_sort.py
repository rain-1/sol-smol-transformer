"""Causal decoder-only key-value pair sort, so value-carry can use an induction head.

The bidirectional in-place encoder (train.py) cannot learn to carry values through a
sort: it collapses to counting the keys and leaves values at chance. Here we reframe
sort as autoregressive generation. The sequence is

    k0 v0 k1 v1 ... k_{n-1} v_{n-1}  SEP  sk_0 sv_0 sk_1 sv_1 ... sk_{n-1} sv_{n-1}  EOS

with a causal mask and loss only on the output region (after SEP). Emitting the
value sv_q after the sorted key sk_q is exactly an induction copy ("find sk_q in the
input, emit the token after it"), the circuit causal transformers learn most readily.

Keys are distinct per sequence (sampled without replacement from KEY_ALPH) so the
value routes by a unique key identity. Values come from a disjoint alphabet.

Usage:
  python research/sort_scaling/causal_sort.py --config 64 4 3 128 --max-len 12 --steps 8000 --save
  python research/sort_scaling/causal_sort.py --search --max-len 12 --steps 6000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"

KEY_ALPH = 50
VAL_ALPH = 20
SEP = KEY_ALPH + VAL_ALPH          # separator between input and output
EOS = KEY_ALPH + VAL_ALPH + 1      # end of output
PAD = KEY_ALPH + VAL_ALPH + 2
VOCAB = KEY_ALPH + VAL_ALPH + 3


# --------------------------------------------------------------------------- #
# Model: minimal causal decoder with attention capture.
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask, key_padding_mask, capture=False):
        normed = self.ln1(x)
        out, w = self.attn(normed, normed, normed, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=capture, average_attn_weights=False)
        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x, w


class CausalTransformer(nn.Module):
    def __init__(self, seq_len, d_model=64, n_heads=4, n_layers=3, d_ff=128):
        super().__init__()
        self.cfg = dict(seq_len=seq_len, vocab_size=VOCAB, d_model=d_model,
                        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        self.token_embedding = nn.Embedding(VOCAB, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB)

    def forward(self, tokens, capture=False):
        L = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding[:, :L]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
        kpm = tokens.eq(PAD)
        internals = []
        for block in self.blocks:
            x, w = block(x, causal, kpm, capture=capture)
            if capture:
                internals.append(w)
        return self.output(self.final_norm(x)), internals


# --------------------------------------------------------------------------- #
# Data.
# --------------------------------------------------------------------------- #
def make_batch(batch_size, max_pairs, device, fixed_len=None):
    seq_len = 4 * max_pairs + 2
    seq = torch.full((batch_size, seq_len), PAD, dtype=torch.long, device=device)
    score = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    if fixed_len is not None:
        counts = torch.full((batch_size,), fixed_len, device=device)
    else:
        counts = torch.randint(2, max_pairs + 1, (batch_size,), device=device)
    for p in range(2, max_pairs + 1):
        rows = counts.eq(p)
        c = int(rows.sum())
        if not c:
            continue
        keys = torch.rand(c, KEY_ALPH, device=device).argsort(dim=1)[:, :p]
        vals = torch.randint(0, VAL_ALPH, (c, p), device=device) + KEY_ALPH
        order = keys.argsort(stable=True, dim=1)
        skeys, svals = keys.gather(1, order), vals.gather(1, order)
        idx = torch.nonzero(rows, as_tuple=True)[0]
        # Interleave input pairs, SEP, interleaved sorted pairs, EOS.
        row = torch.empty(c, seq_len, dtype=torch.long, device=device).fill_(PAD)
        row[:, 0:2 * p:2] = keys
        row[:, 1:2 * p:2] = vals
        row[:, 2 * p] = SEP
        out_start = 2 * p + 1
        row[:, out_start:out_start + 2 * p:2] = skeys
        row[:, out_start + 1:out_start + 2 * p:2] = svals
        row[:, out_start + 2 * p] = EOS
        seq[idx] = row
        # Score the labels in the output region: predicting sk_0..EOS.
        score[idx, 2 * p:out_start + 2 * p] = True
    return seq, score


@torch.inference_mode()
def evaluate_teacher_forced(model, max_pairs, device, batches=10, batch_size=256):
    model.eval()
    tok_correct = tok_total = 0
    for _ in range(batches):
        seq, score = make_batch(batch_size, max_pairs, device)
        logits, _ = model(seq[:, :-1])
        pred = logits.argmax(-1)
        target = seq[:, 1:]
        m = score[:, 1:]
        tok_correct += ((pred == target) & m).sum().item()
        tok_total += m.sum().item()
    return tok_correct / tok_total


@torch.inference_mode()
def generate_exact(model, max_pairs, device, batch_size=256, fixed_len=None):
    """Greedy autoregressive decode from the SEP boundary; exact-match the output.

    Fixed-length rows share the SEP index, so decoding is fully vectorised. Mixed
    lengths are averaged over the per-length results.
    """
    model.eval()
    if fixed_len is None:
        vals = [generate_exact(model, max_pairs, device, batch_size, fl)
                for fl in range(2, max_pairs + 1)]
        return sum(vals) / len(vals)
    p = fixed_len
    seq, _ = make_batch(batch_size, max_pairs, device, fixed_len=p)
    sep = 2 * p                 # SEP index for this length
    pair_len = 2 * p            # sorted pairs, excluding the terminal EOS marker
    work = seq.clone()
    work[:, sep + 1:] = PAD     # hide the gold output
    for t in range(pair_len):
        logits, _ = model(work)
        nxt = logits[:, sep + t, :].argmax(-1)  # token predicted after position sep+t
        if sep + 1 + t < work.shape[1]:
            work[:, sep + 1 + t] = nxt
    gold = seq[:, sep + 1:sep + 1 + pair_len]
    got = work[:, sep + 1:sep + 1 + pair_len]
    return (got == gold).all(1).float().mean().item()


def train(cfg, *, max_pairs, steps, batch_size, lr, seed, device, log_every=0):
    torch.manual_seed(seed)
    seq_len = 4 * max_pairs + 2
    model = CausalTransformer(seq_len, **cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        seq, score = make_batch(batch_size, max_pairs, device)
        logits, _ = model(seq[:, :-1])
        target = seq[:, 1:]
        m = score[:, 1:]
        loss = loss_fn(logits[m], target[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            tf = evaluate_teacher_forced(model, max_pairs, device, batches=4)
            print(f"  step {step:5d} loss {loss.item():.4f} tf_token {tf:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return model, time.perf_counter() - t0


SEARCH = [
    dict(d_model=48, n_heads=2, n_layers=2, d_ff=96),
    dict(d_model=48, n_heads=4, n_layers=3, d_ff=96),
    dict(d_model=64, n_heads=4, n_layers=2, d_ff=128),
    dict(d_model=64, n_heads=4, n_layers=3, d_ff=128),
    dict(d_model=64, n_heads=4, n_layers=4, d_ff=128),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search", action="store_true")
    p.add_argument("--config", nargs=4, type=int, metavar=("D", "H", "L", "FF"))
    p.add_argument("--max-len", type=int, default=12)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", action="store_true")
    p.add_argument("--out", default=str(HERE / "search_causal.json"))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} causal pair-sort max_pairs={a.max_len} steps={a.steps}", flush=True)

    if a.search:
        rows = json.loads(Path(a.out).read_text())["runs"] if Path(a.out).exists() else []
        done = {(r["d_model"], r["n_heads"], r["n_layers"], r["d_ff"]) for r in rows}
        for cfg in SEARCH:
            if (cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["d_ff"]) in done:
                continue
            model, secs = train(cfg, max_pairs=a.max_len, steps=a.steps,
                                 batch_size=a.batch_size, lr=a.lr, seed=a.seed, device=device)
            tf = evaluate_teacher_forced(model, a.max_len, device)
            gen = generate_exact(model, a.max_len, device)
            row = {**cfg, "parameters": sum(x.numel() for x in model.parameters()),
                   "seconds": round(secs, 1), "tf_token": round(tf, 5), "gen_exact": round(gen, 5)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(a.out).write_text(json.dumps({"max_pairs": a.max_len, "steps": a.steps,
                                               "runs": rows}, indent=2))
        best = max(rows, key=lambda r: r["gen_exact"])
        print("\nbest:", json.dumps(best))
        return

    cfg = dict(d_model=a.config[0], n_heads=a.config[1], n_layers=a.config[2], d_ff=a.config[3])
    model, secs = train(cfg, max_pairs=a.max_len, steps=a.steps, batch_size=a.batch_size,
                        lr=a.lr, seed=a.seed, device=device, log_every=max(1, a.steps // 20))
    tf = evaluate_teacher_forced(model, a.max_len, device, batches=20)
    gen = generate_exact(model, a.max_len, device, batch_size=512)
    bylen = {n: generate_exact(model, a.max_len, device, batch_size=256, fixed_len=n)
             for n in range(2, a.max_len + 1)}
    print(f"\nfinal: tf_token {tf:.5f} gen_exact {gen:.5f} ({secs:.1f}s)")
    print("gen exact by length:", {k: round(v, 3) for k, v in bylen.items()})
    if a.save:
        CKPT.mkdir(parents=True, exist_ok=True)
        path = CKPT / f"causal_pairs_L{a.max_len}.pt"
        torch.save({"model_config": model.cfg, "state_dict": model.state_dict(),
                    "key_alph": KEY_ALPH, "val_alph": VAL_ALPH, "max_pairs": a.max_len,
                    "tf_token": tf, "gen_exact": gen, "gen_by_length": bylen}, path)
        print("saved", path)


if __name__ == "__main__":
    main()
