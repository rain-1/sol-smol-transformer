"""Component-by-component walkthrough of the causal pair-sort decoder.

Explains all four attention heads and both MLPs on a concrete example input, with:
  - a token table for one hand-picked example,
  - per-head attention heatmaps on that example (annotated),
  - MLP causal roles (ablate each MLP, split key- vs value-output effect),
  - MLP computational roles (probe the next key / value in the residual BEFORE vs
    AFTER each MLP -> shows the L1 MLP computes the key comparison while the value
    is already carried by the L1 attention/induction path),
  - MLP hidden-activation heatmaps over the example (position/role selectivity).

Roles are stated for the committed length-30 checkpoint (heads permute across
seeds; see reports/sort-scaling.md §7). Figures -> reports/figures/walk_*.png.

Usage: python research/sort_scaling/walkthrough.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, SEQ_BLUE, DIVERGING, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load, batch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One fixed example: 5 (key, value) pairs. Keys distinct 0-49; values 50-69.
EX_KEYS = [12, 3, 41, 27, 8]
EX_VALS = [55, 61, 52, 68, 59]


def build_example():
    p = len(EX_KEYS)
    keys = torch.tensor(EX_KEYS); vals = torch.tensor(EX_VALS)
    order = keys.argsort(stable=True)
    sk, sv = keys[order], vals[order]
    seq = []
    for k, v in zip(EX_KEYS, EX_VALS):
        seq += [k, v]
    seq += [cs.SEP]
    for k, v in zip(sk.tolist(), sv.tolist()):
        seq += [k, v]
    seq += [cs.EOS]
    return torch.tensor([seq], device=DEV), p, order


def tok_label(t):
    if t == cs.SEP:
        return "SEP"
    if t == cs.EOS:
        return "EOS"
    if t == cs.PAD:
        return "·"
    if t < cs.KEY_ALPH:
        return f"k{t}"
    return f"v{t - cs.KEY_ALPH}"


@torch.inference_mode()
def full_capture(model, tokens, mlp_off=None):
    """Manual forward capturing attention, residual before/after each MLP, and MLP
    hidden activations. mlp_off: set of layer indices whose MLP output is zeroed."""
    mlp_off = mlp_off or set()
    L = tokens.shape[1]
    x = model.token_embedding(tokens) + model.position_embedding[:, :L]
    causal = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
    kpm = tokens.eq(cs.PAD)
    caps = []
    for li, block in enumerate(model.blocks):
        normed = block.ln1(x)
        attn_out, w = block.attn(normed, normed, normed, attn_mask=causal,
                                 key_padding_mask=kpm, need_weights=True,
                                 average_attn_weights=False)
        x_mid = x + attn_out
        mlp_in = block.ln2(x_mid)
        h_pre = block.mlp[0](mlp_in)
        hidden = block.mlp[1](h_pre)          # GELU activation [B,L,d_ff]
        mlp_out = block.mlp[2](hidden)
        x_out = x_mid if li in mlp_off else x_mid + mlp_out
        caps.append(dict(attn=w, x_in=x, x_mid=x_mid, hidden=hidden, x_out=x_out))
        x = x_out
    logits = model.output(model.final_norm(x))
    return logits, caps


def bare(ax):
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/" + name)


# --------------------------------------------------------------------------- #
# 1. Per-head attention heatmaps on the example
# --------------------------------------------------------------------------- #
HEAD_ROLE = {
    "L0H0": "precursor: reads input keys at SEP;\nwrites key-identity to value slots",
    "L0H1": "precursor (key side)",
    "L1H0": "induction head: copies the matching value",
    "L1H1": "sort head: reads emitted keys (max-so-far)",
}


def fig_heads(model, tokens, p):
    _, caps = full_capture(model, tokens)
    labels = [tok_label(t) for t in tokens[0].tolist()]
    L = tokens.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    for ax, (li, h) in zip(axes.ravel(), [(0, 0), (0, 1), (1, 0), (1, 1)]):
        A = caps[li]["attn"][0, h].cpu().numpy()
        A = np.tril(A)  # causal
        im = ax.imshow(A, cmap=SEQ_BLUE, vmin=0, vmax=A.max(), aspect="equal")
        ax.set_xticks(range(L)); ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_yticks(range(L)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("attended-to (key)"); ax.set_ylabel("query position")
        name = f"L{li}H{h}"
        ax.set_title(f"{name} — {HEAD_ROLE[name]}", fontsize=10)
        # mark the SEP row/col
        sep = 2 * p
        ax.axhline(sep - 0.5, color=ORANGE, lw=0.6, alpha=0.5)
        ax.axvline(sep + 0.5, color=ORANGE, lw=0.6, alpha=0.5)
        bare(ax)
    fig.suptitle("Every head on one example:  " + " ".join(labels[:2 * p]) +
                 "  →  sort by key", fontsize=13, fontweight="bold", y=0.995)
    save(fig, "walk_heads.png")


# --------------------------------------------------------------------------- #
# 2. MLP causal roles: ablate each MLP, split key vs value output accuracy
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def per_pos_acc(model, p, mlp_off=None, n=512, seed=3):
    seq, _, _ = batch(model, p, n=n, seed=seed)
    logits, _ = full_capture(model, seq[:, :-1], mlp_off=mlp_off)
    pred = logits.argmax(-1); target = seq[:, 1:]
    sep = 2 * p
    kpos = [sep + 2 * q for q in range(p)]
    vpos = [sep + 1 + 2 * q for q in range(p)]
    return ((pred[:, kpos] == target[:, kpos]).float().mean().item(),
            (pred[:, vpos] == target[:, vpos]).float().mean().item())


def fig_mlp_ablation(model, p=12):
    base_k, base_v = per_pos_acc(model, p)
    l0_k, l0_v = per_pos_acc(model, p, mlp_off={0})
    l1_k, l1_v = per_pos_acc(model, p, mlp_off={1})
    both_k, both_v = per_pos_acc(model, p, mlp_off={0, 1})
    rows = ["remove L0 MLP", "remove L1 MLP", "remove both MLPs"]
    dk = [base_k - l0_k, base_k - l1_k, base_k - both_k]
    dv = [base_v - l0_v, base_v - l1_v, base_v - both_v]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.2, dk, 0.4, label="key-output accuracy drop", color=BLUE, zorder=3)
    ax.bar(x + 0.2, dv, 0.4, label="value-output accuracy drop", color=ORANGE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(rows); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy drop"); ax.set_title("What each MLP is causally for")
    ax.legend(fontsize=8)
    save(fig, "walk_mlp_ablation.png")
    return dict(baseline=(base_k, base_v), L0=(l0_k, l0_v), L1=(l1_k, l1_v),
                both=(both_k, both_v))


# --------------------------------------------------------------------------- #
# 3. MLP computational role: is the key/value computed BEFORE or AFTER each MLP?
# --------------------------------------------------------------------------- #
def _probe(X, y):
    n = len(X); idx = np.random.RandomState(0).permutation(n); cut = int(0.75 * n)
    mu, sd = X[idx[:cut]].mean(0), X[idx[:cut]].std(0) + 1e-6
    Xs = (X - mu) / sd
    clf = LogisticRegression(max_iter=1500, C=1.0).fit(Xs[idx[:cut]], y[idx[:cut]])
    return float(clf.score(Xs[idx[cut:]], y[idx[cut:]]))


@torch.inference_mode()
def fig_mlp_probe(model, p=12, n=512):
    seq, keys, order = batch(model, p, n=n, seed=5)
    _, caps = full_capture(model, seq[:, :-1])
    sep = 2 * p
    kq = [sep + 2 * q for q in range(p)]       # emits sk_q
    vq = [sep + 1 + 2 * q for q in range(p)]   # emits sv_q
    next_key = seq[:, [sep + 1 + 2 * q for q in range(p)]]   # sk_q token
    next_val = seq[:, [sep + 2 + 2 * q for q in range(p)]]   # sv_q token
    # residual stream stages around the two MLPs
    stages = [
        ("embed", caps[0]["x_in"]),
        ("after L0 attn", caps[0]["x_mid"]),
        ("after L0 MLP", caps[0]["x_out"]),
        ("after L1 attn", caps[1]["x_mid"]),
        ("after L1 MLP", caps[1]["x_out"]),
    ]
    key_acc, val_acc = [], []
    for _, R in stages:
        Xk = R[:, kq].reshape(-1, R.shape[-1]).cpu().numpy()
        key_acc.append(_probe(Xk, next_key.reshape(-1).cpu().numpy()))
        Xv = R[:, vq].reshape(-1, R.shape[-1]).cpu().numpy()
        val_acc.append(_probe(Xv, next_val.reshape(-1).cpu().numpy()))
    xs = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(xs, key_acc, "-o", color=BLUE, lw=2, ms=7, label="next key decodable (at key slot)", zorder=3)
    ax.plot(xs, val_acc, "-s", color=ORANGE, lw=2, ms=7, label="value-to-emit decodable (at value slot)", zorder=3)
    for xi in (2, 4):
        ax.axvspan(xi - 0.5, xi + 0.5, color=VIOLET, alpha=0.10)
    ax.text(2, 0.5, "L0 MLP", rotation=90, va="center", ha="center", fontsize=8, color=VIOLET)
    ax.text(4, 0.5, "L1 MLP", rotation=90, va="center", ha="center", fontsize=8, color=VIOLET)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in stages], rotation=20, ha="right")
    ax.set_ylim(0, 1.03); ax.set_ylabel("linear-probe accuracy")
    ax.set_title("Both key and value become decodable at the L1 attention step —\nneither MLP is where the answer appears at the query position")
    ax.legend(fontsize=8, loc="center left")
    save(fig, "walk_mlp_probe.png")
    return dict(stages=[s[0] for s in stages], key=key_acc, val=val_acc)


# --------------------------------------------------------------------------- #
# 4. MLP hidden-activation heatmap on the example (position/role selectivity)
# --------------------------------------------------------------------------- #
def fig_mlp_neurons(model, tokens, p):
    _, caps = full_capture(model, tokens)
    labels = [tok_label(t) for t in tokens[0].tolist()]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, li in zip(axes, [0, 1]):
        H = caps[li]["hidden"][0].cpu().numpy().T   # [d_ff, L]
        vmax = np.abs(H).max()
        im = ax.imshow(H, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_ylabel(f"L{li} MLP neuron (of {H.shape[0]})")
        ax.set_xlabel("sequence position")
        ax.set_title(f"L{li} MLP hidden activations on the example")
        sep = 2 * p
        ax.axvline(sep, color=INK, lw=1, ls="--")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle("MLP neuron activations (dashed line = SEP; input pairs left, generated output right)",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "walk_mlp_neurons.png")


def main():
    apply_style()
    model, meta = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
    tokens, p, order = build_example()
    # sanity: does it sort this example correctly (teacher-forced)?
    with torch.inference_mode():
        logits, _ = full_capture(model, tokens[:, :-1])
    pred = logits.argmax(-1)[0]
    print("example tokens:", [tok_label(t) for t in tokens[0].tolist()])
    print("predicted out :", [tok_label(t) for t in pred[2 * p:].tolist()])
    fig_heads(model, tokens, p)
    abl = fig_mlp_ablation(model)
    probe = fig_mlp_probe(model)
    fig_mlp_neurons(model, tokens, p)
    (HERE / "walkthrough.json").write_text(json.dumps(
        {"example_keys": EX_KEYS, "example_values": EX_VALS,
         "mlp_ablation": abl, "mlp_probe": probe}, indent=2))
    print("mlp ablation:", json.dumps(abl))
    print("mlp probe:", json.dumps(probe))


if __name__ == "__main__":
    main()
