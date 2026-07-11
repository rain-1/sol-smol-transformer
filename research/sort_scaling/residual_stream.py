"""Activation (residual) stream analysis for the causal pair-sort decoder.

The stream is d_model = 48 dims. This measures how many of those dims are actually
used (effective dimensionality by stage), decomposes what each component writes into
the stream, maps the stream geometry (PCA by token role), and attributes the output
logits to components (direct logit attribution).

Usage: python research/sort_scaling/residual_stream.py
Figures -> reports/figures/stream_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (apply_style, SEQ_BLUE, DIVERGING, BLUE, ORANGE, VIOLET, GREEN,
                   RED, INK, INK_2)  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load, run_capture, batch  # noqa: E402
from walkthrough import full_capture, tok_label, build_example  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGES = ["embed", "block0", "block1", "final"]


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/" + name)


def bare(ax):
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# 1. Effective dimensionality by stage
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def dimensionality(model, p=12, n=1024):
    seq, _, _ = batch(model, p, n)
    _, _, resid = run_capture(model, seq[:, :-1])
    out = {}
    for st in STAGES:
        X = resid[st].reshape(-1, resid[st].shape[-1]).cpu().numpy()
        Xc = X - X.mean(0)
        ev = np.linalg.eigvalsh(Xc.T @ Xc / len(Xc))[::-1].clip(min=0)
        pr = ev.sum() ** 2 / (ev ** 2).sum()          # participation ratio
        cum = np.cumsum(ev) / ev.sum()
        out[st] = dict(participation_ratio=float(pr),
                       n90=int(np.searchsorted(cum, 0.9) + 1),
                       n95=int(np.searchsorted(cum, 0.95) + 1),
                       cum=cum.tolist())
    return out


def fig_dimensionality(model):
    d = dimensionality(model)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(STAGES))
    pr = [d[s]["participation_ratio"] for s in STAGES]
    ax1.bar(x, pr, 0.6, color=BLUE, zorder=3)
    ax1.axhline(48, ls="--", color=INK_2, lw=1)
    ax1.text(0.05, 46.5, "nominal 48 dims", fontsize=8, color=INK_2)
    ax1.set_xticks(x); ax1.set_xticklabels(STAGES)
    ax1.set_ylim(0, 50); ax1.set_ylabel("effective dimensionality (participation ratio)")
    ax1.set_title("Effective dimensionality by stage")
    for xi, v in zip(x, pr):
        ax1.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    for s, c in zip(STAGES, [BLUE, ORANGE, VIOLET, GREEN]):
        ax2.plot(range(1, 49), d[s]["cum"], "-", color=c, lw=2, label=s)
    ax2.axhline(0.9, ls=":", color=INK_2, lw=1)
    ax2.set_xlabel("number of principal components")
    ax2.set_ylabel("cumulative variance explained")
    ax2.set_title("Cumulative variance by stage")
    fig.suptitle("The 48-dim stream compresses to a ~9–12-dim working subspace",
                 fontsize=12, fontweight="bold", y=1.02)
    ax2.legend(fontsize=8)
    save(fig, "stream_dimensionality.png")
    return d


# --------------------------------------------------------------------------- #
# 2. Component-write decomposition on the example
# --------------------------------------------------------------------------- #
def component_contribs(model, tokens):
    _, caps = full_capture(model, tokens)
    comps = {
        "embed": caps[0]["x_in"],
        "L0 attn": caps[0]["x_mid"] - caps[0]["x_in"],
        "L0 MLP": caps[0]["x_out"] - caps[0]["x_mid"],
        "L1 attn": caps[1]["x_mid"] - caps[1]["x_in"],
        "L1 MLP": caps[1]["x_out"] - caps[1]["x_mid"],
    }
    total = caps[1]["x_out"]
    return comps, total


def fig_component_writes(model, tokens, p):
    labels = [tok_label(t) for t in tokens[0].tolist()]
    comps, total = component_contribs(model, tokens)
    names = list(comps)
    M = np.array([comps[k][0].norm(dim=-1).cpu().numpy() for k in names])  # [C,L]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.4),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    im = ax1.imshow(M, cmap=SEQ_BLUE, aspect="auto", vmin=0, vmax=M.max())
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=9)
    ax1.set_title("What each component writes into the stream (contribution norm by position)")
    sep = 2 * p
    ax1.axvline(sep, color=ORANGE, lw=1.5)
    for i in range(len(names)):
        for j in range(M.shape[1]):
            if M[i, j] > 0.35 * M.max():
                ax1.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                         color="white")
    bare(ax1); fig.colorbar(im, ax=ax1, fraction=0.02, pad=0.01).set_label("‖write‖", fontsize=8)
    ax2.plot(range(M.shape[1]), total[0].norm(dim=-1).cpu().numpy(), "-o", color=INK,
             ms=3, lw=1.5)
    ax2.axvline(sep, color=ORANGE, lw=1.5)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=6, rotation=90)
    ax2.set_ylabel("‖residual‖", fontsize=8)
    ax2.set_title("Total residual-stream norm (orange line = SEP)", fontsize=9)
    save(fig, "stream_component_writes.png")


# --------------------------------------------------------------------------- #
# 3. Stream geometry: PCA colored by token role and key value
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def fig_geometry(model, p=10, n=400, stage="final"):
    seq, keys, order = batch(model, p, n)
    _, _, resid = run_capture(model, seq[:, :-1])
    R = resid[stage]                                   # [B,L-1,48]
    toks = seq[:, :-1]
    B, L, D = R.shape
    X = R.reshape(-1, D).cpu().numpy()
    tk = toks.reshape(-1).cpu().numpy()
    pos = np.tile(np.arange(L), B)
    sep = 2 * p
    # roles
    role = np.full(len(tk), "other", dtype=object)
    role[(tk < cs.KEY_ALPH) & (pos < sep)] = "input key"
    role[(tk >= cs.KEY_ALPH) & (tk < cs.KEY_ALPH + cs.VAL_ALPH) & (pos < sep)] = "input value"
    role[tk == cs.SEP] = "SEP"
    role[(tk < cs.KEY_ALPH) & (pos > sep)] = "output key"
    role[(tk >= cs.KEY_ALPH) & (tk < cs.KEY_ALPH + cs.VAL_ALPH) & (pos > sep)] = "output value"
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    palette = {"input key": BLUE, "input value": ORANGE, "SEP": RED,
               "output key": VIOLET, "output value": GREEN, "other": "#cccccc"}
    for rname, col in palette.items():
        m = role == rname
        if m.sum():
            ax1.scatter(pcs[m, 0], pcs[m, 1], s=4, c=col, label=rname, alpha=0.5)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.set_title(f"Stream geometry ({stage}): token roles separate")
    ax1.legend(fontsize=7, markerscale=2)
    # color key tokens by key value
    km = (role == "input key") | (role == "output key")
    sc = ax2.scatter(pcs[km, 0], pcs[km, 1], s=6, c=tk[km], cmap="viridis", alpha=0.6)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title("Key tokens colored by key value")
    fig.colorbar(sc, ax=ax2, fraction=0.046).set_label("key id", fontsize=8)
    save(fig, "stream_geometry.png")


# --------------------------------------------------------------------------- #
# 4. Direct logit attribution at output positions
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def fig_logit_attribution(model, p=12, n=512):
    seq, _, _ = batch(model, p, n)
    comps, total = component_contribs(model, seq[:, :-1])
    ln = model.final_norm
    gamma = ln.weight.detach()
    # freeze the LN 1/std at the full residual's value; treat LN as linear per position.
    mu = total.mean(-1, keepdim=True)
    std = (total.var(-1, keepdim=True, unbiased=False) + ln.eps).sqrt()
    WU = model.output.weight.detach()                  # [vocab,48]
    sep = 2 * p
    tgt = seq[:, 1:]                                    # next-token targets
    names = list(comps)
    kpos = [sep + 2 * q for q in range(p)]             # predict sk_q
    vpos = [sep + 1 + 2 * q for q in range(p)]         # predict sv_q
    res = {"key step": {}, "value step": {}}
    for step, positions in [("key step", kpos), ("value step", vpos)]:
        for nm in names:
            c = comps[nm]
            cc = (c - c.mean(-1, keepdim=True)) / std * gamma        # LN-folded contribution
            # contribution to the correct-token logit at each position
            contrib = 0.0
            cnt = 0
            for t in positions:
                wu = WU[tgt[:, t]]                       # [B,48]
                contrib += (cc[:, t] * wu).sum(-1).mean().item()
                cnt += 1
            res[step][nm] = contrib / cnt
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [res["key step"][n] for n in names], 0.4, color=BLUE,
           label="key-emit step", zorder=3)
    ax.bar(x + 0.2, [res["value step"][n] for n in names], 0.4, color=ORANGE,
           label="value-emit step", zorder=3)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("contribution to correct-token logit")
    ax.set_title("Direct logit attribution: L1 attention writes the answer\n"
                 "(LN gain folded, per-token scale frozen)")
    ax.legend(fontsize=8)
    save(fig, "stream_logit_attribution.png")
    return res


def main():
    apply_style()
    model, _ = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
    tokens, p, _ = build_example()
    dim = fig_dimensionality(model)
    fig_component_writes(model, tokens, p)
    fig_geometry(model)
    attr = fig_logit_attribution(model)
    print("effective dims:", {s: round(dim[s]["participation_ratio"], 1) for s in STAGES})
    print("90% variance in N pcs:", {s: dim[s]["n90"] for s in STAGES})
    print("logit attribution:", json.dumps({k: {n: round(v, 2) for n, v in d.items()}
                                             for k, d in attr.items()}))
    (HERE / "residual_stream.json").write_text(json.dumps(
        {"nominal_dims": 48, "dimensionality": {s: {k: dim[s][k] for k in
         ("participation_ratio", "n90", "n95")} for s in STAGES},
         "logit_attribution": attr}, indent=2))


if __name__ == "__main__":
    main()
