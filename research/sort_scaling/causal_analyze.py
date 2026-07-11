"""Attention interpretability for the causal pair-sort decoder.

The hypothesis: value-carry is an induction copy. The position that must emit the
value sv_q sits on the freshly generated sorted key sk_q; to produce sv_q it should
attend back to the input occurrence of sk_q and copy the value token that follows it.
This script quantifies that per head, runs causal head ablations, and renders the
attention heatmap over the output->input positions (the induction map).

Usage: python research/sort_scaling/causal_analyze.py [--checkpoint path] [--length 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, SEQ_BLUE, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["model_config"]
    model = cs.CausalTransformer(c["seq_len"], d_model=c["d_model"], n_heads=c["n_heads"],
                                 n_layers=c["n_layers"], d_ff=c["d_ff"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck


@torch.inference_mode()
def teacher_forced_attention(model, p, device, n=256):
    """Capture attention on gold sequences at fixed length p. Returns attn, seq, order."""
    seq, _ = cs.make_batch(n, p, device, fixed_len=p)
    _, attn = model(seq[:, :-1], capture=True)  # list of [B,H,Q,K]
    # Recover per-row key order to know where each value lives in the input.
    keys = seq[:, 0:2 * p:2]                      # [B,p] input keys in original order
    order = keys.argsort(stable=True, dim=1)      # order[b,q] = input index of rank q
    return attn, seq, order


def induction_scores(model, p, device, n=256):
    """Per head: mean attention from the sk_q position to the matching input value token."""
    attn, seq, order = teacher_forced_attention(model, p, device, n)
    sep = 2 * p
    B = seq.shape[0]
    # Output token j is at index sep+1+j and is predicted from position sep+j.
    # sv_q is output token 2q+1, so it is predicted from position sep+2q+1 (which
    # holds the just-generated sorted key sk_q).
    qpos = sep + 2 * torch.arange(p, device=device) + 1
    val_src = 2 * order + 1                                    # input value position for rank q
    scores = {}
    for li, A in enumerate(attn):
        H = A.shape[1]
        for h in range(H):
            a = A[:, h]                                        # [B,Q,K]
            q = qpos.view(1, -1).expand(B, -1)
            picked = a[torch.arange(B)[:, None], q, val_src]   # [B,p]
            key_src = a[torch.arange(B)[:, None], q, 2 * order]  # attention to matching key
            scores[f"L{li}H{h}"] = {
                "induction_value_mass": picked.mean().item(),
                "matching_key_mass": key_src.mean().item()}
    return scores, (attn, seq, order)


@torch.inference_mode()
def head_ablations(model, p, device, batch_size=256):
    base = cs.generate_exact(model, p, device, batch_size, fixed_len=p)
    result = {"baseline_gen_exact": base, "heads": {}}
    width = model.cfg["d_model"] // model.cfg["n_heads"]
    for li, block in enumerate(model.blocks):
        for h in range(model.cfg["n_heads"]):
            sl = slice(h * width, (h + 1) * width)
            saved = block.attn.out_proj.weight[:, sl].detach().clone()
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].zero_()
            ex = cs.generate_exact(model, p, device, batch_size, fixed_len=p)
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].copy_(saved)
            result["heads"][f"L{li}H{h}"] = {"gen_exact": ex, "drop": base - ex}
    return result


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_induction_heatmap(attn, seq, order, p, head, device):
    li, h = head
    b = 0
    sep = 2 * p
    A = attn[li][b, h].cpu().numpy()                # [Q,K] over seq[:-1]
    # Rows: output positions (SEP .. last sorted token). Cols: input pair positions.
    out_q = list(range(sep, sep + 2 * p))           # queries generating the sorted pairs
    in_k = list(range(0, 2 * p))                    # input pair tokens
    sub = A[np.ix_(out_q, in_k)]
    keys = seq[b, 0:2 * p:2].cpu().numpy()
    vals = seq[b, 1:2 * p:2].cpu().numpy() - cs.KEY_ALPH
    in_labels = []
    for j in range(p):
        in_labels += [f"k{keys[j]}", f"v{vals[j]}"]
    ordb = order[b].cpu().numpy()
    out_labels = []
    for q in range(p):
        out_labels += [f"→k{keys[ordb[q]]}", f"→v{vals[ordb[q]]}"]
    fig, ax = plt.subplots(figsize=(min(14, 1 + 0.5 * 2 * p), min(12, 1 + 0.45 * 2 * p)))
    im = ax.imshow(sub, cmap=SEQ_BLUE, vmin=0, vmax=sub.max(), aspect="equal")
    ax.set_xticks(range(2 * p)); ax.set_xticklabels(in_labels, fontsize=7, rotation=90)
    ax.set_yticks(range(2 * p)); ax.set_yticklabels(out_labels, fontsize=7)
    ax.set_xlabel("input position (keys k·, carried values v·)")
    ax.set_ylabel("output position (what it is about to emit)")
    ax.set_title(f"Induction copy in L{li}H{h}: each 'emit vX' row lights up the input vX\n"
                 f"(length-{p} example)", fontsize=11, fontweight="bold")
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("attention", fontsize=9)
    fig.savefig(FIG / "sort_causal_induction_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/sort_causal_induction_heatmap.png")


def fig_scores_and_ablation(scores, ablation):
    heads = list(scores)
    ind = [scores[h]["induction_value_mass"] for h in heads]
    drop = [ablation["heads"].get(h, {}).get("drop", 0) for h in heads]
    y = np.arange(len(heads))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(3, 0.42 * len(heads))))
    ax1.barh(y, ind, color=BLUE, height=0.6, zorder=3)
    ax1.set_yticks(y); ax1.set_yticklabels(heads); ax1.invert_yaxis()
    ax1.set_xlim(0, 1); ax1.set_title("Induction score:\nattention on the matching input value")
    ax1.set_xlabel("mean attention mass")
    for yi, v in zip(y, ind):
        ax1.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color=INK)
    ax2.barh(y, drop, color=ORANGE, height=0.6, zorder=3)
    ax2.set_yticks(y); ax2.set_yticklabels(heads); ax2.invert_yaxis()
    ax2.set_xlim(0, 1); ax2.set_title("Causal ablation:\ngeneration-exact drop")
    ax2.set_xlabel("drop when head zeroed")
    for yi, v in zip(y, drop):
        ax2.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color=INK)
    fig.savefig(FIG / "sort_causal_head_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/sort_causal_head_metrics.png")


def fig_by_length(meta):
    bylen = meta.get("gen_by_length")
    if not bylen:
        return
    ns = sorted(int(k) for k in bylen)
    ys = [bylen[str(n)] if str(n) in bylen else bylen[n] for n in ns]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, ys, "-o", color=VIOLET, lw=2, ms=6, zorder=3)
    ax.set_ylim(0, 1.02); ax.set_xlabel("number of pairs"); ax.set_ylabel("generation exact")
    ax.set_title("Autoregressive pair-sort accuracy by length")
    fig.savefig(FIG / "sort_causal_by_length.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/sort_causal_by_length.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--length", type=int, default=12)
    a = ap.parse_args()
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = a.checkpoint or sorted((HERE / "checkpoints").glob("causal_pairs_*.pt"))[-1]
    model, meta = load_model(ckpt, device)
    print(f"loaded {ckpt} gen_exact={meta.get('gen_exact'):.4f} cfg={model.cfg}")
    p = min(a.length, meta["max_pairs"])
    scores, (attn, seq, order) = induction_scores(model, p, device)
    abl = head_ablations(model, p, device)
    best = max(scores, key=lambda k: scores[k]["induction_value_mass"])
    print("best induction head:", best, scores[best])
    print(json.dumps({"scores": scores, "ablation": abl}, indent=2))
    (HERE / "causal_analysis.json").write_text(json.dumps(
        {"checkpoint": str(ckpt), "config": model.cfg, "length": p,
         "induction_scores": scores, "ablation": abl, "best_head": best}, indent=2))
    li, h = int(best[1]), int(best[3])
    fig_induction_heatmap(attn, seq, order, p, (li, h), device)
    fig_scores_and_ablation(scores, abl)
    fig_by_length(meta)


if __name__ == "__main__":
    main()
