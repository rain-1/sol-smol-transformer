"""In-depth interpretability battery for the 2-layer causal pair-sort decoder.

Runs a range of analyses in one GPU pass (the model is tiny):
  1. head atlas          - attention of every head on one example
  2. head roles          - per-head ablation split into key- vs value-output effect
  3. key-sorting          - where key-emitting positions attend (the sort mechanism)
  4. induction QK          - content vs. position; does L1H0 match on key identity?
  5. precursor path        - ablate each other head, re-measure L1H0 induction
  6. residual probes       - decode key-rank and next-value across the stream
  7. robustness            - tied keys at test, induction sharpness vs. length

Writes figures to reports/figures/deep_*.png and metrics to deep_interp.json.
Usage: python research/sort_scaling/deep_interp.py [--checkpoint path] [--length 12]
"""
from __future__ import annotations

import argparse
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
from style import apply_style, SEQ_BLUE, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    c = ck["model_config"]
    m = cs.CausalTransformer(c["seq_len"], d_model=c["d_model"], n_heads=c["n_heads"],
                             n_layers=c["n_layers"], d_ff=c["d_ff"]).to(DEV).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


@torch.inference_mode()
def run_capture(model, tokens):
    """Forward pass returning logits, per-layer attention, and residual stream."""
    L = tokens.shape[1]
    x = model.token_embedding(tokens) + model.position_embedding[:, :L]
    causal = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
    kpm = tokens.eq(cs.PAD)
    res = {"embed": x}
    attn = []
    for i, block in enumerate(model.blocks):
        x, w = block(x, causal, kpm, capture=True)
        attn.append(w)
        res[f"block{i}"] = x
    xf = model.final_norm(x)
    res["final"] = xf
    return model.output(xf), attn, res


@torch.inference_mode()
def batch(model, p, n=256, seed=0):
    torch.manual_seed(seed)
    seq, _ = cs.make_batch(n, p, DEV, fixed_len=p)
    keys = seq[:, 0:2 * p:2]
    order = keys.argsort(stable=True, dim=1)          # order[b,q] = input idx of rank q
    return seq, keys, order


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/" + name)


def bare(ax):
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# 1. Head atlas
# --------------------------------------------------------------------------- #
def head_atlas(model, p):
    seq, keys, order = batch(model, p, n=64)
    _, attn, _ = run_capture(model, seq[:, :-1])
    sep = 2 * p
    b = 0
    heads = [(li, h) for li in range(len(attn)) for h in range(attn[li].shape[1])]
    fig, axes = plt.subplots(1, len(heads), figsize=(3.1 * len(heads), 3.4))
    if len(heads) == 1:
        axes = [axes]
    out_q = list(range(sep, sep + 2 * p))
    for ax, (li, h) in zip(axes, heads):
        A = attn[li][b, h].cpu().numpy()
        sub = A[np.ix_(out_q, list(range(0, 2 * p)))]
        ax.imshow(sub, cmap=SEQ_BLUE, vmin=0, vmax=sub.max(), aspect="equal")
        ax.set_title(f"L{li}H{h}", fontsize=11)
        ax.set_xlabel("input pos"); bare(ax)
        if ax is axes[0]:
            ax.set_ylabel("output query")
    fig.suptitle("Attention atlas: every head, output queries → input positions "
                 f"(length {p})", fontsize=12, fontweight="bold", y=1.04)
    save(fig, "deep_head_atlas.png")


# --------------------------------------------------------------------------- #
# 2. Head roles: per-head ablation split into key vs value output positions
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def per_position_acc(model, p, n=512, seed=1):
    seq, _, _ = batch(model, p, n=n, seed=seed)
    logits, _, _ = run_capture(model, seq[:, :-1])
    pred = logits.argmax(-1)
    target = seq[:, 1:]
    sep = 2 * p
    kpos = [sep + 2 * q for q in range(p)]           # predict sk_q
    vpos = [sep + 1 + 2 * q for q in range(p)]       # predict sv_q
    key_acc = (pred[:, kpos] == target[:, kpos]).float().mean().item()
    val_acc = (pred[:, vpos] == target[:, vpos]).float().mean().item()
    return key_acc, val_acc


def head_roles(model, p):
    base_k, base_v = per_position_acc(model, p)
    width = model.cfg["d_model"] // model.cfg["n_heads"]
    rows, dk, dv = [], [], []
    for li, block in enumerate(model.blocks):
        for h in range(model.cfg["n_heads"]):
            sl = slice(h * width, (h + 1) * width)
            saved = block.attn.out_proj.weight[:, sl].detach().clone()
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].zero_()
            k, v = per_position_acc(model, p)
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].copy_(saved)
            rows.append(f"L{li}H{h}"); dk.append(base_k - k); dv.append(base_v - v)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.2, dk, 0.4, label="key-output accuracy drop", color=BLUE, zorder=3)
    ax.bar(x + 0.2, dv, 0.4, label="value-output accuracy drop", color=ORANGE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(rows)
    ax.set_ylabel("teacher-forced accuracy drop"); ax.set_ylim(0, 1.02)
    ax.set_title("Causal head roles: which head sorts keys vs. carries values")
    ax.legend(fontsize=8)
    save(fig, "deep_head_roles.png")
    return {"baseline_key_acc": base_k, "baseline_value_acc": base_v,
            "ablation": {r: {"key_drop": k, "value_drop": v} for r, k, v in zip(rows, dk, dv)}}


# --------------------------------------------------------------------------- #
# 3. Key-sorting mechanism
# --------------------------------------------------------------------------- #
POS_CLASSES = ["input keys", "input values", "SEP", "output (generated)"]
POS_COLORS = [BLUE, ORANGE, INK_2, GREEN]


def _class_masses(a, qpos, p, sep):
    """Mean attention mass onto each position class for queries in qpos."""
    aa = a[:, qpos]
    return np.array([
        aa[:, :, 0:2 * p:2].sum(-1).mean().item(),   # input keys
        aa[:, :, 1:2 * p:2].sum(-1).mean().item(),   # input values
        aa[:, :, sep].mean().item(),                 # SEP
        aa[:, :, sep + 1:].sum(-1).mean().item(),    # output region (generated so far)
    ])


def circuit_roles(model, p):
    """Where each head attends, split by whether it is about to emit a key or a value.

    Reveals the division of labour: precursor heads read the input keys at value
    steps, the induction head reads the matching input value, and the sort head
    reads the already-generated output to choose the next key.
    """
    seq, keys, order = batch(model, p, n=256)
    _, attn, _ = run_capture(model, seq[:, :-1])
    sep = 2 * p
    kq = [sep + 2 * q for q in range(p)]
    vq = [sep + 1 + 2 * q for q in range(p)]
    heads = [f"L{li}H{h}" for li in range(len(attn)) for h in range(attn[li].shape[1])]
    key_m, val_m = {}, {}
    for li in range(len(attn)):
        for h in range(attn[li].shape[1]):
            key_m[f"L{li}H{h}"] = _class_masses(attn[li][:, h], kq, p, sep)
            val_m[f"L{li}H{h}"] = _class_masses(attn[li][:, h], vq, p, sep)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mass, title in [(axes[0], key_m, "About to emit a KEY (sort step)"),
                            (axes[1], val_m, "About to emit a VALUE (carry step)")]:
        x = np.arange(len(heads))
        bottom = np.zeros(len(heads))
        for ci, cls in enumerate(POS_CLASSES):
            vals = np.array([mass[h][ci] for h in heads])
            ax.bar(x, vals, 0.6, bottom=bottom, color=POS_COLORS[ci], label=cls, zorder=3)
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(heads)
        ax.set_ylim(0, 1.02); ax.set_title(title)
    axes[0].set_ylabel("attention mass")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Attention role map: what each head reads at each generation step",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "deep_circuit_roles.png")
    return {"key_step": {h: key_m[h].tolist() for h in heads},
            "value_step": {h: val_m[h].tolist() for h in heads},
            "classes": POS_CLASSES}


# --------------------------------------------------------------------------- #
# 4. Induction QK: content vs position
# --------------------------------------------------------------------------- #
def induction_qk(model, p, head=(1, 0)):
    """Does the value-carry head attend by key identity (content) or by position?

    Content dependence: std of attention across examples at fixed (query,key) grid
    positions -- high => the target depends on the data, not the slot. Also verify
    the attended input key equals the sorted key sk_q the query is sitting on.
    """
    li, h = head
    seq, keys, order = batch(model, p, n=256)
    _, attn, _ = run_capture(model, seq[:, :-1])
    a = attn[li][:, h]                                # [B,Q,K]
    sep = 2 * p
    vq = torch.tensor([sep + 1 + 2 * q for q in range(p)], device=DEV)  # emits sv_q
    B = seq.shape[0]
    sub = a[:, vq][:, :, :2 * p]                      # [B,p,2p] attention to input region
    content_sd = sub.std(0).mean().item()
    # argmax attended input position, and whether it is the matching value token.
    val_src = (2 * order + 1)                         # matching value input pos
    picked = a[torch.arange(B)[:, None], vq.view(1, -1).expand(B, -1)].argmax(-1)  # [B,p]
    on_match = (picked == val_src).float().mean().item()
    # Compare with a positional head (min content_sd across heads) for context.
    sds = {}
    for L in range(len(attn)):
        for hh in range(attn[L].shape[1]):
            s = attn[L][:, hh][:, vq][:, :, :2 * p].std(0).mean().item()
            sds[f"L{L}H{hh}"] = s
    return {"head": f"L{li}H{h}", "content_sd": content_sd, "argmax_on_match": on_match,
            "content_sd_all_heads": sds}


# --------------------------------------------------------------------------- #
# 5. Precursor path: ablate each other head, re-measure L1H0 induction
# --------------------------------------------------------------------------- #
def induction_score(model, p, head=(1, 0), n=256):
    li, h = head
    seq, keys, order = batch(model, p, n=n)
    _, attn, _ = run_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    vq = torch.tensor([sep + 1 + 2 * q for q in range(p)], device=DEV)
    val_src = 2 * order + 1
    a = attn[li][:, h]
    return a[torch.arange(B)[:, None], vq.view(1, -1).expand(B, -1), val_src].mean().item()


def precursor_path(model, p, target=(1, 0)):
    base = induction_score(model, p, target)
    width = model.cfg["d_model"] // model.cfg["n_heads"]
    rows, drops = [], []
    for li, block in enumerate(model.blocks):
        for h in range(model.cfg["n_heads"]):
            if (li, h) == target:
                continue
            sl = slice(h * width, (h + 1) * width)
            saved = block.attn.out_proj.weight[:, sl].detach().clone()
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].zero_()
            s = induction_score(model, p, target)
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].copy_(saved)
            rows.append(f"L{li}H{h}"); drops.append(base - s)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(rows))
    ax.bar(x, drops, 0.6, color=VIOLET, zorder=3)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(rows)
    ax.set_ylabel("drop in L1H0 induction score")
    ax.set_title(f"Precursor path: how ablating each head degrades the\ninduction head "
                 f"(baseline score {base:.2f})")
    save(fig, "deep_precursor.png")
    return {"baseline": base, "drops": dict(zip(rows, drops))}


# --------------------------------------------------------------------------- #
# 6. Residual probes: key-rank and next-value across the stream
# --------------------------------------------------------------------------- #
def residual_probes(model, p, n=512):
    seq, keys, order = batch(model, p, n=n)
    _, _, res = run_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    # rank label for each input key position: rank[b, j] = position of key j in sort.
    rank_of = torch.empty_like(keys)
    rank_of.scatter_(1, order, torch.arange(p, device=DEV).expand(B, -1))
    key_pos = list(range(0, 2 * p, 2))
    # value to emit at each value-query position (label = value token id).
    vq = [sep + 1 + 2 * q for q in range(p)]
    val_label = seq[:, [sep + 2 + 2 * q for q in range(p)]]  # sv_q token
    stages = ["embed", "block0", "block1", "final"]
    out = {"key_rank": {}, "next_value": {}}
    for st in stages:
        R = res[st]
        # key-rank probe at input key positions
        X = R[:, key_pos].reshape(-1, R.shape[-1]).cpu().numpy()
        y = rank_of.reshape(-1).cpu().numpy()
        out["key_rank"][st] = _probe(X, y)
        # next-value probe at value-query positions
        Xv = R[:, vq].reshape(-1, R.shape[-1]).cpu().numpy()
        yv = val_label.reshape(-1).cpu().numpy()
        out["next_value"][st] = _probe(Xv, yv)
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(x, [out["key_rank"][s] for s in stages], "-o", color=BLUE, lw=2, ms=7,
            label="key rank (at input key positions)", zorder=3)
    ax.plot(x, [out["next_value"][s] for s in stages], "-s", color=ORANGE, lw=2, ms=7,
            label="value to emit (at value-query positions)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylim(0, 1.02); ax.set_ylabel("linear-probe accuracy")
    ax.set_title("What the residual stream encodes, by depth")
    ax.legend(fontsize=8)
    save(fig, "deep_probes.png")
    return out


def _probe(X, y, train=0.75):
    n = len(X)
    idx = np.random.RandomState(0).permutation(n)
    cut = int(train * n)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xs[tr], y[tr])
    return float(clf.score(Xs[te], y[te]))


# --------------------------------------------------------------------------- #
# 7. Robustness: tied keys at test, and induction sharpness vs length
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def robustness(model, max_p):
    # (a) Tied keys: the model trained on distinct keys; test on keys with duplicates.
    def gen_exact_tied(p, n=256, tie=True):
        eq_ok = 0
        seq = torch.full((n, 4 * max_p + 2), cs.PAD, dtype=torch.long, device=DEV)
        if tie:
            keys = torch.randint(0, max(3, p // 2), (n, p), device=DEV)  # forced ties
        else:
            keys = torch.rand(n, cs.KEY_ALPH, device=DEV).argsort(1)[:, :p]
        vals = torch.randint(0, cs.VAL_ALPH, (n, p), device=DEV) + cs.KEY_ALPH
        order = keys.argsort(stable=True, dim=1)
        sk, sv = keys.gather(1, order), vals.gather(1, order)
        seq[:, 0:2 * p:2] = keys; seq[:, 1:2 * p:2] = vals; seq[:, 2 * p] = cs.SEP
        os = 2 * p + 1
        seq[:, os:os + 2 * p:2] = sk; seq[:, os + 1:os + 2 * p:2] = sv
        work = seq.clone(); sep = 2 * p; work[:, sep + 1:] = cs.PAD
        for t in range(2 * p):
            lo, _ = model(work)
            nxt = lo[:, sep + t].argmax(-1)
            if sep + 1 + t < work.shape[1]:
                work[:, sep + 1 + t] = nxt
        gold = seq[:, sep + 1:sep + 1 + 2 * p]; got = work[:, sep + 1:sep + 1 + 2 * p]
        return (got == gold).all(1).float().mean().item()
    lens = list(range(2, max_p + 1, 2))
    tied = [gen_exact_tied(p, tie=True) for p in lens]
    distinct = [gen_exact_tied(p, tie=False) for p in lens]
    # (b) induction sharpness vs length
    sharp = [induction_score(model, p) for p in lens]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(lens, distinct, "-o", color=BLUE, lw=2, ms=6, label="distinct keys (trained)")
    ax1.plot(lens, tied, "-s", color=ORANGE, lw=2, ms=6, label="tied keys (OOD)")
    ax1.set_ylim(0, 1.02); ax1.set_xlabel("number of pairs"); ax1.set_ylabel("generation exact")
    ax1.set_title("Robustness to duplicate keys (out of distribution)"); ax1.legend(fontsize=8)
    ax2.plot(lens, sharp, "-o", color=VIOLET, lw=2, ms=6, zorder=3)
    ax2.set_ylim(0, 1.02); ax2.set_xlabel("number of pairs")
    ax2.set_ylabel("induction score (L1H0)")
    ax2.set_title("Induction stays sharp as length grows")
    save(fig, "deep_robustness.png")
    return {"lengths": lens, "distinct_exact": distinct, "tied_exact": tied,
            "induction_by_length": sharp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "checkpoints" / "causal_pairs_L30.pt"))
    ap.add_argument("--length", type=int, default=12)
    a = ap.parse_args()
    apply_style()
    model, meta = load(a.checkpoint)
    p = min(a.length, meta["max_pairs"])
    print(f"loaded {a.checkpoint} cfg={model.cfg} analysing at length {p}")
    result = {"checkpoint": a.checkpoint, "config": model.cfg, "length": p}
    head_atlas(model, p)
    result["head_roles"] = head_roles(model, p)
    # Route the value/key heads by their causal role rather than hard-coding.
    abl = result["head_roles"]["ablation"]
    value_head = max(abl, key=lambda k: abl[k]["value_drop"])
    key_head = max(abl, key=lambda k: abl[k]["key_drop"])
    vh = (int(value_head[1]), int(value_head[3]))
    kh = (int(key_head[1]), int(key_head[3]))
    print(f"value head={value_head} key head={key_head}")
    result["circuit_roles"] = circuit_roles(model, p)
    result["induction_qk"] = induction_qk(model, p, head=vh)
    result["precursor_path"] = precursor_path(model, p, target=vh)
    result["residual_probes"] = residual_probes(model, p)
    result["robustness"] = robustness(model, meta["max_pairs"])
    (HERE / "deep_interp.json").write_text(json.dumps(result, indent=2))
    print("\n=== summary ===")
    print("head roles (value-drop):", {k: round(v["value_drop"], 2)
          for k, v in result["head_roles"]["ablation"].items()})
    print("circuit roles (value step):", {h: [round(x, 2) for x in v]
          for h, v in result["circuit_roles"]["value_step"].items()})
    print("induction content_sd:", round(result["induction_qk"]["content_sd"], 3),
          "argmax_on_match:", round(result["induction_qk"]["argmax_on_match"], 3))
    print("probes key_rank final:", round(result["residual_probes"]["key_rank"]["final"], 3),
          "next_value block1:", round(result["residual_probes"]["next_value"]["block1"], 3))
    print("tied-key exact:", [round(x, 2) for x in result["robustness"]["tied_exact"]])


if __name__ == "__main__":
    main()
