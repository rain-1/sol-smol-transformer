"""Reverse-engineering the SORT mechanism (next-key choice) of the causal pair-sort decoder.

Query geometry (fixed length p, sep=2p):
  - sk_q (sorted key rank q) is emitted from query position sep+2q.
    q=0 -> position sep (SEP itself). q>=1 -> that position holds token sv_{q-1}.
  - previously emitted sorted keys sit at seq index sep+1+2j (j=0..q-1); sk_{q-1} is most recent.

Analyses:
  1. Where key-emitting query L1H1 attends inside the output: recency & emitted-rank profiles.
  2. "min input key greater than previous key" rule: logit-lens + causal patch of last key.
  3. Role of layer-0 heads at key steps (SEP parking) and how key info still arrives.
  4. Linear probes for last-emitted key value and next-key value across layers.

Run: python research/sort_scaling/agent_sorthead.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, SEQ_BLUE, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2, DIVERGING  # noqa
import causal_sort as cs  # noqa
import deep_interp as di  # noqa

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = ROOT / "reports" / "figures"
OUT = Path(__file__).resolve().parent / "agent_findings"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "causal_pairs_L30.pt"


def bare(ax):
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# 1. Where the key-emitting query attends inside the output region.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def attn_profiles(model, p=16, n=256):
    seq, keys, order = di.batch(model, p, n=n)
    _, attn, _ = di.run_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    # emitted sorted-key seq indices: sep+1+2j for j=0..p-1
    ekey_pos = [sep + 1 + 2 * j for j in range(p)]
    res = {}
    for li in range(len(attn)):
        for h in range(attn[li].shape[1]):
            a = attn[li][:, h]  # [B,Q,K]
            # for each key-step q (>=1), query = sep+2q, look at emitted keys j<q
            rec = np.full((p, p), np.nan)   # rec[q, r] mass on emitted key at recency r (0=most recent sk_{q-1})
            rnk = np.full((p, p), np.nan)   # rnk[q, j] mass on emitted key of emitted-rank j
            for q in range(1, p):
                qpos = sep + 2 * q
                for j in range(q):
                    m = a[:, qpos, ekey_pos[j]].mean().item()
                    rnk[q, j] = m
                    rec[q, q - 1 - j] = m
            res[f"L{li}H{h}"] = {"rec": rec, "rnk": rnk}
    return res, p


# --------------------------------------------------------------------------- #
# 2. "min greater than previous" rule: logit lens at key query positions.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def logit_lens_rule(model, p=16, n=512):
    seq, keys, order = di.batch(model, p, n=n)
    logits, _, _ = di.run_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    skeys = keys.gather(1, order)  # [B,p] sorted key ids
    # for key-step q>=1: prev emitted key = skeys[:,q-1], correct next = skeys[:,q]
    # check argmax over full vocab and whether predicted == min input key > prev.
    correct = 0; total = 0
    ranks_of_pred = []  # rank (relative to prev) of predicted key among input keys
    for q in range(1, p):
        qpos = sep + 2 * q
        lg = logits[:, qpos]  # [B,V]
        pred = lg.argmax(-1)
        gold = skeys[:, q]
        correct += (pred == gold).sum().item(); total += B
    return correct / total


# --------------------------------------------------------------------------- #
# 2b. Causal patch: override the last emitted key, read the next-key prediction.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def causal_patch_lastkey(model, p=12, n=300, qstep=6, nb=200):
    """Teacher-force the true prefix up to sk_{q-1}, then replace sk_{q-1}'s token
    with each candidate input key and record the argmax next-key prediction.

    Splits by whether the substitution keeps the emitted keys monotone
    (sub >= all earlier emitted keys). Compares two candidate rules:
      last-rule : next = min input key > substituted last key
      max-rule  : next = min input key > max(all emitted keys incl sub)
    """
    seq, keys, order = di.batch(model, p, n=n)
    sep = 2 * p
    skeys = keys.gather(1, order)
    qpos = sep + 2 * qstep
    lastpos = sep + 1 + 2 * (qstep - 1)
    mono, non = [], []
    for b in range(min(seq.shape[0], nb)):
        base = seq[b:b+1].clone()
        ik = sorted(keys[b].tolist())
        prev = skeys[b, :qstep-1].tolist()
        for sub in ik:
            work = base.clone(); work[0, lastpos] = sub
            pred = model(work)[0][0, qpos].argmax(-1).item()
            gl = [k for k in ik if k > sub]
            gm = [k for k in ik if k > max(prev + [sub])]
            if not gl:
                continue
            row = (pred == min(gl), pred == (min(gm) if gm else -1))
            (mono if all(sub >= e for e in prev) else non).append(row)
    mono, non = np.array(mono), np.array(non)
    return {"mono": mono, "non": non,
            "mono_last": float(mono[:, 0].mean()), "mono_max": float(mono[:, 1].mean()),
            "non_last": float(non[:, 0].mean()), "non_max": float(non[:, 1].mean()),
            "n_mono": len(mono), "n_non": len(non)}


# --------------------------------------------------------------------------- #
# 3. Layer-0 head behaviour at key steps.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def layer0_roles(model, p=16, n=256):
    seq, keys, order = di.batch(model, p, n=n)
    _, attn, _ = di.run_capture(model, seq[:, :-1])
    sep = 2 * p
    kq = [sep + 2 * q for q in range(1, p)]  # key-emitting queries (exclude q=0=SEP)
    masses = {}
    for li in range(len(attn)):
        for h in range(attn[li].shape[1]):
            a = attn[li][:, h][:, kq]  # [B,len(kq),K]
            masses[f"L{li}H{h}"] = {
                "input_keys": a[:, :, 0:2*p:2].sum(-1).mean().item(),
                "input_values": a[:, :, 1:2*p:2].sum(-1).mean().item(),
                "SEP": a[:, :, sep].mean().item(),
                "output": a[:, :, sep+1:].sum(-1).mean().item(),
            }
    return masses


# --------------------------------------------------------------------------- #
# 4. Linear probes for last-emitted key and next-key value at key query positions.
# --------------------------------------------------------------------------- #
def _probe(X, y):
    n = len(X)
    idx = np.random.RandomState(0).permutation(n)
    cut = int(0.75 * n); tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xs[tr], y[tr])
    return float(clf.score(Xs[te], y[te]))


@torch.inference_mode()
def threshold_probes(model, p=16, n=600):
    seq, keys, order = di.batch(model, p, n=n)
    _, _, res = di.run_capture(model, seq[:, :-1])
    sep = 2 * p
    skeys = keys.gather(1, order)  # [B,p]
    # key-query positions for q=1..p-1
    qs = list(range(1, p))
    qpos = [sep + 2 * q for q in qs]
    lastkey = skeys[:, [q-1 for q in qs]]   # value fed as "previous key"
    nextkey = skeys[:, qs]                   # key about to be emitted
    stages = ["embed", "block0", "block1", "final"]
    out = {"last_key": {}, "next_key": {}}
    for st in stages:
        R = res[st][:, qpos].reshape(-1, res[st].shape[-1]).cpu().numpy()
        out["last_key"][st] = _probe(R, lastkey.reshape(-1).cpu().numpy())
        out["next_key"][st] = _probe(R, nextkey.reshape(-1).cpu().numpy())
    return out


# --------------------------------------------------------------------------- #
def main():
    apply_style()
    model, meta = di.load(CKPT)
    model.cfg = model.cfg  # ensure attr
    print("loaded", model.cfg)
    result = {}

    prof, p = attn_profiles(model, p=16)
    result["attn_note"] = "profiles computed"

    ll = logit_lens_rule(model, p=16)
    result["final_logit_nextkey_acc"] = ll
    print("final-logit next-key acc:", round(ll, 4))

    cp = causal_patch_lastkey(model, p=12, qstep=6)
    result["causal_patch"] = {k: v for k, v in cp.items() if k not in ("mono", "non")}
    print("causal patch: monotone last-rule=%.3f max-rule=%.3f | non-monotone last=%.3f max=%.3f"
          % (cp["mono_last"], cp["mono_max"], cp["non_last"], cp["non_max"]))

    l0 = layer0_roles(model, p=16)
    result["layer0_roles_keystep"] = l0
    print("layer0 roles at key step:")
    for h, v in l0.items():
        print(" ", h, {k: round(x, 3) for k, x in v.items()})

    thr = threshold_probes(model, p=16)
    result["threshold_probes"] = thr
    print("threshold probes:", json.dumps({k: {s: round(x,3) for s,x in v.items()} for k,v in thr.items()}))

    # ---- figures ----
    make_fig_profiles(prof, p)
    make_fig_probes_rule(l0, thr, ll, cp["mono_last"])
    make_fig_causal_rule(cp)

    (OUT / "_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    print("done")
    return result, prof, p


def make_fig_profiles(prof, p):
    # recency & emitted-rank profile for L1H1 (sort head) averaged over q
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, head in zip(axes[:2], ["L1H1", "L1H0"]):
        rec = prof[head]["rec"]
        recmean = np.nanmean(rec, axis=0)
        ax.plot(np.arange(p), recmean, "-o", color=BLUE if head=="L1H1" else ORANGE, lw=2, ms=5)
        ax.set_xlabel("recency (0 = most recent emitted key sk$_{q-1}$)")
        ax.set_ylabel("mean attention mass")
        ax.set_title(f"{head}: attention vs recency of emitted key")
        ax.set_xlim(-0.5, min(p-1, 12))
        bare(ax)
    # heatmap of L1H1 rank profile
    ax = axes[2]
    rnk = prof["L1H1"]["rnk"]
    im = ax.imshow(rnk, cmap=SEQ_BLUE, aspect="auto", origin="lower")
    ax.set_xlabel("emitted-rank j of attended key")
    ax.set_ylabel("key-step q")
    ax.set_title("L1H1: attention[q, emitted key j]")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Sort head L1H1 reads the most-recently emitted key (recency 0)",
                 fontweight="bold", y=1.02)
    fig.savefig(FIG / "agent_sort_attn_profiles.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote agent_sort_attn_profiles.png")


def make_fig_probes_rule(l0, thr, ll, mm):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # probe curves
    ax = axes[0]
    stages = ["embed", "block0", "block1", "final"]
    x = np.arange(len(stages))
    ax.plot(x, [thr["last_key"][s] for s in stages], "-o", color=VIOLET, lw=2, ms=7,
            label="last-emitted key value")
    ax.plot(x, [thr["next_key"][s] for s in stages], "-s", color=GREEN, lw=2, ms=7,
            label="next key value (to emit)")
    ax.set_xticks(x); ax.set_xticklabels(stages); ax.set_ylim(0, 1.02)
    ax.set_ylabel("linear-probe accuracy")
    ax.set_title("Threshold representation at key-emitting positions")
    ax.legend(fontsize=8); bare(ax)
    # layer0 mass at key step
    ax = axes[1]
    heads = list(l0.keys())
    classes = ["input_keys", "input_values", "SEP", "output"]
    colors = [BLUE, ORANGE, INK_2, GREEN]
    xh = np.arange(len(heads)); bottom = np.zeros(len(heads))
    for ci, c in enumerate(classes):
        vals = np.array([l0[h][c] for h in heads])
        ax.bar(xh, vals, 0.6, bottom=bottom, color=colors[ci], label=c, zorder=3)
        bottom += vals
    ax.set_xticks(xh); ax.set_xticklabels(heads); ax.set_ylim(0, 1.02)
    ax.set_ylabel("attention mass"); ax.set_title("Attention at KEY steps (q>=1)")
    ax.legend(fontsize=7, loc="upper right"); bare(ax)
    fig.suptitle(f"Next-key logit acc={ll:.3f}  |  causal min-greater match={mm:.3f}",
                 fontweight="bold", y=1.03)
    fig.savefig(FIG / "agent_sort_probes_rule.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote agent_sort_probes_rule.png")


def make_fig_causal_rule(cp):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    labels = ["monotone sub\n(sub >= all emitted)", "non-monotone sub\n(sub < some emitted)"]
    last = [cp["mono_last"], cp["non_last"]]
    mx = [cp["mono_max"], cp["non_max"]]
    x = np.arange(2)
    ax.bar(x - 0.2, last, 0.4, color=BLUE, label="pred = min input key > substituted last key", zorder=3)
    ax.bar(x + 0.2, mx, 0.4, color=GREEN, label="pred = min input key > max(emitted keys)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of substitutions matching rule")
    ax.set_title("Causal test of next-key rule (patch last emitted key, step q=6)")
    ax.legend(fontsize=7, loc="lower center"); bare(ax)
    fig.savefig(FIG / "agent_sort_causal_rule.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote agent_sort_causal_rule.png")


if __name__ == "__main__":
    main()
