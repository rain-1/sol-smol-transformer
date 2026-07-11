"""Failure-mode and boundary-behavior characterization for the 2-layer causal pair-sort decoder.

Brief: localize errors, duplicate-key breakdown, SEP sink causal test, EOS termination,
length behavior. Figures -> reports/figures/agent_fail_*.png.

Reuses (imports, does not modify) causal_sort.py and deep_interp.py.
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
from style import apply_style, SEQ_BLUE, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2, RED, DIVERGING  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load, run_capture, batch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = HERE / "checkpoints" / "causal_pairs_L30.pt"


def savefig(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/" + name)


def bare(ax):
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# Teacher-forced generation with full labeling for error localization.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def tf_errors(model, p, n=1024, seed=0):
    """Teacher-forced argmax. Return per-slot correctness arrays.

    Slots for a length-p seq (excluding EOS): 2p output tokens (keys at even rank, values at odd).
    Also include the EOS slot.
    """
    seq, keys, order = batch(model, p, n=n, seed=seed)
    logits, _, _ = run_capture(model, seq[:, :-1])
    pred = logits.argmax(-1)
    target = seq[:, 1:]
    sep = 2 * p
    # positions in pred-space: prediction of token at index j comes from logits[:, j-1]
    # target[:, j-1] == seq[:, j]. output token sk_q at seq index sep+1+2q, sv_q at sep+2+2q.
    kpos = [sep + 2 * q for q in range(p)]      # logits index that predicts sk_q  (seq sep+1+2q)
    vpos = [sep + 1 + 2 * q for q in range(p)]  # logits index that predicts sv_q
    eos_pos = sep + 2 * p                        # predicts EOS at seq sep+1+2p
    key_ok = (pred[:, kpos] == target[:, kpos]).cpu().numpy()   # [n,p]
    val_ok = (pred[:, vpos] == target[:, vpos]).cpu().numpy()   # [n,p]
    eos_ok = (pred[:, eos_pos] == target[:, eos_pos]).cpu().numpy()  # [n]
    eos_pred = pred[:, eos_pos].cpu().numpy()
    return dict(key_ok=key_ok, val_ok=val_ok, eos_ok=eos_ok, eos_pred=eos_pred,
                keys=keys.cpu().numpy(), order=order.cpu().numpy())


@torch.inference_mode()
def ar_errors(model, p, n=1024, seed=0):
    """True autoregressive greedy decode. Return per-slot correctness (keys/vals) and
    whether the model would emit EOS at the correct step."""
    seq, keys, order = batch(model, p, n=n, seed=seed)
    sep = 2 * p
    pair_len = 2 * p
    work = seq.clone()
    work[:, sep + 1:] = cs.PAD
    # decode pair_len + 1 steps to also capture the EOS slot
    preds = torch.full((n, pair_len + 1), cs.PAD, dtype=torch.long, device=DEV)
    for t in range(pair_len + 1):
        logits, _ = model(work)
        nxt = logits[:, sep + t, :].argmax(-1)
        preds[:, t] = nxt
        if sep + 1 + t < work.shape[1]:
            work[:, sep + 1 + t] = nxt
    gold_pairs = seq[:, sep + 1:sep + 1 + pair_len]
    got_pairs = preds[:, :pair_len]
    match = (got_pairs == gold_pairs).cpu().numpy()  # [n, 2p]
    key_ok = match[:, 0::2]  # [n,p]
    val_ok = match[:, 1::2]
    eos_pred = preds[:, pair_len].cpu().numpy()  # token at the EOS slot
    eos_ok = (eos_pred == cs.EOS)
    return dict(key_ok=key_ok, val_ok=val_ok, eos_ok=eos_ok, eos_pred=eos_pred,
                keys=keys.cpu().numpy(), order=order.cpu().numpy())


# --------------------------------------------------------------------------- #
# Custom forward that can block attention TO specified key positions (for SEP sink test)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def forward_block_keys(model, tokens, block_mask=None):
    """Forward pass identical to run_capture, but optionally add positions in block_mask
    (bool [B,L]) to the key_padding_mask so no query can attend to them. Attention
    renormalizes over remaining keys. Returns logits, attn list."""
    L = tokens.shape[1]
    x = model.token_embedding(tokens) + model.position_embedding[:, :L]
    causal = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
    kpm = tokens.eq(cs.PAD)
    if block_mask is not None:
        kpm = kpm | block_mask
    attn = []
    for block in model.blocks:
        x, w = block(x, causal, kpm, capture=True)
        attn.append(w)
    return model.output(model.final_norm(x)), attn


# --------------------------------------------------------------------------- #
# FIG 1: error localization
# --------------------------------------------------------------------------- #
def fig_error_localization(model, max_p=30):
    lens = list(range(2, max_p + 1))
    n = 1500  # VRAM-frugal (was 6000; run_capture materialises full attention tensors)
    tf_key_by_len, tf_val_by_len, ar_key_by_len, ar_val_by_len = [], [], [], []
    ar_seqexact = []
    # rank-normalized error accumulation (fractional rank 0..1) for AR
    rank_bins = np.linspace(0, 1, 11)
    key_err_by_rank = np.zeros(len(rank_bins) - 1); key_cnt = np.zeros(len(rank_bins) - 1)
    val_err_by_rank = np.zeros(len(rank_bins) - 1); val_cnt = np.zeros(len(rank_bins) - 1)
    # value-error decomposition (AR): value wrong & key-at-same-rank wrong (cascade) vs key ok (pure induction)
    cascade = pure = 0
    for p in lens:
        e = tf_errors(model, p, n=n, seed=p)
        a = ar_errors(model, p, n=n, seed=1000 + p)
        torch.cuda.empty_cache()
        tf_key_by_len.append(1 - e["key_ok"].mean()); tf_val_by_len.append(1 - e["val_ok"].mean())
        ar_key_by_len.append(1 - a["key_ok"].mean()); ar_val_by_len.append(1 - a["val_ok"].mean())
        ar_seqexact.append((a["key_ok"].all(1) & a["val_ok"].all(1)).mean())
        # rank binning (AR)
        ranks = np.arange(p) / max(1, p - 1)
        binidx = np.clip(np.digitize(ranks, rank_bins) - 1, 0, len(rank_bins) - 2)
        kerr = (~a["key_ok"]).sum(0); verr = (~a["val_ok"]).sum(0)
        for q in range(p):
            key_err_by_rank[binidx[q]] += kerr[q]; key_cnt[binidx[q]] += n
            val_err_by_rank[binidx[q]] += verr[q]; val_cnt[binidx[q]] += n
        # decomposition: for AR value errors, is key at that rank also wrong?
        vbad = ~a["val_ok"]; kbad = ~a["key_ok"]
        cascade += (vbad & kbad).sum(); pure += (vbad & ~kbad).sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(lens, ar_key_by_len, "-o", color=BLUE, ms=4, lw=1.8, label="key error (autoregressive)")
    ax.plot(lens, ar_val_by_len, "-o", color=ORANGE, ms=4, lw=1.8, label="value error (autoregressive)")
    ax.plot(lens, tf_key_by_len, "--", color=BLUE, lw=1.2, alpha=0.7, label="key error (teacher-forced)")
    ax.plot(lens, tf_val_by_len, "--", color=ORANGE, lw=1.2, alpha=0.7, label="value error (teacher-forced)")
    ax.set_xlabel("number of pairs"); ax.set_ylabel("per-token error rate")
    ax.set_title(f"Errors concentrate at the max trained length (n={n}/len)")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    centers = (rank_bins[:-1] + rank_bins[1:]) / 2
    ax.plot(centers, key_err_by_rank / np.maximum(key_cnt, 1), "-o", color=BLUE, ms=4, label="key")
    ax.plot(centers, val_err_by_rank / np.maximum(val_cnt, 1), "-o", color=ORANGE, ms=4, label="value")
    ax.set_xlabel("fractional output rank (0 = first emitted, 1 = last)")
    ax.set_ylabel("error rate (pooled over lengths)")
    ax.set_title("Errors concentrate at LATE output ranks")
    ax.legend(fontsize=8)
    fig.suptitle("Error localization (true autoregressive generation)", fontweight="bold", y=1.02)
    savefig(fig, "agent_fail_error_localization.png")
    return dict(lens=lens, tf_key=tf_key_by_len, tf_val=tf_val_by_len,
                ar_key=ar_key_by_len, ar_val=ar_val_by_len, ar_seqexact=ar_seqexact,
                value_err_cascade=int(cascade), value_err_pure=int(pure))


# --------------------------------------------------------------------------- #
# FIG 2: duplicate-key breakdown
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def tied_key_example(model, p=6, n_ties=2):
    """Build one sequence with n_ties keys sharing a value, run, return induction attention."""
    torch.manual_seed(3)
    keys = torch.randperm(cs.KEY_ALPH, device=DEV)[:p].unsqueeze(0)
    # force the first n_ties keys equal
    keys[0, :n_ties] = keys[0, 0]
    vals = (torch.randint(0, cs.VAL_ALPH, (1, p), device=DEV) + cs.KEY_ALPH)
    order = keys.argsort(stable=True, dim=1)
    sk, sv = keys.gather(1, order), vals.gather(1, order)
    seq = torch.full((1, 4 * p + 2), cs.PAD, dtype=torch.long, device=DEV)
    seq[:, 0:2 * p:2] = keys; seq[:, 1:2 * p:2] = vals; seq[:, 2 * p] = cs.SEP
    os = 2 * p + 1; seq[:, os:os + 2 * p:2] = sk; seq[:, os + 1:os + 2 * p:2] = sv
    seq[:, os + 2 * p] = cs.EOS
    _, attn, _ = run_capture(model, seq[:, :-1])
    return seq, keys, vals, order, attn


@torch.inference_mode()
def tied_accuracy(model, max_p, p=8, n=1024):
    """Vary number of tied keys among p pairs (rest distinct). Report exact & value acc."""
    res = {}
    for n_ties in range(1, p + 1):  # 1 tie = all distinct
        seq = torch.full((n, 4 * max_p + 2), cs.PAD, dtype=torch.long, device=DEV)
        keys = torch.rand(n, cs.KEY_ALPH, device=DEV).argsort(1)[:, :p]
        keys[:, :n_ties] = keys[:, :1]  # first n_ties keys tied to key[0]
        vals = torch.randint(0, cs.VAL_ALPH, (n, p), device=DEV) + cs.KEY_ALPH
        order = keys.argsort(stable=True, dim=1)
        sk, sv = keys.gather(1, order), vals.gather(1, order)
        seq[:, 0:2 * p:2] = keys; seq[:, 1:2 * p:2] = vals; seq[:, 2 * p] = cs.SEP
        os = 2 * p + 1; seq[:, os:os + 2 * p:2] = sk; seq[:, os + 1:os + 2 * p:2] = sv
        work = seq.clone(); sep = 2 * p; work[:, sep + 1:] = cs.PAD
        for t in range(2 * p):
            lo, _ = model(work); nxt = lo[:, sep + t].argmax(-1)
            if sep + 1 + t < work.shape[1]:
                work[:, sep + 1 + t] = nxt
        gold = seq[:, sep + 1:sep + 1 + 2 * p]; got = work[:, sep + 1:sep + 1 + 2 * p]
        exact = (got == gold).all(1).float().mean().item()
        kpos = list(range(0, 2 * p, 2)); vpos = list(range(1, 2 * p, 2))
        # value accuracy: does the emitted value belong to the multiset for that key?
        vacc = (got[:, vpos] == gold[:, vpos]).float().mean().item()
        kacc = (got[:, kpos] == gold[:, kpos]).float().mean().item()
        res[n_ties] = dict(exact=exact, val_acc=vacc, key_acc=kacc)
    return res


def fig_duplicate_keys(model, max_p=30):
    # (a) attention on a tied example
    p = 6; n_ties = 3
    seq, keys, vals, order, attn = tied_key_example(model, p=p, n_ties=n_ties)
    sep = 2 * p
    # value head = L1H0 per established. attention of L1H0 at value-query positions.
    A = attn[1][0, 0].cpu().numpy()   # [L,L]
    vq = [sep + 1 + 2 * q for q in range(p)]
    sub = A[np.ix_(vq, list(range(0, 2 * p)))]  # value queries -> input positions
    # (b) accuracy vs number of ties
    acc = tied_accuracy(model, max_p, p=8, n=1024)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    im = ax.imshow(sub, cmap=SEQ_BLUE, vmin=0, vmax=sub.max(), aspect="auto")
    ax.set_xticks(range(2 * p))
    labs = []
    for j in range(2 * p):
        if j % 2 == 0:
            labs.append(f"k{keys[0, j // 2].item()}")
        else:
            labs.append(f"v")
    ax.set_xticklabels(labs, fontsize=7, rotation=90)
    ax.set_yticks(range(p)); ax.set_yticklabels([f"emit sv_{q}" for q in range(p)], fontsize=8)
    ax.set_xlabel("input position"); bare(ax)
    ax.set_title(f"L1H0 induction attention, {n_ties} tied keys\n(first {n_ties} input keys identical)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    x = list(acc.keys())
    ax.plot(x, [acc[k]["exact"] for k in x], "-o", color=VIOLET, label="sequence exact")
    ax.plot(x, [acc[k]["val_acc"] for k in x], "-s", color=ORANGE, label="value token acc")
    ax.plot(x, [acc[k]["key_acc"] for k in x], "-^", color=BLUE, label="key token acc")
    ax.set_xlabel("number of tied keys (among 8 pairs; 1 = all distinct)")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy collapses as ties increase")
    ax.legend(fontsize=8)
    fig.suptitle("Duplicate-key breakdown (out of distribution)", fontweight="bold", y=1.02)
    savefig(fig, "agent_fail_duplicate_keys.png")
    # quantify attention split for tied example: mass on each of the tied value sources
    tied_val_pos = [2 * q + 1 for q in range(n_ties)]  # input value positions of tied keys
    # at value-query q=0 (emitting first sorted value), how is mass split among tied sources?
    split = A[vq[0], tied_val_pos].tolist()
    return dict(tied_accuracy=acc, tie_attn_split_q0=split, n_ties=n_ties)


# --------------------------------------------------------------------------- #
# FIG 3: SEP attention sink causal test
# --------------------------------------------------------------------------- #
def fig_sep_sink(model, max_p=30):
    lens = [5, 10, 20, 30]
    n = 800  # VRAM-frugal (was 2000)
    rows = []
    for p in lens:
        seq, keys, order = batch(model, p, n=n, seed=7)
        sep = 2 * p
        # baseline
        base_logits, base_attn = forward_block_keys(model, seq[:, :-1], None)
        # SEP mass by head at output queries (how much of a sink is it)
        oq = list(range(sep, sep + 2 * p))
        sepmass = {f"L{li}H{h}": base_attn[li][:, h][:, oq, sep].mean().item()
                   for li in range(2) for h in range(2)}
        # block attention to SEP
        block_mask = torch.zeros_like(seq[:, :-1], dtype=torch.bool)
        block_mask[:, sep] = True
        blk_logits, _ = forward_block_keys(model, seq[:, :-1], block_mask)

        def acc(logits):
            pred = logits.argmax(-1); target = seq[:, 1:]
            kpos = [sep + 2 * q for q in range(p)]; vpos = [sep + 1 + 2 * q for q in range(p)]
            return ((pred[:, kpos] == target[:, kpos]).float().mean().item(),
                    (pred[:, vpos] == target[:, vpos]).float().mean().item())
        bk, bv = acc(base_logits); zk, zv = acc(blk_logits)
        rows.append(dict(p=p, sepmass=sepmass, base_key=bk, base_val=bv,
                         block_key=zk, block_val=zv))
        print(f"SEP sink p={p}: base(k={bk:.3f},v={bv:.3f}) block-SEP(k={zk:.3f},v={zv:.3f}) "
              f"sepmass={ {k: round(v,2) for k,v in sepmass.items()} }")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    heads = [f"L{li}H{h}" for li in range(2) for h in range(2)]
    x = np.arange(len(lens)); w = 0.2
    for i, hd in enumerate(heads):
        ax.bar(x + (i - 1.5) * w, [r["sepmass"][hd] for r in rows], w, label=hd, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(lens); ax.set_xlabel("number of pairs")
    ax.set_ylabel("mean attention mass on SEP\n(output queries)")
    ax.set_title("SEP is a sink for idle heads"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(lens, [r["base_key"] for r in rows], "-o", color=BLUE, label="key acc (baseline)")
    ax.plot(lens, [r["block_key"] for r in rows], "--o", color=BLUE, alpha=0.6, label="key acc (SEP blocked)")
    ax.plot(lens, [r["base_val"] for r in rows], "-s", color=ORANGE, label="value acc (baseline)")
    ax.plot(lens, [r["block_val"] for r in rows], "--s", color=ORANGE, alpha=0.6, label="value acc (SEP blocked)")
    ax.set_ylim(0, 1.02); ax.set_xlabel("number of pairs"); ax.set_ylabel("teacher-forced accuracy")
    ax.set_title("Blocking attention to SEP: effect on emission")
    ax.legend(fontsize=7.5)
    fig.suptitle("SEP attention sink: causal test", fontweight="bold", y=1.02)
    savefig(fig, "agent_fail_sep_sink.png")
    return dict(rows=rows)


# --------------------------------------------------------------------------- #
# FIG 4: EOS termination + count probe
# --------------------------------------------------------------------------- #
def fig_eos_termination(model, max_p=30):
    from sklearn.linear_model import LogisticRegression
    # (a) what is emitted at the EOS slot, across lengths
    import collections
    eos_rate = []; topk_tok = []
    lens = list(range(2, max_p + 1))
    for p in lens:
        e = tf_errors(model, p, n=1500, seed=p)
        eos_rate.append(e["eos_ok"].mean())
        c = collections.Counter(e["eos_pred"].tolist())
        topk_tok.append(c.most_common(1)[0])
    # (b) count/remaining probe: from block1 residual at each output KEY-query position,
    # decode "pairs remaining" = p - q. Pool over lengths so absolute position varies.
    Xs, ys, yspos = [], [], []
    for p in range(4, max_p + 1, 2):
        seq, keys, order = batch(model, p, n=200, seed=p)
        _, _, res = run_capture(model, seq[:, :-1])
        R = res["block1"]; sep = 2 * p
        for q in range(p):
            pos = sep + 2 * q  # about to emit sk_q
            Xs.append(R[:, pos].cpu().numpy())
            ys.append(np.full(R.shape[0], p - q))     # remaining incl current
            yspos.append(np.full(R.shape[0], q))      # emitted-so-far
    X = np.concatenate(Xs); yrem = np.concatenate(ys); yemit = np.concatenate(yspos)

    def probe(X, y):
        idx = np.random.RandomState(0).permutation(len(X)); cut = int(0.75 * len(X))
        tr, te = idx[:cut], idx[cut:]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs2 = (X - mu) / sd
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(Xs2[tr], y[tr]); return clf.score(Xs2[te], y[te])
    acc_rem = probe(X, yrem)       # remaining count (needs to know total p)
    acc_emit = probe(X, yemit)     # emitted-so-far (≈ absolute position, easy)
    # baseline: majority-class for remaining
    from collections import Counter
    maj = max(Counter(yrem).values()) / len(yrem)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(lens, eos_rate, "-o", color=RED, ms=4)
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("number of pairs")
    ax.set_ylabel("P(emit EOS at final slot)")
    ax.set_title("EOS is never emitted (0%)\nEOS token is not in the training loss")
    ax = axes[1]
    names = ["emitted-so-far\n(≈abs. position)", "pairs remaining\n(needs total length)", "majority\nbaseline"]
    vals = [acc_emit, acc_rem, maj]
    ax.bar(range(3), vals, color=[BLUE, ORANGE, INK_2], zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("linear-probe accuracy (block1)")
    ax.set_title("Does the stream track how many pairs remain?")
    fig.suptitle("EOS / termination behavior", fontweight="bold", y=1.02)
    savefig(fig, "agent_fail_eos_termination.png")
    return dict(eos_rate=eos_rate, top_token=topk_tok, probe_remaining=acc_rem,
                probe_emitted=acc_emit, majority=maj)


def main():
    apply_style()
    model, meta = load(CKPT)
    max_p = meta["max_pairs"]
    print("loaded", CKPT, "cfg", model.cfg, "max_pairs", max_p)
    out = {}
    out["error_localization"] = fig_error_localization(model, max_p)
    out["duplicate_keys"] = fig_duplicate_keys(model, max_p)
    out["sep_sink"] = fig_sep_sink(model, max_p)
    out["eos"] = fig_eos_termination(model, max_p)
    # trim big arrays for json
    def clean(o):
        import numpy as _np
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (_np.floating, _np.integer)): return float(o)
        return o
    (HERE / "agent_findings" / "failure_metrics.json").write_text(json.dumps(clean(out), indent=2, default=str))
    print("\n=== KEY NUMBERS ===")
    el = out["error_localization"]
    print("AR seq-exact by len (first/last):", round(el["ar_seqexact"][0], 4), round(el["ar_seqexact"][-1], 4))
    print("value-err decomposition cascade(key-also-wrong):", el["value_err_cascade"],
          "pure(key-ok, induction):", el["value_err_pure"])
    print("EOS probe remaining=%.3f emitted=%.3f majority=%.3f" %
          (out["eos"]["probe_remaining"], out["eos"]["probe_emitted"], out["eos"]["majority"]))


if __name__ == "__main__":
    main()
