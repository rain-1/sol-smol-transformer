"""How does the decoder actually compare keys and pick the next one?

Tests the "attention-as-comparator" hypothesis: at a key-emit step with threshold
theta = last emitted key, does some head attend to candidate INPUT keys with a
"just above theta" profile (peak at the smallest positive gap), and is there a
monotonic key-order axis that turns ">" into a linear projection?

Prints where key-step attention goes (input keys / SEP / emitted output), the
attention-vs-(key-theta) profile per head, and the key-order axis strength.
Figure -> reports/figures/walk_compare.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load, batch  # noqa: E402
from walkthrough import full_capture  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def analyze(model, p=14, n=1024):
    seq, keys, order = batch(model, p, n=n, seed=11)
    logits, caps = full_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    # emitted sorted keys sk_q at seq index sep+1+2q; key-emit query for sk_q at sep+2q.
    sk = seq[:, [sep + 1 + 2 * q for q in range(p)]]        # [B,p] sorted keys
    # threshold at query emitting sk_q (q>=1) is sk_{q-1}; skip q=0 (no threshold).
    heads = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # (1) where does key-step attention go? input-key cols, SEP, emitted-output cols.
    key_cols = list(range(0, 2 * p, 2))
    where = {}
    for (li, h) in heads:
        A = caps[li]["attn"][:, h]                          # [B,Q,K]
        qmass_in = qmass_sep = qmass_out = 0.0
        cnt = 0
        for q in range(1, p):
            qpos = sep + 2 * q
            a = A[:, qpos]                                   # [B,K]
            qmass_in += a[:, key_cols].sum(1).mean().item()
            qmass_sep += a[:, sep].mean().item()
            qmass_out += a[:, sep + 1:].sum(1).mean().item()
            cnt += 1
        where[f"L{li}H{h}"] = dict(input_keys=qmass_in / cnt, SEP=qmass_sep / cnt,
                                   emitted_output=qmass_out / cnt)
    # (2) attention-vs-gap profile: for the query emitting sk_q, attention to input key
    #     position j as a function of (key_j - threshold). Aggregate over q>=1, all j.
    gaps = np.arange(-40, 41)
    prof = {f"L{li}H{h}": {g: [] for g in gaps} for (li, h) in heads}
    for (li, h) in heads:
        A = caps[li]["attn"][:, h]
        for q in range(1, p):
            qpos = sep + 2 * q
            theta = sk[:, q - 1]                             # [B]
            a = A[:, qpos]                                   # [B,K]
            for jj, kc in enumerate(key_cols):
                gap = (keys[:, jj] - theta).cpu().numpy()
                av = a[:, kc].cpu().numpy()
                for g in np.unique(gap):
                    if -40 <= g <= 40:
                        prof[f"L{li}H{h}"][g].extend(av[gap == g].tolist())
    prof_mean = {hd: np.array([np.mean(prof[hd][g]) if prof[hd][g] else np.nan
                               for g in gaps]) for hd in prof}
    # (2b) Is L1H1's (weak) input-key attention a selective pointer to the next key?
    #      Among input keys strictly above threshold, is the argmax-attended one the
    #      true next key (= the smallest such key)?
    sel_hits = sel_tot = 0
    frac_on_next = []
    A11 = caps[1]["attn"][:, 1]                             # L1H1
    for q in range(1, p):
        qpos = sep + 2 * q
        theta = sk[:, q - 1]                                # [B]
        a_in = A11[:, qpos][:, key_cols]                    # [B,p] attn to input keys
        above = keys > theta[:, None]                       # [B,p] candidate mask
        nxt = sk[:, q]                                      # [B] the true next key
        for b in range(B):
            cand = above[b]
            if cand.sum() == 0:
                continue
            am = a_in[b].clone()
            am[~cand] = -1
            picked_key = keys[b, am.argmax()]
            sel_hits += int(picked_key.item() == nxt[b].item())
            sel_tot += 1
            # fraction of above-threshold input-key attention that sits on the next key
            tot = a_in[b][cand].sum().item()
            if tot > 1e-9:
                on_next = a_in[b][keys[b] == nxt[b]].sum().item()
                frac_on_next.append(on_next / tot)
    pointer = dict(argmax_is_next_key=sel_hits / max(1, sel_tot),
                   mean_frac_attn_on_next=float(np.mean(frac_on_next)),
                   n=sel_tot)
    # (3) key-order axis: does a linear direction in key embeddings encode key value?
    WE = model.token_embedding.weight[:cs.KEY_ALPH].detach().cpu().numpy()  # [50,48]
    kv = np.arange(cs.KEY_ALPH)
    # best single direction: regress key value on embedding, report correlation of fit
    from numpy.linalg import lstsq
    Xc = WE - WE.mean(0)
    w, *_ = lstsq(Xc, kv - kv.mean(), rcond=None)
    fit = Xc @ w
    r_full = np.corrcoef(fit, kv)[0, 1]
    # per-PC correlation
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc_r = [abs(np.corrcoef(U[:, i], kv)[0, 1]) for i in range(min(8, U.shape[1]))]
    return where, gaps, prof_mean, pointer, dict(r_full_fit=float(r_full),
                                                 best_pc_r=float(max(pc_r)),
                                                 best_pc=int(np.argmax(pc_r)))


def main():
    apply_style()
    model, _ = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
    where, gaps, prof, pointer, order = analyze(model)
    print("=== where key-step attention goes (mean mass) ===")
    for hd, w in where.items():
        print(f"  {hd}: input_keys={w['input_keys']:.3f} SEP={w['SEP']:.3f} "
              f"emitted_output={w['emitted_output']:.3f}")
    print("=== L1H1 weak-pointer selectivity ===", pointer)
    print("=== key-order axis ===", order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"L0H0": BLUE, "L0H1": GREEN, "L1H0": ORANGE, "L1H1": VIOLET}
    for hd, pm in prof.items():
        ax1.plot(gaps, pm, "-", color=colors[hd], lw=1.8, label=hd)
    ax1.axvline(0, color=INK_2, lw=1, ls="--")
    ax1.set_xlim(-25, 25)
    ax1.set_xlabel("candidate key − threshold (last emitted key)")
    ax1.set_ylabel("mean attention to that input key")
    ax1.set_title(f"L1H1's faint input-key attention peaks just above threshold\n"
                  f"(its argmax = the true next key {pointer['argmax_is_next_key']:.0%} of the time)")
    ax1.legend(fontsize=8)
    # order axis scatter
    WE = model.token_embedding.weight[:cs.KEY_ALPH].detach().cpu().numpy()
    Xc = WE - WE.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    kv = np.arange(cs.KEY_ALPH)
    pc = order["best_pc"]
    ax2.scatter(kv, U[:, pc], c=kv, cmap="viridis", s=30)
    ax2.set_xlabel("key value (token id)")
    ax2.set_ylabel(f"embedding PC{pc+1} (best order axis)")
    ax2.set_title(f"Key value on the best embedding axis\n(single-axis |r|={order['best_pc_r']:.2f}; "
                  f"distributed code, no clean 1-D line)")
    fig.savefig(FIG / "walk_compare.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/walk_compare.png")


if __name__ == "__main__":
    main()
