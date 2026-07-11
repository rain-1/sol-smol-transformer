"""Representation-geometry study of the 2-layer causal pair-sort decoder.

Covers: embedding/unembedding geometry, positional geometry, the sort "frontier"
(linear probes for last-emitted key, next key value, count emitted/remaining),
value-copy geometry (logit-lens on the induction OV path), and content-vs-position.

Run: python research/sort_scaling/agent_geometry.py
Writes figures to reports/figures/agent_geom_*.png and prints numbers used in
agent_findings/geometry.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, SEQ_BLUE, DIVERGING, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load, run_capture, batch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = HERE / "checkpoints" / "causal_pairs_L30.pt"

KEY_IDS = np.arange(0, cs.KEY_ALPH)          # 0..49
VAL_IDS = np.arange(cs.KEY_ALPH, cs.SEP)     # 50..69


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/" + name)


def bare(ax):
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# 1. Embedding & unembedding geometry
# --------------------------------------------------------------------------- #
def embedding_geometry(model):
    E = model.token_embedding.weight.detach().cpu().numpy()   # [73,48]
    U = model.output.weight.detach().cpu().numpy()            # [73,48]
    out = {}

    def analyse(M, name):
        # center over all real tokens (drop PAD which may be untrained/outlier)
        real = np.arange(cs.SEP + 2)  # 0..71 (exclude PAD=72)
        Mc = M[real] - M[real].mean(0)
        U_, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        coords = Mc @ Vt.T  # [n, k] pc coords, indexed by real-token id
        # key vs value separation: is there any direction linearly separating them?
        keymask = real < cs.KEY_ALPH
        valmask = (real >= cs.KEY_ALPH) & (real < cs.SEP)
        # LDA-ish: logistic on full embedding, key vs value
        from sklearn.linear_model import LogisticRegression as LR
        y = np.where(keymask, 0, np.where(valmask, 1, 2))
        kv = keymask | valmask
        # cross-validated key-vs-value separability (guards against overfit: 48 dims)
        sep_acc = float(cross_val_score(LR(max_iter=2000), M[real][kv], y[kv], cv=5).mean())
        # correlate each of top PCs with key numeric value (id 0..49)
        kcoords = coords[:cs.KEY_ALPH]  # key rows in same order as id
        cors = [np.corrcoef(kcoords[:, k], KEY_IDS)[0, 1] for k in range(min(6, coords.shape[1]))]
        best_pc = int(np.argmax(np.abs(cors)))
        best_cor = cors[best_pc]
        # dedicated 1-D fit: regress key id onto full embedding. In-sample r is
        # meaningless (50 pts, 48 dims -> interpolation); report CV r instead.
        from sklearn.model_selection import cross_val_predict
        pred_cv = cross_val_predict(LinearRegression(), M[:cs.KEY_ALPH], KEY_IDS, cv=5)
        r_fit_cv = float(np.corrcoef(pred_cv, KEY_IDS)[0, 1])
        out[name] = dict(sv=S[:8].tolist(), sep_acc=float(sep_acc),
                         pc_key_corrs=[float(c) for c in cors], best_pc=best_pc,
                         best_cor=float(best_cor), r_fit_cv=r_fit_cv)
        return Mc, coords, cors, best_pc

    Ec, ecoords, ecors, ebest = analyse(E, "embedding")
    Uc, ucoords, ucors, ubest = analyse(U, "unembedding")

    # figure: PCA scatter (key vs value colored) + key-id gradient along best axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    real = np.arange(cs.SEP + 2)
    kc = ecoords[:cs.KEY_ALPH]
    vc = ecoords[cs.KEY_ALPH:cs.SEP]
    sc = ax.scatter(kc[:, 0], kc[:, 1], c=KEY_IDS, cmap=SEQ_BLUE, s=45, label="keys (0-49)", zorder=3)
    ax.scatter(vc[:, 0], vc[:, 1], c=ORANGE, marker="^", s=55, label="values (50-69)", zorder=3)
    ax.scatter(ecoords[cs.SEP, 0], ecoords[cs.SEP, 1], c=GREEN, marker="*", s=180, label="SEP")
    ax.scatter(ecoords[cs.SEP + 1, 0], ecoords[cs.SEP + 1, 1], c=VIOLET, marker="P", s=120, label="EOS")
    ax.set_xlabel("emb PC1"); ax.set_ylabel("emb PC2")
    ax.set_title("Token embedding PCA: keys vs values"); ax.legend(fontsize=7)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="key id")

    ax = axes[1]
    proj = ecoords[:cs.KEY_ALPH, ebest]
    ax.scatter(KEY_IDS, proj, c=KEY_IDS, cmap=SEQ_BLUE, s=45, zorder=3)
    ax.set_xlabel("key numeric id (0-49)")
    ax.set_ylabel(f"embedding PC{ebest+1} coord")
    ax.set_title(f"Embedding order axis (r={ecors[ebest]:+.2f})")

    ax = axes[2]
    uproj = ucoords[:cs.KEY_ALPH, ubest]
    ax.scatter(KEY_IDS, uproj, c=KEY_IDS, cmap=SEQ_BLUE, s=45, zorder=3)
    ax.set_xlabel("key numeric id (0-49)")
    ax.set_ylabel(f"unembedding PC{ubest+1} coord")
    ax.set_title(f"Unembedding order axis (r={ucors[ubest]:+.2f})")
    fig.suptitle("Embedding geometry: key subspace and a numeric-order direction",
                 fontsize=13, fontweight="bold", y=1.03)
    save(fig, "agent_geom_embedding.png")
    return out


# --------------------------------------------------------------------------- #
# 2. Positional embedding geometry
# --------------------------------------------------------------------------- #
def positional_geometry(model, p=30):
    P = model.position_embedding.detach().cpu().numpy()[0]   # [122,48]
    Pc = P - P.mean(0)
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    coords = Pc @ Vt.T
    L = P.shape[0]
    positions = np.arange(L)
    sep = 2 * p  # =60 for p=30 (SEP index); input region 0..59, output 61..

    # Does position encode parity (key vs value slot)? even input pos = key slot.
    inreg = positions < sep
    parity = positions % 2  # 0 even
    # Does position encode "which rank slot" q in output? output key positions sep+2q
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(np.arange(8), S[:8], "-o", color=BLUE)
    ax.set_xlabel("PC index"); ax.set_ylabel("singular value")
    ax.set_title("Position embedding spectrum")

    ax = axes[1]
    for k in range(4):
        ax.plot(positions, coords[:, k], lw=1.5, label=f"PC{k+1}")
    ax.axvline(sep, color=INK_2, ls="--", lw=1, label="SEP")
    ax.set_xlabel("sequence position"); ax.set_ylabel("PC coordinate")
    ax.set_title("Positional PCs vs position (low-freq curves?)"); ax.legend(fontsize=7)

    ax = axes[2]
    ax.scatter(coords[:, 0], coords[:, 1], c=positions, cmap=SEQ_BLUE, s=25, zorder=3)
    ax.scatter(coords[sep, 0], coords[sep, 1], c="red", marker="*", s=160, label="SEP")
    ax.set_xlabel("pos PC1"); ax.set_ylabel("pos PC2")
    ax.set_title("Positional manifold"); ax.legend(fontsize=7)
    fig.suptitle("Positional embedding geometry", fontsize=13, fontweight="bold", y=1.03)
    save(fig, "agent_geom_positional.png")

    # Quantify: probe absolute position from pos-emb (trivially high). More useful:
    # can we linearly decode (a) parity (key/value slot) (b) region (in/out)
    # (c) rank-slot q at output key positions?
    def lin_probe_cls(X, y):
        clf = LogisticRegression(max_iter=3000)
        return float(cross_val_score(clf, X, y, cv=4).mean())

    def lin_probe_reg(X, y):
        reg = LinearRegression()
        pred = reg.fit(X, y).predict(X)
        return float(np.corrcoef(pred, y)[0, 1])

    res = {}
    res["sv"] = S[:8].tolist()
    res["parity_acc"] = lin_probe_cls(P, parity)
    res["region_acc"] = lin_probe_cls(P, inreg.astype(int))
    # rank-slot at output key positions
    okpos = np.array([sep + 2 * q for q in range(p)])
    okq = np.arange(p)
    res["outkey_rankslot_r"] = lin_probe_reg(P[okpos], okq)
    # rank-slot at input key positions (should also just be linear in pos)
    ikpos = np.array([2 * q for q in range(p)])
    res["inkey_slot_r"] = lin_probe_reg(P[ikpos], np.arange(p))
    # smoothness: correlation of adjacent position embeddings
    cos_adj = np.mean([np.dot(Pc[i], Pc[i+1]) /
                       (np.linalg.norm(Pc[i]) * np.linalg.norm(Pc[i+1]) + 1e-9)
                       for i in range(L - 1)])
    res["mean_adjacent_cos"] = float(cos_adj)
    return res


# --------------------------------------------------------------------------- #
# 3. Sort frontier probes
# --------------------------------------------------------------------------- #
def _probe_reg(X, y):
    """CV R^2 and pearson r of a ridge-like linear regression."""
    reg = LinearRegression()
    from sklearn.model_selection import cross_val_predict
    pred = cross_val_predict(reg, X, y, cv=4)
    r = np.corrcoef(pred, y)[0, 1]
    ss = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r), float(ss)


def _probe_cls(X, y):
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, X, y, cv=4).mean())


def _partial_out_q(X, y, qidx):
    """Remove the per-rank-slot (q) mean from features and label. This kills the
    positional confound: at a key-emit step the sorted key sk_q is monotone in q,
    and position encodes q, so an uncontrolled probe reads the rank-slot prior, not
    content. After centering within each q, any residual predictability is genuine
    content (the actual key value at the frontier, beyond its positional prior)."""
    Xc = X.copy().astype(float)
    yc = y.copy().astype(float)
    for q in np.unique(qidx):
        mrows = qidx == q
        Xc[mrows] -= Xc[mrows].mean(0)
        yc[mrows] -= yc[mrows].mean()
    return Xc, yc


def frontier_probes(model, p=20, n=512):
    seq, keys, order = batch(model, p, n=n)
    _, _, res = run_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    # sorted keys / values sequence
    skeys = keys.gather(1, order)                 # [B,p] rank-ordered key ids
    # key-emitting query positions: predict sk_q at pos sep+2q (q=0..p-1)
    kq = [sep + 2 * q for q in range(p)]
    stages = ["embed", "block0", "block1", "final"]
    # raw probes (position-confounded) and q-controlled probes (genuine content)
    out = {"next_key": {}, "last_key": {}, "count": {},
           "next_key_ctrl": {}, "last_key_ctrl": {}}
    next_key = skeys.cpu().numpy()                              # sk_q
    last_key = np.concatenate([-np.ones((B, 1)), skeys.cpu().numpy()[:, :-1]], axis=1)
    count = np.tile(np.arange(p), (B, 1)).astype(float)
    qidx = np.tile(np.arange(p), (B, 1)).reshape(-1)
    mq1 = (np.tile(np.arange(p), (B, 1)) >= 1).reshape(-1)     # q>=1 for last_key
    for st in stages:
        R = res[st][:, kq].cpu().numpy()   # [B,p,d]
        d = R.shape[-1]
        Xall = R.reshape(-1, d)
        out["next_key"][st] = _probe_reg(Xall, next_key.reshape(-1))
        out["last_key"][st] = _probe_reg(Xall[mq1], last_key.reshape(-1)[mq1])
        out["count"][st] = _probe_reg(Xall, count.reshape(-1))
        # position-controlled: partial out q, then probe residual content
        Xc, yc = _partial_out_q(Xall, next_key.reshape(-1), qidx)
        out["next_key_ctrl"][st] = _probe_reg(Xc, yc)
        Xc2, yc2 = _partial_out_q(Xall[mq1], last_key.reshape(-1)[mq1], qidx[mq1])
        out["last_key_ctrl"][st] = _probe_reg(Xc2, yc2)
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(stages))
    ax = axes[0]
    ax.plot(x, [out["next_key"][s][0] for s in stages], "-o", color=BLUE, lw=2, ms=7,
            label="next key value sk_q")
    ax.plot(x, [out["last_key"][s][0] for s in stages], "-s", color=ORANGE, lw=2, ms=7,
            label="last emitted key sk_{q-1}")
    ax.plot(x, [out["count"][s][0] for s in stages], "-^", color=GREEN, lw=2, ms=7,
            label="count emitted (q)")
    ax.set_ylim(0, 1.02); ax.set_ylabel("linear-probe r (4-fold CV)")
    ax.set_title("Raw (position-confounded)")
    ax.set_xticks(x); ax.set_xticklabels(stages); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(x, [out["next_key_ctrl"][s][0] for s in stages], "-o", color=BLUE, lw=2, ms=7,
            label="next key sk_q | q partialled out")
    ax.plot(x, [out["last_key_ctrl"][s][0] for s in stages], "-s", color=ORANGE, lw=2, ms=7,
            label="last key sk_{q-1} | q partialled out")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylim(-0.05, 1.02); ax.set_ylabel("content-only probe r (within rank slot)")
    ax.set_title("Position-controlled (genuine content of frontier)")
    ax.set_xticks(x); ax.set_xticklabels(stages); ax.legend(fontsize=8)
    fig.suptitle("The sort frontier: what is linearly present at key-emit steps",
                 fontsize=13, fontweight="bold", y=1.03)
    save(fig, "agent_geom_frontier.png")
    return out


# --------------------------------------------------------------------------- #
# 4. Value-copy geometry (logit lens on induction OV output)
# --------------------------------------------------------------------------- #
def value_copy_geometry(model, p=20, n=256):
    seq, keys, order = batch(model, p, n=n)
    logits, attn, res = run_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    # value-emit positions: pos sep+1+2q predicts sv_q
    vq = [sep + 1 + 2 * q for q in range(p)]
    correct_val = seq[:, [sep + 2 + 2 * q for q in range(p)]]   # sv_q token id (50-69)
    U = model.output.weight.detach()                            # [73,48]
    bias = model.output.bias.detach()
    # logit lens: apply final_norm + unembed to block1 residual at value positions
    R1 = res["block1"][:, vq]                                   # [B,p,d]
    normed = model.final_norm(R1)
    lens_logits = normed @ U.T + bias                          # [B,p,73]
    lens_pred = lens_logits.argmax(-1)
    lens_acc = (lens_pred == correct_val).float().mean().item()
    # restrict to value vocab
    lens_acc_valvocab = (lens_logits[..., cs.KEY_ALPH:cs.SEP].argmax(-1) + cs.KEY_ALPH
                         == correct_val).float().mean().item()
    # cosine between block1 residual (post final_norm) and unembed row of correct value
    Un = U / U.norm(dim=-1, keepdim=True)
    normed_n = normed / normed.norm(dim=-1, keepdim=True)
    cos_correct = (normed_n * Un[correct_val]).sum(-1).mean().item()
    # cosine to a random other value token (baseline)
    rand_val = torch.randint(cs.KEY_ALPH, cs.SEP, correct_val.shape, device=DEV)
    cos_rand = (normed_n * Un[rand_val]).sum(-1).mean().item()
    # compare: also compute at block0 to show copy appears at block1
    R0 = model.final_norm(res["block0"][:, vq])
    R0n = R0 / R0.norm(dim=-1, keepdim=True)
    cos_block0 = (R0n * Un[correct_val]).sum(-1).mean().item()
    acc0 = ((R0 @ U.T + bias)[..., cs.KEY_ALPH:cs.SEP].argmax(-1) + cs.KEY_ALPH
            == correct_val).float().mean().item()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    labels = ["block0", "block1"]
    ax.bar([0, 1], [acc0, lens_acc_valvocab], color=[INK_2, BLUE], width=0.55, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("logit-lens value accuracy")
    ax.set_title("Value copy is written at block1")
    for i, v in enumerate([acc0, lens_acc_valvocab]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax = axes[1]
    ax.bar([0, 1, 2], [cos_block0, cos_correct, cos_rand],
           color=[INK_2, GREEN, ORANGE], width=0.55, zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["block1→correct", "block1→correct", "block1→random"], rotation=0, fontsize=8)
    ax.set_xticklabels(["block0\n→correct", "block1\n→correct", "block1\n→random"])
    ax.set_ylabel("cosine (post-final-norm resid, unembed row)")
    ax.set_title("Residual aligns with correct value's unembed direction")
    for i, v in enumerate([cos_block0, cos_correct, cos_rand]):
        ax.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("Value-copy geometry: induction OV path is a copy in representation space",
                 fontsize=12, fontweight="bold", y=1.03)
    save(fig, "agent_geom_valuecopy.png")
    return dict(lens_acc_full=lens_acc, lens_acc_valvocab=lens_acc_valvocab,
                cos_correct=cos_correct, cos_rand=cos_rand, cos_block0=cos_block0,
                block0_acc=acc0)


# --------------------------------------------------------------------------- #
# 5. Content vs position for each head
# --------------------------------------------------------------------------- #
def content_vs_position(model, p=20, n=256):
    """For each head, decompose attention variance into position-explained vs
    content-driven. Attention pattern averaged over the batch at fixed (q,k) grid
    is the position-only prediction; residual std is content-driven."""
    seq, keys, order = batch(model, p, n=n)
    _, attn, _ = run_capture(model, seq[:, :-1])
    sep = 2 * p
    heads = [(li, h) for li in range(len(attn)) for h in range(attn[li].shape[1])]
    out = {}
    for (li, h) in heads:
        a = attn[li][:, h]                        # [B,Q,K]
        mean_a = a.mean(0, keepdim=True)          # position-only prediction
        total_var = a.var(0).mean().item()
        # fraction of attention variance NOT explained by mean (i.e. content)
        resid_var = ((a - mean_a) ** 2).mean().item()
        # by construction resid_var == total_var (mean removed) -> use different metric:
        # ratio of std to mean magnitude, over cells with appreciable mass
        content_sd = a.std(0).mean().item()
        pos_signal = mean_a.abs().mean().item()
        out[f"L{li}H{h}"] = dict(content_sd=float(content_sd),
                                 pos_signal=float(pos_signal),
                                 content_frac=float(content_sd / (content_sd + pos_signal + 1e-9)))
    return out


def content_vs_position_causal(model, p=12, n=200):
    """Cleaner test: hold position fixed, vary content. Take fixed-length batches;
    for each head, measure how much the argmax attended source moves when we permute
    the key identities but keep positions. Report the std of the argmax source index
    across random content (position fixed) -- large => content driven."""
    seq, keys, order = batch(model, p, n=n)
    _, attn, _ = run_capture(model, seq[:, :-1])
    sep = 2 * p
    kq = [sep + 2 * q for q in range(p)]     # key emit
    vq = [sep + 1 + 2 * q for q in range(p)]  # value emit
    out = {}
    for li in range(len(attn)):
        for h in range(attn[li].shape[1]):
            a = attn[li][:, h]
            # at value-emit queries, where argmax lands, std over batch (fixed positions)
            src_v = a[:, vq].argmax(-1).float()      # [B,p]
            src_k = a[:, kq].argmax(-1).float()
            out[f"L{li}H{h}"] = dict(
                argmax_src_std_valstep=float(src_v.std(0).mean().item()),
                argmax_src_std_keystep=float(src_k.std(0).mean().item()))
    return out


def main():
    apply_style()
    model, meta = load(CKPT)
    print(f"loaded cfg={model.cfg}")
    results = {}
    print("\n== 1. Embedding geometry ==")
    results["embedding"] = embedding_geometry(model)
    for k, v in results["embedding"].items():
        print(f"  {k}: sep_acc={v['sep_acc']:.3f} best_pc={v['best_pc']} "
              f"best_cor={v['best_cor']:+.3f} r_fit_cv={v['r_fit_cv']:.3f} sv[:4]={[round(x,2) for x in v['sv'][:4]]}")
    print("\n== 2. Positional geometry ==")
    results["positional"] = positional_geometry(model)
    for k, v in results["positional"].items():
        if k != "sv":
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n== 3. Frontier probes ==")
    results["frontier"] = frontier_probes(model)
    for lab in ["next_key", "last_key", "count", "next_key_ctrl", "last_key_ctrl"]:
        print(f"  {lab}: " + ", ".join(f"{s}={results['frontier'][lab][s][0]:.2f}"
              for s in ["embed", "block0", "block1", "final"]))
    print("\n== 4. Value-copy geometry ==")
    results["valuecopy"] = value_copy_geometry(model)
    for k, v in results["valuecopy"].items():
        print(f"  {k}: {v:.3f}")
    print("\n== 5. Content vs position ==")
    results["content_pos"] = content_vs_position(model)
    for k, v in results["content_pos"].items():
        print(f"  {k}: content_sd={v['content_sd']:.3f} pos_signal={v['pos_signal']:.3f} "
              f"content_frac={v['content_frac']:.3f}")
    results["content_pos_causal"] = content_vs_position_causal(model)
    for k, v in results["content_pos_causal"].items():
        print(f"  {k}: argmax_std_val={v['argmax_src_std_valstep']:.2f} "
              f"argmax_std_key={v['argmax_src_std_keystep']:.2f}")
    import json
    (HERE / "agent_findings").mkdir(exist_ok=True)
    (HERE / "agent_findings" / "geometry_metrics.json").write_text(json.dumps(results, indent=2))
    print("\nwrote agent_findings/geometry_metrics.json")


if __name__ == "__main__":
    main()
