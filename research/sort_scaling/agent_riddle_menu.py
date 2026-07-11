"""The "menu" half of the sort riddle: what the SEP residual encodes about the
present-key SET, and whether that code supports a thresholded "smallest present
key > theta" readout.

Establishes four things (see agent_findings/riddle_menu.md):
  1. Composition  - is the SEP residual (block-0 output at SEP) an additive/linear
                    function of the present-key multi-hot? R^2, and is it the sum of
                    per-key codes tied to the key embeddings' L0 OV image?
  2. Presence     - per-key decode "is key k present?" from the SEP residual (AUC),
                    uniform across the key range or not.
  3. Threshold    - the crux: a FIXED readout on [SEP menu, one-hot(theta)] predicting
                    "smallest present key > theta"; sweep theta; linear vs MLP; vs a
                    multi-hot oracle and a theta-only control.
  4. L0 MLP       - do tasks 1-3 improve from before (x_mid) to after (x_out) the L0 MLP
                    at the SEP position?

Analysis length p=12 (VRAM-safe). All numbers printed and dumped to
agent_findings/riddle_menu_metrics.json. Figures -> reports/figures/riddle_menu_*.png.

Usage: python research/sort_scaling/agent_riddle_menu.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, SEQ_BLUE, DIVERGING, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load  # noqa: E402
from walkthrough import full_capture  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
OUT = HERE / "agent_findings"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.RandomState(0)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/" + name)


# --------------------------------------------------------------------------- #
# Data collection: SEP residual before/after L0 MLP + present-key sets.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def collect(model, p, n_total, batch=512, seed0=0):
    """Return dict with xmid, xout (SEP residual before/after L0 MLP) [N,48] and
    keys [N,p] (present key ids), gathered over several forward passes."""
    xmids, xouts, keyss = [], [], []
    got = 0
    s = seed0
    sep = 2 * p
    while got < n_total:
        b = min(batch, n_total - got)
        torch.manual_seed(s); s += 1
        seq, _ = cs.make_batch(b, p, DEV, fixed_len=p)
        keys = seq[:, 0:2 * p:2]
        _, caps = full_capture(model, seq[:, :-1])
        xmids.append(caps[0]["x_mid"][:, sep].cpu())
        xouts.append(caps[0]["x_out"][:, sep].cpu())
        keyss.append(keys.cpu())
        got += b
        torch.cuda.empty_cache()
    return dict(xmid=torch.cat(xmids).numpy(),
                xout=torch.cat(xouts).numpy(),
                keys=torch.cat(keyss).numpy())


def multihot(keys, K=cs.KEY_ALPH):
    N = keys.shape[0]
    m = np.zeros((N, K), dtype=np.float32)
    m[np.arange(N)[:, None], keys] = 1.0
    return m


# --------------------------------------------------------------------------- #
# Task 1: composition (additive-set structure).
# --------------------------------------------------------------------------- #
def held_out_r2(X, Y, seed=0):
    """R^2 of a ridge map X->Y on a held-out split (overall, variance-weighted)."""
    n = len(X); idx = np.random.RandomState(seed).permutation(n); cut = int(0.75 * n)
    tr, te = idx[:cut], idx[cut:]
    reg = Ridge(alpha=1.0).fit(X[tr], Y[tr])
    pred = reg.predict(X[te])
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot, reg


def task_composition(model, data, p):
    X = data["xout"]; mh = multihot(data["keys"])
    # (a) additive per-key code: SEP ~ multi-hot (each column = per-key vector u_k)
    r2_add, reg_add = held_out_r2(mh, X)
    # (b) is it specifically the SUM of the L0 OV image of the key embeddings?
    #     build sum over present keys of OV0-image of (folded) key embedding, then
    #     allow one fixed 48->48 linear map (rank<=48) instead of 50 free vectors.
    ov_img = l0_ov_key_image(model)                      # [50,48], OV image per key id
    sum_ov = mh @ ov_img                                 # [N,48]
    r2_ovsum, _ = held_out_r2(sum_ov, X)
    # control: fixed sum of raw (folded) key embeddings through a free linear map
    emb_img = folded_key_embeddings(model)               # [50,48]
    sum_emb = mh @ emb_img
    r2_embsum, _ = held_out_r2(sum_emb, X)
    # cosine of learned per-key code u_k to OV image, per key
    U = reg_add.coef_.T                                  # [50,48] u_k rows
    cos = np.array([_cos(U[k], ov_img[k]) for k in range(cs.KEY_ALPH)])
    return dict(r2_multihot=float(r2_add), r2_ov_sum=float(r2_ovsum),
                r2_emb_sum=float(r2_embsum),
                cos_uk_ovimage_mean=float(np.nanmean(cos)),
                cos_uk_ovimage=cos, U=U)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-9))


def _fold_ln(E, ln):
    """Fold a LayerNorm gain into embeddings: e -> gamma*(e-mean(e)). Drops 1/std
    and beta (standard argmax-preserving approximation, see qkov.md)."""
    g = ln.weight.detach().cpu().numpy()
    Ec = E - E.mean(1, keepdims=True)
    return Ec * g[None, :]


def folded_key_embeddings(model):
    E = model.token_embedding.weight.detach().cpu().numpy()[:cs.KEY_ALPH]  # [50,48]
    return _fold_ln(E, model.blocks[0].ln1)


def l0_ov_key_image(model):
    """Approx L0 OV image of each key token, summed over the two heads, averaged over
    input key POSITIONS (positions differ per sequence; take the mean position code).
    Uses folded ln1 on the token embedding + mean key-position embedding."""
    d = model.cfg["d_model"]; H = model.cfg["n_heads"]; w = d // H
    W_in = model.blocks[0].attn.in_proj_weight.detach().cpu().numpy()   # [3d,d]
    W_o = model.blocks[0].attn.out_proj.weight.detach().cpu().numpy()   # [d,d]
    Wv = W_in[2 * d:3 * d]                                              # [d,d]
    g = model.blocks[0].ln1.weight.detach().cpu().numpy()
    b = model.blocks[0].ln1.bias.detach().cpu().numpy()
    E = model.token_embedding.weight.detach().cpu().numpy()[:cs.KEY_ALPH]  # [50,48]
    pos = model.position_embedding.detach().cpu().numpy()[0]            # [seq,48]
    # mean key-slot position embedding over even input positions up to 2*30
    kpos = pos[0:60:2].mean(0)                                          # [48]
    imgs = []
    for k in range(cs.KEY_ALPH):
        x = E[k] + kpos
        xc = (x - x.mean()) * g + b                                     # folded ln (keep beta here for realism)
        v = Wv @ xc                                                     # [d]
        o = W_o @ v                                                     # [d]
        imgs.append(o)
    return np.stack(imgs)


# --------------------------------------------------------------------------- #
# Task 2: per-key presence decode.
# --------------------------------------------------------------------------- #
def task_presence(data, feat="xout"):
    X = data[feat]; keys = data["keys"]
    n = len(X); idx = RNG.permutation(n); cut = int(0.75 * n)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    mh = multihot(keys)
    aucs, accs = [], []
    for k in range(cs.KEY_ALPH):
        y = mh[:, k]
        if y[tr].sum() < 5 or y[tr].sum() > len(tr) - 5:
            aucs.append(np.nan); accs.append(np.nan); continue
        clf = LogisticRegression(max_iter=500, C=1.0).fit(Xs[tr], y[tr])
        prob = clf.predict_proba(Xs[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        accs.append(clf.score(Xs[te], y[te]))
    return dict(auc=np.array(aucs), acc=np.array(accs),
                auc_mean=float(np.nanmean(aucs)), acc_mean=float(np.nanmean(accs)))


# --------------------------------------------------------------------------- #
# Task 3: thresholded readout (the crux).
# --------------------------------------------------------------------------- #
def build_threshold_dataset(data, feat="xout", thetas=None):
    """For each sequence and each theta, target = smallest present key > theta
    (class 50 = 'none'). Returns features/labels stacked over (seq, theta)."""
    X = data[feat]; keys = data["keys"]; K = cs.KEY_ALPH
    if thetas is None:
        thetas = list(range(-1, K - 1))          # -1 .. 48  (50 values)
    mh = multihot(keys)
    # smallest present key > theta, vectorised per theta
    keys_sorted = np.sort(keys, axis=1)          # [N,p] ascending
    rows_feat, rows_theta, rows_y = [], [], []
    for th in thetas:
        gt = keys_sorted > th                    # [N,p]
        has = gt.any(1)
        first = np.where(has, np.argmax(gt, axis=1), 0)
        y = np.where(has, keys_sorted[np.arange(len(keys_sorted)), first], K)  # K=none
        rows_feat.append(X)
        rows_theta.append(np.full(len(X), th + 1))   # onehot index 0..49
        rows_y.append(y)
    F = np.concatenate(rows_feat)
    T = np.concatenate(rows_theta).astype(int)
    Y = np.concatenate(rows_y).astype(int)
    Toh = np.zeros((len(T), K), dtype=np.float32)
    Toh[np.arange(len(T)), T] = 1.0
    return F, Toh, Y, mh, np.concatenate(rows_theta)


def _split_by_seq(nseq, ntheta, seed=0):
    """Split so a sequence's rows are entirely train or test (no leakage)."""
    idx = np.random.RandomState(seed).permutation(nseq); cut = int(0.75 * nseq)
    tr_seq, te_seq = set(idx[:cut].tolist()), set(idx[cut:].tolist())
    seq_of = np.tile(np.arange(nseq), ntheta)
    tr = np.array([i for i in range(len(seq_of)) if seq_of[i] in tr_seq])
    te = np.array([i for i in range(len(seq_of)) if seq_of[i] in te_seq])
    return tr, te


def build_threshold_onlattice(data, feat="xout"):
    """Faithful regime: theta is drawn only from the present keys (plus -1), exactly
    as in generation where theta = the previously EMITTED key. For each sequence:
    theta in {-1} U {present keys except the max}; target = smallest present key >
    theta (always valid). Uniform p thetas per sequence."""
    X = data[feat]; keys = data["keys"]; K = cs.KEY_ALPH
    ks = np.sort(keys, axis=1)                    # [N,p] ascending
    N, p = ks.shape
    feats, thetas, ys, seqid = [], [], [], []
    for i in range(N):
        present = ks[i]
        cand = np.concatenate([[-1], present[:-1]])   # -1 and every non-max key
        for th in cand:
            succ = present[present > th][0]
            feats.append(X[i]); thetas.append(th + 1); ys.append(succ); seqid.append(i)
    F = np.stack(feats); T = np.array(thetas, int); Y = np.array(ys, int)
    Toh = np.zeros((len(T), K), dtype=np.float32); Toh[np.arange(len(T)), T] = 1.0
    mh = multihot(keys)[np.array(seqid)]
    return F, Toh, Y, mh, np.array(thetas) - 1, np.array(seqid)


def task_threshold(data, feat="xout", regime="uniform"):
    K = cs.KEY_ALPH
    if regime == "uniform":
        F, Toh, Y, mh, theta_col = build_threshold_dataset(data, feat)
        nseq = len(data["xout"]); ntheta = len(Y) // nseq
        tr, te = _split_by_seq(nseq, ntheta)
        oracle_mh = mh_tiled(mh, ntheta)
    else:  # on-lattice (faithful): theta = a present key, as in generation
        F, Toh, Y, mh, theta_col, seqid = build_threshold_onlattice(data, feat)
        nseq = len(data["xout"])
        idx = np.random.RandomState(0).permutation(nseq); cut = int(0.75 * nseq)
        tr_seq = set(idx[:cut].tolist())
        mask_tr = np.array([s in tr_seq for s in seqid])
        tr = np.where(mask_tr)[0]; te = np.where(~mask_tr)[0]
        oracle_mh = mh
    # standardise menu features on train
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Fs = (F - mu) / sd
    menu = np.concatenate([Fs, Toh], 1)              # [., 48+50]
    oracle = np.concatenate([oracle_mh, Toh], 1)     # multi-hot + theta
    only = Toh                                        # theta only (control)

    def run(Xf, kind="lin"):
        if kind == "lin":
            clf = LogisticRegression(max_iter=300, C=1.0)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=120,
                                random_state=0, early_stopping=True)
        clf.fit(Xf[tr], Y[tr])
        pred = clf.predict(Xf[te])
        acc = float((pred == Y[te]).mean())
        # per-theta accuracy on test
        tt = theta_col[te]
        pa = {}
        for th in np.unique(tt):
            m = tt == th
            pa[float(th)] = float((pred[m] == Y[te][m]).mean())
        return acc, pa

    res = {}
    res["menu_lin"] = run(menu, "lin")
    res["menu_mlp"] = run(menu, "mlp")
    res["oracle_lin"] = run(oracle, "lin")
    res["oracle_mlp"] = run(oracle, "mlp")
    res["theta_only"] = run(only, "lin")
    return res


def mh_tiled(mh, ntheta):
    return np.tile(mh, (ntheta, 1))


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def fig_composition(comp, comp_mid):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    names = ["multi-hot\n(50 free codes)", "sum of L0 OV\nkey images",
             "sum of key\nembeddings"]
    vals = [comp["r2_multihot"], comp["r2_ov_sum"], comp["r2_emb_sum"]]
    ax.bar(np.arange(3), vals, color=[BLUE, VIOLET, INK_2], zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("held-out $R^2$ (SEP residual)")
    ax.set_title("SEP residual is an additive function\nof the present-key set")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax = axes[1]
    cos = comp["cos_uk_ovimage"]
    ax.plot(np.arange(cs.KEY_ALPH), cos, "o", color=VIOLET, ms=4, zorder=3)
    ax.axhline(np.nanmean(cos), color=ORANGE, lw=1.2,
               label=f"mean {np.nanmean(cos):.2f}")
    ax.set_xlabel("key id"); ax.set_ylabel("cos(per-key code, L0 OV image)")
    ax.set_title("Per-key code aligns with the\nL0 attention OV image of the key")
    ax.legend(fontsize=8)
    save(fig, "riddle_menu_composition.png")


def fig_presence(pres_out, pres_mid):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(np.arange(cs.KEY_ALPH), pres_out["auc"], "-o", color=BLUE, ms=4,
            label=f"after L0 MLP (mean AUC {pres_out['auc_mean']:.3f})", zorder=3)
    ax.plot(np.arange(cs.KEY_ALPH), pres_mid["auc"], "-o", color=INK_2, ms=3, alpha=0.7,
            label=f"before L0 MLP (mean AUC {pres_mid['auc_mean']:.3f})", zorder=2)
    ax.axhline(0.5, color=INK_2, lw=0.8, ls="--")
    ax.set_ylim(0.45, 1.02); ax.set_xlabel("key id")
    ax.set_ylabel("presence decode AUC (held-out)")
    ax.set_title("Is key k present? — decoded per key from the SEP residual")
    ax.legend(fontsize=8)
    save(fig, "riddle_menu_presence.png")


def fig_threshold(thr_uni, thr_lat):
    labels = [("menu_lin", "menu (linear)", BLUE),
              ("menu_mlp", "menu (MLP)", GREEN),
              ("oracle_lin", "multi-hot oracle (linear)", VIOLET),
              ("oracle_mlp", "multi-hot oracle (MLP)", ORANGE),
              ("theta_only", "θ only (control)", INK_2)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(labels)); w = 0.4
    uni = [thr_uni[k][0] for k, _, _ in labels]
    lat = [thr_lat[k][0] for k, _, _ in labels]
    ax = axes[0]
    ax.bar(x - w / 2, uni, w, color=INK_2, zorder=3, label="all θ (uniform 0–49)")
    ax.bar(x + w / 2, lat, w, color=BLUE, zorder=3,
           label="θ = a present key (model's regime)")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in labels], rotation=25,
                                         ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy")
    ax.set_title("Predict 'smallest present key > θ'")
    for i, (a, b) in enumerate(zip(uni, lat)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    for k, l, c in labels:
        pa = thr_lat[k][1]
        xs = sorted(pa); ys = [pa[x] for x in xs]
        ax.plot([x for x in xs], ys, "-o", color=c, lw=1.6, ms=3, label=l)
    ax.set_xlabel("threshold θ (a present key)"); ax.set_ylabel("accuracy at θ")
    ax.set_ylim(0, 1.05); ax.set_title("Accuracy vs threshold (model's regime)")
    ax.legend(fontsize=7, loc="lower center")
    save(fig, "riddle_menu_threshold.png")


def fig_mlp_delta(comp_out, comp_mid, pres_out, pres_mid, thr_out, thr_mid):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    tasks = ["composition\n$R^2$", "presence\nmean AUC", "threshold\nmenu-linear acc"]
    before = [comp_mid["r2_multihot"], pres_mid["auc_mean"], thr_mid["menu_lin"][0]]
    after = [comp_out["r2_multihot"], pres_out["auc_mean"], thr_out["menu_lin"][0]]
    x = np.arange(len(tasks))
    ax.bar(x - 0.2, before, 0.4, label="before L0 MLP (x_mid)", color=INK_2, zorder=3)
    ax.bar(x + 0.2, after, 0.4, label="after L0 MLP (x_out)", color=BLUE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylim(0, 1.05)
    ax.set_ylabel("metric"); ax.set_title("What the L0 MLP adds at the SEP position")
    ax.legend(fontsize=8)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
    save(fig, "riddle_menu_mlp.png")


# --------------------------------------------------------------------------- #
def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    model, meta = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
    p = 12
    print(f"cfg={model.cfg} length p={p}")
    data = collect(model, p, n_total=2048, batch=512)
    print("collected", data["xout"].shape, "sequences")

    comp_out = task_composition(model, data, p)
    # task 1 before MLP (uses xmid; OV/emb comparisons are for xout only)
    data_mid = dict(xout=data["xmid"], xmid=data["xmid"], keys=data["keys"])
    comp_mid = task_composition(model, data_mid, p)
    print("composition xout:", {k: round(v, 3) for k, v in comp_out.items()
                                if isinstance(v, float)})
    print("composition xmid R2:", round(comp_mid["r2_multihot"], 3))

    pres_out = task_presence(data, "xout")
    pres_mid = task_presence(data, "xmid")
    print("presence xout mean AUC:", round(pres_out["auc_mean"], 3),
          "xmid:", round(pres_mid["auc_mean"], 3))

    thr_out = task_threshold(data, "xout", regime="uniform")
    thr_mid = task_threshold(data, "xmid", regime="uniform")
    print("threshold[uniform] xout:", {k: round(v[0], 3) for k, v in thr_out.items()})
    print("threshold[uniform] xmid menu_lin:", round(thr_mid["menu_lin"][0], 3))

    thr_out_lat = task_threshold(data, "xout", regime="onlattice")
    thr_mid_lat = task_threshold(data, "xmid", regime="onlattice")
    print("threshold[onlattice] xout:", {k: round(v[0], 3) for k, v in thr_out_lat.items()})
    print("threshold[onlattice] xmid menu_lin:", round(thr_mid_lat["menu_lin"][0], 3))

    fig_composition(comp_out, comp_mid)
    fig_presence(pres_out, pres_mid)
    fig_threshold(thr_out, thr_out_lat)
    fig_mlp_delta(comp_out, comp_mid, pres_out, pres_mid, thr_out_lat, thr_mid_lat)

    def clean(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in d.items() if k not in ("U",)}
    metrics = dict(
        length=p, n_seq=int(data["xout"].shape[0]),
        composition_xout=clean(comp_out), composition_xmid=clean(comp_mid),
        presence_xout=clean(pres_out), presence_xmid=clean(pres_mid),
        threshold_uniform_xout={k: {"acc": v[0], "per_theta": v[1]} for k, v in thr_out.items()},
        threshold_uniform_xmid={k: {"acc": v[0]} for k, v in thr_mid.items()},
        threshold_onlattice_xout={k: {"acc": v[0], "per_theta": v[1]} for k, v in thr_out_lat.items()},
        threshold_onlattice_xmid={k: {"acc": v[0]} for k, v in thr_mid_lat.items()},
    )
    (OUT / "riddle_menu_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", OUT / "riddle_menu_metrics.json")


if __name__ == "__main__":
    main()
