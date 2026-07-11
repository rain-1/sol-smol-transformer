"""The threshold + readout half of the sort riddle.

Reverse-engineers HOW the running-max threshold theta is encoded at the L1 attention
output (caps[1]['x_mid']) and HOW that threshold gates a linear readout to select the
smallest present key strictly above it.

Established upstream (agent_findings/sorthead.md, geometry.md): next key = smallest
input key > running-max of emitted keys; L1H1 reads emitted keys (recency-weighted)
to recover the threshold; SEP carries a static menu of present keys; next key is
decodable ~1.0 from x_mid (before the L1 MLP).

This script answers:
  1. theta direction  - linear-probe theta from x_mid at key-emit steps; dimensionality;
                        is it added ~linearly by L1H1? (head-ablation of x_mid)
  2. theta sweep       - THE decisive experiment: fix a menu, inject different theta
                        (token patch on the emitted key, and x_mid theta-axis shift);
                        confirm the selected next key is a staircase = smallest present > theta.
  3. readout form      - fit logit(k) ~ present/above/gap functional form; argmax match.
  4. linear vs MLP     - is the gating linear at x_mid, or does it need the L1 MLP?

Figures -> reports/figures/riddle_threshold_*.png ; metrics -> agent_findings/riddle_threshold_metrics.json
Usage: python research/sort_scaling/agent_riddle_threshold.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, SEQ_BLUE, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load, batch  # noqa: E402
from walkthrough import full_capture  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
AF = HERE / "agent_findings"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 24  # per-head width


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/" + name)


# --------------------------------------------------------------------------- #
# Manual layer-1 completion from a (possibly patched) x_mid.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def complete_from_xmid(model, x_mid, mlp_off=False):
    """x_mid is the post-L1-attn residual [B,L,d]. Finish the forward -> logits."""
    block = model.blocks[1]
    x_out = x_mid if mlp_off else x_mid + block.mlp(block.ln2(x_mid))
    return model.output(model.final_norm(x_out))


@torch.inference_mode()
def capture_xmid(model, seq, head_off=None):
    """full_capture but optionally zero an L1 head's OV contribution to x_mid.
    head_off: (layer, head) to ablate (only its out_proj slice), or None."""
    if head_off is None:
        _, caps = full_capture(model, seq)
        return caps[1]["x_mid"], caps
    li, h = head_off
    block = model.blocks[li]
    sl = slice(h * W, (h + 1) * W)
    saved = block.attn.out_proj.weight[:, sl].detach().clone()
    block.attn.out_proj.weight[:, sl].zero_()
    _, caps = full_capture(model, seq)
    block.attn.out_proj.weight[:, sl].copy_(saved)
    return caps[1]["x_mid"], caps


# --------------------------------------------------------------------------- #
# Data collection at key-emit steps (q>=1, so a running-max threshold exists).
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def collect(model, p, n, seed, head_off=None):
    """Return dict of arrays gathered over key-emit steps q=1..p-1 (teacher forced)."""
    seq, keys, order = batch(model, p, n=n, seed=seed)
    sep = 2 * p
    x_mid, caps = capture_xmid(model, seq[:, :-1], head_off=head_off)
    logits = complete_from_xmid(model, x_mid)  # identical to model logits (sanity)
    # sorted keys sk_0..sk_{p-1} at seq index sep+1+2j
    sk = seq[:, sep + 1:sep + 1 + 2 * p:2]              # [n,p]  sorted keys
    rows = []
    B = seq.shape[0]
    for q in range(1, p):
        qpos = sep + 2 * q
        theta = sk[:, q - 1]                            # running max of emitted keys = sk_{q-1}
        nxt = sk[:, q]                                  # true next key
        rows.append(dict(
            x=x_mid[:, qpos].float().cpu().numpy(),     # [B,d]
            logit=logits[:, qpos, :cs.KEY_ALPH].float().cpu().numpy(),  # [B,50] key logits
            theta=theta.cpu().numpy(),
            nxt=nxt.cpu().numpy(),
            menu=keys.cpu().numpy(),                    # [B,p] present keys (same each q)
            q=np.full(B, q),
        ))
    out = {k: np.concatenate([r[k] for r in rows], 0) for k in ["x", "logit", "theta", "nxt", "q"]}
    out["menu"] = np.concatenate([r["menu"] for r in rows], 0)  # [N,p]
    torch.cuda.empty_cache()
    return out


def present_matrix(menu, alph=cs.KEY_ALPH):
    """[N, alph] indicator of which keys are present."""
    P = np.zeros((menu.shape[0], alph), dtype=np.float32)
    rows = np.repeat(np.arange(menu.shape[0]), menu.shape[1])
    P[rows, menu.reshape(-1)] = 1.0
    return P


def smallest_present_above(menu_row, theta):
    cand = [k for k in menu_row if k > theta]
    return min(cand) if cand else -1


# --------------------------------------------------------------------------- #
# 1. Threshold direction + dimensionality + additivity from L1H1
# --------------------------------------------------------------------------- #
def cv_r2(X, y, model_ctor=lambda: Ridge(alpha=10.0), k=5):
    kf = KFold(k, shuffle=True, random_state=0)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m = model_ctor().fit((X[tr] - mu) / sd, y[tr])
        preds[te] = m.predict((X[te] - mu) / sd)
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, np.sqrt(((y - preds) ** 2).mean()), preds


def partial_out_q(X, y, q):
    """Center features and label within each rank-slot q (removes the positional
    confound: theta = sk_{q-1} is monotone in q, and q is fully positional)."""
    Xc, yc = X.copy().astype(float), y.copy().astype(float)
    for qq in np.unique(q):
        m = q == qq
        Xc[m] -= Xc[m].mean(0)
        yc[m] -= yc[m].mean()
    return Xc, yc


def part1_theta_direction(model, p=12, n=384):
    print("\n=== Part 1: threshold direction ===")
    full = collect(model, p, n, seed=0)
    noH1 = collect(model, p, n, seed=0, head_off=(1, 1))   # zero L1H1 OV
    noH0 = collect(model, p, n, seed=0, head_off=(1, 0))   # zero L1H0 OV (control)
    theta = full["theta"].astype(float)

    q = full["q"]
    # Raw decode (confounded: theta is monotone in the positional rank-slot q).
    r2_full, rmse_full, _ = cv_r2(full["x"], theta)
    r2_noH1, rmse_noH1, _ = cv_r2(noH1["x"], theta)
    r2_noH0, rmse_noH0, _ = cv_r2(noH0["x"], theta)
    dH1 = full["x"] - noH1["x"]     # exactly the L1H1 OV contribution (linear in x_mid)
    r2_dH1, rmse_dH1, _ = cv_r2(dH1, theta)

    # Position-controlled decode (partial out rank-slot q) -> genuine CONTENT threshold.
    Xc_full, yc = partial_out_q(full["x"], theta, q)
    Xc_noH1, _ = partial_out_q(noH1["x"], theta, q)
    Xc_noH0, _ = partial_out_q(noH0["x"], theta, q)
    Xc_dH1, _ = partial_out_q(dH1, theta, q)
    r2c_full, _, _ = cv_r2(Xc_full, yc)
    r2c_noH1, _, _ = cv_r2(Xc_noH1, yc)
    r2c_noH0, _, _ = cv_r2(Xc_noH0, yc)
    r2c_dH1, _, _ = cv_r2(Xc_dH1, yc)

    # Dimensionality of the (position-controlled) DECODE. NB decode dims != causal dims.
    U, S, Vt = np.linalg.svd(Xc_full - Xc_full.mean(0), full_matrices=False)
    dim_r2 = {}
    for kdim in [1, 2, 3, 5, 10, 48]:
        Z = (Xc_full - Xc_full.mean(0)) @ Vt[:kdim].T
        r2k, _, _ = cv_r2(Z, yc)
        dim_r2[kdim] = float(r2k)

    res = dict(
        raw=dict(r2_full=float(r2_full), rmse_full=float(rmse_full),
                 r2_noH1=float(r2_noH1), r2_noH0=float(r2_noH0),
                 r2_L1H1_contrib_alone=float(r2_dH1)),
        q_controlled=dict(r2_full=float(r2c_full), r2_noH1=float(r2c_noH1),
                          r2_noH0=float(r2c_noH0), r2_L1H1_contrib_alone=float(r2c_dH1)),
        content_decode_dim_r2=dim_r2,
        note="raw R2 is inflated by q (theta monotone in rank slot); q_controlled is the "
             "genuine content threshold. Decode R2 != causal steerability (see part 2).",
    )
    print(json.dumps(res, indent=2))

    # Figure: q-controlled theta decode by dim + head-ablation bars (q-controlled)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ks = sorted(dim_r2)
    ax1.plot(ks, [dim_r2[k] for k in ks], "-o", color=BLUE, lw=2, ms=7, zorder=3)
    ax1.set_xscale("log"); ax1.set_xticks(ks); ax1.set_xticklabels(ks)
    ax1.set_ylim(0, 1.02); ax1.set_xlabel("# PCA dims of x_mid used")
    ax1.set_ylabel("content-theta decode R^2 (CV, q-controlled)")
    ax1.set_title("Content threshold is linearly decodable from x_mid")
    labels = ["x_mid\n(full)", "x_mid\n-L1H1", "x_mid\n-L1H0", "L1H1\ncontrib only"]
    vals = [r2c_full, r2c_noH1, r2c_noH0, r2c_dH1]
    cols = [BLUE, ORANGE, INK_2, VIOLET]
    ax2.bar(np.arange(4), vals, 0.6, color=cols, zorder=3)
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("content-theta decode R^2 (q-controlled)")
    ax2.set_title("L1H1 carries the threshold content")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle("Part 1 — how the running-max threshold theta is encoded at x_mid",
                 fontweight="bold", y=1.02)
    save(fig, "riddle_threshold_direction.png")
    return res, full


# --------------------------------------------------------------------------- #
# 2. THE decisive experiment: theta sweep on a fixed menu.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def build_fixed_seq(model, menu_keys, vals, device):
    """Teacher-forced sequence with given input keys (menu) and values; sorted output."""
    p = len(menu_keys)
    keys = torch.tensor(menu_keys); v = torch.tensor(vals)
    order = keys.argsort(stable=True)
    sk, sv = keys[order], v[order]
    row = [cs.PAD] * (4 * p + 2)
    for i, (k, vv) in enumerate(zip(menu_keys, vals)):
        row[2 * i] = k; row[2 * i + 1] = vv
    row[2 * p] = cs.SEP
    os = 2 * p + 1
    for j in range(p):
        row[os + 2 * j] = int(sk[j]); row[os + 2 * j + 1] = int(sv[j])
    row[os + 2 * p] = cs.EOS
    return torch.tensor([row], device=device), sk.tolist()


@torch.inference_mode()
def sweep_token_patch(model, seq, p, q, mlp_off=False):
    """Overwrite the emitted key sk_{q-1} (seq index sep+1+2(q-1)) with theta=0..49;
    read the argmax next key at query sep+2q. q=1 => threshold is exactly that token."""
    sep = 2 * p
    last_pos = sep + 1 + 2 * (q - 1)
    qpos = sep + 2 * q
    sel = []
    for theta in range(cs.KEY_ALPH):
        s = seq.clone()
        s[0, last_pos] = theta
        logits, caps = full_capture(model, s[:, :-1], mlp_off={1} if mlp_off else None)
        sel.append(int(logits[0, qpos, :cs.KEY_ALPH].argmax()))
    return np.array(sel)


@torch.inference_mode()
def xmid_traj(model, seq, p, q=1):
    """Capture x_mid[qpos] for each token-patch theta=0..49 (fixed menu). The real,
    on-manifold set of threshold states the readout actually receives."""
    sep = 2 * p
    last_pos = sep + 1 + 2 * (q - 1)
    qpos = sep + 2 * q
    base_xm, _ = capture_xmid(model, seq[:, :-1])
    traj = []
    for theta in range(cs.KEY_ALPH):
        s = seq.clone(); s[0, last_pos] = theta
        xm, _ = capture_xmid(model, s[:, :-1])
        traj.append(xm[0, qpos].float().cpu().numpy())
    return np.array(traj), base_xm, qpos


@torch.inference_mode()
def select_from_patched_xmid(model, base_xm, qpos, vec, mlp_off=False):
    xm = base_xm.clone()
    xm[0, qpos] = torch.tensor(vec, device=base_xm.device, dtype=base_xm.dtype)
    lg = complete_from_xmid(model, xm, mlp_off=mlp_off)
    return int(lg[0, qpos, :cs.KEY_ALPH].argmax())


def part2_sweep(model, full, p=12):
    print("\n=== Part 2: decisive theta sweep ===")
    rng = np.random.RandomState(3)
    menu = sorted(rng.choice(np.arange(cs.KEY_ALPH), size=p, replace=False).tolist())
    vals = (rng.randint(0, cs.VAL_ALPH, size=p) + cs.KEY_ALPH).tolist()
    seq, sk = build_fixed_seq(model, menu, vals, DEV)
    ideal = np.array([smallest_present_above(menu, th) for th in range(cs.KEY_ALPH)])
    valid = ideal >= 0

    # --- q=1 token patch: threshold == single emitted token (maximally in-dist) ---
    sel_tok = sweep_token_patch(model, seq, p, q=1, mlp_off=False)
    sel_tok_nomlp = sweep_token_patch(model, seq, p, q=1, mlp_off=True)
    match_tok = float((sel_tok[valid] == ideal[valid]).mean())
    match_tok_nomlp = float((sel_tok_nomlp[valid] == ideal[valid]).mean())

    # --- aggregate token-patch match over many random menus (q=1) ---
    agg = []
    for s in range(40):
        rr = np.random.RandomState(100 + s)
        m = sorted(rr.choice(np.arange(cs.KEY_ALPH), size=p, replace=False).tolist())
        vv = (rr.randint(0, cs.VAL_ALPH, size=p) + cs.KEY_ALPH).tolist()
        sq, _ = build_fixed_seq(model, m, vv, DEV)
        st = sweep_token_patch(model, sq, p, q=1)
        idl = np.array([smallest_present_above(m, th) for th in range(cs.KEY_ALPH)])
        vd = idl >= 0
        agg.append((st[vd] == idl[vd]).mean())
        torch.cuda.empty_cache()
    agg_match = float(np.mean(agg))

    # --- Is the causal threshold code in x_mid low-dimensional / a single axis? ---
    # Capture the REAL x_mid threshold states (token patch), then ask how many PCA dims
    # of that trajectory are needed to reproduce the staircase when transplanted.
    traj, base_xm, qpos = xmid_traj(model, seq, p, q=1)
    # sanity: transplant captured x_mid -> reproduces the token-patch selection exactly.
    sel_transplant = np.array([select_from_patched_xmid(model, base_xm, qpos, traj[t])
                               for t in range(cs.KEY_ALPH)])
    transplant_match = float((sel_transplant == sel_tok).mean())
    c = traj - traj.mean(0)
    U, S, Vt = np.linalg.svd(c, full_matrices=False)
    var_frac = (S ** 2 / (S ** 2).sum())
    topk_match = {}
    for kk in [1, 2, 3, 5, 8, 12]:
        rec = traj.mean(0) + (c @ Vt[:kk].T) @ Vt[:kk]
        selk = np.array([select_from_patched_xmid(model, base_xm, qpos, rec[t])
                         for t in range(cs.KEY_ALPH)])
        topk_match[kk] = float((selk[valid] == ideal[valid]).mean())
    # 1-D "additive theta axis" injection (linear-probe direction) -> the naive hypothesis.
    mu, sd = full["x"].mean(0), full["x"].std(0) + 1e-6
    reg = Ridge(alpha=10.0).fit((full["x"] - mu) / sd, full["theta"].astype(float))
    wdir = (reg.coef_ / sd).astype(np.float32)
    b_raw = float(reg.intercept_ - (reg.coef_ / sd * mu).sum())
    unit = wdir / (wdir @ wdir)
    base_vec = base_xm[0, qpos].float().cpu().numpy()
    cur = float(base_vec @ wdir + b_raw)
    sel_1d = np.array([select_from_patched_xmid(model, base_xm, qpos,
                       base_vec + (t - cur) * unit) for t in range(cs.KEY_ALPH)])
    match_1d = float((sel_1d[valid] == ideal[valid]).mean())

    res = dict(menu=menu, match_token_q1=match_tok, match_token_q1_mlp_off=match_tok_nomlp,
               agg_token_q1_match_40menus=agg_match,
               xmid_transplant_reproduces_tokenpatch=transplant_match,
               xmid_1d_probe_axis_injection_match=match_1d,
               xmid_traj_var_frac_top6=[float(v) for v in var_frac[:6]],
               xmid_topk_pc_reconstruction_match=topk_match)
    print(json.dumps(res, indent=2))

    # ---- Figure: staircase + causal dimensionality ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    present = np.array(menu)
    ax = axes[0]
    th = np.arange(cs.KEY_ALPH)
    ax.step(th, ideal, where="post", color=INK_2, lw=3, alpha=0.4,
            label="ideal: min present > theta")
    ax.scatter(th[valid], sel_tok[valid], s=30, color=BLUE, zorder=3, label="model selection")
    for k in present:
        ax.axhline(k, color=GREEN, lw=0.6, alpha=0.35)
    ax.plot([0, 49], [0, 49], "--", color=ORANGE, lw=0.8, alpha=0.6, label="theta (y=x)")
    ax.set_xlabel("injected threshold theta (emitted-key token)")
    ax.set_ylabel("selected next key")
    ax.set_title(f"Token patch (q=1): staircase, match={match_tok:.3f}\n"
                 f"single static menu emits every present key in order")
    ax.set_xlim(0, 49); ax.set_ylim(-1, 50); ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ks = sorted(topk_match)
    ax.plot(ks, [topk_match[k] for k in ks], "-o", color=BLUE, lw=2, ms=7, zorder=3,
            label="top-k PC reconstruction of\nreal x_mid threshold trajectory")
    ax.axhline(match_1d, color=ORANGE, ls="--", lw=1.5,
               label=f"naive 1-D theta-axis inject ({match_1d:.2f})")
    ax.axhline(transplant_match, color=GREEN, ls=":", lw=1.5,
               label=f"full x_mid transplant ({transplant_match:.2f})")
    ax.set_xlabel("# PCA dims of x_mid threshold code used")
    ax.set_ylabel("staircase match to ideal")
    ax.set_ylim(0, 1.05)
    ax.set_title("The causal threshold code is NOT 1-D:\n~5-8 dims needed to drive the readout")
    ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Part 2 (decisive) — a static menu emits a staircase as theta sweeps; "
                 "the x_mid threshold code is multi-dimensional\n"
                 f"present keys (green): {present.tolist()}", fontweight="bold", y=1.08)
    save(fig, "riddle_threshold_sweep.png")
    return res


# --------------------------------------------------------------------------- #
# 3. Readout functional form.
# --------------------------------------------------------------------------- #
def part3_readout_form(model, p=12, n=384):
    print("\n=== Part 3: readout functional form ===")
    d = collect(model, p, n, seed=1)
    N = d["theta"].shape[0]
    theta = d["theta"].astype(float)
    pres = present_matrix(d["menu"])                    # [N,50]
    ks = np.arange(cs.KEY_ALPH)[None, :].repeat(N, 0)   # [N,50]
    th = theta[:, None]
    above = (ks > th).astype(np.float32)
    gap_above = np.maximum(ks - th, 0).astype(np.float32)
    gap_below = np.maximum(th - ks, 0).astype(np.float32)
    y = d["logit"].astype(np.float32)                   # [N,50] true key logits

    # Features per (example,key). Present interactions matter (absent keys are dead).
    feats = np.stack([
        np.ones_like(above), pres, above, pres * above,
        gap_above, pres * gap_above, gap_below, pres * gap_below,
    ], -1)                                              # [N,50,F]
    Xf = feats.reshape(-1, feats.shape[-1])
    yf = y.reshape(-1)
    reg = LinearRegression().fit(Xf, yf)
    yhat = reg.predict(Xf).reshape(N, cs.KEY_ALPH)
    r2_all = float(reg.score(Xf, yf))

    # The selection is an argmax over PRESENT keys, so fit + evaluate there (the
    # all-key R2 is dominated by dead absent keys and is not what drives the sort).
    pmask = pres.reshape(-1) > 0.5
    feat3 = np.stack([above, gap_above, gap_below], -1).reshape(-1, 3)  # [N*50,3]
    feat_p = feat3[pmask]
    reg_p = LinearRegression().fit(feat_p, yf[pmask])
    r2_present = float(reg_p.score(feat_p, yf[pmask]))

    # argmax agreement (restricted to present keys, which is how the model chooses).
    NEG = -1e9
    ym = np.where(pres > 0.5, y, NEG)                    # true logits, absent masked out
    form = reg_p.predict(feat3).reshape(N, cs.KEY_ALPH)  # theta-gated functional form
    form_m = np.where(pres > 0.5, form, NEG)
    true_sel = ym.argmax(1)
    fit_sel = form_m.argmax(1)
    ideal = np.array([smallest_present_above(d["menu"][i], theta[i]) for i in range(N)])
    argmax_fit_vs_true = float((fit_sel == true_sel).mean())
    argmax_true_vs_ideal = float((true_sel == ideal).mean())
    argmax_fit_vs_ideal = float((fit_sel == ideal).mean())

    # How cleanly does the readout separate the two present-key groups?
    present_above = [y[i, k] for i in range(N) for k in d["menu"][i] if k > theta[i]]
    present_below = [y[i, k] for i in range(N) for k in d["menu"][i] if k <= theta[i]]
    # margin: winner logit minus best present competitor.
    win = ym.max(1)
    tmp = ym.copy(); tmp[np.arange(N), true_sel] = NEG
    second = tmp.max(1)
    margin = float(np.mean(win - second))

    coefs = dict(zip(["bias", "present", "above", "present*above", "gap_above",
                      "present*gap_above", "gap_below", "present*gap_below"],
                     [float(c) for c in reg.coef_]))
    coefs_present = dict(zip(["above", "gap_above", "gap_below"],
                             [float(c) for c in reg_p.coef_]))
    res = dict(r2_all_keys=r2_all, r2_present_keys=r2_present,
               argmax_fit_vs_true=argmax_fit_vs_true,
               argmax_true_vs_ideal=argmax_true_vs_ideal,
               argmax_fit_vs_ideal=argmax_fit_vs_ideal,
               mean_logit_present_above=float(np.mean(present_above)),
               mean_logit_present_below=float(np.mean(present_below)),
               mean_winner_margin=margin,
               coefs_allkeys=coefs, coefs_present=coefs_present)
    print(json.dumps(res, indent=2))
    r2 = r2_present

    # Figure: mean key-logit profile vs (k-theta), present vs absent.
    rel = (ks - th).astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, mask, col in [("present keys", pres > 0.5, BLUE), ("absent keys", pres < 0.5, INK_2)]:
        xs = np.arange(-15, 16)
        m = [np.mean(y[(rel == r) & mask]) if ((rel == r) & mask).any() else np.nan for r in xs]
        ax1.plot(xs, m, "-o", ms=3, color=col, label=name, zorder=3)
    ax1.axvline(0, color=ORANGE, lw=1, ls="--", label="k = theta")
    ax1.set_xlabel("key value minus theta  (k - theta)")
    ax1.set_ylabel("mean model logit")
    ax1.set_title("Readout: sharp cliff at theta, present keys win just above")
    ax1.legend(fontsize=8)
    # scatter true vs fitted logit (PRESENT keys — what the argmax ranges over)
    yp = yf[pmask]
    idx = np.random.RandomState(0).choice(yp.shape[0], min(4000, yp.shape[0]), replace=False)
    ax2.scatter(yp[idx], reg_p.predict(feat_p[idx]), s=6, alpha=0.3, color=VIOLET)
    lo, hi = yp.min(), yp.max()
    ax2.plot([lo, hi], [lo, hi], "--", color=INK, lw=1)
    ax2.set_xlabel("true logit (present keys)"); ax2.set_ylabel("fitted (theta-gated form)")
    ax2.set_title(f"Functional form fit (present keys)  R^2 = {r2_present:.3f}")
    fig.suptitle("Part 3 — logit(k) ~ present(k) with a theta-gated penalty",
                 fontweight="bold", y=1.02)
    save(fig, "riddle_threshold_readout.png")
    return res


# --------------------------------------------------------------------------- #
# 4. Linear vs MLP: is the gating done before the L1 MLP?
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def part4_linear_vs_mlp(model, p=12, n=384):
    print("\n=== Part 4: linear vs MLP ===")
    seq, keys, order = batch(model, p, n=n, seed=2)
    sep = 2 * p
    _, caps = full_capture(model, seq[:, :-1])
    x_mid = caps[1]["x_mid"]; x_out = caps[1]["x_out"]
    kq = [sep + 2 * q for q in range(1, p)]
    tgt = seq[:, [sep + 1 + 2 * q for q in range(1, p)]]  # true next key sk_q  [n,p-1]

    # model's OWN unembedding as a logit lens on x_mid vs x_out (direct-path readout).
    def lens_acc(R):
        lg = model.output(model.final_norm(R))[:, kq, :cs.KEY_ALPH]
        pred = lg.argmax(-1)
        return float((pred == tgt).float().mean())
    acc_xmid = lens_acc(x_mid)      # before L1 MLP
    acc_xout = lens_acc(x_out)      # after L1 MLP (true residual)

    # Ablate the L1 MLP entirely and measure generation-relevant key accuracy.
    def key_acc(mlp_off):
        logits, _ = full_capture(model, seq[:, :-1], mlp_off=mlp_off)
        pred = logits[:, kq, :cs.KEY_ALPH].argmax(-1)
        return float((pred == tgt).float().mean())
    base_key = key_acc(None)
    key_l1off = key_acc({1})
    key_l0off = key_acc({0})

    res = dict(lens_acc_xmid_before_mlp=acc_xmid, lens_acc_xout_after_mlp=acc_xout,
               key_acc_baseline=base_key, key_acc_L1_mlp_off=key_l1off,
               key_acc_L0_mlp_off=key_l0off)
    print(json.dumps(res, indent=2))
    torch.cuda.empty_cache()
    return res


def main():
    apply_style()
    AF.mkdir(exist_ok=True)
    model, meta = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
    print(f"loaded cfg={model.cfg} gen_exact={meta.get('gen_exact'):.5f}")
    p = 12
    r1, full = part1_theta_direction(model, p=p)
    r2 = part2_sweep(model, full, p=p)
    r3 = part3_readout_form(model, p=p)
    r4 = part4_linear_vs_mlp(model, p=p)
    out = dict(checkpoint="causal_pairs_L30.pt", length=p,
               part1_threshold_direction=r1, part2_sweep=r2,
               part3_readout_form=r3, part4_linear_vs_mlp=r4)
    (AF / "riddle_threshold_metrics.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", AF / "riddle_threshold_metrics.json")


if __name__ == "__main__":
    main()
