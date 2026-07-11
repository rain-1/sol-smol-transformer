"""Causal confirmation of the two-factor (menu x threshold) sort mechanism.

The established descriptive picture (agent_findings/sorthead.md, geometry.md):
  - The next key = smallest input key strictly greater than the running max of the
    keys already emitted.
  - MENU: layer-0 heads aggregate the set of present input keys into the SEP-position
    residual (block-0 output at SEP); higher layers read this forward.
  - THRESHOLD: sort head L1H1 reads the recently-emitted keys at the key-emit query to
    recover the running-max threshold.
  - The decision is fully linearly decodable at the L1-attention output (caps[1].x_mid,
    post-L1-attn, pre-MLP) -> the L1 MLP is a minor refinement.

This script CAUSALLY tests that decomposition by activation patching (all patches are
off-distribution -- see caveats in the report):

  1a. MENU swap  : replace the SEP block-0 residual of run A with that of run B and
      confirm A now selects from B's candidate key set (cut stays at A's threshold).
  1b. THRESHOLD swap : replace the L1-attention output at a key-emit query of A with the
      L1-attention output from a *different step* of the same sequence, and confirm the
      cut point moves to the other step's threshold while the candidate set stays A's.
  2.  DAS-lite   : at caps[1].x_mid (the key-emit query), find the low-rank subspace
      whose transplant transfers the next-key decision between sequences; report the
      rank needed.
  3.  Sanity     : patching the whole caps[1].x_mid vector at the query sets the next key
      (downstream of L1-attn is position-wise: MLP+norm+unembed), and the next key is a
      near-linear readout of x_mid.

Reuses causal_sort (model/data) and deep_interp (load/batch). Does NOT modify the
committed model: a tiny forward is re-implemented here so patched tensors can be injected.

Run: python research/sort_scaling/agent_riddle_causal.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, BLUE, ORANGE, VIOLET, GREEN, INK, INK_2  # noqa
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
# Re-implemented forward with injection hooks (does NOT touch committed weights).
# Residual staging matches walkthrough.full_capture:
#   block0 output == caps[0].x_out ;  x_mid == caps[1].x_mid (post-L1-attn, pre-MLP)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def forward_patched(model, tokens, *, sep=None, sep_block0=None,
                    attn1_patch=None, xmid1_patch=None):
    """Full forward; optional injections:
      sep_block0  : [B,d] written into block-0 output at position `sep` (MENU swap).
      attn1_patch : (pos, [B,d]) replaces the block-1 attention output at `pos`.
      xmid1_patch : (pos, [B,d]) replaces x_mid (post-L1-attn) at `pos`.
    Returns dict with logits, x_mid, and block-1 attn_out (unpatched capture)."""
    L = tokens.shape[1]
    x = model.token_embedding(tokens) + model.position_embedding[:, :L]
    causal = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
    kpm = tokens.eq(cs.PAD)
    b0 = model.blocks[0]
    normed = b0.ln1(x)
    ao, _ = b0.attn(normed, normed, normed, attn_mask=causal,
                    key_padding_mask=kpm, need_weights=False)
    x = x + ao
    x = x + b0.mlp(b0.ln2(x))                 # block-0 output (caps[0].x_out)
    if sep_block0 is not None:
        x = x.clone(); x[:, sep] = sep_block0
    b1 = model.blocks[1]
    normed = b1.ln1(x)
    ao1, _ = b1.attn(normed, normed, normed, attn_mask=causal,
                     key_padding_mask=kpm, need_weights=False)
    ao1_cap = ao1
    if attn1_patch is not None:
        pos, val = attn1_patch
        ao1 = ao1.clone(); ao1[:, pos] = val
    x_mid = x + ao1
    if xmid1_patch is not None:
        pos, val = xmid1_patch
        x_mid = x_mid.clone(); x_mid[:, pos] = val
    x_out = x_mid + b1.mlp(b1.ln2(x_mid))
    logits = model.output(model.final_norm(x_out))
    return {"logits": logits, "x_mid": x_mid, "attn1": ao1_cap}


@torch.inference_mode()
def block0_output(model, tokens):
    L = tokens.shape[1]
    x = model.token_embedding(tokens) + model.position_embedding[:, :L]
    causal = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
    kpm = tokens.eq(cs.PAD)
    b0 = model.blocks[0]
    normed = b0.ln1(x)
    ao, _ = b0.attn(normed, normed, normed, attn_mask=causal,
                    key_padding_mask=kpm, need_weights=False)
    x = x + ao
    return x + b0.mlp(b0.ln2(x))


@torch.inference_mode()
def downstream_from_xmid(model, xmid_vec):
    """Map an x_mid vector [N,d] at the query position to logits [N,V].
    Everything after L1-attn (ln2, MLP, residual, final_norm, unembed) is position-wise,
    so the next-key prediction depends only on x_mid at that single position."""
    b1 = model.blocks[1]
    x = xmid_vec + b1.mlp(b1.ln2(xmid_vec))
    return model.output(model.final_norm(x))


# --------------------------------------------------------------------------- #
# 1a. MENU swap: patch the SEP block-0 residual from B into A.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def forward_patched_positions(model, tokens, patch_pos, patch_vals):
    """Full forward, overwriting block-0 output at `patch_pos` (list of indices) with
    `patch_vals` [B, len(patch_pos), d] before block 1 runs. Generalises the SEP menu
    swap to an arbitrary set of positions (e.g. the whole input region)."""
    L = tokens.shape[1]
    x = model.token_embedding(tokens) + model.position_embedding[:, :L]
    causal = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
    kpm = tokens.eq(cs.PAD)
    b0 = model.blocks[0]
    normed = b0.ln1(x)
    ao, _ = b0.attn(normed, normed, normed, attn_mask=causal,
                    key_padding_mask=kpm, need_weights=False)
    x = x + ao
    x = x + b0.mlp(b0.ln2(x))
    x = x.clone(); x[:, patch_pos] = patch_vals
    b1 = model.blocks[1]
    normed = b1.ln1(x)
    ao1, _ = b1.attn(normed, normed, normed, attn_mask=causal,
                     key_padding_mask=kpm, need_weights=False)
    x_mid = x + ao1
    x_out = x_mid + b1.mlp(b1.ln2(x_mid))
    return model.output(model.final_norm(x_out))


@torch.inference_mode()
def menu_swap(model, p=10, n=1024, qsteps=None, region="sep"):
    """Teacher-force true prefixes. At each key-emit step q, A's threshold T=sk_A[q-1].
    A's own next key = min(A_keys > T). After transplanting B's menu residual, the
    two-factor model predicts min(B_keys > T). Report, on the *discriminative* subset
    where those two answers differ, how often the patched prediction flips to the B-menu
    answer.

    region selects which block-0-output positions carry B's menu into A's run:
      'sep'   : only the SEP position (the sorthead.md localization claim).
      'input' : all input-pair positions 0..2p-1 (keys+values), SEP left as A's.
      'input_sep' : input pairs 0..2p-1 plus SEP.
    A's output region (emitted keys = threshold) is always left untouched."""
    if qsteps is None:
        qsteps = list(range(1, p))
    seqA, keysA, orderA = di.batch(model, p, n=n, seed=11)
    seqB, keysB, orderB = di.batch(model, p, n=n, seed=22)
    sep = 2 * p
    skA = keysA.gather(1, orderA)                 # [n,p] sorted A keys
    b0B = block0_output(model, seqB[:, :-1])      # B block-0 output (menu source)

    base = forward_patched(model, seqA[:, :-1])
    if region == "sep":
        pos = [sep]
    elif region == "input":
        pos = list(range(0, 2 * p))
    elif region == "input_sep":
        pos = list(range(0, 2 * p)) + [sep]
    else:
        raise ValueError(region)
    patched_logits = forward_patched_positions(model, seqA[:, :-1], pos, b0B[:, pos])
    patched = {"logits": patched_logits}

    keysA_np = keysA.cpu().numpy(); keysB_np = keysB.cpu().numpy()
    skA_np = skA.cpu().numpy()
    rec = []  # per (example, step): base pred, patched pred, A-answer, B-menu-answer
    for q in qsteps:
        qpos = sep + 2 * q
        pb = base["logits"][:, qpos].argmax(-1).cpu().numpy()
        pp = patched["logits"][:, qpos].argmax(-1).cpu().numpy()
        for i in range(n):
            T = skA_np[i, q - 1]
            a_gt = keysA_np[i][keysA_np[i] > T]
            b_gt = keysB_np[i][keysB_np[i] > T]
            a_ans = int(a_gt.min()) if a_gt.size else cs.EOS
            b_ans = int(b_gt.min()) if b_gt.size else cs.EOS
            rec.append((int(pb[i]), int(pp[i]), a_ans, b_ans))
    rec = np.array(rec)
    pb, pp, a_ans, b_ans = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    # sanity: unpatched model follows the A menu
    base_A = float((pb == a_ans).mean())
    # discriminative subset: A and B menus give different answers, both a real key
    disc = (a_ans != b_ans) & (a_ans != cs.EOS) & (b_ans != cs.EOS)
    d = dict(
        n_total=int(len(rec)), n_disc=int(disc.sum()),
        base_follows_A=base_A,
        patched_follows_B=float((pp[disc] == b_ans[disc]).mean()),
        patched_follows_A=float((pp[disc] == a_ans[disc]).mean()),
        base_follows_B_on_disc=float((pb[disc] == b_ans[disc]).mean()),
        patched_in_B_keys=None,
    )
    # menu preserved-as-B check: is patched pred a member of B's key set (or EOS)?
    inB = []
    for k, q in enumerate(qsteps):
        for i in range(n):
            idx = k * n + i
            inB.append(pp[idx] in set(keysB_np[i].tolist()) or pp[idx] == cs.EOS)
    d["patched_in_B_keys"] = float(np.mean(inB))
    d["patched_in_A_keys"] = float(np.mean([
        pp[k * n + i] in set(keysA_np[i].tolist()) or pp[k * n + i] == cs.EOS
        for k, q in enumerate(qsteps) for i in range(n)]))
    return d


@torch.inference_mode()
def plumbing_control(model, p=10, n=1024):
    """Positive control that block-0-output patching plumbing works: transplant the WHOLE
    block-0 output (all positions) of B into A. Block 1 then runs entirely on B's state,
    so the next-key at each step must equal B's own next key sk_B[q]."""
    seqA, keysA, orderA = di.batch(model, p, n=n, seed=11)
    seqB, keysB, orderB = di.batch(model, p, n=n, seed=22)
    sep = 2 * p
    skB = keysB.gather(1, orderB).cpu().numpy()
    b0B = block0_output(model, seqB[:, :-1])
    L = seqA.shape[1] - 1
    logits = forward_patched_positions(model, seqA[:, :-1], list(range(L)), b0B)
    hit = []
    for q in range(1, p):
        pp = logits[:, sep + 2 * q].argmax(-1).cpu().numpy()
        hit.append((pp == skB[:, q]).mean())
    return float(np.mean(hit))


# --------------------------------------------------------------------------- #
# 1b. THRESHOLD swap: replace the L1-attn output at step q1 with that of step q2.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def threshold_swap(model, p=12, n=1024):
    """Within one sequence the menu (SEP) is constant across steps, so transplanting the
    whole L1-attention output from a later step q2 into an earlier step q1 changes only
    the threshold read (running-max). Expect the cut to move to sk[q2] while the answer
    stays inside A's own key set."""
    seq, keys, order = di.batch(model, p, n=n, seed=7)
    sep = 2 * p
    sk = keys.gather(1, order).cpu().numpy()      # [n,p]
    keys_np = keys.cpu().numpy()
    cap = forward_patched(model, seq[:, :-1])
    attn1 = cap["attn1"]                           # [n,L,d]
    base_logits = cap["logits"]

    rows = []
    for q1 in range(1, p - 1):
        for q2 in range(q1 + 1, p):
            pos1, pos2 = sep + 2 * q1, sep + 2 * q2
            src = attn1[:, pos2]                    # threshold read from step q2
            pt = forward_patched(model, seq[:, :-1], attn1_patch=(pos1, src))
            pp = pt["logits"][:, pos1].argmax(-1).cpu().numpy()
            pb = base_logits[:, pos1].argmax(-1).cpu().numpy()
            for i in range(n):
                T2 = sk[i, q2 - 1]
                gt = keys_np[i][keys_np[i] > T2]
                swap_ans = int(gt.min()) if gt.size else cs.EOS      # min(A_keys > T2)
                a_ans = int(sk[i, q1])                                # min(A_keys > T1)
                rows.append((int(pb[i]), int(pp[i]), a_ans, swap_ans,
                             pp[i] in set(keys_np[i].tolist())))
    rows = np.array(rows)
    pb, pp, a_ans, swap_ans, in_keys = rows.T
    disc = (a_ans != swap_ans) & (swap_ans != cs.EOS)
    return dict(
        n_total=int(len(rows)), n_disc=int(disc.sum()),
        base_follows_orig=float((pb == a_ans).mean()),
        patched_follows_swapped=float((pp[disc] == swap_ans[disc]).mean()),
        patched_follows_orig=float((pp[disc] == a_ans[disc]).mean()),
        patched_in_own_keys=float(in_keys.mean()),
    )


# --------------------------------------------------------------------------- #
# 2/3. DAS-lite + sanity: minimal x_mid subspace that transfers the decision.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def collect_xmid(model, p=12, n=1024, seed=5):
    """x_mid at every key-emit query (q>=1) across a batch, with the next-key label and
    the model's own next-key argmax (to define transfer targets)."""
    seq, keys, order = di.batch(model, p, n=n, seed=seed)
    sep = 2 * p
    sk = keys.gather(1, order)
    cap = forward_patched(model, seq[:, :-1])
    xmid = cap["x_mid"]; logits = cap["logits"]
    X, y_next, y_step, pred = [], [], [], []
    for q in range(1, p):
        qpos = sep + 2 * q
        X.append(xmid[:, qpos])
        y_next.append(sk[:, q])
        y_step.append(torch.full((n,), q, device=DEV))
        pred.append(logits[:, qpos].argmax(-1))
    return (torch.cat(X).float(), torch.cat(y_next), torch.cat(y_step),
            torch.cat(pred), p)


@torch.inference_mode()
def das_lite(model, p=12, n=768, n_pairs=4000, ranks=(1, 2, 3, 4, 6, 8, 12, 16, 24, 48)):
    """Transfer test: patched = x_T + U U^T (x_S - x_T) at the query position; success if
    the resulting next-key argmax equals the model's own next key for source S.

    Subspace U (r dims) is the top-r PCA of *same-step difference vectors* x_S - x_T
    (differences hold position fixed, isolating the menu/threshold decision variation).
    Controls: random r-dim subspace, and full-rank (r=48) = whole-vector transplant."""
    X, y_next, y_step, pred, _ = collect_xmid(model, p=p, n=n, seed=5)
    X = X.cpu(); y_step = y_step.cpu(); pred = pred.cpu()
    N, d = X.shape
    g = torch.Generator().manual_seed(0)

    # sample same-step source/target pairs whose model-predictions differ
    steps = y_step
    pairs = []
    order_idx = torch.randperm(N, generator=g)
    by_step = {int(q): (steps == q).nonzero(as_tuple=True)[0] for q in steps.unique()}
    while len(pairs) < n_pairs:
        q = int(steps[order_idx[len(pairs) % N]])
        idxs = by_step[q]
        a, b = idxs[torch.randint(len(idxs), (2,), generator=g)]
        if pred[a] != pred[b]:
            pairs.append((int(a), int(b)))
    pairs = torch.tensor(pairs)
    T_idx, S_idx = pairs[:, 0], pairs[:, 1]
    xT, xS = X[T_idx], X[S_idx]
    tgt = pred[S_idx]                              # want patched argmax == source pred

    diffs = xS - xT                                # decision-relevant difference vectors
    # PCA basis of differences (mean-centered)
    D = diffs - diffs.mean(0, keepdim=True)
    U_full, Sv, _ = torch.linalg.svd(D, full_matrices=False)   # columns of Vh^T
    # right singular vectors are the directions in d-space:
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    basis = Vh                                      # [d,d] rows = directions

    def transfer_rate(Umat):
        # Umat [d,r] orthonormal columns
        proj = (diffs @ Umat) @ Umat.T             # [P,d]
        patched = (xT + proj).to(DEV)
        lg = downstream_from_xmid(model, patched)
        return float((lg.argmax(-1).cpu() == tgt).float().mean())

    res = {"pca": {}, "random": {}}
    for r in ranks:
        Umat = basis[:r].T.contiguous()             # [d,r]
        res["pca"][r] = transfer_rate(Umat)
        # random orthonormal r-subspace control (mean over a few draws)
        rr = []
        for s in range(3):
            g2 = torch.Generator().manual_seed(100 + s)
            M = torch.randn(d, r, generator=g2)
            Q, _ = torch.linalg.qr(M)
            rr.append(transfer_rate(Q))
        res["random"][r] = float(np.mean(rr))
    # baselines
    res["identity_target_pred"] = float((downstream_from_xmid(model, xT.to(DEV))
                                         .argmax(-1).cpu() == tgt).float().mean())
    res["full_transplant"] = transfer_rate(torch.eye(d))
    res["n_pairs"] = int(len(pairs))
    res["singular_energy"] = (Sv[:16] ** 2 / (Sv ** 2).sum()).tolist()
    return res


@torch.inference_mode()
def sanity_xmid(model, p=12, n=1024):
    """(3) Whole-vector x_mid transplant sets the next key; and the next key is a
    near-linear readout of x_mid (agreement of a linear probe with the model)."""
    from sklearn.linear_model import LogisticRegression
    X, y_next, y_step, pred, _ = collect_xmid(model, p=p, n=n, seed=9)
    Xn = X.cpu().numpy(); yn = pred.cpu().numpy()
    # cross-seq whole-vector transplant: replace target x_mid with source's -> source pred
    N = X.shape[0]
    g = torch.Generator().manual_seed(3)
    perm = torch.randperm(N, generator=g)
    lg = downstream_from_xmid(model, X[perm].to(DEV))
    transplant = float((lg.argmax(-1).cpu() == pred[perm].cpu()).float().mean())
    # linear decodability of the model's next-key from x_mid
    idx = np.random.RandomState(0).permutation(N); cut = int(0.75 * N)
    mu, sd = Xn[idx[:cut]].mean(0), Xn[idx[:cut]].std(0) + 1e-6
    Xs = (Xn - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[idx[:cut]], yn[idx[:cut]])
    lin_acc = float(clf.score(Xs[idx[cut:]], yn[idx[cut:]]))
    return dict(whole_vector_transplant=transplant, linear_probe_xmid_vs_model=lin_acc,
                n=int(N))


# --------------------------------------------------------------------------- #
# Diagnostic: which head brings the SEP menu to the key-emit query at layer 1.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def sep_read_by_head(model, p=12, n=256):
    seq, keys, order = di.batch(model, p, n=n, seed=1)
    _, attn, _ = di.run_capture(model, seq[:, :-1])
    sep = 2 * p
    kq = [sep + 2 * q for q in range(1, p)]
    out = {}
    for li in range(len(attn)):
        for h in range(attn[li].shape[1]):
            a = attn[li][:, h][:, kq]
            out[f"L{li}H{h}"] = float(a[:, :, sep].mean())
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_swaps(menu_regions, thr, name="riddle_causal_swaps.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # Left: THRESHOLD swap -- the clean two-factor dissociation.
    ax = axes[0]
    cats = ["unpatched\nfollows orig cut", "patched\nfollows orig cut",
            "patched\nfollows swapped cut", "patched key\nin own menu"]
    vals = [thr["base_follows_orig"], thr["patched_follows_orig"],
            thr["patched_follows_swapped"], thr["patched_in_own_keys"]]
    ax.bar(range(4), vals, color=[INK_2, ORANGE, VIOLET, GREEN], zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 1.06); ax.set_ylabel("rate")
    ax.set_title("1b. THRESHOLD swap (L1-attn out, step q2->q1)\n"
                 "cut moves, candidate set preserved")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    # Right: MENU localization -- candidate set is NOT transplantable from a residual.
    ax = axes[1]
    regs = ["sep", "input", "input_sep", "plumbing_full_block0_to_B"]
    labs = ["SEP\nblock-0", "input pairs\nblock-0", "input+SEP\nblock-0",
            "FULL block-0\n(control)"]
    vals = [menu_regions["sep"]["patched_follows_B"],
            menu_regions["input"]["patched_follows_B"],
            menu_regions["input_sep"]["patched_follows_B"],
            menu_regions["plumbing_full_block0_to_B"]]
    cols = [ORANGE, ORANGE, ORANGE, INK_2]
    ax.bar(range(4), vals, color=cols, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0, 1.06); ax.set_ylabel("rate follows B's candidate set")
    ax.set_title("1a. MENU swap: install B's key set into A\n"
                 "no input-side residual carries a swappable menu")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for a in axes:
        bare(a)
    fig.suptitle("Causal two-factor test: the threshold is a transplantable factor; "
                 "the candidate menu is distributed",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print("wrote reports/figures/" + name)


def fig_das(das, name="riddle_causal_das.png"):
    ranks = sorted(das["pca"].keys())
    pca = [das["pca"][r] for r in ranks]
    rnd = [das["random"][r] for r in ranks]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ranks, pca, "-o", color=BLUE, lw=2, ms=6, label="difference-PCA subspace", zorder=3)
    ax1.plot(ranks, rnd, "-s", color=INK_2, lw=1.6, ms=5, label="random subspace (control)", zorder=3)
    ax1.axhline(das["full_transplant"], color=GREEN, ls="--", lw=1.2,
                label=f"full transplant {das['full_transplant']:.2f}")
    ax1.axhline(das["identity_target_pred"], color=ORANGE, ls=":", lw=1.2,
                label=f"no patch (target keeps own key) {das['identity_target_pred']:.2f}")
    ax1.set_xlabel("subspace rank r"); ax1.set_ylabel("decision-transfer rate")
    ax1.set_ylim(0, 1.03); ax1.legend(fontsize=7.5, loc="center right")
    ax1.set_title("2. Minimal x_mid subspace transfers the next-key decision")
    en = das["singular_energy"]
    ax2.plot(range(1, len(en) + 1), np.cumsum(en), "-o", color=VIOLET, lw=2, ms=5, zorder=3)
    ax2.set_xlabel("rank"); ax2.set_ylabel("cumulative variance of\ndifference vectors")
    ax2.set_ylim(0, 1.03); ax2.set_title("Difference-vector spectrum (x_mid)")
    for a in (ax1, ax2):
        bare(a)
    fig.suptitle("DAS-lite: the sort decision lives in a few dimensions of the L1-attn output",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print("wrote reports/figures/" + name)


def main():
    apply_style()
    model, meta = di.load(CKPT)
    print(f"loaded {CKPT.name} cfg={model.cfg} gen_exact={meta.get('gen_exact'):.4f}")

    print("\n[diag] L1 attention-to-SEP mass at key steps (which head fetches the menu):")
    sepd = sep_read_by_head(model, p=12)
    print("  " + json.dumps({k: round(v, 3) for k, v in sepd.items()}))
    torch.cuda.empty_cache()

    print("\n[1a] MENU swap (localizing the candidate set) ...")
    menu_regions = {}
    for region in ("sep", "input", "input_sep"):
        m = menu_swap(model, p=10, n=1024, region=region)
        menu_regions[region] = m
        print(f"  [{region}] follows_B={m['patched_follows_B']:.3f} "
              f"follows_A={m['patched_follows_A']:.3f} "
              f"in_B_keys={m['patched_in_B_keys']:.3f} "
              f"in_A_keys={m['patched_in_A_keys']:.3f} n_disc={m['n_disc']}")
        torch.cuda.empty_cache()
    menu = menu_regions["input_sep"]   # the effective menu swap for the summary figure
    plumb = plumbing_control(model, p=10, n=1024)
    print(f"  [plumbing] full block-0 swap -> B's next key: {plumb:.3f}")
    menu_regions["plumbing_full_block0_to_B"] = plumb
    torch.cuda.empty_cache()

    print("\n[1b] THRESHOLD swap ...")
    thr = threshold_swap(model, p=12, n=1024)
    print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in thr.items()}))
    torch.cuda.empty_cache()

    print("\n[2] DAS-lite subspace ...")
    das = das_lite(model, p=12, n=768)
    print("  pca:", {r: round(v, 3) for r, v in das["pca"].items()})
    print("  random:", {r: round(v, 3) for r, v in das["random"].items()})
    print("  full_transplant", round(das["full_transplant"], 3),
          "no_patch", round(das["identity_target_pred"], 3))
    torch.cuda.empty_cache()

    print("\n[3] sanity x_mid ...")
    san = sanity_xmid(model, p=12, n=1024)
    print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in san.items()}))
    torch.cuda.empty_cache()

    fig_swaps(menu_regions, thr)
    fig_das(das)

    metrics = dict(checkpoint=str(CKPT), config=model.cfg,
                   sep_read_by_head=sepd, menu_swap=menu, menu_regions=menu_regions,
                   threshold_swap=thr, das_lite=das, sanity=san)
    (OUT / "riddle_causal_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nwrote", OUT / "riddle_causal_metrics.json")


if __name__ == "__main__":
    main()
