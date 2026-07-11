"""Weight-level QK/OV analysis of the value-carrying induction circuit L0H0 -> L1H0.

Imports (does not modify) deep_interp/causal_sort. Extracts per-head Wq,Wk,Wv,Wo,
folds LayerNorm gains (with stated approximations), and tests:
  (2) OV circuit of L1H0 in token space -> copy matrix over value tokens
  (3) QK circuit of L1H0 -> induction key-identity match + positional contribution
  (4) K-composition of L0H0 OV into L1H0 K input

Figures -> reports/figures/agent_qkov_*.png
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
from style import apply_style, SEQ_BLUE, DIVERGING, BLUE, ORANGE, VIOLET  # noqa: E402
import causal_sort as cs  # noqa: E402
import deep_interp as di  # noqa: E402

DEV = di.DEV
FIG = ROOT / "reports" / "figures"
HEAD_W = 24  # per-head width

# Token groups
KEY_TOKENS = list(range(0, cs.KEY_ALPH))          # 0..49
VAL_TOKENS = list(range(cs.KEY_ALPH, cs.SEP))     # 50..69


def head_slice(h):
    return slice(h * HEAD_W, (h + 1) * HEAD_W)


def get_qkv(block, h):
    """Return Wq,Wk,Wv [HEAD_W, d_model] and Wo [d_model, HEAD_W] for head h.

    in_proj_weight is [3*d, d] = concat(Wq,Wk,Wv along rows). out_proj.weight is
    [d, d]; head h reads columns h*24:(h+1)*24.
    """
    d = block.attn.in_proj_weight.shape[1]
    W = block.attn.in_proj_weight.detach()
    Wq = W[0:d][head_slice(h)]
    Wk = W[d:2 * d][head_slice(h)]
    Wv = W[2 * d:3 * d][head_slice(h)]
    Wo = block.attn.out_proj.weight.detach()[:, head_slice(h)]
    return Wq, Wk, Wv, Wo


def ln_fold(W_rows, ln):
    """Fold a LayerNorm applied to the *input* of a linear map into the embedding.

    APPROXIMATION: LayerNorm(x) = gamma * (x - mean(x)) / std(x) + beta. We keep the
    mean-centering (exact, linear) and the gain gamma (diag), and DROP the per-token
    1/std(x) rescale (data-dependent, not a fixed weight) and the additive beta
    (constant shift; contributes an offset, not token-to-token structure). So we map
    an embedding row e -> gamma * (e - mean(e)). Returns rows in the same shape.
    """
    gamma = ln.weight.detach()
    E = W_rows.detach()
    E = E - E.mean(-1, keepdim=True)
    return E * gamma


def main():
    apply_style()
    model, meta = di.load(str(Path(__file__).resolve().parent / "checkpoints" / "causal_pairs_L30.pt"))
    b0, b1 = model.blocks[0], model.blocks[1]
    W_E = model.token_embedding.weight.detach()          # [vocab, d]
    W_U = model.output.weight.detach()                   # [vocab, d]
    d = W_E.shape[1]

    out = []
    def log(s):
        print(s); out.append(s)

    # -------- extract heads --------
    Wq1, Wk1, Wv1, Wo1 = get_qkv(b1, 0)   # L1H0
    Wq0, Wk0, Wv0, Wo0 = get_qkv(b0, 0)   # L0H0
    log(f"shapes L1H0: Wq{tuple(Wq1.shape)} Wk{tuple(Wk1.shape)} Wv{tuple(Wv1.shape)} Wo{tuple(Wo1.shape)}")

    # ================================================================= #
    # (2) OV circuit of L1H0 over value tokens
    #   effective: W_U @ Wo1 @ Wv1 @ (LN1-folded W_E)
    #   Attention onto value token X should raise logit of X (copy).
    # ================================================================= #
    E_ln1 = ln_fold(W_E, b1.ln1)                         # input to L1 attn is ln1(resid)
    # value written by head into residual, then read by final_norm+output.
    # Fold final_norm gain into W_U (centering + gain; drop 1/std, beta).
    gammaU = model.final_norm.weight.detach()
    W_U_f = (W_U - W_U.mean(0, keepdim=True) * 0)  # placeholder; U acts on normed vec
    # For OV we care about U @ (fold of final_norm on the OV-written direction).
    # final_norm normalizes the vector v=Wo1@Wv1@e. Approx: v -> gammaU*(v-mean(v)).
    def U_of(v):  # v: [..., d]
        vv = v - v.mean(-1, keepdim=True)
        return vv * gammaU @ W_U.T
    OV = (Wo1 @ Wv1 @ E_ln1.T).T                          # [vocab, d] rows = source token
    OV_logits = U_of(OV)                                  # [vocab_src, vocab_out]
    M = OV_logits[np.ix_(VAL_TOKENS, VAL_TOKENS)].cpu().numpy()  # [20,20]
    diag = np.diag(M)
    off = M.copy(); np.fill_diagonal(off, np.nan)
    argmax_correct = (M.argmax(1) == np.arange(len(VAL_TOKENS))).mean()
    # diagonal dominance: row-centered diag z-score
    row_mean = np.nanmean(off, 1); row_std = np.nanstd(off, 1)
    diag_z = ((diag - row_mean) / row_std).mean()
    log("\n(2) OV COPY over value tokens 50-69:")
    log(f"  argmax-correct fraction = {argmax_correct:.3f} (chance {1/len(VAL_TOKENS):.3f})")
    log(f"  mean diag = {diag.mean():.3f}, mean off-diag = {np.nanmean(off):.3f}")
    log(f"  mean diagonal z-score over row = {diag_z:.2f}")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    ax.set_xlabel("output value logit"); ax.set_ylabel("attended value token")
    ax.set_xticks(range(0, 20, 4)); ax.set_xticklabels(range(50, 70, 4))
    ax.set_yticks(range(0, 20, 4)); ax.set_yticklabels(range(50, 70, 4))
    ax.set_title(f"L1H0 OV circuit (copy matrix)\nargmax-correct={argmax_correct:.0%}, diag z={diag_z:.1f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIG / "agent_qkov_ov_copy.png", bbox_inches="tight"); plt.close(fig)
    log("  wrote reports/figures/agent_qkov_ov_copy.png")

    # Control: OV over key tokens should NOT be a value copy (sanity)
    OV_kk = U_of((Wo1 @ Wv1 @ E_ln1.T).T)
    Mkk = OV_kk[np.ix_(KEY_TOKENS, KEY_TOKENS)].cpu().numpy()
    log(f"  control: OV key->key argmax-correct = {(Mkk.argmax(1)==np.arange(len(KEY_TOKENS))).mean():.3f}")

    # ================================================================= #
    # (3) QK circuit of L1H0: induction key-identity match.
    #   Query on a generated sorted key sk_q (token = some key id) should match the
    #   Key of the input KEY token of the SAME identity.
    #   score(q_tok, k_tok) = (E_q @ Wq1.T) @ (E_k @ Wk1.T).T   over key tokens.
    # ================================================================= #
    E_ln1_q = E_ln1  # same LN1 fold for both q and k side (both at layer-1 input)
    Q = E_ln1_q @ Wq1.T                                   # [vocab, HEAD_W]
    K = E_ln1_q @ Wk1.T
    scale = 1.0 / np.sqrt(HEAD_W)
    QK = (Q @ K.T) * scale                                # [vocab_q, vocab_k]
    QKk = QK[np.ix_(KEY_TOKENS, KEY_TOKENS)].cpu().numpy()
    diagk = np.diag(QKk)
    offk = QKk.copy(); np.fill_diagonal(offk, np.nan)
    qk_argmax = (QKk.argmax(1) == np.arange(len(KEY_TOKENS))).mean()
    qk_diag_z = ((diagk - np.nanmean(offk, 1)) / np.nanstd(offk, 1)).mean()
    log("\n(3) QK identity match over key tokens 0-49:")
    log(f"  argmax-on-identity fraction = {qk_argmax:.3f} (chance {1/len(KEY_TOKENS):.3f})")
    log(f"  mean diag = {diagk.mean():.3f}, mean off-diag = {np.nanmean(offk):.3f}")
    log(f"  mean diagonal z-score = {qk_diag_z:.2f}")

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(QKk, cmap=DIVERGING, vmin=-np.abs(QKk).max(), vmax=np.abs(QKk).max())
    ax.set_xlabel("input key token (K side)"); ax.set_ylabel("query key token (Q side)")
    ax.set_title(f"L1H0 QK circuit (identity match)\nargmax-on-identity={qk_argmax:.0%}, diag z={qk_diag_z:.1f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIG / "agent_qkov_qk_identity.png", bbox_inches="tight"); plt.close(fig)
    log("  wrote reports/figures/agent_qkov_qk_identity.png")

    # positional contribution: use position_embedding through Wq1/Wk1 (LN1 folded).
    P = model.position_embedding.detach()[0]              # [seq_len, d]
    P_ln = ln_fold(P, b1.ln1)
    Qp = P_ln @ Wq1.T; Kp = P_ln @ Wk1.T
    QKpos = (Qp @ Kp.T) * scale                           # [seq,seq]
    # magnitude of positional vs content QK bias
    log(f"  content QK std over key-token grid = {QKk.std():.3f}")
    log(f"  positional QK std over all pos pairs = {QKpos.std().item():.3f}")
    log(f"  --> content/positional std ratio = {QKk.std()/QKpos.std().item():.2f}")

    # ================================================================= #
    # (4) K-composition: L0H0 OV -> L1H0 K input.
    #   L1H0 reads keys from ln1(resid). L0H0 writes Wo0@Wv0 into resid. Composition
    #   operator = Wk1 @ diag(gamma_ln1) @ (Wo0 @ Wv0).  Compare Frobenius norm to
    #   baselines: (a) Wk1 @ direct embedding path, (b) other-head OV, (c) random.
    # ================================================================= #
    g1 = b1.ln1.weight.detach()
    def comp_score(Wo_s, Wv_s):
        M = Wk1 @ torch.diag(g1) @ (Wo_s @ Wv_s)          # [HEAD_W, d]
        return M.norm().item()
    # normalize by operand norms to make it a genuine "alignment" score.
    def comp_norm(Wo_s, Wv_s):
        OVs = Wo_s @ Wv_s
        raw = (Wk1 @ torch.diag(g1) @ OVs).norm().item()
        return raw / ((Wk1 @ torch.diag(g1)).norm().item() * OVs.norm().item() + 1e-9)
    s_l0h0 = comp_norm(Wo0, Wv0)
    Wq0b, Wk0b, Wv0b, Wo0b = get_qkv(b0, 1)               # L0H1 as control
    s_l0h1 = comp_norm(Wo0b, Wv0b)
    # random control: random OV maps of same norm
    torch.manual_seed(0)
    rand = []
    for _ in range(200):
        Wr = torch.randn_like(Wv0); Wor = torch.randn_like(Wo0)
        rand.append(comp_norm(Wor, Wr))
    rand = np.array(rand)
    log("\n(4) K-composition L0H0 OV -> L1H0 K:")
    log(f"  normalized comp score L0H0 = {s_l0h0:.4f}")
    log(f"  normalized comp score L0H1 (control) = {s_l0h1:.4f}")
    log(f"  random OV baseline = {rand.mean():.4f} +/- {rand.std():.4f}")
    log(f"  L0H0 z vs random = {(s_l0h0-rand.mean())/rand.std():.2f}")
    log(f"  L0H1 z vs random = {(s_l0h1-rand.mean())/rand.std():.2f}")
    # also raw Frobenius for reference
    log(f"  raw Frobenius: L0H0={comp_score(Wo0,Wv0):.3f} L0H1={comp_score(Wo0b,Wv0b):.3f}")

    # ---- composed QK: induction match mediated by L0H0 ----
    # Query on generated key token x: q(x) = ln1(E_x) @ Wq1.T
    # Key side at an input VALUE position that (via L0H0) attended to its preceding
    # KEY token x': the residual there carries L0H0 OV of E_{x'}. So the K vector is
    #   Wk1 @ diag(g1) @ (Wo0 @ Wv0 @ ln0(E_{x'})).
    E_ln0 = ln_fold(W_E, b0.ln1)
    Kcomp = (Wk1 @ torch.diag(g1) @ (Wo0 @ Wv0 @ E_ln0.T)).T   # [vocab, HEAD_W]
    QKc = (Q @ Kcomp.T) * scale
    QKck = QKc[np.ix_(KEY_TOKENS, KEY_TOKENS)].cpu().numpy()
    diagc = np.diag(QKck); offc = QKck.copy(); np.fill_diagonal(offc, np.nan)
    c_argmax = (QKck.argmax(1) == np.arange(len(KEY_TOKENS))).mean()
    c_z = ((diagc - np.nanmean(offc, 1)) / np.nanstd(offc, 1)).mean()
    log("\n(3b) COMPOSED QK (query key token -> L0H0-mediated key-side):")
    log(f"  argmax-on-identity = {c_argmax:.3f} (chance {1/len(KEY_TOKENS):.3f})")
    log(f"  mean diag = {diagc.mean():.3f}, off-diag = {np.nanmean(offc):.3f}, diag z = {c_z:.2f}")
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(QKck, cmap=DIVERGING, vmin=-np.abs(QKck).max(), vmax=np.abs(QKck).max())
    ax.set_xlabel("input key id (via L0H0 OV, K side)"); ax.set_ylabel("query key id (Q side)")
    ax.set_title(f"L1H0 composed QK via L0H0\nargmax-on-identity={c_argmax:.0%}, diag z={c_z:.1f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIG / "agent_qkov_qk_composed.png", bbox_inches="tight"); plt.close(fig)
    log("  wrote reports/figures/agent_qkov_qk_composed.png")

    Path(ROOT / "research/sort_scaling/agent_findings").mkdir(exist_ok=True)
    return "\n".join(out)


if __name__ == "__main__":
    text = main()
