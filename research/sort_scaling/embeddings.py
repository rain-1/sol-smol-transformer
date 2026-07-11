"""Embedding geometry: do keys carry order structure that values do not?

Keys must be compared (the `>` readout), so their embeddings should carry numeric
order; values are pure payload, copied by identity and never compared, so they
should NOT need order structure. This tests that asymmetry and shows the per-id
similarity structure of both alphabets.

Usage: python research/sort_scaling/embeddings.py
Figure -> reports/figures/emb_key_value_order.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, DIVERGING, BLUE, ORANGE, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"


def cv_order_r(E, ids, k=5, shuffle=True):
    """CV correlation of a linear embedding->id decode.

    shuffle=True -> interspersed held-out ids (interpolation); shuffle=False ->
    contiguous held-out id ranges (extrapolation, the stricter test).
    """
    E = E - E.mean(0)
    preds = np.zeros(len(ids))
    for tr, te in KFold(k, shuffle=shuffle, random_state=0 if shuffle else None).split(E):
        preds[te] = LinearRegression().fit(E[tr], ids[tr]).predict(E[te])
    return float(np.corrcoef(preds, ids)[0, 1])


def best_axis_r(E, ids):
    Ec = E - E.mean(0)
    U = np.linalg.svd(Ec, full_matrices=False)[0]
    return max(abs(np.corrcoef(U[:, i], ids)[0, 1]) for i in range(min(8, U.shape[1])))


def cos_matrix(E):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return En @ En.T


def main():
    apply_style()
    model, _ = load(HERE / "checkpoints" / "causal_pairs_L30.pt")
    W = model.token_embedding.weight.detach().cpu().numpy()
    Ek = W[:cs.KEY_ALPH]                        # keys 0-49
    Ev = W[cs.KEY_ALPH:cs.KEY_ALPH + cs.VAL_ALPH]  # values 50-69
    kid = np.arange(cs.KEY_ALPH)
    vid = np.arange(cs.VAL_ALPH)

    stats = dict(
        key_interp_r=cv_order_r(Ek, kid, shuffle=True),
        key_extrap_r=cv_order_r(Ek, kid, shuffle=False),
        value_interp_r=cv_order_r(Ev, vid, shuffle=True),
        value_extrap_r=cv_order_r(Ev, vid, shuffle=False),
        key_best_axis_r=best_axis_r(Ek, kid), value_best_axis_r=best_axis_r(Ev, vid),
    )
    print(json.dumps(stats, indent=2))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4),
                                        gridspec_kw={"width_ratios": [1, 1, 1.15],
                                                     "wspace": 0.5})
    Ck, Cv = cos_matrix(Ek), cos_matrix(Ev)
    for ax, C, ttl, ids in [(ax1, Ck, "Key–key cosine (ordered by key id)", "key id"),
                            (ax2, Cv, "Value–value cosine (ordered by value id)", "value id")]:
        im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xlabel(ids); ax.set_ylabel(ids); ax.set_title(ttl, fontsize=10)
        ax.grid(False); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cosine", fontsize=8)

    x = np.arange(2)
    interp = [stats["key_interp_r"], stats["value_interp_r"]]
    extrap = [stats["key_extrap_r"], stats["value_extrap_r"]]
    ax3.bar(x - 0.2, interp, 0.4, color=BLUE, label="interpolation CV r", zorder=3)
    ax3.bar(x + 0.2, extrap, 0.4, color=ORANGE, label="extrapolation CV r", zorder=3)
    ax3.axhline(0, color=INK_2, lw=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(["keys\n(compared)", "values\n(payload)"])
    ax3.set_ylim(-0.65, 1.05)
    ax3.set_ylabel("linear id-decode correlation")
    ax3.set_title("Only keys carry numeric-order structure", fontsize=10)
    ax3.legend(fontsize=8, loc="upper right")
    for xi, v in zip(x - 0.2, interp):
        ax3.text(xi, v + (0.03 if v >= 0 else -0.09), f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, extrap):
        ax3.text(xi, v + (0.03 if v >= 0 else -0.09), f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle("Keys are laid out by value; values are an arbitrary code",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(FIG / "emb_key_value_order.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/emb_key_value_order.png")
    (HERE / "embeddings.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
