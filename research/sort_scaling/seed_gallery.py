"""Cross-seed attention gallery: all heads, all 5 seeds, on fixed example inputs.

For each of 7 example sequences (2 increasing, 2 decreasing, 3 random) produces one
figure: rows = the 5 universality seeds, columns = the 4 attention heads, each cell
the head's attention on that sequence (teacher-forced on the correct sorted output).
The per-seed value/induction head (from seeds_metrics.json) is highlighted, so the
"clean single sort head in seed 42 but not the others" story is visible by eye.

Usage: python research/sort_scaling/seed_gallery.py
Figures -> reports/figures/gallery_*.png
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
from style import apply_style, SEQ_BLUE, ORANGE, VIOLET, INK, INK_2  # noqa: E402
import causal_sort as cs  # noqa: E402
from deep_interp import load  # noqa: E402
from walkthrough import full_capture, tok_label  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (seed, checkpoint, induction/value head, clean sort head or None).
# seed 42 is the main checkpoint the deep study used: it uniquely has a clean single
# sort head (L1H1); the fresh universality seeds do not (their sort channel is diffuse).
SEED_SPECS = [
    (42, HERE / "checkpoints" / "causal_pairs_L30.pt", "L1H0", "L1H1"),
    (0, HERE / "checkpoints" / "seed_0.pt", "L1H0", None),
    (1, HERE / "checkpoints" / "seed_1.pt", "L1H0", None),
    (2, HERE / "checkpoints" / "seed_2.pt", "L1H1", None),
    (3, HERE / "checkpoints" / "seed_3.pt", "L1H1", None),
    (7, HERE / "checkpoints" / "seed_7.pt", "L1H1", None),
]

# 7 example inputs (distinct keys, as trained). value ids are 50-69.
EXAMPLES = [
    ("inc1", "increasing", [3, 8, 15, 22, 30], [55, 62, 51, 68, 59]),
    ("inc2", "increasing", [1, 12, 20, 35, 44], [60, 52, 66, 58, 63]),
    ("dec1", "decreasing", [44, 33, 21, 10, 2], [53, 61, 57, 69, 50]),
    ("dec2", "decreasing", [40, 28, 19, 9, 5], [64, 51, 67, 54, 59]),
    ("rnd1", "random", [17, 3, 41, 27, 8], [55, 61, 52, 68, 59]),
    ("rnd2", "random", [23, 5, 38, 11, 46], [58, 66, 51, 63, 69]),
    ("rnd3", "random", [30, 14, 2, 49, 21], [62, 50, 57, 68, 53]),
]


def build_tokens(keys, vals):
    p = len(keys)
    keys_t, vals_t = torch.tensor(keys), torch.tensor(vals)
    order = keys_t.argsort(stable=True)
    sk, sv = keys_t[order], vals_t[order]
    seq = []
    for k, v in zip(keys, vals):
        seq += [k, v]
    seq += [cs.SEP]
    for k, v in zip(sk.tolist(), sv.tolist()):
        seq += [k, v]
    seq += [cs.EOS]
    return torch.tensor([seq], device=DEV), p


def main():
    apply_style()
    models = {s: load(path)[0] for s, path, _, _ in SEED_SPECS}
    heads = [(0, 0), (0, 1), (1, 0), (1, 1)]
    head_names = ["L0H0", "L0H1", "L1H0", "L1H1"]
    nrows = len(SEED_SPECS)

    for name, kind, keys, vals in EXAMPLES:
        tokens, p = build_tokens(keys, vals)
        labels = [tok_label(t) for t in tokens[0].tolist()]
        L = tokens.shape[1]
        fig, axes = plt.subplots(nrows, 4, figsize=(15, 4 + 2.6 * nrows))
        for r, (s, _, ind, sort) in enumerate(SEED_SPECS):
            model = models[s]
            _, caps = full_capture(model, tokens)
            # did this seed sort it right (teacher-forced)?
            with torch.inference_mode():
                pred = model(tokens[:, :-1])[0].argmax(-1)[0]
            ok = (pred[2 * p:2 * p + 2 * p] == tokens[0, 2 * p + 1:2 * p + 1 + 2 * p]).all().item()
            for c, (li, h) in enumerate(heads):
                ax = axes[r, c]
                A = np.tril(caps[li]["attn"][0, h].cpu().numpy())
                ax.imshow(A, cmap=SEQ_BLUE, vmin=0, vmax=A.max(), aspect="equal")
                ax.set_xticks(range(L)); ax.set_yticks(range(L))
                if r == nrows - 1:
                    ax.set_xticklabels(labels, fontsize=5, rotation=90)
                else:
                    ax.set_xticklabels([])
                if c == 0:
                    ax.set_yticklabels(labels, fontsize=5)
                    tag = "  (main)" if s == 42 else ""
                    ax.set_ylabel(f"seed {s}{tag}\n{'sorted OK' if ok else 'ERR'}",
                                  fontsize=9, color=INK if ok else "#c02c2b")
                else:
                    ax.set_yticklabels([])
                is_ind = head_names[c] == ind
                is_sort = sort is not None and head_names[c] == sort
                if is_ind:
                    color, note = ORANGE, "  ← induction/value"
                elif is_sort:
                    color, note = VIOLET, "  ← clean sort head"
                else:
                    color, note = INK_2, ""
                ax.set_title(head_names[c] + note, fontsize=8.5, color=color,
                             fontweight="bold" if (is_ind or is_sort) else "normal")
                ax.tick_params(length=0)
                for sp in ax.spines.values():
                    sp.set_visible(is_ind or is_sort)
                    sp.set_color(color)
                    sp.set_linewidth(2 if (is_ind or is_sort) else 0.5)
        ktxt = " ".join(f"k{k}" for k in keys)
        fig.suptitle(f"Attention across seeds (42 = main) — {kind} input:  {ktxt}\n"
                     f"(rows = seed, cols = head; orange = induction/value head, "
                     f"violet = seed-42's clean sort head; length-{p}, teacher-forced)",
                     fontsize=12, fontweight="bold", y=0.995)
        fig.savefig(FIG / f"gallery_{name}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote reports/figures/gallery_{name}.png")


if __name__ == "__main__":
    main()
