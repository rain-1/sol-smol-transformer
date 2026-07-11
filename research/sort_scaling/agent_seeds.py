"""Universality study: train several fresh minimal models with different seeds and
re-run the seed-42 interpretability battery on each, aligning heads by behavior.

Reuses causal_sort.train/make_batch and deep_interp analysis functions unchanged.

Usage:
  python research/sort_scaling/agent_seeds.py --train      # train all seeds
  python research/sort_scaling/agent_seeds.py --analyze    # analyze + figure
  python research/sort_scaling/agent_seeds.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "figviz"))
sys.path.insert(0, str(HERE))

import causal_sort as cs  # noqa: E402
import deep_interp as di  # noqa: E402
from style import apply_style, BLUE, ORANGE, VIOLET, GREEN, INK_2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CKPT = HERE / "checkpoints"
FINDINGS = HERE / "agent_findings"
FIG = ROOT / "reports" / "figures"
DEV = di.DEV

CFG = dict(d_model=48, n_heads=2, n_layers=2, d_ff=96)
SEEDS = [0, 1, 2, 3, 7]
MAX_PAIRS = 16
STEPS = 11000
ANALYSIS_LEN = 12


def ckpt_path(seed):
    return CKPT / f"seed_{seed}.pt"


def train_all():
    CKPT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        path = ckpt_path(seed)
        print(f"\n=== training seed {seed} (steps={STEPS}, max_pairs={MAX_PAIRS}) ===",
              flush=True)
        model, secs = cs.train(CFG, max_pairs=MAX_PAIRS, steps=STEPS, batch_size=256,
                               lr=0.002, seed=seed, device=DEV,
                               log_every=STEPS // 5)
        tf = cs.evaluate_teacher_forced(model, MAX_PAIRS, DEV, batches=20)
        gen = cs.generate_exact(model, MAX_PAIRS, DEV, batch_size=512)
        print(f"seed {seed}: tf_token {tf:.4f} gen_exact {gen:.4f} ({secs:.0f}s)", flush=True)
        torch.save({"model_config": model.cfg, "state_dict": model.state_dict(),
                    "key_alph": cs.KEY_ALPH, "val_alph": cs.VAL_ALPH,
                    "max_pairs": MAX_PAIRS, "seed": seed,
                    "tf_token": tf, "gen_exact": gen}, path)
        print("saved", path, flush=True)


@torch.inference_mode()
def all_head_induction(model, p, n=256):
    """Induction score (attn on matching input value at value-emit positions) for
    every head, returned as {'L{li}H{h}': score}."""
    seq, keys, order = di.batch(model, p, n=n)
    _, attn, _ = di.run_capture(model, seq[:, :-1])
    sep = 2 * p
    B = seq.shape[0]
    vq = torch.tensor([sep + 1 + 2 * q for q in range(p)], device=DEV)
    val_src = 2 * order + 1
    out = {}
    for li in range(len(attn)):
        for h in range(attn[li].shape[1]):
            a = attn[li][:, h]
            s = a[torch.arange(B)[:, None], vq.view(1, -1).expand(B, -1), val_src].mean().item()
            out[f"L{li}H{h}"] = s
    return out


def analyze_seed(seed):
    model, meta = di.load(ckpt_path(seed))
    p = ANALYSIS_LEN
    res = {"seed": seed, "config": model.cfg, "length": p,
           "tf_token": meta.get("tf_token"), "gen_exact": meta.get("gen_exact")}

    # 1. head roles: per-head key- vs value-output ablation drops
    base_k, base_v = di.per_position_acc(model, p)
    width = model.cfg["d_model"] // model.cfg["n_heads"]
    abl = {}
    for li, block in enumerate(model.blocks):
        for h in range(model.cfg["n_heads"]):
            sl = slice(h * width, (h + 1) * width)
            saved = block.attn.out_proj.weight[:, sl].detach().clone()
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].zero_()
            k, v = di.per_position_acc(model, p)
            with torch.no_grad():
                block.attn.out_proj.weight[:, sl].copy_(saved)
            abl[f"L{li}H{h}"] = {"key_drop": base_k - k, "value_drop": base_v - v}
    res["baseline_key_acc"], res["baseline_value_acc"] = base_k, base_v
    res["ablation"] = abl

    # 2. per-head induction score
    ind = all_head_induction(model, p)
    res["induction_per_head"] = ind

    # Align heads by behavior.
    value_head = max(abl, key=lambda k: abl[k]["value_drop"])
    key_head = max(abl, key=lambda k: abl[k]["key_drop"])
    induction_head = max(ind, key=lambda k: ind[k])
    res["value_head"] = value_head
    res["key_head"] = key_head
    res["induction_head"] = induction_head

    vh = (int(value_head[1]), int(value_head[3]))

    # 3. precursor path: ablate each other head, re-measure induction head's score
    pp = di.precursor_path(model, p, target=(int(induction_head[1]), int(induction_head[3])))
    # rename the figure produced by precursor_path so seeds don't clobber each other
    (FIG / "deep_precursor.png").replace(FIG / f"agent_seeds_precursor_s{seed}.png")
    res["precursor"] = pp
    res["precursor_top"] = max(pp["drops"], key=lambda k: pp["drops"][k])
    res["precursor_top_drop"] = pp["drops"][res["precursor_top"]]

    # 4. circuit roles: which head reads generated output at key-emit steps = SORT head
    cr = di.circuit_roles(model, p)
    (FIG / "deep_circuit_roles.png").replace(FIG / f"agent_seeds_circuit_s{seed}.png")
    # class order: [input keys, input values, SEP, output(generated)]
    res["keystep_mass_all"] = {h: cr["key_step"][h] for h in cr["key_step"]}
    # behavioral sort head = head that reads the generated output most at key steps
    sort_head = max(cr["key_step"], key=lambda h: cr["key_step"][h][3])
    res["sort_head"] = sort_head
    res["sort_head_reads_output"] = cr["key_step"][sort_head][3]
    # the ablation-defined key head (max key-output drop) recorded for reference
    res["key_head_reads_output"] = cr["key_step"][key_head][3]
    res["key_head_keystep_mass"] = cr["key_step"][key_head]
    # SEP attention-sink: max mass any head parks on SEP at key steps
    res["max_sep_sink"] = max(cr["key_step"][h][2] for h in cr["key_step"])

    # 5. key-rank linear probe at final residual (low => sort-by-generation)
    # residual_probes writes a figure; run and capture only key_rank
    seq, keys, order = di.batch(model, p, n=512)
    _, _, resid = di.run_capture(model, seq[:, :-1])
    B = seq.shape[0]
    rank_of = torch.empty_like(keys)
    rank_of.scatter_(1, order, torch.arange(p, device=DEV).expand(B, -1))
    key_pos = list(range(0, 2 * p, 2))
    kr = {}
    for st in ["embed", "block0", "block1", "final"]:
        R = resid[st]
        X = R[:, key_pos].reshape(-1, R.shape[-1]).cpu().numpy()
        y = rank_of.reshape(-1).cpu().numpy()
        kr[st] = di._probe(X, y)
    res["key_rank_probe"] = kr

    # argmax-on-match for the induction head (fraction of value steps where the
    # induction head's argmax attention lands on the matching input value)
    seq, keys, order = di.batch(model, p, n=256)
    B = seq.shape[0]
    _, attn, _ = di.run_capture(model, seq[:, :-1])
    li, h = int(induction_head[1]), int(induction_head[3])
    a = attn[li][:, h]
    sep = 2 * p
    vq = torch.tensor([sep + 1 + 2 * q for q in range(p)], device=DEV)
    val_src = (2 * order + 1)
    picked = a[torch.arange(B)[:, None], vq.view(1, -1).expand(B, -1)].argmax(-1)
    res["induction_argmax_on_match"] = (picked == val_src).float().mean().item()

    return res


def analyze_all():
    apply_style()
    FINDINGS.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        print(f"\n=== analyzing seed {seed} ===", flush=True)
        r = analyze_seed(seed)
        results.append(r)
        print(f"seed {seed}: gen={r['gen_exact']:.3f} value_head={r['value_head']} "
              f"key_head={r['key_head']} induction_head={r['induction_head']} "
              f"precursor={r['precursor_top']}({r['precursor_top_drop']:.2f}) "
              f"key_reads_out={r['key_head_reads_output']:.2f} "
              f"keyrank_final={r['key_rank_probe']['final']:.2f}", flush=True)
    (FINDINGS / "seeds_metrics.json").write_text(json.dumps(results, indent=2))
    make_figure(results)
    return results


def make_figure(results):
    apply_style()
    seeds = [r["seed"] for r in results]
    n = len(seeds)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: value head value-drop vs key head key-drop (specialization)
    ax = axes[0]
    vdrop = [r["ablation"][r["value_head"]]["value_drop"] for r in results]
    kdrop = [r["ablation"][r["key_head"]]["key_drop"] for r in results]
    # cross terms (should be small)
    v_keydrop = [r["ablation"][r["value_head"]]["key_drop"] for r in results]
    k_valdrop = [r["ablation"][r["key_head"]]["value_drop"] for r in results]
    ax.bar(x - 0.3, vdrop, 0.2, label="value-head: value-drop", color=ORANGE, zorder=3)
    ax.bar(x - 0.1, v_keydrop, 0.2, label="value-head: key-drop", color=ORANGE, alpha=0.4, zorder=3)
    ax.bar(x + 0.1, kdrop, 0.2, label="key-head: key-drop", color=BLUE, zorder=3)
    ax.bar(x + 0.3, k_valdrop, 0.2, label="key-head: value-drop", color=BLUE, alpha=0.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy drop")
    ax.set_title("Head specialization (aligned by role)")
    ax.legend(fontsize=7)

    # Panel B: induction score of top head + precursor collapse
    ax = axes[1]
    ind = [r["induction_per_head"][r["induction_head"]] for r in results]
    prec_after = [r["precursor"]["baseline"] - r["precursor_top_drop"] for r in results]
    ax.bar(x - 0.2, ind, 0.4, label="induction head score (intact)", color=VIOLET, zorder=3)
    ax.bar(x + 0.2, prec_after, 0.4, label="after ablating top precursor", color=INK_2, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylim(0, 1.02); ax.set_ylabel("induction score")
    ax.set_title("Induction head + precursor dependency")
    ax.legend(fontsize=7)

    # Panel C: sort-by-generation evidence
    ax = axes[2]
    reads = [r["sort_head_reads_output"] for r in results]
    krank = [r["key_rank_probe"]["final"] for r in results]
    ax.bar(x - 0.2, reads, 0.4, label="sort-head reads generated output (mass)", color=GREEN, zorder=3)
    ax.bar(x + 0.2, krank, 0.4, label="key-rank probe acc (final resid)", color=BLUE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylim(0, 1.02); ax.set_ylabel("mass / accuracy")
    ax.set_title("Sort-by-generation signature")
    ax.legend(fontsize=7)

    fig.suptitle("Circuit universality across seeds (config d48 h2 l2 ff96, max_pairs=16)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.savefig(FIG / "agent_seeds_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/agent_seeds_summary.png")

    # Panel: head-role matrix (heat) of value-drop / key-drop / induction per head
    heads = [f"L{li}H{h}" for li in range(CFG["n_layers"]) for h in range(CFG["n_heads"])]
    fig, axm = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, metric, title, cmap in [
        (axm[0], "value_drop", "value-output drop", "Oranges"),
        (axm[1], "key_drop", "key-output drop", "Blues"),
        (axm[2], "induction", "induction score", "Purples")]:
        M = np.zeros((n, len(heads)))
        for i, r in enumerate(results):
            for j, hd in enumerate(heads):
                if metric == "induction":
                    M[i, j] = r["induction_per_head"][hd]
                else:
                    M[i, j] = r["ablation"][hd][metric]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(heads))); ax.set_xticklabels(heads, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels([f"s{s}" for s in seeds])
        ax.set_title(title, fontsize=10)
        for i in range(n):
            for j in range(len(heads)):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if M[i, j] < 0.6 else "white")
    fig.suptitle("Per-seed head-role matrix (heads NOT aligned; note which index carries each role)",
                 fontsize=12, fontweight="bold", y=1.06)
    fig.savefig(FIG / "agent_seeds_head_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/figures/agent_seeds_head_matrix.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.train or a.all:
        train_all()
    if a.analyze or a.all:
        analyze_all()
        # analyze_all() renames deep_interp's default-named outputs per seed, which
        # would clobber the committed main-model figures reports/figures/deep_*.png.
        # Regenerate the canonical versions from the length-30 checkpoint.
        main_ckpt = CKPT / "causal_pairs_L30.pt"
        if main_ckpt.exists():
            m, _ = di.load(main_ckpt)
            di.circuit_roles(m, 12)     # rewrites deep_circuit_roles.png
            di.precursor_path(m, 12)    # rewrites deep_precursor.png
            print("regenerated canonical deep_circuit_roles.png / deep_precursor.png")


if __name__ == "__main__":
    main()
