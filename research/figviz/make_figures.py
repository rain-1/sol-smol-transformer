"""Generate every figure embedded in reports/ from the research results.

Run: python research/figviz/make_figures.py

Attention heatmaps are captured live from the checkpoints in
research/attention/checkpoints/; all other figures are drawn from the
committed results.json files. Output PNGs land in reports/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import (  # noqa: E402
    apply_style, SEQ_BLUE, DIVERGING, TASK_COLORS, TASK_LABELS, TASK_ORDER,
    BLUE, ORANGE, VIOLET, GREEN, INK, INK_2,
)
from app.model import ModelConfig, TinyTransformer  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    return json.loads((ROOT / "research" / name / "results.json").read_text())


def save(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def heatmap(ax, M, cmap=SEQ_BLUE, vmin=0.0, vmax=1.0, xt=None, yt=None,
            xlabel=None, ylabel=None, cbar=False, fig=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    if xt is not None:
        ax.set_xticklabels(xt, fontsize=8)
    if yt is not None:
        ax.set_yticklabels(yt, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar and fig is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# --------------------------------------------------------------------------- #
# Attention circuits: live attention heatmaps from the checkpoints.
# --------------------------------------------------------------------------- #
ATTN_CFG = {
    "copy": dict(d_model=8, n_heads=1, n_layers=1, d_ff=8),
    "reverse": dict(d_model=16, n_heads=2, n_layers=2, d_ff=16),
    "sort": dict(d_model=16, n_heads=1, n_layers=1, d_ff=16),
    "rotate_left": dict(d_model=8, n_heads=1, n_layers=1, d_ff=8),
}


@torch.inference_mode()
def capture_attention(task: str, digits: list[int]):
    """Return list of [H,Q,K] attention arrays and the token labels for one string."""
    ckpt = torch.load(ROOT / "research/attention/checkpoints" / f"{task}.pt",
                      map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ckpt["model_config"].items()})
    model = TinyTransformer(cfg).eval()
    model.load_state_dict(ckpt["state_dict"])
    x = torch.tensor([digits + [10]], dtype=torch.long)  # 10 == '='
    _, layers = model(x, capture=True)
    labels = [str(d) for d in digits] + ["="]
    return [layer["attention"][0].numpy() for layer in layers], labels


def fig_attention_maps():
    """One representative attention matrix per key head, captured live."""
    digits = [3, 1, 4, 1, 5, 9]  # length-6 example
    panels = [
        ("reverse", 0, 0, "reverse · L0H0"),
        ("reverse", 0, 1, "reverse · L0H1"),
        ("reverse", 1, 0, "reverse · L1H0"),
        ("reverse", 1, 1, "reverse · L1H1"),
        ("copy", 0, 0, "copy · L0H0"),
        ("rotate_left", 0, 0, "rotate-left · L0H0"),
        ("sort", 0, 0, "sort · L0H0"),
    ]
    caches = {t: capture_attention(t, digits) for t in ATTN_CFG}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    axes = axes.ravel()
    im = None
    for ax, (task, layer, head, title) in zip(axes, panels):
        attn, labels = caches[task]
        M = attn[layer][head]
        im = heatmap(ax, M, xt=labels, yt=labels,
                     xlabel="key (attended-to)", ylabel="query (output pos)")
        ax.set_title(title, fontsize=10)
    axes[-1].axis("off")
    fig.suptitle("Learned attention patterns (length-6 example “314159=”)",
                 fontsize=13, fontweight="bold", y=1.0)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("attention weight", fontsize=9)
    save(fig, "attention_maps.png")


def fig_attention_metrics():
    d = load("attention")["tasks"]
    # Required-source mass and top-1 for the routing heads.
    rows = [
        ("reverse L1H0", "reverse", "L1H0"),
        ("reverse L1H1", "reverse", "L1H1"),
        ("rotate L0H0", "rotate_left", "L0H0"),
        ("copy L0H0", "copy", "L0H0"),
    ]
    labels = [r[0] for r in rows]
    mass = [d[t]["heads"][h]["mean"].get("required_source_mass", 0) for _, t, h in rows]
    top1 = [d[t]["heads"][h]["mean"].get("required_source_top1", 0) for _, t, h in rows]
    y = np.arange(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    ax1.barh(y, mass, color=BLUE, height=0.6, zorder=3)
    ax1.set_yticks(y); ax1.set_yticklabels(labels)
    ax1.set_xlim(0, 1); ax1.invert_yaxis()
    ax1.set_title("Mean attention on required source")
    ax1.set_xlabel("attention mass")
    for yi, v in zip(y, mass):
        ax1.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax2.barh(y, top1, color=ORANGE, height=0.6, zorder=3)
    ax2.set_yticks(y); ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 1); ax2.invert_yaxis()
    ax2.set_title("Required source ranked top-1")
    ax2.set_xlabel("fraction of digit queries")
    for yi, v in zip(y, top1):
        ax2.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK)
    save(fig, "attention_routing_metrics.png")


def fig_attention_ablation():
    d = load("attention")["tasks"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    xs, labels, colors = [], [], []
    i = 0
    for task in TASK_ORDER:
        heads = d[task]["causal_ablation"]["heads"]
        for h, v in heads.items():
            xs.append((i, v["exact_drop"]))
            labels.append(f"{TASK_LABELS[task]}\n{h}")
            colors.append(TASK_COLORS[task])
            i += 1
    pos = [x[0] for x in xs]
    ax.bar(pos, [x[1] for x in xs], color=colors, zorder=3, width=0.72)
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("exact-accuracy drop")
    ax.set_title("Causal head ablation: exact-accuracy lost when zeroing each head")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TASK_COLORS[t]) for t in TASK_ORDER]
    ax.legend(handles, [TASK_LABELS[t] for t in TASK_ORDER], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    save(fig, "attention_ablation.png")


def fig_attention_by_length():
    d = load("attention")["tasks"]
    fig, ax = plt.subplots(figsize=(7, 4))
    series = [
        ("reverse L1H0", "reverse", "L1H0", ORANGE),
        ("reverse L1H1", "reverse", "L1H1", "#c04a12"),
        ("rotate L0H0", "rotate_left", "L0H0", GREEN),
    ]
    for label, t, h, c in series:
        bl = d[t]["heads"][h]["by_length"]
        ns = sorted(int(n) for n in bl)
        ys = [bl[str(n)]["required_source_mass"] for n in ns]
        ax.plot(ns, ys, "-o", color=c, label=label, lw=2, ms=6, zorder=3)
    ax.set_xlabel("sequence length"); ax.set_ylabel("required-source mass")
    ax.set_ylim(0, 1.02)
    ax.set_title("Routing sharpness vs. sequence length")
    ax.legend()
    save(fig, "attention_by_length.png")


def grouped_bars(ax, groups, series, values, colors, ylabel, title, ylim=None,
                 pct=False):
    """values[s][g]; series legend."""
    n_s = len(series)
    x = np.arange(len(groups))
    w = 0.8 / n_s
    for si, s in enumerate(series):
        off = (si - (n_s - 1) / 2) * w
        ax.bar(x + off, values[si], w, label=s, color=colors[si], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()


def annotate_heatmap(ax, M, fmt="{:.2f}", thr=0.5, textcolors=("#0b0b0b", "#ffffff")):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color=textcolors[int(v > thr)])


# --------------------------------------------------------------------------- #
# Causal interventions
# --------------------------------------------------------------------------- #
def fig_causal_components():
    d = load("causal")["tasks"]
    # Build (task,layer) rows.
    rows, attn, mlp = [], [], []
    for t in TASK_ORDER:
        comps = d[t]["layer_components"]
        by = {}
        for c in comps:
            by[(c["layer"], c["component"])] = c["effect"]["exact"]
        n_layers = 1 + max(c["layer"] for c in comps)
        for L in range(n_layers):
            rows.append(f"{TASK_LABELS[t]}\nL{L}")
            attn.append(by.get((L, "attention"), np.nan))
            mlp.append(by.get((L, "mlp"), np.nan))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, attn, 0.4, label="remove attention", color=BLUE, zorder=3)
    ax.bar(x + 0.2, mlp, 0.4, label="remove MLP", color=ORANGE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(rows, fontsize=8)
    ax.set_ylabel("exact-accuracy drop"); ax.set_ylim(0, 1.05)
    ax.set_title("Whole-component necessity (zero-ablation)")
    ax.legend()
    ax.annotate("reverse L1 MLP\ndispensable (0.0)", xy=(2.2, 0.0), xytext=(2.3, 0.4),
                fontsize=8, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1))
    save(fig, "causal_components.png")


def fig_causal_delimiter():
    d = load("causal")["tasks"]
    groups = [TASK_LABELS[t] for t in TASK_ORDER]
    base = [d[t]["baseline"]["digit_exact"] for t in TASK_ORDER]
    eq = [d[t]["equals_to_zero"]["ablated"]["digit_exact"] for t in TASK_ORDER]
    pos = [d[t]["zero_position_embedding"]["ablated"]["digit_exact"] for t in TASK_ORDER]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    grouped_bars(ax, groups, ["baseline", "'=' → '0'", "zero position emb."],
                 [base, eq, pos], [BLUE, ORANGE, VIOLET],
                 "digit-exact accuracy", "Delimiter and position-embedding interventions",
                 ylim=(0, 1.05))
    save(fig, "causal_delimiter.png")


def fig_causal_neurons():
    d = load("causal")["tasks"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for t, c in [("reverse", ORANGE), ("sort", VIOLET)]:
        neurons = d[t]["neurons"]
        eff = sorted((n["effect"]["exact"] for n in neurons), reverse=True)
        ax.plot(range(1, len(eff) + 1), eff, "-o", color=c, ms=5, lw=2,
                label=TASK_LABELS[t], zorder=3)
    ax.set_xlabel("MLP neuron rank (layer 0)")
    ax.set_ylabel("exact-accuracy drop")
    ax.set_title("Single-neuron ablation effects (sorted)")
    ax.legend()
    save(fig, "causal_neurons.png")


# --------------------------------------------------------------------------- #
# MLP & representation study
# --------------------------------------------------------------------------- #
FACTORS = ["digit", "position", "length"]


def _mlp_layer_table(runs, extract):
    """extract(layer_dict, factor)->float; returns rows (task x layer) averaged over seeds."""
    acc = {}
    for run in runs:
        t = run["task"]
        for li, ld in enumerate(run["layers"]):
            for f in FACTORS:
                acc.setdefault((t, li), {}).setdefault(f, []).append(extract(ld, f))
    row_labels, M = [], []
    for t in TASK_ORDER:
        for li in range(2):
            row_labels.append(f"{TASK_LABELS[t]} · L{li}")
            M.append([float(np.mean(acc[(t, li)][f])) for f in FACTORS])
    return row_labels, np.array(M)


def fig_mlp_probes():
    runs = load("mlp")["runs"]
    row_labels, M = _mlp_layer_table(
        runs, lambda ld, f: ld["mlp_probes"][f]["accuracy"])
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    heatmap(ax, M, vmin=0, vmax=1, xt=FACTORS, yt=row_labels, xlabel="decoded variable")
    annotate_heatmap(ax, M, "{:.2f}", thr=0.6)
    ax.set_title("Linear probe accuracy in MLP activations")
    save(fig, "mlp_probes.png")


def fig_mlp_selectivity():
    runs = load("mlp")["runs"]
    row_labels, M = _mlp_layer_table(
        runs, lambda ld, f: ld["selectivity"][f]["max_eta2"])
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    heatmap(ax, M, vmin=0, vmax=1, xt=FACTORS, yt=row_labels, xlabel="label")
    annotate_heatmap(ax, M, "{:.2f}", thr=0.6)
    ax.set_title("Max neuron selectivity (η²)")
    save(fig, "mlp_selectivity.png")


def fig_mlp_cka():
    d = load("mlp")["cka"]
    M = np.array([[d[a][b] for b in TASK_ORDER] for a in TASK_ORDER])
    labels = [TASK_LABELS[t] for t in TASK_ORDER]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    heatmap(ax, M, cmap=SEQ_BLUE, vmin=0, vmax=1, xt=labels, yt=labels)
    annotate_heatmap(ax, M, "{:.2f}", thr=0.6)
    ax.set_title("Cross-task final-residual CKA")
    save(fig, "mlp_cka.png")


def fig_mlp_ablation():
    runs = load("mlp")["runs"]
    agg = {}
    for run in runs:
        t = run["task"]
        for li, ld in enumerate(run["layers"]):
            a = ld["ablation"]
            agg.setdefault((t, li), {"sel": [], "rnd": []})
            agg[(t, li)]["sel"].append(a["top_selective"]["exact"])
            agg[(t, li)]["rnd"].append(a["random_mean"]["exact"])
    rows = [(t, li) for t in TASK_ORDER for li in range(2)]
    labels = [f"{TASK_LABELS[t]} · L{li}" for t, li in rows]
    sel = [np.mean(agg[r]["sel"]) for r in rows]
    rnd = [np.mean(agg[r]["rnd"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, sel, 0.4, label="most-selective 10%", color=BLUE, zorder=3)
    ax.bar(x + 0.2, rnd, 0.4, label="random 10% (mean)", color=ORANGE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("exact accuracy after ablation")
    ax.set_ylim(0.9, 1.005)
    ax.set_title("Selective vs. random neuron ablation")
    ax.legend()
    save(fig, "mlp_ablation.png")


# --------------------------------------------------------------------------- #
# Multi-task routing
# --------------------------------------------------------------------------- #
def fig_routing_prefix_attention():
    d = load("multitask_routing")
    obs = d["observations"]
    # attention_to_operation: [n_layers][n_heads]
    heads = []
    sample = obs[TASK_ORDER[0]]["attention_to_operation"]
    for L in range(len(sample)):
        for h in range(len(sample[L])):
            heads.append(f"L{L+1}H{h}")
    M = np.array([[obs[t]["attention_to_operation"][int(hd[1]) - 1][int(hd[3])]
                   for hd in heads] for t in TASK_ORDER])
    labels = [TASK_LABELS[t] for t in TASK_ORDER]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    heatmap(ax, M, cmap=SEQ_BLUE, vmin=0, vmax=1, xt=heads, yt=labels,
            xlabel="attention head")
    annotate_heatmap(ax, M, "{:.2f}", thr=0.55)
    ax.set_title("Mean attention from digit queries to the operation prefix")
    save(fig, "routing_prefix_attention.png")


def fig_routing_head_ablation():
    d = load("multitask_routing")["head_ablations"]
    heads = [f"L{r['layer']+1}H{r['head']}" for r in d]
    M = np.array([[r["exact_effect"][t] for t in TASK_ORDER] for r in d])
    labels = [TASK_LABELS[t] for t in TASK_ORDER]
    fig, ax = plt.subplots(figsize=(5.6, 6.6))
    heatmap(ax, M, cmap=SEQ_BLUE, vmin=0, vmax=1, xt=labels, yt=heads,
            ylabel="ablated head")
    annotate_heatmap(ax, M, "{:.2f}", thr=0.55)
    ax.set_title("Head-ablation exact-accuracy drop\n(shared multi-task model)")
    save(fig, "routing_head_ablation.png")


# --------------------------------------------------------------------------- #
# Multi-task causal control
# --------------------------------------------------------------------------- #
def fig_causal_control_patching():
    d = load("multitask_causal")["residual_patching"]
    site_keys = ["operation_only", "nonoperation", "all_real"]
    site_labels = ["operation pos only", "digit positions", "all real positions"]
    stage_keys = ["0", "1", "2"]
    stage_labels = ["embedding", "after block 1", "after block 2"]
    M = np.zeros((3, 3))
    for si, sk in enumerate(site_keys):
        for gi, gk in enumerate(stage_keys):
            vals = []
            for src, dsts in d.items():
                for dst, sd in dsts.items():
                    vals.append(sd[gk][sk]["destination_target"]["exact"])
            M[si, gi] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    heatmap(ax, M, cmap=SEQ_BLUE, vmin=0, vmax=1, xt=stage_labels, yt=site_labels,
            xlabel="patch depth", ylabel="patched positions")
    annotate_heatmap(ax, M, "{:.2f}", thr=0.55)
    ax.set_title("Residual patching: mean destination-task exact accuracy")
    save(fig, "causal_control_patching.png")


def fig_causal_control_interpolation():
    d = load("multitask_causal")["embedding_interpolation"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    colors = [BLUE, ORANGE, VIOLET, GREEN, "#199e70", "#d55181"]
    for (pair, steps), c in zip(d.items(), colors):
        src, dst = pair.split("->")
        keys = sorted(steps, key=float)
        xs = [float(k) for k in keys]
        ys = [steps[k][dst] for k in keys]
        ax.plot(xs, ys, "-o", color=c, ms=5, lw=1.8,
                label=pair.replace("->", " → ").replace("_left", ""), zorder=3)
    ax.set_xlabel("interpolation toward destination embedding")
    ax.set_ylabel("destination-task exact accuracy")
    ax.set_title("Operation-embedding interpolation is cliff-like, not a smooth blend")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "causal_control_interpolation.png")


def fig_causal_control_ablation():
    d = load("multitask_causal")["component_ablations"]
    sample = d[TASK_ORDER[0]]
    rows = [("mlps", L, f"{L} MLP") for L in sorted(sample["mlps"])]
    rows += [("heads", h, h) for h in sorted(sample["heads"])]
    labels, M = [], []
    for group, key, label in rows:
        labels.append(label)
        M.append([d[t][group][key]["exact"] for t in TASK_ORDER])
    M = np.array(M)
    tlabels = [TASK_LABELS[t] for t in TASK_ORDER]
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    heatmap(ax, M, cmap=SEQ_BLUE, vmin=0, vmax=1, xt=tlabels, yt=labels,
            ylabel="ablated component")
    annotate_heatmap(ax, M, "{:.2f}", thr=0.55)
    ax.set_title("Exact accuracy after single-component ablation")
    save(fig, "causal_control_ablation.png")


# --------------------------------------------------------------------------- #
# Multi-task representation geometry (across depth)
# --------------------------------------------------------------------------- #
STAGES = ["embedding", "layer_1", "layer_2", "final"]
STAGE_LABELS = ["embed", "block 1", "block 2", "final"]


def _geom_mean_over_lengths(lengths, path):
    """path: callable(stage_dict)->float ; returns list over STAGES averaged over lengths."""
    out = []
    for st in STAGES:
        vals = []
        for L, ld in lengths.items():
            try:
                vals.append(path(ld, st))
            except Exception:
                pass
        out.append(np.mean(vals) if vals else np.nan)
    return out


def fig_geom_probe_and_pca():
    d = load("multitask_geometry")["lengths"]
    probe = _geom_mean_over_lengths(d, lambda ld, st: ld["probe"][st])
    pc1 = _geom_mean_over_lengths(d, lambda ld, st: ld["pca"][st]["pc1"])
    erank = _geom_mean_over_lengths(d, lambda ld, st: ld["pca"][st]["effective_rank"])
    x = np.arange(len(STAGES))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))
    ax1.plot(x, probe, "-o", color=BLUE, lw=2, ms=7, zorder=3)
    ax1.axhline(0.25, ls="--", color=INK_2, lw=1)
    ax1.text(0.05, 0.28, "chance (25%)", fontsize=8, color=INK_2)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("task-probe accuracy")
    ax1.set_title("Task identity decodable after block 1")
    ax2.plot(x, pc1, "-o", color=ORANGE, lw=2, ms=7, zorder=3)
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("PC1 variance fraction")
    ax2.set_title("Task signal concentrated early")
    ax3.plot(x, erank, "-s", color=VIOLET, lw=2, ms=6, zorder=3)
    ax3.set_ylabel("effective rank")
    ax3.set_title("Representation spreads with depth")
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(x); ax.set_xticklabels(STAGE_LABELS)
    save(fig, "geometry_probe_pca.png")


def fig_geom_cka():
    d = load("multitask_geometry")["lengths"]
    pairs = ["copy:sort", "reverse:rotate_left", "copy:reverse",
             "copy:rotate_left", "reverse:sort", "sort:rotate_left"]
    colors = [BLUE, ORANGE, VIOLET, GREEN, "#199e70", "#d55181"]
    x = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for pair, c in zip(pairs, colors):
        ys = _geom_mean_over_lengths(d, lambda ld, st, p=pair: ld["cka"][st][p])
        ax.plot(x, ys, "-o", color=c, lw=1.9, ms=5,
                label=pair.replace("rotate_left", "rotate").replace(":", "–"), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylim(0, 1.05); ax.set_ylabel("linear CKA")
    ax.set_title("Cross-task similarity collapses after block 1")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "geometry_cka.png")


def fig_geom_steering():
    d = load("multitask_geometry")["lengths"]
    tasks = ["reverse", "sort", "rotate_left"]
    l1 = [np.mean([d[L]["steering"]["layer_1"][t]["target"] for L in d]) for t in tasks]
    l2 = [np.mean([d[L]["steering"]["layer_2"][t]["target"] for L in d]) for t in tasks]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(6.6, 4))
    ax.bar(x - 0.2, l1, 0.4, label="add block-1 task vector", color=BLUE, zorder=3)
    ax.bar(x + 0.2, l2, 0.4, label="add block-2 task vector", color=ORANGE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylabel("target-task exact accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("Mean-vector steering works from block 1, not block 2")
    ax.legend(fontsize=8)
    save(fig, "geometry_steering.png")


# --------------------------------------------------------------------------- #
# Translation & rotation manifolds
# --------------------------------------------------------------------------- #
def fig_manifold_translation():
    d = load("manifolds")["translation"]["lengths"]
    stages = [("context_input", "input\ncontext"), ("layer_1", "block 1"),
              ("final", "final")]
    lens = sorted(int(k) for k in d)
    r2 = [np.mean([d[str(n)]["stages"][s]["digit_to_letter"]["r2"] for n in lens])
          for s, _ in stages]
    rmse = [np.mean([d[str(n)]["stages"][s]["digit_to_letter"]["relative_rmse"] for n in lens])
            for s, _ in stages]
    x = np.arange(len(stages))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.bar(x, r2, 0.55, color=BLUE, zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels([s[1] for s in stages])
    ax1.set_ylim(0.9, 1.001); ax1.set_ylabel("held-out R²")
    ax1.set_title("Digit→letter affine map fits almost perfectly")
    for xi, v in zip(x, r2):
        ax1.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.bar(x, rmse, 0.55, color=ORANGE, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels([s[1] for s in stages])
    ax2.set_ylabel("relative RMSE")
    ax2.set_title("Small residual error")
    for xi, v in zip(x, rmse):
        ax2.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "manifold_translation.png")


def fig_manifold_rotation():
    d = load("manifolds")["cyclic_rotation"]["lengths"]
    stages = [("context_input", "input"), ("layer_1", "block 1"),
              ("layer_2", "block 2"), ("final", "final")]
    lens = sorted(int(k) for k in d)
    follow = [np.mean([d[str(n)]["stages"][s]["token_following_affine"]["r2"] for n in lens])
              for s, _ in stages]
    same = [np.mean([d[str(n)]["stages"][s]["same_absolute_position_affine"]["r2"] for n in lens])
            for s, _ in stages]
    x = np.arange(len(stages))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(x, follow, "-o", color=BLUE, lw=2, ms=7, label="token-following", zorder=3)
    ax1.plot(x, same, "-s", color=ORANGE, lw=2, ms=6, label="same absolute position", zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels([s[1] for s in stages])
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("affine R²")
    ax1.set_title("Late states follow the moved token, not the position")
    ax1.legend(fontsize=8)
    # Fourier power of the offset kernel (final stage), averaged over lengths.
    fp = np.mean([d[str(n)]["stages"]["final"]["offset_kernel_fourier_power"][:2]
                  for n in lens], axis=0)
    bx = np.arange(len(fp))
    ax2.bar(bx, fp, 0.5, color=VIOLET, zorder=3)
    ax2.set_xticks(bx); ax2.set_xticklabels(["DC", "1st harmonic"][:len(fp)])
    ax2.set_ylabel("relative power")
    ax2.set_xlabel("offset-kernel harmonic")
    ax2.set_title("Shift structure concentrates in low harmonics")
    for xi, v in zip(bx, fp):
        ax2.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    save(fig, "manifold_rotation.png")


if __name__ == "__main__":
    apply_style()
    fig_attention_maps()
    fig_attention_metrics()
    fig_attention_ablation()
    fig_attention_by_length()
    fig_causal_components()
    fig_causal_delimiter()
    fig_causal_neurons()
    fig_mlp_probes()
    fig_mlp_selectivity()
    fig_mlp_cka()
    fig_mlp_ablation()
    fig_routing_prefix_attention()
    fig_routing_head_ablation()
    fig_causal_control_patching()
    fig_causal_control_interpolation()
    fig_causal_control_ablation()
    fig_geom_probe_and_pca()
    fig_geom_cka()
    fig_geom_steering()
    fig_manifold_translation()
    fig_manifold_rotation()
    print("all figures done")

