"""Representation geometry study for a trained operation-token multi-task model.

Uses paired examples (identical digit strings under every operation), reports
linear task decodability, linear CKA, mean task-vector coherence, PCA variance,
and causal mean-vector steering. No training is performed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.model import ModelConfig, OPERATION_IDS, PAD_ID, TinyTransformer  # noqa: E402
from app.training import OPERATIONS, transform  # noqa: E402


def paired_batch(length: int, count: int, seed: int, device: torch.device):
    g = torch.Generator(device="cpu").manual_seed(seed + length)
    digits = torch.randint(0, 10, (count, length), generator=g).to(device)
    batches = {}
    for task in OPERATIONS:
        x = torch.full((count, 10), PAD_ID, dtype=torch.long, device=device)
        y = torch.full_like(x, PAD_ID)
        x[:, 0] = OPERATION_IDS[task]
        x[:, 1:length + 1] = digits
        x[:, length + 1] = 10
        y[:, 1:length + 1] = transform(digits, task)
        y[:, length + 1] = 10
        batches[task] = (x, y)
    return batches


@torch.no_grad()
def states(model, x):
    out = {}
    handles = []
    out["embedding"] = (model.token_embedding(x) + model.position_embedding[:, :x.shape[1]]).detach()
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(
            lambda _m, _a, value, i=i: out.__setitem__(f"layer_{i+1}", value[0].detach())))
    logits, _ = model(x)
    for h in handles:
        h.remove()
    out["final"] = model.final_norm(out[f"layer_{len(model.blocks)}"]).detach()
    return logits, out


def accuracy(logits, y, length):
    good = logits.argmax(-1)[:, 1:length + 2].eq(y[:, 1:length + 2])
    return {"token": good.float().mean().item(), "exact": good.all(1).float().mean().item()}


def linear_cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    xy = (x.T @ y).square().sum()
    denom = ((x.T @ x).square().sum() * (y.T @ y).square().sum()).sqrt()
    return (xy / denom.clamp_min(1e-12)).item()


def ridge_probe(x, labels, groups, seed=0):
    # Split by underlying digit-string group, preventing paired-task leakage.
    g = torch.Generator().manual_seed(seed)
    unique = groups.unique()
    test_groups = unique[torch.randperm(len(unique), generator=g)[:len(unique)//4]]
    test = torch.isin(groups, test_groups)
    train = ~test
    mu, sd = x[train].mean(0), x[train].std(0).clamp_min(1e-5)
    z = (x - mu) / sd
    z = torch.cat([z, torch.ones(len(z), 1)], 1)
    onehot = torch.nn.functional.one_hot(labels[train], len(OPERATIONS)).float()
    reg = 1e-2 * torch.eye(z.shape[1]); reg[-1, -1] = 0
    w = torch.linalg.solve(z[train].T @ z[train] + reg, z[train].T @ onehot)
    return z[test].matmul(w).argmax(1).eq(labels[test]).float().mean().item()


@torch.no_grad()
def steering(model, x, y_source, y_target, delta, layer, length):
    def hook(_m, _a, value):
        h, extra = value
        h = h.clone()
        h[:, :length + 2] += delta
        return h, extra
    handle = model.blocks[layer].register_forward_hook(hook)
    logits, _ = model(x)
    handle.remove()
    return {"target": accuracy(logits, y_target, length)["exact"],
            "source": accuracy(logits, y_source, length)["exact"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/multi_task-20260711-103650.pt")
    p.add_argument("--output", default="research/multitask_geometry/results.json")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = TinyTransformer(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    result = {"checkpoint": args.checkpoint, "samples_per_length": args.samples,
              "seed": args.seed, "tasks": list(OPERATIONS), "lengths": {},
              "aggregate": {}}
    all_repr = {stage: [] for stage in ["embedding", "layer_1", "layer_2", "final"]}
    all_labels, all_groups = [], []
    vector_by_stage_task_length = {}
    global_group = 0
    for length in range(2, 9):
        batches = paired_batch(length, args.samples, args.seed, device)
        reps, ys = {}, {}
        row = {"accuracy": {}, "cka": {}, "probe": {}, "pca": {}, "task_vectors": {}, "steering": {}}
        for ti, task in enumerate(OPERATIONS):
            x, y = batches[task]; logits, rep = states(model, x)
            reps[task], ys[task] = rep, y
            row["accuracy"][task] = accuracy(logits, y, length)
            for stage in all_repr:
                # Average digit positions: compact example-level representation.
                all_repr[stage].append(rep[stage][:, 1:length+1].mean(1).cpu())
            all_labels.append(torch.full((args.samples,), ti, dtype=torch.long))
            all_groups.append(torch.arange(global_group, global_group + args.samples))
        global_group += args.samples
        for stage in all_repr:
            matrix = torch.cat([reps[t][stage][:, 1:length+1].mean(1).cpu() for t in OPERATIONS])
            labels = torch.arange(4).repeat_interleave(args.samples)
            groups = torch.arange(args.samples).repeat(4)
            row["probe"][stage] = ridge_probe(matrix, labels, groups, args.seed)
            centered = matrix - matrix.mean(0)
            s = torch.linalg.svdvals(centered)
            variance = s.square() / s.square().sum()
            row["pca"][stage] = {"pc1": variance[0].item(), "pc2": variance[1].item(),
                                  "effective_rank": torch.exp(-(variance * variance.clamp_min(1e-12).log()).sum()).item()}
            for i, a in enumerate(OPERATIONS):
                for b in OPERATIONS[i+1:]:
                    key = f"{a}:{b}"
                    xa = reps[a][stage][:, 1:length+1].reshape(-1, 32).cpu()
                    xb = reps[b][stage][:, 1:length+1].reshape(-1, 32).cpu()
                    row["cka"].setdefault(stage, {})[key] = linear_cka(xa, xb)
            base = reps["copy"][stage][:, :length+2]
            for task in OPERATIONS[1:]:
                diffs = reps[task][stage][:, :length+2] - base
                mean = diffs.mean(0)
                explained = 1 - (diffs - mean).square().sum() / diffs.square().sum().clamp_min(1e-12)
                row["task_vectors"].setdefault(stage, {})[task] = {
                    "mean_fraction_difference_energy": explained.item(),
                    "mean_norm": mean.norm(dim=-1).mean().item()}
                vector_by_stage_task_length[(stage, task, length)] = mean.cpu()
        # Mean-vector causal steering at both residual layers, source copy -> target.
        for li, stage in enumerate(["layer_1", "layer_2"]):
            for target in OPERATIONS[1:]:
                delta = (reps[target][stage][:, :length+2] - reps["copy"][stage][:, :length+2]).mean(0)
                row["steering"].setdefault(stage, {})[target] = steering(
                    model, batches["copy"][0], ys["copy"], ys[target], delta, li, length)
        result["lengths"][str(length)] = row
    # Cross-length task-vector cosine, aligned at prefix and relative digit positions via mean over tokens.
    coherence = {}
    for stage in ["embedding", "layer_1", "layer_2", "final"]:
        coherence[stage] = {}
        for task in OPERATIONS[1:]:
            vecs = [vector_by_stage_task_length[(stage, task, l)].mean(0) for l in range(2, 9)]
            cos = [torch.nn.functional.cosine_similarity(vecs[i], vecs[j], dim=0).item()
                   for i in range(7) for j in range(i+1, 7)]
            coherence[stage][task] = {"mean_pairwise_cosine": sum(cos)/len(cos),
                                      "min_pairwise_cosine": min(cos)}
    result["aggregate"]["task_vector_cross_length_coherence"] = coherence
    # Overall probes across all lengths.
    labels, groups = torch.cat(all_labels), torch.cat(all_groups)
    result["aggregate"]["task_probe_all_lengths"] = {
        stage: ridge_probe(torch.cat(xs), labels, groups, args.seed) for stage, xs in all_repr.items()}
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
