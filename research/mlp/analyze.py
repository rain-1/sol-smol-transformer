#!/usr/bin/env python3
"""Reproducible representation and MLP analysis for the tiny task transformers."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.model import ModelConfig, PAD_ID, TinyTransformer, VOCAB
from app.training import make_batch


TASKS = ("copy", "reverse", "sort", "rotate_left")


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(task, seed, device, steps, cfg):
    seed_all(seed)
    model = TinyTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.01)
    for _ in range(steps):
        x, y = make_batch(256, cfg.seq_len, task, device)
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(VOCAB)), y.reshape(-1), ignore_index=PAD_ID)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval()


def dataset(seed, n, max_len, device):
    # Identical inputs across tasks/models; targets are generated separately.
    g = torch.Generator(device=device).manual_seed(seed)
    lengths = torch.randint(2, max_len, (n,), generator=g, device=device)
    x = torch.full((n, max_len), PAD_ID, dtype=torch.long, device=device)
    for i, length in enumerate(lengths.tolist()):
        x[i, :length] = torch.randint(0, 10, (length,), generator=g, device=device)
        x[i, length] = 10
    return x, lengths


def targets(x, lengths, task):
    y = torch.full_like(x, PAD_ID)
    for i, length in enumerate(lengths.tolist()):
        d = x[i, :length]
        if task == "copy": out = d
        elif task == "reverse": out = d.flip(0)
        elif task == "sort": out = d.sort().values
        else: out = torch.roll(d, -1)
        y[i, :length] = out; y[i, length] = 10
    return y


@torch.inference_mode()
def collect(model, x):
    residuals, handles = [], []
    for block in model.blocks:
        handles.append(block.register_forward_hook(lambda _m, _i, o: residuals.append(o[0].detach())))
    logits, internals = model(x, capture=True)
    for h in handles: h.remove()
    mlps = [d["mlp"].detach() for d in internals]
    return logits, residuals, mlps


def eta_squared(a, labels):
    """Per-neuron fraction of activation variance explained by a categorical label."""
    a = np.asarray(a); labels = np.asarray(labels)
    grand = a.mean(0); total = ((a-grand)**2).sum(0) + 1e-12
    between = np.zeros(a.shape[1])
    for value in np.unique(labels):
        group = a[labels == value]
        between += len(group) * (group.mean(0)-grand)**2
    return between / total


def ridge_probe(x, y, groups, seed=0, ridge=1e-2):
    """Sequence-held-out linear least-squares classifier."""
    rng = np.random.default_rng(seed); unique = np.unique(groups); rng.shuffle(unique)
    test_groups = set(unique[:max(1, len(unique)//4)].tolist())
    test = np.array([g in test_groups for g in groups]); train = ~test
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    xt = (x[train]-mean)/std; xv = (x[test]-mean)/std
    classes = np.unique(y); onehot = (y[train, None] == classes[None, :]).astype(float)
    xb = np.c_[xt, np.ones(len(xt))]; xvb = np.c_[xv, np.ones(len(xv))]
    reg = np.eye(xb.shape[1])*ridge; reg[-1,-1] = 0
    w = np.linalg.solve(xb.T@xb + reg, xb.T@onehot)
    pred = classes[(xvb@w).argmax(1)]
    return float((pred == y[test]).mean()), float(max(np.bincount(y[train]))/len(y[train]))


def linear_cka(x, y):
    x=x-x.mean(0); y=y-y.mean(0)
    cross=np.linalg.norm(x.T@y, "fro")**2
    return float(cross / (np.linalg.norm(x.T@x,"fro")*np.linalg.norm(y.T@y,"fro") + 1e-12))


@torch.inference_mode()
def scores(model, x, y):
    p=model(x)[0].argmax(-1); valid=y.ne(PAD_ID)
    tok=((p==y)&valid).sum()/valid.sum()
    exact=(((p==y)|~valid).all(1)).float().mean()
    return {"token": float(tok), "exact": float(exact)}


@torch.inference_mode()
def ablate(model, layer, neurons, x, y):
    neurons=torch.as_tensor(neurons, device=x.device)
    def hook(_m, _i, out):
        out=out.clone(); out[..., neurons]=0; return out
    h=model.blocks[layer].mlp[1].register_forward_hook(hook)
    result=scores(model,x,y); h.remove(); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--seeds",type=int,nargs="+",default=[11,29,47]); p.add_argument("--eval",type=int,default=2048)
    p.add_argument("--output",type=Path,default=Path("research/mlp/results.json")); a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=ModelConfig(seq_len=9,d_model=32,n_heads=4,n_layers=2,d_ff=64)
    x,lengths=dataset(20260711,a.eval,cfg.seq_len,device)
    digit_mask=(x<10); rows=torch.arange(len(x),device=device)[:,None].expand_as(x)[digit_mask].cpu().numpy()
    token=x[digit_mask].cpu().numpy(); pos=torch.arange(cfg.seq_len,device=device)[None,:].expand_as(x)[digit_mask].cpu().numpy()
    lens=lengths[:,None].expand_as(x)[digit_mask].cpu().numpy()
    result={"method":{"architecture":cfg.dict(),"steps":a.steps,"eval_sequences":a.eval,"seeds":a.seeds,
             "probe_split":"25% held-out sequences","device":str(device)},"runs":[],"cka":{}}
    representations={}
    for task in TASKS:
        task_reps=[]
        for seed in a.seeds:
            model=train(task,seed,device,a.steps,cfg); y=targets(x,lengths,task)
            logits,resids,mlps=collect(model,x); base=scores(model,x,y)
            layers=[]
            for li,(r,m) in enumerate(zip(resids,mlps)):
                rv=r[digit_mask].cpu().numpy(); mv=m[digit_mask].cpu().numpy()
                probes={name:ridge_probe(rv,lab,rows,seed) for name,lab in (("digit",token),("position",pos),("length",lens))}
                mlp_probes={name:ridge_probe(mv,lab,rows,seed) for name,lab in (("digit",token),("position",pos),("length",lens))}
                select={name:eta_squared(mv,lab) for name,lab in (("digit",token),("position",pos),("length",lens))}
                # Causal test: ablate the 10% most label-selective neurons; compare 20 matched random sets.
                combined=np.maximum.reduce(list(select.values())); k=max(1,cfg.d_ff//10); top=np.argsort(combined)[-k:]
                top_score=ablate(model,li,top,x,y)
                rng=np.random.default_rng(seed+li); random_scores=[ablate(model,li,rng.choice(cfg.d_ff,k,replace=False),x,y) for _ in range(20)]
                layers.append({"residual_probes":{k:{"accuracy":v[0],"majority":v[1]} for k,v in probes.items()},
                    "mlp_probes":{k:{"accuracy":v[0],"majority":v[1]} for k,v in mlp_probes.items()},
                    "selectivity":{k:{"max_eta2":float(v.max()),"median_eta2":float(np.median(v)),
                    "neurons_eta2_gt_0.2":int((v>.2).sum())} for k,v in select.items()},
                    "ablation":{"fraction":k/cfg.d_ff,"top_selective":top_score,
                    "random_mean":{q:float(np.mean([z[q] for z in random_scores])) for q in ("token","exact")},
                    "random_sd":{q:float(np.std([z[q] for z in random_scores])) for q in ("token","exact")}}})
            result["runs"].append({"task":task,"seed":seed,"baseline":base,"layers":layers})
            task_reps.append(resids[-1][digit_mask].cpu().numpy())
        representations[task]=task_reps
    for t1 in TASKS:
        result["cka"][t1]={t2:float(np.mean([linear_cka(x1,x2) for x1,x2 in zip(representations[t1],representations[t2])])) for t2 in TASKS}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(a.output)

if __name__ == "__main__": main()
