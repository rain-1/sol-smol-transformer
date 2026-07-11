"""Train tiny task models and quantify their attention algorithms.

Run: python research/attention/analyze.py --steps 2000 --examples 256
"""
import argparse, json, math, random, sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.model import ModelConfig, PAD_ID, TinyTransformer
from app.training import make_batch

CONFIGS = {
    "copy": dict(d_model=8, n_heads=1, n_layers=1, d_ff=8),
    "reverse": dict(d_model=16, n_heads=2, n_layers=2, d_ff=16),
    "sort": dict(d_model=16, n_heads=1, n_layers=1, d_ff=16),
    "rotate_left": dict(d_model=8, n_heads=1, n_layers=1, d_ff=8),
}

def train(task, device, steps, seed):
    random.seed(seed); torch.manual_seed(seed)
    cfg = ModelConfig(seq_len=9, **CONFIGS[task]); model = TinyTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    lossfn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    model.train()
    for _ in range(steps):
        x,y=make_batch(256,9,task,device); logits,_=model(x)
        loss=lossfn(logits.flatten(0,1),y.flatten()); opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    return model.eval()

@torch.inference_mode()
def evaluate(model, task, device, examples):
    out={"exact_by_length":{},"heads":{}}
    # Fixed length batches avoid padding effects and support clean positional metrics.
    for n in range(2,9):
        digits=torch.randint(0,10,(examples,n),device=device)
        x=torch.cat([digits,torch.full((examples,1),10,device=device,dtype=torch.long)],1)
        if task=="copy": target=digits
        elif task=="reverse": target=digits.flip(1)
        elif task=="sort": target=digits.sort(1).values
        else: target=torch.roll(digits,-1,1)
        y=torch.cat([target,torch.full((examples,1),10,device=device,dtype=torch.long)],1)
        logits,layers=model(x,capture=True); pred=logits.argmax(-1)
        out["exact_by_length"][str(n)]=(pred.eq(y).all(1).float().mean().item())
        for li,layer in enumerate(layers):
            A=layer["attention"] # B,H,Q,K
            for h in range(A.shape[1]):
                key=f"L{li}H{h}"; d=out["heads"].setdefault(key,{"by_length":{}})
                a=A[:,h,:n,:n+1]; q=torch.arange(n,device=device)
                diag=a[:,q,q].mean().item(); anti=a[:,q,n-1-q].mean().item()
                eq=a[:,:,-1].mean().item()
                ent=(-(a.clamp_min(1e-9)*a.clamp_min(1e-9).log()).sum(-1)/math.log(n+1)).mean().item()
                expected=None
                if task=="copy": src=q
                elif task=="reverse": src=n-1-q
                elif task=="rotate_left": src=(q+1)%n
                else: src=None
                if src is not None:
                    mass=a[:,q,src].mean().item(); top=a.argmax(-1).eq(src).float().mean().item()
                    expected={"required_source_mass":mass,"required_source_top1":top}
                # Variation across examples at fixed positions is evidence of content dependence.
                content_sd=a.std(0).mean().item()
                d["by_length"][str(n)]={"diagonal_mass":diag,"anti_diagonal_mass":anti,
                    "equals_mass":eq,"normalized_entropy":ent,"content_sd":content_sd,**(expected or {})}
    for d in out["heads"].values():
        rows=list(d["by_length"].values())
        d["mean"]={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
    return out

@torch.inference_mode()
def exact_score(model, task, device, examples=512):
    scores=[]
    for n in range(2,9):
        # Re-seeding makes every ablation use identical cases.
        g=torch.Generator(device=device).manual_seed(1000+n)
        digits=torch.randint(0,10,(examples,n),device=device,generator=g)
        x=torch.cat([digits,torch.full((examples,1),10,device=device,dtype=torch.long)],1)
        target={"copy":lambda z:z,"reverse":lambda z:z.flip(1),"sort":lambda z:z.sort(1).values,
                "rotate_left":lambda z:torch.roll(z,-1,1)}[task](digits)
        y=torch.cat([target,torch.full((examples,1),10,device=device,dtype=torch.long)],1)
        scores.append(model(x)[0].argmax(-1).eq(y).all(1).float().mean().item())
    return sum(scores)/len(scores)

def ablations(model, task, device):
    """Zero each head's columns in W_O, a causal intervention on that head's output."""
    base=exact_score(model,task,device); result={"baseline_mean_exact":base,"heads":{}}
    for li,block in enumerate(model.blocks):
        width=model.cfg.d_model//model.cfg.n_heads
        for h in range(model.cfg.n_heads):
            sl=slice(h*width,(h+1)*width); saved=block.attn.out_proj.weight[:,sl].detach().clone()
            with torch.no_grad(): block.attn.out_proj.weight[:,sl].zero_()
            score=exact_score(model,task,device)
            with torch.no_grad(): block.attn.out_proj.weight[:,sl].copy_(saved)
            result["heads"][f"L{li}H{h}"]={"ablated_mean_exact":score,"exact_drop":base-score}
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=2000); p.add_argument("--examples",type=int,default=256); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result={"method":{"steps":a.steps,"examples_per_length":a.examples,"seed":a.seed,"device":str(device)},"tasks":{}}
    ck=ROOT/"research/attention/checkpoints"; ck.mkdir(parents=True,exist_ok=True)
    for task in CONFIGS:
        model=train(task,device,a.steps,a.seed); result["tasks"][task]=evaluate(model,task,device,a.examples)
        result["tasks"][task]["causal_ablation"]=ablations(model,task,device)
        torch.save({"model_config":model.cfg.dict(),"state_dict":model.state_dict()},ck/f"{task}.pt")
        print(task,result["tasks"][task]["exact_by_length"])
    path=ROOT/"research/attention/results.json"; path.write_text(json.dumps(result,indent=2)); print(path)
if __name__=="__main__": main()
