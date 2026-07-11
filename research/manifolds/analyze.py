"""Quantitative manifold study for translation and cyclic rotation.

The analysis deliberately distinguishes the *raw* learned token table from
contextual residual-stream states.  It uses held-out random strings and writes
only numerical summaries, so the report does not depend on visual impressions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.model import (LETTER_IDS, MULTITASK_VOCAB, OPERATION_IDS, PAD_ID,
                       ModelConfig, TinyTransformer)  # noqa: E402

DIGITS = torch.tensor([1, 2, 3])
LETTERS = torch.tensor([LETTER_IDS['a'], LETTER_IDS['b'], LETTER_IDS['c']])
EQ = 10


def load(path, device):
    data = torch.load(ROOT / path, map_location=device, weights_only=False)
    model = TinyTransformer(ModelConfig(**data['model_config'])).to(device)
    model.load_state_dict(data['state_dict']); model.eval()
    return model, data


@torch.no_grad()
def states(model, x):
    """Return raw contextual input and residual states, prior to final logits."""
    result = {'context_input': model.token_embedding(x) + model.position_embedding[:, :x.shape[1]]}
    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            lambda _m, _a, value, i=i: result.__setitem__(f'layer_{i + 1}', value[0].detach())))
    logits, _ = model(x)
    for hook in hooks: hook.remove()
    result['final'] = model.final_norm(result[f'layer_{len(model.blocks)}']).detach()
    return logits.detach(), result


def acc(logits, y, active):
    good = logits.argmax(-1)[active].eq(y[active])
    per_row = ((logits.argmax(-1) == y) | ~active).all(1)
    return {'token': good.float().mean().item(), 'exact': per_row.float().mean().item()}


def fit_affine(x_train, y_train, x_test, y_test, ridge=1e-3):
    """Fit x -> y; report held-out R2, cosine and relative error."""
    mu_x, mu_y = x_train.mean(0), y_train.mean(0)
    a, b = x_train - mu_x, y_train - mu_y
    eye = torch.eye(a.shape[1], device=a.device)
    w = torch.linalg.solve(a.T @ a + ridge * eye, a.T @ b)
    pred = (x_test - mu_x) @ w + mu_y
    err = (pred - y_test).square().sum()
    total = (y_test - y_test.mean(0)).square().sum().clamp_min(1e-12)
    cosine = F.cosine_similarity(pred, y_test, dim=-1).mean()
    rel = err.sqrt() / y_test.square().sum().sqrt().clamp_min(1e-12)
    return {'r2': (1 - err / total).item(), 'mean_cosine': cosine.item(),
            'relative_rmse': rel.item(), 'w': w, 'mu_x': mu_x, 'mu_y': mu_y}


def fit_orthogonal(x_train, y_train, x_test, y_test):
    """Rigid centered alignment; suitable for testing a putative group action."""
    mx, my = x_train.mean(0), y_train.mean(0)
    u, _, vh = torch.linalg.svd((x_train-mx).T @ (y_train-my), full_matrices=False)
    q = u @ vh
    pred = (x_test-mx) @ q + my
    return {'mean_cosine': F.cosine_similarity(pred, y_test, dim=-1).mean().item(),
            'relative_rmse': ((pred-y_test).norm()/y_test.norm().clamp_min(1e-12)).item(),
            'q': q, 'mean_source': mx, 'mean_target': my}


def translate_batch(length, n, seed, device):
    g = torch.Generator().manual_seed(seed + length)
    d = torch.randint(1, 4, (n, length), generator=g).to(device)
    l = LETTERS.to(device)[d - 1]
    x = torch.full((2*n, length + 1), EQ, dtype=torch.long, device=device)
    x[:n, :length], x[n:, :length] = d, l
    y = x.clone(); y[:n, :length], y[n:, :length] = l, d
    return x, y, d


def raw_embedding_translation(model):
    table = model.token_embedding.weight.detach()
    d, l = table[DIGITS.to(table.device)], table[LETTERS.to(table.device)]
    cos = F.cosine_similarity(d, l, dim=1)
    pair_dist = (d-l).norm(dim=1)
    all_dist = torch.cdist(d, l)
    # Orthogonal Procrustes is appropriate only as a descriptive 3-point fit.
    dc, lc = d-d.mean(0), l-l.mean(0)
    u, _, vh = torch.linalg.svd(dc.T @ lc, full_matrices=False)
    q = u @ vh
    aligned = dc @ q
    return {'paired_cosine': cos.tolist(), 'mean_paired_cosine': cos.mean().item(),
            'paired_distance': pair_dist.tolist(), 'mean_paired_distance': pair_dist.mean().item(),
            'distance_matrix_digit_rows_letter_columns': all_dist.tolist(),
            'procrustes_three_point_relative_error': ((aligned-lc).norm()/lc.norm()).item(),
            'note': 'The three paired vocabulary vectors are insufficient to establish a general linear map.'}


def translation_study(model, n, seed, device):
    result = {'raw_token_embeddings': raw_embedding_translation(model), 'lengths': {}, 'aggregate': {}}
    cross_train, cross_test = {}, {}
    all_d_to_l, all_l_to_d = {}, {}
    for length in range(2, 9):
        x, y, _ = translate_batch(length, n, seed, device)
        logits, reps = states(model, x)
        active = torch.zeros_like(x, dtype=torch.bool); active[:, :length+1] = True
        # Behavioral involution is tested directly, rather than inferring it
        # from separately fitted (and rank-limited) linear maps.
        first = logits.argmax(-1)
        second, _ = states(model, first)
        row = {'accuracy': acc(logits, y, active),
               'behavioral_involution': acc(second, x, active), 'stages': {}}
        split = torch.arange(n, device=device) < (n * 3 // 4)
        for stage, h in reps.items():
            # Pair exact same abstract sequence in the two alphabets, position aligned.
            hd, hl = h[:n, :length].reshape(n*length, -1), h[n:, :length].reshape(n*length, -1)
            train = split.repeat_interleave(length); test = ~train
            dl = fit_affine(hd[train], hl[train], hd[test], hl[test])
            ld = fit_affine(hl[train], hd[train], hl[test], hd[test])
            row['stages'][stage] = {
                'digit_to_letter': {k: v for k, v in dl.items() if k not in ('w', 'mu_x', 'mu_y')},
                'letter_to_digit': {k: v for k, v in ld.items() if k not in ('w', 'mu_x', 'mu_y')},
                'paired_state_cosine_before_mapping': F.cosine_similarity(hd[test], hl[test], dim=1).mean().item(),
            }
            all_d_to_l.setdefault(stage, []).append((hd, hl, train))
            all_l_to_d.setdefault(stage, []).append((hl, hd, train))
        result['lengths'][str(length)] = row
    # Fit only short strings, evaluate long ones: this isolates length dependence.
    for stage in all_d_to_l:
        train_x = torch.cat([a[t] for a, _b, t in all_d_to_l[stage][:3]])
        train_y = torch.cat([b[t] for _a, b, t in all_d_to_l[stage][:3]])
        test_x = torch.cat([a[~t] for a, _b, t in all_d_to_l[stage][3:]])
        test_y = torch.cat([b[~t] for _a, b, t in all_d_to_l[stage][3:]])
        result['aggregate'][stage] = {
            'short_length_train_to_long_length_test_digit_to_letter': {
                k: v for k, v in fit_affine(train_x, train_y, test_x, test_y).items() if k not in ('w', 'mu_x', 'mu_y')},
        }
    return result


def rotate_batch(length, n, seed, device):
    g = torch.Generator().manual_seed(seed + 100 + length)
    payload = torch.randint(0, 10, (n, length), generator=g).to(device)
    x = torch.full((n, length + 2), PAD_ID, dtype=torch.long, device=device)
    x[:, 0], x[:, 1:length+1], x[:, length+1] = OPERATION_IDS['rotate_left'], payload, EQ
    y = x.clone(); y[:, 1:length+1] = torch.roll(payload, -1, 1)
    return x, y, payload


def rotation_study(model, n, seed, device):
    result = {'lengths': {}, 'aggregate': {}}
    # Compare h(x) against h(Rx), with token-following alignment. Prefix and = are excluded.
    for length in range(3, 9):
        x, y, payload = rotate_batch(length, n, seed, device)
        xr = x.clone(); xr[:, 1:length+1] = torch.roll(payload, -1, 1)
        logits, reps = states(model, x); _, reps_r = states(model, xr)
        active = torch.zeros_like(x, dtype=torch.bool); active[:, 1:length+2] = True
        row = {'accuracy': acc(logits, y, active), 'stages': {}}
        split = torch.arange(n, device=device) < n * 3 // 4
        for stage in reps:
            h, hr = reps[stage][:, 1:length+1], reps_r[stage][:, 1:length+1]
            # After left rotation, token originally at p occurs at p-1.
            followed = torch.roll(hr, shifts=1, dims=1).reshape(n*length, -1)
            same_pos = hr.reshape(n*length, -1)
            source = h.reshape(n*length, -1)
            tr, te = split.repeat_interleave(length), (~split).repeat_interleave(length)
            follow_fit = fit_affine(source[tr], followed[tr], source[te], followed[te])
            spatial_fit = fit_affine(source[tr], same_pos[tr], source[te], same_pos[te])
            # An orthogonal one-step map is a stricter group-action test than an
            # unconstrained affine interpolant (the latter can be unstable under powers).
            ortho = fit_orthogonal(source[tr], followed[tr], source[te], followed[te])
            z = source[te] - ortho['mean_source']
            for _ in range(length): z = z @ ortho['q']
            z = z + ortho['mean_source']
            closure = (z-source[te]).norm() / source[te].norm().clamp_min(1e-12)
            # Offset matrix: mean cosine, using the same sample across positions.
            offset = torch.empty(length, length, device=device)
            for i in range(length):
                for j in range(length): offset[i, j] = F.cosine_similarity(h[:, i], hr[:, j], dim=1).mean()
            # A circulant matrix has constant wrapped diagonals; report residual after diagonal averaging.
            circular = torch.stack([torch.stack([offset[i, (i+k) % length] for i in range(length)]).mean()
                                    for k in range(length)])
            recon = torch.stack([torch.stack([circular[(j-i) % length] for j in range(length)]) for i in range(length)])
            circ_error = (offset-recon).norm()/offset.norm().clamp_min(1e-12)
            # Fourier concentration of the offset kernel: descriptive, not proof of Fourier features.
            power = torch.fft.rfft(circular).abs().square(); power = power / power.sum().clamp_min(1e-12)
            row['stages'][stage] = {
                'token_following_affine': {k: v for k, v in follow_fit.items() if k not in ('w','mu_x','mu_y')},
                'token_following_orthogonal': {k: v for k, v in ortho.items() if k not in ('q','mean_source','mean_target')},
                'same_absolute_position_affine': {k: v for k, v in spatial_fit.items() if k not in ('w','mu_x','mu_y')},
                'one_step_map_orbit_closure_relative_error': closure.item(),
                'position_similarity_circulant_relative_error': circ_error.item(),
                'offset_kernel_fourier_power': power.tolist(),
            }
        result['lengths'][str(length)] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--translate-checkpoint', default='checkpoints/translate-20260711-104508.pt')
    parser.add_argument('--multitask-checkpoint', default='checkpoints/multi_task-20260711-104631.pt')
    parser.add_argument('--output', default='research/manifolds/results.json')
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=31415)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(); device = torch.device(args.device)
    torch.manual_seed(args.seed)
    translation, tdata = load(args.translate_checkpoint, device)
    multi, mdata = load(args.multitask_checkpoint, device)
    output = {'protocol': {'samples_per_length': args.samples, 'seed': args.seed,
                           'translation_lengths': [2,3,4,5,6,7,8], 'rotation_lengths': [3,4,5,6,7,8]},
              'checkpoints': {'translation': args.translate_checkpoint, 'multitask': args.multitask_checkpoint,
                              'translation_config': tdata['model_config'], 'multitask_config': mdata['model_config']},
              'translation': translation_study(translation, args.samples, args.seed, device),
              'cyclic_rotation': rotation_study(multi, args.samples, args.seed, device)}
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__': main()
