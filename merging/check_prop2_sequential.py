"""Numerical check of the sequential-projection claims in Proposition 2 (app B.2).

For r=8 on ViT-B/16 own checkpoints, over the 53 merged tensors:
  1. pairwise Frobenius inner products between hat_tau_i and hat_tau_j  -> exact zero?
  2. ||sum hat_tau_i||^2  vs  sum ||hat_tau_i||^2                       -> Pythagoras?
  3. ||hat_tau_i|| non-increasing in i                                  -> monotone decay?
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_merge import DATASETS, load_task_vectors, get_merge_scope, top_r_basis, qr_orthonormalize, project_out

cli = argparse.ArgumentParser()
cli.add_argument('--model_location', required=True, help='path to the ViT-B-16 pretrained dir')
cli.add_argument('--rank', type=int, default=8)
args = cli.parse_args()

MODEL = args.model_location
RANK = args.rank

base_sd, tvs = load_task_vectors(MODEL, DATASETS)
keys = get_merge_scope(base_sd, 'jnsp53')
print(f'tensors={len(keys)} rank={RANK}')

# Sequential projections, kept per task so we can inspect them.
seq = {}   # key -> list of 8 projected tensors
for k in keys:
    deltas = [tv[k] for tv in tvs]
    acc, proj = None, []
    for i, d in enumerate(deltas):
        p = project_out(d, acc)
        proj.append(p)
        cat = torch.cat([acc, top_r_basis(d, RANK)], dim=1) if acc is not None else top_r_basis(d, RANK)
        acc = qr_orthonormalize(cat)
    seq[k] = proj

# 1. pairwise inner products, relative to the product of norms
worst = 0.0
max_abs_offdiag = 0.0
for k in keys:
    P = seq[k]
    for i in range(8):
        for j in range(i + 1, 8):
            ip = torch.sum(P[i] * P[j]).item()
            denom = torch.norm(P[i]).item() * torch.norm(P[j]).item()
            cos = ip / denom if denom > 0 else 0.0
            if abs(cos) > abs(max_abs_offdiag):
                max_abs_offdiag = cos
            if abs(ip) > worst:
                worst = abs(ip)
print(f'[1] max |<tau_i,tau_j>_F| over all tensors/pairs = {worst:.6e}')
print(f'    max |cosine| between sequential increments        = {max_abs_offdiag:.6e}')

# 2. Pythagoras: ||sum||^2 vs sum ||.||^2
lhs = sum(torch.sum(torch.stack(seq[k]).sum(0) ** 2).item() for k in keys)
rhs = sum(sum(torch.sum(p ** 2).item() for p in seq[k]) for k in keys)
print(f'[2] ||sum hat_tau||_F^2 = {lhs:.6e}')
print(f'    sum ||hat_tau_i||_F^2 = {rhs:.6e}   ratio lhs/rhs = {lhs / rhs:.6f}')

# 3. monotone norm decay along the list
norms = []
for i in range(8):
    tot = sum(torch.norm(seq[k][i]).item() ** 2 for k in keys) ** 0.5
    norms.append(tot)
raws = []
for i in range(8):
    tot = sum(torch.norm(tvs[i][k]).item() ** 2 for k in keys) ** 0.5
    raws.append(tot)
print('[3] position : ||hat_tau_i||_F : retention ||hat||/||tau||')
for i in range(8):
    print(f'    {i + 1} {DATASETS[i]:>8s} : {norms[i]:.4f} : {norms[i] / raws[i]:.4f}')
mono = all(norms[i] >= norms[i + 1] for i in range(7))
print(f'    monotone non-increasing? {mono}')

retention_mono = [norms[i] / raws[i] for i in range(8)]
print(f'    retention monotone? {all(retention_mono[i] >= retention_mono[i+1] for i in range(7))}')
