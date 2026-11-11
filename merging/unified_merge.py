#!/usr/bin/env python3
"""Unified merging harness: fair comparison of 8 merge methods on ViT-B/16.

Methods:
  - ta: Task Arithmetic (scaling * sum)
  - ties: TIES-Merging (trim + sign election + disjoint mean)
  - dare: DARE (drop + rescale + sum)
  - jnsp: Joint Null-Space Projection (rank r per task)
  - tsvm: TSV-M (SVD compression + polar orthogonalization)
  - iso_c: Iso-C (TA + flatten spectrum to mean)
  - iso_cts: Iso-CTS (common 80% + task-specific subspaces + flatten)
  - esm: ESM/ESD (activation-PCA principal directions + scaling)

Scope:
  - all2d: all 2-D tensors except text_projection (52 tensors)
  - jnsp53: 2-D tensors with min(shape)>=64 (53 tensors, includes text_projection)

Non-2D handling:
  - zero: leave at pretrained (JNSP native)
  - avg: running mean of deltas (TSV/Iso/ESM native)

Usage:
  python unified_merge.py --methods ta ties dare jnsp tsvm iso_c iso_cts esm \\
      --scope all2d --non2d zero --eval --save_merged
  python unified_merge.py --eval_only merged_ta_all2d_zero.pt merged_ties_all2d_zero.pt
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, VISION_ROOT)

from fda_args import parse_arguments
from src.eval import eval_single_dataset
from src.modeling import ImageEncoder

DATASETS = ['MNIST', 'EuroSAT', 'GTSRB', 'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']

# ============================================================================
# Loading and evaluation
# ============================================================================

def load_task_vectors(model_location, datasets):
    """Load pretrained and 8 finetuned checkpoints, return base_sd and list of deltas."""
    pretrained = torch.load(f'{model_location}/zeroshot.pt', map_location='cpu', weights_only=False)
    base_sd = pretrained.state_dict()
    tvs = []
    for ds in datasets:
        ft_path = f'{model_location}/{ds}/finetuned.pt'
        ft = torch.load(ft_path, map_location='cpu', weights_only=False).state_dict()
        tv = {}
        for k in base_sd:
            if base_sd[k].dtype not in (torch.float32, torch.float16, torch.bfloat16):
                continue
            tv[k] = (ft[k].float() - base_sd[k].float())
        tvs.append(tv)
    return base_sd, tvs


def get_merge_scope(base_sd, scope='all2d'):
    """Return list of keys to merge."""
    if scope == 'all2d':
        return [k for k in base_sd if base_sd[k].dim() == 2 and 'text_projection' not in k]
    elif scope == 'jnsp53':
        return [k for k in base_sd if base_sd[k].dim() == 2 and min(base_sd[k].shape) >= 64]
    else:
        raise ValueError(f"Unknown scope: {scope}")


def assemble_state_dict(base_sd, merged_delta, non2d_deltas=None, non2d_mode='zero'):
    """Assemble final state dict from base + merged deltas.

    Args:
        base_sd: pretrained state dict
        merged_delta: dict of merged deltas for scoped keys
        non2d_deltas: list of 8 task deltas (full dicts) for non-2D handling
        non2d_mode: 'zero' (leave at pretrained) or 'avg' (mean of deltas)
    """
    sd = {k: v.clone() for k, v in base_sd.items()}
    # Apply merged deltas
    for k, dv in merged_delta.items():
        sd[k] = (base_sd[k].float() + dv).to(base_sd[k].dtype)
    # Handle non-2D tensors not in merged_delta
    if non2d_mode == 'avg' and non2d_deltas is not None:
        for k in base_sd:
            if k not in merged_delta and base_sd[k].dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Average delta across tasks
                deltas = [tv.get(k, torch.zeros_like(base_sd[k])) for tv in non2d_deltas]
                avg_delta = sum(deltas) / len(deltas)
                sd[k] = (base_sd[k].float() + avg_delta).to(base_sd[k].dtype)
    return sd


@torch.no_grad()
def evaluate_all(model_location, state_dict, args, datasets, preds_dir=None):
    """Evaluate merged model on all datasets, return per-dataset accuracies and average."""
    encoder = torch.load(f'{model_location}/zeroshot.pt', map_location='cpu', weights_only=False)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.model.to(args.device).eval()

    if preds_dir is not None:
        os.makedirs(preds_dir, exist_ok=True)

    scores = {}
    for ds in datasets:
        m = eval_single_dataset(encoder, f'{ds}ValfromTrain', args, ds, None, None,
                                collect_preds=preds_dir is not None)
        scores[ds] = 100.0 * m['top1']
        if preds_dir is not None:
            np.savez(os.path.join(preds_dir, f'{ds}.npz'),
                     preds=m['preds'], labels=m['labels'])
    scores['avg'] = sum(scores[d] for d in datasets) / len(datasets)

    encoder.model.to('cpu')
    del encoder
    torch.cuda.empty_cache()
    return scores


# ============================================================================
# Merge methods
# ============================================================================

def merge_ta(tvs, keys, scaling=1.0):
    """Task Arithmetic: scaled sum of deltas."""
    merged = {}
    for k in keys:
        merged[k] = scaling * sum(tv[k] for tv in tvs)
    return merged


def merge_ties(tvs, keys, density=0.2, scaling=1.0):
    """TIES-Merging: trim -> sign election -> disjoint mean."""
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        trimmed = []
        for d in deltas:
            flat = d.abs().flatten()
            keep_k = max(1, int(density * flat.numel()))
            thresh = torch.topk(flat, keep_k, largest=True).values.min()
            mask = d.abs() >= thresh
            trimmed.append(d * mask)
        stacked = torch.stack(trimmed, dim=0)
        elected_sign = torch.sign(stacked.sum(dim=0))
        elected_sign[elected_sign == 0] = 1.0
        agree = (torch.sign(stacked) == elected_sign.unsqueeze(0)) & (stacked != 0)
        count = agree.sum(dim=0).clamp(min=1)
        summed = (stacked * agree).sum(dim=0)
        merged[k] = scaling * (summed / count)
    return merged


def merge_dare(tvs, keys, drop_rate=0.9, scaling=1.0, seed=0):
    """DARE: drop + rescale + sum."""
    merged = {}
    gen = torch.Generator().manual_seed(seed)
    for k in keys:
        total = None
        for tv in tvs:
            d = tv[k]
            mask = torch.bernoulli(torch.full_like(d, 1.0 - drop_rate), generator=gen)
            d_sparse = d * mask / (1.0 - drop_rate)
            total = d_sparse if total is None else total + d_sparse
        merged[k] = scaling * total
    return merged


def top_r_basis(w2d, rank):
    """Top-r right-singular vectors."""
    u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
    return vh[:rank].T.contiguous()


def project_out(w2d, vbasis):
    """Remove component in span(vbasis) from w2d."""
    if vbasis is None or vbasis.shape[1] == 0:
        return w2d
    return w2d - (w2d @ vbasis) @ vbasis.T


def qr_orthonormalize(m):
    q, _ = torch.linalg.qr(m)
    return q


def merge_jnsp(tvs, keys, rank=16):
    """Joint Null-Space Projection: rank r per task, order-independent."""
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        total = torch.zeros_like(deltas[0])
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            total = total + project_out(d, vb)
        merged[k] = total
    return merged


def _row_norms(d, eps=1e-12):
    return d.float().norm(dim=1, keepdim=True).clamp(min=eps)


def _col_norms(d, eps=1e-12):
    return d.float().norm(dim=0, keepdim=True).clamp(min=eps)


def merge_jnsp_dora(tvs, keys, rank=16, alpha=0.0, mode='row'):
    """DoRA-style magnitude/direction split on top of JNSP.

    DoRA (Liu et al., ICML 2024) writes a weight as a per-channel magnitude
    times a unit direction. Transplanted to merging: JNSP supplies the
    direction, and a separately merged magnitude restores the per-channel norm
    profile that the peer projection removed.

        M   = merge_jnsp(deltas)
        n_M = per-channel norm of M
        n_T = per-channel norm of sum_i d_i   (the naive TA merge)
        n   = (1 - alpha) * n_M + alpha * n_T
        out = M * n / n_M

    alpha=0 reproduces merge_jnsp exactly (checked in the driver). alpha=1
    keeps JNSP directions but takes TA's magnitude profile.

    mode='row' -- per OUTPUT-channel norm, shape [d_out]: the literal DoRA m.
    mode='col' -- per INPUT-channel norm, shape [d_in]: the space the peer
                  basis V_r lives in, i.e. DoRA applied on the A (=V) side.
    """
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        m = torch.zeros_like(deltas[0]).float()
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            m = m + project_out(d, vb).float()
        t = sum(d.float() for d in deltas)
        norm = _row_norms if mode == 'row' else _col_norms
        n_m, n_t = norm(m), norm(t)
        n = (1.0 - alpha) * n_m + alpha * n_t
        merged[k] = (m * (n / n_m)).to(deltas[0].dtype)
    return merged


def merge_jnsp_spec(tvs, keys, rank=16, power=1.0):
    """JNSP with a power-law reshape of each survivor's own singular spectrum.

    JNSP sums the survivors S_i = (I - P_i) d_i. Its external-reweighting
    failures (PCB, norm restoration, jnsp_abs) all reweighted the summed result
    in *coordinate* space. This instead reshapes each survivor along its own
    singular directions before summing:

        S_i = U diag(s) V^T  ->  U diag(s^power) V^T,  normalised so that
        ||S_i||_F is unchanged

    power=1 is exactly merge_jnsp (checked in the driver). power<1 flattens the
    spectrum (peer projection removed mass unevenly from the leading
    directions); power>1 sharpens it. One global scalar, no per-task tuning.
    """
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        total = torch.zeros_like(deltas[0]).float()
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            s_i = project_out(d, vb).float()
            if power != 1.0:
                u, sv, vh = torch.linalg.svd(s_i, full_matrices=False)
                scale = float(s_i.norm()) / max(float((sv ** (2 * power)).sum().sqrt()), 1e-12)
                s_i = (u * (sv ** power * scale)) @ vh
            total = total + s_i
        merged[k] = total.to(deltas[0].dtype)
    return merged


def merge_jnsp_topcut(tvs, keys, rank=16, cut=0.0):
    """JNSP with each survivor's own leading singular direction attenuated.

    S_i = (I - P_i) d_i is the identity content JNSP keeps for task i. Its top
    singular pair is where that content is largest, and the energy diagnostic
    says leading directions are also where peer subspaces overlap most, so they
    are where residual interference hides. This removes within each survivor,
    unlike peer-side deletion which removes across tasks:

        S_i -> S_i - cut * sigma_1 u1 v1^T   (Frobenius norm restored)

    cut=0 is exactly merge_jnsp (checked in the driver). cut=1 deletes the
    leading direction outright. The norm restore keeps the comparison from
    being confounded by the magnitude drop.
    """
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        total = torch.zeros_like(deltas[0]).float()
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            s_i = project_out(d, vb).float()
            if cut != 0.0:
                keep = float(s_i.norm())
                u, sv, vh = torch.linalg.svd(s_i, full_matrices=False)
                s_i = s_i - cut * sv[0] * torch.outer(u[:, 0], vh[0])
                s_i = s_i * (keep / max(float(s_i.norm()), 1e-12))
            total = total + s_i
        merged[k] = total.to(deltas[0].dtype)
    return merged


def merge_jnsp_rgate(tvs, keys, rank=16, gate=0.0):
    """JNSP with a per-coordinate gate driven by how much peer span explains it.

    JNSP deletes peer energy wholesale: 37.8% of the total, but it also drags
    22.7% identity-exclusive energy with it. That loss is not uniform across
    coordinates -- a coordinate lying deep inside the peer span loses nearly all
    of itself, a coordinate orthogonal to it loses nothing but still passes
    through the sum unchanged. This weights each *surviving* coordinate by its
    own residual ratio:

        p_i = P_i d_i,  s_i = d_i - p_i
        ratio_i = |p_i| / (|d_i| + eps)      in [0, 1], per coordinate
        out = sum_i s_i * (1 - gate * ratio_i)

    gate=0 is exactly merge_jnsp (checked in the driver). gate=1 fully drops
    coordinates the peer span explains and leaves untouched coordinates intact.
    Unlike every closed reweighting (global scalar, per-coordinate factor from
    the merged result), this factor is computed *before* summing and is defined
    on JNSP's own deletion, not on an external statistic.
    """
    eps = 1e-12
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        total = torch.zeros_like(deltas[0]).float()
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            df = d.float()
            if vb.shape[1] == 0:
                total = total + df
                continue
            proj = (df @ vb) @ vb.T
            s_i = df - proj
            if gate != 0.0:
                ratio = proj.abs() / (df.abs() + eps)
                s_i = s_i * (1.0 - gate * torch.clamp(ratio, 0.0, 1.0))
            total = total + s_i
        merged[k] = total.to(deltas[0].dtype)
    return merged


def merge_jnsp_align(tvs, keys, rank=16, keep=None, mode='align', seed=0):
    """JNSP with per-pair, alignment-ranked peer basis selection.

    JNSP projects out the union of every peer's full top-r space, 8*16 = 128
    columns per tensor. It treats all peer directions as equally worth deleting.
    This keeps only `keep` columns from each peer, chosen by how much they
    actually explain task i's delta:

        score_ij(c) = |v_ij[:, c]^T d_i|   for column c of peer j's basis
        B_ij = [v_ij[:, c] for the `keep` largest scores]

    mode='random' draws the same number of columns uniformly instead, so any
    gain over it is selection, not budget (the lesson of the pairspan control).
    keep=rank (default) reproduces merge_jnsp exactly: every column is kept and
    QR over the concatenation is the same space.
    """
    keep = rank if keep is None else keep
    merged = {}
    g = torch.Generator().manual_seed(seed)
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        n = len(deltas)
        total = torch.zeros_like(deltas[0]).float()
        for i, d in enumerate(deltas):
            sel = []
            for j in range(n):
                if j == i:
                    continue
                bj = bases[j].float()
                if keep >= bj.shape[1]:
                    sel.append(bj)
                elif mode == 'align':
                    # bj is [d_in, r]: column c explains d through d @ v_c,
                    # so its alignment is the norm of that output vector.
                    score = (d.float() @ bj).norm(dim=0)
                    idx = torch.topk(score, keep).indices
                    sel.append(bj[:, idx])
                elif mode == 'energy':
                    # control: keep peer j's own top-`keep` singular directions.
                    sel.append(bases[j][:, :keep].float())
                else:
                    idx = torch.randperm(bj.shape[1], generator=g)[:keep]
                    sel.append(bj[:, idx])
            vb = qr_orthonormalize(torch.cat(sel, dim=1))
            total = total + project_out(d, vb).float()
        merged[k] = total.to(deltas[0].dtype)
    return merged


def merge_jnsp_iter(tvs, keys, rank=16, iters=1):
    """JNSP iterated to a fixed point on the survivor basis.

    Round 1 is exactly merge_jnsp (bases from the raw deltas). Each later round
    rebuilds every task's peer basis from the *survivors* of the previous
    round, then re-projects the raw delta against that cleaner basis:

        B_i^(t) = top_r(S_i^(t-1)),  S_i^(t) = project_out(d_i, qr([B_j^(t)]_{j!=i}))

    The hope is that a survivor's top-r space is less contaminated by peer
    interference than a raw delta's, so round 2 deletes less identity energy.
    iters=1 reproduces merge_jnsp bit-for-bit (checked in the driver).
    """
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        n = len(deltas)
        survivors = deltas
        for _ in range(iters):
            bases = [top_r_basis(s, rank) for s in survivors]
            new = []
            for i, d in enumerate(deltas):
                others = [bases[j] for j in range(n) if j != i]
                vb = qr_orthonormalize(torch.cat(others, dim=1))
                new.append(project_out(d, vb))
            survivors = new
        merged[k] = sum(survivors)
    return merged



    """Functional-overlap merger (exploratory; not reported in the paper).

        Delta_final = sum_i (I - P_i) Delta_i   +   sum_i beta_i * P_i Delta_i

    The first term is exactly JNSP (P_i projects onto the union of the OTHER
    seven tasks' top-r right-singular bases, so (I - P_i) Delta_i is what JNSP
    keeps). The second term is the contested part JNSP throws away, re-added
    with a strength that depends on closed-form statistics only.

    beta_mode:
      'cos'   -- per (tensor, task) cosine of the contested component with the
                 other tasks' combined delta projected into the same span:
                     beta_i = clip(<P_i d_i, P_i S_{-i}> /
                                   (||P_i d_i|| ||P_i S_{-i}||), 0, 1)
                 where S_{-i} = sum_{j != i} d_j. Alignment with what the other
                 tasks jointly want is a shared-beneficial signal; anti-
                 alignment (negative cosine) is clipped to 0, i.e. treated as
                 conflicting and dropped exactly as JNSP does.
      'const' -- beta_i = beta_const for every (tensor, task). Control showing
                 the result is not a product of per-cell tampering.
      'none'  -- beta_i = 0 for all (degenerates to JNSP exactly; sanity check).

    No per-task hand tuning: 'cos' has no free parameter, 'const' is one global
    scalar identical for every task and tensor.
    """
    merged = {}
    betas = []
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        total = torch.zeros_like(deltas[0])
        s_all = sum(deltas)
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            pd = (d @ vb) @ vb.T                       # contested part P_i d_i
            kept = d - pd
            if beta_mode == 'none':
                beta = 0.0
            elif beta_mode == 'const':
                beta = beta_const
            else:
                s_mi = s_all - d
                ps = (s_mi @ vb) @ vb.T
                denom = float(pd.norm()) * float(ps.norm())
                beta = float((pd * ps).sum()) / denom if denom > 0 else 0.0
                beta = min(max(beta, 0.0), 1.0)
            betas.append(beta)
            total = total + kept + beta * pd
        merged[k] = total
    print(f'[merge] fo beta_mode={beta_mode} mean beta={sum(betas) / len(betas):.4f} '
          f'min={min(betas):.4f} max={max(betas):.4f}')
    return merged


@torch.no_grad()
def build_func_bases(tvs, datasets, model_location, data_location, device, rank=16,
                     proxy_n=64, seed=1, proxy='train'):
    """Per task/layer INPUT-activation subspace of the finetuned backbone.

    For a linear layer y = W x, JNSP's basis lives in the input (row) space of
    dW. To split the same dW by function rather than by weight energy we need a
    subspace in that SAME input space, so this PCA's the layer inputs x under
    task i's own finetuned model -- not the output displacement Delta O = X dW^T
    that ESM/`build_esd` uses (that one lives in the output space and is only
    comparable to dW's left singular vectors).

    Only hooked layers get an activation basis (attn.in_proj, mlp.c_fc,
    mlp.c_proj -> 36 of the 52 `all2d` tensors). Everything else is filled in by
    the caller with the SVD basis.
    """
    from src.datasets.registry import get_dataset

    pretrained = torch.load(f'{model_location}/zeroshot.pt', map_location='cpu', weights_only=False)
    preprocess = pretrained.val_preprocess
    del pretrained

    bases = {}
    for ti, ds in enumerate(datasets):
        print(f"[func] {ds}")
        ft = torch.load(f'{model_location}/{ds}/finetuned.pt', map_location='cpu', weights_only=False)
        ft.model.to(device).eval()

        captured = {}
        handles = []

        def make_hook(module_name):
            def hook(module, inp, out):
                x = inp[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                captured[module_name] = x.float().cpu()
            return hook

        for name, module in ft.model.named_modules():
            if name.endswith('.attn'):
                handles.append(module.register_forward_hook(make_hook(f'model.{name}.in_proj_weight')))
            elif name.endswith('.mlp.c_fc') or name.endswith('.mlp.c_proj'):
                handles.append(module.register_forward_hook(make_hook(f'model.{name}.weight')))

        dataset = get_dataset(ds, preprocess, data_location, batch_size=256)
        src_ds = dataset.test_dataset if proxy == 'test' else dataset.train_dataset
        n_avail = len(src_ds)
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_avail, size=min(proxy_n, n_avail), replace=False)
        x = torch.stack([src_ds[i][0] for i in indices]).to(device)

        _ = ft(x)
        for h in handles:
            h.remove()
        ft.model.to('cpu')
        del ft
        torch.cuda.empty_cache()

        task_bases = {}
        for name, acts in captured.items():
            if name not in tvs[ti]:
                continue
            in_dim = tvs[ti][name].shape[1]
            if acts.shape[1] != in_dim:
                continue
            acts = acts - acts.mean(0, keepdim=True)
            _, _, vh = torch.linalg.svd(acts, full_matrices=False)
            k = min(rank, vh.shape[0])
            task_bases[name] = vh[:k].T.contiguous()
        bases[ds] = task_bases
        del captured
    return bases


def merge_ff(tvs, keys, func_bases, datasets, rank=16, beta_mode='none', beta_const=0.5):
    """Function-partitioned merger (exploratory; not reported in the paper).

        Delta_final = sum_i (I - P_i) Delta_i  +  sum_i beta_i P_i Delta_i

    where P_i now projects onto the union of the OTHER tasks' FUNCTIONAL bases
    (top-r input-activation PCA of each of the other seven finetuned models)
    rather than onto the union of their weight SVD bases. Same algebra and same
    budget as JNSP (7 x 16 removed directions per layer); the only thing that
    changes is which subspace defines "contested".

    Layers without an activation basis fall back to the SVD basis, so the JNSP
    arithmetic is preserved there and any delta comes only from the 36 hooked
    layers.

    beta_mode: 'none' (beta=0, pure deletion), 'const', or 'cos'
    (closed-form alignment of the contested block with the other tasks'
    contested blocks, clipped to [0,1] -- no per-task tuning).
    """
    fallback = set(tvs[0].keys()) - set(func_bases[datasets[0]].keys())
    T = len(datasets)
    merged = {}
    betas = []
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        if k in fallback:
            bases = [top_r_basis(d, rank) for d in deltas]
        else:
            bases = [func_bases[datasets[i]][k] for i in range(T)]
        total = torch.zeros_like(deltas[0])
        s_all = sum(deltas)
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(T) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            pd = (d @ vb) @ vb.T
            kept = d - pd
            if beta_mode == 'none':
                beta = 0.0
            elif beta_mode == 'const':
                beta = beta_const
            else:
                s_mi = s_all - d
                ps = (s_mi @ vb) @ vb.T
                denom = float(pd.norm()) * float(ps.norm())
                beta = float((pd * ps).sum()) / denom if denom > 0 else 0.0
                beta = min(max(beta, 0.0), 1.0)
            betas.append(beta)
            total = total + kept + beta * pd
        merged[k] = total
    print(f'[merge] ff beta_mode={beta_mode} mean beta={sum(betas) / len(betas):.4f} '
          f'svd_fallback_layers={len([k for k in keys if k in fallback])}')
    return merged


def merge_jnsp_sequential(tvs, keys, rank=16, grow_basis=True):
    """NUFILT-style sequential null-space filtering (D5 control).

    Same operator as JNSP, I - V V^T, but applied cumulatively in merge order:
    task i is projected against the basis accumulated from tasks 0..i-1.
    grow_basis=True  -> basis grows by <=rank per step (true NUFILT behaviour)
    grow_basis=False -> basis stays at rank r (truncated variant)
    Order-dependent by construction; tvs order = DATASETS order.
    """
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        total = torch.zeros_like(deltas[0])
        acc = None
        for i, d in enumerate(deltas):
            total = total + project_out(d, acc)
            cat = torch.cat([acc, bases[i]], dim=1) if acc is not None else bases[i]
            acc = qr_orthonormalize(cat)
            if not grow_basis and acc.shape[1] > rank:
                acc = acc[:, :rank].contiguous()
        merged[k] = total
    return merged


def merge_jnsp_global(tvs, keys, rank=16):
    """Shared-basis control: one rank-r basis of the summed delta, removed from all."""
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        vb = top_r_basis(sum(deltas), rank)
        merged[k] = sum(project_out(d, vb) for d in deltas)
    return merged


def merge_ta_normmatched(tvs, keys, target, per_tensor=True):
    """TA rescaled to match the norm of `target` (JNSP).

    per_tensor: rescale each (key, task) delta so ||scaled delta|| == ||target delta||
                for that key, then sum. Isolates *direction* from *magnitude*.
    """
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        tgt_norm = torch.norm(target[k])
        if per_tensor:
            raw = torch.stack(deltas).sum(dim=0)
            raw_norm = torch.norm(raw)
            scale = (tgt_norm / raw_norm) if raw_norm > 0 else torch.tensor(1.0)
            merged[k] = raw * scale
        else:
            out = torch.zeros_like(deltas[0])
            for d in deltas:
                n = torch.norm(d)
                s = (tgt_norm / n) if n > 0 else torch.tensor(1.0)
                out = out + d * s
            merged[k] = out
    return merged


def merge_random_proj(tvs, keys, rank=16, seed=0):
    """Same machinery as JNSP but the per-task subspaces are random, not SVD-based."""
    gen = torch.Generator().manual_seed(seed)
    merged = {}
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = []
        for d in deltas:
            g = torch.randn(d.shape[1], min(rank, d.shape[1]), generator=gen)
            q, _ = torch.linalg.qr(g)
            bases.append(q.contiguous())
        total = torch.zeros_like(deltas[0])
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            total = total + project_out(d, vb)
        merged[k] = total
    return merged


def merge_ta_normalized(tvs, keys, ref_scaling=0.2):
    """Per-task global normalisation, then rescaled to TA's global norm.

    Each task's delta is divided by its own Frobenius norm over the whole merge
    scope, the normalised deltas are summed, and the sum is rescaled so its
    global norm equals that of TA at `ref_scaling`. Isolates "task vectors just
    have unequal norms" from "the merge removes interfering directions".
    """
    task_norms = []
    for tv in tvs:
        s = sum(float((tv[k] ** 2).sum()) for k in keys)
        task_norms.append(s ** 0.5)

    merged = {}
    for k in keys:
        out = torch.zeros_like(tvs[0][k])
        for ti, tv in enumerate(tvs):
            if task_norms[ti] > 0:
                out = out + tv[k] / task_norms[ti]
        merged[k] = out

    ta = merge_ta(tvs, keys, scaling=ref_scaling)
    m = sum(float((merged[k] ** 2).sum()) for k in keys) ** 0.5
    t = sum(float((ta[k] ** 2).sum()) for k in keys) ** 0.5
    scale = (t / m) if m > 0 else 1.0
    print(f"[ta_normalized] normalised norm={m:.4f} TA(ref={ref_scaling}) norm={t:.4f} "
          f"scale={scale:.4f}; per-task norms={[round(x, 3) for x in task_norms]}")
    return {k: v * scale for k, v in merged.items()}


def merge_jnsp_scaled(tvs, keys, rank=16, ref_scaling=0.2):
    """JNSP output rescaled to TA's global norm.

    If accuracy survives the rescale, JNSP's gain is directional, not a
    magnitude/shrinkage effect.
    """
    merged = merge_jnsp(tvs, keys, rank=rank)
    ta = merge_ta(tvs, keys, scaling=ref_scaling)
    m = sum(float((merged[k] ** 2).sum()) for k in keys) ** 0.5
    t = sum(float((ta[k] ** 2).sum()) for k in keys) ** 0.5
    scale = (t / m) if m > 0 else 1.0
    print(f"[jnsp_scaled] JNSP norm={m:.4f} TA(ref={ref_scaling}) norm={t:.4f} "
          f"scale={scale:.4f}")
    return {k: v * scale for k, v in merged.items()}


def apply_jnsp(tvs, keys, rank=16):
    """Return the JNSP-projected task vectors instead of their sum.

    Same operator as merge_jnsp, but keeps the per-task outputs so a second
    merger can be applied on top of them.
    """
    projected = [{} for _ in tvs]
    for k in keys:
        deltas = [tv[k] for tv in tvs]
        bases = [top_r_basis(d, rank) for d in deltas]
        for i, d in enumerate(deltas):
            others = [bases[j] for j in range(len(deltas)) if j != i]
            vb = qr_orthonormalize(torch.cat(others, dim=1))
            projected[i][k] = project_out(d, vb)
    return projected


def _wudi_solve(vectors, rel_ridge=1e-6):
    """Closed form of sum_i ||(m - t_i) t_i^T||^2 / ||t_i||^2.

    Normal equations m M = B with M = sum_i t_i^T t_i / ||t_i||^2 come from an
    eigendecomposition of M; directions with negligible mass are dropped rather
    than amplified by the ridge, so the returned m is the minimum-norm solution.
    """
    n = vectors.shape[0]
    norms = vectors.reshape(n, -1).pow(2).sum(dim=1).clamp_min(1e-30)
    w = (1.0 / norms).to(vectors.dtype)

    M = torch.zeros(vectors.shape[2], vectors.shape[2], dtype=vectors.dtype)
    for i in range(n):
        M += w[i] * (vectors[i].t() @ vectors[i])
    M = 0.5 * (M + M.t())

    lam, V = torch.linalg.eigh(M)
    lam_max = float(lam.max().clamp_min(1e-30))
    keep = lam > rel_ridge * lam_max
    lam_inv = torch.zeros_like(lam)
    lam_inv[keep] = 1.0 / lam[keep]

    B = torch.zeros_like(vectors[0])
    for i in range(n):
        B += w[i] * (vectors[i] @ (vectors[i].t() @ vectors[i]))
    return (B @ V * lam_inv.unsqueeze(0)) @ V.t()


def merge_wudi(tvs, keys, rel_ridge=1e-6):
    """WUDI (Weight Update Direction Interference): closed-form least-squares merge."""
    return {k: _wudi_solve(torch.stack([tv[k] for tv in tvs]), rel_ridge)
            for k in keys}


def merge_jnsp_wudi(tvs, keys, rank=16, rel_ridge=1e-6):
    """JNSP projection, then WUDI on the projected task vectors.

    Holds the second-stage merger fixed at WUDI so the comparison isolates the
    projection design (right-side full union vs. AlphaMerging's left-side top-k).
    """
    projected = apply_jnsp(tvs, keys, rank=rank)
    return {k: _wudi_solve(torch.stack([p[k] for p in projected]), rel_ridge)
            for k in keys}


def _top_l_basis(w2d, rank):
    """Top-r left-singular vectors (output directions), as AlphaMerging uses."""
    u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
    return u[:, :rank].contiguous()


def apply_alpha(tvs, keys, rank=16, k=16):
    """AlphaMerging Algorithm 1: project out the compressed peer output-space.

    For task i: B_{S,-i} = concat_j!=i U_j^{1:r1} (left singular vectors), then
    SVD(B) and keep the top-k left directions U_S^{1:k}; tau_i <- (I - U_S U_S^T) tau_i.
    """
    projected = [{} for _ in tvs]
    n = len(tvs)
    for key in keys:
        deltas = [tv[key] for tv in tvs]
        lb = [_top_l_basis(d, rank) for d in deltas]
        for i, d in enumerate(deltas):
            peers = torch.cat([lb[j] for j in range(n) if j != i], dim=1)
            kk = min(k, peers.shape[1])
            us = _top_l_basis(peers, kk)
            projected[i][key] = d - us @ (us.t() @ d)
    return projected


def merge_alpha(tvs, keys, rank=16, k=16):
    """AlphaMerging projection followed by plain summation."""
    projected = apply_alpha(tvs, keys, rank=rank, k=k)
    return {key: sum(p[key] for p in projected) for key in keys}


def merge_alpha_wudi(tvs, keys, rank=16, k=16, rel_ridge=1e-6):
    """AlphaMerging projection, then WUDI on the projected task vectors."""
    projected = apply_alpha(tvs, keys, rank=rank, k=k)
    return {key: _wudi_solve(torch.stack([p[key] for p in projected]), rel_ridge)
            for key in keys}


def merge_jnsp_wudi_free(tvs, keys, rank=16, rel_ridge=1e-6, verbose=True):
    """JNSP-projected WUDI, with the objective's null space given back.

    Losing the peer top-r row spaces is what JNSP does, which makes the union of
    every task's own top-r directions a null space of WUDI's M: the minimum-norm
    solve zeroes exactly each task's highest-energy identity directions. Those
    directions are free in WUDI's objective, so the JNSP sum's component there is
    the natural (and only non-arbitrary) completion.
    """
    projected = apply_jnsp(tvs, keys, rank=rank)
    merged = {}
    free_frac = []
    norm_ratio = []
    for key in keys:
        stack = torch.stack([p[key] for p in projected])
        m = _wudi_solve(stack, rel_ridge)
        ssum = stack.sum(0)
        ub = qr_orthonormalize(torch.cat([top_r_basis(s, rank) for s in stack], dim=1))
        free = ssum - (ssum @ ub) @ ub.t()
        merged[key] = m + free
        free_frac.append(float(free.pow(2).sum() / ssum.pow(2).sum().clamp_min(1e-30)))
        norm_ratio.append(float(m.pow(2).sum() / ssum.pow(2).sum().clamp_min(1e-30)))
    if verbose:
        import statistics
        print(f'[jnsp_wudi_free] nullspace={statistics.mean(free_frac):.3f} '
              f'wudi_norm_frac={statistics.mean(norm_ratio):.3f} '
              f'(energy-weighted mean over {len(keys)} tensors)', flush=True)
    return merged


def merge_jnsp_wudi_norm(tvs, keys, rank=16, rel_ridge=1e-6):
    """JNSP-projected WUDI rescaled to the JNSP sum's norm, per tensor.

    Control for merge_jnsp_wudi_free: if restoring only the magnitude recovers
    most of the accuracy, the collapse is a scale artefact of the minimum-norm
    solve; if it does not, the lost identity directions are the cause.
    """
    projected = apply_jnsp(tvs, keys, rank=rank)
    merged = {}
    for key in keys:
        stack = torch.stack([p[key] for p in projected])
        m = _wudi_solve(stack, rel_ridge)
        target = stack.sum(0).norm()
        merged[key] = m * (target / m.norm().clamp_min(1e-30))
    return merged


def merge_tsvm(tvs, keys, device='cpu'):
    """TSV-M: SVD compression (r = min(d)/T) + polar orthogonalization."""
    sv_reduction = 1 / len(tvs)
    merged = {}
    for k in keys:
        deltas = [tv[k].to(device) for tv in tvs]
        u, s, v = torch.linalg.svd(deltas[0], full_matrices=False)
        sum_u = torch.zeros_like(u, device=device)
        sum_s = torch.zeros_like(s, device=device)
        sum_v = torch.zeros_like(v, device=device)

        for i, d in enumerate(deltas):
            u, s, v = torch.linalg.svd(d, full_matrices=False)
            r = int(s.shape[0] * sv_reduction)
            sum_u[:, i*r:(i+1)*r] = u[:, :r]
            sum_s[i*r:(i+1)*r] = s[:r]
            sum_v[i*r:(i+1)*r, :] = v[:r, :]

        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        merged[k] = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).cpu()
    return merged


def merge_iso_c(tvs, keys, device='cpu'):
    """Iso-C: TA + flatten spectrum to mean."""
    merged = {}
    for k in keys:
        deltas = [tv[k].to(device) for tv in tvs]
        combined = sum(deltas)
        U, S, V = torch.linalg.svd(combined, full_matrices=False)
        S_mean = torch.ones_like(S) * S.mean()
        merged[k] = torch.linalg.multi_dot((U, torch.diag(S_mean), V)).cpu()
    return merged


def merge_iso_cts(tvs, keys, common_frac=0.8, device='cpu'):
    """Iso-CTS: common 80% + task-specific subspaces + flatten spectrum.

    One RoBERTa tensor (layer.1.intermediate.dense, 4096x1024) has singular values
    degenerate enough that LAPACK's float32 gesdd refuses to converge. Retry that
    tensor in float64; results are unchanged where float32 already converged.
    """
    simple, hard = [], []
    for k in keys:
        try:
            simple.append((k, _merge_iso_cts_one([tv[k].to(device) for tv in tvs],
                                                 common_frac)))
        except RuntimeError as e:
            if 'failed to converge' not in str(e) and '_LinAlgError' not in type(e).__name__:
                raise
            hard.append(k)

    merged = {k: v.cpu() for k, v in simple}
    for k in hard:
        deltas = [tv[k].to(device).double() for tv in tvs]
        merged[k] = _merge_iso_cts_one(deltas, common_frac).float().cpu()
    return merged


def _merge_iso_cts_one(deltas, common_frac=0.8):
    """Body of Iso-CTS for one tensor; deltas is a list of pre-moved tensors."""
    shape = deltas[0].shape
    combined = sum(deltas)

    # Common space
    min_d = min(shape)
    common_idx = int(min_d * common_frac)
    task_total = round((min_d - common_idx) / len(deltas)) * len(deltas)
    common_idx = min_d - task_total

    u, s, v = torch.linalg.svd(combined, full_matrices=False)
    common_u = u[:, :common_idx]
    common_s = s[:common_idx]
    common_v = v[:common_idx, :]

    # Task-specific spaces
    n_dims_per_task = int((min_d - common_idx) / len(deltas))
    combined_u = torch.zeros_like(u)
    combined_s = torch.zeros_like(s)
    combined_v = torch.zeros_like(v)

    for i, d in enumerate(deltas):
        # Project out common space
        w_ts = d - common_u @ common_u.T @ d
        u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)
        combined_u[:, i*n_dims_per_task:(i+1)*n_dims_per_task] = u_ts[:, :n_dims_per_task]
        combined_s[i*n_dims_per_task:(i+1)*n_dims_per_task] = s_ts[:n_dims_per_task]
        combined_v[i*n_dims_per_task:(i+1)*n_dims_per_task, :] = v_ts[:n_dims_per_task, :]

    # Append common space
    offset = len(deltas) * n_dims_per_task
    combined_u[:, offset:offset+common_idx] = common_u
    combined_s[offset:offset+common_idx] = common_s
    combined_v[offset:offset+common_idx, :] = common_v

    # Orthogonalize and flatten spectrum
    u_u, s_u, v_u = torch.linalg.svd(combined_u, full_matrices=False)
    u_v, s_v, v_v = torch.linalg.svd(combined_v, full_matrices=False)
    combined_u = u_u @ v_u
    combined_v = u_v @ v_v
    combined_s = torch.ones_like(combined_s) * combined_s.mean()

    return torch.linalg.multi_dot((combined_u, torch.diag(combined_s), combined_v))


ESD_ATTRS = ('attn.in_proj_weight', 'attn.out_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')


def _compute_esd(act_data):
    """PCA on activation displacement. Returns eigenvalues, eigenvectors (desc)."""
    feature_dim = act_data.shape[-1]
    if act_data.dim() > 2:
        act_data = act_data.reshape(-1, feature_dim)
    n = act_data.shape[0]
    _, S, Vh = torch.linalg.svd(act_data, full_matrices=False)
    eigenvalues = S ** 2 / (n - 1)
    eigenvectors = Vh.T
    if eigenvectors.shape[1] < feature_dim:
        full_eig = torch.eye(feature_dim, device=act_data.device, dtype=act_data.dtype)
        full_eig[:, :eigenvectors.shape[1]] = eigenvectors
        eigenvectors = full_eig
        full_vals = torch.zeros(feature_dim, device=eigenvalues.device, dtype=eigenvalues.dtype)
        full_vals[:len(eigenvalues)] = eigenvalues
        eigenvalues = full_vals
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order], eigenvectors[:, order]


@torch.no_grad()
def build_esd(tvs, datasets, model_location, data_location, device, proxy_n=32, seed=1, proxy='test'):
    """Per task/layer principal directions of ΔO = X ΔWᵀ (X from the finetuned model).

    Mirrors esm_esd.esd: hooks on every `.mlp` and `.attn` module, one forward pass
    over `proxy_n` images drawn with RandomState(seed=1). `proxy='test'` reproduces
    the upstream setting (test-set leakage); `proxy='train'` is the disjoint control.
    """
    from src.datasets.registry import get_dataset

    pretrained = torch.load(f'{model_location}/zeroshot.pt', map_location='cpu', weights_only=False)
    preprocess = pretrained.val_preprocess
    del pretrained

    esd = {}
    for ti, ds in enumerate(datasets):
        print(f"[ESD] {ds}")
        ft = torch.load(f'{model_location}/{ds}/finetuned.pt', map_location='cpu', weights_only=False)
        ft.model.to(device).eval()

        captured = {}
        handles = []

        def make_attn_hook(module_name):
            def hook(module, inp, out):
                x = inp[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                dW = tvs[ti][f'{module_name}.in_proj_weight'].to(device)
                captured[f'{module_name}.in_proj_weight'] = x @ dW.T
            return hook

        def make_mlp_hook(module_name):
            def hook(module, inp, out):
                x = inp[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                dW = tvs[ti][f'{module_name}.weight'].to(device)
                captured[f'{module_name}.weight'] = x @ dW.T
            return hook

        for name, module in ft.model.named_modules():
            if name.endswith('.attn'):
                handles.append(module.register_forward_hook(make_attn_hook('model.' + name)))
            elif name.endswith('.mlp.c_fc') or name.endswith('.mlp.c_proj'):
                handles.append(module.register_forward_hook(make_mlp_hook('model.' + name)))

        dataset = get_dataset(ds, preprocess, data_location, batch_size=256)
        src_ds = dataset.test_dataset if proxy == 'test' else dataset.train_dataset
        n_avail = len(src_ds)
        n_samples = min(proxy_n, n_avail)
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_avail, size=n_samples, replace=False)
        x = torch.stack([src_ds[i][0] for i in indices]).to(device)

        _ = ft(x)
        for h in handles:
            h.remove()
        ft.model.to('cpu')
        del ft
        torch.cuda.empty_cache()

        task_esd = {}
        for key, act_diff in captured.items():
            eigvals, eigvecs = _compute_esd(act_diff.float())
            r = int(tvs[ti][key].shape[0] / len(datasets))
            task_esd[key] = {'eigenvalues': eigvals[:r].cpu(), 'eigenvectors': eigvecs[:, :r].cpu()}
        esd[ds] = task_esd
        del captured
    return esd


@torch.no_grad()
def merge_esm(tvs, keys, esd, datasets, device='cpu'):
    """ESM merge, element-for-element per esm_utils.esm_merge."""
    T = len(tvs)
    sv_reduction = 1 / T
    norm_power = 2
    principal_keys = set(esd[datasets[0]].keys())
    merged = {}

    for key in keys:
        sum_u, sum_s, sum_v, norms_v = [], [], [], []
        is_principal = key in principal_keys
        for i in range(T):
            vec = tvs[i][key].to(device)
            if is_principal:
                P = esd[datasets[i]][key]['eigenvectors'].to(device)
                reduced = int(vec.shape[0] * sv_reduction) if not key.endswith('.proj') \
                    else int(vec.shape[1] * sv_reduction)
                u = P
                v = P.T @ vec
                norm_v = torch.norm(v)
                norms_v.append(norm_v)
                v = v * pow(norm_v, norm_power)
                lam = esd[datasets[i]][key]['eigenvalues'].to(device)[:reduced]
                scale = torch.pow(lam / lam.mean(), 0.15)
                scale = scale / scale.mean()
                sum_u.append(u[:, :reduced] * scale.unsqueeze(0))
                sum_v.append(v[:reduced, :] * scale.unsqueeze(1))
            else:
                u, s, v = torch.linalg.svd(vec, full_matrices=False)
                reduced = int(s.shape[0] * sv_reduction)
                sum_u.append(u[:, :reduced])
                sum_s.append(s[:reduced])
                sum_v.append(v[:reduced, :])

        if is_principal:
            sum_u = torch.cat(sum_u, dim=1)
            sum_v = torch.cat(sum_v, dim=0)
            norms_v_mean = torch.tensor([float(n) for n in norms_v], device=device).mean()
            sum_v = sum_v / pow(norms_v_mean, norm_power)
            col_norms = torch.norm(sum_v, dim=0)
            col_scales = (col_norms / col_norms.mean()).unsqueeze(0)
            sum_v = sum_v * pow(col_scales, norm_power)
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            U_ortho = u_u @ v_u * s_u.mean()
            V_ortho = u_v @ v_v * s_v.mean()
            merged[key] = (U_ortho @ V_ortho * 2).cpu()
        else:
            sum_u = torch.cat(sum_u, dim=1)
            sum_s = torch.cat(sum_s, dim=0)
            sum_v = torch.cat(sum_v, dim=0)
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            merged[key] = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).cpu()

    # inter-layer scaling over the four block families
    fam_means = {}
    for attr in ESD_ATTRS:
        norms = [torch.norm(merged[k]) for k in merged if attr in k]
        if norms:
            fam_means[attr] = torch.stack(norms).mean()
    for k in list(merged.keys()):
        for attr, mu in fam_means.items():
            if attr in k:
                ratio = torch.norm(merged[k]) / mu
                merged[k] = merged[k] * pow(ratio, norm_power)
    return merged


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=['ta', 'ties', 'dare', 'jnsp'],
                    choices=['ta', 'ties', 'dare', 'jnsp', 'tsvm', 'iso_c', 'iso_cts', 'esm',
                             'ta_norm', 'ta_tasknorm', 'ta_normalized', 'jnsp_scaled', 'rand_proj',
                             'jnsp_seq', 'jnsp_seq_trunc', 'jnsp_global',
                             'fo', 'fo_const', 'fo_none',
                             'ff', 'ff_const', 'ff_cos', 'alpha'])
    ap.add_argument('--scope', default='all2d', choices=['all2d', 'jnsp53'])
    ap.add_argument('--non2d', default='zero', choices=['zero', 'avg'])
    ap.add_argument('--rank', type=int, default=16, help='Rank for JNSP')
    ap.add_argument('--seed', type=int, default=0, help='Seed for random-projection controls')
    ap.add_argument('--ties_density', type=float, default=0.2)
    ap.add_argument('--ties_scaling', type=float, default=1.0)
    ap.add_argument('--dare_drop', type=float, default=0.9)
    ap.add_argument('--dare_scaling', type=float, default=1.0)
    ap.add_argument('--ta_scaling', type=float, default=1.0)
    ap.add_argument('--ta_ref_scaling', type=float, default=0.2,
                    help='TA lambda whose global norm ta_normalized matches')
    ap.add_argument('--iso_cts_common_frac', type=float, default=0.8)
    ap.add_argument('--alpha_r1', type=int, default=32,
                    help='AlphaMerging per-task rank r1 (best of the B/16 sweep)')
    ap.add_argument('--alpha_k', type=int, default=64,
                    help='AlphaMerging compressed peer rank k')
    ap.add_argument('--fo_beta', type=float, default=0.5,
                    help='Fixed contested-weight used by --methods fo_const (control only)')
    ap.add_argument('--proxy', default='test', choices=['test', 'train'],
                    help='ESM proxy source: test reproduces upstream (leaky); train is disjoint')
    ap.add_argument('--esd_seed', type=int, default=1,
                    help='Seed for ESM proxy-image draw (upstream uses 1)')
    ap.add_argument('--func_proxy_n', type=int, default=64,
                    help='Images used to build each task functional basis (ff methods)')
    ap.add_argument('--func_proxy', default='train', choices=['test', 'train'],
                    help='Proxy split for functional bases; train keeps it disjoint from eval')
    ap.add_argument('--func_seed', type=int, default=1,
                    help='Seed for functional-basis proxy draw')
    ap.add_argument('--model', default='ViT-B-16')
    ap.add_argument('--model_location', default=os.environ.get('MODEL_LOCATION', ''))
    ap.add_argument('--data_location', default=os.environ.get('DATA_LOCATION', ''))
    ap.add_argument('--save_merged', action='store_true')
    ap.add_argument('--save_preds', action='store_true',
                    help='Dump per-example preds/labels per dataset to preds_dump/<model>_<tag>/')
    ap.add_argument('--preds_tag', default='',
                    help='Subdirectory name for --save_preds (default: <method>_<scope>)')
    ap.add_argument('--tag', default='', help='Suffix for the merged checkpoint filename')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--eval_only', nargs='+', help='Evaluate existing checkpoints without merging')
    ap.add_argument('--out', default='unified_results.json')
    cli = ap.parse_args()

    # Setup fda_args for evaluation
    _saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        args = parse_arguments()
    finally:
        sys.argv = _saved_argv
    args.model = cli.model
    args.model_location = f'{cli.model_location}/{cli.model}'
    args.data_location = cli.data_location

    print(f"[unified_merge] model={cli.model} scope={cli.scope} non2d={cli.non2d}")
    print(f"[unified_merge] model_location={args.model_location}")
    print(f"[unified_merge] data_location={args.data_location}")

    results = {}

    # Eval-only mode
    if cli.eval_only:
        for ckpt_path in cli.eval_only:
            name = os.path.basename(ckpt_path).replace('.pt', '')
            print(f"\n[eval_only] {name}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = ckpt if isinstance(ckpt, dict) and 'model.visual.proj' in ckpt else ckpt['state_dict']
            preds_dir = None
            if cli.save_preds:
                tag = cli.preds_tag or (name[len('merged_'):] if name.startswith('merged_') else name)
                preds_dir = os.path.join(SCRIPT_DIR, 'preds_dump', cli.model, tag)
            scores = evaluate_all(args.model_location, sd, args, DATASETS, preds_dir=preds_dir)
            if preds_dir:
                print(f"[preds] {preds_dir}")
            results[name] = {'scores': scores}
            print(f"[eval_only] {name}: AVG={scores['avg']:.2f}")
        with open(cli.out, 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Load task vectors
    base_sd, tvs = load_task_vectors(args.model_location, DATASETS)
    keys = get_merge_scope(base_sd, cli.scope)
    print(f"[unified_merge] Merge scope: {len(keys)} tensors")

    # Merge
    results_merged = {}
    func_bases = None
    for method in cli.methods:
        print(f"\n[merge] === {method} ===")
        t0 = time.time()

        if method == 'ta':
            merged = merge_ta(tvs, keys, scaling=cli.ta_scaling)
        elif method == 'ties':
            merged = merge_ties(tvs, keys, density=cli.ties_density, scaling=cli.ties_scaling)
        elif method == 'dare':
            merged = merge_dare(tvs, keys, drop_rate=cli.dare_drop, scaling=cli.dare_scaling)
        elif method == 'jnsp':
            merged = merge_jnsp(tvs, keys, rank=cli.rank)
        elif method == 'ta_norm':
            if 'jnsp' not in results_merged:
                print("[merge] ta_norm needs JNSP as the norm target; computing it")
                results_merged['jnsp'] = merge_jnsp(tvs, keys, rank=cli.rank)
            merged = merge_ta_normmatched(tvs, keys, results_merged['jnsp'])
        elif method == 'ta_tasknorm':
            if 'jnsp' not in results_merged:
                print("[merge] ta_tasknorm needs JNSP as the norm target; computing it")
                results_merged['jnsp'] = merge_jnsp(tvs, keys, rank=cli.rank)
            merged = merge_ta_normmatched(tvs, keys, results_merged['jnsp'], per_tensor=False)
        elif method == 'jnsp_scaled':
            merged = merge_jnsp_scaled(tvs, keys, rank=cli.rank, ref_scaling=cli.ta_ref_scaling)
        elif method == 'ta_normalized':
            merged = merge_ta_normalized(tvs, keys, ref_scaling=cli.ta_ref_scaling)
        elif method == 'rand_proj':
            merged = merge_random_proj(tvs, keys, rank=cli.rank, seed=cli.seed)
        elif method == 'jnsp_seq':
            merged = merge_jnsp_sequential(tvs, keys, rank=cli.rank, grow_basis=True)
        elif method == 'jnsp_seq_trunc':
            merged = merge_jnsp_sequential(tvs, keys, rank=cli.rank, grow_basis=False)
        elif method == 'jnsp_global':
            merged = merge_jnsp_global(tvs, keys, rank=cli.rank)
        elif method == 'fo':
            merged = merge_fo(tvs, keys, rank=cli.rank, beta_mode='cos')
        elif method == 'fo_const':
            merged = merge_fo(tvs, keys, rank=cli.rank, beta_mode='const',
                              beta_const=cli.fo_beta)
        elif method == 'fo_none':
            merged = merge_fo(tvs, keys, rank=cli.rank, beta_mode='none')
        elif method in ('ff', 'ff_cos', 'ff_const'):
            # Functional bases come from PCA of each task's layer INPUTS under its
            # own finetuned model, so they live in the same input/row space JNSP's
            # (I - P_i) acts on. Built once and reused across the three beta modes.
            if func_bases is None:
                func_bases = build_func_bases(
                    tvs, DATASETS, args.model_location, args.data_location, args.device,
                    rank=cli.rank, proxy_n=cli.func_proxy_n, seed=cli.func_seed,
                    proxy=cli.func_proxy)
            beta_mode = {'ff': 'none', 'ff_cos': 'cos', 'ff_const': 'const'}[method]
            merged = merge_ff(tvs, keys, func_bases, DATASETS, rank=cli.rank,
                              beta_mode=beta_mode, beta_const=cli.fo_beta)
        elif method == 'tsvm':
            merged = merge_tsvm(tvs, keys, device=args.device)
        elif method == 'iso_c':
            merged = merge_iso_c(tvs, keys, device=args.device)
        elif method == 'iso_cts':
            merged = merge_iso_cts(tvs, keys, common_frac=cli.iso_cts_common_frac, device=args.device)
        elif method == 'alpha':
            merged = merge_alpha_wudi(tvs, keys, rank=cli.alpha_r1, k=cli.alpha_k)
        elif method == 'esm':
            esd = build_esd(tvs, DATASETS, args.model_location, args.data_location, args.device,
                            proxy=cli.proxy, seed=cli.esd_seed)
            merged = merge_esm(tvs, keys, esd, DATASETS, device=args.device)
            del esd
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - t0
        print(f"[merge] {method}: {elapsed:.1f}s")
        results_merged[method] = merged

        sd = assemble_state_dict(base_sd, merged, non2d_deltas=tvs, non2d_mode=cli.non2d)

        if cli.save_merged:
            suffix = f'_{cli.tag}' if cli.tag else ''
            out_name = f"merged_{method}_{cli.scope}_{cli.non2d}{suffix}.pt"
            out_path = os.path.join(SCRIPT_DIR, out_name)
            torch.save({'state_dict': sd}, out_path)
            print(f"[save] {out_path}")

        results[method] = {'merge_time': elapsed, 'scores': None}

        if cli.eval:
            print(f"[eval] {method}")
            preds_dir = None
            if cli.save_preds:
                tag = cli.preds_tag or f'{method}_{cli.scope}'
                preds_dir = os.path.join(SCRIPT_DIR, 'preds_dump', cli.model, tag)
            scores = evaluate_all(args.model_location, sd, args, DATASETS, preds_dir=preds_dir)
            if preds_dir:
                print(f"[preds] {preds_dir}")
            results[method]['scores'] = scores
            print(f"[eval] {method}: " + ' '.join(f'{d}={scores[d]:.2f}' for d in DATASETS) + f" | AVG={scores['avg']:.2f}")

    with open(os.path.join(SCRIPT_DIR, cli.out), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {cli.out}")


if __name__ == '__main__':
    main()
