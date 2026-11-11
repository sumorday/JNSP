# Joint Null-Space Projection (JNSP) for Data-Free Model Merging

Code for the paper *Joint Null-Space Projection for Data-Free Model Merging*.

JNSP processes every two-dimensional weight tensor separately. It takes the top-`r` right singular vectors of each task's delta, which span the input directions along which that task moves the representation most. For each task, the singular vectors of all *other* tasks are stacked and orthonormalized. The task's delta is projected off that span, and the projected deltas are summed. The method is closed-form, needs no data or forward pass, and has a single hyperparameter, `r` (16 everywhere in the paper).

The core operator is `merge_jnsp` in `merging/unified_merge.py`. Its helpers `top_r_basis`, `qr_orthonormalize` and `project_out` are in the same file.

## Layout

```
merging/unified_merge.py           merging harness: JNSP, controls, baselines, evaluation
merging/check_prop2_sequential.py  numerical check of the sequential-vs-joint analysis
tools/prepare_tall_checkpoints.py  converts the TALL task-vector release to the layout below
src/                               CLIP image encoder, classification heads, dataset loaders, evaluator
fda_args.py                        shared evaluation arguments (batch size, device, ...)
```

The `src/` directory and `fda_args.py` come from the public task-arithmetic / FDA evaluation code. They are lightly modified.

## Environment

Tested with Python 3.9, PyTorch 2.3.0, torchvision 0.18.0, numpy 1.26.4, scipy 1.13.0 and `open_clip_torch`.

```bash
pip install torch torchvision open_clip_torch numpy scipy tqdm
```

## Checkpoints and data

```
$MODEL_LOCATION/<MODEL>/            # MODEL in {ViT-B-16, ViT-B-32, ViT-L-14}
    zeroshot.pt                     # pretrained CLIP image encoder
    <Dataset>/finetuned.pt          # one fine-tuned encoder per task
    head_<Dataset>.pt               # classification heads
$DATA_LOCATION/
    MNIST/ EuroSAT/ GTSRB/ SVHN/ DTD/ RESISC45/ Cars/ SUN397/
```

The eight tasks are MNIST, EuroSAT, GTSRB, SVHN, DTD, RESISC45, Cars and SUN397. Data preparation follows the task-arithmetic benchmark.

For the TALL release, run the conversion script first:

```bash
python tools/prepare_tall_checkpoints.py --tall_root /path/to/tall --fda_root /path/to/checkpoints --dest_root /path/to/tall_converted
```

Then point `--model_location` at the converted directory.

## Usage

```bash
export MODEL_LOCATION=/path/to/checkpoints
export DATA_LOCATION=/path/to/data

# JNSP, r = 16 (main result)
python merging/unified_merge.py --methods jnsp --scope all2d --non2d zero --rank 16 \
    --model ViT-B-16 --eval --out results_jnsp.json
```

Change `--model` to `ViT-B-32` or `ViT-L-14` for the other backbones. Pass `--save_merged` to also write the merged state dict.

| Paper item | Command |
|---|---|
| Main table / backbone table, JNSP | `--methods jnsp --scope all2d --non2d zero --rank 16` |
| Subspace baselines (TSV-M, Iso-C, Iso-CTS) | `--methods tsvm iso_c iso_cts --scope all2d --non2d zero` |
| AlphaMerging re-implementation | `--methods alpha --scope all2d --non2d zero --alpha_r1 32 --alpha_k 64` |
| Task Arithmetic / TIES / DARE | `--methods ta ties dare --ta_scaling <λ> --ties_density 0.2 --ties_scaling <λ> --dare_drop 0.9 --dare_scaling <λ>`. Merge coefficients are swept and the best is reported. |
| Rank sweep | `--methods jnsp jnsp_global --rank <r>` for r in 8 … 32 |
| Shared-basis control | `--methods jnsp_global --rank 16` |
| Order dependence (sequential) | `--methods jnsp jnsp_seq jnsp_global --rank 8` |
| Random-projection control | `--methods rand_proj --rank 16 --seed <s>` |
| Re-evaluate saved merges | `--eval_only merged_*.pt` |
| Sequential analysis check | `python merging/check_prop2_sequential.py --model_location $MODEL_LOCATION/ViT-B-16 --rank 8` |

The WUDI baseline row uses the authors' released WUDI operator and is not reimplemented here. `unified_merge.py` also contains some exploratory variants that the paper does not report (`fo*`, `ff*`, `ta_norm*`, `jnsp_scaled`, `esm`).

## Notes

- `--scope all2d` merges every two-dimensional tensor of the image encoder except the text projection. `--non2d zero` leaves the one-dimensional parameters (LayerNorm, biases) at their pretrained values.
- JNSP is deterministic. For fixed checkpoints and `r`, repeated runs give identical merges.
