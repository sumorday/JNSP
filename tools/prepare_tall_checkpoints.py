"""Convert nik-dim/tall_masks checkpoints into the layout quick_demo_test.py reads.

TALL (Wang et al., ICML 2024) ships, per backbone under `single_task/<BB>/`:
  - `<Dataset>Val/nonlinear_finetuned.pt`  : plain OrderedDict, 158 `model.*` keys
  - `MNISTVal/nonlinear_zeroshot.pt`       : same format, the pretrained base
  - `head_<Dataset>Val.pt`                 : pickled `src.models.modeling.ClassificationHead`

This script writes `{dest}/zeroshot.pt`, `{dest}/{Dataset}/finetuned.pt` and
`{dest}/head_{Dataset}.pt` as `src.modeling` objects so the existing loaders
work unchanged. Heads are re-saved from our own `_fda_line` copy when present
(TALL's ViT-B/16 head_MNISTVal is bit-identical to ours), else rebuilt from the
TALL pickle via the `src.models` shim.

Usage (from Vision/):
    python experiments/null_space_filtering/prepare_tall_checkpoints.py --backbone VIT-B-16
"""
import argparse
import os
import shutil
import sys
import types

import torch

VISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VISION_ROOT)

from fda_args import parse_arguments  # noqa: E402
from src.modeling import ImageEncoder, ClassificationHead  # noqa: E402

DATASETS = ['MNIST', 'EuroSAT', 'GTSRB', 'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']


def _tall_module_shim():
    """TALL pickles reference `src.models.modeling`; alias it to plain torch modules."""
    pkg = types.ModuleType('src.models')
    pkg.__path__ = []
    mod = types.ModuleType('src.models.modeling')

    class _Stub(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    for name in ('ImageEncoder', 'ImageClassifier', 'ClassificationHead'):
        setattr(mod, name, _Stub)
    pkg.modeling = mod
    sys.modules['src.models'] = pkg
    sys.modules['src.models.modeling'] = mod


def load_raw_state_dict(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        return obj
    return obj.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True, help='TALL dir name, e.g. VIT-B-16 / ViT-B-32 / VIT-L-14')
    ap.add_argument('--tall_root', required=True)
    ap.add_argument('--fda_root', required=True)
    ap.add_argument('--dest_root', required=True)
    cli = ap.parse_args()

    model_name = {'VIT-B-16': 'ViT-B-16', 'VIT-B-32': 'ViT-B-32', 'ViT-B-32': 'ViT-B-32',
                  'VIT-L-14': 'ViT-L-14', 'ViT-L-14': 'ViT-L-14'}[cli.backbone]
    src = os.path.join(cli.tall_root, cli.backbone)
    dest = os.path.join(cli.dest_root, model_name)
    os.makedirs(dest, exist_ok=True)
    _saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        args = parse_arguments()
    finally:
        sys.argv = _saved
    # `__pretrained__` with an empty suffix tells ImageEncoder to skip the OpenAI
    # download; every weight is overwritten by the TALL state dict (strict=True).
    args.model = f'{model_name}__pretrained__'

    # TALL checkpoints contain only the image tower + text_projection (158 keys),
    # matching our own zeroshot.pt, so the text transformer must not be kept.
    encoder = ImageEncoder(args, keep_lang=False)

    base_sd = load_raw_state_dict(os.path.join(src, 'MNISTVal', 'nonlinear_zeroshot.pt'))
    encoder.load_state_dict(base_sd, strict=True)
    out = os.path.join(dest, 'zeroshot.pt')
    torch.save(encoder, out)
    print(f'  saved {out} ({len(base_sd)} keys)')

    fda_heads = os.path.join(cli.fda_root, model_name)
    for ds in DATASETS:
        ft_sd = load_raw_state_dict(os.path.join(src, f'{ds}Val', 'nonlinear_finetuned.pt'))
        encoder.load_state_dict(ft_sd, strict=True)
        os.makedirs(os.path.join(dest, ds), exist_ok=True)
        out = os.path.join(dest, ds, 'finetuned.pt')
        torch.save(encoder, out)
        print(f'  saved {out}')

        head_out = os.path.join(dest, f'head_{ds}.pt')
        local_head = os.path.join(fda_heads, f'head_{ds}.pt')
        if model_name == 'ViT-B-16' and os.path.exists(local_head):
            shutil.copyfile(local_head, head_out)
            print(f'  copied head from {local_head}')
        else:
            _tall_module_shim()
            raw = torch.load(os.path.join(src, f'head_{ds}Val.pt'), map_location='cpu', weights_only=False)
            head = ClassificationHead(normalize=raw.normalize, weights=raw.state_dict()['weight'],
                                      biases=raw.state_dict()['bias'])
            head.save(head_out)
            print(f'  rebuilt head -> {head_out}')

    print(f'[done] {cli.backbone} -> {dest}')


if __name__ == '__main__':
    main()
