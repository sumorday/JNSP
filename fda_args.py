import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--name", type=str, default='init_by_gauss_random', help="init FDA")
    parser.add_argument("--model", type=str, default='ViT-B-32', help="Model type (e.g. ViT-B-32, ViT-B-16, ViT-L-14)")
    parser.add_argument("--init_params", type=str, default='pretrained')

    # file location
    parser.add_argument("--data_location", type=str, default='/path/to/data',help="Root directory for datasets")
    parser.add_argument("--model_location", type=str, default='/path/to/checkpoints', help="Checkpoint directory")
    parser.add_argument("--read_anchors_path", type=str,default='/path/to/anchors',help='Read anchor path')

    # Anchor
    parser.add_argument("--anchor_num", type=int, default=64, help="Number of anchors")
    parser.add_argument("--token_num", type=int, default=50, help="Token number (for embedding, etc.)")
    parser.add_argument("--anchor_loss", type=str, default='cos', choices=['cos', 'mse', 'l1'], help="Loss for anchor generation")
    parser.add_argument("--opt_steps", type=int, default=800, help="Optimization steps for anchor generation")
    parser.add_argument("--opt_lr", type=float, default=1e-2, help="Learning rate for anchor optimization")
    parser.add_argument("--scale", type=float, default=1, help="Scaling factor for Gauss")

    # Adaptation
    parser.add_argument("--adapt_batch_size", type=int, default=128, help="Batch size for adaptation")
    parser.add_argument("--adapt_epochs", type=int, default=500, help="Epochs for adaptation")
    parser.add_argument("--adapt_lr", type=float, default=1e-2, help="Learning rate for adaptation")
    parser.add_argument("--adapt_loss", type=str, default='mse', help="Loss function for adaptation")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha coefficient for adaptation")

    # Few-shot Hybrid + Iterative Refinement (Stage-2 extensions; all off by default
    # so the official data-free baseline stays fully reproducible)
    parser.add_argument("--use_fewshot", action="store_true", help="Mix real few-shot train images into adaptation")
    parser.add_argument("--fewshot_num", type=int, default=32, help="Max real training images sampled per task")
    parser.add_argument("--lambda_fda", type=float, default=1.0, help="Loss weight for synthetic FDA anchors")
    parser.add_argument("--lambda_real", type=float, default=1.0, help="Loss weight for real few-shot samples")
    parser.add_argument("--iterative_rounds", type=int, default=1, help="Number of adaptation rounds (>=2 triggers anchor reconstruction)")
    parser.add_argument("--fewshot_seed", type=int, default=-1, help="Seed for few-shot sampling (default: reuse --seed)")
    parser.add_argument("--real_only", action="store_true", help="Tier-1a ablation: use ONLY real few-shot samples (no synthetic anchors); requires --use_fewshot")
    parser.add_argument("--project_real_onto_anchors", action="store_true", help="Experiment D ablation: project each real sample's layer input onto the existing anchor subspace before using it as a training target; requires --use_fewshot and not --real_only")
    parser.add_argument("--save_path_suffix", type=str, default='', help="Optional suffix appended to save_path, for isolating new experiment variants (e.g. E3/E4) without disturbing existing path conventions")

    # Gradient-Norm-Matching anchor construction (off by
    # default so lambda_norm=0.0 reproduces construct_fda.py's original
    # dropped-norm_diff behavior exactly -- see
    # experiments/gradient_norm_matching/construct_fda_gnm.py)
    parser.add_argument("--lambda_norm", type=float, default=0.0, help="Weight on the gradient/task-vector norm-matching term added to grad_diff_loss during anchor construction (0.0 = baseline angle-loss-only behavior)")

    # Saving
    parser.add_argument("--save_model", action="store_true", help="Save model after training")
    parser.add_argument("--save_dual_anchors", action="store_true", help="Save dual anchors")
    parser.add_argument("--save_anchors_path", type=str, default='/path/to/anchors',help="Anchor save path")
    parser.add_argument("--save_merged_model_path", type=str,default='/path/to/merged_models')

    # Semantic anchor initialization (Variant A2, data-free). Consumed only by
    # construct_fda_semantic.py; absent from the official construct_fda.py path.
    parser.add_argument("--anchors_per_task", type=int, default=8, help="Semantic anchors sampled per task (anchor_num must equal 8 * this)")
    parser.add_argument("--subspace_rank", type=str, default='auto', help="'auto' = one component per class embedding, else an integer")
    parser.add_argument("--alpha_scale", type=float, default=1.0, help="Std of coefficients along text principal directions")
    parser.add_argument("--noise_sigma", type=float, default=0.1, help="Per-patch-token noise std, relative to the semantic vector's RMS")
    parser.add_argument("--openclip_cachedir", type=str, default=None, help="open_clip download cache directory")
    parser.add_argument("--dataset_subset", type=str, default='', help="Comma-separated task names for a reduced-scope pilot; empty = all tasks in utils.dataset_list")

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args