"""
Full fine-tuning gradient / delta importance (no adapters).

Control for the LoRA delta measurement. Produces two ImportanceResult files:

    full_ft_grad_{run}.json   mean per-layer full-parameter gradient norm
                              (extra: step-0 matrix and a LoRA-target-modules-only matrix)
    full_ft_delta_{run}.json  per-layer ||Delta W||_F after --n_steps of full FT
                              (only if --n_steps > 0)

Paper setting: 500 steps/task, embeddings + lm_head frozen,
  <= 4B models : --dtype float32  --optimizer adamw            --lr 1e-5
  >= 7B models : --dtype float32  --optimizer paged_adamw8bit  --lr 1e-5 --gradient_checkpointing

    python experiments/run_full_ft.py --model_name_or_path Qwen/Qwen3-0.6B \
        --n_steps 500 --lr 1e-5 --optimizer adamw --dtype float32 \
        --output_dir results/full_ft/qwen3-0.6b

    # Gradient norms at the pretrained point only (no updates):
    python experiments/run_full_ft.py --model_name_or_path Qwen/Qwen3-0.6B --n_steps 0 ...
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch

from layerimp.data import ALL_TASKS, TASK_GROUPS, build_task_dataloaders
from layerimp.full_ft import FullFTGradAnalyzer
from layerimp.importance import ImportanceResult
from layerimp.models import get_layer_types, load_from_args
from layerimp.utils import init_wandb as _init_wandb
from layerimp.utils import save_result, setup_logging

setup_logging()
logger = logging.getLogger("full_ft")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full fine-tuning gradient / delta importance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument(
        "--attn_implementation",
        type=str, default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="float32 recommended for models ≤3B: bf16 weight updates below "
             "~4e-3 relative magnitude are lost to rounding, biasing ΔW.",
    )
    p.add_argument("--hf_cache_dir", type=str, default=None)

    # ── Full FT hyperparameters ───────────────────────────────────────────
    p.add_argument(
        "--n_steps", type=int, default=30,
        help="Fine-tuning steps per task (paper: 500). "
             "0 = measure pretrained-point gradient norms only, no updates.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate. Typical: 1e-5 for adamw, 1e-3 for sgd.",
    )
    p.add_argument(
        "--optimizer", type=str, default="adamw",
        choices=["adamw", "sgd", "adamw8bit", "paged_adamw8bit"],
        help="adamw8bit/paged_adamw8bit (bitsandbytes) give Adam-family "
             "updates at 2 bytes/param state — use for ≥7B models so ΔW is "
             "comparable to the AdamW runs. sgd deltas track raw gradient "
             "magnitude and are NOT comparable to Adam-family deltas.",
    )
    p.add_argument(
        "--train_scope", type=str, default="decoder", choices=["decoder", "all"],
        help="decoder = train all decoder-layer params, freeze embeddings/"
             "lm_head (per-layer attribution is then exhaustive).",
    )
    p.add_argument(
        "--target_modules", nargs="+", default=None,
        help="Module names for the LoRA-comparable gradient submatrix "
             "(default: auto-detect, same rule as run_lora_delta.py).",
    )

    # ── Tasks ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--tasks", nargs="+",
        default=ALL_TASKS,
        help=f"Task names. Available: {ALL_TASKS}",
    )
    p.add_argument(
        "--task_group", type=str, default=None,
        choices=list(TASK_GROUPS.keys()),
        help="Expand a named task group instead of --tasks.",
    )

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--n_examples", type=int, default=256)
    p.add_argument("--seq_len",    type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable non-reentrant gradient checkpointing "
                        "(works with frozen embeddings).")
    p.add_argument("--shuffle_labels", action="store_true",
                   help="Random-label control: shuffle targets within each batch.")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="results/full_ft")
    p.add_argument("--run_name",   type=str, default=None)

    # ── W&B ──────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags",     nargs="*", default=[])

    return p


def init_wandb(args, run_name: str):
    return _init_wandb(
        args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        tags=["full_ft"] + (args.wandb_tags or []), config=vars(args),
    )


def log_to_wandb(run, result: ImportanceResult, key_prefix: str):
    """Log one importance matrix (table + heatmap) to W&B."""
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    cols = ["layer_idx", "layer_type"] + result.task_names
    table = wandb.Table(columns=cols)
    for l_idx in range(result.n_layers):
        row = [l_idx, result.layer_types[l_idx]] + result.matrix_norm[l_idx].tolist()
        table.add_data(*row)
    run.log({f"{key_prefix}_importance_norm": table})

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(
            figsize=(max(12, result.n_tasks * 1.2), result.n_layers * 0.35 + 2)
        )
        sns.heatmap(
            result.matrix_norm,
            ax=ax,
            xticklabels=result.task_names,
            yticklabels=[f"L{i}[{t}]" for i, t in enumerate(result.layer_types)],
            cmap="viridis",
        )
        ax.set_title(f"{result.method} importance (normalised) — {result.model_name}")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        run.log({f"{key_prefix}_heatmap": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not generate heatmap: {e}")


def print_summary(result: ImportanceResult, label: str):
    print(f"\nTop-5 layers per task ({label}, normalised):")
    for task in result.task_names:
        top5 = result.top_layers_for_task(task, k=5)
        entries = ", ".join(
            f"L{l}[{result.layer_types[l]}]={s:.3f}" for l, s in top5
        )
        print(f"  {task:<30}: {entries}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.task_group is not None:
        args.tasks = TASK_GROUPS[args.task_group]
        logger.info(f"Expanded task group '{args.task_group}' → {args.tasks}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_short = Path(args.model_name_or_path).name
    run_name = args.run_name or (
        f"{model_short}_full_ft_{args.optimizer}_s{args.n_steps}_{ts}"
    )
    logger.info(f"Run: {run_name}")
    logger.info(
        f"Config: steps={args.n_steps}, lr={args.lr}, optimizer={args.optimizer}, "
        f"scope={args.train_scope}, dtype={args.dtype}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No CUDA device found — running on CPU (very slow).")

    # Full FT with frozen embeddings breaks reentrant checkpointing (block
    # inputs don't require grad), so enable the non-reentrant variant here
    # instead of inside the model loader.
    model, tokenizer = load_from_args(args, gradient_checkpointing=False)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled (non-reentrant).")

    layer_types = get_layer_types(model)

    dataloaders = build_task_dataloaders(
        tasks=args.tasks,
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        cache_dir=args.hf_cache_dir,
        num_workers=args.num_workers,
    )

    analyzer = FullFTGradAnalyzer(
        model=model,
        lr=args.lr,
        n_steps=args.n_steps,
        optimizer=args.optimizer,
        train_scope=args.train_scope,
        target_modules=args.target_modules,
        shuffle_labels=args.shuffle_labels,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run = init_wandb(args, run_name)
    t0 = time.time()

    results = analyzer.run(
        task_dataloaders=dataloaders,
        device=device,
        layer_types=layer_types,
        model_name=args.model_name_or_path,
        n_samples=args.n_examples,
        seq_len=args.seq_len,
        checkpoint_path=os.path.join(
            args.output_dir, f"checkpoint_{run_name}.npz"
        ),
    )

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed/60:.1f} min.")

    grad_result, delta_result = results["grad"], results["delta"]
    save_result(grad_result, args.output_dir, f"full_ft_grad_{run_name}")
    if delta_result is not None:
        save_result(delta_result, args.output_dir, f"full_ft_delta_{run_name}")

    if run is not None:
        run.summary["elapsed_minutes"] = elapsed / 60
        log_to_wandb(run, grad_result, "full_ft_grad")
        if delta_result is not None:
            log_to_wandb(run, delta_result, "full_ft_delta")
        run.finish()

    # ── Quick summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Model      : {grad_result.model_name}")
    print(f"Layers     : {grad_result.n_layers}  Tasks: {grad_result.n_tasks}")
    print(f"Full FT    : optimizer={args.optimizer}, steps={args.n_steps}, "
          f"lr={args.lr}, scope={args.train_scope}, dtype={args.dtype}")
    print_summary(grad_result, "full_ft_grad — mean grad norm")
    if delta_result is not None:
        print_summary(delta_result, "full_ft_delta — ||ΔW||_F")
    print("=" * 60)


if __name__ == "__main__":
    main()
