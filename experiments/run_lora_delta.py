"""
LoRA delta (plasticity) importance.

Fine-tunes rank-r LoRA adapters for --n_steps on each task (adapters reset
between tasks) and records I[l, t] = sum_{modules in layer l} ||Delta W||_F.

Paper setting (rank 8, alpha 16, 200 steps, lr 1e-4, 512 examples x 512 tokens):
    python experiments/run_lora_delta.py --model_name_or_path Qwen/Qwen3-8B \
        --lora_rank 8 --lora_alpha 16 --n_steps 200 --lr 1e-4 \
        --n_examples 512 --seq_len 512 --batch_size 4 --seed 42 \
        --output_dir results/lora_delta/qwen3-8b

Output: {output_dir}/lora_delta_{run_name}.json (+ .npy)
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from layerimp.data import ALL_TASKS, TASK_GROUPS, build_task_dataloaders
from layerimp.importance import ImportanceResult
from layerimp.lora_delta import LoRADeltaAnalyzer
from layerimp.models import get_layer_types, load_from_args
from layerimp.utils import init_wandb as _init_wandb
from layerimp.utils import save_result, setup_logging

setup_logging()
logger = logging.getLogger("lora_delta")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA delta (plasticity) importance",
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
    )
    p.add_argument("--hf_cache_dir", type=str, default=None)

    # ── LoRA hyperparameters ───────────────────────────────────────────────
    p.add_argument(
        "--lora_rank", type=int, default=8,
        help="LoRA rank r.  Higher = more expressive but more memory.",
    )
    p.add_argument(
        "--lora_alpha", type=int, default=16,
        help="LoRA alpha (scaling = alpha/rank).  Typically 2× rank.",
    )
    p.add_argument(
        "--n_steps", type=int, default=200,
        help="Fine-tuning steps per task.  200 is sufficient to see divergence.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="AdamW learning rate for LoRA params.",
    )
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help=(
            "Linear module names to apply LoRA to.  "
            "Default (None) auto-detects by architecture: "
            "transformers → gate_proj/up_proj/down_proj, "
            "Mamba/Falcon-Mamba → in_proj/x_proj/dt_proj, "
            "RWKV → receptance/key/value/output/gate."
        ),
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
    p.add_argument("--n_examples", type=int, default=512,
                   help="Calibration examples per task (used for fine-tuning).")
    p.add_argument("--seq_len",    type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--num_workers",type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce GPU memory at the cost of speed.")
    p.add_argument("--shuffle_labels", action="store_true",
                   help="Random-label control: shuffle targets within each batch before backward.")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="results/lora_delta")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Optional suffix for output filenames.")

    # ── W&B ──────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags",     nargs="*", default=[])

    return p


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(args, run_name: str):
    return _init_wandb(
        args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        tags=["lora_delta"] + (args.wandb_tags or []), config=vars(args),
    )


def log_to_wandb(run, result: ImportanceResult):
    """Log importance matrix, top-layers, and task similarity to W&B."""
    try:
        import wandb
    except ImportError:
        return
    if run is None:
        return

    cols = ["layer_idx", "layer_type"] + result.task_names
    table = wandb.Table(columns=cols)
    for l_idx in range(result.n_layers):
        row = [l_idx, result.layer_types[l_idx]] + result.matrix_norm[l_idx].tolist()
        table.add_data(*row)
    run.log({"lora_delta_importance_norm": table})

    for task in result.task_names:
        top5 = result.top_layers_for_task(task, k=5)
        run.log({f"top5_{task}/L{l}": s for l, s in top5})

    sim = result.task_similarity_matrix()
    sim_table = wandb.Table(
        columns=result.task_names,
        data=sim.tolist(),
    )
    run.log({"task_similarity": sim_table})

    # Heatmap
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
        ax.set_title(f"LoRA delta importance (normalised) — {result.model_name}")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        run.log({"lora_delta_heatmap": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not generate heatmap: {e}")

    run.summary["n_layers"] = result.n_layers
    run.summary["n_tasks"]  = result.n_tasks
    run.summary["method"]   = result.method


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.task_group is not None:
        args.tasks = TASK_GROUPS[args.task_group]
        logger.info(f"Expanded task group '{args.task_group}' → {args.tasks}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_short = Path(args.model_name_or_path).name
    run_name = args.run_name or (
        f"{model_short}_lora_delta_r{args.lora_rank}_s{args.n_steps}_{ts}"
    )
    logger.info(f"Run: {run_name}")
    logger.info(
        f"Config: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"steps={args.n_steps}, lr={args.lr}, targets={args.target_modules}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No CUDA device found — running on CPU (very slow).")

    model, tokenizer = load_from_args(args)
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

    analyzer = LoRADeltaAnalyzer(
        model=model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lr=args.lr,
        n_steps=args.n_steps,
        shuffle_labels=args.shuffle_labels,
    )

    run = init_wandb(args, run_name)
    t0 = time.time()

    result = analyzer.run(
        task_dataloaders=dataloaders,
        device=device,
        layer_types=layer_types,
        model_name=args.model_name_or_path,
        n_samples=args.n_examples,
        seq_len=args.seq_len,
    )

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed/60:.1f} min.")

    save_result(result, args.output_dir, f"lora_delta_{run_name}")

    if run is not None:
        run.summary["elapsed_minutes"] = elapsed / 60
        log_to_wandb(run, result)
        run.finish()

    # ── Quick summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Method     : {result.method}")
    print(f"Model      : {result.model_name}")
    print(f"Layers     : {result.n_layers}  Tasks: {result.n_tasks}")
    print(f"LoRA       : rank={args.lora_rank}, steps={args.n_steps}, lr={args.lr}")
    print()
    print("Top-5 layers per task (normalised ||ΔW||_F):")
    for task in result.task_names:
        top5 = result.top_layers_for_task(task, k=5)
        entries = ", ".join(
            f"L{l}[{result.layer_types[l]}]={s:.3f}" for l, s in top5
        )
        print(f"  {task:<30}: {entries}")
    print()
    sim = result.task_similarity_matrix()
    tnames = result.task_names
    min_sim = 1.0
    min_pair = ("", "")
    for i in range(len(tnames)):
        for j in range(i + 1, len(tnames)):
            if sim[i, j] < min_sim:
                min_sim = sim[i, j]
                min_pair = (tnames[i], tnames[j])
    print(f"Most divergent tasks: {min_pair[0]} ↔ {min_pair[1]} (cosine={min_sim:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
