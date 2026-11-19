"""
Zero-shot layer importance: layer ablation (necessity) or gradient x activation.

Produces an importance matrix I[n_layers, n_tasks] saved as an ImportanceResult
JSON (+ .npy of the raw matrix).

Ablation, all layers in one job:
    python experiments/run_ablation.py --model_name_or_path Qwen/Qwen3-0.6B \
        --method ablation --output_dir results/ablation/qwen3-0.6b

Ablation, one layer per job (cluster array), then merge:
    python experiments/run_ablation.py --model_name_or_path Qwen/Qwen3-8B \
        --method ablation --ablation_layer_idx 14 --output_dir results/ablation/qwen3-8b
    python experiments/run_ablation.py --mode merge \
        --merge_dir results/ablation/qwen3-8b/ablation_layers \
        --output_dir results/ablation/qwen3-8b          # -> ablation_merged.json

Gradient x activation (Taylor) importance:
    python experiments/run_ablation.py --model_name_or_path Qwen/Qwen3-0.6B \
        --method gradient --output_dir results/gradient/qwen3-0.6b
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch

from layerimp.data import ALL_TASKS, TASK_GROUPS, build_task_dataloaders
from layerimp.importance import (
    AblationImportanceAnalyzer,
    GradientImportanceAnalyzer,
    ImportanceResult,
    merge_ablation_results,
)
from layerimp.models import get_layer_types, load_model_and_tokenizer
from layerimp.utils import init_wandb as _init_wandb
from layerimp.utils import load_result, save_result, setup_logging

setup_logging()
logger = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zero-shot layer importance (ablation / gradient)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["run", "merge"],
        default="run",
        help=(
            "'run'   : compute importance scores (gradient or ablation). "
            "'merge' : aggregate per-layer ablation results from --merge_dir."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help=(
            "Attention implementation. Use 'sdpa' or 'eager' for gradient mode "
            "(flash_attention_2 does not support retain_grad on intermediates). "
            "'flash_attention_2' is fine for ablation mode."
        ),
    )
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--hf_cache_dir", type=str, default=None, help="HuggingFace cache directory.")

    # ── Method ────────────────────────────────────────────────────────────
    p.add_argument(
        "--method",
        choices=["gradient", "ablation"],
        default="gradient",
        help="Importance method to run.",
    )
    p.add_argument(
        "--ablation_layer_idx",
        type=int,
        default=None,
        help=(
            "Ablation mode: index of the single layer to bypass in this job. "
            "If None, all layers are ablated in sequence (slow, single job)."
        ),
    )

    # ── Tasks ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        help=f"Task names. Available: {ALL_TASKS}",
    )
    p.add_argument(
        "--task_group",
        type=str,
        default=None,
        choices=list(TASK_GROUPS.keys()),
        help="Expand a named task group instead of specifying --tasks individually.",
    )

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--n_examples", type=int, default=512, help="Calibration examples per task.")
    p.add_argument("--seq_len",    type=int, default=512, help="Max token length per example.")
    p.add_argument("--batch_size", type=int, default=4,   help="Batch size for forward passes.")
    p.add_argument("--seed",       type=int, default=42,  help="Random seed for data sampling.")
    p.add_argument("--num_workers",type=int, default=2,   help="DataLoader num_workers.")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="results/ablation")
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name suffix. Defaults to method + timestamp.",
    )

    # ── Merge mode ────────────────────────────────────────────────────────
    p.add_argument(
        "--merge_dir",
        type=str,
        default=None,
        help="Directory containing per-layer ablation JSON files to merge.",
    )

    # ── W&B ──────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name. Set to disable W&B.")
    p.add_argument("--wandb_entity",  type=str, default=None, help="W&B entity (team or username).")
    p.add_argument("--wandb_run_name",type=str, default=None, help="W&B run name (defaults to --run_name).")
    p.add_argument("--wandb_tags",    nargs="*", default=[],   help="Additional W&B tags.")

    return p


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def init_wandb(args, run_name: str):
    return _init_wandb(
        args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        tags=["ablation_gradient", args.method] + (args.wandb_tags or []),
        config=vars(args),
    )


def log_importance_to_wandb(run, result: ImportanceResult):
    """Log the importance matrix as tables, heatmaps, and summary metrics."""
    try:
        import wandb
    except ImportError:
        return

    if run is None:
        return

    # ── Normalized importance matrix as a W&B Table ───────────────────
    cols = ["layer_idx", "layer_type"] + result.task_names
    table = wandb.Table(columns=cols)
    for l_idx in range(result.n_layers):
        row = [l_idx, result.layer_types[l_idx]] + result.matrix_norm[l_idx].tolist()
        table.add_data(*row)
    run.log({"importance_matrix_norm": table})

    # ── Raw matrix table ─────────────────────────────────────────────
    table_raw = wandb.Table(columns=cols)
    for l_idx in range(result.n_layers):
        row = [l_idx, result.layer_types[l_idx]] + result.matrix[l_idx].tolist()
        table_raw.add_data(*row)
    run.log({"importance_matrix_raw": table_raw})

    # ── Per-task top-5 layers ─────────────────────────────────────────
    for task in result.task_names:
        top5 = result.top_layers_for_task(task, k=5)
        summary = {f"top5_{task}/layer_{l}": score for l, score in top5}
        run.log(summary)

    # ── Task similarity matrix ────────────────────────────────────────
    sim = result.task_similarity_matrix()
    sim_table = wandb.Table(
        columns=result.task_names,
        data=sim.tolist(),
        rows=result.task_names,
    )
    run.log({"task_similarity": sim_table})

    # ── Heatmap image (requires matplotlib) ──────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(max(14, result.n_tasks * 1.5), result.n_layers * 0.4 + 2))

        # Raw
        sns.heatmap(
            result.matrix,
            ax=axes[0],
            xticklabels=result.task_names,
            yticklabels=[f"L{i} [{t}]" for i, t in enumerate(result.layer_types)],
            cmap="viridis",
            annot=False,
        )
        axes[0].set_title(f"Importance (raw) — {result.method}")
        axes[0].tick_params(axis="x", rotation=45)

        # Normalized
        sns.heatmap(
            result.matrix_norm,
            ax=axes[1],
            xticklabels=result.task_names,
            yticklabels=[f"L{i} [{t}]" for i, t in enumerate(result.layer_types)],
            cmap="viridis",
            annot=False,
        )
        axes[1].set_title(f"Importance (normalized) — {result.method}")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        run.log({"importance_heatmap": wandb.Image(fig)})
        plt.close(fig)

    except Exception as e:
        logger.warning(f"Could not generate heatmap: {e}")

    # ── Task-similarity heatmap ───────────────────────────────────────
    try:
        fig2, ax = plt.subplots(figsize=(max(6, result.n_tasks), max(6, result.n_tasks)))
        sns.heatmap(
            sim, ax=ax,
            xticklabels=result.task_names,
            yticklabels=result.task_names,
            cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f",
        )
        ax.set_title("Task similarity (cosine of importance profiles)")
        plt.tight_layout()
        run.log({"task_similarity_heatmap": wandb.Image(fig2)})
        plt.close(fig2)
    except Exception as e:
        logger.warning(f"Could not generate similarity heatmap: {e}")

    run.summary["n_layers"] = result.n_layers
    run.summary["n_tasks"]  = result.n_tasks
    run.summary["method"]   = result.method


# ---------------------------------------------------------------------------
# Main execution modes
# ---------------------------------------------------------------------------

def run_gradient(args, model, tokenizer, device, run_name: str):
    tasks = args.tasks
    logger.info(f"Running gradient importance on {len(tasks)} tasks: {tasks}")

    dataloaders = build_task_dataloaders(
        tasks=tasks,
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        cache_dir=args.hf_cache_dir,
        num_workers=args.num_workers,
    )

    analyzer = GradientImportanceAnalyzer(model)
    result = analyzer.run(
        task_dataloaders=dataloaders,
        device=device,
        layer_types=get_layer_types(model),
        model_name=args.model_name_or_path,
        n_samples=args.n_examples,
        seq_len=args.seq_len,
    )

    return result, run_name


def run_ablation(args, model, tokenizer, device, run_name: str):
    tasks = args.tasks
    layer_indices = (
        [args.ablation_layer_idx] if args.ablation_layer_idx is not None
        else None  # all layers
    )

    if layer_indices is not None:
        logger.info(f"Running ablation for layer {layer_indices[0]} on tasks: {tasks}")
        fname = f"ablation_layer{layer_indices[0]:02d}"
        out_subdir = os.path.join(args.output_dir, "ablation_layers")
    else:
        logger.info(f"Running ablation for ALL layers on tasks: {tasks}")
        fname = "ablation_all"
        out_subdir = args.output_dir

    dataloaders = build_task_dataloaders(
        tasks=tasks,
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        cache_dir=args.hf_cache_dir,
        num_workers=args.num_workers,
    )

    analyzer = AblationImportanceAnalyzer(model)
    result = analyzer.run(
        task_dataloaders=dataloaders,
        device=device,
        layer_indices=layer_indices,
        layer_types=get_layer_types(model),
        model_name=args.model_name_or_path,
        n_samples=args.n_examples,
        seq_len=args.seq_len,
    )

    # Save immediately (important for per-layer cluster jobs)
    save_result(result, out_subdir, fname)

    return result, fname


def run_merge(args):
    """Merge per-layer ablation JSON files into a single importance matrix."""
    merge_dir = args.merge_dir
    if merge_dir is None or not os.path.isdir(merge_dir):
        raise ValueError(f"--merge_dir must point to a directory containing ablation JSON files. Got: {merge_dir}")

    json_files = sorted(Path(merge_dir).glob("ablation_layer*.json"))
    if not json_files:
        raise FileNotFoundError(f"No ablation_layer*.json files found in {merge_dir}")

    logger.info(f"Merging {len(json_files)} partial results from {merge_dir}")
    partial = [load_result(str(p)) for p in json_files]
    merged = merge_ablation_results(partial)

    save_result(merged, args.output_dir, "ablation_merged")
    logger.info("Merge complete.")
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve task group shorthand
    if args.task_group is not None:
        args.tasks = TASK_GROUPS[args.task_group]
        logger.info(f"Expanded task group '{args.task_group}' → {args.tasks}")

    # Build run name
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_short = Path(args.model_name_or_path).name if args.model_name_or_path else "model"
    if args.mode == "run":
        layer_suffix = (
            f"_L{args.ablation_layer_idx:02d}" if args.ablation_layer_idx is not None else ""
        )
        run_name = args.run_name or f"{model_short}_{args.method}{layer_suffix}_{ts}"
    else:
        run_name = args.run_name or f"{model_short}_merge_{ts}"

    logger.info(f"Run name: {run_name}")

    # ── Merge mode ────────────────────────────────────────────────────────
    if args.mode == "merge":
        result = run_merge(args)
        run = init_wandb(args, run_name)
        if run is not None:
            log_importance_to_wandb(run, result)
            run.finish()
        return

    # ── Run mode ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No CUDA device found. Running on CPU (very slow).")

    if args.model_name_or_path is None:
        parser.error("--model_name_or_path is required in run mode")
    attn_impl = args.attn_implementation
    if args.method == "gradient" and attn_impl == "flash_attention_2":
        logger.warning("flash_attention_2 does not support retain_grad; using sdpa for gradient mode.")
        attn_impl = "sdpa"
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, dtype=args.dtype, attn_implementation=attn_impl,
        cache_dir=args.hf_cache_dir,
    )

    run = init_wandb(args, run_name)

    t0 = time.time()

    if args.method == "gradient":
        result, fname = run_gradient(args, model, tokenizer, device, run_name)
        save_result(result, args.output_dir, f"gradient_{run_name}")

    elif args.method == "ablation":
        result, fname = run_ablation(args, model, tokenizer, device, run_name)
        # Already saved inside run_ablation for per-layer jobs; save full result too
        if args.ablation_layer_idx is None:
            save_result(result, args.output_dir, f"ablation_full_{run_name}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed/60:.1f} min. Results in {args.output_dir}")

    # ── W&B logging ───────────────────────────────────────────────────────
    if run is not None:
        run.summary["elapsed_minutes"] = elapsed / 60
        log_importance_to_wandb(run, result)
        run.finish()

    # ── Print quick summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Method     : {result.method}")
    print(f"Model      : {result.model_name}")
    print(f"Layers     : {result.n_layers}")
    print(f"Tasks      : {result.task_names}")
    print()
    print("Top-3 layers per task (normalized importance):")
    for task in result.task_names:
        top3 = result.top_layers_for_task(task, k=3)
        entries = ", ".join(
            f"L{l}[{result.layer_types[l]}]={s:.3f}" for l, s in top3
        )
        print(f"  {task:30s}: {entries}")
    print("=" * 60)


if __name__ == "__main__":
    main()
