"""
Causal validation: does the importance ranking predict where adaptation helps?

For each task, LoRA adapters are trained on only the top-k or only the bottom-k
layers of an importance matrix (same rank, steps, lr, data), and evaluated on a
separately sampled set (perplexity: seed 1, MCQ: seed 2; training uses seed 42):
  - winogrande / arc_challenge / mmlu / piqa / gpqa : multiple-choice accuracy
    (log-likelihood scoring)
  - humaneval / mbpp / gsm8k / squad / longbench    : perplexity on gold answers

Usage:
    python experiments/run_causal_validation.py \
        --importance_json "results/lora_delta/qwen3-8b/lora_delta_*.json" \
        --model_name_or_path Qwen/Qwen3-8B \
        --tasks humaneval winogrande --k 6 --n_train_steps 300 --lr 1e-4 \
        --output_dir results/causal_validation/qwen3-8b

    # Uniform (all-layer) LoRA reference:
    python experiments/run_causal_validation.py --uniform \
        --model_name_or_path Qwen/Qwen3-8B --tasks humaneval winogrande \
        --output_dir results/causal_validation/qwen3-8b

State-spaces Mamba checkpoints are held in float32 here (weights rounded to
--dtype), matching the paper runs.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from layerimp.data import build_task_dataloaders
from layerimp.finetune import (
    MCQ_TASKS,
    apply_lora_to_layers,
    eval_mcq_accuracy,
    eval_perplexity,
    finetune,
    remove_lora,
)
from layerimp.models import (
    default_target_modules,
    get_decoder_layers,
    load_from_args,
)
from layerimp.utils import init_wandb, load_result, resolve_path, setup_logging

setup_logging()
logger = logging.getLogger("causal_validation")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Causal accuracy validation of layer importance matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--importance_json", type=str, default=None,
                   help="Path to the LoRA delta importance JSON (glob OK). "
                        "Required unless --uniform is set.")
    p.add_argument("--uniform", action="store_true",
                   help="Run uniform (all-layer) LoRA baseline instead of top-k vs bottom-k. "
                        "When set, --importance_json is not required.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--hf_cache_dir", type=str, default=None)

    p.add_argument("--tasks", nargs="+", default=["humaneval", "winogrande"],
                   help="Tasks to validate. Default: humaneval winogrande (most divergent pair).")
    p.add_argument("--k", type=int, default=6,
                   help="Number of top/bottom layers to use for LoRA. "
                        "~6 ≈ top-17%% for Qwen3-8B (36 layers), top-21%% for Llama-3B (28 layers).")

    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--n_train_steps", type=int, default=300,
                   help="Fine-tuning steps for each LoRA-on-subset run.")
    p.add_argument("--n_eval_steps", type=int, default=50,
                   help="Batches used for evaluation (perplexity / MCQ accuracy).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target_modules", nargs="+", default=None,
                   help="LoRA target modules (default: architecture-specific, see "
                        "layerimp.models.default_target_modules).")

    p.add_argument("--n_train_examples", type=int, default=256,
                   help="Examples for fine-tuning (train split, seed=42).")
    p.add_argument("--n_eval_examples", type=int, default=256,
                   help="Examples for evaluation (eval split, seed=1).")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce GPU memory at the cost of speed.")

    p.add_argument("--output_dir", type=str, default="results/causal_validation")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", nargs="*", default=[])

    return p


# ---------------------------------------------------------------------------
# Single (task, layers) validation run
# ---------------------------------------------------------------------------

def run_one(
    base_model: nn.Module,
    tokenizer,
    task_name: str,
    layer_indices: List[int],
    label: str,  # "top_k" or "bottom_k"
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    args,
) -> Dict:
    """
    Fine-tune LoRA restricted to layer_indices on task_name, then evaluate.

    Returns a dict with loss/ppl/accuracy metrics.
    """
    logger.info(f"  [{label}] Layers: {layer_indices}")

    # Apply LoRA to the subset of layers
    peft_model = apply_lora_to_layers(
        base_model,
        layer_indices=layer_indices,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )

    # Fine-tune
    train_losses = finetune(
        peft_model,
        train_loader,
        device,
        n_steps=args.n_train_steps,
        lr=args.lr,
        desc=f"[{label}] {task_name}",
    )

    # Evaluate perplexity on held-out eval split
    ppl = eval_perplexity(peft_model, eval_loader, device, n_steps=args.n_eval_steps)

    # Evaluate MCQ accuracy if applicable
    accuracy = None
    if task_name in MCQ_TASKS:
        accuracy = eval_mcq_accuracy(
            peft_model,
            tokenizer,
            task_name=task_name,
            n_examples=args.n_eval_examples,
            seed=2,  # seed 2 = independent from train (42) and eval ppl (1)
            device=device,
            cache_dir=args.hf_cache_dir,
            seq_len=args.seq_len,
        )
        logger.info(f"  [{label}] {task_name}: ppl={ppl:.2f}, accuracy={accuracy:.4f}")
    else:
        logger.info(f"  [{label}] {task_name}: ppl={ppl:.2f}")

    # Remove LoRA from the base model so the next run starts from scratch
    remove_lora(peft_model)
    del peft_model
    torch.cuda.empty_cache()

    # Re-freeze all parameters in base_model for next iteration
    for p in base_model.parameters():
        p.requires_grad_(False)

    return {
        "label": label,
        "layers": layer_indices,
        "final_train_loss": float(np.mean(train_losses[-20:])),
        "perplexity": ppl,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Validate mode ─────────────────────────────────────────────────────
    if not args.uniform and args.importance_json is None:
        parser.error("Either --importance_json or --uniform must be provided.")

    importance = None
    json_path = None

    if not args.uniform:
        json_path = resolve_path(args.importance_json)  # glob OK: newest match
        logger.info(f"Loading importance result: {json_path}")
        importance = load_result(json_path)

        logger.info(f"  Model: {importance.model_name}, {importance.n_layers} layers, "
                    f"{importance.n_tasks} tasks, method={importance.method}")

        # ── Validate tasks ─────────────────────────────────────────────────
        for task in args.tasks:
            if task not in importance.task_names:
                raise ValueError(
                    f"Task '{task}' not in importance result. "
                    f"Available: {importance.task_names}"
                )

        # ── Validate k ────────────────────────────────────────────────────
        if args.k >= importance.n_layers // 2:
            raise ValueError(
                f"k={args.k} must be < n_layers//2 = {importance.n_layers//2} "
                f"to avoid overlap between top-k and bottom-k."
            )

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No CUDA device — running on CPU (very slow).")

    # ── Load model ────────────────────────────────────────────────────────
    base_model, tokenizer = load_from_args(args, upcast_state_spaces=True)

    # ── LoRA target modules ───────────────────────────────────────────────
    if args.target_modules is None:
        args.target_modules = default_target_modules(base_model)
    if getattr(base_model.config, "model_type", "") in ("mamba", "mamba2"):
        # PEFT does not support LoRA on Mamba's out_proj / conv1d.
        args.target_modules = [m for m in args.target_modules if m not in {"out_proj", "conv1d"}]
    logger.info(f"LoRA target modules: {args.target_modules}")

    # ── Run name / output dir ─────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_short = Path(args.model_name_or_path).name
    if args.uniform:
        run_name = args.run_name or f"{model_short}_uniform_lora_{ts}"
    else:
        run_name = args.run_name or f"{model_short}_causal_val_k{args.k}_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────────
    wb_run = init_wandb(
        args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        tags=["uniform_lora" if args.uniform else "causal_validation"] + (args.wandb_tags or []),
        config={**vars(args), "importance_json": json_path},
    )

    # ── Uniform LoRA mode ─────────────────────────────────────────────────
    if args.uniform:
        n_layers = len(get_decoder_layers(base_model))
        all_layer_indices = list(range(n_layers))
        logger.info(f"Uniform LoRA mode: applying LoRA to all {n_layers} layers.")

        all_results = []
        for task_name in args.tasks:
            logger.info(f"\n{'='*60}\nTask: {task_name} [uniform]\n{'='*60}")

            train_loaders = build_task_dataloaders(
                tasks=[task_name], tokenizer=tokenizer,
                n_examples=args.n_train_examples, seq_len=args.seq_len,
                batch_size=args.batch_size, seed=42,
                cache_dir=args.hf_cache_dir, num_workers=args.num_workers,
            )
            eval_loaders = build_task_dataloaders(
                tasks=[task_name], tokenizer=tokenizer,
                n_examples=args.n_eval_examples, seq_len=args.seq_len,
                batch_size=args.batch_size, seed=1,
                cache_dir=args.hf_cache_dir, num_workers=args.num_workers,
            )
            train_loader = train_loaders[task_name]
            eval_loader = eval_loaders[task_name]

            base_ppl = eval_perplexity(base_model, eval_loader, device, n_steps=args.n_eval_steps)
            base_acc = None
            if task_name in MCQ_TASKS:
                base_acc = eval_mcq_accuracy(
                    base_model, tokenizer, task_name,
                    n_examples=args.n_eval_examples, seed=2, device=device,
                    cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
                )
            logger.info(f"  [baseline] {task_name}: ppl={base_ppl:.2f}"
                        + (f", accuracy={base_acc:.4f}" if base_acc is not None else ""))

            uniform_result = run_one(
                base_model, tokenizer, task_name, all_layer_indices, "uniform",
                train_loader, eval_loader, device, args
            )

            logger.info(
                f"\n  SUMMARY for {task_name} [uniform]:\n"
                f"    Baseline   : ppl={base_ppl:.2f}"
                + (f", acc={base_acc:.4f}" if base_acc is not None else "") + "\n"
                f"    Uniform LoRA: ppl={uniform_result['perplexity']:.2f}"
                + (f", acc={uniform_result['accuracy']:.4f}" if uniform_result['accuracy'] is not None else "")
            )

            task_record = {
                "task": task_name,
                "mode": "uniform",
                "n_layers": n_layers,
                "baseline": {"perplexity": base_ppl, "accuracy": base_acc},
                "uniform": uniform_result,
            }
            all_results.append(task_record)

            if wb_run is not None:
                wb_run.log({
                    f"{task_name}/baseline_ppl": base_ppl,
                    f"{task_name}/uniform_ppl": uniform_result["perplexity"],
                })

        # Save results
        out_path = Path(args.output_dir) / f"{run_name}.json"
        payload = {
            "run_name": run_name,
            "model": args.model_name_or_path,
            "mode": "uniform",
            "n_layers": n_layers,
            "n_train_steps": args.n_train_steps,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "results": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"\nUniform LoRA results saved → {out_path}")
        if wb_run is not None:
            wb_run.finish()
        return

    # ── Per-task loop (top-k vs bottom-k) ─────────────────────────────────
    all_results = []

    for task_name in args.tasks:
        logger.info(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")

        # Get top-k and bottom-k layer indices from importance matrix
        t_idx = importance.task_names.index(task_name)
        scores = importance.matrix_norm[:, t_idx]
        ranked = np.argsort(scores)[::-1]          # descending
        top_k_layers = sorted(ranked[:args.k].tolist())
        bottom_k_layers = sorted(ranked[-args.k:].tolist())

        logger.info(f"  Top-{args.k}    layers: {top_k_layers}")
        logger.info(f"  Bottom-{args.k} layers: {bottom_k_layers}")
        logger.info(f"  Score range — top min: {scores[ranked[args.k-1]]:.4f}, "
                    f"bottom max: {scores[ranked[-args.k]]:.4f}")

        # Build dataloaders: train seed=42, eval seed=1 (different examples)
        train_loaders = build_task_dataloaders(
            tasks=[task_name],
            tokenizer=tokenizer,
            n_examples=args.n_train_examples,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=42,
            cache_dir=args.hf_cache_dir,
            num_workers=args.num_workers,
        )
        eval_loaders = build_task_dataloaders(
            tasks=[task_name],
            tokenizer=tokenizer,
            n_examples=args.n_eval_examples,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=1,  # held-out split
            cache_dir=args.hf_cache_dir,
            num_workers=args.num_workers,
        )
        train_loader = train_loaders[task_name]
        eval_loader = eval_loaders[task_name]

        # Baseline: no fine-tuning
        base_ppl = eval_perplexity(base_model, eval_loader, device, n_steps=args.n_eval_steps)
        base_acc = None
        if task_name in MCQ_TASKS:
            base_acc = eval_mcq_accuracy(
                base_model, tokenizer, task_name,
                n_examples=args.n_eval_examples, seed=2, device=device,
                cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
            )
        logger.info(f"  [baseline] {task_name}: ppl={base_ppl:.2f}"
                    + (f", accuracy={base_acc:.4f}" if base_acc is not None else ""))

        # Top-k run
        top_result = run_one(
            base_model, tokenizer, task_name, top_k_layers, "top_k",
            train_loader, eval_loader, device, args
        )
        torch.cuda.empty_cache()

        # Bottom-k run
        bot_result = run_one(
            base_model, tokenizer, task_name, bottom_k_layers, "bottom_k",
            train_loader, eval_loader, device, args
        )
        torch.cuda.empty_cache()

        # Summarise
        ppl_gap = bot_result["perplexity"] - top_result["perplexity"]
        acc_gap = None
        if top_result["accuracy"] is not None and bot_result["accuracy"] is not None:
            acc_gap = top_result["accuracy"] - bot_result["accuracy"]

        logger.info(
            f"\n  SUMMARY for {task_name}:\n"
            f"    Baseline    : ppl={base_ppl:.2f}"
            + (f", acc={base_acc:.4f}" if base_acc is not None else "") + "\n"
            f"    Top-{args.k} LoRA : ppl={top_result['perplexity']:.2f}"
            + (f", acc={top_result['accuracy']:.4f}" if top_result['accuracy'] is not None else "") + "\n"
            f"    Bot-{args.k} LoRA : ppl={bot_result['perplexity']:.2f}"
            + (f", acc={bot_result['accuracy']:.4f}" if bot_result['accuracy'] is not None else "") + "\n"
            f"    PPL gap (bot-top) : {ppl_gap:+.2f}  "
            + (f"ACC gap (top-bot) : {acc_gap:+.4f}" if acc_gap is not None else "")
        )

        task_record = {
            "task": task_name,
            "k": args.k,
            "top_k_layers": top_k_layers,
            "bottom_k_layers": bottom_k_layers,
            "baseline": {"perplexity": base_ppl, "accuracy": base_acc},
            "top_k": top_result,
            "bottom_k": bot_result,
            "ppl_gap_bottom_minus_top": ppl_gap,
            "acc_gap_top_minus_bottom": acc_gap,
            "hypothesis_supported": (
                ppl_gap > 0  # top-k gives lower (better) ppl
                if acc_gap is None else acc_gap > 0  # top-k gives higher accuracy
            ),
        }
        all_results.append(task_record)

        if wb_run is not None:
            try:
                wb_run.log({
                    f"{task_name}/baseline_ppl": base_ppl,
                    f"{task_name}/top_k_ppl": top_result["perplexity"],
                    f"{task_name}/bottom_k_ppl": bot_result["perplexity"],
                    f"{task_name}/ppl_gap": ppl_gap,
                    **(
                        {
                            f"{task_name}/baseline_acc": base_acc,
                            f"{task_name}/top_k_acc": top_result["accuracy"],
                            f"{task_name}/bottom_k_acc": bot_result["accuracy"],
                            f"{task_name}/acc_gap": acc_gap,
                        }
                        if task_name in MCQ_TASKS else {}
                    ),
                })
            except Exception as e:
                logger.warning(f"W&B log failed: {e}")

    # ── Save final results ─────────────────────────────────────────────────
    output = {
        "run_name": run_name,
        "model": args.model_name_or_path,
        "importance_json": json_path,
        "k": args.k,
        "n_train_steps": args.n_train_steps,
        "lr": args.lr,
        "n_layers": importance.n_layers,
        "results": all_results,
        "hypothesis_supported_all": all(r["hypothesis_supported"] for r in all_results),
    }
    out_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved results → {out_path}")

    if wb_run is not None:
        wb_run.summary["hypothesis_supported_all"] = output["hypothesis_supported_all"]
        wb_run.finish()

    # ── Print final table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"CAUSAL VALIDATION RESULTS — {model_short} | k={args.k}")
    print("=" * 70)
    print(f"{'Task':<25} {'Baseline':>10} {'Top-k':>10} {'Bottom-k':>10} {'Gap':>10} {'Supported':>10}")
    print("-" * 70)
    for r in all_results:
        if r["acc_gap_top_minus_bottom"] is not None:
            bl = f"{r['baseline']['accuracy']:.4f}"
            tk = f"{r['top_k']['accuracy']:.4f}"
            bk = f"{r['bottom_k']['accuracy']:.4f}"
            gap = f"{r['acc_gap_top_minus_bottom']:+.4f}"
            metric = "ACC"
        else:
            bl = f"{r['baseline']['perplexity']:.1f}"
            tk = f"{r['top_k']['perplexity']:.1f}"
            bk = f"{r['bottom_k']['perplexity']:.1f}"
            gap = f"{r['ppl_gap_bottom_minus_top']:+.1f}"
            metric = "PPL"
        supported = "YES ✓" if r["hypothesis_supported"] else "NO ✗"
        print(f"{r['task']:<25} {bl:>10} {tk:>10} {bk:>10} {gap:>10} {supported:>10}  [{metric}]")
    print("=" * 70)
    print(f"Overall hypothesis supported: {output['hypothesis_supported_all']}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
