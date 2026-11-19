"""
Head-level ablation (transformers only).

For each attention head h in each layer l, temporarily zeros the head's
contribution to the output projection and measures the per-task loss delta.

  I_head(h, l, t) = L(model with head h of layer l zeroed) - L(baseline)

This reveals whether terminal layers have individually-small head impacts
(consistent with redundancy / low ablation importance at layer level) while
individually large head impacts appear in early layers.

Zeroing mechanism: o_proj.weight[:, h*d_head:(h+1)*d_head] = 0
This cleanly zeros head h's contribution to the output without touching other
heads. Works for any standard transformer with a single o_proj linear layer.

Supports GQA (Qwen3, Llama 3.x) where num_kv_heads < num_q_heads — the query
heads still all contribute through o_proj equally.

Parallelization: --layer_idx runs a single layer (all heads), designed for
SLURM array jobs. Use merge mode after all layers complete.

Usage:
    # Single layer (SLURM array):
    python experiments/run_head_ablation.py \\
        --model_name_or_path Qwen/Qwen3-8B \\
        --layer_idx 5 \\
        --output_dir results/head_ablation/qwen3-8b

    # Merge after all layers complete:
    python experiments/run_head_ablation.py \\
        --mode merge \\
        --merge_dir results/head_ablation/qwen3-8b/layers \\
        --output_dir results/head_ablation/qwen3-8b
"""

import argparse
import json
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from layerimp.data import ALL_TASKS, build_task_dataloaders
from layerimp.models import load_from_args
from layerimp.utils import setup_logging

setup_logging()
logger = logging.getLogger("head_ablation")


# ---------------------------------------------------------------------------
# Transformer layer / attention module discovery
# ---------------------------------------------------------------------------

def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    inner = model.model if hasattr(model, "model") else model
    for attr in ("layers", "h", "blocks"):
        if hasattr(inner, attr):
            return getattr(inner, attr)
    raise ValueError("Cannot locate transformer layers in model.")


def get_attention_module(layer: nn.Module) -> Optional[nn.Module]:
    """Return the self-attention submodule from a decoder layer."""
    for attr in ("self_attn", "attention", "attn", "self_attention"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def get_o_proj(attn: nn.Module) -> Optional[nn.Linear]:
    """Return the output projection linear layer from an attention module."""
    for attr in ("o_proj", "out_proj", "c_proj", "dense"):
        if hasattr(attn, attr):
            mod = getattr(attn, attr)
            if isinstance(mod, nn.Linear):
                return mod
    return None


def get_head_config(model: nn.Module) -> Tuple[int, int]:
    """Return (num_attention_heads, head_dim) from the model config."""
    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads",
               getattr(cfg, "n_head",
               getattr(cfg, "num_heads", None)))
    d_model = getattr(cfg, "hidden_size",
               getattr(cfg, "d_model",
               getattr(cfg, "n_embd", None)))
    if n_heads is None or d_model is None:
        raise ValueError(f"Cannot infer n_heads or d_model from config: {cfg}")
    # Some models (e.g. Qwen3-0.6B/4B/32B) set head_dim != hidden_size // n_heads.
    head_dim = getattr(cfg, "head_dim", None) or d_model // n_heads
    return n_heads, head_dim


# ---------------------------------------------------------------------------
# Head zeroing context manager
# ---------------------------------------------------------------------------

@contextmanager
def zero_head(o_proj: nn.Linear, head_idx: int, head_dim: int):
    """
    Temporarily zero the columns of o_proj.weight corresponding to head_idx.
    Those columns are: o_proj.weight[:, head_idx*head_dim : (head_idx+1)*head_dim]

    Restores exact original values on exit even if an exception occurs.
    """
    start = head_idx * head_dim
    end   = start + head_dim
    orig = o_proj.weight.data[:, start:end].clone()
    try:
        o_proj.weight.data[:, start:end] = 0.0
        yield
    finally:
        o_proj.weight.data[:, start:end] = orig


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        flat_log = shift_logits.view(-1, shift_logits.size(-1))
        flat_lbl = shift_labels.view(-1)
        mask = flat_lbl != -100
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(flat_log, flat_lbl, ignore_index=-100, reduction="sum")
        total_loss   += loss.item()
        total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Head-level ablation importance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["run", "merge"], default="run")
    p.add_argument("--model_name_or_path", type=str, required=False, default=None)
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--hf_cache_dir", type=str, default=None)

    p.add_argument("--layer_idx", type=int, default=None,
                   help="Layer index to ablate (all heads). For SLURM array jobs.")
    p.add_argument("--tasks", nargs="+",
                   default=ALL_TASKS)
    p.add_argument("--n_examples", type=int, default=256)
    p.add_argument("--seq_len",    type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_batches",  type=int, default=50,
                   help="Batches for loss evaluation per head.")
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--output_dir", type=str, default="results/head_ablation")
    p.add_argument("--merge_dir",  type=str, default=None,
                   help="Directory of per-layer JSONs to merge (merge mode).")
    return p


# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------

def merge(merge_dir: str, output_dir: str):
    """Merge per-layer JSON files into a single I_head[n_heads, n_layers, n_tasks] array."""
    files = sorted(Path(merge_dir).glob("head_ablation_layer*.json"))
    if not files:
        raise FileNotFoundError(f"No head_ablation_layer*.json in {merge_dir}")

    all_layers = []
    task_names = None
    model_name = None

    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if task_names is None:
            task_names = d["task_names"]
            model_name = d["model"]
        all_layers.append(d)

    # Sort by layer_idx
    all_layers.sort(key=lambda x: x["layer_idx"])
    n_layers = max(d["layer_idx"] for d in all_layers) + 1
    if n_layers != len(all_layers):
        logger.warning(f"Only {len(all_layers)}/{n_layers} layers found; missing layers are zero.")
    n_heads  = len(all_layers[0]["heads"])
    n_tasks  = len(task_names)

    # Build I_head[n_layers, n_heads, n_tasks]
    importance = np.zeros((n_layers, n_heads, n_tasks), dtype=np.float32)
    baselines  = np.zeros((n_layers, n_tasks), dtype=np.float32)

    for layer_d in all_layers:
        l = layer_d["layer_idx"]
        for task_idx, task in enumerate(task_names):
            baselines[l, task_idx] = layer_d["baselines"][task]
        for h, head_d in enumerate(layer_d["heads"]):
            for task_idx, task in enumerate(task_names):
                importance[l, h, task_idx] = head_d["delta"][task]

    # Per-layer sum of head importances (for comparison with layer ablation)
    head_sum = importance.sum(axis=1)   # [n_layers, n_tasks]

    os.makedirs(output_dir, exist_ok=True)
    out = {
        "model":      model_name,
        "n_layers":   n_layers,
        "n_heads":    n_heads,
        "n_tasks":    n_tasks,
        "task_names": task_names,
        # [n_layers, n_heads, n_tasks] — raw importance (loss delta per head)
        "importance": importance.tolist(),
        # [n_layers, n_tasks] — sum of head importances per layer
        "head_sum":   head_sum.tolist(),
        # [n_layers, n_tasks] — baseline perplexity (no ablation)
        "baselines":  baselines.tolist(),
    }
    np.save(os.path.join(output_dir, "head_ablation.npy"), importance)
    with open(os.path.join(output_dir, "head_ablation_merged.json"), "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Merged {n_layers} layers × {n_heads} heads × {n_tasks} tasks")
    logger.info(f"Saved → {output_dir}/head_ablation_merged.json")

    # Quick summary: top-5 layers by mean head-sum importance
    mean_head_sum = head_sum.mean(axis=1)
    top5 = np.argsort(mean_head_sum)[::-1][:5]
    logger.info("Top-5 layers by mean head-sum importance:")
    for l in top5:
        logger.info(f"  L{l:03d}: {mean_head_sum[l]:.4f}")


# ---------------------------------------------------------------------------
# Main run mode
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "merge":
        merge(args.merge_dir, args.output_dir)
        return

    # ── Load model ────────────────────────────────────────────────────────
    if args.model_name_or_path is None:
        parser.error("--model_name_or_path is required in run mode")
    model, tokenizer = load_from_args(args)
    device = next(model.parameters()).device

    # ── Discover architecture ─────────────────────────────────────────────
    layers   = get_transformer_layers(model)
    n_layers = len(layers)
    n_heads, head_dim = get_head_config(model)

    logger.info(f"Architecture: {n_layers} layers, {n_heads} heads, head_dim={head_dim}")

    # Determine which layers to process
    if args.layer_idx is not None:
        layer_indices = [args.layer_idx]
    else:
        layer_indices = list(range(n_layers))

    # ── Load data ─────────────────────────────────────────────────────────
    loaders = build_task_dataloaders(
        tasks=args.tasks,
        tokenizer=tokenizer,
        n_examples=args.n_examples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=42,
        cache_dir=args.hf_cache_dir,
        num_workers=args.num_workers,
    )
    task_names = list(loaders.keys())

    # ── Per-layer loop ────────────────────────────────────────────────────
    layers_dir = os.path.join(args.output_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    for l_idx in layer_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {l_idx}/{n_layers-1}")
        logger.info(f"{'='*60}")

        layer = layers[l_idx]
        attn  = get_attention_module(layer)
        if attn is None:
            logger.warning(f"  Layer {l_idx}: no attention module found — skipping")
            continue
        o_proj = get_o_proj(attn)
        if o_proj is None:
            logger.warning(f"  Layer {l_idx}: no o_proj found — skipping")
            continue

        # Baseline perplexity (no ablation)
        baselines = {}
        for task in task_names:
            ppl = compute_loss(model, loaders[task], device, args.n_batches)
            baselines[task] = ppl
            logger.info(f"  [baseline] {task}: ppl={ppl:.3f}")

        # Per-head ablation
        heads_results = []
        for h in tqdm(range(n_heads), desc=f"L{l_idx:03d} heads"):
            head_delta = {}
            with zero_head(o_proj, h, head_dim):
                for task in task_names:
                    ppl_ablated = compute_loss(model, loaders[task], device, args.n_batches)
                    head_delta[task] = round(ppl_ablated - baselines[task], 5)
            heads_results.append({"head_idx": h, "delta": head_delta})

        # Save per-layer result
        layer_out = {
            "model":      args.model_name_or_path,
            "layer_idx":  l_idx,
            "n_heads":    n_heads,
            "head_dim":   head_dim,
            "task_names": task_names,
            "baselines":  {t: round(baselines[t], 5) for t in task_names},
            "heads":      heads_results,
        }
        out_path = os.path.join(layers_dir, f"head_ablation_layer{l_idx:03d}.json")
        with open(out_path, "w") as f:
            json.dump(layer_out, f, indent=2)
        logger.info(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
