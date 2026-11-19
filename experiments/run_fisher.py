"""
Diagonal Fisher information per layer (curvature proxy).

For each task, computes the diagonal Fisher information aggregated per layer:

    F_l(task) = (1/N) sum_{x in D_task} sum_{i in layer l} (dL/d theta_i)^2

This is the expected squared gradient — an unbiased estimator of the Hessian
diagonal under the Fisher-Hessian equivalence at a local minimum, and a valid
curvature proxy everywhere.

Usage:
    python experiments/run_fisher.py --model_name_or_path Qwen/Qwen3-8B --model_slug qwen3-8b

Saved: {output_dir}/{model_slug}/fisher_per_layer.json
  {
    "model_name":       str,
    "task_names":       [10 tasks],
    "fisher_per_layer": [[F_0,...,F_{L-1}] per task],   # shape (n_tasks, n_layers)
    "n_layers":         int,
  }
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from layerimp.data import TaskDataRegistry
from layerimp.models import load_model_and_tokenizer

TASK_NAMES = [
    "mmlu", "gsm8k", "arc_challenge", "piqa",
    "humaneval", "mbpp", "squad", "winogrande", "gpqa", "longbench_single_doc",
]


# ── Layer index detection ──────────────────────────────────────────────────────

def _layer_idx(name: str) -> int | None:
    """Return the layer index from a parameter name, or None."""
    m = re.search(r'\.layers\.(\d+)\.', name)   # covers backbone.layers.N and model.layers.N
    if m:
        return int(m.group(1))
    m = re.search(r'\bh\.(\d+)\.', name)         # GPT-2 style
    if m:
        return int(m.group(1))
    return None


def _count_layers(model) -> int:
    return max(
        (_layer_idx(n) for n, _ in model.named_parameters()
         if _layer_idx(n) is not None),
        default=-1,
    ) + 1


# ── Fisher computation ─────────────────────────────────────────────────────────

def fisher_per_layer(
    model,
    tokenizer,
    task_name: str,
    n_layers: int,
    n_examples: int = 64,
    seq_len: int = 256,
    batch_size: int = 4,
    device: str = "cuda",
    cache_dir: str | None = None,
) -> list[float]:
    """Return [F_0, ..., F_{L-1}]: Fisher trace per layer averaged over n_examples."""
    loader = TaskDataRegistry(cache_dir=cache_dir)
    texts  = loader.load_texts(task_name, n_examples=n_examples, seed=42)

    accumulator = [0.0] * n_layers
    n_batches   = 0

    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if not batch:
            break

        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        ).to(device)

        input_ids = enc["input_ids"]
        labels    = input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        model.zero_grad()
        with torch.enable_grad():
            out  = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if loss is None or not loss.requires_grad:
                print(f"    [warn] no grad on loss for task={task_name} batch={i}")
                continue
            loss.backward()

        for pname, param in model.named_parameters():
            if param.grad is None:
                continue
            li = _layer_idx(pname)
            if li is not None and li < n_layers:
                accumulator[li] += param.grad.detach().pow(2).sum().item()

        n_batches += 1
        if n_batches * batch_size >= n_examples:
            break

    if n_batches == 0:
        return [float("nan")] * n_layers

    return [v / n_batches for v in accumulator]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--model_slug",         required=True)
    ap.add_argument("--n_examples", type=int, default=64)
    ap.add_argument("--seq_len",    type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype",      default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--output_dir", default="results/fisher")
    args = ap.parse_args()

    out_dir  = Path(args.output_dir) / args.model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fisher_per_layer.json"

    if out_path.exists():
        print(f"Already exists: {out_path}  — delete to rerun.")
        return

    print(f"Loading {args.model_name_or_path} (dtype={args.dtype}) ...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, dtype=args.dtype, cache_dir=args.hf_cache_dir,
        device_map=args.device, pad_to_eos=False,
    )

    # Enable grads for all parameters
    for p in model.parameters():
        p.requires_grad_(True)

    n_layers = _count_layers(model)
    print(f"Detected {n_layers} layers.", flush=True)

    result = {
        "model_name":       args.model_name_or_path,
        "model_slug":       args.model_slug,
        "n_examples":       args.n_examples,
        "seq_len":          args.seq_len,
        "n_layers":         n_layers,
        "task_names":       TASK_NAMES,
        "fisher_per_layer": [],
    }

    for task in TASK_NAMES:
        print(f"  [{task}] ...", end=" ", flush=True)
        try:
            f = fisher_per_layer(
                model, tokenizer, task, n_layers,
                n_examples=args.n_examples,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=args.device,
                cache_dir=args.hf_cache_dir,
            )
            max_f = max(v for v in f if np.isfinite(v))
            max_l = next(i for i, v in enumerate(f) if v == max_f)
            print(f"ok | F_max={max_f:.3e} at L{max_l}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            f = [float("nan")] * n_layers
        result["fisher_per_layer"].append(f)

    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
