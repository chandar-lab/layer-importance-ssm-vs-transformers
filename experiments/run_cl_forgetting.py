"""
Continual-learning forgetting: does layer importance predict where forgetting happens?

Protocol:
  1. Load layer importance JSON (LoRA delta or ablation).
  2. For each (task_A, task_B) pair and each layer selection method:
       a. Apply LoRA restricted to selected layers.
       b. Fine-tune on task_A for n_train_steps.
       c. Evaluate ppl on task_A eval split (pre-B).
       d. Continue fine-tuning on task_B (NO LoRA reset) for n_train_steps.
       e. Evaluate ppl on task_A eval split (post-B).
       f. Backward transfer = ppl_A_post / ppl_A_pre  (>1 = forgetting).
  3. Layer selection methods: top_k, bottom_k, random_k, plus an all-layer
     EWC baseline.
  4. Save results JSON.

Usage example::

    python experiments/run_cl_forgetting.py \\
        --importance_json "results/lora_delta/qwen3-8b/lora_delta_*.json" \\
        --model_name_or_path Qwen/Qwen3-8B \\
        --task_pairs humaneval winogrande  gsm8k piqa  mbpp winogrande \\
        --k 6 --n_train_steps 30 --lr 1e-4 \\
        --batch_size 2 --gradient_checkpointing \\
        --output_dir results/cl_forgetting/qwen3-8b
"""

import argparse
import json
import logging
import os
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from layerimp.data import build_task_dataloaders
from layerimp.finetune import (
    MCQ_TASKS,
    apply_lora_to_layers,
    eval_mcq_accuracy,
    eval_perplexity,
    finetune,
    remove_lora,
)
from layerimp.models import default_target_modules, is_mamba, is_rwkv, load_from_args
from layerimp.utils import load_result, resolve_path, setup_logging

setup_logging()
logger = logging.getLogger("cl_forgetting")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Continual-learning forgetting vs. layer importance")

    p.add_argument("--importance_json", type=str, required=True,
                   help="Path to LoRA delta importance JSON (glob OK).")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--hf_cache_dir", type=str, default=None)

    p.add_argument("--task_pairs", nargs="+", required=True,
                   help="Even-length list of task pairs: 'A1 B1 A2 B2 ...' "
                        "Each consecutive pair (A, B) is one forgetting run.")
    p.add_argument("--target_modules", nargs="+", default=None,
                   help="LoRA target modules. Default: attention + MLP projections for "
                        "transformers, layerimp.models.default_target_modules otherwise.")
    p.add_argument("--k", type=int, default=6,
                   help="Number of top/bottom layers to select.")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--n_train_steps", type=int, default=30,
                   help="Steps for task_A fine-tuning (same repeated for task_B).")
    p.add_argument("--n_eval_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--n_train_examples", type=int, default=256)
    p.add_argument("--n_eval_examples", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--ewc_lambda", type=float, default=1000.0,
                   help="EWC regularization strength (λ). Only used when EWC method is active.")
    p.add_argument("--ewc_fisher_steps", type=int, default=100,
                   help="Batches used to estimate Fisher diagonal for EWC.")

    p.add_argument("--output_dir", type=str, default="results/cl_forgetting")
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# LoRA update overlap helpers
# ---------------------------------------------------------------------------

def _snapshot_lora(peft_model) -> torch.Tensor:
    """Flat CPU float32 vector of all LoRA parameter values."""
    parts = [p.detach().float().cpu().flatten()
             for n, p in peft_model.named_parameters()
             if "lora_" in n and p.requires_grad]
    return torch.cat(parts) if parts else torch.zeros(1)


def _update_overlap(snap_init, snap_A, snap_B) -> Dict:
    """
    Cosine similarity between the task-A LoRA update and the task-B LoRA update.

    delta_A = snap_A - snap_init  (what task-A training wrote)
    delta_B = snap_B - snap_A     (what task-B training wrote, starting from snap_A)

    High cosine  → tasks update the same directions → low conflict.
    Near-zero    → orthogonal updates → independent.
    Negative     → opposing updates → forgetting expected.
    """
    delta_A = snap_A - snap_init
    delta_B = snap_B - snap_A
    norm_A  = delta_A.norm().item()
    norm_B  = delta_B.norm().item()
    if norm_A < 1e-12 or norm_B < 1e-12:
        cosine = float("nan")
    else:
        cosine = (delta_A * delta_B).sum().item() / (norm_A * norm_B)
    return {"update_cosine": cosine, "norm_delta_A": norm_A, "norm_delta_B": norm_B}


# ---------------------------------------------------------------------------
# One forgetting run
# ---------------------------------------------------------------------------

def run_forgetting_pair(
    base_model: nn.Module,
    tokenizer,
    task_A: str,
    task_B: str,
    layer_indices: List[int],
    method_label: str,
    train_A_loader,
    train_B_loader,
    eval_A_loader,
    eval_B_loader,
    device: torch.device,
    args,
) -> Dict:
    """
    Fine-tune on A, measure ppl_A (+ acc_A if MCQ), continue on B, measure again.
    Also measures task_B metrics after B training to confirm learning happened.
    Returns forgetting metrics.
    """
    logger.info(f"  [{method_label}] {task_A}→{task_B}: layers={layer_indices}")

    # Apply LoRA to selected layers
    peft_model = apply_lora_to_layers(
        base_model,
        layer_indices=layer_indices,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    snap_init = _snapshot_lora(peft_model)

    # Phase 1: fine-tune on task_A
    finetune(peft_model, train_A_loader, device,
             n_steps=args.n_train_steps, lr=args.lr,
             desc=f"[{method_label}] FT-A({task_A})")
    snap_A = _snapshot_lora(peft_model)

    # Measure task_A metrics before B (PPL always; accuracy if MCQ)
    ppl_A_before = eval_perplexity(peft_model, eval_A_loader, device,
                                   n_steps=args.n_eval_steps)
    acc_A_before = None
    if task_A in MCQ_TASKS:
        acc_A_before = eval_mcq_accuracy(
            peft_model, tokenizer, task_A,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )
    _a_str = f"ppl={ppl_A_before:.3f}" + (f", acc={acc_A_before:.4f}" if acc_A_before is not None else "")
    logger.info(f"  [{method_label}] task_A before B: {_a_str}")

    # Measure task_B baseline (PPL always; accuracy if MCQ) — before B training
    ppl_B_before = eval_perplexity(peft_model, eval_B_loader, device,
                                   n_steps=args.n_eval_steps)
    acc_B_before = None
    if task_B in MCQ_TASKS:
        acc_B_before = eval_mcq_accuracy(
            peft_model, tokenizer, task_B,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )
    _b_str = f"ppl={ppl_B_before:.3f}" + (f", acc={acc_B_before:.4f}" if acc_B_before is not None else "")
    logger.info(f"  [{method_label}] task_B before B: {_b_str}")

    # Phase 2: continue fine-tuning on task_B (no LoRA reset)
    finetune(peft_model, train_B_loader, device,
             n_steps=args.n_train_steps, lr=args.lr,
             desc=f"[{method_label}] FT-B({task_B})")
    snap_B = _snapshot_lora(peft_model)
    overlap = _update_overlap(snap_init, snap_A, snap_B)

    # Measure task_A metrics after B (forgetting signal)
    ppl_A_after = eval_perplexity(peft_model, eval_A_loader, device,
                                  n_steps=args.n_eval_steps)
    acc_A_after = None
    if task_A in MCQ_TASKS:
        acc_A_after = eval_mcq_accuracy(
            peft_model, tokenizer, task_A,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )

    # Measure task_B metrics after B (learning signal)
    ppl_B_after = eval_perplexity(peft_model, eval_B_loader, device,
                                  n_steps=args.n_eval_steps)
    acc_B_after = None
    if task_B in MCQ_TASKS:
        acc_B_after = eval_mcq_accuracy(
            peft_model, tokenizer, task_B,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )

    # Forgetting: task_A degradation
    forgetting_ppl = ppl_A_after - ppl_A_before
    forgetting_ratio = ppl_A_after / ppl_A_before if ppl_A_before > 0 else float("nan")
    forgetting_acc = None
    if acc_A_before is not None and acc_A_after is not None:
        forgetting_acc = acc_A_after - acc_A_before  # negative = forgetting

    # Learning: task_B improvement
    learning_ppl = ppl_B_before - ppl_B_after  # positive = improved
    learning_acc = None
    if acc_B_before is not None and acc_B_after is not None:
        learning_acc = acc_B_after - acc_B_before  # positive = learned

    _a_after_str = f"ppl={ppl_A_after:.3f}" + (f", acc={acc_A_after:.4f}" if acc_A_after is not None else "")
    _b_after_str = f"ppl={ppl_B_after:.3f}" + (f", acc={acc_B_after:.4f}" if acc_B_after is not None else "")
    logger.info(f"  [{method_label}] task_A after  B: {_a_after_str}")
    logger.info(f"  [{method_label}] task_B after  B: {_b_after_str}")
    logger.info(f"  [{method_label}] Forgetting — Δppl={forgetting_ppl:+.3f} (ratio {forgetting_ratio:.3f})"
                + (f", Δacc={forgetting_acc:+.4f}" if forgetting_acc is not None else ""))
    logger.info(f"  [{method_label}] Learning   — Δppl={learning_ppl:+.3f}"
                + (f", Δacc={learning_acc:+.4f}" if learning_acc is not None else ""))

    # Clean up LoRA from base model
    remove_lora(peft_model)
    del peft_model
    torch.cuda.empty_cache()
    for p in base_model.parameters():
        p.requires_grad_(False)

    return {
        "method": method_label,
        "task_A": task_A,
        "task_B": task_B,
        "layers": layer_indices,
        # Task A (forgetting)
        "ppl_A_before": float(ppl_A_before),
        "ppl_A_after": float(ppl_A_after),
        "forgetting_delta_ppl": float(forgetting_ppl),
        "forgetting_ratio": float(forgetting_ratio),
        "acc_A_before": float(acc_A_before) if acc_A_before is not None else None,
        "acc_A_after": float(acc_A_after) if acc_A_after is not None else None,
        "forgetting_delta_acc": float(forgetting_acc) if forgetting_acc is not None else None,
        # Task B (learning)
        "ppl_B_before": float(ppl_B_before),
        "ppl_B_after": float(ppl_B_after),
        "learning_delta_ppl": float(learning_ppl),
        "acc_B_before": float(acc_B_before) if acc_B_before is not None else None,
        "acc_B_after": float(acc_B_after) if acc_B_after is not None else None,
        "learning_delta_acc": float(learning_acc) if learning_acc is not None else None,
        # Update overlap
        **overlap,
    }


# ---------------------------------------------------------------------------
# EWC helpers
# ---------------------------------------------------------------------------

def compute_ewc_fisher(peft_model, train_loader, device, n_steps: int = 100):
    """Diagonal Fisher over LoRA parameters estimated on the current task's data."""
    lora_named = [(n, p) for n, p in peft_model.named_parameters()
                  if "lora_" in n and p.requires_grad]
    fisher = {n: torch.zeros_like(p) for n, p in lora_named}

    peft_model.train()
    data_iter = cycle(train_loader)
    for _ in range(n_steps):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        peft_model.zero_grad()
        outputs = peft_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits  = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().float().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        loss.backward()

        for n, p in lora_named:
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)

    for n in fisher:
        fisher[n] /= n_steps
    peft_model.eval()
    return fisher


def finetune_ewc(peft_model, dataloader, device, n_steps, lr,
                 ewc_fisher, ewc_anchor, ewc_lambda, desc="finetune_ewc"):
    """Cross-entropy fine-tuning with EWC penalty on LoRA parameters."""
    lora_params = [p for n, p in peft_model.named_parameters() if "lora_" in n]
    optimizer   = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)

    peft_model.train()
    data_iter = cycle(dataloader)
    pbar = tqdm(total=n_steps, desc=desc, leave=False)
    for _ in range(n_steps):
        batch     = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        outputs  = peft_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits   = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        ce_loss  = F.cross_entropy(
            logits[..., :-1, :].contiguous().float().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )

        ewc_pen = torch.tensor(0.0, device=device)
        for n, p in peft_model.named_parameters():
            if "lora_" in n and p.requires_grad and n in ewc_fisher:
                ewc_pen = ewc_pen + (ewc_fisher[n] * (p - ewc_anchor[n]).pow(2)).sum()

        loss = ce_loss + ewc_lambda * ewc_pen
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({"ce": f"{ce_loss.item():.3f}", "ewc": f"{ewc_pen.item():.2e}"})
        pbar.update(1)

    pbar.close()
    peft_model.eval()


def run_ewc_forgetting_pair(
    base_model, tokenizer,
    task_A, task_B,
    train_A_loader, train_B_loader,
    eval_A_loader, eval_B_loader,
    device, args,
) -> Dict:
    """
    EWC baseline: LoRA on all layers, EWC regularization during task_B.

    Theory prediction:
      Transformer: EWC protects early high-curvature (Necessary) layers;
                   adaptation writes to late low-curvature (Plastic) layers →
                   EWC and adaptation are spatially decoupled → low forgetting.
      SSM:         curvature is aligned with plasticity → EWC fights the
                   adaptation gradient → relatively higher forgetting.
    """
    n_layers   = args._ewc_n_layers
    all_layers = list(range(n_layers))
    logger.info(f"  [ewc] {task_A}→{task_B}: all {n_layers} layers, λ={args.ewc_lambda}")

    peft_model = apply_lora_to_layers(
        base_model,
        layer_indices=all_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    snap_init = _snapshot_lora(peft_model)

    # Phase 1: fine-tune on task_A
    finetune(peft_model, train_A_loader, device,
             n_steps=args.n_train_steps, lr=args.lr,
             desc=f"[ewc] FT-A({task_A})")
    snap_A = _snapshot_lora(peft_model)

    # Compute Fisher and anchor after task_A
    logger.info(f"  [ewc] Computing Fisher diagonal ({args.ewc_fisher_steps} steps)…")
    ewc_fisher = compute_ewc_fisher(peft_model, train_A_loader, device,
                                    n_steps=args.ewc_fisher_steps)
    ewc_anchor = {n: p.detach().clone()
                  for n, p in peft_model.named_parameters()
                  if "lora_" in n and p.requires_grad}

    # Eval task_A before B
    ppl_A_before = eval_perplexity(peft_model, eval_A_loader, device, n_steps=args.n_eval_steps)
    acc_A_before = None
    if task_A in MCQ_TASKS:
        acc_A_before = eval_mcq_accuracy(
            peft_model, tokenizer, task_A,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )
    logger.info(f"  [ewc] task_A before B: ppl={ppl_A_before:.3f}"
                + (f", acc={acc_A_before:.4f}" if acc_A_before else ""))

    # Eval task_B before B
    ppl_B_before = eval_perplexity(peft_model, eval_B_loader, device, n_steps=args.n_eval_steps)
    acc_B_before = None
    if task_B in MCQ_TASKS:
        acc_B_before = eval_mcq_accuracy(
            peft_model, tokenizer, task_B,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )

    # Phase 2: fine-tune on task_B with EWC
    finetune_ewc(peft_model, train_B_loader, device,
                 n_steps=args.n_train_steps, lr=args.lr,
                 ewc_fisher=ewc_fisher, ewc_anchor=ewc_anchor,
                 ewc_lambda=args.ewc_lambda,
                 desc=f"[ewc] FT-B({task_B})")
    snap_B = _snapshot_lora(peft_model)
    overlap = _update_overlap(snap_init, snap_A, snap_B)

    # Eval after B
    ppl_A_after = eval_perplexity(peft_model, eval_A_loader, device, n_steps=args.n_eval_steps)
    acc_A_after = None
    if task_A in MCQ_TASKS:
        acc_A_after = eval_mcq_accuracy(
            peft_model, tokenizer, task_A,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )

    ppl_B_after = eval_perplexity(peft_model, eval_B_loader, device, n_steps=args.n_eval_steps)
    acc_B_after = None
    if task_B in MCQ_TASKS:
        acc_B_after = eval_mcq_accuracy(
            peft_model, tokenizer, task_B,
            n_examples=args.n_eval_examples, seed=2, device=device,
            cache_dir=args.hf_cache_dir, seq_len=args.seq_len,
        )

    forgetting_ppl   = ppl_A_after - ppl_A_before
    forgetting_ratio = ppl_A_after / ppl_A_before if ppl_A_before > 0 else float("nan")
    forgetting_acc   = (acc_A_after - acc_A_before) if (acc_A_before and acc_A_after) else None
    learning_ppl     = ppl_B_before - ppl_B_after
    learning_acc     = (acc_B_after - acc_B_before) if (acc_B_before and acc_B_after) else None

    logger.info(f"  [ewc] task_A after  B: ppl={ppl_A_after:.3f}"
                + (f", acc={acc_A_after:.4f}" if acc_A_after else ""))
    logger.info(f"  [ewc] Forgetting — Δppl={forgetting_ppl:+.3f} (ratio {forgetting_ratio:.3f})")

    remove_lora(peft_model)
    del peft_model, ewc_fisher, ewc_anchor
    torch.cuda.empty_cache()
    for p in base_model.parameters():
        p.requires_grad_(False)

    return {
        "method": "ewc",
        "task_A": task_A,
        "task_B": task_B,
        "layers": all_layers,
        "ewc_lambda": args.ewc_lambda,
        "ppl_A_before": float(ppl_A_before),
        "ppl_A_after":  float(ppl_A_after),
        "forgetting_delta_ppl": float(forgetting_ppl),
        "forgetting_ratio":     float(forgetting_ratio),
        "acc_A_before": float(acc_A_before) if acc_A_before is not None else None,
        "acc_A_after":  float(acc_A_after)  if acc_A_after  is not None else None,
        "forgetting_delta_acc": float(forgetting_acc) if forgetting_acc is not None else None,
        "ppl_B_before": float(ppl_B_before),
        "ppl_B_after":  float(ppl_B_after),
        "learning_delta_ppl": float(learning_ppl),
        "acc_B_before": float(acc_B_before) if acc_B_before is not None else None,
        "acc_B_after":  float(acc_B_after)  if acc_B_after  is not None else None,
        "learning_delta_acc": float(learning_acc) if learning_acc is not None else None,
        **overlap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if len(args.task_pairs) % 2 != 0:
        raise ValueError("--task_pairs must have an even number of entries (A B A B ...).")

    pairs = [(args.task_pairs[i], args.task_pairs[i + 1])
             for i in range(0, len(args.task_pairs), 2)]

    # ── Load importance ──────────────────────────────────────────────────
    json_path = resolve_path(args.importance_json)  # glob OK: newest match
    logger.info(f"Loading importance: {json_path}")
    importance = load_result(json_path)
    logger.info(f"  {importance.n_layers} layers, {importance.n_tasks} tasks, "
                f"method={importance.method}")

    # ── Output path ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_short = Path(args.model_name_or_path).name
    run_name = args.run_name or f"{model_short}_cl_forgetting_{ts}"
    out_path = Path(args.output_dir) / f"{run_name}.json"

    meta = {
        "run_name": run_name,
        "model": args.model_name_or_path,
        "importance_json": json_path,
        "k": args.k,
        "n_train_steps": args.n_train_steps,
        "pairs": [{"A": a, "B": b} for a, b in pairs],
    }

    # ── Phase 1: Pre-plan all work items (consumes RNG deterministically) ─
    # Must happen before any checkpoint loading so random_k layers are always
    # drawn in the same order regardless of which items are already done.
    n_layers = importance.n_layers
    rng = np.random.default_rng(args.seed)
    work_items = []  # (task_A, task_B, method_label, layer_indices)
    for task_A, task_B in pairs:
        if task_A in importance.task_names:
            task_idx = importance.task_names.index(task_A)
            importance_vec = importance.matrix[:, task_idx]
        else:
            logger.warning(f"Task {task_A} not in importance; using mean.")
            importance_vec = importance.matrix.mean(axis=1)

        sorted_desc = np.argsort(importance_vec)[::-1].tolist()
        top_k_layers    = sorted_desc[:args.k]
        bottom_k_layers = sorted_desc[-args.k:]
        random_k_layers = sorted(rng.choice(n_layers, size=args.k, replace=False).tolist())

        work_items.extend([
            (task_A, task_B, "top_k",    top_k_layers),
            (task_A, task_B, "bottom_k", bottom_k_layers),
            (task_A, task_B, "random_k", random_k_layers),
            (task_A, task_B, "ewc",      list(range(n_layers))),
        ])

    # ── Phase 2: Load checkpoint (partial results from a previous run) ───
    all_results = []
    done_keys: set = set()
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            all_results = existing.get("results", [])
            done_keys = {(r["method"], r["task_A"], r["task_B"]) for r in all_results}
            logger.info(f"Checkpoint found: {len(done_keys)}/{len(work_items)} items already done.")
        except Exception as e:
            logger.warning(f"Could not read checkpoint {out_path}: {e}. Starting fresh.")

    if len(done_keys) == len(work_items):
        logger.info("All items already complete. Nothing to do.")
        return

    # ── Device + Model (only load if there is work to do) ────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tokenizer = load_from_args(args, upcast_state_spaces=True)
    if args.target_modules is None:
        if is_mamba(base_model) or is_rwkv(base_model):
            args.target_modules = default_target_modules(base_model)
        else:
            args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]
    logger.info(f"LoRA target modules: {args.target_modules}")
    args._ewc_n_layers = n_layers  # pass layer count to EWC runner without extra arg

    # ── Phase 3: Execute remaining items, checkpoint after each ──────────
    current_pair = None
    train_A_loader = train_B_loader = eval_A_loader = eval_B_loader = None

    for task_A, task_B, method_label, layer_indices in work_items:
        key = (method_label, task_A, task_B)
        if key in done_keys:
            logger.info(f"SKIP (cached): [{method_label}] {task_A}→{task_B}")
            continue

        # Rebuild data loaders only when the (task_A, task_B) pair changes
        if (task_A, task_B) != current_pair:
            logger.info(f"\n{'='*60}")
            logger.info(f"Pair: {task_A} → {task_B}")
            logger.info(f"{'='*60}")
            train_A_loader = build_task_dataloaders(
                tasks=[task_A], tokenizer=tokenizer, n_examples=args.n_train_examples,
                seq_len=args.seq_len, batch_size=args.batch_size,
                seed=args.seed, cache_dir=args.hf_cache_dir,
                num_workers=args.num_workers,
            )[task_A]
            train_B_loader = build_task_dataloaders(
                tasks=[task_B], tokenizer=tokenizer, n_examples=args.n_train_examples,
                seq_len=args.seq_len, batch_size=args.batch_size,
                seed=args.seed + 1, cache_dir=args.hf_cache_dir,
                num_workers=args.num_workers,
            )[task_B]
            eval_A_loader = build_task_dataloaders(
                tasks=[task_A], tokenizer=tokenizer, n_examples=args.n_eval_examples,
                seq_len=args.seq_len, batch_size=args.batch_size,
                seed=1, cache_dir=args.hf_cache_dir,
                num_workers=args.num_workers,
            )[task_A]
            eval_B_loader = build_task_dataloaders(
                tasks=[task_B], tokenizer=tokenizer, n_examples=args.n_eval_examples,
                seq_len=args.seq_len, batch_size=args.batch_size,
                seed=1, cache_dir=args.hf_cache_dir,
                num_workers=args.num_workers,
            )[task_B]
            current_pair = (task_A, task_B)

        if method_label == "ewc":
            result = run_ewc_forgetting_pair(
                base_model=base_model,
                tokenizer=tokenizer,
                task_A=task_A,
                task_B=task_B,
                train_A_loader=train_A_loader,
                train_B_loader=train_B_loader,
                eval_A_loader=eval_A_loader,
                eval_B_loader=eval_B_loader,
                device=device,
                args=args,
            )
        else:
            result = run_forgetting_pair(
                base_model=base_model,
                tokenizer=tokenizer,
                task_A=task_A,
                task_B=task_B,
                layer_indices=sorted(layer_indices),
                method_label=method_label,
                train_A_loader=train_A_loader,
                train_B_loader=train_B_loader,
                eval_A_loader=eval_A_loader,
                eval_B_loader=eval_B_loader,
                device=device,
                args=args,
            )
        all_results.append(result)
        done_keys.add(key)

        # Write checkpoint immediately so progress survives preemption/timeout
        with open(out_path, "w") as f:
            json.dump({**meta, "results": all_results}, f, indent=2)
        logger.info(f"Checkpoint saved ({len(all_results)}/{len(work_items)} done): {out_path}")

    # Summary table
    logger.info("\n=== Forgetting Summary (task_A) ===")
    logger.info(f"{'Pair':<25} {'Method':<12} {'ppl_A_pre':>10} {'ppl_A_post':>11} {'Δppl':>8} {'ratio':>7} {'acc_A_pre':>10} {'acc_A_post':>11} {'Δacc':>8}")
    for r in all_results:
        pair_str = f"{r['task_A']}→{r['task_B']}"
        acc_pre  = f"{r['acc_A_before']:>10.4f}" if r['acc_A_before'] is not None else f"{'N/A':>10}"
        acc_post = f"{r['acc_A_after']:>11.4f}"  if r['acc_A_after']  is not None else f"{'N/A':>11}"
        d_acc    = f"{r['forgetting_delta_acc']:>+8.4f}" if r['forgetting_delta_acc'] is not None else f"{'N/A':>8}"
        logger.info(
            f"{pair_str:<25} {r['method']:<12} "
            f"{r['ppl_A_before']:>10.3f} {r['ppl_A_after']:>11.3f} "
            f"{r['forgetting_delta_ppl']:>+8.3f} {r['forgetting_ratio']:>7.3f} "
            f"{acc_pre} {acc_post} {d_acc}"
        )

    logger.info("\n=== Learning Summary (task_B) ===")
    logger.info(f"{'Pair':<25} {'Method':<12} {'ppl_B_pre':>10} {'ppl_B_post':>11} {'Δppl':>8} {'acc_B_pre':>10} {'acc_B_post':>11} {'Δacc':>8}")
    for r in all_results:
        pair_str = f"{r['task_A']}→{r['task_B']}"
        acc_pre  = f"{r['acc_B_before']:>10.4f}" if r['acc_B_before'] is not None else f"{'N/A':>10}"
        acc_post = f"{r['acc_B_after']:>11.4f}"  if r['acc_B_after']  is not None else f"{'N/A':>11}"
        d_acc    = f"{r['learning_delta_acc']:>+8.4f}" if r['learning_delta_acc'] is not None else f"{'N/A':>8}"
        logger.info(
            f"{pair_str:<25} {r['method']:<12} "
            f"{r['ppl_B_before']:>10.3f} {r['ppl_B_after']:>11.3f} "
            f"{r['learning_delta_ppl']:>+8.3f} "
            f"{acc_pre} {acc_post} {d_acc}"
        )


if __name__ == "__main__":
    main()
