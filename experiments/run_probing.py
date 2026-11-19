"""
Linear task-identity probes per layer.

For each layer of a language model, trains a multi-class linear probe that predicts
which of the T tasks an example belongs to, using that layer's mean-pooled hidden
states as features.

  I[l, t] = per-class probe accuracy for task t at layer l
           = how distinctively layer l represents task-t examples vs. all others

This yields an importance matrix [n_layers × n_tasks] where high values mean the
layer has developed task-specific representations — a sharper signal than zero-shot
gradient attribution or ablation.

Usage:
    python experiments/run_probing.py --model_name_or_path Qwen/Qwen3-8B \\
        --n_examples 256 --seq_len 512 --batch_size 8 \\
        --output_dir results/probing/qwen3-8b
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from layerimp.data import ALL_TASKS, build_task_dataloaders
from layerimp.importance import ImportanceResult, _normalize_columns
from layerimp.models import get_decoder_layers, get_layer_types, load_from_args
from layerimp.utils import init_wandb, save_result, setup_logging

setup_logging()
logger = logging.getLogger("probing")


# ---------------------------------------------------------------------------
# Hidden-state collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_hidden_states(
    model: nn.Module,
    task_dataloaders: Dict[str, DataLoader],
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Single-pass hidden-state collection with hooks on every layer.

    Registers forward hooks on all decoder layers simultaneously. For each
    example, captures the mean-pooled (over sequence) hidden state at every
    layer. Only the pooled vector is stored, keeping peak RAM low.

    Returns:
        states: dict  layer_idx -> ndarray [n_examples, d_model]  (float32, CPU)
        labels: ndarray [n_examples]  task label (0 … T-1)
    """
    model.eval()
    layers = get_decoder_layers(model)
    n_layers = len(layers)

    # Will accumulate pooled states per layer across all batches
    layer_states: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_layers)}
    all_labels: List[int] = []

    # ── Register hooks on full layer outputs ─────────────────────────────────
    # Hook the entire decoder layer (after MLP + residual) so we capture the
    # full residual stream, not just the attention sub-layer output.
    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            # Transformer layers return (hidden_state, ...) or just hidden_state.
            h = output[0] if isinstance(output, tuple) else output
            # Mean-pool over sequence dimension: [B, T, D] → [B, D]
            pooled = h.float().mean(dim=1).detach().cpu().numpy()  # [B, D]
            layer_states[layer_idx].append(pooled)
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    try:
        task_names = list(task_dataloaders.keys())
        for t_idx, task_name in enumerate(task_names):
            loader = task_dataloaders[task_name]
            pbar = tqdm(loader, desc=f"[probing collect] {task_name}", leave=False)
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                # Record task label for each example in the batch
                all_labels.extend([t_idx] * input_ids.shape[0])
    finally:
        for h in handles:
            h.remove()

    # Stack per-layer lists into arrays
    states = {i: np.concatenate(layer_states[i], axis=0) for i in range(n_layers)}
    labels = np.array(all_labels, dtype=np.int64)
    logger.info(f"Collected {labels.shape[0]} examples × {n_layers} layers")
    return states, labels


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------

def train_probes(
    states: Dict[int, np.ndarray],
    labels: np.ndarray,
    task_names: List[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """
    Train one logistic regression probe per layer.

    For each layer, fits a multiclass (one-vs-rest) linear classifier that
    predicts the task label from the mean-pooled hidden state. Reports
    per-class accuracy on a held-out test split.

    Returns:
        matrix  ndarray [n_layers, n_tasks]  per-class probe accuracy in [0, 1]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    n_layers = len(states)
    n_tasks = len(task_names)
    matrix = np.zeros((n_layers, n_tasks), dtype=np.float32)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)

    for l_idx in tqdm(range(n_layers), desc="[probing] training probes"):
        X = states[l_idx]           # [n_examples, d_model]
        y = labels                   # [n_examples]

        # Stratified train/test split
        train_idx, test_idx = next(splitter.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardise features (important for LR convergence)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Fit multiclass logistic regression
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Per-class accuracy
        for t_idx, task_name in enumerate(task_names):
            mask = y_test == t_idx
            if mask.sum() == 0:
                matrix[l_idx, t_idx] = float("nan")
            else:
                matrix[l_idx, t_idx] = float((y_pred[mask] == t_idx).mean())

        overall = float((y_pred == y_test).mean())
        logger.debug(f"  L{l_idx:02d}: overall acc={overall:.3f}  "
                     f"per-task=[{', '.join(f'{matrix[l_idx,t]:.2f}' for t in range(n_tasks))}]")

    return matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Linear task-identity probes per layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--hf_cache_dir", type=str, default=None)
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    p.add_argument("--n_examples", type=int, default=256,
                   help="Examples per task for hidden-state collection.")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--test_fraction", type=float, default=0.2,
                   help="Fraction of examples held out for probe evaluation.")
    p.add_argument("--output_dir", type=str, default="results/probing")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = init_wandb(args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_run_name, tags=["probing"], config=vars(args))

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model_name_or_path} [{args.dtype}]")
    model, tokenizer = load_from_args(args)
    layer_types = get_layer_types(model)
    n_layers = len(get_decoder_layers(model))
    logger.info(f"Model loaded: {n_layers} layers  types={set(layer_types)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading data for {len(args.tasks)} tasks × {args.n_examples} examples each")
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
    task_names = list(dataloaders.keys())
    logger.info(f"Tasks: {task_names}")

    # ── Collect hidden states ─────────────────────────────────────────────────
    t0 = time.time()
    states, labels = collect_hidden_states(model, dataloaders, device)
    logger.info(f"Hidden-state collection done in {time.time()-t0:.1f}s")

    # Free model memory before sklearn fitting
    del model
    torch.cuda.empty_cache()

    # ── Train probes ──────────────────────────────────────────────────────────
    t1 = time.time()
    matrix = train_probes(
        states, labels, task_names,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    logger.info(f"Probe training done in {time.time()-t1:.1f}s")

    # ── Build ImportanceResult ───────────────────────────────────────────────
    result = ImportanceResult(
        matrix=matrix,
        matrix_norm=_normalize_columns(np.nan_to_num(matrix, nan=0.0)),
        layer_types=layer_types,
        task_names=task_names,
        method="probing",
        model_name=args.model_name_or_path,
        n_samples=args.n_examples,
        seq_len=args.seq_len,
        extra={"test_fraction": args.test_fraction},
    )

    # ── Save results ──────────────────────────────────────────────────────────
    model_tag = os.path.basename(args.model_name_or_path)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_result(result, args.output_dir, f"probing_{model_tag}_{ts}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=== Probe accuracy summary ===")
    chance = 1.0 / len(task_names)
    logger.info(f"Chance level: {chance:.3f}")
    for t_idx, task in enumerate(task_names):
        col = matrix[:, t_idx]
        logger.info(f"  {task:30s}: mean={np.nanmean(col):.3f}  "
                    f"max={np.nanmax(col):.3f} @ L{int(np.nanargmax(col)):02d}  "
                    f"min={np.nanmin(col):.3f}")

    sim_matrix = result.task_similarity_matrix()
    triu_idx = np.triu_indices(len(task_names), k=1)
    sims = sim_matrix[triu_idx]
    most_sim_i, most_sim_j = triu_idx[0][np.argmax(sims)], triu_idx[1][np.argmax(sims)]
    most_div_i, most_div_j = triu_idx[0][np.argmin(sims)], triu_idx[1][np.argmin(sims)]
    logger.info(f"Most similar:  {task_names[most_sim_i]} ↔ {task_names[most_sim_j]}  (cos={sims.max():.4f})")
    logger.info(f"Most divergent: {task_names[most_div_i]} ↔ {task_names[most_div_j]}  (cos={sims.min():.4f})")

    # ── W&B log ───────────────────────────────────────────────────────────────
    if run is not None:
        for t_idx, task in enumerate(task_names):
            for l_idx in range(n_layers):
                run.log({f"probe_acc/{task}/L{l_idx:02d}": float(matrix[l_idx, t_idx])}, step=l_idx)
        run.log({"most_divergent_sim": float(sims.min()),
                 "mean_probe_acc": float(np.nanmean(matrix))})
        run.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
