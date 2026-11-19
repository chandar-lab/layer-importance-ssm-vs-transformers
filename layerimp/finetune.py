"""
LoRA fine-tuning on a subset of layers + evaluation helpers.

Shared by the causal-validation and continual-learning experiments.
"""

import logging
import math
import random
from itertools import cycle
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from layerimp.data import load_task_dataset
from layerimp.models import sanitize_tied_weights_keys

logger = logging.getLogger(__name__)

# Tasks scored by multiple-choice accuracy; the rest are scored by perplexity.
MCQ_TASKS = {"winogrande", "arc_challenge", "mmlu", "piqa", "gpqa"}
PPL_TASKS = {"humaneval", "mbpp", "gsm8k", "squad", "longbench_single_doc"}


# ---------------------------------------------------------------------------
# LoRA on layer subset
# ---------------------------------------------------------------------------

def apply_lora_to_layers(
    model: nn.Module,
    layer_indices: List[int],
    lora_rank: int,
    lora_alpha: int,
    target_modules: List[str],
):
    """
    Wrap model with PEFT LoRA, restricted to the given layer indices.

    Uses LoraConfig.layers_to_transform to constrain adapter placement.
    Returns the peft_model.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError("peft>=0.10.0 required") from e

    for p in model.parameters():
        p.requires_grad_(False)

    sanitize_tied_weights_keys(model)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        layers_to_transform=layer_indices,
    )
    peft_model = get_peft_model(model, config)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info(
        f"  LoRA on layers {layer_indices}: rank={lora_rank}, "
        f"trainable params={n_trainable:,}"
    )
    return peft_model


def remove_lora(peft_model: nn.Module) -> nn.Module:
    """Merge LoRA back into base weights and return the unwrapped model."""
    try:
        from peft import PeftModel
    except ImportError:
        return peft_model

    if isinstance(peft_model, PeftModel):
        # Reset adapters to zero (B=0) before merging so base weights are unchanged
        for module in peft_model.modules():
            if hasattr(module, "lora_B") and "default" in module.lora_B:
                nn.init.zeros_(module.lora_B["default"].weight)
        merged = peft_model.merge_and_unload()
        return merged
    return peft_model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(
    peft_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_steps: int,
    lr: float,
    desc: str = "finetune",
) -> List[float]:
    """Fine-tune LoRA for n_steps steps, return per-step losses."""
    lora_params = [p for n, p in peft_model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)

    peft_model.train()
    losses = []
    data_iter = cycle(dataloader)

    pbar = tqdm(total=n_steps, desc=desc, leave=False)
    for _ in range(n_steps):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        outputs = peft_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        pbar.update(1)

    pbar.close()
    peft_model.eval()
    return losses


# ---------------------------------------------------------------------------
# Evaluation: perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_steps: int = 50,
) -> float:
    """Evaluate perplexity on up to n_steps batches from dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    steps = 0

    for batch in dataloader:
        if steps >= n_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()

        # Compute token-level loss without reduction
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        mask = flat_labels != -100
        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        total_tokens += mask.sum().item()
        steps += 1

    if total_tokens == 0:
        return float("inf")
    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)


# ---------------------------------------------------------------------------
# Evaluation: multiple-choice accuracy (log-likelihood)
# ---------------------------------------------------------------------------

def _build_mcq_eval_examples(task_name: str, n_examples: int, seed: int, cache_dir: Optional[str]):
    """
    Return a list of (premise_text, [choice_text, ...], correct_idx) tuples.
    Only for MCQ_TASKS.
    """
    dataset = load_task_dataset(task_name, cache_dir)
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:n_examples]

    examples = []

    if task_name == "winogrande":
        for i in indices:
            ex = dataset[i]
            sentence = ex["sentence"]
            # Present the sentence with each option substituted for "_"
            c1 = sentence.replace("_", ex["option1"], 1)
            c2 = sentence.replace("_", ex["option2"], 1)
            correct = 0 if ex["answer"] == "1" else 1
            examples.append(("", [c1, c2], correct))

    elif task_name == "arc_challenge":
        for i in indices:
            ex = dataset[i]
            q = ex["question"]
            texts = ex["choices"]["text"]
            options = [f"{q} {t}" for t in texts]
            # answerKey is a letter or number
            ak = ex["answerKey"]
            if ak.isdigit():
                correct = int(ak) - 1
            else:
                correct = ord(ak) - ord("A")
            correct = max(0, min(correct, len(options) - 1))
            examples.append(("", options, correct))

    elif task_name == "mmlu":
        for i in indices:
            ex = dataset[i]
            q = ex["question"]
            options = [f"{q} {c}" for c in ex["choices"]]
            examples.append(("", options, int(ex["answer"])))

    elif task_name == "piqa":
        for i in indices:
            ex = dataset[i]
            goal = ex["goal"]
            options = [f"{goal} {ex['sol1']}", f"{goal} {ex['sol2']}"]
            examples.append(("", options, int(ex["label"])))

    elif task_name == "gpqa":
        for i in indices:
            ex = dataset[i]
            q = ex["Question"]
            options = [
                f"{q} {ex['Correct Answer']}",
                f"{q} {ex['Incorrect Answer 1']}",
                f"{q} {ex['Incorrect Answer 2']}",
                f"{q} {ex['Incorrect Answer 3']}",
            ]
            examples.append(("", options, 0))  # correct is always index 0 in our format

    return examples


@torch.no_grad()
def eval_mcq_accuracy(
    model: nn.Module,
    tokenizer,
    task_name: str,
    n_examples: int,
    seed: int,
    device: torch.device,
    cache_dir: Optional[str] = None,
    seq_len: int = 512,
) -> float:
    """
    Evaluate multiple-choice accuracy via log-likelihood scoring.

    For each example, compute log P(choice_i | context) for each option;
    pick the highest as the model's answer; compare to ground truth.
    """
    examples = _build_mcq_eval_examples(task_name, n_examples, seed, cache_dir)
    if not examples:
        logger.warning(f"No MCQ examples loaded for {task_name}")
        return float("nan")

    model.eval()
    correct = 0

    for context, choices, gold_idx in tqdm(examples, desc=f"[mcq eval] {task_name}", leave=False):
        log_probs = []

        for choice in choices:
            text = (context + " " + choice).strip() if context else choice
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
            ).to(device)

            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                log_probs.append(float("-inf"))
                continue

            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()

            lp = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            log_probs.append(-lp.item())  # higher = more likely

        pred = int(np.argmax(log_probs))
        if pred == gold_idx:
            correct += 1

    return correct / len(examples) if examples else float("nan")
