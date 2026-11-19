"""
LoRA delta (plasticity) importance

Measures per-layer importance by fine-tuning LoRA adapters on each task and
recording how much each layer's weights change:

    I[l, t] = sum_{module in layer_l} ||ΔW_module||_F

where ΔW_module = scale * lora_B @ lora_A after K gradient steps.

Rationale: zero-shot gradient/ablation measures what the pre-trained model
already does uniformly.  Fine-tuning delta measures *adaptation pressure* —
where the model needs to change to specialize for task t.  This produces a
sharper, more task-discriminative importance matrix.

Key design choices:
  - LoRA applied once; weights reset (A~kaiming, B=0) before each task.
  - Only LoRA params are trained; base model is fully frozen.
  - Frobenius norm computed as  sqrt(trace((B^T B)(A A^T))) — exact, O(r²·d).
  - Returns an ImportanceResult with method="lora_delta".
"""

import logging
import math
from itertools import cycle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from layerimp.importance import ImportanceResult, _normalize_columns
from layerimp.models import (
    default_target_modules,
    get_decoder_layers,
    sanitize_tied_weights_keys,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frobenius norm of a LoRA update  ||scale * B @ A||_F
# ---------------------------------------------------------------------------

def _frobenius_lora(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
) -> float:
    """
    Compute ||scale * B @ A||_F without forming the full (out × in) matrix.

    Uses the identity:
        ||BA||_F² = trace(A^T B^T B A) = trace((B^T B)(A A^T))

    Both inner products are (r × r) and trivially cheap for small rank r.

    Args:
        A: lora_A weight, shape (r, in_features).
        B: lora_B weight, shape (out_features, r).
        scale: lora_alpha / lora_rank.

    Returns:
        Scalar Frobenius norm of the full update.
    """
    with torch.no_grad():
        BtB = B.t() @ B          # (r, r)
        AAt = A @ A.t()           # (r, r)
        sq = torch.trace(BtB @ AAt).clamp(min=0.0)
        return scale * sq.sqrt().item()


# ---------------------------------------------------------------------------
# LoRA Delta Analyzer
# ---------------------------------------------------------------------------

class LoRADeltaAnalyzer:
    """
    Computes the layer importance matrix I[n_layers, n_tasks] via LoRA
    fine-tuning delta attribution.

    Usage::

        analyzer = LoRADeltaAnalyzer(
            model, lora_rank=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lr=1e-4, n_steps=200,
        )
        result = analyzer.run(task_dataloaders, device, layer_types=..., model_name=...)

    The returned ImportanceResult uses method="lora_delta" and is serialisation-
    compatible with the existing gradient/ablation results.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        lr: float = 1e-4,
        n_steps: int = 200,
        shuffle_labels: bool = False,
    ):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA delta attribution. "
                "Install with: pip install peft>=0.10.0"
            ) from e

        self.lora_rank      = lora_rank
        self.lora_alpha     = lora_alpha
        self.lr             = lr
        self.n_steps        = n_steps
        self.shuffle_labels = shuffle_labels

        if target_modules is None:
            target_modules = default_target_modules(model)
        self.target_modules = target_modules

        # Get decoder layers BEFORE PEFT modifies submodule references.
        # PEFT replaces only the target linear modules (in-place on the same
        # layer objects), so this reference remains valid after wrapping.
        self._decoder_layers = get_decoder_layers(model)
        self.n_layers = len(self._decoder_layers)

        # Freeze entire base model before wrapping — PEFT will unfreeze LoRA params.
        for p in model.parameters():
            p.requires_grad_(False)

        sanitize_tied_weights_keys(model)

        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,   # no dropout: we measure norms, not generalisation
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.peft_model = get_peft_model(model, config)
        logger.info(
            f"LoRA applied — rank={lora_rank}, alpha={lora_alpha}, "
            f"targets={target_modules}, trainable params: "
            f"{sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad):,}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_lora(self):
        """
        Re-initialise all LoRA adapters to standard LoRA init:
          lora_A ~ kaiming_uniform,  lora_B = 0
        so that the effective update ΔW = scale*B@A = 0 at the start of
        each task's fine-tuning run.
        """
        for module in self.peft_model.modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                nn.init.kaiming_uniform_(
                    module.lora_A["default"].weight, a=math.sqrt(5)
                )
                nn.init.zeros_(module.lora_B["default"].weight)

    def _lora_params(self) -> List[torch.nn.Parameter]:
        """Return only the LoRA trainable parameters."""
        return [p for n, p in self.peft_model.named_parameters() if "lora_" in n]

    def _compute_layer_norms(self) -> np.ndarray:
        """
        For each decoder layer, aggregate Frobenius norms of all LoRA updates.

        Returns ndarray of shape [n_layers].
        """
        norms = np.zeros(self.n_layers, dtype=np.float64)
        for l_idx, layer in enumerate(self._decoder_layers):
            for module in layer.modules():
                if hasattr(module, "lora_A") and "default" in module.lora_A:
                    A = module.lora_A["default"].weight   # (r, in)
                    B = module.lora_B["default"].weight   # (out, r)
                    scale = module.scaling["default"]
                    norms[l_idx] += _frobenius_lora(A, B, scale)
        return norms

    # ------------------------------------------------------------------
    # Per-task fine-tuning
    # ------------------------------------------------------------------

    def compute_task_importance(
        self,
        dataloader: DataLoader,
        task_name: str,
        device: torch.device,
    ) -> np.ndarray:
        """
        Fine-tune LoRA adapters on one task for self.n_steps steps, then
        measure per-layer ||ΔW||_F.

        Returns ndarray of shape [n_layers].
        """
        self._reset_lora()

        optimizer = torch.optim.AdamW(
            self._lora_params(), lr=self.lr, weight_decay=0.0
        )

        self.peft_model.train()
        losses = []
        data_iter = cycle(dataloader)

        pbar = tqdm(
            total=self.n_steps,
            desc=f"[lora_delta] {task_name:<30}",
            leave=False,
        )
        for step in range(self.n_steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            outputs = self.peft_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            if self.shuffle_labels:
                # Random-label control: permute labels within batch to break
                # task-specific signal while preserving gradient norm magnitude.
                flat = shift_labels.view(-1)
                valid = flat != -100
                flat[valid] = flat[valid][torch.randperm(valid.sum(), device=flat.device)]
                shift_labels = flat.view(shift_labels.shape)
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
        self.peft_model.eval()

        norms = self._compute_layer_norms()
        avg_loss = float(np.mean(losses[-min(20, len(losses)):]))
        logger.info(
            f"  {task_name}: steps={self.n_steps}, "
            f"final_loss={avg_loss:.3f}, "
            f"total_norm={norms.sum():.4f}, "
            f"top_layer=L{int(norms.argmax())} ({norms.max():.4f})"
        )
        return norms

    # ------------------------------------------------------------------
    # Full matrix
    # ------------------------------------------------------------------

    def run(
        self,
        task_dataloaders: Dict[str, DataLoader],
        device: torch.device,
        layer_types: Optional[List[str]] = None,
        model_name: str = "unknown",
        n_samples: int = 0,
        seq_len: int = 0,
    ) -> ImportanceResult:
        """
        Compute the full importance matrix over all tasks.

        Args:
            task_dataloaders: Dict mapping task_name → DataLoader.
            device:           CUDA device.
            layer_types:      Optional list of layer type strings for annotation.
            model_name:       For metadata.
            n_samples:        For metadata.
            seq_len:          For metadata.

        Returns:
            ImportanceResult with method="lora_delta".  Compatible with
            analyse_results.py via --gradient_result <path>.
        """
        task_names = list(task_dataloaders.keys())
        matrix = np.zeros((self.n_layers, len(task_names)), dtype=np.float64)

        for t_idx, task_name in enumerate(task_names):
            logger.info(
                f"[{t_idx+1}/{len(task_names)}] LoRA delta for task: {task_name}"
            )
            col = self.compute_task_importance(
                task_dataloaders[task_name], task_name, device
            )
            matrix[:, t_idx] = col

        if layer_types is None:
            layer_types = ["unknown"] * self.n_layers

        return ImportanceResult(
            matrix=matrix,
            matrix_norm=_normalize_columns(matrix),
            layer_types=layer_types,
            task_names=task_names,
            method="lora_delta",
            model_name=model_name,
            n_samples=n_samples,
            seq_len=seq_len,
            extra={
                "lora_rank":       self.lora_rank,
                "lora_alpha":      self.lora_alpha,
                "lora_n_steps":    self.n_steps,
                "lora_lr":         self.lr,
                "target_modules":  self.target_modules,
            },
        )
