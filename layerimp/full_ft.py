"""
Full fine-tuning gradient / delta importance.

Control for the LoRA delta measurement: per-layer adaptation pressure
measured WITHOUT any adapter parameterization.

Two matrices are produced per run:

  1. Gradient-norm importance (method="full_ft_grad"):
         I_grad[l, t] = E_steps [ sqrt( sum_{p in layer_l} ||∇_p L_t||² ) ]
     i.e. the Frobenius norm of the full-parameter gradient of layer l,
     averaged over fine-tuning steps.  At n_steps=0 this reduces to the
     zero-shot per-layer gradient norm at the pretrained point (one pass
     over the calibration set, no weight updates).

  2. Full-FT delta importance (method="full_ft_delta", only if n_steps > 0):
         I_delta[l, t] = sqrt( sum_{p in layer_l} ||p_final - p_init||²_F )
     the direct full-parameter counterpart of the LoRA ||ΔW||_F measure.

Rationale: if the per-layer plasticity profile (terminal concentration in
transformers, distributed in SSMs) replicates under full fine-tuning, the
finding is a property of the model/task — not of the LoRA parameterization
(B=0 init, alpha scaling, restricted target modules).

Key design choices:
  - All parameters of every decoder layer are trained (train_scope="decoder";
    embeddings/lm_head frozen so per-layer attribution is exhaustive), or
    everything with train_scope="all".
  - Initial weights are snapshotted to CPU once; restored before each task
    so every task fine-tunes from the same pretrained point.
  - extra["grad_matrix_step0"] records the pretrained-point gradient norms
    (before any update) — the pure "gradient norm per layer" measurement.
  - extra["target_matrix"] restricts the gradient norm to the same modules
    LoRA targeted, for an apples-to-apples comparison with lora_delta.
  - Results are ImportanceResult objects, directly comparable with the
    ablation / lora_delta results.
"""

import logging
from itertools import cycle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from layerimp.importance import ImportanceResult, _normalize_columns
from layerimp.models import default_target_modules, get_decoder_layers

logger = logging.getLogger(__name__)


class FullFTGradAnalyzer:
    """
    Computes layer importance matrices via full-parameter fine-tuning.

    Usage::

        analyzer = FullFTGradAnalyzer(model, lr=1e-5, n_steps=30, optimizer="adamw")
        results = analyzer.run(task_dataloaders, device, layer_types=..., model_name=...)
        results["grad"]   # ImportanceResult, method="full_ft_grad"
        results["delta"]  # ImportanceResult, method="full_ft_delta" (None if n_steps=0)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        n_steps: int = 30,
        optimizer: str = "adamw",
        train_scope: str = "decoder",
        target_modules: Optional[List[str]] = None,
        shuffle_labels: bool = False,
    ):
        assert optimizer in ("adamw", "sgd", "adamw8bit", "paged_adamw8bit"), \
            f"Unknown optimizer: {optimizer}"
        assert train_scope in ("decoder", "all"), f"Unknown train_scope: {train_scope}"

        self.model          = model
        self.lr             = lr
        self.n_steps        = n_steps
        self.optimizer_name = optimizer
        self.train_scope    = train_scope
        self.shuffle_labels = shuffle_labels

        if target_modules is None:
            target_modules = default_target_modules(model)
        self.target_modules = target_modules

        self._decoder_layers = get_decoder_layers(model)
        self.n_layers = len(self._decoder_layers)

        # ── Set trainable scope ────────────────────────────────────────────
        if train_scope == "decoder":
            for p in model.parameters():
                p.requires_grad_(False)
            for layer in self._decoder_layers:
                for p in layer.parameters():
                    p.requires_grad_(True)
        else:  # "all"
            for p in model.parameters():
                p.requires_grad_(True)

        # ── Per-layer parameter registry ───────────────────────────────────
        # layer_params[l] = list of (param, is_target_module_param)
        self._layer_params: List[List[Tuple[torch.nn.Parameter, bool]]] = []
        for layer in self._decoder_layers:
            target_param_ids = set()
            for mod_name, module in layer.named_modules():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in self.target_modules:
                    for p in module.parameters(recurse=False):
                        target_param_ids.add(id(p))
            entries = [
                (p, id(p) in target_param_ids)
                for p in layer.parameters()
                if p.requires_grad
            ]
            self._layer_params.append(entries)

        # ── CPU snapshot of all trainable params (reset + delta) ──────────
        self._trainable: List[torch.nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self._snapshot: List[torch.Tensor] = [
            p.detach().to("cpu", copy=True) for p in self._trainable
        ]

        self._optimizer = None  # created once; state reset in-place per task

        n_trainable = sum(p.numel() for p in self._trainable)
        snapshot_gb = sum(t.numel() * t.element_size() for t in self._snapshot) / 1e9
        logger.info(
            f"Full FT — scope={train_scope}, optimizer={optimizer}, lr={lr}, "
            f"steps={n_steps}, trainable params: {n_trainable:,}, "
            f"CPU snapshot: {snapshot_gb:.1f} GB, "
            f"target-module breakdown: {self.target_modules}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _restore_snapshot(self):
        """Copy the pretrained weights back so each task starts identically."""
        for p, init in zip(self._trainable, self._snapshot):
            p.copy_(init.to(p.device, p.dtype), non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _make_optimizer(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self._trainable, lr=self.lr, weight_decay=0.0)
        if self.optimizer_name in ("adamw8bit", "paged_adamw8bit"):
            # bitsandbytes 8-bit AdamW: same per-parameter normalization as
            # AdamW at 2 bytes/param of state; the paged variant spills state
            # to pinned CPU memory under GPU pressure. Lets ≥7B models use the
            # same Adam-family protocol as the ≤4B fp32 runs.
            import bitsandbytes as bnb
            cls = (bnb.optim.PagedAdamW8bit
                   if self.optimizer_name == "paged_adamw8bit"
                   else bnb.optim.AdamW8bit)
            return cls(self._trainable, lr=self.lr, weight_decay=0.0)
        return torch.optim.SGD(self._trainable, lr=self.lr, momentum=0.0)

    # State keys safe to zero for a fresh-optimizer reset. bnb 8-bit keeps
    # (state1/2, absmax1/2) quantized moments; qmap1/2 are constant codebooks
    # and must be left alone.
    _RESETTABLE_STATE = {
        "state1", "state2", "absmax1", "absmax2",
        "exp_avg", "exp_avg_sq", "max_exp_avg_sq", "unorm_vec", "momentum_buffer",
    }

    def _get_optimizer(self):
        """Create the optimizer once and reset its state in-place per task.

        Re-creating a bnb paged optimizer per task leaks host RAM: every paged
        state buffer is appended to the GlobalPageManager singleton
        (bitsandbytes/optim/optimizer.py) and never released, so a 10-task run
        pins 10× the optimizer state.
        """
        if self._optimizer is None:
            self._optimizer = self._make_optimizer()
            return self._optimizer
        opt = self._optimizer
        for state in opt.state.values():
            for k, v in list(state.items()):
                if k == "step":
                    state[k] = torch.zeros_like(v) if torch.is_tensor(v) else 0
                elif k in self._RESETTABLE_STATE and torch.is_tensor(v):
                    v.zero_()
        return opt

    @torch.no_grad()
    def _layer_grad_norms(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-layer Frobenius norm of the current gradients.

        Returns (all_params_norms, target_module_only_norms), each [n_layers].
        Accumulation in float32 regardless of model dtype.
        """
        norms_all    = np.zeros(self.n_layers, dtype=np.float64)
        norms_target = np.zeros(self.n_layers, dtype=np.float64)
        for l_idx, entries in enumerate(self._layer_params):
            sq_all, sq_target = 0.0, 0.0
            for p, is_target in entries:
                if p.grad is None:
                    continue
                sq = p.grad.detach().float().pow(2).sum().item()
                sq_all += sq
                if is_target:
                    sq_target += sq
            norms_all[l_idx]    = sq_all ** 0.5
            norms_target[l_idx] = sq_target ** 0.5
        return norms_all, norms_target

    @torch.no_grad()
    def _layer_delta_norms(self) -> np.ndarray:
        """Per-layer ||W_final - W_init||_F against the CPU snapshot."""
        init_by_id = {
            id(p): init for p, init in zip(self._trainable, self._snapshot)
        }
        norms = np.zeros(self.n_layers, dtype=np.float64)
        for l_idx, entries in enumerate(self._layer_params):
            sq = 0.0
            for p, _ in entries:
                init = init_by_id[id(p)].to(p.device)
                sq += (p.detach().float() - init.float()).pow(2).sum().item()
            norms[l_idx] = sq ** 0.5
        return norms

    def _loss_from_batch(self, batch: dict, device: torch.device) -> torch.Tensor:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=False,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        if self.shuffle_labels:
            # Random-label control (same as LoRADeltaAnalyzer): permute labels
            # within batch to break task signal, preserve gradient magnitude.
            flat = shift_labels.view(-1)
            valid = flat != -100
            flat[valid] = flat[valid][torch.randperm(valid.sum(), device=flat.device)]
            shift_labels = flat.view(shift_labels.shape)

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------
    # Per-task fine-tuning
    # ------------------------------------------------------------------

    def compute_task_importance(
        self,
        dataloader: DataLoader,
        task_name: str,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        Fine-tune the full model on one task for self.n_steps steps.

        Returns dict with keys:
            grad        [n_layers]  mean per-step gradient norm
            grad_step0  [n_layers]  gradient norm at the pretrained point
            grad_target [n_layers]  mean grad norm, LoRA target modules only
            delta       [n_layers]  ||ΔW||_F after training (zeros if n_steps=0)
        """
        self._restore_snapshot()
        optimizer = self._get_optimizer() if self.n_steps > 0 else None

        self.model.train()
        grad_sum        = np.zeros(self.n_layers, dtype=np.float64)
        grad_target_sum = np.zeros(self.n_layers, dtype=np.float64)
        grad_step0      = np.zeros(self.n_layers, dtype=np.float64)
        losses          = []

        if self.n_steps > 0:
            data_iter = cycle(dataloader)
            total = self.n_steps
        else:
            data_iter = iter(dataloader)
            total = len(dataloader)

        pbar = tqdm(total=total, desc=f"[full_ft] {task_name:<30}", leave=False)
        n_measured = 0
        for step in range(total):
            batch = next(data_iter)

            loss = self._loss_from_batch(batch, device)
            loss.backward()

            norms_all, norms_target = self._layer_grad_norms()
            grad_sum        += norms_all
            grad_target_sum += norms_target
            if step == 0:
                grad_step0 = norms_all.copy()
            n_measured += 1

            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                self.model.zero_grad(set_to_none=True)

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            pbar.update(1)

        pbar.close()
        self.model.eval()

        grad_mean        = grad_sum / max(n_measured, 1)
        grad_target_mean = grad_target_sum / max(n_measured, 1)
        delta = (
            self._layer_delta_norms()
            if self.n_steps > 0
            else np.zeros(self.n_layers, dtype=np.float64)
        )

        avg_loss = float(np.mean(losses[-min(20, len(losses)):])) if losses else float("nan")
        logger.info(
            f"  {task_name}: steps={n_measured}, final_loss={avg_loss:.3f}, "
            f"top_grad=L{int(grad_mean.argmax())} ({grad_mean.max():.4f}), "
            f"top_delta=L{int(delta.argmax())} ({delta.max():.4f})"
        )
        return {
            "grad":        grad_mean,
            "grad_step0":  grad_step0,
            "grad_target": grad_target_mean,
            "delta":       delta,
        }

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
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Optional[ImportanceResult]]:
        """
        Compute importance matrices over all tasks.

        Returns:
            {"grad": ImportanceResult(method="full_ft_grad"),
             "delta": ImportanceResult(method="full_ft_delta") or None}
        """
        task_names = list(task_dataloaders.keys())
        n_tasks = len(task_names)
        grad_matrix        = np.zeros((self.n_layers, n_tasks), dtype=np.float64)
        grad_step0_matrix  = np.zeros((self.n_layers, n_tasks), dtype=np.float64)
        grad_target_matrix = np.zeros((self.n_layers, n_tasks), dtype=np.float64)
        delta_matrix       = np.zeros((self.n_layers, n_tasks), dtype=np.float64)

        for t_idx, task_name in enumerate(task_names):
            logger.info(f"[{t_idx+1}/{n_tasks}] Full FT for task: {task_name}")
            cols = self.compute_task_importance(
                task_dataloaders[task_name], task_name, device
            )
            grad_matrix[:, t_idx]        = cols["grad"]
            grad_step0_matrix[:, t_idx]  = cols["grad_step0"]
            grad_target_matrix[:, t_idx] = cols["grad_target"]
            delta_matrix[:, t_idx]       = cols["delta"]

            if checkpoint_path is not None:
                np.savez(
                    checkpoint_path,
                    grad=grad_matrix,
                    grad_step0=grad_step0_matrix,
                    grad_target=grad_target_matrix,
                    delta=delta_matrix,
                    tasks=np.array(task_names),
                    tasks_done=t_idx + 1,
                )

        # Leave weights pristine for any downstream use of the model object.
        self._restore_snapshot()

        if layer_types is None:
            layer_types = ["unknown"] * self.n_layers

        common_extra = {
            "optimizer":       self.optimizer_name,
            "lr":              self.lr,
            "n_steps":         self.n_steps,
            "train_scope":     self.train_scope,
            "target_modules":  self.target_modules,
            "shuffle_labels":  self.shuffle_labels,
        }

        grad_result = ImportanceResult(
            matrix=grad_matrix,
            matrix_norm=_normalize_columns(grad_matrix),
            layer_types=layer_types,
            task_names=task_names,
            method="full_ft_grad",
            model_name=model_name,
            n_samples=n_samples,
            seq_len=seq_len,
            extra={
                **common_extra,
                "grad_matrix_step0":       grad_step0_matrix.tolist(),
                "grad_matrix_step0_norm":  _normalize_columns(grad_step0_matrix).tolist(),
                "target_matrix":           grad_target_matrix.tolist(),
                "target_matrix_norm":      _normalize_columns(grad_target_matrix).tolist(),
            },
        )

        delta_result = None
        if self.n_steps > 0:
            delta_result = ImportanceResult(
                matrix=delta_matrix,
                matrix_norm=_normalize_columns(delta_matrix),
                layer_types=layer_types,
                task_names=task_names,
                method="full_ft_delta",
                model_name=model_name,
                n_samples=n_samples,
                seq_len=seq_len,
                extra=dict(common_extra),
            )

        return {"grad": grad_result, "delta": delta_result}
