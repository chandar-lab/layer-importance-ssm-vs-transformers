"""
Zero-shot layer importance.

  - AblationImportanceAnalyzer : necessity,  I[l,t] = L_t(bypass layer l) - L_t(baseline)
  - GradientImportanceAnalyzer : Taylor criterion,  I[l,t] = E|h_l * dL_t/dh_l|

Both return an :class:`ImportanceResult` holding a [n_layers, n_tasks] matrix.
All other analyzers (LoRA delta, full fine-tuning, ...) reuse this container.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from layerimp.models import get_decoder_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ImportanceResult:
    """
    Stores the full importance matrix plus associated metadata.

    Attributes:
        matrix:         ndarray [n_layers, n_tasks], raw importance scores.
        matrix_norm:    ndarray [n_layers, n_tasks], column-normalized scores
                        (each task's scores sum to 1).
        layer_types:    Per-layer type strings (metadata only).
        task_names:     Ordered list of task names.
        method:         e.g. "ablation", "gradient", "lora_delta", "full_ft_delta".
        model_name:     HF model identifier used.
        n_samples:      Number of calibration samples used per task.
        seq_len:        Sequence length used.
        extra:          Method-specific metadata, serialised at the top level.
    """
    matrix:       np.ndarray
    matrix_norm:  np.ndarray
    layer_types:  List[str]
    task_names:   List[str]
    method:       str
    model_name:   str
    n_samples:    int
    seq_len:      int
    extra:        Dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_tasks(self) -> int:
        return self.matrix.shape[1]

    def top_layers_for_task(self, task: str, k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k (layer_idx, score) pairs for a given task."""
        t_idx = self.task_names.index(task)
        scores = self.matrix_norm[:, t_idx]
        ranked = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in ranked]

    def task_similarity_matrix(self) -> np.ndarray:
        """Cosine similarity between task importance profiles."""
        normed = self.matrix_norm / (np.linalg.norm(self.matrix_norm, axis=0, keepdims=True) + 1e-8)
        return normed.T @ normed  # [n_tasks, n_tasks]

    def to_dict(self) -> dict:
        return {
            "matrix":      self.matrix.tolist(),
            "matrix_norm": self.matrix_norm.tolist(),
            "layer_types": self.layer_types,
            "task_names":  self.task_names,
            "method":      self.method,
            "model_name":  self.model_name,
            "n_samples":   self.n_samples,
            "seq_len":     self.seq_len,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImportanceResult":
        matrix = np.array(d["matrix"])
        matrix_norm = np.array(d["matrix_norm"])
        return cls(
            matrix=matrix,
            matrix_norm=matrix_norm,
            layer_types=d["layer_types"],
            task_names=d["task_names"],
            method=d["method"],
            model_name=d["model_name"],
            n_samples=d["n_samples"],
            seq_len=d["seq_len"],
            extra={k: v for k, v in d.items() if k not in cls.__dataclass_fields__},
        )


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize each column (task) so scores sum to 1."""
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    return matrix / col_sums


# ---------------------------------------------------------------------------
# Gradient Importance Analyzer
# ---------------------------------------------------------------------------

class GradientImportanceAnalyzer:
    """
    Computes per-layer importance using the Taylor (gradient × activation) criterion.

    For each layer l and task t:
        I[l, t] = E_{x ~ D_t} [ mean | h_l(x) · ∇_{h_l} L_t(x) | ]

    where h_l is the output tensor of the attention sublayer (before residual add).

    This is a single-GPU method.  The model should be loaded with attn_implementation
    that supports per-tensor gradient retention (sdpa or eager).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._layers = get_decoder_layers(model)
        self.n_layers = len(self._layers)

    def _register_hooks(self) -> Tuple[List, Dict[int, torch.Tensor]]:
        """Register forward hooks on every primary sublayer (attention or SSM mixer)."""
        handles = []
        attn_outputs: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(module, inp, output):
                # output from self_attn/mixer is either a tensor or a tuple whose
                # first element is the hidden-state tensor.
                out = output[0] if isinstance(output, tuple) else output
                # Clone to a float32 leaf that retains grad
                out_f32 = out.float()
                out_f32.retain_grad()
                attn_outputs[layer_idx] = out_f32
                # We need to inject the float32 copy back so that the residual
                # connection and subsequent layers receive it.
                # Return the same dtype as the input to avoid dtype mismatches.
                if isinstance(output, tuple):
                    return (out_f32.to(out.dtype),) + output[1:]
                return out_f32.to(out.dtype)
            return hook

        for l_idx, layer in enumerate(self._layers):
            # Pick the primary sublayer to hook, in architecture priority order.
            if hasattr(layer, "self_attn"):
                sublayer = layer.self_attn          # transformer
            elif hasattr(layer, "mixer"):
                sublayer = layer.mixer              # original Mamba
            elif hasattr(layer, "mamba"):
                sublayer = layer.mamba              # Falcon Mamba
            elif hasattr(layer, "attention"):
                sublayer = layer.attention          # RWKV
            else:
                logger.warning(f"Layer {l_idx} ({type(layer).__name__}): no hookable sublayer, skipping.")
                continue
            h = sublayer.register_forward_hook(make_hook(l_idx))
            handles.append(h)

        return handles, attn_outputs

    @torch.enable_grad()
    def compute_task_importance(
        self,
        dataloader: DataLoader,
        task_name: str,
        device: torch.device,
        fp32_loss: bool = True,
    ) -> np.ndarray:
        """
        Compute importance vector of shape [n_layers] for one task.

        Args:
            dataloader: Calibration DataLoader for the task.
            task_name:  Used only for progress bar labeling.
            device:     CUDA device.
            fp32_loss:  Cast logits to float32 before computing loss.
        """
        self.model.eval()
        importances = torch.zeros(self.n_layers, dtype=torch.float64)
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"[gradient] {task_name}", leave=False)
        for batch in pbar:
            input_ids  = batch["input_ids"].to(device)
            labels     = batch["labels"].to(device)
            attn_mask  = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            self.model.zero_grad()
            handles, attn_outputs = self._register_hooks()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                # Shift for causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if fp32_loss:
                    shift_logits = shift_logits.float()

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss.backward()

                # Taylor criterion: |activation * gradient|
                for l_idx in range(self.n_layers):
                    if l_idx in attn_outputs:
                        h = attn_outputs[l_idx]
                        if h.grad is not None:
                            score = (h.detach() * h.grad).abs().mean().item()
                            importances[l_idx] += score

                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

            finally:
                for h in handles:
                    h.remove()
                self.model.zero_grad()

        if n_batches == 0:
            logger.warning(f"No batches processed for task '{task_name}'")
            return importances.numpy()

        return (importances / n_batches).numpy()

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
        """
        task_names = list(task_dataloaders.keys())
        matrix = np.zeros((self.n_layers, len(task_names)))

        for t_idx, task_name in enumerate(task_names):
            logger.info(f"Computing gradient importance for task: {task_name}")
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
            method="gradient",
            model_name=model_name,
            n_samples=n_samples,
            seq_len=seq_len,
        )


# ---------------------------------------------------------------------------
# Ablation Importance Analyzer
# ---------------------------------------------------------------------------

class _AttentionBypassContext:
    """
    Context manager that monkey-patches the forward of one decoder layer.

      - Transformer layers: the attention sub-layer is skipped (identity on the
        residual stream); the MLP sub-layer is kept.
      - Mamba / RWKV / Zamba2 layers: the whole block is replaced by identity.
    """

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self._original_forward = None

    def __enter__(self):
        layer = self.layer
        original_forward = layer.forward

        if hasattr(layer, "post_attention_layernorm") and hasattr(layer, "mlp"):
            # Transformer-style: skip attention, keep MLP + norm (sub-layer 2)
            def bypassed_forward(*args, **kwargs):
                hidden_states = args[0]
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                # Handle multi-GPU: residual and hidden_states may be on different devices
                return (residual.to(hidden_states.device) + hidden_states,)
        elif hasattr(layer, "attention") and hasattr(layer, "feed_forward") and not hasattr(layer, "self_attn"):
            # RWKV-style: bypass entire block, thread recurrent state through unchanged.
            # Rwkv6Block.forward returns (hidden_states, state); state must be passed back
            # so the outer Rwkv6Model loop can store it in state[i].
            def bypassed_forward(*args, **kwargs):
                hidden_states = args[0]
                state = kwargs.get("state", None)
                # Rwkv6Block always returns (hidden, state, attentions_or_None) — 3 elements
                return hidden_states, state, None
        elif "zamba" in type(layer).__name__.lower():
            # Any Zamba2 decoder layer (Zamba2MambaDecoderLayer or Zamba2AttentionDecoderLayer):
            # Zamba2Model.forward does `hidden_states = layer_outputs[0]`, so the bypass must
            # return a tuple. Returning a bare tensor causes layer_outputs[0] to index into
            # the tensor's first batch dimension (2D), breaking downstream shape checks.
            def bypassed_forward(*args, **kwargs):
                return (args[0],)
        else:
            # Mamba/SSM-style (original Mamba and Falcon Mamba): bypass entire block.
            # MambaForCausalLM block returns a plain tensor; FalconMambaDecoderLayer also
            # returns a plain tensor. The outer model handles both via isinstance(out, tuple).
            def bypassed_forward(*args, **kwargs):
                return args[0]

        self._original_forward = original_forward
        layer.forward = bypassed_forward
        return self

    def __exit__(self, *args):
        self.layer.forward = self._original_forward


class AblationImportanceAnalyzer:
    """
    Computes per-layer importance by measuring the increase in LM loss
    when each layer's attention sublayer is bypassed:

        I[l, t] = L_t(model with layer l bypassed) - L_t(baseline)

    This is a loss-based proxy (not accuracy-based) — fast and sufficient
    for layer ranking.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._layers = get_decoder_layers(model)
        self.n_layers = len(self._layers)

    @torch.no_grad()
    def _compute_loss(
        self,
        dataloader: DataLoader,
        device: torch.device,
        bypass_layer_idx: Optional[int] = None,
        desc: str = "",
    ) -> float:
        """Average causal LM loss over the calibration set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        ctx = (
            _AttentionBypassContext(self._layers[bypass_layer_idx])
            if bypass_layer_idx is not None
            else _null_context()
        )

        with ctx:
            for batch in tqdm(dataloader, desc=desc, leave=False):
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

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def compute_task_importance(
        self,
        dataloader: DataLoader,
        task_name: str,
        device: torch.device,
        layer_indices: Optional[List[int]] = None,
        baseline_loss: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute importance vector for one task.

        Args:
            dataloader:     Calibration DataLoader.
            task_name:      Label for progress bar.
            device:         CUDA device.
            layer_indices:  Subset of layer indices to ablate (None = all).
            baseline_loss:  Pre-computed baseline loss (avoids recomputation).

        Returns:
            (importance_vector [n_layers], baseline_loss)
        """
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))

        if baseline_loss is None:
            logger.info(f"  Computing baseline loss for task: {task_name}")
            baseline_loss = self._compute_loss(
                dataloader, device, bypass_layer_idx=None,
                desc=f"[ablation baseline] {task_name}",
            )

        importances = np.zeros(self.n_layers)

        for l_idx in tqdm(layer_indices, desc=f"[ablation layers] {task_name}", leave=False):
            ablated_loss = self._compute_loss(
                dataloader, device, bypass_layer_idx=l_idx,
                desc=f"  layer {l_idx}",
            )
            # Importance = increase in loss when this layer is removed
            importances[l_idx] = max(ablated_loss - baseline_loss, 0.0)

        return importances, baseline_loss

    def run(
        self,
        task_dataloaders: Dict[str, DataLoader],
        device: torch.device,
        layer_indices: Optional[List[int]] = None,
        layer_types: Optional[List[str]] = None,
        model_name: str = "unknown",
        n_samples: int = 0,
        seq_len: int = 0,
    ) -> ImportanceResult:
        """
        Compute the full importance matrix over all tasks.

        Tip: Pass layer_indices=[l] and run once per layer to parallelise
        across cluster jobs.  Merge results using merge_ablation_results().
        """
        task_names = list(task_dataloaders.keys())
        matrix = np.zeros((self.n_layers, len(task_names)))
        baseline_losses = {}

        for t_idx, task_name in enumerate(task_names):
            logger.info(f"Running ablation for task: {task_name}")
            col, b_loss = self.compute_task_importance(
                task_dataloaders[task_name],
                task_name,
                device,
                layer_indices=layer_indices,
            )
            matrix[:, t_idx] = col
            baseline_losses[task_name] = b_loss

        if layer_types is None:
            layer_types = ["unknown"] * self.n_layers

        return ImportanceResult(
            matrix=matrix,
            matrix_norm=_normalize_columns(matrix),
            layer_types=layer_types,
            task_names=task_names,
            method="ablation",
            model_name=model_name,
            n_samples=n_samples,
            seq_len=seq_len,
            extra={"baseline_losses": baseline_losses},
        )


# ---------------------------------------------------------------------------
# Utility: merge partial ablation results (one per layer_idx job)
# ---------------------------------------------------------------------------

def merge_ablation_results(partial_results: List[ImportanceResult]) -> ImportanceResult:
    """
    Merge a list of ImportanceResult objects produced by running AblationImportanceAnalyzer
    with different layer_indices subsets (e.g., one per cluster job).

    All partial results must share the same task_names, model_name, and n_layers.
    """
    assert len(partial_results) > 0
    ref = partial_results[0]
    n_layers = ref.n_layers
    n_tasks = ref.n_tasks

    merged_matrix = np.zeros((n_layers, n_tasks))
    for res in partial_results:
        assert res.task_names == ref.task_names, "task_names mismatch in partial results"
        assert res.n_layers == n_layers, "n_layers mismatch in partial results"
        # Each partial result has non-zero values only for its layer_indices
        merged_matrix += res.matrix  # non-overlapping layers, so addition = merge

    merged_extra = {}
    for res in partial_results:
        merged_extra.update(res.extra)

    return ImportanceResult(
        matrix=merged_matrix,
        matrix_norm=_normalize_columns(merged_matrix),
        layer_types=ref.layer_types,
        task_names=ref.task_names,
        method="ablation",
        model_name=ref.model_name,
        n_samples=ref.n_samples,
        seq_len=ref.seq_len,
        extra=merged_extra,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _null_context():
    yield
