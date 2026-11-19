"""
Model loading and architecture helpers shared by every experiment.

Supported families (all loaded through HuggingFace ``transformers``):
  - Transformers : Qwen3, Llama 3.x
  - SSMs         : Mamba (state-spaces), Mamba-2 (state-spaces), Falcon-Mamba
  - Linear RNN   : RWKV-6
  - Hybrid       : Zamba2
"""

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# ---------------------------------------------------------------------------
# Model registry used in the paper (slug -> HF id, family)
# ---------------------------------------------------------------------------

MODELS = {
    # Transformers
    "qwen3-0.6b":      ("Qwen/Qwen3-0.6B",              "Transformer"),
    "qwen3-1.7b":      ("Qwen/Qwen3-1.7B",              "Transformer"),
    "qwen3-4b":        ("Qwen/Qwen3-4B",                "Transformer"),
    "qwen3-8b":        ("Qwen/Qwen3-8B",                "Transformer"),
    "qwen3-14b":       ("Qwen/Qwen3-14B",               "Transformer"),
    "qwen3-32b":       ("Qwen/Qwen3-32B",               "Transformer"),
    "llama-3.2-1b":    ("meta-llama/Llama-3.2-1B",      "Transformer"),
    "llama-3.2-3b":    ("meta-llama/Llama-3.2-3B",      "Transformer"),
    "llama-3.1-8b":    ("meta-llama/Llama-3.1-8B",      "Transformer"),
    "llama-3.1-70b":   ("meta-llama/Llama-3.1-70B",     "Transformer"),
    # State-space models
    "mamba-130m":      ("state-spaces/mamba-130m",      "SSM"),
    "mamba-370m":      ("state-spaces/mamba-370m",      "SSM"),
    "mamba-790m":      ("state-spaces/mamba-790m",      "SSM"),
    "mamba-2.8b":      ("state-spaces/mamba-2.8b",      "SSM"),
    "mamba2-1.3b":     ("state-spaces/mamba2-1.3b",     "SSM"),
    "falcon-mamba-7b": ("tiiuae/falcon-mamba-7b",       "SSM"),
    # Linear RNN / hybrid
    "rwkv6-3b":        ("RWKV/rwkv-6-world-3b",         "RNN"),
    "zamba2-2.7b":     ("Zyphra/Zamba2-2.7B",           "Hybrid"),
}


# ---------------------------------------------------------------------------
# state-spaces Mamba checkpoints
# ---------------------------------------------------------------------------
# The original state-spaces/mamba-* and mamba2-* repos are not in HF format:
#   * config.json uses legacy field names (d_model, n_layer) that transformers
#     silently ignores, so the configs are given explicitly here;
#   * the embedding key is 'backbone.embedding.' instead of 'backbone.embeddings.';
#   * lm_head.weight is tied and absent from the checkpoint;
#   * the weights use a padded vocabulary (50280 / 50288);
#   * there is no tokenizer.json -> we use EleutherAI/gpt-neox-20b, the tokenizer
#     Mamba was trained with.

_MAMBA_CONFIGS = {
    "mamba-130m": {"vocab_size": 50280, "hidden_size": 768,  "num_hidden_layers": 24},
    "mamba-370m": {"vocab_size": 50280, "hidden_size": 1024, "num_hidden_layers": 48},
    "mamba-790m": {"vocab_size": 50280, "hidden_size": 1536, "num_hidden_layers": 48},
    "mamba-2.8b": {"vocab_size": 50280, "hidden_size": 2560, "num_hidden_layers": 64},
}
_MAMBA2_CONFIGS = {
    "mamba2-1.3b": {
        "hidden_size": 2048, "num_hidden_layers": 48, "vocab_size": 50288,
        "num_heads": 64, "head_dim": 64, "state_size": 128,
        "conv_kernel": 4, "expand": 2, "n_groups": 1,
    },
}
_MAMBA_TOKENIZER = "EleutherAI/gpt-neox-20b"


def is_state_spaces_checkpoint(model_name_or_path: str) -> bool:
    return "state-spaces" in model_name_or_path.lower()


def _load_state_spaces_mamba(
    model_name_or_path: str,
    dtype: torch.dtype,
    cache_dir: Optional[str],
    upcast: bool,
) -> nn.Module:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    tag = os.path.basename(model_name_or_path.rstrip("/")).lower()
    if tag in _MAMBA2_CONFIGS:
        from transformers import Mamba2Config, Mamba2ForCausalLM
        model = Mamba2ForCausalLM(Mamba2Config(**_MAMBA2_CONFIGS[tag]))
    elif tag in _MAMBA_CONFIGS:
        from transformers import MambaConfig, MambaForCausalLM
        model = MambaForCausalLM(MambaConfig(**_MAMBA_CONFIGS[tag]))
    else:
        raise ValueError(
            f"Unknown state-spaces checkpoint '{model_name_or_path}'. "
            f"Known: {sorted(_MAMBA_CONFIGS) + sorted(_MAMBA2_CONFIGS)}"
        )
    if not upcast:
        model = model.to(dtype)

    if os.path.isdir(model_name_or_path):
        local_dir = model_name_or_path
    else:
        local_dir = snapshot_download(
            model_name_or_path,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
    files = sorted(os.listdir(local_dir))
    st_files = [f for f in files if f.endswith(".safetensors")]
    bin_files = [f for f in files if f.endswith(".bin") and "pytorch_model" in f]

    def _rename(k: str) -> str:
        return k.replace("backbone.embedding.", "backbone.embeddings.")

    state_dict = {}
    if st_files:
        for fname in st_files:
            with safe_open(os.path.join(local_dir, fname), framework="pt") as f:
                for k in f.keys():
                    state_dict[_rename(k)] = f.get_tensor(k).to(dtype)
    elif bin_files:
        for fname in bin_files:
            raw = torch.load(os.path.join(local_dir, fname), map_location="cpu", weights_only=False)
            for k, v in raw.items():
                state_dict[_rename(k)] = v.to(dtype)
    else:
        raise FileNotFoundError(f"No .safetensors / pytorch_model*.bin in {local_dir}: {files}")

    if "lm_head.weight" not in state_dict and "backbone.embeddings.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["backbone.embeddings.weight"]

    model.load_state_dict(state_dict, strict=True)
    return model.cuda()


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    cache_dir: Optional[str] = None,
    device_map: str = "auto",
    gradient_checkpointing: bool = False,
    upcast_state_spaces: bool = False,
    pad_to_eos: bool = True,
) -> Tuple[nn.Module, "AutoTokenizer"]:
    """
    Load any model from :data:`MODELS` (or any standard HF causal LM).

    Args:
        model_name_or_path:  HF hub id or local path.
        dtype:               "bfloat16" | "float16" | "float32".
        attn_implementation: Passed to transformers for attention-based models
                             (ignored for Mamba / RWKV).
        cache_dir:           Optional HF cache directory.
        device_map:          Passed to ``from_pretrained``.
        gradient_checkpointing: Enable gradient checkpointing.
        upcast_state_spaces: For state-spaces Mamba checkpoints only: round the
                             weights to ``dtype`` but keep the module in float32.
                             This is the setting used for the causal-validation
                             and continual-learning experiments in the paper.
        pad_to_eos:          Always use EOS as the pad token. If False, EOS is only
                             used when the tokenizer has no pad token.
    """
    torch_dtype = DTYPES[dtype]
    name_lower = model_name_or_path.lower()
    state_spaces = is_state_spaces_checkpoint(model_name_or_path)
    # Mamba-family and RWKV models do not accept `attn_implementation`.
    no_attn_impl = "mamba" in name_lower or "rwkv" in name_lower

    logger.info(
        f"Loading model: {model_name_or_path} "
        f"[{dtype}, {'n/a' if no_attn_impl else attn_implementation}]"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        _MAMBA_TOKENIZER if state_spaces else model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if pad_to_eos or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if state_spaces:
        model = _load_state_spaces_mamba(model_name_or_path, torch_dtype, cache_dir, upcast_state_spaces)
    else:
        kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        if not no_attn_impl:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    model.eval()
    return model, tokenizer


def load_from_args(args, **overrides):
    """:func:`load_model_and_tokenizer` driven by the common CLI arguments."""
    kwargs = dict(
        model_name_or_path=args.model_name_or_path,
        dtype=getattr(args, "dtype", "bfloat16"),
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        cache_dir=getattr(args, "hf_cache_dir", None),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
    )
    kwargs.update(overrides)
    return load_model_and_tokenizer(**kwargs)


# ---------------------------------------------------------------------------
# Architecture introspection
# ---------------------------------------------------------------------------

def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Return the stack of residual blocks of an (un-wrapped) causal LM.

        Qwen3/Llama/Zamba2ForCausalLM .model.layers
        Mamba/Mamba2/FalconMambaForCausalLM .backbone.layers
        Rwkv6ForCausalLM .rwkv.blocks
    """
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return inner.model.layers
    if hasattr(inner, "backbone") and hasattr(inner.backbone, "layers"):
        return inner.backbone.layers
    if hasattr(inner, "rwkv") and hasattr(inner.rwkv, "blocks"):
        return inner.rwkv.blocks
    if hasattr(inner, "blocks"):
        return inner.blocks
    raise AttributeError(f"Cannot locate decoder layers in {type(model).__name__}.")


def get_layer_types(model: nn.Module) -> List[str]:
    """Per-layer type annotation (used for plots/metadata only)."""
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "layer_types", None) is not None:
        return list(cfg.layer_types)
    types = []
    for layer in get_decoder_layers(model):
        if hasattr(layer, "self_attn"):
            types.append(type(layer.self_attn).__name__.lower())
        elif hasattr(layer, "mixer"):
            types.append("mamba")
        elif hasattr(layer, "mamba"):
            types.append("falcon_mamba")
        elif hasattr(layer, "attention"):
            types.append("rwkv")
        else:
            types.append("unknown")
    return types


def is_mamba(model: nn.Module) -> bool:
    """True for Mamba / Mamba-2 / Falcon-Mamba."""
    cfg = getattr(model, "config", None)
    if cfg is not None and "mamba" in getattr(cfg, "model_type", "").lower():
        return True
    inner = model.model if hasattr(model, "model") else model
    return hasattr(inner, "backbone") and hasattr(inner.backbone, "layers")


def is_rwkv(model: nn.Module) -> bool:
    cfg = getattr(model, "config", None)
    if cfg is not None and "rwkv" in getattr(cfg, "model_type", "").lower():
        return True
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "rwkv") and hasattr(inner.rwkv, "blocks"):
        return True
    return hasattr(inner, "blocks") and not hasattr(inner, "layers")


def default_target_modules(model: nn.Module) -> List[str]:
    """
    LoRA target modules used in the paper.

      RWKV         : receptance, key, value, output, gate
      Mamba family : in_proj, x_proj, dt_proj   (PEFT blocks out_proj / conv1d)
      Transformers : gate_proj, up_proj, down_proj  (MLP projections)
    """
    if is_rwkv(model):
        return ["receptance", "key", "value", "output", "gate"]
    if is_mamba(model):
        return ["in_proj", "x_proj", "dt_proj"]
    return ["gate_proj", "up_proj", "down_proj"]


def sanitize_tied_weights_keys(model: nn.Module) -> None:
    """Zamba2 stores compiled regexes in ``_tied_weights_keys``; PEFT expects strings."""
    if getattr(model, "_tied_weights_keys", None):
        model._tied_weights_keys = [k for k in model._tied_weights_keys if isinstance(k, str)]
