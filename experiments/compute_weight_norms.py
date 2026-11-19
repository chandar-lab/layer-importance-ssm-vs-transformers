"""
Per-layer pretrained weight norms ||W_l||_F, for update-to-weight normalised
plasticity:  plasticity_norm[l, t] = ||Delta W_l||_F / ||W_l||_F.

Two denominators are saved per layer:
    norm_all    — all parameters of decoder layer l (the full-FT trainable set)
    norm_target — LoRA target-module parameters only (the set LoRA updates)

No forward pass; runs on CPU if no GPU is available.

    python experiments/compute_weight_norms.py --model_name_or_path Qwen/Qwen3-0.6B --slug qwen3-0.6b
    -> results/full_ft/qwen3-0.6b/layer_weight_norms.json
"""

import argparse
import json
import logging
import os

import torch

from layerimp.models import default_target_modules, get_decoder_layers, load_model_and_tokenizer
from layerimp.utils import setup_logging

setup_logging()
logger = logging.getLogger("weight_norms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output_root", default="results/full_ft")
    p.add_argument("--hf_cache_dir", default=None)
    p.add_argument("--target_modules", nargs="+", default=None,
                   help="Override the default LoRA target-module list "
                        "(match the model's actual lora_delta run)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        # The state-spaces loading path calls model.cuda(); norms only need CPU.
        torch.nn.Module.cuda = lambda self, *a, **k: self
    model, _ = load_model_and_tokenizer(
        args.model_name_or_path, dtype=args.dtype, attn_implementation="eager",
        cache_dir=args.hf_cache_dir,
    )

    layers = get_decoder_layers(model)
    target_modules = args.target_modules or default_target_modules(model)

    norm_all, norm_target = [], []
    with torch.no_grad():
        for layer in layers:
            target_param_ids = set()
            for mod_name, module in layer.named_modules():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in target_modules:
                    for q in module.parameters(recurse=False):
                        target_param_ids.add(id(q))
            sq_all = sq_tgt = 0.0
            for q in layer.parameters():
                s = float(q.detach().float().pow(2).sum())
                sq_all += s
                if id(q) in target_param_ids:
                    sq_tgt += s
            norm_all.append(sq_all ** 0.5)
            norm_target.append(sq_tgt ** 0.5)

    out_dir = os.path.join(args.output_root, args.slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "layer_weight_norms.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_name": args.model_name_or_path,
                "dtype": args.dtype,
                "n_layers": len(layers),
                "target_modules": target_modules,
                "norm_all": norm_all,
                "norm_target": norm_target,
            },
            f,
            indent=2,
        )
    logger.info(f"{args.slug}: {len(layers)} layers -> {out_path}")


if __name__ == "__main__":
    main()
