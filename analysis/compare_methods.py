"""
Compare two layer-importance maps with per-task Spearman rank correlation.

For each task t shared by both results, computes rho_t = Spearman(A[:, t], B[:, t])
over layers, and reports the mean over tasks. With A = plasticity (LoRA delta or
full-FT delta) and B = necessity (ablation) this is the plasticity–necessity
alignment reported in the paper: negative in transformers, positive in SSMs.

Two files:
    python analysis/compare_methods.py \
        --a "results/lora_delta/qwen3-8b/lora_delta_*.json" \
        --b results/ablation/qwen3-8b/ablation_merged.json

All models in the standard results layout ({results_dir}/{method}/{slug}/...):
    python analysis/compare_methods.py --all --a_method lora_delta --b_method ablation
    python analysis/compare_methods.py --all --a_method full_ft_delta --b_method lora_delta
"""

import argparse
import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from layerimp.models import MODELS

# method name -> glob relative to {results_dir}/<subdir>/{slug}/
METHOD_FILES = {
    "ablation":      ("ablation",   "ablation_merged.json"),
    "gradient":      ("gradient",   "gradient_*.json"),
    "lora_delta":    ("lora_delta", "lora_delta_*.json"),
    "full_ft_delta": ("full_ft",    "full_ft_delta_*.json"),
    "full_ft_grad":  ("full_ft",    "full_ft_grad_*.json"),
    "probing":       ("probing",    "probing_*.json"),
    "fisher":        ("fisher",     "fisher_importance_*.json"),
}


def load(path_or_glob: str) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    matches = [path_or_glob] if os.path.exists(path_or_glob) else sorted(glob.glob(path_or_glob))
    if not matches:
        return None, None, None
    path = matches[-1]  # newest timestamp
    with open(path) as f:
        d = json.load(f)
    return np.array(d["matrix"], dtype=float), d["task_names"], path


def per_task_spearman(A: np.ndarray, ta: List[str], B: np.ndarray, tb: List[str]) -> dict:
    n = min(A.shape[0], B.shape[0])
    out = {}
    for t in ta:
        if t in tb:
            out[t] = float(spearmanr(A[:n, ta.index(t)], B[:n, tb.index(t)]).statistic)
    return out


def method_glob(results_dir: str, method: str, slug: str) -> str:
    subdir, pattern = METHOD_FILES[method]
    return os.path.join(results_dir, subdir, slug, pattern)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", help="Importance JSON (glob OK).")
    ap.add_argument("--b", help="Importance JSON (glob OK).")
    ap.add_argument("--all", action="store_true", help="Loop over all models in the results layout.")
    ap.add_argument("--a_method", default="lora_delta", choices=list(METHOD_FILES))
    ap.add_argument("--b_method", default="ablation", choices=list(METHOD_FILES))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    if args.all:
        pairs = [(m, method_glob(args.results_dir, args.a_method, m),
                  method_glob(args.results_dir, args.b_method, m)) for m in args.models]
    elif args.a and args.b:
        pairs = [("custom", args.a, args.b)]
    else:
        ap.error("Provide --a and --b, or --all.")

    rows, by_family = [], {}
    print(f"{'model':<16} {'family':<12} {'mean rho':>9} {'min':>7} {'max':>7} {'#tasks':>7}")
    for slug, pa, pb in pairs:
        A, ta, fa = load(pa)
        B, tb, fb = load(pb)
        if A is None or B is None:
            if not args.all:
                raise FileNotFoundError(pa if A is None else pb)
            continue
        rhos = per_task_spearman(A, ta, B, tb)
        vals = np.array([v for v in rhos.values() if not np.isnan(v)])
        family = MODELS[slug][1] if slug in MODELS else "-"
        mean = float(vals.mean()) if len(vals) else float("nan")
        by_family.setdefault(family, []).append(mean)
        rows.append({"model": slug, "family": family, "a": fa, "b": fb,
                     "mean_rho": mean, "per_task": rhos})
        print(f"{slug:<16} {family:<12} {mean:>+9.3f} {vals.min():>+7.2f} {vals.max():>+7.2f} {len(vals):>7}")

    if len(rows) > 1:
        print()
        for fam, v in by_family.items():
            print(f"{fam + ' mean':<29} {np.nanmean(v):>+9.3f}  (n={len(v)})")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()
