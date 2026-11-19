"""
Coefficient of variation (CV) of ablation importance across layers.

Tests whether SSM layers are more functionally homogeneous than transformer
layers by computing the Coefficient of Variation (CV) of ablation importance
scores across layers, per model and task.

Hypothesis: CV(SSM) < CV(Transformer) → SSM interior layers are
interchangeable; transformer layers are functionally differentiated.

Uses existing ablation_merged.json files — no new experiments needed.

Usage:
    python analysis/cv_analysis.py --output_dir results/analysis/cv
"""

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np

from layerimp.models import MODELS

# Models included in the CV analysis of the paper.
DEFAULT_MODELS = [
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    "llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.1-70b",
    "mamba-130m", "mamba-370m", "mamba-790m", "mamba-2.8b",
]


def cv(x: np.ndarray) -> float:
    """Coefficient of Variation = std / mean. Returns nan if mean≈0."""
    m = np.mean(x)
    if abs(m) < 1e-10:
        return float("nan")
    return float(np.std(x) / m)


def load_ablation(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="CV analysis of ablation importance homogeneity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model slugs. Available: {list(MODELS)}")
    parser.add_argument("--output_dir", type=str, default="results/analysis/cv")
    parser.add_argument("--exclude_l0", action="store_true",
                        help="Exclude Layer 0 (embedding) from CV computation "
                             "to avoid L0-dominance artifact.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    trans_cvs, ssm_cvs = [], []

    print(f"\n{'Model':<18} {'Family':<12} {'Mean CV':>9} {'Min CV':>9} "
          f"{'Max CV':>9} {'L0 share':>10} {'N layers':>9}")
    print("─" * 85)

    for model_label in args.models:
        family = MODELS[model_label][1]
        abl_path = os.path.join(args.results_dir, "ablation", model_label, "ablation_merged.json")
        d = load_ablation(abl_path)
        if d is None:
            print(f"  {model_label:<16} MISSING")
            continue

        mat = np.array(d["matrix_norm"])        # [n_layers, n_tasks]
        task_names = d.get("task_names", [])
        n_layers, n_tasks = mat.shape

        # L0 share: how much of the total importance mass sits in layer 0
        l0_share = float(np.mean(mat[0, :]))    # mean across tasks

        # CV per task (optionally excluding L0)
        src = mat[1:] if args.exclude_l0 else mat
        task_cvs = [cv(src[:, t]) for t in range(n_tasks)]
        mean_cv = float(np.nanmean(task_cvs))
        min_cv  = float(np.nanmin(task_cvs))
        max_cv  = float(np.nanmax(task_cvs))

        per_task = {task_names[t]: round(task_cvs[t], 3) for t in range(n_tasks)}

        rows.append({
            "model":    model_label,
            "family":   family,
            "n_layers": n_layers,
            "mean_cv":  mean_cv,
            "min_cv":   min_cv,
            "max_cv":   max_cv,
            "l0_share": l0_share,
            "per_task": per_task,
            "exclude_l0": args.exclude_l0,
        })

        if family == "Transformer":
            trans_cvs.append(mean_cv)
        else:
            ssm_cvs.append(mean_cv)

        print(f"  {model_label:<16} {family:<12} {mean_cv:>9.3f} {min_cv:>9.3f} "
              f"{max_cv:>9.3f} {l0_share:>10.3f} {n_layers:>9}")

    print("─" * 85)
    if trans_cvs:
        print(f"  {'Transformer mean':<28} {np.mean(trans_cvs):>9.3f}")
    if ssm_cvs:
        print(f"  {'SSM mean':<28} {np.mean(ssm_cvs):>9.3f}")

    # Hypothesis test: Mann-Whitney U (non-parametric, SSM CV < Transformer CV)
    if trans_cvs and ssm_cvs:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(ssm_cvs, trans_cvs, alternative="less")
        print(f"\n  Mann-Whitney U (SSM < Trans): U={stat:.1f}, p={p:.4f}")
        if p < 0.05:
            print("  ✓ Hypothesis SUPPORTED: SSM layers are significantly more homogeneous")
        else:
            print("  ✗ Hypothesis NOT supported at p<0.05")

    # Save
    out_path = os.path.join(args.output_dir, "cv_analysis.json")
    with open(out_path, "w") as f:
        json.dump({
            "rows": rows,
            "transformer_mean_cv": float(np.mean(trans_cvs)) if trans_cvs else None,
            "ssm_mean_cv":         float(np.mean(ssm_cvs))   if ssm_cvs   else None,
            "exclude_l0":          args.exclude_l0,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
