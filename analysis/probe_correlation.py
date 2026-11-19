"""
Correlation between linear-probe accuracy and layer importance.

After running experiments/run_probing.py on a model, this script loads:
  1. The probe accuracy matrix P[L × T]  (from probing output)
  2. The LoRA delta importance matrix    (from lora_delta output)
  3. The ablation importance matrix      (from experiments/run_ablation.py)

And computes per-task Spearman rank correlations:
  - rho(P[:, t], I_delta[:, t])   → does probe accuracy align with LoRA delta?
  - rho(P[:, t], I_abl[:, t])     → does probe accuracy align with ablation?

Hypothesis: In transformers, probe accuracy aligns with LoRA delta (both peak
at terminal layers). In SSMs, alignment is more uniform across methods.

Usage:
    # All models, standard results layout ({results_dir}/{method}/{slug}/...):
    python analysis/probe_correlation.py --all

    # Explicit files:
    python analysis/probe_correlation.py \\
        --probe_json "results/probing/qwen3-8b/probing_*.json" \\
        --lora_json "results/lora_delta/qwen3-8b/lora_delta_*.json" \\
        --ablation_json results/ablation/qwen3-8b/ablation_merged.json \\
        --model_label qwen3-8b --family Transformer
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from layerimp.models import MODELS

# Models included in the probing analysis of the paper.
DEFAULT_MODELS = [
    "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    "llama-3.2-3b", "llama-3.1-8b",
    "mamba-370m", "mamba-790m", "mamba-2.8b",
]


def registry_entry(slug: str, results_dir: str) -> Dict:
    return {
        "family":   MODELS[slug][1],
        "probe":    f"{results_dir}/probing/{slug}/probing_*.json",
        "lora":     f"{results_dir}/lora_delta/{slug}/lora_delta_*.json",
        "ablation": f"{results_dir}/ablation/{slug}/ablation_merged.json",
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_matrix(path_glob: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Load matrix_norm from a JSON file (supports glob)."""
    files = sorted(glob.glob(path_glob))
    if not files:
        return None, None
    with open(files[-1]) as f:
        d = json.load(f)
    mat = d.get("matrix_norm") or d.get("matrix")
    if mat is None:
        return None, None
    return np.array(mat), d.get("task_names", [])


def align_tasks(
    mat_a: np.ndarray, tasks_a: List[str],
    mat_b: np.ndarray, tasks_b: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return columns of mat_a and mat_b that share the same task names."""
    common = [t for t in tasks_a if t in tasks_b]
    if not common:
        return None, None, []
    idx_a = [tasks_a.index(t) for t in common]
    idx_b = [tasks_b.index(t) for t in common]
    n = min(mat_a.shape[0], mat_b.shape[0])
    return mat_a[:n, :][:, idx_a], mat_b[:n, :][:, idx_b], common


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyse_one(
    model_label: str,
    family: str,
    probe_mat: np.ndarray,
    probe_tasks: List[str],
    lora_mat:  np.ndarray,
    lora_tasks: List[str],
    abl_mat:   np.ndarray,
    abl_tasks:  List[str],
) -> Dict:
    """
    Compute per-task correlations between probe accuracy and each importance method.
    Also computes overall probe peak layer vs importance peak layer agreement.
    """
    # Align all three matrices to common tasks
    probe_a, lora_a, common1 = align_tasks(probe_mat, probe_tasks, lora_mat, lora_tasks)
    if probe_a is None:
        return {"error": "no common tasks between probe and lora_delta"}

    probe_b, abl_a, common2 = align_tasks(probe_mat, probe_tasks, abl_mat, abl_tasks)
    if probe_b is None:
        return {"error": "no common tasks between probe and ablation"}

    # Common tasks across all three
    common = [t for t in common1 if t in common2]
    idx1 = [common1.index(t) for t in common]
    idx2 = [common2.index(t) for t in common]
    p1 = probe_a[:, idx1]
    l1 = lora_a[:, idx1]
    p2 = probe_b[:, idx2]
    a1 = abl_a[:, idx2]

    n_layers = p1.shape[0]
    results = {}

    rho_probe_lora_list = []
    rho_probe_abl_list  = []

    for i, task in enumerate(common):
        rho_pl = spearmanr(p1[:, i], l1[:, i]).statistic
        rho_pa = spearmanr(p2[:, i], a1[:, i]).statistic
        rho_probe_lora_list.append(rho_pl)
        rho_probe_abl_list.append(rho_pa)

        # Peak layer agreement: does argmax(probe) == argmax(lora_delta)?
        peak_probe = int(np.argmax(p1[:, i]))
        peak_lora  = int(np.argmax(l1[:, i]))
        peak_abl   = int(np.argmax(a1[:, i]))

        results[task] = {
            "rho_probe_vs_lora":    round(float(rho_pl), 4),
            "rho_probe_vs_ablation": round(float(rho_pa), 4),
            "peak_probe_layer":     peak_probe,
            "peak_lora_layer":      peak_lora,
            "peak_abl_layer":       peak_abl,
            "peak_agrees_with_lora": abs(peak_probe - peak_lora) <= 2,
            "peak_agrees_with_abl":  abs(peak_probe - peak_abl)  <= 2,
        }

    return {
        "model":                    model_label,
        "family":                   family,
        "n_layers":                 n_layers,
        "n_tasks":                  len(common),
        "mean_rho_probe_vs_lora":   round(float(np.nanmean(rho_probe_lora_list)), 4),
        "mean_rho_probe_vs_abl":    round(float(np.nanmean(rho_probe_abl_list)),  4),
        "per_task":                 results,
        # Key metric: does probe align more with LoRA delta or ablation?
        "probe_aligns_with": (
            "lora_delta"
            if np.nanmean(rho_probe_lora_list) > np.nanmean(rho_probe_abl_list)
            else "ablation"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Probe-importance correlation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--all", action="store_true",
                   help="Run all --models from the standard results layout.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--model_label", type=str, default=None,
                   help="Single model slug (standard layout), or a label for explicit files.")
    p.add_argument("--probe_json",    type=str, default=None)
    p.add_argument("--lora_json",     type=str, default=None)
    p.add_argument("--ablation_json", type=str, default=None)
    p.add_argument("--family",        type=str, default=None,
                   choices=["Transformer", "SSM", "RNN", "Hybrid"])
    p.add_argument("--output_dir",    type=str,
                   default="results/analysis/probe_correlation")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build work list
    if args.all:
        work = [(m, registry_entry(m, args.results_dir)) for m in args.models]
    elif args.model_label and not args.probe_json:
        work = [(args.model_label, registry_entry(args.model_label, args.results_dir))]
    else:
        if not (args.probe_json and args.lora_json and args.ablation_json and args.family):
            p.error("Provide --all, --model_label, or all of "
                    "--probe_json --lora_json --ablation_json --family --model_label")
        work = [(args.model_label or "custom", {
            "family":   args.family,
            "probe":    args.probe_json,
            "lora":     args.lora_json,
            "ablation": args.ablation_json,
        })]

    all_results = []
    trans_rows, ssm_rows = [], []

    print(f"\n{'Model':<18} {'Family':<12} {'ρ(probe,Δ)':>12} {'ρ(probe,abl)':>13} {'Aligns with':<14}")
    print("─" * 75)

    for model_label, cfg in work:
        family = cfg["family"]
        probe_mat, probe_tasks   = load_matrix(cfg["probe"])
        lora_mat,  lora_tasks    = load_matrix(cfg["lora"])
        abl_mat,   abl_tasks     = load_matrix(cfg["ablation"])

        if any(m is None for m in [probe_mat, lora_mat, abl_mat]):
            missing = [k for k, m in [("probe", probe_mat), ("lora", lora_mat),
                                       ("ablation", abl_mat)] if m is None]
            print(f"  {model_label:<16} SKIP (missing: {missing})")
            continue

        row = analyse_one(
            model_label, family,
            probe_mat, probe_tasks,
            lora_mat,  lora_tasks,
            abl_mat,   abl_tasks,
        )
        if "error" in row:
            print(f"  {model_label:<16} ERROR: {row['error']}")
            continue

        all_results.append(row)
        if family == "Transformer":
            trans_rows.append(row)
        else:
            ssm_rows.append(row)

        print(f"  {model_label:<16} {family:<12} "
              f"{row['mean_rho_probe_vs_lora']:>12.3f} "
              f"{row['mean_rho_probe_vs_abl']:>13.3f} "
              f"{row['probe_aligns_with']:<14}")

    print("─" * 75)
    if trans_rows:
        m_l = np.mean([r["mean_rho_probe_vs_lora"] for r in trans_rows])
        m_a = np.mean([r["mean_rho_probe_vs_abl"]  for r in trans_rows])
        print(f"  {'Transformer mean':<28} {m_l:>12.3f} {m_a:>13.3f}")
    if ssm_rows:
        m_l = np.mean([r["mean_rho_probe_vs_lora"] for r in ssm_rows])
        m_a = np.mean([r["mean_rho_probe_vs_abl"]  for r in ssm_rows])
        print(f"  {'SSM mean':<28} {m_l:>12.3f} {m_a:>13.3f}")

    print("\nInterpretation:")
    print("  ρ(probe, LoRA delta) > 0  → terminal-layer specialization (layers that")
    print("  change most also have highest task-discriminative representations)")
    print("  ρ(probe, ablation) > 0    → early-layer necessity (layers that break the")
    print("  model when removed are also most representationally distinctive)")

    # Save
    out_path = os.path.join(args.output_dir, "probe_correlation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
