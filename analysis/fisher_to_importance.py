"""
Convert fisher_per_layer.json (experiments/run_fisher.py) into an ImportanceResult
JSON, so Fisher curvature can be used as a layer-selection signal (e.g. by
experiments/run_cl_forgetting.py --importance_json).

    python analysis/fisher_to_importance.py --slug llama-3.2-3b --model_name_or_path meta-llama/Llama-3.2-3B
    -> results/fisher/llama-3.2-3b/fisher_importance_llama-3.2-3b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug",   required=True, help="Model slug, e.g. llama-3.2-3b")
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    fisher_path = Path(args.results_dir) / "fisher" / args.slug / "fisher_per_layer.json"
    if not fisher_path.exists():
        print(f"Fisher file not found: {fisher_path}")
        sys.exit(1)

    fd = json.load(open(fisher_path))
    fisher_raw  = np.array(fd["fisher_per_layer"])   # (n_tasks, n_layers)
    task_names  = fd["task_names"]
    n_layers    = fd["n_layers"]

    # Transpose to (n_layers, n_tasks), replace NaN with 0
    matrix = fisher_raw.T[:n_layers, :]              # (n_layers, n_tasks)
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Column-normalize (each task column sums to 1)
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    matrix_norm = matrix / col_sums

    result = {
        "matrix":      matrix.tolist(),
        "matrix_norm": matrix_norm.tolist(),
        "layer_types": ["layer"] * n_layers,
        "task_names":  task_names,
        "method":      "fisher",
        "model_name":  args.model_name_or_path,
        "n_samples":   fd.get("n_examples", 64),
        "seq_len":     fd.get("seq_len", 128),
    }

    out_dir  = fisher_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fisher_importance_{args.slug}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_path}  ({n_layers} layers × {len(task_names)} tasks)")


if __name__ == "__main__":
    main()
