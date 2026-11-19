"""
Update-to-weight normalised plasticity.

For each model with a layer_weight_norms.json (experiments/compute_weight_norms.py),
recompute the plasticity-vs-necessity correlations with per-layer normalised
plasticity:

    full-FT:  ||dW_l|| / ||W_l||        (norm_all — matches trainable set)
    LoRA:     ||dW_l|| / ||W_l^target|| (norm_target — the params LoRA updates)

Prints mean per-task Spearman vs ablation necessity, raw vs normalized.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr


def load(path, key="matrix"):
    d = json.load(open(path))
    return np.array(d[key]), d["task_names"]


def mean_rho(A, ta, B, tb):
    common = [t for t in ta if t in tb]
    rhos = [spearmanr(A[:, ta.index(t)], B[:, tb.index(t)]).statistic for t in common]
    rhos = [r for r in rhos if not np.isnan(r)]
    return np.mean(rhos) if rhos else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    R = ap.parse_args().results_dir

    header = (
        f"{'model':<14} {'FFT raw':>8} {'FFT/||W||':>9} "
        f"{'LoRA raw':>9} {'LoRA/||W||':>10} {'FFTn~LoRAn':>11}"
    )
    print("mean per-task Spearman rho(plasticity, ablation necessity)")
    print(header)
    for wn_path in sorted(glob.glob(f"{R}/full_ft/*/layer_weight_norms.json")):
        slug = os.path.basename(os.path.dirname(wn_path))
        wn = json.load(open(wn_path))
        w_all = np.array(wn["norm_all"])[:, None]
        w_tgt = np.array(wn["norm_target"])[:, None]

        abl_path = f"{R}/ablation/{slug}/ablation_merged.json"
        if not os.path.exists(abl_path):
            print(f"{slug:<14} (no ablation baseline)")
            continue
        A, at = load(abl_path)

        row = {"fft": np.nan, "fftn": np.nan, "lora": np.nan, "loran": np.nan,
               "cross": np.nan}
        Fn = Ln = None
        fft = sorted(glob.glob(
            f"{R}/full_ft/{slug}/full_ft_delta_*.json"))
        if fft:
            F, ft = load(fft[0])
            Fn = F / w_all
            row["fft"] = mean_rho(F, ft, A, at)
            row["fftn"] = mean_rho(Fn, ft, A, at)
        lora = sorted(glob.glob(f"{R}/lora_delta/{slug}/lora_delta_*.json"))
        if lora:
            L, lt = load(lora[0])
            Ln = L / w_tgt
            row["lora"] = mean_rho(L, lt, A, at)
            row["loran"] = mean_rho(Ln, lt, A, at)
        if Fn is not None and Ln is not None:
            row["cross"] = mean_rho(Fn, ft, Ln, lt)

        print(
            f"{slug:<14} {row['fft']:>8.3f} {row['fftn']:>9.3f} "
            f"{row['lora']:>9.3f} {row['loran']:>10.3f} {row['cross']:>11.3f}"
        )


if __name__ == "__main__":
    main()
