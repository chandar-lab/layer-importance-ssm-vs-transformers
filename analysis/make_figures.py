"""
Main-paper figures.

Reads results in the standard layout ({results_dir}/{method}/{slug}/...) written
by the experiment scripts and writes PDF + PNG files to --output_dir.

    python analysis/make_figures.py --results_dir results --output_dir figures
"""

import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NeurIPS-friendly defaults
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'text.usetex': False,
})

RESULTS = 'results'
OUTDIR = 'figures'

TASK_DISPLAY = {
    'mmlu': 'MMLU', 'gsm8k': 'GSM8K', 'arc_challenge': 'ARC',
    'piqa': 'PIQA', 'humaneval': 'HumanEval', 'mbpp': 'MBPP',
    'squad': 'SQuAD', 'winogrande': 'WinoGrande', 'gpqa': 'GPQA',
    'longbench_single_doc': 'LongBench',
}

TASK_COLORS = {
    'Code': '#2171b5',
    'Commonsense': '#cb181d',
    'Math': '#238b45',
    'Knowledge': '#6a51a3',
    'Science': '#d94801',
    'Retrieval': '#737373',
}

TASK_CATEGORY = {
    'humaneval': 'Code', 'mbpp': 'Code',
    'arc_challenge': 'Commonsense', 'piqa': 'Commonsense', 'winogrande': 'Commonsense',
    'gsm8k': 'Math', 'mmlu': 'Knowledge', 'gpqa': 'Science',
    'squad': 'Retrieval', 'longbench_single_doc': 'Retrieval',
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def latest(pattern):
    """Last match (sorted by name, i.e. newest timestamp) of a glob pattern."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern}")
    return matches[-1]


# ============================================================
# Figure 1: Qwen3-8B LoRA delta heatmap (terminal cluster + LongBench isolation)
# ============================================================
def figure1():
    data = load_json(latest(f'{RESULTS}/lora_delta/qwen3-8b/lora_delta_*_r8_s200_*.json'))
    matrix = np.array(data['matrix'])  # (36, 10)
    tasks = data['task_names']
    n_layers = matrix.shape[0]

    # Reorder tasks: code, retrieval, math, knowledge, science, commonsense
    order = ['humaneval', 'mbpp', 'longbench_single_doc', 'squad', 'gsm8k', 'mmlu', 'gpqa', 'arc_challenge', 'piqa', 'winogrande']
    idx = [tasks.index(t) for t in order]
    matrix_ordered = matrix[:, idx]
    task_labels = [TASK_DISPLAY[t] for t in order]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(matrix_ordered.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xlabel('Layer index')
    ax.set_ylabel('')
    ax.set_yticks(range(len(task_labels)))
    ax.set_yticklabels(task_labels)

    # X-axis: show every 5th layer
    xticks = list(range(0, n_layers, 5)) + [n_layers - 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    # Annotate terminal cluster region
    ax.axvline(x=n_layers * 0.8, color='white', linewidth=1.5, linestyle='--', alpha=0.9)
    ax.text(n_layers * 0.81, -1.2, 'Terminal cluster', fontsize=7, color='#333333',
            ha='left', va='bottom', fontweight='bold', rotation=0)

    # Annotate LongBench wider band
    lb_idx = order.index('longbench_single_doc')
    ax.annotate('', xy=(24, lb_idx - 0.45), xytext=(32, lb_idx - 0.45),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.2))
    ax.text(28, lb_idx - 0.8, 'wider band', fontsize=6.5, color='white',
            ha='center', va='bottom', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r'LoRA $\|\Delta W\|_F$', fontsize=8)

    ax.set_title('Qwen3-8B: Per-layer LoRA delta norms', fontsize=10, pad=8)
    fig.savefig(f'{OUTDIR}/fig1_qwen8b_lora_delta_heatmap.pdf')
    fig.savefig(f'{OUTDIR}/fig1_qwen8b_lora_delta_heatmap.png')
    plt.close(fig)
    print('Figure 1 saved.')


# ============================================================
# Figure 2: Mamba-790m ablation bar chart (boundary-layer pattern)
# ============================================================
def figure2():
    data = load_json(f'{RESULTS}/ablation/mamba-790m/ablation_merged.json')
    matrix = np.array(data['matrix'])  # (48, 10)
    tasks = data['task_names']
    n_layers = matrix.shape[0]

    # Average importance across tasks
    mean_importance = matrix.mean(axis=1)
    # Also show per-category averages
    code_idx = [tasks.index('humaneval'), tasks.index('mbpp')]
    comm_idx = [tasks.index('arc_challenge'), tasks.index('piqa'), tasks.index('winogrande')]
    math_idx = [tasks.index('gsm8k')]

    fig, ax = plt.subplots(figsize=(6, 2.8))

    colors = []
    for i in range(n_layers):
        if i in [0, 1]:
            colors.append('#2171b5')  # early boundary
        elif i in [46, 47]:
            colors.append('#cb181d')  # late boundary
        elif i == 34:
            colors.append('#d94801')  # mid-critical
        else:
            colors.append('#bdbdbd')  # interior

    ax.bar(range(n_layers), mean_importance, color=colors, width=0.8, edgecolor='none')

    # Annotate boundary layers
    ax.annotate('L0, L1\n(input boundary)', xy=(0.5, mean_importance[0]),
                xytext=(6, mean_importance[0] + 0.15),
                fontsize=7, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2171b5', lw=1))
    ax.annotate('L46, L47\n(output boundary)', xy=(46.5, mean_importance[47]),
                xytext=(40, mean_importance[47] + 0.15),
                fontsize=7, ha='center',
                arrowprops=dict(arrowstyle='->', color='#cb181d', lw=1))
    ax.annotate('L34', xy=(34, mean_importance[34]),
                xytext=(34, mean_importance[34] + 0.12),
                fontsize=7, ha='center',
                arrowprops=dict(arrowstyle='->', color='#d94801', lw=0.8))

    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean ablation impact')
    ax.set_title('Mamba-790m: Per-layer ablation importance (boundary-layer pattern)', fontsize=10, pad=8)
    ax.set_xlim(-0.5, n_layers - 0.5)

    xticks = list(range(0, n_layers, 5)) + [n_layers - 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2171b5', label='Input boundary (L0\u2013L1)'),
        Patch(facecolor='#cb181d', label='Output boundary (L46\u2013L47)'),
        Patch(facecolor='#d94801', label='Mid-critical (L34)'),
        Patch(facecolor='#bdbdbd', label='Interior layers'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=7)

    fig.savefig(f'{OUTDIR}/fig2_mamba790m_ablation_boundary.pdf')
    fig.savefig(f'{OUTDIR}/fig2_mamba790m_ablation_boundary.png')
    plt.close(fig)
    print('Figure 2 saved.')


# ============================================================
# Figure 3: Causal validation — flipping tasks (LoRA delta vs ablation)
# ============================================================
def figure3():
    # Collect causal validation results across models and methods
    # LoRA delta-based causal validations
    # One file per seed/run in results/causal_validation/{slug}/
    def runs(slug, k):
        return sorted(glob.glob(f'{RESULTS}/causal_validation/{slug}/*_causal_val_k{k}_*.json'))

    # Layers selected by LoRA delta importance
    lora_delta_files = {
        'Qwen3-8B': runs('qwen3-8b', 6),
        'Llama-3.1-8B': runs('llama-3.1-8b', 6),
        'Llama-3.2-3B': runs('llama-3.2-3b', 5),
    }
    # Layers selected by ablation importance
    ablation_files = {
        'Mamba-790m': runs('mamba-790m', 5),
    }

    # Parse all results into a unified structure
    def parse_cv_files(file_dict, method_label):
        results = {}  # task -> list of (model, top_ppl, bot_ppl, supported)
        for model, files in file_dict.items():
            for fpath in files:
                if not os.path.exists(fpath):
                    continue
                d = load_json(fpath)
                for r in d['results']:
                    task = r['task']
                    if task not in results:
                        results[task] = []
                    top_ppl = r['top_k'].get('perplexity')
                    bot_ppl = r['bottom_k'].get('perplexity')
                    if top_ppl and bot_ppl:
                        # Compute recovery ratio: how much better top-k is
                        base = r['baseline'].get('perplexity', 1)
                        recovery = (bot_ppl - top_ppl) / max(bot_ppl, 0.01)
                        results[task].append({
                            'model': model,
                            'top_ppl': top_ppl,
                            'bot_ppl': bot_ppl,
                            'base_ppl': base,
                            'recovery': recovery,
                            'supported': r['hypothesis_supported'],
                            'method': method_label,
                        })
        return results

    ld_results = parse_cv_files(lora_delta_files, 'LoRA delta')
    ab_results = parse_cv_files(ablation_files, 'Ablation')

    # Focus on the three flipping tasks + two stable ones
    flipping_tasks = ['longbench_single_doc', 'piqa', 'squad']
    stable_tasks = ['humaneval', 'mbpp']
    focus_tasks = stable_tasks + flipping_tasks

    # Build paired bar data: for each task, compute mean PPL gap (bot - top) / bot
    # under LoRA delta and ablation
    fig, axes = plt.subplots(1, len(focus_tasks), figsize=(6.5, 2.8), sharey=False)

    for i, task in enumerate(focus_tasks):
        ax = axes[i]

        # LoRA delta results
        ld_entries = ld_results.get(task, [])
        ld_gaps = [(e['bot_ppl'] - e['top_ppl']) for e in ld_entries]

        # Ablation results
        ab_entries = ab_results.get(task, [])
        ab_gaps = [(e['bot_ppl'] - e['top_ppl']) for e in ab_entries]

        bars = []
        colors = []
        labels = []
        if ld_gaps:
            bars.append(np.mean(ld_gaps))
            colors.append('#2171b5')
            labels.append('LoRA\ndelta')
        else:
            bars.append(0)
            colors.append('#2171b5')
            labels.append('LoRA\ndelta')

        if ab_gaps:
            bars.append(np.mean(ab_gaps))
            colors.append('#cb181d')
            labels.append('Ablation')
        else:
            bars.append(0)
            colors.append('#cb181d')
            labels.append('Ablation')

        x = [0, 0.6]
        rects = ax.bar(x, bars, width=0.45, color=colors, edgecolor='white', linewidth=0.5)

        # Add support indicators
        for j, (gap, entries) in enumerate([(ld_gaps, ld_entries), (ab_gaps, ab_entries)]):
            if entries:
                n_supported = sum(1 for e in entries if e['supported'])
                n_total = len(entries)
                marker = f'{n_supported}/{n_total}'
                y_pos = max(gap) if gap else 0
                ax.text(x[j], max(bars[j] + 0.05, 0.05), marker,
                        ha='center', va='bottom', fontsize=6.5, fontweight='bold',
                        color='#238b45' if n_supported > 0 else '#cb181d')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=8, fontweight='bold')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if task in flipping_tasks:
            ax.set_facecolor('#fff5f5')

        if i == 0:
            ax.set_ylabel('PPL gap\n(bottom$_k$ $-$ top$_k$)', fontsize=7.5)

    # Add annotation for flipping tasks
    fig.text(0.72, -0.04, 'shaded background = flipping tasks', fontsize=7, fontstyle='italic',
             ha='center', color='#666666')

    fig.suptitle('Causal validation: PPL gap under selective fine-tuning', fontsize=10, y=1.05)
    fig.tight_layout(rect=[0, 0.02, 1, 1.0])
    fig.savefig(f'{OUTDIR}/fig3_causal_validation_flipping.pdf')
    fig.savefig(f'{OUTDIR}/fig3_causal_validation_flipping.png')
    plt.close(fig)
    print('Figure 3 saved.')


# ============================================================
# Figure 4: Qwen3-8B vs Qwen3-32B bimodality at scale
# ============================================================
def figure4():
    d8b = load_json(latest(f'{RESULTS}/lora_delta/qwen3-8b/lora_delta_*_r8_s200_*.json'))
    d32b = load_json(latest(f'{RESULTS}/lora_delta/qwen3-32b/lora_delta_*_r8_s200_*.json'))

    m8b = np.array(d8b['matrix'])    # (36, 10)
    m32b = np.array(d32b['matrix'])  # (64, 10)
    tasks = d8b['task_names']

    # Reorder tasks
    order = ['humaneval', 'mbpp', 'longbench_single_doc', 'squad', 'gsm8k', 'mmlu', 'gpqa', 'arc_challenge', 'piqa', 'winogrande']
    idx = [tasks.index(t) for t in order]
    m8b_ordered = m8b[:, idx]
    m32b_ordered = m32b[:, idx]
    task_labels = [TASK_DISPLAY[t] for t in order]

    # Normalize each to relative position [0, 1] for comparison
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.5), gridspec_kw={'hspace': 0.35})

    # Panel A: Qwen3-8B
    ax = axes[0]
    im1 = ax.imshow(m8b_ordered.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_ylabel('')
    ax.set_yticks(range(len(task_labels)))
    ax.set_yticklabels(task_labels)
    xticks8 = list(range(0, 36, 5)) + [35]
    ax.set_xticks(xticks8)
    ax.set_xticklabels([str(x) for x in xticks8])
    ax.set_xlabel('Layer index')
    ax.set_title('(a) Qwen3-8B (36 layers) — terminal cluster', fontsize=9, fontweight='bold', pad=6)
    ax.axvline(x=36 * 0.8, color='white', linewidth=1.2, linestyle='--', alpha=0.85)
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.85, pad=0.02)
    cbar1.set_label(r'$\|\Delta W\|_F$', fontsize=7)

    # Panel B: Qwen3-32B
    ax = axes[1]
    im2 = ax.imshow(m32b_ordered.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_ylabel('')
    ax.set_yticks(range(len(task_labels)))
    ax.set_yticklabels(task_labels)
    xticks32 = list(range(0, 64, 10)) + [63]
    ax.set_xticks(xticks32)
    ax.set_xticklabels([str(x) for x in xticks32])
    ax.set_xlabel('Layer index')
    ax.set_title('(b) Qwen3-32B (64 layers) — bimodal pattern', fontsize=9, fontweight='bold', pad=6)
    cbar2 = fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.02)
    cbar2.set_label(r'$\|\Delta W\|_F$', fontsize=7)

    # Annotate the two peaks in 32B
    # Early peak around L2-L6
    ax.axvline(x=4, color='white', linewidth=1, linestyle=':', alpha=0.8)
    ax.text(4, len(task_labels) + 0.3, 'early peak', fontsize=6.5, color='#333333',
            ha='center', va='top', fontweight='bold')
    # Terminal peak
    ax.axvline(x=64 * 0.8, color='white', linewidth=1.2, linestyle='--', alpha=0.85)
    ax.text(64 * 0.82, len(task_labels) + 0.3, 'terminal cluster', fontsize=6.5, color='#333333',
            ha='left', va='top', fontweight='bold')

    fig.savefig(f'{OUTDIR}/fig4_qwen_8b_vs_32b_bimodality.pdf')
    fig.savefig(f'{OUTDIR}/fig4_qwen_8b_vs_32b_bimodality.png')
    plt.close(fig)
    print('Figure 4 saved.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Generate the main-paper figures.")
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--output_dir', default='figures')
    a = ap.parse_args()
    RESULTS, OUTDIR = a.results_dir, a.output_dir
    os.makedirs(OUTDIR, exist_ok=True)
    figure1()
    figure2()
    figure3()
    figure4()
    print(f'\nAll figures saved to {OUTDIR}/')
