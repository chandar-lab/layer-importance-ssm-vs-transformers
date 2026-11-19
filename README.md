# Layer importance in SSMs vs. Transformers

Code for measuring **which layers matter** for a capability in transformers, state-space
models (SSMs), RNNs and hybrids, using two complementary notions of layer importance:

| Notion | Question | Measurement |
|---|---|---|
| **Necessity** | What breaks if the layer is removed? | Zero-shot layer ablation: `L_t(bypass layer l) − L_t(baseline)` |
| **Plasticity** | Where does the model change when it learns the task? | `‖ΔW_l‖_F` after LoRA (or full) fine-tuning on task `t` |

Each method produces an importance matrix `I[n_layers, n_tasks]` over ten tasks. The code
also includes the follow-up experiments: causal validation (LoRA on top-k vs. bottom-k
layers), continual-learning forgetting, linear probes, head-level ablation, Fisher
curvature and update-to-weight normalisation.

## Installation

```bash
git clone https://github.com/chandar-lab/layer-importance-ssm-vs-transformers
cd layer-importance-ssm-vs-transformers
pip install -e .            # or: pip install -r requirements.txt
pip install bitsandbytes    # only for full fine-tuning of >=7B models
```

Tested with Python 3.10, PyTorch 2.7, `transformers==4.53.0` and `peft 0.20`. The
`transformers` version is pinned because the Mamba / Mamba-2 / Falcon-Mamba / RWKV-6 /
Zamba2 implementations change between releases. Llama 3 and GPQA are gated on the
HuggingFace Hub, so run `huggingface-cli login` first.

## Models

All models load through `layerimp.models.load_model_and_tokenizer` (slug → HF id in
`layerimp.models.MODELS`):

| Family | Models |
|---|---|
| Transformer | Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B, Llama-3.2-1B / 3B, Llama-3.1-8B / 70B |
| SSM | Mamba-130m / 370m / 790m / 2.8b (`state-spaces/*`), Mamba2-1.3B, Falcon-Mamba-7B |
| RNN | RWKV-6-3B (`RWKV/rwkv-6-world-3b`) |
| Hybrid | Zamba2-2.7B |

The original `state-spaces/mamba-*` checkpoints are not in HF format; the loader builds
the config explicitly, remaps the weight names and uses the `EleutherAI/gpt-neox-20b`
tokenizer.

## Tasks

`mmlu, gsm8k, arc_challenge, piqa, humaneval, mbpp, squad, winogrande, gpqa,
longbench_single_doc` (see `layerimp/data.py`). Each example is rendered as plain text
(question + answer) and scored with the causal-LM loss. PIQA is read from the Hub's
parquet export. LongBench (Qasper) is downloaded from `THUDM/LongBench` on first use, or
read from `$LONGBENCH_DATA_DIR/qasper.jsonl` if that variable is set.

## Repository layout

```
layerimp/                 library
  importance.py           ImportanceResult, ablation + gradient analyzers
  lora_delta.py           LoRA delta (plasticity) analyzer
  full_ft.py              full fine-tuning gradient / delta analyzer
  finetune.py             LoRA-on-layer-subset training + perplexity / MCQ evaluation
  data.py                 task datasets and calibration dataloaders
  models.py               model registry, loading, architecture helpers
experiments/              one script per experiment (GPU)
analysis/                 statistics and figures from saved results (CPU)
```

Every experiment writes JSON files to `results/{method}/{model_slug}/`. The analysis
scripts read that layout (override it with `--results_dir`).

## Reproducing the experiments

The commands below use the paper's hyper-parameters. Swap `qwen3-8b` / `Qwen/Qwen3-8B`
for any model in the table above. All scripts take `--help`, and the ones that train or
evaluate models also take `--wandb_project` for optional logging.

### 1. Necessity: layer ablation

One job per layer (e.g. a SLURM array over `--ablation_layer_idx`), then merge:

```bash
for L in $(seq 0 35); do
  python experiments/run_ablation.py --model_name_or_path Qwen/Qwen3-8B --method ablation \
      --ablation_layer_idx $L --n_examples 256 --seq_len 256 --batch_size 8 \
      --output_dir results/ablation/qwen3-8b
done
python experiments/run_ablation.py --mode merge \
    --merge_dir results/ablation/qwen3-8b/ablation_layers --output_dir results/ablation/qwen3-8b
# -> results/ablation/qwen3-8b/ablation_merged.json
```

Leave out `--ablation_layer_idx` to ablate every layer in a single job. For a transformer
layer, the ablation skips the attention sub-layer and keeps the MLP. For Mamba, RWKV and
Zamba2, the whole block is replaced by the identity. `--method gradient` computes the
gradient × activation (Taylor) score instead.

### 2. Plasticity: LoRA delta

```bash
python experiments/run_lora_delta.py --model_name_or_path Qwen/Qwen3-8B \
    --lora_rank 8 --lora_alpha 16 --n_steps 200 --lr 1e-4 \
    --n_examples 512 --seq_len 512 --batch_size 4 --seed 42 \
    --output_dir results/lora_delta/qwen3-8b
```

The LoRA targets depend on the architecture (`layerimp.models.default_target_modules`):
MLP projections for transformers, `in_proj/x_proj/dt_proj` for Mamba models, and
`receptance/key/value/output/gate` for RWKV. The adapters are re-initialised before each
task. Use `--shuffle_labels` for the random-label control.

### 3. Plasticity without adapters: full fine-tuning

```bash
# <= 4B models
python experiments/run_full_ft.py --model_name_or_path Qwen/Qwen3-0.6B \
    --n_steps 500 --lr 1e-5 --optimizer adamw --dtype float32 \
    --output_dir results/full_ft/qwen3-0.6b
# >= 7B models
python experiments/run_full_ft.py --model_name_or_path Qwen/Qwen3-8B \
    --n_steps 500 --lr 1e-5 --optimizer paged_adamw8bit --dtype float32 --gradient_checkpointing \
    --output_dir results/full_ft/qwen3-8b
# pretrained weight norms, for update-to-weight normalisation
python experiments/compute_weight_norms.py --model_name_or_path Qwen/Qwen3-8B --slug qwen3-8b
```

This writes `full_ft_delta_*.json` (per-layer `‖ΔW‖_F`) and `full_ft_grad_*.json`
(per-layer gradient norms). The embeddings and `lm_head` are frozen, so every update is
attributed to some layer. Use float32 weights: with bf16, many small updates round to zero.

### 4. Causal validation

LoRA restricted to the top-k vs. bottom-k layers of an importance map (same budget):

```bash
python experiments/run_causal_validation.py --model_name_or_path Qwen/Qwen3-8B \
    --importance_json "results/lora_delta/qwen3-8b/lora_delta_*.json" \
    --tasks humaneval mbpp mmlu arc_challenge gpqa piqa winogrande gsm8k squad longbench_single_doc \
    --k 6 --n_train_steps 300 --lr 1e-4 --output_dir results/causal_validation/qwen3-8b
# all-layer LoRA reference
python experiments/run_causal_validation.py --uniform --model_name_or_path Qwen/Qwen3-8B \
    --tasks humaneval winogrande --output_dir results/causal_validation/qwen3-8b
```

### 5. Continual-learning forgetting

```bash
python experiments/run_cl_forgetting.py --model_name_or_path Qwen/Qwen3-8B \
    --importance_json "results/lora_delta/qwen3-8b/lora_delta_*.json" \
    --task_pairs humaneval winogrande gsm8k piqa mbpp winogrande \
    --k 6 --n_train_steps 30 --lr 1e-4 --batch_size 2 --gradient_checkpointing \
    --output_dir results/cl_forgetting/qwen3-8b
```

Each pair trains LoRA on task A and then on task B, and reports the change in task-A
perplexity for top-k, bottom-k, random-k and all-layer EWC placements. Any importance
JSON works as `--importance_json`, including the Fisher map from step 6.

### 6. Mechanistic analyses

```bash
python experiments/run_probing.py --model_name_or_path Qwen/Qwen3-8B \
    --n_examples 256 --seq_len 512 --batch_size 8 --output_dir results/probing/qwen3-8b
python experiments/run_head_ablation.py --model_name_or_path Qwen/Qwen3-8B --layer_idx 5 \
    --output_dir results/head_ablation/qwen3-8b          # one job per layer, then --mode merge
python experiments/run_fisher.py --model_name_or_path Qwen/Qwen3-8B --model_slug qwen3-8b
python analysis/fisher_to_importance.py --slug qwen3-8b --model_name_or_path Qwen/Qwen3-8B
```

## Analysis

```bash
# Plasticity–necessity alignment: per-task Spearman(LoRA delta, ablation), averaged over tasks
python analysis/compare_methods.py --all --a_method lora_delta --b_method ablation
# Does full fine-tuning give the same plasticity map as LoRA?
python analysis/compare_methods.py --all --a_method full_ft_delta --b_method lora_delta
python analysis/update_to_weight.py          # same correlations with ||ΔW|| / ||W||
python analysis/cv_analysis.py               # spread of ablation importance across layers
python analysis/probe_correlation.py --all   # probe accuracy vs. importance
python analysis/make_figures.py --output_dir figures
```

## Output format

Every importance map is saved as an `ImportanceResult` JSON (`layerimp/importance.py`):

```json
{
  "matrix":      [[...]],   // [n_layers][n_tasks] raw scores
  "matrix_norm": [[...]],   // each task column normalised to sum to 1
  "layer_types": [...], "task_names": [...], "method": "lora_delta",
  "model_name": "Qwen/Qwen3-8B", "n_samples": 512, "seq_len": 512,
  "...": "method-specific metadata"
}
```

Load one with `layerimp.utils.load_result(path_or_glob)`.

## Citation

```bibtex
TBD
```
