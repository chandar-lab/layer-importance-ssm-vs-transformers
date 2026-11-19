"""
Calibration data for the ten tasks used in the paper.

Every example is rendered as plain text (question + answer) and scored with the
standard causal-LM loss, which gives the task-specific signal used by all
importance methods.
"""

import os
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Dataset configs: (hf_path, hf_name, split, text_field, max_examples)
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "split": "test",
        "formatter": "mmlu",
    },
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "train",
        "formatter": "gsm8k",
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split": "test",
        "formatter": "arc",
    },
    "piqa": {
        "hf_path": "parquet",
        "hf_name": None,
        "split": "train",
        "formatter": "piqa",
        # Parquet export of ybisk/piqa (the original repo uses a loading script).
        "data_files": {"train": "hf://datasets/ybisk/piqa@refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet"},
    },
    "squad": {
        "hf_path": "rajpurkar/squad_v2",
        "hf_name": None,
        "split": "validation",
        "formatter": "squad",
    },
    "humaneval": {
        "hf_path": "evalplus/humanevalplus",
        "hf_name": None,
        "split": "test",
        "formatter": "humaneval",
    },
    "mbpp": {
        "hf_path": "evalplus/mbppplus",
        "hf_name": None,
        "split": "test",
        "formatter": "mbpp",
    },
    "longbench_single_doc": {
        "hf_path": "json",
        "hf_name": None,
        "split": "train",
        "formatter": "longbench",
        # LongBench Qasper subset; resolved at load time by _longbench_file().
        "data_files": {"train": "longbench:qasper.jsonl"},
    },
    "winogrande": {
        "hf_path": "allenai/winogrande",
        "hf_name": "winogrande_xl",
        "split": "train",
        "formatter": "winogrande",
    },
    "gpqa": {
        "hf_path": "Idavidrein/gpqa",
        "hf_name": "gpqa_main",
        "split": "train",
        "formatter": "gpqa",
    },
}

# Task groups (capability families)
TASK_GROUPS = {
    "language":    ["mmlu"],
    "math":        ["gsm8k", "gpqa"],
    "code":        ["humaneval", "mbpp"],
    "commonsense": ["arc_challenge", "piqa", "winogrande"],
    "retrieval":   ["squad", "longbench_single_doc"],
}

ALL_TASKS = list(TASK_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Formatters: convert raw HuggingFace examples to plain text strings
# ---------------------------------------------------------------------------

def _fmt_mmlu(ex: dict) -> str:
    choices = ex["choices"]
    options = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    answer = chr(65 + ex["answer"])
    return f"Question: {ex['question']}\n{options}\nAnswer: {answer}"


def _fmt_gsm8k(ex: dict) -> str:
    return f"Problem: {ex['question']}\nSolution: {ex['answer']}"


def _fmt_arc(ex: dict) -> str:
    labels = ex["choices"]["label"]
    texts  = ex["choices"]["text"]
    options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"Question: {ex['question']}\n{options}\nAnswer: {ex['answerKey']}"


def _fmt_piqa(ex: dict) -> str:
    label = "Solution 1" if ex["label"] == 0 else "Solution 2"
    return (
        f"Goal: {ex['goal']}\n"
        f"Solution 1: {ex['sol1']}\n"
        f"Solution 2: {ex['sol2']}\n"
        f"Answer: {label}"
    )


def _fmt_squad(ex: dict) -> str:
    answer = ex["answers"]["text"][0] if ex["answers"]["text"] else "unanswerable"
    return (
        f"Context: {ex['context']}\n"
        f"Question: {ex['question']}\n"
        f"Answer: {answer}"
    )


def _fmt_humaneval(ex: dict) -> str:
    return ex["prompt"] + ex["canonical_solution"]


def _fmt_mbpp(ex: dict) -> str:
    prompt = ex.get("text") or ex.get("prompt", "")
    return f"# {prompt}\n{ex['code']}"


def _fmt_longbench(ex: dict) -> str:
    context = ex.get("context", ex.get("input", ""))
    question = ex.get("input", "")
    answer = ex.get("answers", [""])[0] if isinstance(ex.get("answers"), list) else ex.get("answers", "")
    return f"Context: {context[:4096]}\nQuestion: {question}\nAnswer: {answer}"


def _fmt_winogrande(ex: dict) -> str:
    answer = ex["option1"] if ex["answer"] == "1" else ex["option2"]
    return f"Sentence: {ex['sentence']}\nAnswer: {answer}"


def _fmt_gpqa(ex: dict) -> str:
    choices = "\n".join([
        f"A. {ex['Correct Answer']}",
        f"B. {ex['Incorrect Answer 1']}",
        f"C. {ex['Incorrect Answer 2']}",
        f"D. {ex['Incorrect Answer 3']}",
    ])
    return f"Question: {ex['Question']}\n{choices}\nAnswer: A"


FORMATTERS = {
    "mmlu":        _fmt_mmlu,
    "gsm8k":       _fmt_gsm8k,
    "arc":         _fmt_arc,
    "piqa":        _fmt_piqa,
    "squad":       _fmt_squad,
    "humaneval":   _fmt_humaneval,
    "mbpp":        _fmt_mbpp,
    "longbench":   _fmt_longbench,
    "winogrande":  _fmt_winogrande,
    "gpqa":        _fmt_gpqa,
}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CalibrationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.encodings = self._tokenize(texts)

    # Short examples are packed together so every sequence has >= MIN_SEQ_LEN
    # tokens (some fused linear-attention kernels have no backward pass for
    # q_len <= 64). Kept for all models so data is identical across them.
    MIN_SEQ_LEN = 65

    def _tokenize(self, texts: List[str]) -> List[torch.Tensor]:
        # First pass: tokenize all texts
        raw = []
        for text in texts:
            ids = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.seq_len,
                padding=False,
            ).input_ids.squeeze(0)
            if len(ids) > 1:
                raw.append(ids)

        # Second pass: pack consecutive short examples so every sequence
        # has at least MIN_SEQ_LEN tokens.
        encoded = []
        buf = []
        buf_len = 0
        for ids in raw:
            buf.append(ids)
            buf_len += len(ids)
            if buf_len >= self.MIN_SEQ_LEN:
                packed = torch.cat(buf)[:self.seq_len]
                encoded.append(packed)
                buf = []
                buf_len = 0
        # Flush remaining buffer if long enough
        if buf_len >= self.MIN_SEQ_LEN:
            packed = torch.cat(buf)[:self.seq_len]
            encoded.append(packed)

        return encoded

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.encodings[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Left-pad sequences in a batch to the same length."""
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, -seq_len:] = item["input_ids"]
        labels[i, -seq_len:]    = item["labels"]
        attn_mask[i, -seq_len:] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


# ---------------------------------------------------------------------------
# Registry and public builder
# ---------------------------------------------------------------------------

def _longbench_file(name: str) -> str:
    """
    Path to a LongBench jsonl file (e.g. ``qasper.jsonl``).

    Uses ``$LONGBENCH_DATA_DIR/{name}`` if set; otherwise downloads
    THUDM/LongBench ``data.zip`` once and extracts it next to the HF cache.
    """
    local_dir = os.environ.get("LONGBENCH_DATA_DIR")
    if local_dir:
        return os.path.join(local_dir, name)

    import zipfile

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    out_dir = os.path.join(os.path.dirname(zip_path), "longbench_extracted")
    target = os.path.join(out_dir, "data", name)
    if not os.path.exists(target):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(f"data/{name}", out_dir)
    return target


def load_task_dataset(task_name: str, cache_dir: Optional[str] = None):
    """Load the raw HuggingFace split configured for ``task_name``."""
    from datasets import load_dataset  # lazy import

    cfg = TASK_CONFIGS[task_name]
    load_kwargs = dict(path=cfg["hf_path"], split=cfg["split"])
    if cfg["hf_name"] is not None:
        load_kwargs["name"] = cfg["hf_name"]
    if "data_files" in cfg:
        load_kwargs["data_files"] = {
            k: _longbench_file(v.split(":", 1)[1]) if v.startswith("longbench:") else v
            for k, v in cfg["data_files"].items()
        }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    try:
        return load_dataset(**load_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset for task '{task_name}': {e}\nConfig: {cfg}")


class TaskDataRegistry:
    """Loads and caches HuggingFace datasets for calibration."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def load_texts(self, task_name: str, n_examples: int, seed: int = 42) -> List[str]:
        formatter = FORMATTERS[TASK_CONFIGS[task_name]["formatter"]]
        dataset = load_task_dataset(task_name, self.cache_dir)

        indices = list(range(len(dataset)))
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[:n_examples]

        texts = []
        for idx in indices:
            try:
                texts.append(formatter(dataset[idx]))
            except Exception:
                continue  # skip malformed examples

        return texts


def build_task_dataloaders(
    tasks: List[str],
    tokenizer: PreTrainedTokenizerBase,
    n_examples: int = 512,
    seq_len: int = 512,
    batch_size: int = 4,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Returns a dict mapping task_name → DataLoader with tokenized calibration data.

    Args:
        tasks:      List of task names from TASK_CONFIGS.
        tokenizer:  HuggingFace tokenizer.
        n_examples: Number of examples per task.
        seq_len:    Max token length per example.
        batch_size: Batch size for the dataloader.
        seed:       Random seed for example selection.
        cache_dir:  Optional HF datasets cache directory.
        num_workers: DataLoader num_workers.
    """
    registry = TaskDataRegistry(cache_dir=cache_dir)
    dataloaders = {}

    for task in tasks:
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Available: {list(TASK_CONFIGS.keys())}")

        texts = registry.load_texts(task, n_examples=n_examples, seed=seed)
        dataset = CalibrationDataset(texts, tokenizer, seq_len=seq_len)

        dataloaders[task] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
