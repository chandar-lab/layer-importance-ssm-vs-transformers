"""Small I/O, logging and W&B helpers shared by the experiment scripts."""

import glob
import json
import logging
import os
from typing import List, Optional

import numpy as np

from layerimp.importance import ImportanceResult

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def save_result(result: ImportanceResult, output_dir: str, filename: str) -> str:
    """Write ``{filename}.json`` (full result) and ``{filename}.npy`` (raw matrix)."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, filename + ".json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    np.save(os.path.join(output_dir, filename + ".npy"), result.matrix)
    logger.info(f"Saved importance result → {json_path}")
    return json_path


def resolve_path(path_or_glob: str) -> str:
    """Return ``path_or_glob`` itself, or the last (newest-named) match of a glob."""
    if os.path.exists(path_or_glob):
        return path_or_glob
    matches = sorted(glob.glob(path_or_glob))
    if not matches:
        raise FileNotFoundError(f"No file matches: {path_or_glob}")
    return matches[-1]


def load_result(path_or_glob: str) -> ImportanceResult:
    with open(resolve_path(path_or_glob)) as f:
        return ImportanceResult.from_dict(json.load(f))


def init_wandb(
    project: Optional[str],
    entity: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[dict] = None,
):
    """Start a W&B run, or return None if ``project`` is None / wandb is missing."""
    if project is None:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging.")
        return None
    return wandb.init(project=project, entity=entity, name=name, tags=tags or [], config=config or {})
