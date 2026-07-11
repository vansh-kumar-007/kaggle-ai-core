"""
checkpoint.py

Lightweight state persistence so the pipeline can resume from the
execute -> repair -> publish stage without redoing dataset selection,
planning, and section generation -- the expensive, already-paid-for work
-- if the process is interrupted (crash, Ctrl+C, or an outage that
outlasts even NemotronClient's extended DEGRADED-retry budget).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PipelineCheckpoint:
    """Everything needed to resume the execute/repair/publish stage."""

    notebook_path: str
    dataset_ref: str
    dataset_title: str
    notebook_title: str
    prior_variables: list[str]


def save_checkpoint(checkpoint: PipelineCheckpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(checkpoint), indent=2), encoding="utf-8")
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(path: Path) -> PipelineCheckpoint | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PipelineCheckpoint(**data)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to load checkpoint from %s: %s", path, exc)
        return None


def clear_checkpoint(path: Path) -> None:
    if path.exists():
        path.unlink()
        logger.info("Checkpoint cleared: %s", path)
