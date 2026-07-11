"""
file_utils.py

Simple JSON-backed persistence helpers used across the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_published_refs(path: Path) -> set[str]:
    """
    Load the set of previously-published dataset references.

    Args:
        path: Path to the JSON file storing published refs.

    Returns:
        A set of dataset ref strings (e.g. "owner/dataset-slug").
        Returns an empty set if the file does not exist or is invalid.
    """
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("published_refs", []))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load published refs from %s: %s", path, exc)
        return set()


def add_published_ref(path: Path, ref: str) -> None:
    """
    Add a dataset ref to the published-refs tracking file.

    Creates the file (and any parent directories) if it does not exist yet.

    Args:
        path: Path to the JSON file storing published refs.
        ref: Dataset reference to record as published.
    """
    refs = load_published_refs(path)
    refs.add(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"published_refs": sorted(refs)}, f, indent=2)
    logger.info("Recorded '%s' as published in %s", ref, path)
