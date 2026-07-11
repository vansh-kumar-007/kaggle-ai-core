"""
helpers.py

Small shared utility functions used across the pipeline.
"""

from __future__ import annotations

import re


def extract_json_from_text(text: str) -> str:
    """
    Extract a JSON object/array from raw LLM output text.

    Handles the common case where the model wraps its JSON in markdown
    code fences (```json ... ```) despite being instructed not to, and
    falls back to locating the first '{' and last '}' if no fences are
    present.

    Args:
        text: Raw text returned by the LLM.

    Returns:
        The extracted JSON string (not yet parsed/validated).

    Raises:
        ValueError: if no JSON-like content could be located at all.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    raise ValueError("No JSON object found in text.")

def slugify(text: str, max_length: int = 60) -> str:
    """
    Convert arbitrary text into a Kaggle-kernel-slug-safe string:
    lowercase, alphanumeric and single hyphens only, no leading/trailing
    hyphens, at least 5 characters (Kaggle's minimum).
    """
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    slug = slug[:max_length].rstrip("-")
    if len(slug) < 5:
        slug = (slug + "-notebook")[:max_length].rstrip("-")
    return slug
