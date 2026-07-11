"""
dataset_selector.py

Intelligent dataset discovery and selection.

Searches Kaggle across a rotating set of topics, applies strict hard
filters to eliminate low-quality/tutorial/broken datasets, scores the
remaining candidates, and returns the single best dataset to use for
notebook generation. Also skips datasets that have already been
published, to avoid repeats.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

from kaggle_ai_core.kaggle_client import DatasetSearchResult, KaggleClient
from kaggle_ai_core.utils.file_utils import load_published_refs

logger = logging.getLogger(__name__)

# --- Search term strategy -----------------------------------------------

CORE_TOPICS: list[str] = [
    "finance",
    "healthcare",
    "sports analytics",
    "climate",
    "real estate",
    "retail sales",
    "customer churn",
    "energy consumption",
    "education",
    "transportation",
]

RANDOM_TOPIC_POOL: list[str] = [
    "cybersecurity",
    "social media",
    "e-commerce",
    "insurance",
    "manufacturing",
    "agriculture",
    "tourism",
    "employment",
    "supply chain",
    "public health",
    "banking",
    "telecommunications",
]

RANDOM_TOPIC_CHANCE: float = 0.3  # 30% chance to add a wildcard topic each run

# --- Hard filter thresholds ----------------------------------------------

MIN_DATASET_SIZE_BYTES: int = 50_000       # reject tiny/toy datasets
MAX_DATASET_SIZE_BYTES: int = 1_000_000_000  # reject datasets too large for CI (GitHub Actions ~14GB disk)
MIN_USABILITY_RATING: float = 0.5          # reject low-quality metadata
MIN_TAG_COUNT: int = 1                     # reject completely untagged datasets

BLACKLISTED_TITLE_KEYWORDS: list[str] = [
    "test", "sample", "dummy", "practice", "my first", "tutorial",
    "meme", "joke", "template", "playground", "toy dataset", "demo",
]


@dataclass
class ScoredDataset:
    """A dataset candidate paired with its computed quality score."""

    dataset: DatasetSearchResult
    score: float


def _get_search_terms() -> list[str]:
    """
    Build today's list of search terms: the full fixed core topic list,
    plus (with some probability) one random wildcard topic for variety.
    """
    terms = list(CORE_TOPICS)
    if random.random() < RANDOM_TOPIC_CHANCE:
        wildcard = random.choice(RANDOM_TOPIC_POOL)
        terms.append(wildcard)
        logger.info("Added wildcard search topic for today: '%s'", wildcard)
    return terms


def _hard_filter_reason(ds: DatasetSearchResult, published_refs: set[str]) -> str | None:
    """
    Check a dataset against strict hard-filter rules.

    Returns:
        A human-readable rejection reason if the dataset should be
        disqualified, or None if it passes all hard filters.
    """
    if ds.ref in published_refs:
        return "already published previously"

    if ds.size_bytes and ds.size_bytes < MIN_DATASET_SIZE_BYTES:
        return f"too small ({ds.size_bytes} bytes < {MIN_DATASET_SIZE_BYTES})"

    if ds.size_bytes and ds.size_bytes > MAX_DATASET_SIZE_BYTES:
        return f"too large ({ds.size_bytes} bytes > {MAX_DATASET_SIZE_BYTES})"

    if ds.usability_rating < MIN_USABILITY_RATING:
        return f"usability too low ({ds.usability_rating} < {MIN_USABILITY_RATING})"

    if len(ds.tags) < MIN_TAG_COUNT:
        return "no tags present"

    title_lower = ds.title.lower()
    for keyword in BLACKLISTED_TITLE_KEYWORDS:
        if keyword in title_lower:
            return f"blacklisted title keyword: '{keyword}'"

    return None


def _score_dataset(ds: DatasetSearchResult) -> float:
    """
    Compute a quality score for a dataset that has already passed hard
    filters. Higher is better.

    Weighting rationale:
        - usability_rating is the strongest single quality signal (weight: 100,
          since it's already a 0.0-1.0 Kaggle-computed score).
        - vote_count is log-scaled so a handful of mega-popular "classic"
          datasets don't permanently dominate every selection.
        - kernel_count (public notebooks built on this dataset) is a distinct
          engagement signal from votes -- it reflects how many people have
          actually done real analysis on this data, which correlates with
          it being genuinely analyzable. Log-scaled for the same reason as votes.
        - size is log-scaled -- raw byte counts span many orders of magnitude
          and a 10x larger dataset isn't necessarily 10x better.
        - recency gives a modest bonus to datasets updated more recently,
          since actively maintained datasets are less likely to have stale
          or broken download links. Decays smoothly rather than a hard cutoff.
        - tag_count gives a small bonus for well-categorized datasets.
    """
    usability_score = ds.usability_rating * 100.0
    vote_score = math.log1p(max(ds.vote_count, 0)) * 15.0
    kernel_score = math.log1p(max(ds.kernel_count, 0)) * 10.0
    size_score = math.log1p(max(ds.size_bytes, 0)) * 5.0
    tag_score = len(ds.tags) * 3.0
    recency_score = _recency_score(ds.last_updated)

    return (
        usability_score
        + vote_score
        + kernel_score
        + size_score
        + tag_score
        + recency_score
    )


def _recency_score(last_updated: str) -> float:
    """
    Compute a smooth recency bonus based on how long ago a dataset was
    last updated. Newer datasets score higher, decaying gradually --
    there is no hard cutoff, since an older but excellent dataset should
    still be able to win on its other merits.

    Returns:
        A bonus in the range [0.0, 20.0], or 0.0 if the date is missing
        or unparseable.
    """
    if not last_updated:
        return 0.0

    try:
        updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - updated_dt).days
    except (ValueError, TypeError):
        return 0.0

    if age_days < 0:
        return 20.0
    # Half-life style decay: ~20 points for brand new, ~10 at 2 years, approaching 0 after ~6 years.
    return max(0.0, 20.0 * math.exp(-age_days / 730.0))


class DatasetSelector:
    """
    Selects the single best Kaggle dataset to generate a notebook for,
    using multi-topic search, strict quality filtering, and scoring.
    """

    def __init__(self, kaggle_client: KaggleClient, published_refs_path: Path) -> None:
        self._client = kaggle_client
        self._published_refs_path = published_refs_path

    def get_shortlist(self, top_n: int = 5, max_results_per_term: int = 20) -> list[ScoredDataset]:
        """
        Search, filter, and score datasets, returning the top N candidates
        ordered best-first. This is Stage A of the two-stage selection
        process -- Stage B (DatasetAnalyzer) downloads and profiles these
        candidates' actual data to pick the final winner.

        Args:
            top_n: Number of top-scoring candidates to return.
            max_results_per_term: Max results to fetch per search term.

        Returns:
            Up to `top_n` ScoredDataset objects, best-first. Empty list
            if no dataset passed the hard filters.
        """
        candidates = self._gather_scored_candidates(max_results_per_term)
        return candidates[:top_n]

    def select_best_dataset(self, max_results_per_term: int = 20) -> ScoredDataset | None:
        """
        Search, filter, score, and select the single best dataset by
        metadata score alone (Stage A only, no data profiling).

        Kept for standalone use/testing; the full pipeline uses
        get_shortlist() + DatasetAnalyzer instead.

        Returns:
            The highest-scoring ScoredDataset, or None if no dataset
            passed the hard filters.
        """
        candidates = self._gather_scored_candidates(max_results_per_term)
        if not candidates:
            logger.error("No datasets passed hard filters. Cannot select a dataset today.")
            return None

        best = candidates[0]
        logger.info(
            "Selected dataset: '%s' (%s) with score %.2f",
            best.dataset.title, best.dataset.ref, best.score,
        )
        return best

    def _gather_scored_candidates(self, max_results_per_term: int) -> list[ScoredDataset]:
        """Run the full search -> filter -> score pipeline, sorted best-first."""
        published_refs = load_published_refs(self._published_refs_path)
        search_terms = _get_search_terms()

        seen_refs: set[str] = set()
        candidates: list[ScoredDataset] = []
        rejected_count = 0

        for term in search_terms:
            try:
                results = self._client.search_datasets(term, max_results=max_results_per_term)
            except Exception:
                logger.warning("Search failed for term '%s', skipping.", term)
                continue

            for ds in results:
                if ds.ref in seen_refs:
                    continue
                seen_refs.add(ds.ref)

                reason = _hard_filter_reason(ds, published_refs)
                if reason is not None:
                    logger.info("Rejected '%s': %s", ds.ref, reason)
                    rejected_count += 1
                    continue

                score = _score_dataset(ds)
                candidates.append(ScoredDataset(dataset=ds, score=score))

        logger.info(
            "Evaluated %d unique dataset(s) across %d search term(s): %d passed, %d rejected.",
            len(seen_refs), len(search_terms), len(candidates), rejected_count,
        )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
