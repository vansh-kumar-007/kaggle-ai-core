"""
dataset_analyzer.py

Stage B of dataset selection: downloads and profiles the top candidates
from Stage A's metadata-based scoring, computes a real data-quality
score from the actual CSV contents, and picks the final winner using a
combined score. Produces a structured DatasetProfile consumed directly
by the notebook planning stage (Step 5).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from kaggle_ai_core.dataset_selector import ScoredDataset
from kaggle_ai_core.kaggle_client import KaggleClient

logger = logging.getLogger(__name__)

# --- Column role heuristics -----------------------------------------------

ID_NAME_PATTERN = re.compile(r"(^|_)(id|uuid|guid|index)($|_)", re.IGNORECASE)
TIME_NAME_PATTERN = re.compile(r"(^|_)(date|time|timestamp|year|month|day)($|_)", re.IGNORECASE)
GEO_NAME_PATTERN = re.compile(
    r"(country|state|city|region|province|latitude|longitude|\blat\b|\blon\b|\blng\b|zip|postal)",
    re.IGNORECASE,
)
TARGET_NAME_CANDIDATES = {
    "target", "label", "class", "y", "outcome", "result", "churn",
    "survived", "price", "sales", "default", "fraud", "response",
}

TEXT_AVG_LENGTH_THRESHOLD = 50  # avg string length above this suggests free text, not a category


@dataclass
class ColumnProfile:
    """Profile of a single column within a CSV file."""

    name: str
    dtype: str
    missing_pct: float
    unique_count: int
    role: str  # one of: id, time, geographic, categorical, numerical, text, unknown
    is_target_candidate: bool


@dataclass
class CsvProfile:
    """Profile of a single CSV file within a dataset."""

    filename: str
    num_rows: int
    num_cols: int
    missing_pct_overall: float
    duplicate_pct: float
    columns: list[ColumnProfile] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class DatasetProfile:
    """
    Full profile of a dataset: its Stage A metadata score plus Stage B
    real-data quality analysis, combined into a final score.
    """

    ref: str
    title: str
    metadata_score: float
    quality_score: float
    combined_score: float
    csv_profiles: list[CsvProfile] = field(default_factory=list)

CURRENCY_PATTERN = re.compile(r"^\$?-?[\d,]+\.?\d*$")
CURRENCY_SAMPLE_SIZE = 200
CURRENCY_MATCH_THRESHOLD = 0.9  # 90%+ of sampled non-null values must look like currency/numeric strings


def _looks_like_currency(series: pd.Series) -> bool:
    """
    Check whether an object-dtype column actually holds currency-formatted
    numeric strings (e.g. "$1,234.56") rather than genuine categorical or
    free-text values. Samples up to CURRENCY_SAMPLE_SIZE non-null values
    rather than scanning the whole column, since this only needs to be a
    reliable heuristic, not an exhaustive check.
    """
    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.astype(str).sample(
        n=min(CURRENCY_SAMPLE_SIZE, len(non_null)), random_state=42
    )
    match_count = sample.str.match(CURRENCY_PATTERN).sum()
    return (match_count / len(sample)) >= CURRENCY_MATCH_THRESHOLD


def _classify_column(series: pd.Series, name: str, num_rows: int) -> ColumnProfile:
    """
    Classify a single column's role using name heuristics and dtype/
    cardinality signals. Also flags whether it looks like a plausible
    prediction target.
    """
    missing_pct = float(series.isna().mean() * 100.0)
    unique_count = int(series.nunique(dropna=True))
    dtype_str = str(series.dtype)

    is_id = bool(ID_NAME_PATTERN.search(name)) and (unique_count >= int(num_rows * 0.95))
    is_time_name = bool(TIME_NAME_PATTERN.search(name))
    is_geo = bool(GEO_NAME_PATTERN.search(name))

    if is_id:
        role = "id"
    elif is_time_name:
        role = "time"
    elif is_geo:
        role = "geographic"
    elif pd.api.types.is_numeric_dtype(series):
        role = "numerical"
    elif pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        if _looks_like_currency(series):
            role = "numerical"
        else:
            avg_len = series.dropna().astype(str).str.len().mean() if series.notna().any() else 0.0
            role = "text" if (avg_len or 0.0) > TEXT_AVG_LENGTH_THRESHOLD else "categorical"
    else:
        role = "unknown"

    name_lower = name.lower().strip()
    is_target_candidate = name_lower in TARGET_NAME_CANDIDATES and role != "id"

    return ColumnProfile(
        name=name,
        dtype=dtype_str,
        missing_pct=round(missing_pct, 2),
        unique_count=unique_count,
        role=role,
        is_target_candidate=is_target_candidate,
    )


def _profile_csv(path: Path) -> CsvProfile | None:
    """
    Load a single CSV file and compute its full profile: shape, missing
    values, duplicates, and per-column classification.

    Returns:
        A CsvProfile, or None if the file could not be read (e.g. corrupt,
        not actually CSV despite the extension, encoding issues) or is empty.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.warning("Failed to read CSV '%s': %s", path, exc)
        return None

    if df.empty or df.shape[1] == 0:
        logger.warning("CSV '%s' is empty, skipping.", path)
        return None

    num_rows, num_cols = df.shape
    missing_pct_overall = float(df.isna().mean().mean() * 100.0)
    duplicate_pct = float(df.duplicated().mean() * 100.0)

    columns = [_classify_column(df[col], col, num_rows) for col in df.columns]

    quality_score = _score_csv_quality(
        num_rows=num_rows,
        missing_pct_overall=missing_pct_overall,
        duplicate_pct=duplicate_pct,
        columns=columns,
    )

    return CsvProfile(
        filename=path.name,
        num_rows=num_rows,
        num_cols=num_cols,
        missing_pct_overall=round(missing_pct_overall, 2),
        duplicate_pct=round(duplicate_pct, 2),
        columns=columns,
        quality_score=quality_score,
    )


def _score_csv_quality(
    num_rows: int,
    missing_pct_overall: float,
    duplicate_pct: float,
    columns: list[ColumnProfile],
) -> float:
    """
    Compute a real-data quality score for a single CSV.

    Weighting rationale:
        - completeness_score rewards low missing-value rates (max 100).
        - duplicate_penalty subtracts for high duplicate rates.
        - row_score log-scales row count (more rows -> more statistical
          power -> generally more interesting analysis).
        - diversity_score rewards a mix of column roles (numerical,
          categorical, time, text, geographic), since that opens up more
          possible notebook sections in later steps.
    """
    completeness_score = max(0.0, 100.0 - missing_pct_overall)
    duplicate_penalty = duplicate_pct * 1.5
    row_score = math.log1p(max(num_rows, 0)) * 8.0

    distinct_roles = {c.role for c in columns if c.role != "unknown"}
    diversity_score = len(distinct_roles) * 10.0

    return completeness_score - duplicate_penalty + row_score + diversity_score


def _sanitize_ref_for_path(ref: str) -> str:
    """Convert a dataset ref like 'owner/slug' into a filesystem-safe folder name."""
    return ref.replace("/", "__")


class DatasetAnalyzer:
    """
    Stage B analyzer: downloads and profiles a shortlist of candidate
    datasets, then selects the final winner using a combined score.
    """

    def __init__(self, kaggle_client: KaggleClient, datasets_dir: Path) -> None:
        self._client = kaggle_client
        self._datasets_dir = datasets_dir

    def analyze_shortlist(self, shortlist: list[ScoredDataset]) -> DatasetProfile | None:
        """
        Download and profile each candidate in the shortlist, in order,
        and return the DatasetProfile with the highest combined score.

        Candidates that fail to download, contain no readable CSVs, or
        error out during profiling are logged and skipped -- the pipeline
        continues with the remaining candidates rather than aborting.

        Args:
            shortlist: Top-N candidates from Stage A metadata scoring,
                ordered best-first.

        Returns:
            The winning DatasetProfile, or None if every candidate failed.
        """
        profiles: list[DatasetProfile] = []

        for candidate in shortlist:
            ref = candidate.dataset.ref
            try:
                profile = self._analyze_one(candidate)
            except Exception:
                logger.exception("Unexpected error analyzing '%s', skipping.", ref)
                continue

            if profile is None:
                logger.warning("Skipping '%s': no usable data found.", ref)
                continue

            profiles.append(profile)

        if not profiles:
            logger.error("No candidate in the shortlist could be successfully profiled.")
            return None

        profiles.sort(key=lambda p: p.combined_score, reverse=True)
        winner = profiles[0]
        logger.info(
            "Final selection: '%s' (%s) | metadata_score=%.2f quality_score=%.2f combined_score=%.2f",
            winner.title, winner.ref, winner.metadata_score, winner.quality_score, winner.combined_score,
        )
        return winner

    def _analyze_one(self, candidate: ScoredDataset) -> DatasetProfile | None:
        """Download and profile a single candidate dataset."""
        ref = candidate.dataset.ref
        dest_dir = self._datasets_dir / _sanitize_ref_for_path(ref)

        logger.info("Analyzing candidate '%s' (metadata_score=%.2f)", ref, candidate.score)

        try:
            self._client.download_dataset_files(ref, dest_dir)
        except Exception:
            logger.exception("Download failed for '%s', skipping.", ref)
            return None

        csv_paths = sorted(dest_dir.rglob("*.csv"))
        if not csv_paths:
            logger.warning("No CSV files found for '%s' (non-tabular dataset), skipping.", ref)
            return None

        csv_profiles: list[CsvProfile] = []
        for csv_path in csv_paths:
            profile = _profile_csv(csv_path)
            if profile is not None:
                csv_profiles.append(profile)

        if not csv_profiles:
            logger.warning("All CSVs unreadable/empty for '%s', skipping.", ref)
            return None

        # Weight each CSV's quality score by its row count so large,
        # well-populated files matter more than tiny auxiliary ones.
        total_rows = sum(cp.num_rows for cp in csv_profiles) or 1
        quality_score = sum(cp.quality_score * (cp.num_rows / total_rows) for cp in csv_profiles)

        combined_score = candidate.score + quality_score

        return DatasetProfile(
            ref=ref,
            title=candidate.dataset.title,
            metadata_score=candidate.score,
            quality_score=round(quality_score, 2),
            combined_score=round(combined_score, 2),
            csv_profiles=csv_profiles,
        )
