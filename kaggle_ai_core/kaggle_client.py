"""
kaggle_client.py

Thin wrapper around the Kaggle API's dataset-related endpoints.

Centralizes all direct interaction with the `kaggle` package so the rest
of the application never touches the raw API. This keeps logging,
error handling, and (later) retry logic in one place.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
import builtins
from contextlib import contextmanager
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

from kaggle_ai_core.utils.retry import with_retry

@contextmanager
def _force_utf8_default_encoding():
    """
    Work around a Windows-specific bug in the kaggle package: kernels_push()
    opens the notebook file via open(path) with no explicit encoding, which
    on Windows falls back to the system locale codepage (cp1252) rather than
    UTF-8. Notebooks containing legitimate non-ASCII characters (degree
    signs, Greek letters, emojis) then crash with UnicodeDecodeError deep
    inside the third-party library, which we can't patch directly.

    Patching locale.getpreferredencoding() directly doesn't work here --
    modern CPython resolves open()'s default encoding through an internal
    C-level path that bypasses that public function entirely. Instead,
    this temporarily wraps the builtins.open() function itself: any text-mode
    open() call with no explicit encoding gets "utf-8" forced in, for the
    duration of the push only. Restored afterward regardless of outcome.
    """
    original_open = builtins.open

    def _utf8_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if encoding is None and "b" not in mode:
            encoding = "utf-8"
        return original_open(file, mode, buffering, encoding, *args, **kwargs)

    builtins.open = _utf8_open
    try:
        yield
    finally:
        builtins.open = original_open


logger = logging.getLogger(__name__)


@dataclass
class DatasetSearchResult:
    """
    Structured representation of a single dataset search result.

    Attributes:
        ref: Kaggle dataset reference in "owner/dataset-slug" form.
        title: Human-readable dataset title.
        subtitle: Short description/subtitle, if provided.
        owner_name: Username or organization that published the dataset.
        size_bytes: Total dataset size in bytes, as reported by Kaggle.
        vote_count: Number of upvotes the dataset has received.
        usability_rating: Kaggle's usability score (0.0 - 1.0).
        total_views: Total view count (often 0 -- not always populated by this API version).
        total_downloads: Total download count (often 0 -- same caveat as total_views).
        file_count: Number of files in the dataset (often 0 -- same caveat).
        tags: List of tag names associated with the dataset.
        license_name: Name of the dataset's license (e.g. "CC0-1.0"), if provided.
        last_updated: ISO-format string of when the dataset was last updated, if provided.
        current_version_number: The dataset's current version number.
        kernel_count: Number of public Kaggle notebooks built on this dataset --
            a strong signal of real community engagement and prior art to learn from.
        topic_count: Number of discussion threads on the dataset's forum page --
            another engagement signal, distinct from raw votes.
    """

    ref: str
    title: str
    subtitle: str
    owner_name: str
    size_bytes: int
    vote_count: int
    usability_rating: float
    total_views: int
    total_downloads: int
    file_count: int
    tags: list[str]
    license_name: str
    last_updated: str
    current_version_number: int
    kernel_count: int
    topic_count: int


class KaggleClient:
    """
    Wrapper around the Kaggle API for dataset discovery and management.

    Handles authentication once at construction time. All public methods
    log their operation and raise on failure — callers are responsible
    for retry/error-handling policy (added in a later step).
    """

    def __init__(self) -> None:
        self._api = KaggleApi()
        self._api.authenticate()
        logger.info("KaggleClient authenticated successfully.")

    def search_datasets(
        self,
        search_term: str,
        max_results: int = 20,
    ) -> list[DatasetSearchResult]:
        """
        Search Kaggle for datasets matching a search term.

        Args:
            search_term: Free-text search query (e.g. "finance", "health").
            max_results: Maximum number of results to return (Kaggle
                paginates at 20 per page; we only fetch page 1 for now).

        Returns:
            A list of DatasetSearchResult objects, one per dataset found.

        Raises:
            Exception: propagated from the underlying Kaggle API call if
                the search request fails (e.g. network error, auth issue).
        """
        logger.info("Searching Kaggle datasets for term: '%s'", search_term)

        try:
            raw_results = self._api.dataset_list(search=search_term, page=1)
        except Exception:
            logger.exception("Kaggle dataset search failed for term '%s'", search_term)
            raise

        results: list[DatasetSearchResult] = []
        for item in raw_results[:max_results]:
            try:
                results.append(
                    DatasetSearchResult(
                        ref=item.ref,
                        title=item.title,
                        subtitle=getattr(item, "subtitle", "") or "",
                        owner_name=getattr(item, "ownerName", "") or "",
                        size_bytes=getattr(item, "totalBytes", 0) or 0,
                        vote_count=getattr(item, "voteCount", 0) or 0,
                        usability_rating=getattr(item, "usabilityRating", 0.0) or 0.0,
                        total_views=getattr(item, "totalViews", 0) or 0,
                        total_downloads=getattr(item, "totalDownloads", 0) or 0,
                        file_count=getattr(item, "fileCount", 0) or 0,
                        tags=[tag.name for tag in getattr(item, "tags", [])] or [],
                        license_name=getattr(item, "licenseName", "") or "",
                        last_updated=str(getattr(item, "lastUpdated", "") or ""),
                        current_version_number=getattr(item, "currentVersionNumber", 0) or 0,
                        kernel_count=getattr(item, "kernelCount", 0) or 0,
                        topic_count=getattr(item, "topicCount", 0) or 0,
                    )
                )
            except Exception:
                logger.warning(
                    "Skipping malformed dataset result: %s", getattr(item, "ref", "<unknown>")
                )
                continue

        logger.info("Found %d dataset(s) for term '%s'.", len(results), search_term)
        return results
    def download_dataset_files(self, ref: str, dest_dir: Path) -> None:
        """
        Download and unzip all files for a given dataset into dest_dir.

        Args:
            ref: Dataset reference ("owner/dataset-slug").
            dest_dir: Local directory to download into. Created if missing.

        Raises:
            Exception: propagated from the underlying Kaggle API call if
                the download fails after retries.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading dataset '%s' to %s", ref, dest_dir)
        self._download_with_retry(ref, dest_dir)
        logger.info("Download complete for '%s'.", ref)
        
    def push_kernel(self, folder: Path) -> None:
        """
        Push (upload and publish/update) a kernel from a local folder
        containing kernel-metadata.json and the notebook file.

        Args:
            folder: Directory containing kernel-metadata.json + the .ipynb.

        Raises:
            Exception: propagated from the underlying Kaggle API call if
                the push fails after retries.
        """
        logger.info("Pushing kernel from %s", folder)
        self._push_with_retry(folder)
        logger.info("Kernel push complete for %s", folder)
        
    def create_model(self, folder: Path) -> None:
        """Create a new top-level Kaggle Model entity from model-metadata.json in folder."""
        logger.info("Creating Kaggle Model from %s", folder)
        self._create_model_with_retry(folder)
        logger.info("Model creation complete for %s", folder)

    @with_retry(max_attempts=3)
    def _create_model_with_retry(self, folder: Path) -> None:
        with _force_utf8_default_encoding():
            self._api.model_create_new(str(folder))

    def create_model_instance(self, folder: Path) -> str:
        """
        Create a Model Instance (and upload its files) from
        model-instance-metadata.json in folder.

        Returns:
            The instance URL as reported by Kaggle's own API response
            (e.g. ".../models/owner/slug/ScikitLearn/default"). Note this
            may use different capitalization for the framework segment
            than what was submitted as input (e.g. submitting "scikitLearn"
            can come back as "ScikitLearn" in the URL) -- callers needing
            to reference this instance elsewhere (e.g. linking a notebook
            to it via kernel-metadata.json's model_sources) should parse
            the framework segment FROM THIS RETURNED URL, not reconstruct
            it from the submitted value, to avoid a silent case mismatch.
        """
        logger.info("Creating Kaggle Model Instance from %s", folder)
        result = self._create_model_instance_with_retry(folder)
        logger.info("Model Instance creation complete for %s", folder)
        return str(result)

    @with_retry(max_attempts=3)
    def _create_model_instance_with_retry(self, folder: Path):
        with _force_utf8_default_encoding():
            return self._api.model_instance_create(str(folder))

    def push_kernel(self, folder: Path) -> None:
        """
        Push (upload and publish/update) a kernel from a local folder
        containing kernel-metadata.json and the notebook file.
        """
        logger.info("Pushing kernel from %s", folder)
        self._push_kernel_with_retry(folder)
        logger.info("Kernel push complete for %s", folder)

    @with_retry(max_attempts=3)
    def _push_kernel_with_retry(self, folder: Path) -> None:
        with _force_utf8_default_encoding():
            self._api.kernels_push(str(folder))
    
    @with_retry(max_attempts=3)
    def _create_model_instance_with_retry(self, folder: Path) -> None:
        with _force_utf8_default_encoding():
            self._api.model_instance_create(str(folder))

    @with_retry(max_attempts=3)
    def _push_with_retry(self, folder: Path) -> None:
        with _force_utf8_default_encoding():
            self._api.kernels_push(str(folder))

    @with_retry(max_attempts=3)
    def _download_with_retry(self, ref: str, dest_dir: Path) -> None:
        self._api.dataset_download_files(ref, path=str(dest_dir), unzip=True, quiet=True)
        
