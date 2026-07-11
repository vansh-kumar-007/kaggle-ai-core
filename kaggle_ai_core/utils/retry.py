"""
retry.py

Shared retry decorator built on tenacity, used for network-dependent
operations (Kaggle downloads, API calls) that may fail transiently.
"""

from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def with_retry(max_attempts: int = 3):
    """
    Return a tenacity retry decorator configured for transient network
    failures: exponential backoff, capped attempts, and a log line
    before each retry.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
