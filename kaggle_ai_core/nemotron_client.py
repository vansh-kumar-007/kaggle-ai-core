"""
nemotron_client.py

Thin wrapper around the Nemotron 3 Ultra API (via NVIDIA's OpenAI-compatible
endpoint). Centralizes authentication, retry logic, and error handling for
all LLM calls used later in the pipeline (planning, section generation,
notebook repair).
"""

from __future__ import annotations

import logging

from openai import APIStatusError, OpenAI

from kaggle_ai_core.config import Settings
from kaggle_ai_core.utils.retry import with_retry

logger = logging.getLogger(__name__)

# Errors worth retrying: rate limits, transient server issues, and NVIDIA's
# own "model temporarily degraded" signal.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

NVIDIA_DEGRADED_RETRY_INTERVAL_SECONDS = 90
NVIDIA_DEGRADED_MAX_WAIT_SECONDS = 40 * 60  # ~40 min ceiling for a single call's outage wait -- generous given outages are typically ~30 min


class NemotronUnavailableError(RuntimeError):
    """Raised when the Nemotron endpoint remains DEGRADED beyond our extended wait budget."""


class NemotronClient:
    """
    Wrapper around the Nemotron 3 Ultra chat completion API.

    Uses the OpenAI SDK pointed at NVIDIA's integrate.api.nvidia.com
    endpoint, since Nemotron is served through an OpenAI-compatible API.
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.nemotron_model
        self._client = OpenAI(
            base_url=settings.nemotron_base_url,
            api_key=settings.nvidia_api_key,
        )
        logger.info("NemotronClient initialized with model '%s'", self._model)

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a prompt to Nemotron and return the generated text.

        Args:
            prompt: The user-facing prompt content.
            system_instruction: Optional system message to steer behavior.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens to generate in the response.

        Returns:
            The generated text content.

        Raises:
            Exception: propagated if the API call fails after retries.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        logger.info(
            "Calling Nemotron (model=%s, temperature=%.2f, max_tokens=%d, prompt_len=%d chars)",
            self._model, temperature, max_tokens, len(prompt),
        )

        response = self._call_with_degraded_handling(messages, temperature, max_tokens)
        content = response.choices[0].message.content or ""

        logger.info("Nemotron response received (%d chars).", len(content))
        return content
    
    def _call_with_degraded_handling(self, messages: list[dict], temperature: float, max_tokens: int):
        """
        Wraps _call_with_retry (which already handles ordinary transient
        blips with quick backoff) with patient, long-interval retrying
        specifically for NVIDIA's "DEGRADED function cannot be invoked"
        signal -- a real server-side outage, not a malformed request.
        These outages are typically ~30 minutes; waiting them out here
        means the entire pipeline (including automated GitHub Actions
        runs) simply pauses and continues on its own, rather than crashing.
        """
        waited = 0
        while True:
            try:
                return self._call_with_retry(messages, temperature, max_tokens)
            except APIStatusError as exc:
                if exc.status_code == 400 and "DEGRADED" in str(exc):
                    if waited >= NVIDIA_DEGRADED_MAX_WAIT_SECONDS:
                        raise NemotronUnavailableError(
                            f"Nemotron endpoint still reports DEGRADED after waiting {waited}s; giving up on this call."
                        ) from exc
                    logger.warning(
                        "Nemotron endpoint is DEGRADED (temporary NVIDIA-side outage, typically ~30 min). "
                        "Waiting %ds before retrying (total waited so far: %ds/%ds budget)...",
                        NVIDIA_DEGRADED_RETRY_INTERVAL_SECONDS, waited, NVIDIA_DEGRADED_MAX_WAIT_SECONDS,
                    )
                    time.sleep(NVIDIA_DEGRADED_RETRY_INTERVAL_SECONDS)
                    waited += NVIDIA_DEGRADED_RETRY_INTERVAL_SECONDS
                    continue
                raise

    @with_retry(max_attempts=5)
    def _call_with_retry(self, messages: list[dict], temperature: float, max_tokens: int):
        try:
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIStatusError as exc:
            if exc.status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    "Nemotron call failed with retryable status %d: %s",
                    exc.status_code, exc,
                )
                raise
            logger.error("Nemotron call failed with non-retryable status %d: %s", exc.status_code, exc)
            raise
