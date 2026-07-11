"""
config.py

Centralized configuration management for the Kaggle AI Notebook Agent.

Loads and validates all environment variables required by the application
using pydantic-settings. Fails fast and loudly if required configuration
is missing, rather than letting the app proceed into a pipeline run with
an invalid state.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Project paths -----------------------------------------------------

# Derived from the current working directory, not __file__ -- this file
# lives inside the kaggle-ai-core submodule, physically nested inside
# whichever consuming repo embeds it. Using cwd() means paths like
# NOTEBOOKS_DIR/STATE_DIR correctly resolve to the CONSUMING repo's root
# (where scripts are always run from), not this package's own location.
PROJECT_ROOT: Path = Path.cwd()
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# STATE_DIR holds data that must persist ACROSS GitHub Actions runs (ephemeral
# runners wipe everything else). Unlike logs/ and datasets/, this directory is
# git-tracked and committed back to the repo by the workflow after each run.
STATE_DIR: Path = PROJECT_ROOT / "state"
PUBLISHED_REFS_PATH: Path = STATE_DIR / "published_datasets.json"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Field names are matched case-insensitively against environment
    variable names (e.g. `kaggle_username` <-> KAGGLE_USERNAME).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    kaggle_username: str
    kaggle_key: str
    nvidia_api_key: str

    nemotron_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nemotron_base_url: str = "https://integrate.api.nvidia.com/v1"

    @field_validator("kaggle_username", "kaggle_key", "nvidia_api_key")
    @classmethod
    def _not_empty(cls, value: str, info) -> str:
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} must not be empty.")
        return value.strip()


def get_settings() -> Settings:
    """
    Load and return validated application settings.

    Raises:
        pydantic_core.ValidationError: if required environment variables
            are missing or invalid. This is intentional — the program
            should fail immediately at startup, not deep inside a
            pipeline run.
    """
    return Settings()  # type: ignore[call-arg]


def ensure_directories() -> None:
    """Create required project directories if they do not already exist."""
    for directory in (NOTEBOOKS_DIR, DATASETS_DIR, LOGS_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
