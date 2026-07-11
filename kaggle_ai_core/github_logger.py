"""
github_logger.py

Structured per-run logging. Writes a JSON summary to logs/ for every
pipeline run, and -- when running inside GitHub Actions -- appends a
human-readable markdown summary to $GITHUB_STEP_SUMMARY, so outcomes
are visible directly in the Actions UI without digging through raw logs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Structured record of one pipeline run's outcome."""

    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    success: bool = False
    dataset_ref: str | None = None
    dataset_title: str | None = None
    notebook_title: str | None = None
    kernel_url: str | None = None
    execution_attempts: int = 0
    repairs_applied: int = 0
    failure_reason: str | None = None

    def mark_finished(self, success: bool, failure_reason: str | None = None) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.success = success
        self.failure_reason = failure_reason

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "dataset_ref": self.dataset_ref,
            "dataset_title": self.dataset_title,
            "notebook_title": self.notebook_title,
            "kernel_url": self.kernel_url,
            "execution_attempts": self.execution_attempts,
            "repairs_applied": self.repairs_applied,
            "failure_reason": self.failure_reason,
        }


def write_run_log(summary: RunSummary, logs_dir: Path) -> Path:
    """Persist the run summary as timestamped JSON, and mirror it to GitHub's step summary if present."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{ts}.json"
    log_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    logger.info("Run summary written to %s", log_path)

    _write_github_step_summary(summary)
    return log_path


def _write_github_step_summary(summary: RunSummary) -> None:
    """Append a markdown summary to $GITHUB_STEP_SUMMARY, if set (only present inside GitHub Actions)."""
    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not step_summary_path:
        return

    status_emoji = "✅" if summary.success else "❌"
    lines = [
        f"## {status_emoji} Kaggle AI Notebook Agent — Daily Run",
        "",
        f"- **Status:** {'Success' if summary.success else 'Failed'}",
        f"- **Started:** {summary.started_at}",
        f"- **Finished:** {summary.finished_at}",
    ]
    if summary.dataset_title:
        lines.append(f"- **Dataset:** {summary.dataset_title} (`{summary.dataset_ref}`)")
    if summary.notebook_title:
        lines.append(f"- **Notebook:** {summary.notebook_title}")
    if summary.kernel_url:
        lines.append(f"- **Published:** [{summary.kernel_url}]({summary.kernel_url})")
    lines.append(f"- **Execution attempts:** {summary.execution_attempts}")
    lines.append(f"- **Repairs applied:** {summary.repairs_applied}")
    if summary.failure_reason:
        lines.append(f"- **Failure reason:** {summary.failure_reason}")

    try:
        with open(step_summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        logger.warning("Could not write to GITHUB_STEP_SUMMARY at %s", step_summary_path)
