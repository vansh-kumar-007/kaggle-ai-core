"""
profile_formatting.py

Renders a DatasetProfile (from dataset_analyzer.py) into compact,
LLM-readable text. Shared by any pipeline that needs to hand a profiled
dataset's real structure to an LLM inside a prompt.
"""

from __future__ import annotations

from kaggle_ai_core.dataset_analyzer import DatasetProfile


def render_profile_summary(profile: DatasetProfile) -> str:
    """
    Render a DatasetProfile into a compact, LLM-readable text summary --
    condensed enough to fit comfortably in a prompt, but detailed enough
    for an LLM to reason about real column roles and data quality.
    """
    lines: list[str] = [
        f"Dataset: {profile.title} ({profile.ref})",
        f"Metadata score: {profile.metadata_score:.1f} | Data quality score: {profile.quality_score:.1f}",
        "",
    ]

    for csv in profile.csv_profiles:
        lines.append(f"--- File: {csv.filename} ---")
        lines.append(
            f"Rows: {csv.num_rows} | Columns: {csv.num_cols} | "
            f"Missing: {csv.missing_pct_overall}% | Duplicates: {csv.duplicate_pct}%"
        )
        for col in csv.columns:
            target_note = " (POSSIBLE TARGET)" if col.is_target_candidate else ""
            lines.append(
                f"  - {col.name} | dtype={col.dtype} | role={col.role} | "
                f"missing={col.missing_pct}% | unique={col.unique_count}{target_note}"
            )
        lines.append("")

    return "\n".join(lines)