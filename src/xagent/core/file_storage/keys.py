"""Canonical builders for durable-storage keys.

This module is the single source of truth for the object-storage key layout
(``users/{user_id}/uploads/...`` and ``users/{user_id}/tasks/{task_id}/outputs/...``).
Builders sanitize their inputs so the produced keys always pass strict
normalization (see ``normalize_storage_key``); keys already persisted in the
database are read back verbatim and never re-derived through these builders.
"""

from __future__ import annotations

from pathlib import Path


def safe_storage_filename(filename: str) -> str:
    safe_name = Path(filename).name.strip()
    if safe_name in ("", ".", ".."):
        return "file"
    return _sanitize_component(safe_name) or "file"


def build_upload_storage_key(user_id: int, file_id: str, filename: str) -> str:
    return f"users/{user_id}/uploads/{file_id}/{safe_storage_filename(filename)}"


def build_task_output_storage_key(
    user_id: int, task_id: int, file_id: str, relative_path: str
) -> str:
    return (
        f"users/{user_id}/tasks/{task_id}/outputs/"
        f"{file_id}/{_safe_relative_output_path(relative_path)}"
    )


def _safe_relative_output_path(relative_path: str) -> str:
    components = [
        part for part in relative_path.strip().split("/") if part not in ("", ".")
    ]
    if ".." in components:
        # A traversal component means the structure cannot be trusted;
        # keep only a safe basename.
        return safe_storage_filename(relative_path)
    if not components:
        return "file"
    return "/".join(_sanitize_component(part) for part in components)


def _sanitize_component(component: str) -> str:
    return "".join(
        "_" if ch == "\\" or ord(ch) < 32 or ord(ch) == 127 else ch for ch in component
    )
