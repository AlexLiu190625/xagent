from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from sqlalchemy.orm import Session

from ...core.file_storage import FsspecFileStorage, get_file_storage
from ..models.uploaded_file import UploadedFile
from .managed_file_ref import ManagedFileRef, build_upload_storage_key

logger = logging.getLogger(__name__)

_sync_lock = threading.Lock()
_FILE_LOCK_RETRY_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class StartupFileStorageSyncResult:
    scanned: int = 0
    already_present: int = 0
    uploaded: int = 0
    skipped_missing_local: int = 0
    skipped_backend: int = 0
    failed: int = 0
    locked: bool = False


def sync_registered_files_to_durable_storage(
    db: Session,
    *,
    storage: FsspecFileStorage | Any | None = None,
    batch_size: int = 500,
) -> StartupFileStorageSyncResult:
    """Reconcile registered local files with S3-backed durable storage."""
    resolved_storage = storage or get_file_storage()
    backend = str(getattr(resolved_storage, "backend", ""))
    if not backend:
        backend = str(getattr(resolved_storage, "_backend", ""))

    if backend != "s3":
        logger.info(
            "Skipping startup file storage sync for non-S3 backend: %s",
            backend or "unknown",
        )
        return StartupFileStorageSyncResult(skipped_backend=1)

    if not _sync_lock.acquire(blocking=False):
        logger.info("Startup file storage sync is already running in this process")
        return StartupFileStorageSyncResult(locked=True)

    file_lock = None
    try:
        file_lock = _acquire_file_lock_after_contention()

        return _sync_registered_files(
            db,
            storage=resolved_storage,
            batch_size=batch_size,
        )
    finally:
        if file_lock is not None:
            _release_file_lock(file_lock)
        _sync_lock.release()


def _acquire_file_lock_after_contention() -> Any:
    file_lock = _acquire_file_lock()
    while file_lock is None:
        logger.info(
            "Startup file storage sync is already running in another process; waiting"
        )
        _wait_for_lock_holder()
        file_lock = _acquire_file_lock()
    return file_lock


def _wait_for_lock_holder() -> None:
    time.sleep(_FILE_LOCK_RETRY_INTERVAL_SECONDS)


def _sync_registered_files(
    db: Session,
    *,
    storage: FsspecFileStorage | Any,
    batch_size: int,
) -> StartupFileStorageSyncResult:
    scanned = 0
    already_present = 0
    uploaded = 0
    skipped_missing_local = 0
    failed = 0

    rows = (
        db.query(UploadedFile)
        .order_by(UploadedFile.user_id.asc(), UploadedFile.id.asc())
        .yield_per(batch_size)
    )
    current_user_id: int | None = None
    remote_objects: dict[str, Any] = {}
    batch_updates = 0

    for record in rows:
        scanned += 1
        user_id = int(getattr(record, "user_id"))
        if user_id != current_user_id:
            current_user_id = user_id
            remote_objects = _list_remote_objects_for_user(storage, user_id)

        expected_key = _expected_storage_key(record)
        remote_object = remote_objects.get(expected_key)
        if remote_object is not None:
            if not _has_complete_durable_metadata(record):
                try:
                    adopt_result = ManagedFileRef(
                        record, storage=storage
                    ).adopt_existing_object(expected_key)
                except Exception:
                    failed += 1
                    logger.exception(
                        "Failed startup durable adoption for file_id=%s key=%s",
                        getattr(record, "file_id", None),
                        expected_key,
                    )
                    continue
                if adopt_result == "missing":
                    local_path = Path(str(getattr(record, "storage_path", "")))
                    if not local_path.exists() or not local_path.is_file():
                        skipped_missing_local += 1
                    else:
                        failed += 1
                    continue
                if adopt_result == "uploaded":
                    uploaded += 1
                batch_updates += 1
            already_present += 1
            continue

        local_path = Path(str(getattr(record, "storage_path", "")))
        if not local_path.exists() or not local_path.is_file():
            skipped_missing_local += 1
            logger.warning(
                "Skipping startup durable sync for missing local file: file_id=%s path=%s",
                getattr(record, "file_id", None),
                local_path,
            )
            continue

        try:
            stored_object = ManagedFileRef(record, storage=storage).sync_to_durable(
                storage_key=expected_key,
                mime_type=getattr(record, "mime_type", None),
            )
            remote_objects[expected_key] = stored_object
            uploaded += 1
            batch_updates += 1
        except Exception:
            failed += 1
            logger.exception(
                "Failed startup durable sync for file_id=%s path=%s key=%s",
                getattr(record, "file_id", None),
                local_path,
                expected_key,
            )
            continue

        if batch_updates >= batch_size:
            db.commit()
            batch_updates = 0

    if batch_updates:
        db.commit()

    result = StartupFileStorageSyncResult(
        scanned=scanned,
        already_present=already_present,
        uploaded=uploaded,
        skipped_missing_local=skipped_missing_local,
        failed=failed,
    )
    logger.info(
        "Startup file storage sync complete: scanned=%s already_present=%s uploaded=%s skipped_missing_local=%s failed=%s",
        result.scanned,
        result.already_present,
        result.uploaded,
        result.skipped_missing_local,
        result.failed,
    )
    return result


def _expected_storage_key(record: UploadedFile) -> str:
    existing_key = str(getattr(record, "storage_key", "") or "").strip()
    if existing_key:
        return existing_key
    return build_upload_storage_key(
        int(getattr(record, "user_id")),
        str(getattr(record, "file_id")),
        str(getattr(record, "filename")),
    )


def _has_complete_durable_metadata(record: UploadedFile) -> bool:
    return bool(
        getattr(record, "storage_key", None)
        and getattr(record, "storage_backend", None) == "s3"
        and getattr(record, "storage_status", None) == "available"
        and getattr(record, "checksum", None)
    )


def _list_remote_objects_for_user(
    storage: FsspecFileStorage | Any, user_id: int
) -> dict[str, Any]:
    return {stored.key: stored for stored in storage.list(f"users/{user_id}")}


def _get_lock_file_path() -> str:
    return os.environ.get(
        "XAGENT_FILE_STORAGE_STARTUP_SYNC_LOCK_FILE",
        os.path.join(tempfile.gettempdir(), "xagent_file_storage_startup_sync.lock"),
    )


def _acquire_file_lock() -> Any | None:
    lock_path = _get_lock_file_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    try:
        lock = FileLock(lock_path, timeout=0)
        lock.acquire()
        Path(lock_path).write_text(str(os.getpid()), encoding="utf-8")
        return lock
    except Timeout:
        return None


def _release_file_lock(lock_file: Any) -> None:
    lock_file.release()
