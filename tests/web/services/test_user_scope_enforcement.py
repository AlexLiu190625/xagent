"""End-to-end enforcement of user-scoped storage handles at the call sites.

ManagedFileRef defaults to a handle scoped to ``users/{record.user_id}``;
these tests prove a record whose storage_key targets another user's prefix
cannot read, write, sign, adopt, or delete through any entry point.
"""

from pathlib import Path

import pytest

from xagent.core.file_storage import StorageKeyScopeError
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import (
    DurableStorageOperationError,
    ManagedFileRef,
)


@pytest.fixture
def storage_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    get_unscoped_file_storage.cache_clear()
    yield tmp_path
    get_unscoped_file_storage.cache_clear()


def _record(local_path: Path, **overrides) -> UploadedFile:
    values = {
        "file_id": "file-123",
        "user_id": 7,
        "filename": local_path.name,
        "storage_path": str(local_path),
        "storage_status": "legacy",
        "mime_type": "text/plain",
        "file_size": 0,
    }
    values.update(overrides)
    return UploadedFile(**values)


def _foreign_key_record(local_path: Path) -> UploadedFile:
    return _record(
        local_path,
        storage_key="users/8/uploads/file-123/source.txt",
        storage_backend="file",
        storage_status="available",
    )


def test_round_trip_through_default_user_scoped_storage(storage_env, tmp_path):
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("scoped round trip", encoding="utf-8")
    record = _record(source)

    stored = ManagedFileRef(record).sync_to_durable()
    assert stored.key == "users/7/uploads/file-123/source.txt"
    assert record.storage_status == "available"

    source.unlink()
    restored = ManagedFileRef(record).ensure_local()
    assert restored.read_text(encoding="utf-8") == "scoped round trip"

    ManagedFileRef(record).delete_durable()
    assert not get_unscoped_file_storage().exists(stored.key)


def test_sync_to_durable_rejects_foreign_explicit_key(storage_env, tmp_path):
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("data", encoding="utf-8")
    record = _record(source)

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).sync_to_durable(
            storage_key="users/8/uploads/file-123/source.txt"
        )
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)
    assert not get_unscoped_file_storage().exists("users/8/uploads/file-123/source.txt")


def test_restore_rejects_foreign_storage_key(storage_env, tmp_path):
    record = _foreign_key_record(tmp_path / "uploads" / "missing.txt")

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).ensure_local()
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).materialize()
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)


def test_signed_url_never_issued_for_foreign_storage_key(storage_env, tmp_path):
    # signed_access_url degrades to None when the checksum probe fails; the
    # scope check makes that probe fail for foreign keys, so no URL is issued
    # even though the foreign object exists and the checksum matches.
    foreign = get_unscoped_file_storage().put_bytes(
        b"foreign", "users/8/uploads/file-123/missing.txt"
    )
    record = _record(
        tmp_path / "uploads" / "missing.txt",
        storage_key=foreign.key,
        storage_backend="file",
        storage_status="available",
        checksum=foreign.checksum,
    )

    assert ManagedFileRef(record).signed_access_url(expires=300) is None


def test_delete_durable_rejects_foreign_storage_key(storage_env, tmp_path):
    record = _foreign_key_record(tmp_path / "uploads" / "missing.txt")

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).delete_durable()


def test_adopt_existing_object_rejects_foreign_expected_key(storage_env, tmp_path):
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("data", encoding="utf-8")
    record = _record(source)
    # The foreign object exists, so a scope bypass would return "adopted".
    get_unscoped_file_storage().put_bytes(
        b"foreign", "users/8/uploads/file-123/source.txt"
    )

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).adopt_existing_object(
            "users/8/uploads/file-123/source.txt"
        )
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)


def test_separator_aware_scope_for_record_owner(storage_env, tmp_path):
    # user 1's handle must not admit a users/10 key.
    record = _record(
        tmp_path / "uploads" / "missing.txt",
        user_id=1,
        storage_key="users/10/uploads/file-123/source.txt",
        storage_backend="file",
        storage_status="available",
    )

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).delete_durable()


def test_record_without_owner_cannot_bind_default_scope(storage_env, tmp_path):
    record = _record(tmp_path / "uploads" / "missing.txt", user_id=None)

    with pytest.raises(ValueError, match="user_id is required"):
        ManagedFileRef(record)
