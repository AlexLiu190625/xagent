from __future__ import annotations

import asyncio
import io
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.web.api import files as files_api
from xagent.web.models import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import uploaded_file_store


class _ObservedStream(io.BytesIO):
    def __init__(
        self,
        *,
        content: bytes,
        assert_no_checkout: Any,
    ) -> None:
        super().__init__(content)
        self._assert_no_checkout = assert_no_checkout

    def read(self, size: int = -1) -> bytes:
        self._assert_no_checkout()
        return super().read(size)


class _ObservedUpload(UploadFile):
    def __init__(
        self,
        *,
        filename: str,
        content: bytes,
        assert_no_checkout: Any,
    ) -> None:
        super().__init__(
            file=_ObservedStream(
                content=content,
                assert_no_checkout=assert_no_checkout,
            ),
            filename=filename,
        )


class _CancellingAfterFirstChunk(io.BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads > 1:
            raise asyncio.CancelledError
        return super().read(size)


@pytest.fixture
def isolated_upload_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[sessionmaker[Session], QueuePool, Path, Path]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'uploads.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory.begin() as db:
        db.add(
            User(
                id=7,
                username="upload-boundary-user",
                password_hash="unused",
                is_admin=False,
            )
        )

    uploads_root = tmp_path / "uploads"
    durable_root = tmp_path / "durable"
    uploads_root.mkdir()
    durable_root.mkdir()

    def upload_path(
        filename: str,
        task_id: str | None = None,
        folder: str | None = None,
        user_id: int | None = None,
        **_kwargs: Any,
    ) -> Path:
        del task_id, folder
        target_dir = uploads_root / str(user_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / Path(filename).name

    monkeypatch.setattr(files_api, "get_upload_path", upload_path)
    monkeypatch.setattr(
        uploaded_file_store, "get_session_local", lambda: session_factory
    )
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", durable_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    try:
        yield session_factory, engine.pool, uploads_root, durable_root
    finally:
        get_unscoped_file_storage.cache_clear()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.asyncio
async def test_v1_upload_reads_and_durable_work_without_a_checked_out_connection(
    isolated_upload_runtime: tuple[sessionmaker[Session], QueuePool, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, pool, _uploads_root, _durable_root = isolated_upload_runtime
    event_loop_thread = threading.get_ident()
    checkout_threads: list[int] = []
    durable_threads: list[int] = []

    @event.listens_for(pool, "checkout")
    def record_checkout(*_args: Any) -> None:
        checkout_threads.append(threading.get_ident())

    original_put_file = FsspecFileStorage.put_file

    def observed_put_file(self: FsspecFileStorage, *args: Any, **kwargs: Any):
        assert pool.checkedout() == 0
        durable_threads.append(threading.get_ident())
        return original_put_file(self, *args, **kwargs)

    monkeypatch.setattr(FsspecFileStorage, "put_file", observed_put_file)
    uploads = [
        _ObservedUpload(
            filename="first.txt",
            content=b"first",
            assert_no_checkout=lambda: (
                pool.checkedout() == 0
                or pytest.fail("upload read held a database connection")
            ),
        ),
        _ObservedUpload(
            filename="second.txt",
            content=b"second",
            assert_no_checkout=lambda: (
                pool.checkedout() == 0
                or pytest.fail("upload read held a database connection")
            ),
        ),
    ]

    result = await files_api.store_v1_uploaded_files(
        upload_items=uploads,
        task_type="general",
        folder=None,
        owner_user_id=7,
        single_file_mode=False,
    )

    assert result["total_files"] == 2
    assert pool.checkedout() == 0
    assert checkout_threads
    assert all(thread_id != event_loop_thread for thread_id in checkout_threads)
    assert durable_threads
    assert all(thread_id != event_loop_thread for thread_id in durable_threads)
    with session_factory() as db:
        rows = db.query(UploadedFile).order_by(UploadedFile.filename).all()
        assert [str(row.filename) for row in rows] == ["first.txt", "second.txt"]
        assert all(str(row.storage_status) == "available" for row in rows)
        assert all(row.task_id is None for row in rows)


@pytest.mark.asyncio
async def test_v1_upload_path_and_local_copy_run_off_the_event_loop(
    isolated_upload_runtime: tuple[sessionmaker[Session], QueuePool, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_factory, pool, _uploads_root, _durable_root = isolated_upload_runtime
    resolve_path = getattr(files_api, "_resolve_v1_upload_path_sync", None)
    copy_local = getattr(files_api, "_copy_upload_to_local_path_sync", None)
    assert callable(resolve_path), "missing worker-owned upload path resolver"
    assert callable(copy_local), "missing worker-owned local upload copier"

    loop_thread = threading.get_ident()
    path_threads: list[int] = []
    copy_threads: list[int] = []

    def observed_path(*args: Any, **kwargs: Any):
        path_threads.append(threading.get_ident())
        return resolve_path(*args, **kwargs)

    def observed_copy(*args: Any, **kwargs: Any):
        assert pool.checkedout() == 0
        copy_threads.append(threading.get_ident())
        return copy_local(*args, **kwargs)

    monkeypatch.setattr(files_api, "_resolve_v1_upload_path_sync", observed_path)
    monkeypatch.setattr(files_api, "_copy_upload_to_local_path_sync", observed_copy)

    await files_api.store_v1_uploaded_files(
        upload_items=[
            UploadFile(file=io.BytesIO(b"local staging"), filename="staging.txt")
        ],
        task_type="general",
        folder=None,
        owner_user_id=7,
        single_file_mode=False,
    )

    assert path_threads and all(thread_id != loop_thread for thread_id in path_threads)
    assert copy_threads and all(thread_id != loop_thread for thread_id in copy_threads)


@pytest.mark.asyncio
async def test_v1_upload_read_cancellation_removes_partial_local_file(
    isolated_upload_runtime: tuple[sessionmaker[Session], QueuePool, Path, Path],
) -> None:
    session_factory, pool, uploads_root, durable_root = isolated_upload_runtime
    upload = UploadFile(
        file=_CancellingAfterFirstChunk(b"partial"),
        filename="cancelled.txt",
    )

    with pytest.raises(asyncio.CancelledError):
        await files_api.store_v1_uploaded_files(
            upload_items=[upload],
            task_type="general",
            folder=None,
            owner_user_id=7,
            single_file_mode=False,
        )

    assert pool.checkedout() == 0
    with session_factory() as db:
        assert db.query(UploadedFile).count() == 0
    assert [path for path in uploads_root.rglob("*") if path.is_file()] == []
    assert [path for path in durable_root.rglob("*") if path.is_file()] == []


@pytest.mark.asyncio
async def test_v1_upload_durable_failure_leaves_no_rows_or_artifacts(
    isolated_upload_runtime: tuple[sessionmaker[Session], QueuePool, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, _pool, uploads_root, durable_root = isolated_upload_runtime
    original_put_file = FsspecFileStorage.put_file
    call_count = 0

    def fail_second_put(self: FsspecFileStorage, *args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("durable write failed")
        return original_put_file(self, *args, **kwargs)

    monkeypatch.setattr(FsspecFileStorage, "put_file", fail_second_put)

    with pytest.raises(Exception, match="Durable storage is temporarily unavailable"):
        await files_api.store_v1_uploaded_files(
            upload_items=[
                UploadFile(file=io.BytesIO(b"one"), filename="one.txt"),
                UploadFile(file=io.BytesIO(b"two"), filename="two.txt"),
            ],
            task_type="general",
            folder=None,
            owner_user_id=7,
            single_file_mode=False,
        )

    with session_factory() as db:
        assert db.query(UploadedFile).count() == 0
    assert [path for path in uploads_root.rglob("*") if path.is_file()] == []
    assert [path for path in durable_root.rglob("*") if path.is_file()] == []


@pytest.mark.asyncio
async def test_v1_upload_db_failure_compensates_durable_and_local_artifacts(
    isolated_upload_runtime: tuple[sessionmaker[Session], QueuePool, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, _pool, uploads_root, durable_root = isolated_upload_runtime

    class _CommitFailingSession(Session):
        def commit(self) -> None:
            self.flush()
            raise RuntimeError("database commit failed")

    failing_session_factory = sessionmaker(
        bind=session_factory.kw["bind"],
        class_=_CommitFailingSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr(
        uploaded_file_store, "get_session_local", lambda: failing_session_factory
    )

    with pytest.raises(RuntimeError, match="database commit failed"):
        await files_api.store_v1_uploaded_files(
            upload_items=[UploadFile(file=io.BytesIO(b"one"), filename="one.txt")],
            task_type="general",
            folder=None,
            owner_user_id=7,
            single_file_mode=False,
        )

    with session_factory() as db:
        assert db.query(UploadedFile).count() == 0
    assert [path for path in uploads_root.rglob("*") if path.is_file()] == []
    assert [path for path in durable_root.rglob("*") if path.is_file()] == []


@pytest.mark.asyncio
async def test_v1_upload_cancellation_drains_commit_to_an_all_state(
    isolated_upload_runtime: tuple[sessionmaker[Session], QueuePool, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, pool, uploads_root, durable_root = isolated_upload_runtime
    entered_persist = threading.Event()
    release_persist = threading.Event()
    original_persist = uploaded_file_store.persist_prepared_uploaded_files

    def blocked_persist(files: Any) -> Any:
        entered_persist.set()
        assert release_persist.wait(timeout=5)
        return original_persist(files)

    monkeypatch.setattr(
        uploaded_file_store, "persist_prepared_uploaded_files", blocked_persist
    )

    upload_task = asyncio.create_task(
        files_api.store_v1_uploaded_files(
            upload_items=[UploadFile(file=io.BytesIO(b"one"), filename="one.txt")],
            task_type="general",
            folder=None,
            owner_user_id=7,
            single_file_mode=False,
        )
    )
    assert await asyncio.to_thread(entered_persist.wait, 5)
    upload_task.cancel()
    release_persist.set()
    with pytest.raises(asyncio.CancelledError):
        await upload_task

    assert pool.checkedout() == 0
    with session_factory() as db:
        row = db.query(UploadedFile).one()
        assert row.task_id is None
    assert len([path for path in uploads_root.rglob("*") if path.is_file()]) == 1
    assert len([path for path in durable_root.rglob("*") if path.is_file()]) == 1


@pytest.mark.asyncio
async def test_v1_upload_cancellation_drains_failed_commit_to_a_none_state(
    isolated_upload_runtime: tuple[sessionmaker[Session], QueuePool, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory, pool, uploads_root, durable_root = isolated_upload_runtime
    entered_commit = threading.Event()
    release_commit = threading.Event()

    class _BlockedCommitFailingSession(Session):
        def commit(self) -> None:
            entered_commit.set()
            assert release_commit.wait(timeout=5)
            self.flush()
            raise RuntimeError("database commit failed")

    failing_session_factory = sessionmaker(
        bind=session_factory.kw["bind"],
        class_=_BlockedCommitFailingSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr(
        uploaded_file_store, "get_session_local", lambda: failing_session_factory
    )

    upload_task = asyncio.create_task(
        files_api.store_v1_uploaded_files(
            upload_items=[UploadFile(file=io.BytesIO(b"one"), filename="one.txt")],
            task_type="general",
            folder=None,
            owner_user_id=7,
            single_file_mode=False,
        )
    )
    assert await asyncio.to_thread(entered_commit.wait, 5)
    upload_task.cancel()
    release_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await upload_task

    assert pool.checkedout() == 0
    with session_factory() as db:
        assert db.query(UploadedFile).count() == 0
    assert [path for path in uploads_root.rglob("*") if path.is_file()] == []
    assert [path for path in durable_root.rglob("*") if path.is_file()] == []
