import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.service import AgentService
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.workspace import TaskWorkspace, WorkspaceManager
from xagent.web.models import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


def test_workspace_register_file_writes_durable_storage(
    monkeypatch, tmp_path, mock_workspace_db
):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="workspace-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=123, user_id=user.id, title="Workspace task")
        db.add(task)
        db.commit()

        workspace = TaskWorkspace(
            id="web_task_123", base_dir=str(tmp_path / "workspaces")
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("workspace output", encoding="utf-8")

        file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        assert record.storage_path == str(output_path)
        assert record.storage_status == "available"
        assert record.storage_backend == "file"
        assert record.storage_key == (
            f"users/{user.id}/tasks/123/outputs/{file_id}/output/report.txt"
        )
        assert record.workspace_relative_path == "output/report.txt"
        assert record.workspace_category == "output"

        object_files = [path for path in object_root.rglob("*") if path.is_file()]
        assert len(object_files) == 1
        assert object_files[0].read_text(encoding="utf-8") == "workspace output"
    finally:
        db.close()
        engine.dispose()


def test_agent_workspace_register_file_uses_explicit_db_task_id(
    monkeypatch, tmp_path, mock_workspace_db
):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="delegated-workspace-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=321, user_id=user.id, title="Parent task")
        db.add(task)
        db.commit()

        workspace = TaskWorkspace(
            id="agent_2_abcd1234",
            base_dir=str(tmp_path / "workspaces"),
            db_task_id=321,
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("delegated output", encoding="utf-8")

        file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        canonical_path = (
            tmp_path
            / "workspaces"
            / f"user_{user.id}"
            / "web_task_321"
            / "output"
            / "report.txt"
        )
        assert workspace.current_task_id == 321
        assert record.user_id == user.id
        assert record.task_id == 321
        assert record.storage_path == str(canonical_path)
        assert canonical_path.read_text(encoding="utf-8") == "delegated output"
        assert record.storage_key == (
            f"users/{user.id}/tasks/321/outputs/{file_id}/output/report.txt"
        )
        assert record.workspace_relative_path == "output/report.txt"
        assert record.workspace_category == "output"
    finally:
        db.close()
        engine.dispose()


def test_agent_workspace_register_file_rebinds_existing_output_to_db_task_id(
    monkeypatch, tmp_path, mock_workspace_db
):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="delegated-rebind-user", password_hash="hash")
        db.add(user)
        db.flush()
        parent_task = Task(id=321, user_id=user.id, title="Parent task")
        worker_task = Task(id=322, user_id=user.id, title="Worker task")
        db.add_all([parent_task, worker_task])
        db.commit()

        workspace = TaskWorkspace(
            id="agent_2_abcd1234",
            base_dir=str(tmp_path / "workspaces"),
            db_task_id=321,
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("worker output", encoding="utf-8")

        from xagent.web.services.uploaded_file_store import UploadedFileStore

        record = UploadedFileStore(db).create_from_local_path(
            local_path=output_path,
            user_id=int(user.id),
            file_id="worker-output",
            task_id=322,
            filename="report.txt",
            workspace_relative_path="output/report.txt",
            workspace_category="output",
            mime_type="text/plain",
        )
        db.commit()

        output_path.write_text("parent-visible output", encoding="utf-8")
        file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        assert file_id == "worker-output"
        db.refresh(record)
        canonical_path = (
            tmp_path
            / "workspaces"
            / f"user_{user.id}"
            / "web_task_321"
            / "output"
            / "report.txt"
        )
        assert record.user_id == user.id
        assert record.task_id == 321
        assert record.storage_path == str(canonical_path)
        assert canonical_path.read_text(encoding="utf-8") == "parent-visible output"
        assert record.storage_key == (
            f"users/{user.id}/tasks/321/outputs/worker-output/output/report.txt"
        )
        assert record.workspace_relative_path == "output/report.txt"
        assert record.workspace_category == "output"
        assert record.file_size == len("parent-visible output")
    finally:
        db.close()
        engine.dispose()


def test_agent_workspace_register_file_avoids_parent_output_name_collision(
    monkeypatch, tmp_path, mock_workspace_db
):
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="delegated-collision-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=321, user_id=user.id, title="Parent task")
        db.add(task)
        db.commit()

        parent_output_path = (
            tmp_path
            / "workspaces"
            / f"user_{user.id}"
            / "web_task_321"
            / "output"
            / "report.txt"
        )
        parent_output_path.parent.mkdir(parents=True)
        parent_output_path.write_text("existing parent output", encoding="utf-8")
        db.add(
            UploadedFile(
                file_id="existing-parent-output",
                user_id=user.id,
                task_id=321,
                filename="report.txt",
                storage_path=str(parent_output_path),
                mime_type="text/plain",
                file_size=len("existing parent output"),
                workspace_relative_path="output/report.txt",
                workspace_category="output",
            )
        )
        db.commit()

        workspace = TaskWorkspace(
            id="agent_2_abcd1234",
            base_dir=str(tmp_path / "workspaces"),
            db_task_id=321,
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("delegated output", encoding="utf-8")

        file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        canonical_path = parent_output_path.with_name("report_1.txt")
        assert (
            parent_output_path.read_text(encoding="utf-8") == "existing parent output"
        )
        assert canonical_path.read_text(encoding="utf-8") == "delegated output"
        assert record.storage_path == str(canonical_path)
        assert record.workspace_relative_path == "output/report_1.txt"
        assert record.storage_key == (
            f"users/{user.id}/tasks/321/outputs/{file_id}/output/report_1.txt"
        )
    finally:
        db.close()
        engine.dispose()


def test_agent_workspace_register_file_is_idempotent_after_canonicalization(
    monkeypatch, tmp_path, mock_workspace_db
):
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="delegated-idempotent-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=321, user_id=user.id, title="Parent task")
        db.add(task)
        db.commit()

        workspace = TaskWorkspace(
            id="agent_2_abcd1234",
            base_dir=str(tmp_path / "workspaces"),
            db_task_id=321,
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("first delegated output", encoding="utf-8")

        file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        output_path.write_text("updated delegated output", encoding="utf-8")
        second_file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        assert second_file_id == file_id
        records = db.query(UploadedFile).all()
        assert len(records) == 1
        record = records[0]
        canonical_path = (
            tmp_path
            / "workspaces"
            / f"user_{user.id}"
            / "web_task_321"
            / "output"
            / "report.txt"
        )
        assert record.storage_path == str(canonical_path)
        assert record.workspace_relative_path == "output/report.txt"
        assert record.file_size == len("updated delegated output")
        assert canonical_path.read_text(encoding="utf-8") == "updated delegated output"

        object_files = [path for path in object_root.rglob("*") if path.is_file()]
        assert len(object_files) == 1
        assert object_files[0].read_text(encoding="utf-8") == "updated delegated output"
    finally:
        db.close()
        engine.dispose()


def test_tool_factory_workspace_preserves_db_task_id(tmp_path):
    workspace = ToolFactory._create_workspace(
        {
            "base_dir": str(tmp_path / "workspaces"),
            "task_id": "agent_2_abcd1234",
            "db_task_id": 654,
        }
    )

    assert workspace is not None
    assert workspace.id == "agent_2_abcd1234"
    assert workspace.db_task_id == 654
    assert workspace.current_task_id == 654


def test_workspace_manager_updates_cached_workspace_db_task_id(tmp_path):
    manager = WorkspaceManager()
    workspace = manager.get_or_create_workspace(
        str(tmp_path / "workspaces"),
        "agent_2_abcd1234",
    )
    assert workspace.db_task_id is None
    assert workspace.current_task_id is None

    same_workspace = manager.get_or_create_workspace(
        str(tmp_path / "workspaces"),
        "agent_2_abcd1234",
        db_task_id=654,
    )

    assert same_workspace is workspace
    assert same_workspace.db_task_id == 654
    assert same_workspace.current_task_id == 654


def test_agent_service_workspace_preserves_config_db_task_id(tmp_path):
    class ToolConfig:
        _workspace_config = {
            "base_dir": str(tmp_path / "workspaces"),
            "task_id": "agent_2_abcd1234",
            "db_task_id": 987,
        }

        def get_allowed_skills(self):
            return None

    service = AgentService(
        name="worker",
        id="agent_2_abcd1234",
        tool_config=ToolConfig(),
        enable_workspace=True,
    )

    assert service.workspace is not None
    assert service.workspace.id == "agent_2_abcd1234"
    assert service.workspace.db_task_id == 987
    assert service.workspace.current_task_id == 987


def test_workspace_register_file_uses_uploaded_file_store_create(
    monkeypatch, tmp_path, mock_workspace_db
):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    create_calls = []

    from xagent.web.services.uploaded_file_store import UploadedFileStore

    original_create = UploadedFileStore.create_from_local_path

    def create_spy(self, **kwargs):
        create_calls.append(
            {
                "local_path": kwargs["local_path"],
                "user_id": kwargs["user_id"],
                "file_id": kwargs["file_id"],
                "task_id": kwargs["task_id"],
                "filename": kwargs["filename"],
                "storage_key": kwargs["storage_key"],
                "workspace_relative_path": kwargs["workspace_relative_path"],
                "workspace_category": kwargs["workspace_category"],
                "mime_type": kwargs["mime_type"],
            }
        )
        return original_create(self, **kwargs)

    monkeypatch.setattr(UploadedFileStore, "create_from_local_path", create_spy)

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="workspace-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=456, user_id=user.id, title="Workspace task")
        db.add(task)
        db.commit()

        workspace = TaskWorkspace(
            id="web_task_456", base_dir=str(tmp_path / "workspaces")
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("workspace output", encoding="utf-8")

        file_id = workspace.register_file(str(output_path), db_session=db)

        assert create_calls == [
            {
                "local_path": output_path,
                "user_id": user.id,
                "file_id": file_id,
                "task_id": 456,
                "filename": "report.txt",
                "storage_key": (
                    f"users/{user.id}/tasks/456/outputs/{file_id}/output/report.txt"
                ),
                "workspace_relative_path": "output/report.txt",
                "workspace_category": "output",
                "mime_type": "text/plain",
            }
        ]
    finally:
        db.close()
        engine.dispose()


def test_workspace_register_file_resyncs_existing_modified_file(
    monkeypatch, tmp_path, mock_workspace_db
):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="workspace-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=654, user_id=user.id, title="Workspace task")
        db.add(task)
        db.commit()

        workspace = TaskWorkspace(
            id="web_task_654", base_dir=str(tmp_path / "workspaces")
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("old", encoding="utf-8")

        file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        output_path.write_text("new content", encoding="utf-8")
        second_file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        assert second_file_id == file_id
        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        assert record.file_size == len("new content")
        assert record.storage_status == "available"

        object_files = [path for path in object_root.rglob("*") if path.is_file()]
        assert len(object_files) == 1
        assert object_files[0].read_text(encoding="utf-8") == "new content"
    finally:
        db.close()
        engine.dispose()


def test_auto_register_files_resyncs_modified_existing_file(
    monkeypatch, tmp_path, mock_workspace_db
):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="workspace-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=655, user_id=user.id, title="Workspace task")
        db.add(task)
        db.commit()

        workspace = TaskWorkspace(
            id="web_task_655", base_dir=str(tmp_path / "workspaces")
        )
        output_path = workspace.output_dir / "report.txt"
        output_path.write_text("old", encoding="utf-8")
        file_id = workspace.register_file(str(output_path), db_session=db)
        db.commit()

        workspace.db_session = db
        with workspace.auto_register_files():
            output_path.write_text("new content", encoding="utf-8")
        db.commit()

        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        assert record.file_size == len("new content")
        object_files = [path for path in object_root.rglob("*") if path.is_file()]
        assert len(object_files) == 1
        assert object_files[0].read_text(encoding="utf-8") == "new content"
    finally:
        db.close()
        engine.dispose()


def test_workspace_register_file_resyncs_external_file_without_reclassifying_upload(
    monkeypatch, tmp_path, mock_workspace_db
):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="workspace-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=656, user_id=user.id, title="Workspace task")
        db.add(task)
        db.commit()

        external_dir = tmp_path / "external-uploads"
        external_dir.mkdir()
        external_path = external_dir / "source.txt"
        external_path.write_text("old upload", encoding="utf-8")

        from xagent.web.services.uploaded_file_store import UploadedFileStore

        record = UploadedFileStore(db).create_from_local_path(
            local_path=external_path,
            user_id=int(user.id),
            task_id=int(task.id),
            filename="source.txt",
            mime_type="text/plain",
        )
        file_id = str(record.file_id)
        original_storage_key = str(record.storage_key)
        db.commit()

        workspace = TaskWorkspace(
            id="web_task_656",
            base_dir=str(tmp_path / "workspaces"),
            allowed_external_dirs=[str(external_dir)],
        )

        external_path.write_text("new upload", encoding="utf-8")
        second_file_id = workspace.register_file(
            str(external_path), file_id=file_id, db_session=db
        )
        db.commit()

        assert second_file_id == file_id
        db.refresh(record)
        assert record.storage_key == original_storage_key
        assert record.workspace_relative_path is None
        assert record.workspace_category is None
        assert record.file_size == len("new upload")

        object_files = [path for path in object_root.rglob("*") if path.is_file()]
        assert len(object_files) == 1
        assert object_files[0].read_text(encoding="utf-8") == "new upload"
    finally:
        db.close()
        engine.dispose()


def test_list_all_user_files_includes_durable_only_uploads(tmp_path, mock_workspace_db):
    # Override the global autouse fixture from tests/conftest.py for this module.
    del mock_workspace_db

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="workspace-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=789, user_id=user.id, title="Workspace task")
        db.add(task)
        db.commit()

        missing_local_path = tmp_path / "uploads" / "durable-only.txt"
        assert not missing_local_path.exists()
        file_record = UploadedFile(
            user_id=user.id,
            task_id=task.id,
            filename="durable-only.txt",
            storage_path=str(missing_local_path),
            storage_backend="s3",
            storage_key=f"users/{user.id}/uploads/file-1/durable-only.txt",
            storage_status="available",
            mime_type="text/plain",
            file_size=12,
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)

        workspace = TaskWorkspace(
            id="web_task_789",
            base_dir=str(tmp_path / "workspaces"),
        )
        workspace.db_session = db

        result = workspace.list_all_user_files(include_workspace_files=False)

        assert result["success"] is True
        assert [file_info["file_id"] for file_info in result["files"]] == [
            file_record.file_id
        ]
        assert result["files"][0]["filename"] == "durable-only.txt"
        assert result["files"][0]["in_current_workspace"] is False
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def mock_workspace_db():
    yield
