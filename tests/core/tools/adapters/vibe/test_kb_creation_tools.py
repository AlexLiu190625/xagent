import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from xagent.config import WEB_CRAWL_TLS_IMPERSONATE
from xagent.core.tools.adapters.vibe.agent_kb_service import (
    AgentKnowledgeBaseError,
    AgentKnowledgeBaseService,
)
from xagent.core.tools.adapters.vibe.file_ingestion_tool import (
    CreateKnowledgeBaseFromFileTool,
    UploadedFileSnapshot,
)
from xagent.core.tools.adapters.vibe.web_ingestion_tool import (
    CreateKnowledgeBaseFromUrlTool,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_MODEL_ID,
    CollectionInfo,
    IngestionConfig,
    IngestionResult,
    IngestionStepResult,
    WebIngestionResult,
)
from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.operation_compatibility import (
    KBOperationCompatibilityFacade,
    RollbackStatus,
    SideEffectPlane,
)


class _FakeMetadataStore:
    def __init__(self, collection: CollectionInfo | None) -> None:
        self.collection = collection
        self.saved_configs: list[dict[str, object]] = []
        self.saved_collections: list[CollectionInfo] = []

    async def save_collection_config(
        self,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        self.saved_configs.append(
            {
                "collection": collection,
                "config_json": config_json,
                "user_id": user_id,
            }
        )

    async def get_collection(self, collection: str) -> CollectionInfo:
        if self.collection is None or self.collection.name != collection:
            raise ValueError(f"Collection {collection!r} not found")
        return self.collection

    async def save_collection(self, collection: CollectionInfo) -> None:
        self.saved_collections.append(collection)
        self.collection = collection


class _FakeStorageShim:
    def __init__(self, metadata_store: _FakeMetadataStore) -> None:
        self.metadata_store = metadata_store

    def get_metadata_store(self) -> _FakeMetadataStore:
        return self.metadata_store

    def reset_kb_write_coordinator(self) -> None:
        return None

    def reset_rag_storage_for_tests(self) -> None:
        return None


class _RecordingOperationFacade(KBOperationCompatibilityFacade):
    def __init__(self) -> None:
        super().__init__()
        self.outcomes = []

    @contextmanager
    def start_operation(self, **kwargs):
        with super().start_operation(**kwargs) as operation:
            yield operation
        if operation.outcome is not None:
            self.outcomes.append(operation.outcome)


def _ingestion_step(name: str, **metadata: object) -> IngestionStepResult:
    return IngestionStepResult(name=name, metadata=dict(metadata))


def _successful_ingestion_result(doc_id: str = "doc-ok") -> IngestionResult:
    return IngestionResult(
        status="success",
        doc_id=doc_id,
        parse_hash="parse-ok",
        chunk_count=1,
        embedding_count=1,
        vector_count=1,
        completed_steps=[
            _ingestion_step("initialize_collection", embedding_model_id="model-a"),
            _ingestion_step("register_document", doc_id=doc_id, created=True),
            _ingestion_step("parse_document", parse_hash="parse-ok", written=True),
            _ingestion_step("chunk_document", chunk_count=1, created=True),
            _ingestion_step("write_vectors_to_db", vector_count=1),
        ],
        message="ok",
    )


def _fake_db_generator(db):
    try:
        yield db
    finally:
        db.close()


@pytest.mark.asyncio
async def test_agent_kb_service_prepare_collection_persists_config_and_sanitizes():
    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.get_collection = AsyncMock(
        side_effect=ValueError("Collection 'agent url kb' not found")
    )
    metadata_store.save_collection = AsyncMock()
    service = AgentKnowledgeBaseService(user_id=71, is_admin=False)
    ingest_config = IngestionConfig(embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID)

    with patch(
        "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
        return_value=metadata_store,
    ):
        collection_name = await service.prepare_collection(
            "  agent url kb  ", ingest_config
        )

    assert collection_name == "agent url kb"
    metadata_store.save_collection_config.assert_awaited_once()
    _, save_kwargs = metadata_store.save_collection_config.await_args
    assert save_kwargs["collection"] == "agent url kb"
    assert save_kwargs["user_id"] == 71
    assert json.loads(save_kwargs["config_json"]) == {
        "embedding_model_id": DEFAULT_EMBEDDING_MODEL_ID
    }
    metadata_store.save_collection.assert_awaited_once()
    saved_collection = metadata_store.save_collection.await_args.args[0]
    assert saved_collection.name == "agent url kb"
    assert saved_collection.extra_metadata["kb_storage"] == {"backend": "lancedb"}


@pytest.mark.asyncio
async def test_agent_kb_service_prepare_collection_preserves_existing_backend_binding():
    existing = CollectionInfo(
        name="agent url kb",
        extra_metadata={"kb_storage": {"backend": "postgresql"}, "other": "kept"},
    )
    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.get_collection = AsyncMock(return_value=existing)
    metadata_store.save_collection = AsyncMock()
    service = AgentKnowledgeBaseService(user_id=71, is_admin=False)
    ingest_config = IngestionConfig(embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID)

    with patch(
        "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
        return_value=metadata_store,
    ):
        collection_name = await service.prepare_collection(
            "agent url kb", ingest_config
        )

    assert collection_name == "agent url kb"
    metadata_store.save_collection_config.assert_awaited_once()
    metadata_store.save_collection.assert_not_awaited()
    assert existing.extra_metadata["kb_storage"] == {"backend": "postgresql"}
    assert existing.extra_metadata["other"] == "kept"


@pytest.mark.asyncio
async def test_agent_kb_service_refresh_collection_metadata_forces_realtime_for_admin():
    refresh_metadata = AsyncMock()
    service = AgentKnowledgeBaseService(user_id=71, is_admin=True)

    with patch(
        "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
        new=refresh_metadata,
    ):
        await service.refresh_collection_metadata("agent_url_kb")

    refresh_metadata.assert_awaited_once_with(
        user_id=71,
        is_admin=True,
        force_realtime=True,
    )


@pytest.mark.asyncio
async def test_agent_kb_service_refresh_collection_metadata_skips_non_admin_refresh():
    refresh_metadata = AsyncMock()
    service = AgentKnowledgeBaseService(user_id=71, is_admin=False)

    with patch(
        "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
        new=refresh_metadata,
    ):
        await service.refresh_collection_metadata("agent_url_kb")

    refresh_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_kb_service_prepare_collection_raises_on_config_save_failure():
    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock(
        side_effect=RuntimeError("config save failed")
    )
    service = AgentKnowledgeBaseService(user_id=71, is_admin=False)
    ingest_config = IngestionConfig(embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID)

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        pytest.raises(
            AgentKnowledgeBaseError, match="Failed to save collection config"
        ),
    ):
        await service.prepare_collection("agent kb", ingest_config)


@pytest.mark.asyncio
async def test_agent_kb_service_refresh_collection_metadata_raises_on_failure():
    refresh_metadata = AsyncMock(side_effect=RuntimeError("refresh failed"))
    service = AgentKnowledgeBaseService(user_id=71, is_admin=True)

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
            new=refresh_metadata,
        ),
        pytest.raises(
            AgentKnowledgeBaseError,
            match="Failed to refresh knowledge base metadata",
        ),
    ):
        await service.refresh_collection_metadata("agent_url_kb")


@pytest.mark.asyncio
async def test_create_kb_from_url_uses_shared_service(monkeypatch):
    monkeypatch.setenv(WEB_CRAWL_TLS_IMPERSONATE, "auto")

    ingest_result = WebIngestionResult(
        status="success",
        collection="agent_url_kb",
        total_urls_found=1,
        pages_crawled=1,
        pages_failed=0,
        documents_created=1,
        chunks_created=3,
        embeddings_created=3,
        crawled_urls=["https://example.com"],
        failed_urls={},
        message="ok",
        warnings=[],
        elapsed_time_ms=123,
    )
    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_url_kb")
    service.refresh_collection_metadata = AsyncMock()
    run_web_ingestion_mock = AsyncMock(return_value=ingest_result)

    with (
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_web_ingestion",
            new=run_web_ingestion_mock,
        ),
    ):
        tool = CreateKnowledgeBaseFromUrlTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"url": "https://example.com", "collection_name": "agent_url_kb"}
        )

    assert result["success"] is True
    service.prepare_collection.assert_awaited_once()
    _, prepare_kwargs = service.prepare_collection.await_args
    assert prepare_kwargs["collection_name"] == "agent_url_kb"
    assert (
        prepare_kwargs["ingestion_config"].embedding_model_id
        == DEFAULT_EMBEDDING_MODEL_ID
    )
    service.refresh_collection_metadata.assert_awaited_once_with("agent_url_kb")
    run_web_ingestion_mock.assert_awaited_once()
    _, run_kwargs = run_web_ingestion_mock.await_args
    assert run_kwargs["crawl_config"].tls_impersonate == "auto"


@pytest.mark.asyncio
async def test_create_kb_from_url_returns_error_when_shared_service_fails():
    service = MagicMock()
    service.prepare_collection = AsyncMock(
        side_effect=AgentKnowledgeBaseError("config save failed")
    )
    service.refresh_collection_metadata = AsyncMock()

    with patch(
        "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
        return_value=service,
    ):
        tool = CreateKnowledgeBaseFromUrlTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"url": "https://example.com", "collection_name": "agent_url_kb"}
        )

    assert result["success"] is False
    assert result["message"] == "config save failed"


@pytest.mark.asyncio
async def test_create_kb_from_url_rejects_invalid_start_url():
    service = MagicMock()
    service.prepare_collection = AsyncMock()
    service.refresh_collection_metadata = AsyncMock()

    with patch(
        "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
        return_value=service,
    ):
        tool = CreateKnowledgeBaseFromUrlTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"url": "www.example.com", "collection_name": "agent_url_kb"}
        )

    assert result["success"] is False
    assert (
        result["message"]
        == "Invalid start_url: URL must start with http:// or https://"
    )
    service.prepare_collection.assert_not_awaited()
    service.refresh_collection_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_kb_from_file_uses_shared_service(tmp_path):
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello", encoding="utf-8")
    file_record = SimpleNamespace(
        user_id=7,
        filename="notes.txt",
        storage_path=str(source_file),
        file_id="file-1",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [file_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield from _fake_db_generator(db)

    ingest_result = IngestionResult(
        status="success",
        doc_id="doc-1",
        parse_hash="parse-1",
        chunk_count=2,
        embedding_count=2,
        vector_count=2,
        completed_steps=[],
        failed_step=None,
        message="ok",
        warnings=[],
        file_id="file-1",
    )
    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_file_kb")
    service.refresh_collection_metadata = AsyncMock()

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.document_ingestion.run_document_ingestion",
            new=Mock(return_value=ingest_result),
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-1"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is True
    service.prepare_collection.assert_awaited_once()
    _, prepare_kwargs = service.prepare_collection.await_args
    assert prepare_kwargs["collection_name"] == "agent_file_kb"
    assert (
        prepare_kwargs["ingestion_config"].embedding_model_id
        == DEFAULT_EMBEDDING_MODEL_ID
    )
    service.refresh_collection_metadata.assert_awaited_once_with("agent_file_kb")
    db.close.assert_called_once()


@pytest.mark.asyncio
async def test_create_kb_from_file_continues_after_unexpected_ingest_error(tmp_path):
    bad_file = tmp_path / "bad.txt"
    good_file = tmp_path / "good.txt"
    bad_file.write_text("bad", encoding="utf-8")
    good_file.write_text("good", encoding="utf-8")
    bad_record = SimpleNamespace(
        user_id=7,
        filename="bad.txt",
        storage_path=str(bad_file),
        file_id="file-bad",
    )
    good_record = SimpleNamespace(
        user_id=7,
        filename="good.txt",
        storage_path=str(good_file),
        file_id="file-good",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [bad_record, good_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield from _fake_db_generator(db)

    def fake_run_ingestion(*, source_path: str, file_id: str, **_: object):
        if source_path == str(bad_file):
            raise RuntimeError("parser exploded")
        return IngestionResult(
            status="success",
            doc_id=f"doc-{file_id}",
            parse_hash="parse-good",
            chunk_count=2,
            embedding_count=2,
            vector_count=2,
            completed_steps=[],
            failed_step=None,
            message="ok",
            warnings=[],
            file_id=file_id,
        )

    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_file_kb")
    service.refresh_collection_metadata = AsyncMock()
    run_ingestion = Mock(side_effect=fake_run_ingestion)

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.document_ingestion.run_document_ingestion",
            new=run_ingestion,
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-bad", "file-good"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is True
    assert result["collection_name"] == "agent_file_kb"
    assert result["files_ingested"] == 1
    assert (
        "Failed to ingest bad.txt due to unexpected error: parser exploded"
        in result["message"]
    )
    service.refresh_collection_metadata.assert_awaited_once_with("agent_file_kb")
    assert run_ingestion.call_count == 2
    db.close.assert_called_once()


@pytest.mark.asyncio
async def test_create_kb_from_file_failed_ingest_records_operation_outcome(
    tmp_path,
    monkeypatch,
):
    from xagent.core.tools.core.RAG_tools import kb as kb_module
    from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion

    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello", encoding="utf-8")
    file_record = SimpleNamespace(
        user_id=7,
        filename="notes.txt",
        storage_path=str(source_file),
        file_id="file-1",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [file_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield from _fake_db_generator(db)

    def fake_run_document_ingestion_impl(**_: object) -> IngestionResult:
        return IngestionResult(
            status="error",
            doc_id="doc-failed",
            parse_hash="parse-failed",
            chunk_count=2,
            embedding_count=0,
            vector_count=0,
            completed_steps=[
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-failed", created=True),
                _ingestion_step(
                    "parse_document",
                    parse_hash="parse-failed",
                    written=True,
                ),
                _ingestion_step("chunk_document", chunk_count=2, created=True),
            ],
            failed_step="write_vectors_to_db",
            message="embedding failed",
            file_id="file-1",
        )

    metadata_store = _FakeMetadataStore(CollectionInfo(name="agent_file_kb"))
    operation_facade = _RecordingOperationFacade()
    coordinator = KBCoordinator(
        storage_shim=_FakeStorageShim(metadata_store),
        operation_compatibility=operation_facade,
    )

    monkeypatch.setattr(kb_module, "get_kb_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        document_ingestion,
        "_run_document_ingestion_impl",
        fake_run_document_ingestion_impl,
    )

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.web.services.managed_file_ref.ensure_uploaded_file_local_path",
            return_value=source_file,
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-1"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is False
    assert result["collection_name"] == "agent_file_kb"
    assert "embedding failed" in result["message"]
    assert metadata_store.saved_configs[-1]["collection"] == "agent_file_kb"
    assert metadata_store.saved_configs[-1]["user_id"] == 71
    assert metadata_store.saved_collections
    assert metadata_store.saved_collections[-1].owners == []
    assert metadata_store.saved_collections[-1].extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }

    assert len(operation_facade.outcomes) == 1
    outcome = operation_facade.outcomes[0]
    assert outcome.operation_type == "document_ingestion"
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert {step.plane for step in outcome.compensation_steps} == {
        SideEffectPlane.COLLECTION,
        SideEffectPlane.DOCUMENT,
        SideEffectPlane.STATUS,
        SideEffectPlane.PARSE,
        SideEffectPlane.CHUNK,
    }
    db.close.assert_called_once()


@pytest.mark.asyncio
async def test_create_kb_from_file_preserves_storage_context_in_executor(
    tmp_path,
    monkeypatch,
):
    from xagent.core.tools.core.RAG_tools import kb as kb_module
    from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_bound_storage_shim_for_current_context,
    )

    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello", encoding="utf-8")
    file_record = SimpleNamespace(
        user_id=7,
        filename="notes.txt",
        storage_path=str(source_file),
        file_id="file-1",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [file_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield from _fake_db_generator(db)

    metadata_store = _FakeMetadataStore(CollectionInfo(name="agent_file_kb"))
    coordinator = KBCoordinator(
        storage_shim=_FakeStorageShim(metadata_store),
        operation_compatibility=_RecordingOperationFacade(),
    )

    def fake_run_document_ingestion_impl(**_: object) -> IngestionResult:
        assert get_bound_storage_shim_for_current_context() is coordinator.storage_shim
        return _successful_ingestion_result("doc-1")

    monkeypatch.setattr(kb_module, "get_kb_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        document_ingestion,
        "_run_document_ingestion_impl",
        fake_run_document_ingestion_impl,
    )

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.web.services.managed_file_ref.ensure_uploaded_file_local_path",
            return_value=source_file,
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-1"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is True
    assert result["files_ingested"] == 1
    db.close.assert_called_once()


@pytest.mark.asyncio
async def test_create_kb_from_file_restores_durable_only_upload_before_ingestion(
    tmp_path,
):
    missing_source = tmp_path / "missing-notes.txt"
    restored_source = tmp_path / "restored-notes.txt"
    restored_source.write_text("restored", encoding="utf-8")
    file_record = SimpleNamespace(
        user_id=7,
        filename="notes.txt",
        storage_path=str(missing_source),
        file_id="file-1",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [file_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield from _fake_db_generator(db)

    def fake_ensure_local(record):
        db.close.assert_called_once()
        assert isinstance(record, UploadedFileSnapshot)
        assert record.filename == file_record.filename
        assert record.storage_path == file_record.storage_path
        assert record.file_id == file_record.file_id
        return restored_source

    ingest_result = IngestionResult(
        status="success",
        doc_id="doc-1",
        parse_hash="parse-1",
        chunk_count=2,
        embedding_count=2,
        vector_count=2,
        completed_steps=[],
        failed_step=None,
        message="ok",
        warnings=[],
        file_id="file-1",
    )
    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_file_kb")
    service.refresh_collection_metadata = AsyncMock()
    run_ingestion = Mock(return_value=ingest_result)

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.web.services.managed_file_ref.ensure_uploaded_file_local_path",
            side_effect=fake_ensure_local,
        ) as ensure_local,
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.document_ingestion.run_document_ingestion",
            new=run_ingestion,
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-1"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is True
    ensure_local.assert_called_once()
    _, ingestion_kwargs = run_ingestion.call_args
    assert ingestion_kwargs["source_path"] == str(restored_source)
    db.close.assert_called_once()


@pytest.mark.asyncio
async def test_create_kb_from_file_returns_error_when_metadata_refresh_fails(tmp_path):
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello", encoding="utf-8")
    file_record = SimpleNamespace(
        user_id=7,
        filename="notes.txt",
        storage_path=str(source_file),
        file_id="file-1",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [file_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield from _fake_db_generator(db)

    ingest_result = IngestionResult(
        status="success",
        doc_id="doc-1",
        parse_hash="parse-1",
        chunk_count=2,
        embedding_count=2,
        vector_count=2,
        completed_steps=[],
        failed_step=None,
        message="ok",
        warnings=[],
        file_id="file-1",
    )
    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_file_kb")
    service.refresh_collection_metadata = AsyncMock(
        side_effect=AgentKnowledgeBaseError("metadata refresh failed")
    )

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.document_ingestion.run_document_ingestion",
            new=Mock(return_value=ingest_result),
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-1"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is False
    assert result["message"] == "metadata refresh failed"
    db.close.assert_called_once()


@pytest.mark.asyncio
async def test_create_kb_from_url_partial_failure_preserves_pipeline_policy(
    monkeypatch,
):
    from xagent.core.tools.core.RAG_tools import kb as kb_module
    from xagent.core.tools.core.RAG_tools.pipelines import (
        document_ingestion,
        web_ingestion,
    )

    def fake_run_document_ingestion_impl(
        source_path: str,
        **_: object,
    ) -> IngestionResult:
        if source_path.endswith("ok.md"):
            return _successful_ingestion_result("doc-ok")
        return IngestionResult(
            status="partial",
            doc_id="doc-bad",
            parse_hash=None,
            chunk_count=0,
            embedding_count=0,
            vector_count=0,
            completed_steps=[
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-bad", created=True),
            ],
            failed_step="parse_document",
            message="parse failed",
        )

    async def fake_run_web_ingestion_impl(
        collection: str,
        crawl_config,
        *,
        pipeline_facade,
        **_: object,
    ) -> WebIngestionResult:
        pipeline_facade.run_document_ingestion(collection, "/tmp/ok.md")
        pipeline_facade.run_document_ingestion(collection, "/tmp/bad.md")
        return WebIngestionResult(
            status="partial",
            collection=collection,
            total_urls_found=2,
            pages_crawled=2,
            pages_failed=1,
            documents_created=1,
            chunks_created=1,
            embeddings_created=1,
            crawled_urls=[crawl_config.start_url, "https://example.com/bad"],
            failed_urls={"https://example.com/bad": "parse failed"},
            message="partial web ingest",
            warnings=["parse failed"],
            elapsed_time_ms=1,
        )

    metadata_store = _FakeMetadataStore(CollectionInfo(name="agent_url_kb"))
    operation_facade = _RecordingOperationFacade()
    coordinator = KBCoordinator(
        storage_shim=_FakeStorageShim(metadata_store),
        operation_compatibility=operation_facade,
    )

    monkeypatch.setattr(kb_module, "get_kb_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        document_ingestion,
        "_run_document_ingestion_impl",
        fake_run_document_ingestion_impl,
    )
    monkeypatch.setattr(
        web_ingestion,
        "_run_web_ingestion_impl",
        fake_run_web_ingestion_impl,
    )

    tool = CreateKnowledgeBaseFromUrlTool(user_id=71, is_admin=False)
    result = await tool.run_json_async(
        {"url": "https://example.com", "collection_name": "agent_url_kb"}
    )

    assert result == {
        "success": True,
        "collection_name": "agent_url_kb",
        "message": "Successfully imported website https://example.com into knowledge base 'agent_url_kb'",
        "pages_crawled": 2,
    }
    assert metadata_store.saved_configs[-1]["collection"] == "agent_url_kb"
    assert metadata_store.saved_configs[-1]["user_id"] == 71
    assert metadata_store.saved_collections
    assert metadata_store.saved_collections[-1].owners == []
    assert metadata_store.saved_collections[-1].extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }

    root_outcome = operation_facade.outcomes[-1]
    assert root_outcome.operation_type == "web_ingestion"
    assert root_outcome.status == "partial"
    assert root_outcome.rollback_status is RollbackStatus.SKIPPED_BY_POLICY
    assert root_outcome.side_effects_may_remain is True
    assert [child.status for child in root_outcome.child_outcomes] == [
        "success",
        "partial",
    ]
    assert root_outcome.details["documents_created"] == 1
    assert root_outcome.details["pages_failed"] == 1
