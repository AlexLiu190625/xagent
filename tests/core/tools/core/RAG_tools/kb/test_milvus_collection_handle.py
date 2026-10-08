"""``MilvusCollectionHandle`` skeleton: dispatch, ledger delegation, refusals,
capabilities, and the LanceDB-only paths a Milvus deployment skips.

Milvus rows are not written, searched or deleted yet, so nothing here connects
to Milvus.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.core.RAG_tools import kb
from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    RegisterDocumentRequest,
)
from xagent.core.tools.core.RAG_tools.kb import collection_handle
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    LanceDBCollectionHandle,
    MilvusCollectionHandle,
    ledger_holds_vectors,
)
from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBContextRequest,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_name,
)
from xagent.core.tools.core.RAG_tools.management import (
    collection_manager,
    collections,
)
from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
    _INGEST_TABLES,
    _compact_storage_if_needed,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.utils import migration_utils
from xagent.core.tools.core.RAG_tools.version_management import cascade_cleaner
from xagent.providers.vector_store.milvus import MilvusConnectionManager
from xagent.web.services import kb_file_service

ASYNC_SEARCH = "search_dense_async search_sparse_async search_hybrid_async"
CASCADES = """cleanup_cascade cleanup_document_cascade cleanup_parse_cascade
    cleanup_chunk_cascade cleanup_embed_cascade"""
VERSIONS = """list_candidates promote_version_main capture_candidate_cleanup_snapshot
    restore_candidate_cleanup_snapshot"""
FAMILIES = {
    "supports_documents": """register_document load_document list_documents
        delete_document_record snapshot_document restore_document
        delete_created_document""",
    "supports_parses": """parse_exists read_parse_paragraphs write_parse
        read_latest_parse_record read_parse_paragraph_dicts delete_parse_records
        snapshot_parse restore_parse delete_created_parse""",
    "supports_chunks": """chunk_exists read_existing_chunks write_chunks
        delete_chunk_records snapshot_chunks restore_chunks delete_created_chunks""",
    "supports_embeddings": """read_chunks_needing_embedding write_embeddings
        delete_embedding_records snapshot_embeddings restore_embeddings
        delete_created_embeddings cleanup_embeddings_for_operation""",
    "supports_search": "validate_query_vector search_dense search_sparse search_hybrid",
    "supports_versions": f"{VERSIONS} {CASCADES}",
    "supports_async_search": ASYNC_SEARCH,
}
LEDGER = set(
    f"""{FAMILIES["supports_documents"]} {FAMILIES["supports_parses"]}
    {FAMILIES["supports_chunks"]} rename_collection_data rename_collection_status
    rename_collection_metadata delete_collection_config count_documents
    list_collection_documents write_ingestion_status load_ingestion_status
    clear_ingestion_status write_ingestion_status_async load_ingestion_status_async
    clear_ingestion_status_async get_main_pointer set_main_pointer list_main_pointers
    delete_main_pointer capture_status_snapshot restore_status_snapshot
    clear_status_snapshot capture_main_pointer_snapshot
    restore_main_pointer_snapshot""".split()
)
UNSUPPORTED = {
    name: family
    for family, names in (
        ("async search", ASYNC_SEARCH),
        ("cascade cleanup", CASCADES),
        ("version candidates and promotion", VERSIONS),
    )
    for name in names.split()
}
PENDING = set(
    f"""{FAMILIES["supports_embeddings"]} {FAMILIES["supports_search"]}
    capture_document_rows restore_document_rows delete_documents_data
    delete_collection_data cleanup_collection_data_after_rollback collection_stats
    count_rows_by_document""".split()
)


@pytest.fixture
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for step 8 (#2870), which lets the milvus setting start."""
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
    monkeypatch.setattr(
        collection_handle, "require_implemented_vector_backend", lambda _: None
    )


def _context(backend: KBStorageBackend) -> KBCollectionContext:
    return KBCollectionContext(
        collection="kb",
        user_scope=KBUserScope(user_id=1, is_admin=False),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=get_main_pointer_store(),
        backend=backend,
        capabilities=KBCoordinator._capabilities_for_backend(backend),
    )


def _handle() -> tuple[MilvusCollectionHandle, MagicMock, MagicMock]:
    ledger = MagicMock(spec=KBCollectionHandle)
    connections = MagicMock(spec=MilvusConnectionManager)
    handle = MilvusCollectionHandle(
        _context(KBStorageBackend.MILVUS), ledger=ledger, connections=connections
    )
    return handle, ledger, connections


async def _call(handle: MilvusCollectionHandle, name: str) -> Any:
    result = getattr(handle, name)("arg", key="value")
    return await result if inspect.isawaitable(result) else result


def test_every_interface_method_is_delegated_refused_or_pending() -> None:
    interface = {
        name
        for name, member in vars(KBCollectionHandle).items()
        if inspect.isfunction(member)
    }

    assert (len(LEDGER), len(UNSUPPORTED), len(PENDING)) == (44, 12, 18)
    assert LEDGER | set(UNSUPPORTED) | PENDING == interface
    for name in interface:
        routed = getattr(MilvusCollectionHandle, name)
        declared = getattr(KBCollectionHandle, name)
        assert inspect.iscoroutinefunction(routed) == (
            inspect.iscoroutinefunction(declared)
        ), name
        assert routed.__qualname__ == f"MilvusCollectionHandle.{name}"
        assert routed.__doc__ == declared.__doc__, name


@pytest.mark.parametrize("name", sorted(LEDGER))
async def test_ledger_methods_go_to_the_injected_ledger_handle(name: str) -> None:
    handle, ledger, connections = _handle()

    assert await _call(handle, name) is getattr(ledger, name).return_value
    getattr(ledger, name).assert_called_once_with("arg", key="value")
    assert connections.mock_calls == []


@pytest.mark.parametrize(
    ("name", "error", "message"),
    [
        *(
            (name, ConfigurationError, f"not support {family} \\({name}\\)")
            for name, family in sorted(UNSUPPORTED.items())
        ),
        *(
            (name, NotImplementedError, f"{name} needs Milvus rows and is not")
            for name in sorted(PENDING)
        ),
    ],
)
async def test_other_methods_raise_before_reaching_any_store(
    name: str, error: type[Exception], message: str
) -> None:
    handle, ledger, connections = _handle()

    with pytest.raises(error, match=message):
        await _call(handle, name)
    assert ledger.mock_calls == connections.mock_calls == []


def test_capabilities_report_exactly_what_the_handle_serves() -> None:
    capabilities = KBBackendCapabilities.milvus()

    assert {field.name for field in fields(capabilities)} == {
        *FAMILIES,
        "supports_raw_connection",
    }
    for flag, names in FAMILIES.items():
        assert getattr(capabilities, flag) == (set(names.split()) <= LEDGER), flag
    assert capabilities.supports_raw_connection is False
    assert KBCoordinator._capabilities_for_backend(KBStorageBackend.MILVUS) == (
        capabilities
    )


def test_the_client_comes_from_the_connection_manager_on_first_use() -> None:
    handle, _ledger, connections = _handle()
    connections.get_client_from_env.assert_not_called()

    assert handle.client is connections.get_client_from_env.return_value
    assert handle.client is connections.get_client_from_env.return_value
    connections.get_client_from_env.assert_called_once_with()


def test_the_provider_dispatches_on_the_engine(milvus_deployment: None) -> None:
    context = _context(KBStorageBackend.MILVUS)

    handle = KBHandleProvider().open(context)

    assert isinstance(handle, MilvusCollectionHandle)
    assert handle.context is context
    assert type(handle.ledger) is LanceDBCollectionHandle
    assert handle.ledger.context is context
    assert type(handle.connections) is MilvusConnectionManager
    lancedb = KBHandleProvider().open(_context(KBStorageBackend.LANCEDB))
    assert type(lancedb) is LanceDBCollectionHandle
    with pytest.raises(ValueError, match="'qdrant' is not supported"):
        KBHandleProvider().open(_context(KBStorageBackend.QDRANT))


def test_a_milvus_deployment_opens_milvus_handles(milvus_deployment: None) -> None:
    handle = kb.get_kb_coordinator().open_collection_sync(
        KBContextRequest(collection="new", user_id=1, hide_missing=True)
    )

    assert isinstance(handle, MilvusCollectionHandle)
    assert handle.context.capabilities == KBBackendCapabilities.milvus()
    handle.write_ingestion_status("doc", status="running", user_id=1)
    assert [row["doc_id"] for row in handle.load_ingestion_status(user_id=1)] == ["doc"]


def test_a_lancedb_deployment_refuses_a_milvus_binding(tmp_path: Path) -> None:
    binding = {"kb_storage": {"backend": "milvus"}}
    asyncio.run(
        get_metadata_store().save_collection(
            CollectionInfo(name="kb", extra_metadata=binding)
        )
    )
    source = tmp_path / "a.txt"
    source.write_text("kiwi", encoding="utf-8")
    coordinator = kb.get_kb_coordinator()

    mismatch = "bound to the milvus engine, but this deployment runs lancedb"
    with pytest.raises(ValueError, match=mismatch):
        coordinator.register_document_sync(
            RegisterDocumentRequest(
                collection="kb", source_path=str(source), doc_id="d", user_id=1
            )
        )
    with pytest.raises(ValueError, match=mismatch):
        coordinator.write_ingestion_status_sync("kb", "d", status="running", user_id=1)
    ledger = KBHandleProvider().open(_context(KBStorageBackend.LANCEDB))
    assert ledger.count_documents(None, True) == 0
    assert ledger.load_ingestion_status(is_admin=True) == []


def test_batched_stats_are_pending_on_milvus_and_refused_elsewhere(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(NotImplementedError, match="#2867"):
        KBHandleProvider().aggregate_collection_stats(user_id=None, is_admin=True)
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "qdrant")
    with pytest.raises(ValueError, match="'qdrant' is not supported"):
        KBHandleProvider().aggregate_collection_stats(user_id=None, is_admin=True)


def test_lancedb_only_paths_follow_the_deployment_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ledger_holds_vectors()
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
    with pytest.raises(ConfigurationError, match="not implemented"):
        ledger_holds_vectors()
    monkeypatch.setattr(
        collection_handle, "require_implemented_vector_backend", lambda _: None
    )
    assert not ledger_holds_vectors()


async def test_metadata_rebuild_keeps_the_model_without_lancedb_vectors(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = CollectionInfo(
        name="kb", embeddings=3, embedding_model_id="m", embedding_dimension=3
    )
    monkeypatch.setattr(
        collections,
        "list_collections",
        AsyncMock(return_value=SimpleNamespace(status="success", collections=[listed])),
    )
    store, saved = MagicMock(), AsyncMock()
    monkeypatch.setattr(collection_manager, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(collection_manager.collection_manager, "save_collection", saved)

    await collection_manager._rebuild_collection_metadata_impl()

    assert store.method_calls == []
    saved.assert_awaited_once_with(listed)


def test_file_statuses_skip_the_legacy_indexed_fallback(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MagicMock()
    store.list_document_records_by_file_ids.return_value = [
        DocumentRecord(doc_id="d", file_id="f", user_id=1, collection="kb")
    ]
    monkeypatch.setattr(kb_file_service, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(kb_file_service, "_load_ingestion_status_impl", lambda **_: [])

    assert kb_file_service._aggregate_uploaded_file_statuses_impl(
        file_ids=["f"], user_id=1, is_admin=False, use_cache=False
    ) == {"f": "UNKNOWN"}
    store.list_indexed_doc_refs.assert_not_called()


def test_stale_file_cleanup_deletes_through_the_handle(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = MagicMock()
    query = db.query.return_value.filter.return_value.order_by.return_value
    query.all.return_value = [SimpleNamespace(file_id="f", created_at=None)]
    store = MagicMock()
    store.list_document_records_by_file_ids.return_value = [
        DocumentRecord(doc_id="d", file_id="f", user_id=1, collection="kb")
    ]
    monkeypatch.setattr(kb_file_service, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(
        kb_file_service,
        "_aggregate_uploaded_file_statuses_impl",
        lambda **_: {"f": "FAILED"},
    )
    coordinator = MagicMock()
    coordinator.delete_documents_data_sync.side_effect = NotImplementedError
    monkeypatch.setattr(kb, "get_kb_coordinator", lambda: coordinator)

    result = kb_file_service._reconcile_uploaded_files_impl(
        db, user_id=1, is_admin=False
    )

    coordinator.delete_documents_data_sync.assert_called_once_with(
        "kb", ["d"], user_id=1, is_admin=False
    )
    store.cascade_delete.assert_not_called()
    assert (result["deleted"], result["cleanup_errors"]) == (0, 1)


def test_version_cascade_delete_is_refused(milvus_deployment: None) -> None:
    with pytest.raises(ConfigurationError, match="does not support cascade cleanup"):
        cascade_cleaner.cascade_delete(
            target="collection", collection="kb", preview_only=False, confirm=True
        )


def test_search_time_model_inference_reads_no_lancedb_tables(
    milvus_deployment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = MagicMock()
    monkeypatch.setattr(migration_utils, "get_vector_store_raw_connection", connect)

    assert migration_utils._infer_embedding_config_from_collection("kb") == (
        None,
        None,
    )
    connect.assert_not_called()


def test_compaction_without_embeddings_tables_compacts_the_ledger(
    milvus_deployment: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = get_vector_index_store()
    handle = KBHandleProvider().open(_context(KBStorageBackend.MILVUS))
    handle.write_ingestion_status("doc", status="success", user_id=1)
    requested: list[list[str]] = []
    compact = store.compact_tables
    monkeypatch.setattr(
        store,
        "compact_tables",
        lambda names, policy=None: requested.append(names) or compact(names, policy),
    )

    _compact_storage_if_needed("model")

    assert requested == [[*_INGEST_TABLES, embeddings_table_name("model")]]
    assert not [name for name in store.list_table_names() if "embeddings_" in name]
    assert "compaction skipped" not in caplog.text
