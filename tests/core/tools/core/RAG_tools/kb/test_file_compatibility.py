"""Tests for the KB file compatibility facade."""

from __future__ import annotations

from inspect import signature


def test_kb_file_compatibility_public_surface_imports() -> None:
    """Given the KB package, the file facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb

    assert hasattr(kb, "KBFileCompatibilityFacade")
    assert hasattr(kb.get_kb_coordinator(), "file_compatibility")


def test_kb_file_compatibility_methods_match_public_helper_signatures() -> None:
    """Given legacy helpers, facade methods preserve their call signatures."""
    from xagent.core.tools.core.RAG_tools.kb import KBFileCompatibilityFacade
    from xagent.web.services import kb_collection_service, kb_file_service

    facade = KBFileCompatibilityFacade()
    pairs = [
        (
            facade.upsert_uploaded_file_record,
            kb_file_service.upsert_uploaded_file_record,
        ),
        (facade.list_documents_for_user, kb_file_service.list_documents_for_user),
        (
            facade.build_uploaded_filename_map,
            kb_file_service.build_uploaded_filename_map,
        ),
        (
            facade.get_document_record_file_id,
            kb_file_service.get_document_record_file_id,
        ),
        (facade.resolve_document_filename, kb_file_service.resolve_document_filename),
        (
            facade.delete_uploaded_file_if_orphaned,
            kb_file_service.delete_uploaded_file_if_orphaned,
        ),
        (
            facade.aggregate_uploaded_file_statuses,
            kb_file_service.aggregate_uploaded_file_statuses,
        ),
        (facade.reconcile_uploaded_files, kb_file_service.reconcile_uploaded_files),
        (
            facade.list_collection_uploaded_file_owner_ids,
            kb_collection_service.list_collection_uploaded_file_owner_ids,
        ),
        (
            facade.delete_collection_physical_dir,
            kb_collection_service.delete_collection_physical_dir,
        ),
        (
            facade.delete_collection_uploaded_files,
            kb_collection_service.delete_collection_uploaded_files,
        ),
        (
            facade.rename_collection_storage,
            kb_collection_service.rename_collection_storage,
        ),
    ]

    for facade_method, public_helper in pairs:
        assert signature(facade_method) == signature(public_helper)


def test_public_file_helper_delegates_through_facade(monkeypatch) -> None:
    """Given a public file helper call, it routes through the coordinator facade."""
    from xagent.web.services import kb_file_service

    class _FakeFacade:
        def get_document_record_file_id(self, record):
            assert record == {"file_id": "legacy"}
            return "facade-file-id"

    monkeypatch.setattr(
        kb_file_service,
        "_get_file_compatibility_facade",
        lambda: _FakeFacade(),
    )

    assert (
        kb_file_service.get_document_record_file_id({"file_id": "legacy"})
        == "facade-file-id"
    )


def test_public_collection_helper_delegates_through_facade(monkeypatch) -> None:
    """Given a public collection helper call, it routes through the facade."""
    from xagent.web.services import kb_collection_service

    class _FakeFacade:
        def list_collection_uploaded_file_owner_ids(self, db, *, collection_name: str):
            assert db == "db"
            assert collection_name == "kb"
            return {1, 2}

    monkeypatch.setattr(
        kb_collection_service,
        "_get_file_compatibility_facade",
        lambda: _FakeFacade(),
    )

    assert kb_collection_service.list_collection_uploaded_file_owner_ids(
        "db",
        collection_name="kb",
    ) == {1, 2}
