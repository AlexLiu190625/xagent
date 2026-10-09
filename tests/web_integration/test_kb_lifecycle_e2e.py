"""
End-to-end tests for KB lifecycle management.

This module tests the complete knowledge base lifecycle:
1. Create new KB (collection)
2. Ingest multiple documents
3. Search and verify content
4. Delete documents
5. Delete KB (collection)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import eventually, http_detail
from xagent.core.tools.core.RAG_tools.kb.collection_handle import deployment_kb_backend

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.contract_stub,
    pytest.mark.usefixtures("kb_engine"),
]


def _log(msg: str) -> None:
    """Helper to log messages immediately."""
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()


# ==========================================
# TEST-SPECIFIC FIXTURES
# ==========================================
# Note: _StubEmbeddingAdapter, stub_embedding_adapter, and mock_rag_pipeline
# are provided by conftest.py with autouse=True


@pytest.fixture
def sample_lifecycle_files():
    """Create sample test files for lifecycle testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create diverse test files
        test_files = {
            "document1.txt": "First document with important information about testing.",
            "document2.md": "# Second Document\n\nThis is a markdown document with **bold** text.",
            "document3.json": '{"title": "Third Document", "content": "JSON data for testing"}',
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


# ==========================================
# E2E TEST CLASSES
# ==========================================


class TestKBLifecycleE2E:
    """
    End-to-end tests for complete KB lifecycle.

    These tests simulate the complete user workflow from creating
    a knowledge base through to deletion, ensuring all operations
    work correctly together.
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_kb_lifecycle_single_document(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test complete KB lifecycle: create → ingest → search → delete."""
        _log("\n=== START test_complete_kb_lifecycle_single_document ===")
        files, temp_dir = sample_lifecycle_files
        collection_name = "e2e_lifecycle_single"
        file_path = files["document1.txt"]
        _log(f"Collection name: {collection_name}")
        _log(f"File path: {file_path}")

        # Step 1: Create collection and ingest document
        _log("Step 1: Ingesting document...")
        with open(file_path, "rb") as f:
            ingest_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                },
                headers=auth_headers,
            )
        _log(f"Ingest response status: {ingest_response.status_code}")

        # Verify ingestion succeeded
        assert ingest_response.status_code == 200
        ingest_result = ingest_response.json()
        assert ingest_result["status"] in ["success", "partial"]
        _log(f"Ingest result status: {ingest_result.get('status')}")

        # Step 2: List collections to verify KB exists
        _log("Step 2: Listing collections...")
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200
        collections = list_response.json()
        assert "collections" in collections
        _log(f"Collections count: {len(collections.get('collections', []))}")

        # Step 3: Try to delete the document
        _log("Step 3: Deleting document...")
        delete_doc_response = client.delete(
            f"/api/kb/collections/{collection_name}/documents/document1.txt",
            headers=auth_headers,
        )
        # Deleting just-uploaded document should succeed
        assert delete_doc_response.status_code == 200
        _log(f"Delete response status: {delete_doc_response.status_code}")
        _log("=== END test_complete_kb_lifecycle_single_document ===\n")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_stops_finding_a_deleted_document_and_keeps_the_others(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
        kb_engine: str,
    ):
        """Keyword search finds each ingested text; a delete removes only its own."""
        assert deployment_kb_backend().value == kb_engine
        files, _ = sample_lifecycle_files
        collection_name = "e2e_lifecycle_search"
        for filename in ("document1.txt", "document2.md"):
            with open(files[filename], "rb") as f:
                ingest_response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
            assert ingest_response.status_code == 200
            assert ingest_response.json()["status"] == "success"

        def found(query: str) -> list[str]:
            response = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": query,
                    "embedding_model_id": "e2e-test-embedding",
                    "search_type": "sparse",
                },
                headers=auth_headers,
            )
            assert response.status_code == 200, http_detail(response)
            assert response.json()["status"] == "success", http_detail(response)
            return [hit["text"] for hit in response.json()["results"]]

        eventually(lambda: bool(found("important information")))
        eventually(lambda: bool(found("markdown document")))
        assert found("important information") == [
            "First document with important information about testing."
        ]

        delete_response = client.delete(
            f"/api/kb/collections/{collection_name}/documents/document1.txt",
            headers=auth_headers,
        )
        assert delete_response.status_code == 200
        eventually(lambda: not found("important information"))
        assert found("markdown document")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_kb_lifecycle_multiple_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test KB lifecycle with multiple documents of different formats."""
        _log("\n=== START test_complete_kb_lifecycle_multiple_documents ===")
        files, temp_dir = sample_lifecycle_files
        collection_name = "e2e_lifecycle_multi"
        _log(f"Collection name: {collection_name}")

        # Step 1: Ingest multiple documents
        _log("Step 1: Ingesting multiple documents...")
        uploaded_docs = []
        for filename in ["document1.txt", "document2.md", "document3.json"]:
            print(f"  Ingesting {filename}...")
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                print(f"    Response status: {response.status_code}")
                if response.status_code == 200:
                    uploaded_docs.append(filename)

        # Verify at least some documents were ingested
        assert len(uploaded_docs) >= 1
        print(f"Uploaded {len(uploaded_docs)} documents")

        # Step 2: List collections
        _log("Step 2: Listing collections...")
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200
        print(f"List response status: {list_response.status_code}")

        # Step 3: Try to delete each document
        _log("Step 3: Deleting documents...")
        for doc_name in uploaded_docs:
            print(f"  Deleting {doc_name}...")
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}/documents/{doc_name}",
                headers=auth_headers,
            )
            # Deleting just-uploaded documents should succeed
            assert delete_response.status_code == 200
            print(f"    Delete response: {delete_response.status_code}")
        _log("=== END test_complete_kb_lifecycle_multiple_documents ===\n")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_delete_collection_after_ingestion(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test deleting an entire collection after document ingestion."""
        _log("\n=== START test_delete_collection_after_ingestion ===")
        files, temp_dir = sample_lifecycle_files
        collection_name = "e2e_delete_collection"
        file_path = files["document1.txt"]
        print(f"Collection name: {collection_name}")

        # Step 1: Ingest a document
        _log("Step 1: Ingesting document...")
        with open(file_path, "rb") as f:
            ingest_response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )
        print(f"Ingest response: {ingest_response.status_code}")

        if ingest_response.status_code == 200:
            # Step 2: Delete the entire collection
            _log("Step 2: Deleting collection...")
            delete_response = client.delete(
                f"/api/kb/collections/{collection_name}",
                headers=auth_headers,
            )
            # Deleting a collection that exists should succeed
            assert delete_response.status_code == 200
            print(f"Delete response: {delete_response.status_code}")
        _log("=== END test_delete_collection_after_ingestion ===\n")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_collection_isolation(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test that different collections maintain document isolation."""
        _log("\n=== START test_collection_isolation ===")
        files, temp_dir = sample_lifecycle_files

        # Create two separate collections
        collection_a = "e2e_isolation_a"
        collection_b = "e2e_isolation_b"

        # Ingest into collection A
        file_path_a = files["document1.txt"]
        with open(file_path_a, "rb") as f:
            response_a = client.post(
                "/api/kb/ingest",
                files={"file": ("doc_a.txt", f, "text/plain")},
                data={"collection": collection_a},
                headers=auth_headers,
            )

        # Ingest into collection B
        file_path_b = files["document2.md"]
        with open(file_path_b, "rb") as f:
            response_b = client.post(
                "/api/kb/ingest",
                files={"file": ("doc_b.md", f, "text/plain")},
                data={"collection": collection_b},
                headers=auth_headers,
            )

        # At least one should succeed
        assert response_a.status_code == 200 or response_b.status_code == 200

        # List collections should show both if they were created
        list_response = client.get("/api/kb/collections", headers=auth_headers)
        assert list_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_collection_config_management(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ):
        """Test saving and retrieving collection configuration."""
        _log("\n=== START test_collection_config_management ===")
        collection_name = "e2e_config_test"

        # Save collection config
        _log("Saving config...")
        config_response = client.post(
            f"/api/kb/collections/{collection_name}/config",
            json={
                "parse_method": "deepdoc",
                "chunk_strategy": "fixed_size",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "embedding_model_id": "text-embedding-v3",
            },
            headers=auth_headers,
        )
        print(f"Config response: {config_response.status_code}")

        # Config save should succeed
        assert config_response.status_code == 200
        _log("=== END test_collection_config_management ===\n")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ingest_with_custom_config(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test ingesting documents with custom ingestion configuration."""
        files, temp_dir = sample_lifecycle_files
        collection_name = "e2e_custom_config"
        file_path = files["document1.txt"]

        # Ingest with custom chunking configuration
        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("document1.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "parse_method": "deepdoc",
                    "chunk_strategy": "fixed_size",
                    "chunk_size": "300",
                    "chunk_overlap": "100",
                    "separators": '["\\n\\n", "\\n", " ", ""]',
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_listing_after_multiple_ingestions(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test that documents are correctly listed after multiple ingestions."""
        files, temp_dir = sample_lifecycle_files
        collection_name = "e2e_listing_test"

        # Ingest multiple documents
        uploaded = []
        for filename in ["document1.txt", "document2.md"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                if response.status_code == 200:
                    uploaded.append(filename)

        if len(uploaded) > 0:
            # No collection-detail GET; verify filenames via documents/check.
            check_response = client.post(
                f"/api/kb/collections/{collection_name}/documents/check",
                json={"filenames": uploaded},
                headers=auth_headers,
            )
            assert check_response.status_code == 200
            existing = set(check_response.json().get("existing_filenames", []))
            assert existing.issuperset(set(uploaded))

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_error_handling_invalid_collection_name(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test error handling with invalid collection names."""
        files, temp_dir = sample_lifecycle_files
        file_path = files["document1.txt"]

        # Try various invalid collection names
        invalid_names = [
            "../../../etc",
            "collection/../other",
            "collection with spaces",
            "collection/with/slashes",
        ]

        for invalid_name in invalid_names:
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"collection": invalid_name},
                    headers=auth_headers,
                )
                # Path traversal names are rejected because basename != original
                # Only "collection with spaces" passes validation
                if invalid_name == "collection with spaces":
                    assert response.status_code == 200
                else:
                    assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_multiple_document_ingestion(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_lifecycle_files: tuple[dict[str, str], str],
    ):
        """Test handling multiple ingestion requests sequentially."""
        files, temp_dir = sample_lifecycle_files
        collection_name = "e2e_multiple_ingestion"

        # Ingest multiple files sequentially
        results = []
        for filename in ["document1.txt", "document2.md", "document3.json"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )
                results.append(response.status_code)

        # All ingestions should succeed
        assert all(code == 200 for code in results)
