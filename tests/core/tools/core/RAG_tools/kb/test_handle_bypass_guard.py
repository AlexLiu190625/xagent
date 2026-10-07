"""Upper layers must not read or delete chunk and embedding rows through the
global store; they go through ``KBCollectionHandle`` so each engine owns them.

The scan covers ``src/xagent`` outside the LanceDB store, handle and migration
code. It matches by name, so it is best effort. Existing bypasses are
allowlisted one by one; the allowlist must match the scan exactly, so a fixed
bypass has to leave it.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[6] / "src" / "xagent"
IMPLEMENTATIONS = (
    "xagent/core/tools/core/RAG_tools/storage/",
    "xagent/core/tools/core/RAG_tools/kb/collection_handle.py",
    "xagent/migrations/lancedb/",
)

# VectorIndexStore methods that read or delete chunk/embedding rows.
ROW_METHODS = frozenset(
    """
    aggregate_collection_stats aggregate_document_counts aggregate_document_stats
    cascade_delete cleanup_cascade_by_scope cleanup_cascade_by_scope_async
    delete_chunk_records delete_collection_data delete_document_data
    delete_documents_data delete_embedding_records list_indexed_doc_refs
    list_table_names list_version_candidate_rows list_version_candidate_rows_async
    open_embeddings_table search_fts_async search_fts_by_model_async search_vectors
    search_vectors_async search_vectors_by_model search_vectors_by_model_async
    """.split()
)
# Generic table readers, flagged unless the table is a literal other than
# ``chunks`` or ``embeddings_*``.
TABLE_METHODS = frozenset(
    """
    count_rows count_rows_async count_rows_or_zero get_vector_dimension
    get_vector_dimension_async iter_batches iter_batches_async
    """.split()
)
RAW_ACCESSORS = frozenset(
    {"get_vector_store_raw_connection", "list_embeddings_table_names"}
)
INFERS_MODEL = "infers the embedding model from embeddings_* tables"

# Existing bypasses: how often each one occurs and what it is for.
ALLOWLIST: dict[str, dict[tuple[str, str], tuple[int, str]]] = {
    "xagent/core/tools/core/RAG_tools/management/collections.py": {
        ("_list_collections_impl", "aggregate_collection_stats"): (
            1,
            "collection list stats",
        ),
        ("_list_documents_impl", "aggregate_document_counts"): (
            2,
            "document list counts",
        ),
        ("_list_documents_impl", "list_table_names"): (1, "document list counts"),
        ("_get_document_stats_impl", "aggregate_document_stats"): (
            1,
            "document stats",
        ),
        ("_get_document_stats_impl", "count_rows"): (2, "document stats"),
        ("_get_document_stats_impl", "list_table_names"): (1, "document stats"),
    },
    "xagent/core/tools/core/RAG_tools/management/collection_manager.py": {
        ("_rebuild_collection_stats_impl", "aggregate_collection_stats"): (
            1,
            "stats rebuild (tests only)",
        ),
        ("_rebuild_collection_metadata_impl", "list_table_names"): (1, INFERS_MODEL),
        ("_rebuild_collection_metadata_impl", "count_rows_or_zero"): (1, INFERS_MODEL),
        ("_rebuild_collection_metadata_impl", "get_vector_dimension"): (
            1,
            INFERS_MODEL,
        ),
    },
    "xagent/web/services/kb_file_service.py": {
        ("_aggregate_uploaded_file_statuses_impl", "list_indexed_doc_refs"): (
            1,
            "indexed fallback for legacy files",
        ),
        ("_reconcile_uploaded_files_impl", "cascade_delete"): (
            1,
            "stale-file cleanup (delete_stale=True)",
        ),
    },
    "xagent/core/tools/core/RAG_tools/kb/version_compatibility.py": {
        ("KBVersionCompatibilityFacade.cascade_delete", "cascade_delete"): (
            1,
            "version cleanup",
        ),
    },
    "xagent/core/tools/core/RAG_tools/utils/migration_utils.py": {
        (
            "_infer_embedding_config_from_collection",
            "get_vector_store_raw_connection",
        ): (1, "model inference at search time"),
        ("migrate_embeddings_table", "get_vector_store_raw_connection"): (
            1,
            "body of LanceDBVectorIndexStore.migrate_embeddings_table",
        ),
    },
    "xagent/web/app.py": {
        ("startup_event", "list_embeddings_table_names"): (
            1,
            "startup user_id migration check",
        ),
    },
}


def _callee(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_other_literal_table(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value != "chunks"
        and not node.value.startswith("embeddings_")
    )


class _BypassScanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.store_names: list[set[str]] = [set()]
        self.found: Counter[tuple[str, str]] = Counter()

    def _record(self, name: str) -> None:
        self.found[(".".join(self.scope) or "<module>", name)] += 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        names = {
            arg.arg
            for arg in node.args.args + node.args.kwonlyargs
            if arg.annotation is not None
            and "VectorIndexStore" in ast.unparse(arg.annotation)
        }
        names |= {
            target.id
            for child in ast.walk(node)
            if isinstance(child, (ast.Assign, ast.AnnAssign))
            and child.value is not None
            and self._is_store(child.value)
            for target in (
                child.targets if isinstance(child, ast.Assign) else [child.target]
            )
            if isinstance(target, ast.Name)
        }
        self.scope.append(node.name)
        self.store_names.append(names | self.store_names[-1])
        self.generic_visit(node)
        self.store_names.pop()
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _is_store(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Call):
            return _callee(node) == "get_vector_index_store"
        if isinstance(node, ast.Name):
            return node.id in self.store_names[-1]
        return isinstance(node, ast.Attribute) and node.attr == "vector_index_store"

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ROW_METHODS and self._is_store(node.value):
            self._record(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee(node)
        if callee in RAW_ACCESSORS:
            self._record(callee)
        elif (
            callee in TABLE_METHODS
            and isinstance(node.func, ast.Attribute)
            and self._is_store(node.func.value)
        ):
            table = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg == "table_name":
                    table = keyword.value
            if not _is_other_literal_table(table):
                self._record(callee)
        self.generic_visit(node)


def _scan(source: str) -> Counter[tuple[str, str]]:
    scanner = _BypassScanner()
    scanner.visit(ast.parse(source))
    return scanner.found


def test_upper_layers_reach_chunks_and_embeddings_only_through_the_handle() -> None:
    found: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(SRC_ROOT.parent).as_posix()
        if not relative.startswith(IMPLEMENTATIONS):
            for hit, count in _scan(path.read_text("utf-8")).items():
                found[(relative, *hit)] += count
    allowed = Counter(
        {
            (path, *hit): count
            for path, hits in ALLOWLIST.items()
            for hit, (count, _purpose) in hits.items()
        }
    )

    assert found - allowed == Counter(), "new bypass of KBCollectionHandle"
    assert allowed - found == Counter(), "fixed bypass still allowlisted"


def test_scanner_flags_each_bypass_shape() -> None:
    source = """
def direct():
    get_vector_index_store().delete_document_data("kb", "d", 1, False)

def bound_name():
    store = get_vector_index_store()
    return store.aggregate_collection_stats

def annotated(store: VectorIndexStore):
    return store.count_rows("chunks")

def computed_table(table_name):
    store = get_vector_index_store()
    return store.count_rows_or_zero(table_name)

def context_store(context):
    return context.vector_index_store.cascade_delete(target="collection")

class Facade:
    def enumerate(self, conn):
        return list_embeddings_table_names(conn)

def closure():
    store = get_vector_index_store()

    def _compensate():
        store.delete_documents_data("kb", ["d"], None, True)

    return _compensate

def annotated_assignment():
    store: VectorIndexStore = get_vector_index_store()
    return store.list_indexed_doc_refs("kb")

def other_tables_only(facade):
    store = get_vector_index_store()
    store.count_rows("documents")
    store.iter_batches(table_name="parses")
    return facade.cascade_delete()

async def each_occurrence(store: VectorIndexStore):
    store.count_rows("chunks")
    store.count_rows("chunks")
    return await store.get_vector_dimension_async("embeddings_m")
"""

    assert _scan(source) == {
        ("each_occurrence", "count_rows"): 2,
        ("each_occurrence", "get_vector_dimension_async"): 1,
        ("direct", "delete_document_data"): 1,
        ("bound_name", "aggregate_collection_stats"): 1,
        ("annotated", "count_rows"): 1,
        ("computed_table", "count_rows_or_zero"): 1,
        ("context_store", "cascade_delete"): 1,
        ("Facade.enumerate", "list_embeddings_table_names"): 1,
        ("closure._compensate", "delete_documents_data"): 1,
        ("annotated_assignment", "list_indexed_doc_refs"): 1,
    }
