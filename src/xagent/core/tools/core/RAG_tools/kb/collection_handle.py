"""Collection-scoped KB backend handle skeletons."""

from __future__ import annotations

from dataclasses import dataclass

from ..storage.contracts import MetadataStore, VectorIndexStore
from .models import KBBackendCapabilities, KBCollectionContext, KBStorageBackend


class KBHandleProvider:
    """Open collection-scoped handles for resolved KB contexts."""

    def open(self, context: KBCollectionContext) -> LanceDBCollectionHandle:
        """Return a backend-specific handle for the resolved collection context."""
        if context.backend is KBStorageBackend.LANCEDB:
            return LanceDBCollectionHandle(context)
        raise ValueError(
            f"KB storage backend {context.backend.value!r} is not supported by "
            "KBHandleProvider"
        )

    def reset_for_tests(self) -> None:
        """Clear provider-owned caches for test reset.

        The current provider is stateless, but the hook keeps the coordinator
        reset path ready for backend handle caches.
        """


@dataclass(frozen=True)
class LanceDBCollectionHandle:
    """Thin LanceDB-backed collection handle for #495 compatibility wiring."""

    context: KBCollectionContext

    @property
    def metadata_store(self) -> MetadataStore:
        """Return the metadata store bound to this collection context."""
        return self.context.metadata_store

    @property
    def vector_index_store(self) -> VectorIndexStore:
        """Return the vector index store bound to this collection context."""
        return self.context.vector_index_store

    @property
    def backend(self) -> KBStorageBackend:
        """Return the collection storage backend."""
        return self.context.backend

    @property
    def capabilities(self) -> KBBackendCapabilities:
        """Return backend capabilities for this collection."""
        return self.context.capabilities
