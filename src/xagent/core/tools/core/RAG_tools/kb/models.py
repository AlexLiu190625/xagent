"""Semantic KB coordinator models.

These types describe collection-level KB context without moving existing
storage, API, pipeline, or tool behavior into the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from ..core.schemas import CollectionInfo
from ..storage.contracts import MetadataStore, VectorIndexStore


class KBAccessMode(StrEnum):
    """Semantic access mode requested by a KB caller."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class KBStorageBackend(StrEnum):
    """Collection-level KB storage backend binding."""

    LANCEDB = "lancedb"


@dataclass(frozen=True)
class KBBackendCapabilities:
    """Capabilities exposed by a collection backend handle."""

    supports_documents: bool
    supports_parses: bool
    supports_chunks: bool
    supports_embeddings: bool
    supports_search: bool
    supports_versions: bool
    supports_raw_connection: bool

    @classmethod
    def lancedb(cls) -> KBBackendCapabilities:
        """Return current LanceDB-compatible KB capabilities."""
        return cls(
            supports_documents=True,
            supports_parses=True,
            supports_chunks=True,
            supports_embeddings=True,
            supports_search=True,
            supports_versions=True,
            supports_raw_connection=True,
        )

    @classmethod
    def unsupported(cls) -> KBBackendCapabilities:
        """Return capabilities for a known but unavailable backend."""
        return cls(
            supports_documents=False,
            supports_parses=False,
            supports_chunks=False,
            supports_embeddings=False,
            supports_search=False,
            supports_versions=False,
            supports_raw_connection=False,
        )


@dataclass(frozen=True)
class KBUserScope:
    """Resolved caller scope for KB context operations."""

    user_id: Optional[int]
    is_admin: bool


@dataclass(frozen=True)
class KBContextRequest:
    """Request for resolving collection-level KB context."""

    collection: str
    user_id: Optional[int] = None
    is_admin: Optional[bool] = None
    access_mode: KBAccessMode = KBAccessMode.READ
    allow_create: bool = False
    hide_missing: bool = False


@dataclass(frozen=True)
class KBCollectionContext:
    """Resolved collection-level context for coordinator and handle callers."""

    collection: str
    user_scope: KBUserScope
    access_mode: KBAccessMode
    allow_create: bool
    hide_missing: bool
    metadata_store: MetadataStore
    vector_index_store: VectorIndexStore
    backend: KBStorageBackend
    capabilities: KBBackendCapabilities
    collection_info: Optional[CollectionInfo] = None
