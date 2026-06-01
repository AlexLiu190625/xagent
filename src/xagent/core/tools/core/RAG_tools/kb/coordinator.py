"""Semantic KB coordinator skeleton."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from contextvars import copy_context
from typing import Any, Optional, TypeVar

from ..storage.factory import StorageFactory
from ..utils.user_scope import resolve_user_scope
from .collection_handle import KBHandleProvider, LanceDBCollectionHandle
from .file_compatibility import KBFileCompatibilityFacade
from .models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBContextRequest,
    KBStorageBackend,
    KBUserScope,
)
from .storage_shim import KBStorageShimCompatibilityFacade

T = TypeVar("T")

KB_STORAGE_METADATA_KEY = "kb_storage"


class KBCoordinator:
    """KB-level semantic entry point for future compatibility facades."""

    def __init__(
        self,
        storage_factory: StorageFactory | None = None,
        handle_provider: KBHandleProvider | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
        file_compatibility: KBFileCompatibilityFacade | None = None,
    ) -> None:
        self._storage_factory = storage_factory or StorageFactory.get_factory()
        self._handle_provider = handle_provider or KBHandleProvider()
        self._storage_shim = storage_shim or KBStorageShimCompatibilityFacade(
            storage_factory=self._storage_factory
        )
        self._file_compatibility = file_compatibility or KBFileCompatibilityFacade()

    @property
    def storage_shim(self) -> KBStorageShimCompatibilityFacade:
        """Return the low-level storage compatibility facade."""
        return self._storage_shim

    @property
    def file_compatibility(self) -> KBFileCompatibilityFacade:
        """Return the uploaded-file and physical compatibility facade."""
        return self._file_compatibility

    @property
    def file_compat(self) -> KBFileCompatibilityFacade:
        """Backward-friendly short alias for the file compatibility facade."""
        return self._file_compatibility

    async def get_context(self, request: KBContextRequest) -> KBCollectionContext:
        """Resolve collection, caller scope, stores, backend, and capabilities."""
        collection = self._normalize_collection(request.collection)
        access_mode = self._normalize_access_mode(request.access_mode)
        user_scope = self._resolve_user_scope(request)
        metadata_store = self._storage_shim.get_metadata_store()
        vector_index_store = self._storage_shim.get_vector_index_store()

        collection_info = None
        try:
            collection_info = await metadata_store.get_collection(collection)
        except ValueError as exc:
            if not self._is_missing_collection_error(collection, exc):
                raise
            if not (request.hide_missing or request.allow_create):
                raise ValueError(f"Collection '{collection}' not found") from exc

        backend = self._resolve_backend(collection_info)
        capabilities = self._capabilities_for_backend(backend)

        return KBCollectionContext(
            collection=collection,
            user_scope=user_scope,
            access_mode=access_mode,
            allow_create=bool(request.allow_create),
            hide_missing=bool(request.hide_missing),
            metadata_store=metadata_store,
            vector_index_store=vector_index_store,
            backend=backend,
            capabilities=capabilities,
            collection_info=collection_info,
        )

    def get_context_sync(self, request: KBContextRequest) -> KBCollectionContext:
        """Synchronous wrapper for legacy compatibility surfaces."""
        return _run_in_separate_loop(self.get_context(request))

    async def open_collection(
        self, request: KBContextRequest
    ) -> LanceDBCollectionHandle:
        """Open a thin collection handle for the resolved context."""
        context = await self.get_context(request)
        return self._handle_provider.open(context)

    def open_collection_sync(
        self, request: KBContextRequest
    ) -> LanceDBCollectionHandle:
        """Synchronous wrapper for opening a collection handle."""
        return _run_in_separate_loop(self.open_collection(request))

    @staticmethod
    def _normalize_collection(collection: str) -> str:
        normalized = collection.strip() if isinstance(collection, str) else ""
        if not normalized:
            raise ValueError("collection must be a non-empty string")
        return normalized

    @staticmethod
    def _normalize_access_mode(access_mode: KBAccessMode | str) -> KBAccessMode:
        if isinstance(access_mode, KBAccessMode):
            return access_mode
        try:
            return KBAccessMode(str(access_mode).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in KBAccessMode)
            raise ValueError(
                f"Invalid KB access mode {access_mode!r}; choose one of: {allowed}"
            ) from exc

    @staticmethod
    def _is_missing_collection_error(collection: str, exc: ValueError) -> bool:
        message = str(exc)
        return message in {
            f"Collection '{collection}' not found",
            "Table 'collection_metadata' was not found",
        }

    @staticmethod
    def _resolve_user_scope(request: KBContextRequest) -> KBUserScope:
        scope = resolve_user_scope(user_id=request.user_id, is_admin=request.is_admin)
        return KBUserScope(user_id=scope.user_id, is_admin=bool(scope.is_admin))

    def _resolve_backend(self, collection_info: object | None) -> KBStorageBackend:
        if collection_info is None:
            return KBStorageBackend.LANCEDB

        extra_metadata = getattr(collection_info, "extra_metadata", None) or {}
        binding = extra_metadata.get(KB_STORAGE_METADATA_KEY)
        if binding is None:
            return KBStorageBackend.LANCEDB

        if isinstance(binding, str):
            return self._parse_backend(binding)

        if isinstance(binding, dict):
            raw_backend = binding.get("backend")
            if raw_backend is None or str(raw_backend).strip() == "":
                return KBStorageBackend.LANCEDB
            return self._parse_backend(str(raw_backend))

        raise ValueError(
            f"Invalid {KB_STORAGE_METADATA_KEY} binding shape: {type(binding).__name__}"
        )

    @staticmethod
    def _parse_backend(raw_backend: str) -> KBStorageBackend:
        try:
            return KBStorageBackend(raw_backend.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(backend.value for backend in KBStorageBackend)
            raise ValueError(
                f"Invalid {KB_STORAGE_METADATA_KEY} backend {raw_backend!r}; "
                f"choose one of: {allowed}"
            ) from exc

    @staticmethod
    def _capabilities_for_backend(backend: KBStorageBackend) -> KBBackendCapabilities:
        if backend is KBStorageBackend.LANCEDB:
            return KBBackendCapabilities.lancedb()
        return KBBackendCapabilities.unsupported()

    def reset_compatibility_caches(self) -> None:
        """Clear coordinator-owned compatibility facade caches."""
        self._storage_shim.reset_coordinator_caches()
        self._handle_provider.reset_for_tests()


_coordinator_lock = threading.RLock()
_coordinator: Optional[KBCoordinator] = None


def get_kb_coordinator() -> KBCoordinator:
    """Return the process-global KB semantic coordinator."""
    global _coordinator
    if _coordinator is None:
        with _coordinator_lock:
            if _coordinator is None:
                _coordinator = KBCoordinator()
    return _coordinator


def reset_kb_coordinator_for_tests() -> None:
    """Reset process-global KB coordinator state for tests."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is not None:
            _coordinator.reset_compatibility_caches()
        _coordinator = None


def _run_in_separate_loop(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code, including inside an existing event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    if not loop.is_running():
        return asyncio.run(awaitable)

    result: Optional[T] = None
    error: Optional[BaseException] = None
    context = copy_context()

    def target() -> None:
        nonlocal result, error
        try:
            result = context.run(lambda: asyncio.run(awaitable))
        except BaseException as exc:  # noqa: BLE001 - propagate from worker thread
            error = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result  # type: ignore[return-value]
