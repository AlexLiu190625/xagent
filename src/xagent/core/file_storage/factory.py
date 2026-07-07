from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

import fsspec

from ...config import (
    get_file_materialize_dir,
    get_file_storage_options,
    get_file_storage_uri,
)
from .scoped import ScopedFileStorage
from .storage import FsspecFileStorage

_DEFAULT_S3_CONFIG_KWARGS = {
    "connect_timeout": 3,
    "read_timeout": 10,
    "retries": {"max_attempts": 1},
}


def get_file_storage_backend() -> str:
    """Backend scheme of the configured durable storage ("s3", "file", ...)."""
    return get_unscoped_file_storage().backend


def get_user_file_storage(user_id: int) -> ScopedFileStorage:
    """Get a storage view scoped to a user's key prefix.

    This is the primary storage factory: every operation through the returned
    handle is confined to ``users/{user_id}``. Reach for
    ``get_unscoped_file_storage`` only in infrastructure code inside this
    package.
    """
    return ScopedFileStorage(
        storage=get_unscoped_file_storage(),
        prefix=f"users/{int(user_id)}",
    )


@lru_cache
def get_unscoped_file_storage() -> FsspecFileStorage:
    """Build the configured durable file storage backend (no prefix scope)."""
    uri = get_file_storage_uri()
    options = get_file_storage_options()
    parsed = urlparse(uri)
    backend = parsed.scheme or "file"
    if backend == "s3":
        options = _with_default_s3_config_kwargs(options)

    try:
        fs, root = fsspec.core.url_to_fs(uri, **options)
    except ImportError as exc:
        if backend == "s3":
            raise RuntimeError(
                "XAGENT_FILE_STORAGE_URI uses s3:// but s3fs is not installed"
            ) from exc
        raise

    return FsspecFileStorage(
        fs=fs,
        root=str(root),
        backend=backend,
        base_uri=uri,
        materialize_dir=get_file_materialize_dir(),
    )


def _with_default_s3_config_kwargs(options: dict) -> dict:
    config_kwargs = options.get("config_kwargs")
    if config_kwargs is None:
        return {**options, "config_kwargs": dict(_DEFAULT_S3_CONFIG_KWARGS)}
    if not isinstance(config_kwargs, dict):
        return options
    return {
        **options,
        "config_kwargs": {
            **_DEFAULT_S3_CONFIG_KWARGS,
            **config_kwargs,
        },
    }
