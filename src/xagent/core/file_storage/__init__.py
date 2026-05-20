"""Durable file storage abstraction for user-visible files."""

from .factory import get_file_storage
from .storage import FsspecFileStorage
from .types import StoredObject

__all__ = ["FsspecFileStorage", "StoredObject", "get_file_storage"]
