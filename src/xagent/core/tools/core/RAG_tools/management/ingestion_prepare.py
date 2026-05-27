"""Shared preparation for knowledge-base ingestion entry points."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from ..core.schemas import DEFAULT_EMBEDDING_MODEL_ID, CollectionInfo, IngestionConfig
from ..storage.factory import get_metadata_store
from .collection_manager import collection_manager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedKnowledgeBaseIngestion:
    """Resolved collection name and config for an ingestion request."""

    collection_name: str
    ingestion_config: IngestionConfig
    collection_existed_before: bool
    should_save_config: bool


def _normalize_model_id(model_id: Optional[str]) -> Optional[str]:
    if not isinstance(model_id, str):
        return None
    normalized = model_id.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def _config_with_embedding_model(
    config: IngestionConfig, embedding_model_id: str
) -> IngestionConfig:
    if config.embedding_model_id == embedding_model_id:
        return config
    return config.model_copy(update={"embedding_model_id": embedding_model_id})


def _parse_ingestion_config(config_json: Optional[str]) -> Optional[IngestionConfig]:
    if not config_json:
        return None
    try:
        payload = json.loads(config_json)
        if not isinstance(payload, dict):
            return None
        return IngestionConfig(**payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse saved collection ingestion config: %s", exc)
        return None


async def prepare_kb_ingestion(
    *,
    collection_name: str,
    ingestion_config: IngestionConfig,
    user_id: int,
    is_admin: bool = False,
    fallback_embedding_model_id: Optional[str] = None,
    save_config: bool = True,
) -> PreparedKnowledgeBaseIngestion:
    """Resolve the effective ingestion config for a knowledge-base collection.

    The collection's existing binding is the source of truth once initialized.
    For uninitialized collections, preserve the user's saved or requested model
    identifier; only fall back to a default when no user choice exists.
    """

    metadata_store = get_metadata_store()

    collection_info: Optional[CollectionInfo]
    try:
        collection_info = await collection_manager.get_collection(collection_name)
    except ValueError:
        collection_info = None

    stored_config = _parse_ingestion_config(
        await metadata_store.get_collection_config(
            collection=collection_name,
            user_id=user_id,
            is_admin=is_admin,
        )
    )

    bound_model_id = _normalize_model_id(
        collection_info.embedding_model_id if collection_info is not None else None
    )
    stored_model_id = _normalize_model_id(
        stored_config.embedding_model_id if stored_config is not None else None
    )
    requested_model_id = _normalize_model_id(ingestion_config.embedding_model_id)
    fallback_model_id = _normalize_model_id(fallback_embedding_model_id)

    if (
        collection_info is not None
        and collection_info.is_initialized
        and bound_model_id
    ):
        effective_config = _config_with_embedding_model(
            ingestion_config, bound_model_id
        )
        collection_existed_before = True
        should_save_config = False
    else:
        selected_model_id = (
            stored_model_id
            or requested_model_id
            or fallback_model_id
            or DEFAULT_EMBEDDING_MODEL_ID
        )
        effective_config = _config_with_embedding_model(
            ingestion_config, selected_model_id
        )
        collection_existed_before = (
            collection_info is not None or stored_config is not None
        )
        should_save_config = True

    if save_config and should_save_config:
        await metadata_store.save_collection_config(
            collection=collection_name,
            config_json=effective_config.model_dump_json(exclude_unset=True),
            user_id=user_id,
        )

    return PreparedKnowledgeBaseIngestion(
        collection_name=collection_name,
        ingestion_config=effective_config,
        collection_existed_before=collection_existed_before,
        should_save_config=should_save_config,
    )
