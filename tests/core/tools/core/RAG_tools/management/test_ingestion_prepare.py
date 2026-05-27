import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_MODEL_ID,
    CollectionInfo,
    IngestionConfig,
)
from xagent.core.tools.core.RAG_tools.management.ingestion_prepare import (
    prepare_kb_ingestion,
)


@pytest.mark.asyncio
async def test_prepare_uses_initialized_collection_bound_model():
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(
        return_value=json.dumps({"embedding_model_id": "stored-short"})
    )
    metadata_store.save_collection_config = AsyncMock()
    collection = CollectionInfo(
        name="faq",
        embedding_model_id="bound-long",
        embedding_dimension=1024,
    )

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.collection_manager.get_collection",
            new=AsyncMock(return_value=collection),
        ),
    ):
        prepared = await prepare_kb_ingestion(
            collection_name="faq",
            ingestion_config=IngestionConfig(embedding_model_id="request-short"),
            user_id=7,
            fallback_embedding_model_id="user-default",
        )

    assert prepared.ingestion_config.embedding_model_id == "bound-long"
    assert prepared.collection_existed_before is True
    assert prepared.should_save_config is False
    metadata_store.save_collection_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_uninitialized_collection_uses_saved_config():
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(
        return_value=json.dumps({"embedding_model_id": "stored-choice"})
    )
    metadata_store.save_collection_config = AsyncMock()
    collection = CollectionInfo(name="faq")

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.collection_manager.get_collection",
            new=AsyncMock(return_value=collection),
        ),
    ):
        prepared = await prepare_kb_ingestion(
            collection_name="faq",
            ingestion_config=IngestionConfig(embedding_model_id="request-choice"),
            user_id=7,
            save_config=False,
        )

    assert prepared.ingestion_config.embedding_model_id == "stored-choice"
    assert prepared.collection_existed_before is True
    assert prepared.should_save_config is True
    metadata_store.save_collection_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_new_collection_preserves_requested_model_id():
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.collection_manager.get_collection",
            new=AsyncMock(side_effect=ValueError("not found")),
        ),
    ):
        prepared = await prepare_kb_ingestion(
            collection_name="faq",
            ingestion_config=IngestionConfig(embedding_model_id="text-embedding-v4"),
            user_id=7,
        )

    assert prepared.ingestion_config.embedding_model_id == "text-embedding-v4"
    metadata_store.save_collection_config.assert_awaited_once()
    _, kwargs = metadata_store.save_collection_config.await_args
    assert json.loads(kwargs["config_json"]) == {
        "embedding_model_id": "text-embedding-v4"
    }


@pytest.mark.asyncio
async def test_prepare_new_collection_falls_back_to_user_default_then_system_default():
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.collection_manager.get_collection",
            new=AsyncMock(side_effect=ValueError("not found")),
        ),
    ):
        prepared = await prepare_kb_ingestion(
            collection_name="faq",
            ingestion_config=IngestionConfig(),
            user_id=7,
            fallback_embedding_model_id="user-default",
        )

    assert prepared.ingestion_config.embedding_model_id == "user-default"

    metadata_store.save_collection_config.reset_mock()
    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.ingestion_prepare.collection_manager.get_collection",
            new=AsyncMock(side_effect=ValueError("not found")),
        ),
    ):
        prepared = await prepare_kb_ingestion(
            collection_name="faq",
            ingestion_config=IngestionConfig(),
            user_id=7,
        )

    assert prepared.ingestion_config.embedding_model_id == DEFAULT_EMBEDDING_MODEL_ID
