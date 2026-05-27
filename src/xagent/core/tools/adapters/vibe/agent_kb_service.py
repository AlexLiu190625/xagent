import logging

from ...core.RAG_tools.core.schemas import IngestionConfig
from ...core.RAG_tools.management.ingestion_prepare import (
    PreparedKnowledgeBaseIngestion,
    prepare_kb_ingestion,
)

logger = logging.getLogger(__name__)


class AgentKnowledgeBaseError(RuntimeError):
    """Raised when agent-triggered knowledge base setup cannot be completed."""


class AgentKnowledgeBaseService:
    """Shared collection setup/refresh flow for agent-triggered KB creation."""

    def __init__(
        self,
        user_id: int,
        is_admin: bool = False,
        default_embedding_model_id: str | None = None,
    ) -> None:
        self.user_id = user_id
        self.is_admin = is_admin
        self.default_embedding_model_id = default_embedding_model_id

    async def prepare_collection(
        self,
        collection_name: str,
        ingestion_config: IngestionConfig,
    ) -> PreparedKnowledgeBaseIngestion:
        from .....web.config import sanitize_path_component

        safe_collection = sanitize_path_component(collection_name, "collection")

        try:
            return await prepare_kb_ingestion(
                collection_name=safe_collection,
                ingestion_config=ingestion_config,
                user_id=self.user_id,
                is_admin=self.is_admin,
                fallback_embedding_model_id=self.default_embedding_model_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to prepare agent knowledge base %s: %s",
                safe_collection,
                exc,
            )
            raise AgentKnowledgeBaseError(
                f"Failed to prepare knowledge base '{safe_collection}'"
            ) from exc

    async def refresh_collection_metadata(self, collection_name: str) -> None:
        from ...core.RAG_tools.management.collections import list_collections

        if not self.is_admin:
            # Non-admin realtime refreshes do not persist metadata and only add scan cost.
            return

        try:
            # Refresh metadata cache so agent-created KBs are visible like API-created ones.
            await list_collections(
                user_id=self.user_id,
                is_admin=self.is_admin,
                force_realtime=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to refresh collection metadata after agent ingestion for %s: %s",
                collection_name,
                exc,
            )
            raise AgentKnowledgeBaseError(
                f"Failed to refresh knowledge base metadata for '{collection_name}'"
            ) from exc
