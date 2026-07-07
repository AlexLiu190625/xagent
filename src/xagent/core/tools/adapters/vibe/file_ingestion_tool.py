import asyncio
import logging
import re
import time
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Type

from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.RAG_tools.kb import KBToolCompatibilityFacade


@dataclass(frozen=True)
class UploadedFileSnapshot:
    """Scalar upload metadata retained after closing the DB session."""

    user_id: int
    file_id: str
    filename: str
    storage_path: str
    storage_key: str | None
    storage_status: str | None
    checksum: str | None


class CreateKnowledgeBaseFromFileArgs(BaseModel):
    file_ids: List[str] = Field(
        description="List of uploaded file IDs to ingest into the knowledge base."
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Name for the knowledge base collection. If not provided, one will be generated from the first file name.",
    )


class CreateKnowledgeBaseFromFileResult(BaseModel):
    success: bool
    collection_name: str
    message: str
    files_ingested: int


class CreateKnowledgeBaseFromFileTool(AbstractBaseTool):
    """Tool to create a knowledge base by ingesting already-uploaded files."""

    category = ToolCategory.KNOWLEDGE

    def __init__(self, user_id: int, is_admin: bool = False) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self.user_id = user_id
        self.is_admin = is_admin

    @property
    def name(self) -> str:
        return "create_knowledge_base_from_file"

    @property
    def description(self) -> str:
        return (
            "Create a new knowledge base by ingesting files that the user has already uploaded. "
            "Use this tool when the user has uploaded one or more files and wants to build a knowledge base from them. "
            "Pass the file_ids from the uploaded files. "
            "Returns the collection_name which you should then use when creating or updating the agent."
        )

    def args_type(self) -> Type[BaseModel]:
        return CreateKnowledgeBaseFromFileArgs

    def return_type(self) -> Type[BaseModel]:
        return CreateKnowledgeBaseFromFileResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await _get_tool_compatibility_facade().create_knowledge_base_from_file(
            args,
            user_id=self.user_id,
            is_admin=self.is_admin,
        )


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned KB tool compatibility facade."""
    from ...core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


def _snapshot_uploaded_file_record(record: Any) -> UploadedFileSnapshot:
    return UploadedFileSnapshot(
        user_id=int(record.user_id),
        file_id=str(record.file_id),
        filename=str(record.filename),
        storage_path=str(record.storage_path),
        storage_key=getattr(record, "storage_key", None),
        storage_status=getattr(record, "storage_status", None),
        checksum=getattr(record, "checksum", None),
    )


async def _create_knowledge_base_from_file_impl(
    args: Mapping[str, Any],
    *,
    user_id: int,
    is_admin: bool = False,
) -> Any:
    try:
        from sqlalchemy.orm import Session

        from .....web.models.database import get_db
        from .....web.models.uploaded_file import UploadedFile
        from .....web.services.managed_file_ref import (
            DurableStorageOperationError,
            ensure_uploaded_file_local_path,
        )
        from ...core.RAG_tools.core.schemas import (
            DEFAULT_EMBEDDING_MODEL_ID,
            IngestionConfig,
        )
        from ...core.RAG_tools.pipelines.document_ingestion import (
            run_document_ingestion,
        )
        from .agent_kb_service import AgentKnowledgeBaseService

        tool_args = CreateKnowledgeBaseFromFileArgs.model_validate(args)

        db_gen = get_db()
        db: Session = next(db_gen)

        try:
            query = db.query(UploadedFile).filter(
                UploadedFile.file_id.in_(tool_args.file_ids)
            )
            if not is_admin:
                query = query.filter(UploadedFile.user_id == user_id)
            file_records = [
                _snapshot_uploaded_file_record(record) for record in query.all()
            ]
        finally:
            db_gen.close()

        if not file_records:
            return CreateKnowledgeBaseFromFileResult(
                success=False,
                collection_name="",
                message=f"No files found for the provided file_ids: {tool_args.file_ids}",
                files_ingested=0,
            ).model_dump()

        if tool_args.collection_name:
            collection_name = tool_args.collection_name
        else:
            base_name = re.sub(
                r"[^a-zA-Z0-9_-]",
                "_",
                Path(file_records[0].filename).stem,
            )[:30]
            collection_name = f"{base_name}_{int(time.time())}"

        config = IngestionConfig(embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID)
        kb_service = AgentKnowledgeBaseService(
            user_id=user_id,
            is_admin=is_admin,
        )
        collection_name = await kb_service.prepare_collection(
            collection_name=collection_name,
            ingestion_config=config,
        )

        ingested_count = 0
        errors = []

        for record in file_records:
            try:
                source_path = ensure_uploaded_file_local_path(record)
            except DurableStorageOperationError as exc:
                errors.append(
                    f"Failed to restore {record.filename} from durable storage: {exc}"
                )
                continue
            if not source_path.exists():
                errors.append(
                    f"File not found on disk: {record.filename} (file_id={record.file_id})"
                )
                continue

            loop = asyncio.get_running_loop()
            func = partial(
                run_document_ingestion,
                collection=collection_name,
                source_path=str(source_path),
                ingestion_config=config,
                user_id=user_id,
                is_admin=is_admin,
                file_id=str(record.file_id),
            )
            try:
                request_context = copy_context()
                result = await loop.run_in_executor(None, request_context.run, func)
            except Exception as exc:
                errors.append(
                    f"Failed to ingest {record.filename} due to unexpected error: {exc}"
                )
                logger.exception(
                    "Unexpected error ingesting file %s",
                    record.filename,
                )
                continue

            if result.status == "error":
                errors.append(f"Failed to ingest {record.filename}: {result.message}")
            else:
                ingested_count += 1
                logger.info(
                    "Ingested file %s into collection %s",
                    record.filename,
                    collection_name,
                )

        if ingested_count == 0:
            return CreateKnowledgeBaseFromFileResult(
                success=False,
                collection_name=collection_name,
                message=f"Failed to ingest any files. Errors: {'; '.join(errors)}",
                files_ingested=0,
            ).model_dump()

        message = (
            f"Successfully created knowledge base '{collection_name}' "
            f"with {ingested_count} file(s)."
        )
        if errors:
            message += f" Warnings: {'; '.join(errors)}"

        await kb_service.refresh_collection_metadata(collection_name)

        return CreateKnowledgeBaseFromFileResult(
            success=True,
            collection_name=collection_name,
            message=message,
            files_ingested=ingested_count,
        ).model_dump()

    except Exception as e:
        logger.exception("Error creating knowledge base from file: %s", e)
        return CreateKnowledgeBaseFromFileResult(
            success=False,
            collection_name="",
            message=str(e),
            files_ingested=0,
        ).model_dump()


@register_tool(categories={"knowledge"})
async def create_file_ingestion_tools(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Create file ingestion tools."""
    return await _get_tool_compatibility_facade().create_file_ingestion_tools(config)


async def _create_file_ingestion_tools_impl(
    config: WebToolConfig,
) -> list[AbstractBaseTool]:
    """Create file ingestion tools."""
    try:
        user_id = config.get_user_id()
        is_admin = config.is_admin()
        if not user_id:
            return []

        tool = CreateKnowledgeBaseFromFileTool(
            user_id=user_id,
            is_admin=is_admin,
        )
        logger.debug("Created CreateKnowledgeBaseFromFileTool for user %s", user_id)
        return [tool]
    except Exception as e:
        logger.warning("Failed to create CreateKnowledgeBaseFromFileTool: %s", e)
        return []
