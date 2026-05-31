"""SDK management endpoints for user-owned agents."""

from typing import Tuple

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from ...models.agent import Agent
from ...models.database import get_db
from ...models.user import User
from ...models.user_api_key import UserApiKey
from ...schemas.agent_api_key import APIKeyGenerateResponse
from ...schemas.v1 import (
    RuntimeKeyResponse,
    V1AgentCreateRequest,
    V1AgentCreateResponse,
    V1AgentResponse,
    V1AgentSummary,
    V1AgentTemplateCreateRequest,
)
from ...services.agent_management import (
    AgentManagementService,
    DuplicateAgentNameError,
    InvalidAgentModelConfigError,
    InvalidKnowledgeBaseError,
    TemplateNotFoundError,
)
from ...services.api_keys import KeyRotationConflict
from .deps import get_user_from_personal_key
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/agents")


def _runtime_key_response(api_key: APIKeyGenerateResponse) -> RuntimeKeyResponse:
    return RuntimeKeyResponse(
        full_key=api_key.full_key,
        key_prefix=api_key.key_prefix,
        created_at=api_key.created_at,
    )


def _agent_response(service: AgentManagementService, agent: Agent) -> V1AgentResponse:
    return V1AgentResponse.model_validate(service.store.agent_to_response_dict(agent))


@router.get("", response_model=list[V1AgentSummary])
async def list_agents(
    authed: Tuple[User, UserApiKey] = Depends(get_user_from_personal_key),
    db: Session = Depends(get_db),
) -> list[V1AgentSummary]:
    user, _key = authed
    service = AgentManagementService(db)
    return [
        V1AgentSummary.model_validate(item)
        for item in service.list_agents_for_user(int(user.id))
    ]


@router.post("", response_model=V1AgentCreateResponse)
async def create_agent(
    request: V1AgentCreateRequest,
    authed: Tuple[User, UserApiKey] = Depends(get_user_from_personal_key),
    db: Session = Depends(get_db),
) -> V1AgentCreateResponse:
    user, _key = authed
    service = AgentManagementService(db)
    try:
        # Atomic: KB validation + agent row + optional first runtime key,
        # all behind the single async create entry point.
        agent, api_key = await service.create_agent(
            user_id=int(user.id),
            is_admin=bool(user.is_admin),
            name=request.name,
            description=request.description,
            instructions=request.instructions,
            execution_mode=request.execution_mode,
            models=request.models,
            knowledge_bases=request.knowledge_bases,
            skills=request.skills,
            tool_categories=request.tool_categories,
            suggested_prompts=request.suggested_prompts,
            generate_runtime_key=request.generate_runtime_key,
        )
    except DuplicateAgentNameError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 400, "Agent with this name already exists."
        )
    except InvalidAgentModelConfigError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "Agent models must use integer DB model ids for known model slots.",
        )
    except InvalidKnowledgeBaseError as e:
        raise V1ApiError(V1ErrorCode.INVALID_INPUT, 400, str(e))
    except KeyRotationConflict:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR, 409, "Runtime key rotation conflict."
        )

    return V1AgentCreateResponse(
        agent=_agent_response(service, agent),
        api_key=_runtime_key_response(api_key) if api_key else None,
    )


@router.post("/from-template", response_model=V1AgentCreateResponse)
async def create_agent_from_template(
    request: V1AgentTemplateCreateRequest,
    fastapi_request: Request,
    authed: Tuple[User, UserApiKey] = Depends(get_user_from_personal_key),
    db: Session = Depends(get_db),
) -> V1AgentCreateResponse:
    user, _key = authed
    template_manager = getattr(fastapi_request.app.state, "template_manager", None)
    service = AgentManagementService(db, template_manager=template_manager)
    try:
        # Atomic: template-derived agent + optional first runtime key
        # commit together, same boundary as the plain create path.
        agent, api_key = await service.create_agent_from_template(
            user_id=int(user.id),
            is_admin=bool(user.is_admin),
            template_id=request.template_id,
            name=request.name,
            description=request.description,
            instructions=request.instructions,
            execution_mode=request.execution_mode,
            models=request.models,
            knowledge_bases=request.knowledge_bases,
            skills=request.skills,
            tool_categories=request.tool_categories,
            suggested_prompts=request.suggested_prompts,
            generate_runtime_key=request.generate_runtime_key,
        )
    except TemplateNotFoundError:
        raise V1ApiError(V1ErrorCode.TEMPLATE_NOT_FOUND, 404)
    except DuplicateAgentNameError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 400, "Agent with this name already exists."
        )
    except InvalidAgentModelConfigError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "Agent models must use integer DB model ids for known model slots.",
        )
    except InvalidKnowledgeBaseError as e:
        raise V1ApiError(V1ErrorCode.INVALID_INPUT, 400, str(e))
    except KeyRotationConflict:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR, 409, "Runtime key rotation conflict."
        )

    return V1AgentCreateResponse(
        agent=_agent_response(service, agent),
        api_key=_runtime_key_response(api_key) if api_key else None,
    )


@router.post("/{agent_id}/api-key", response_model=RuntimeKeyResponse)
async def rotate_agent_runtime_key(
    agent_id: int,
    authed: Tuple[User, UserApiKey] = Depends(get_user_from_personal_key),
    db: Session = Depends(get_db),
) -> RuntimeKeyResponse:
    user, _key = authed
    service = AgentManagementService(db)
    try:
        api_key = service.generate_agent_runtime_key(
            user_id=int(user.id), agent_id=agent_id
        )
    except KeyRotationConflict:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR, 409, "Runtime key rotation conflict."
        )
    if api_key is None:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
    return _runtime_key_response(api_key)
