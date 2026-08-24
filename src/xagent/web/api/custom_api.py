"""
Custom API Management API Endpoints

Provides REST API endpoints for managing Custom API configurations
in the web application.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import (
    ConnectorRuntimeError,
    validate_runtime_config_declaration,
)
from ...core.utils.encryption import encrypt_value
from ..auth_dependencies import get_current_user
from ..models.custom_api import CustomApi, UserCustomApi
from ..models.database import get_db
from ..models.user import User

if TYPE_CHECKING:
    from ..services.connector_team_scope import ConnectorAccess
    from .mcp import _TeamOwnedUserApi

logger = logging.getLogger(__name__)


# Pydantic models for API
class CustomApiCreate(BaseModel):
    """Request model for creating a Custom API."""

    name: str = Field(..., min_length=1, max_length=100, description="API name")
    description: Optional[str] = Field(None, description="API description")
    url: Optional[str] = Field(
        None, min_length=1, max_length=500, description="API URL"
    )
    method: Optional[str] = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    body: Optional[str] = Field(None, description="HTTP body (JSON template)")
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (secrets)"
    )
    runtime_input_schema: Optional[Dict[str, Any]] = Field(
        None, description="Runtime input declarations"
    )
    runtime_bindings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Runtime binding declarations"
    )
    allow_delegated_authorization: bool = Field(
        False, description="Allow runtime Authorization header binding"
    )
    is_active: bool = Field(True, description="Whether the API is active")


class CustomApiUpdate(BaseModel):
    """Request model for updating a Custom API."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="API name"
    )
    description: Optional[str] = Field(None, description="API description")
    url: Optional[str] = Field(
        None, min_length=1, max_length=500, description="API URL"
    )
    method: Optional[str] = Field(None, description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    body: Optional[str] = Field(None, description="HTTP body (JSON template)")
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (secrets)"
    )
    runtime_input_schema: Optional[Dict[str, Any]] = Field(
        None, description="Runtime input declarations"
    )
    runtime_bindings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Runtime binding declarations"
    )
    allow_delegated_authorization: Optional[bool] = Field(
        None, description="Allow runtime Authorization header binding"
    )
    is_active: Optional[bool] = Field(None, description="Whether the API is active")


class CustomApiResponse(BaseModel):
    """Response model for Custom API."""

    id: int
    user_id: int
    name: str
    description: Optional[str]
    url: Optional[str]
    method: Optional[str]
    headers: Optional[Dict[str, str]]
    body: Optional[str]
    env: Optional[Dict[str, str]]  # Will return masked values
    runtime_input_schema: Optional[Dict[str, Any]]
    runtime_bindings: Optional[List[Dict[str, Any]]]
    allow_delegated_authorization: bool
    is_active: bool
    is_default: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


# Create router
custom_api_router = APIRouter(prefix="/api/custom-apis", tags=["Custom API Management"])


def _db_api_to_response(
    api: CustomApi,
    user_api: "UserCustomApi | _TeamOwnedUserApi",
) -> CustomApiResponse:
    """Convert database CustomApi to response model with masked env values."""

    # Mask env values for frontend
    masked_env = None
    if api.env and isinstance(api.env, dict):
        masked_env = {k: "********" for k in api.env.keys()}

    return CustomApiResponse(
        id=api.id,
        user_id=user_api.user_id,
        name=api.name,
        description=api.description,
        url=api.url,
        method=api.method,
        headers=api.headers,
        body=api.body,
        env=masked_env,
        runtime_input_schema=api.runtime_input_schema,
        runtime_bindings=api.runtime_bindings,
        allow_delegated_authorization=bool(api.allow_delegated_authorization),
        is_active=user_api.is_active,
        is_default=user_api.is_default,
        created_at=str(api.created_at.isoformat()),
        updated_at=str(api.updated_at.isoformat()),
    )


def _process_env_vars(
    env: Optional[Dict[str, str]], existing_env: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, str]]:
    """Encrypt environment variables, retaining masks only for the same key."""
    if not env:
        return env

    encrypted_env = {}
    existing_env = existing_env or {}

    for k, v in env.items():
        if v == "********":
            # Retain existing encrypted value if masked
            if k in existing_env:
                encrypted_env[k] = existing_env[k]
            else:
                raise ValueError(
                    f"Masked secret '{k}' has no stored value; provide a new value"
                )
        else:
            encrypted_env[k] = encrypt_value(v)

    return encrypted_env


@custom_api_router.get("", response_model=List[CustomApiResponse])
async def list_custom_apis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[CustomApiResponse]:
    """List all Custom APIs for the current user."""
    user_apis = (
        db.query(UserCustomApi).filter(UserCustomApi.user_id == current_user.id).all()
    )

    responses = []
    for user_api in user_apis:
        if user_api.custom_api:
            responses.append(_db_api_to_response(user_api.custom_api, user_api))

    return responses


@custom_api_router.post(
    "", response_model=CustomApiResponse, status_code=status.HTTP_201_CREATED
)
async def create_custom_api(
    api_data: CustomApiCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Create a new Custom API."""

    # Check if name already exists
    existing = db.query(CustomApi).filter(CustomApi.name == api_data.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Custom API with name '{api_data.name}' already exists",
        )

    # A masked value is a same-key retention token, never a transferable secret.
    try:
        encrypted_env = _process_env_vars(api_data.env)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid environment variables: {exc}",
        ) from exc
    try:
        validate_runtime_config_declaration(
            connector_type="custom_api",
            runtime_input_schema=api_data.runtime_input_schema,
            runtime_bindings=api_data.runtime_bindings,
            allow_delegated_authorization=api_data.allow_delegated_authorization,
            static_headers=api_data.headers,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid runtime configuration: {exc}",
        ) from exc

    # Create CustomApi
    new_api = CustomApi(
        name=api_data.name,
        description=api_data.description,
        url=api_data.url,
        method=api_data.method,
        headers=api_data.headers,
        body=api_data.body,
        env=encrypted_env,
        runtime_input_schema=api_data.runtime_input_schema,
        runtime_bindings=api_data.runtime_bindings,
        allow_delegated_authorization=api_data.allow_delegated_authorization,
    )

    db.add(new_api)
    db.flush()

    # Create UserCustomApi link
    user_api = UserCustomApi(
        user_id=current_user.id,
        custom_api_id=new_api.id,
        is_owner=True,
        can_edit=True,
        can_delete=True,
        is_active=api_data.is_active,
    )

    db.add(user_api)

    db.commit()
    db.refresh(new_api)
    db.refresh(user_api)

    return _db_api_to_response(new_api, user_api)


def _resolve_custom_api_for_request(
    db: Session,
    user_id: int,
    api_id: int,
    *,
    skip_resolution_when: "Callable[[UserCustomApi], bool] | None" = None,
) -> "tuple[UserCustomApi | _TeamOwnedUserApi, CustomApi, ConnectorAccess | None]":
    """Resolve the caller's association, the definition row, and the
    caller's team access verdict, for ``GET``/``PUT /api/custom-apis/{id}``.

    Looks up the caller's own personal link row first, with the same query
    both routes have always run. When that row exists and its ``custom_api``
    relationship resolves, the association and the definition row both come
    from it and nothing else runs. When it does not -- no row, or a row
    whose relationship is unexpectedly empty -- the definition row is
    looked up on its own -- a team-owned API's shared row must still be
    found even though this caller has no personal link to it -- and the
    caller's team access verdict decides what happens next:

    - no working personal row and no team access (``access is None``) ->
      404, the same outcome every caller without an association has
      always gotten.
    - no working personal row but the caller's team links the API -> the
      existing ``_TeamOwnedUserApi`` stand-in takes the association's
      place, the same stand-in the aggregate connector list already
      constructs for this case.

    ``skip_resolution_when`` lets a caller declare when its own working
    personal row already decides the answer on its own, so resolving a
    verdict would only add an unnecessary hook call: ``get_custom_api``
    passes a predicate that is always true, because it never reads the
    verdict at all and a personal row -- owner or not -- already decides
    what it returns; ``update_custom_api`` passes one that checks
    ``can_edit``, because only an owner's ``can_edit=True`` decides the
    edit answer on its own -- a non-owner's ``can_edit=False`` personal row
    does not, since a granting team verdict can still widen it. Left
    unset (the default), resolution is never skipped, which is what a
    caller with no working personal row always needs -- the verdict is the
    gate there and must stay fail-closed.

    Raises ``ConnectorRuntimeError`` when access resolution itself fails;
    callers translate that into an ``HTTPException``.
    """
    from ..services.connector_team_scope import resolve_connector_access_or_raise
    from .mcp import _TeamOwnedUserApi

    user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == user_id,
        )
        .first()
    )
    if user_api is not None and user_api.custom_api is not None:
        api: Optional[CustomApi] = user_api.custom_api
    else:
        user_api = None
        api = db.query(CustomApi).filter(CustomApi.id == api_id).first()

    already_decided = user_api is not None and (
        skip_resolution_when is not None and skip_resolution_when(user_api)
    )

    access: "ConnectorAccess | None" = None
    if api is not None and not already_decided:
        access = resolve_connector_access_or_raise(
            db, int(user_id), "custom_api", int(api.id)
        )

    if user_api is None and access is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )

    resolved_user_api: "UserCustomApi | _TeamOwnedUserApi" = (
        user_api if user_api is not None else _TeamOwnedUserApi(int(user_id))
    )
    return resolved_user_api, cast(CustomApi, api), access


@custom_api_router.get("/{api_id}", response_model=CustomApiResponse)
async def get_custom_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Get a specific Custom API by ID."""

    try:
        # This route never reads the verdict at all (see _db_api_to_response),
        # so a working personal row -- owner or not -- always already
        # decides everything this route returns; resolving one would only
        # add an unnecessary hook call.
        user_api, api, _team_access = _resolve_custom_api_for_request(
            db,
            int(current_user.id),
            api_id,
            skip_resolution_when=lambda _user_api: True,
        )
    except ConnectorRuntimeError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.safe_message
        ) from exc

    return _db_api_to_response(api, user_api)


@custom_api_router.put("/{api_id}", response_model=CustomApiResponse)
async def update_custom_api(
    api_id: int,
    api_data: CustomApiUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Update an existing Custom API."""

    try:
        # An owner's can_edit=True already decides the edit answer on its
        # own (below), so resolving a verdict for that row would only add
        # an unnecessary hook call; a non-owner's can_edit=False personal
        # row does not decide it, since a granting team verdict can still
        # widen it.
        user_api, api, team_access = _resolve_custom_api_for_request(
            db,
            int(current_user.id),
            api_id,
            skip_resolution_when=lambda ua: bool(ua.can_edit),
        )
    except ConnectorRuntimeError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.safe_message
        ) from exc

    is_stand_in = not isinstance(user_api, UserCustomApi)
    can_edit = bool(user_api.can_edit) or bool(
        team_access is not None and team_access.can_edit
    )
    if not can_edit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to edit this Custom API",
        )

    # is_active lives on the personal association row; a caller with no
    # personal row (the stand-in) has none to hold it, so a payload
    # carrying it must be rejected outright -- writing it onto the
    # stand-in would only set a shadowing instance attribute that
    # persists nothing, and the response below would then read that
    # shadow back and report a change that never happened.
    if is_stand_in and api_data.is_active is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No personal connection exists to configure is_active for this API",
        )

    old_name = str(api.name)
    # The row's declared type from here on is loosened for mypy's sake: the
    # column-typed attributes below (name, description, env, ...) are all
    # mutated directly by this route, exactly as before this gate existed.
    mutable_api = cast(Any, api)

    # Check name uniqueness if name is changed
    if api_data.name and api_data.name != api.name:
        existing = db.query(CustomApi).filter(CustomApi.name == api_data.name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Custom API with name '{api_data.name}' already exists",
            )
        mutable_api.name = api_data.name

    # Update fields
    if api_data.description is not None:
        mutable_api.description = api_data.description
    if api_data.url is not None:
        mutable_api.url = api_data.url
    if api_data.method is not None:
        mutable_api.method = api_data.method
    if api_data.headers is not None:
        mutable_api.headers = api_data.headers
    if api_data.body is not None:
        mutable_api.body = api_data.body

    # Process env variables
    if api_data.env is not None:
        existing_env: Dict[str, str] = (
            mutable_api.env if isinstance(api.env, dict) else {}
        )
        try:
            processed_env = _process_env_vars(api_data.env, existing_env)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid environment variables: {exc}",
            ) from exc
        mutable_api.env = processed_env

    fields_set = api_data.model_fields_set
    runtime_input_schema = (
        api_data.runtime_input_schema
        if "runtime_input_schema" in fields_set
        else api.runtime_input_schema
    )
    runtime_bindings = (
        api_data.runtime_bindings
        if "runtime_bindings" in fields_set
        else api.runtime_bindings
    )
    allow_delegated_authorization = (
        bool(api_data.allow_delegated_authorization)
        if "allow_delegated_authorization" in fields_set
        else bool(api.allow_delegated_authorization)
    )
    try:
        validate_runtime_config_declaration(
            connector_type="custom_api",
            runtime_input_schema=runtime_input_schema,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=allow_delegated_authorization,
            static_headers=mutable_api.headers,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid runtime configuration: {exc}",
        ) from exc
    if "runtime_input_schema" in fields_set:
        mutable_api.runtime_input_schema = runtime_input_schema
    if "runtime_bindings" in fields_set:
        mutable_api.runtime_bindings = runtime_bindings
    if "allow_delegated_authorization" in fields_set:
        mutable_api.allow_delegated_authorization = allow_delegated_authorization

    from ..services.connector_team_scope import rename_team_connector

    rename_team_connector(
        db,
        int(current_user.id),
        "custom_api",
        int(api_id),
        old_name,
        str(api.name),
    )

    # Update UserCustomApi link
    if api_data.is_active is not None:
        user_api.is_active = api_data.is_active

    db.commit()
    db.refresh(api)

    return _db_api_to_response(api, user_api)


@custom_api_router.delete("/{api_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete a Custom API."""

    user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == current_user.id,
        )
        .first()
    )

    if not user_api or not user_api.custom_api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )

    if not user_api.can_delete:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this Custom API",
        )

    api = user_api.custom_api

    from ..services.connector_team_scope import delete_team_connector

    team_delete = delete_team_connector(
        db, int(current_user.id), "custom_api", int(api_id)
    )
    if team_delete.blocked_reason:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=team_delete.blocked_reason,
        )
    if team_delete.team_owned and not team_delete.authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only a team admin can delete a team Custom API",
        )
    if team_delete.team_owned:
        db.delete(user_api)
        db.flush([user_api])
        with db.no_autoflush:
            remaining = (
                db.query(UserCustomApi)
                .filter(UserCustomApi.custom_api_id == api_id)
                .first()
            )
        if remaining is None and team_delete.delete_definition:
            db.delete(api)
    else:
        db.delete(api)  # Will cascade to UserCustomApi
    db.commit()

    return None
