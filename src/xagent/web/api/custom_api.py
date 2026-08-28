"""
Custom API Management API Endpoints

Provides REST API endpoints for managing Custom API configurations
in the web application.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import (
    validate_runtime_config_declaration,
)
from ...core.utils.encryption import encrypt_value
from ..auth_dependencies import get_current_user
from ..models.custom_api import CustomApi, UserCustomApi
from ..models.database import get_db
from ..models.user import User

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
    user_api: UserCustomApi,
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


@custom_api_router.get("/{api_id}", response_model=CustomApiResponse)
async def get_custom_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Get a specific Custom API by ID."""

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

    return _db_api_to_response(user_api.custom_api, user_api)


@custom_api_router.put("/{api_id}", response_model=CustomApiResponse)
def update_custom_api(
    api_id: int,
    api_data: CustomApiUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Update an existing Custom API."""

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

    if not user_api.can_edit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to edit this Custom API",
        )

    # A second, single-table lock on the definition row, taken before any
    # field below reads or mutates it. The read above comes through the
    # personal link row's relationship and cannot itself lock just this
    # table; this is a fresh statement, so a row deleted between the two
    # still yields None here (handled as the same 404) rather than
    # surfacing as an unrelated error out of the write path below.
    # ``populate_existing()`` makes the locked row the one the rest of this
    # route reads: without it the already-identity-mapped instance the
    # relationship loaded would be returned unrefreshed, and every field
    # below would still be the pre-lock snapshot.
    locked_api = (
        db.query(CustomApi)
        .filter(CustomApi.id == api_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if locked_api is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )
    api = locked_api

    # Read only after the lock: rename_team_connector's "old" argument must
    # be the name this transaction actually holds locked, not whatever the
    # pre-lock read above saw -- a concurrent committed rename in between
    # would otherwise make this stale, and the rewrite below would look for
    # a name that no longer exists anywhere, leaving the previous renamer's
    # selectors dangling with no error.
    old_name = str(api.name)

    # The row's declared type from here on is loosened for mypy's sake: the
    # column-typed attributes below (name, description, env, ...) are all
    # mutated directly by this route, exactly as they were when this local
    # came off the relationship instead of off the locked query.
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
        user_api.is_active = api_data.is_active  # type: ignore[assignment]

    db.commit()
    db.refresh(api)

    return _db_api_to_response(api, user_api)


@custom_api_router.delete("/{api_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_custom_api(
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

    # One global lock order over this pair of tables. ``update_custom_api``
    # locks the ``CustomApi`` definition row first and writes the
    # ``UserCustomApi`` link row afterwards; both branches below delete the
    # link row first and the definition row second, inside one transaction.
    # Without this statement the two routes take the same two rows in
    # opposite orders and a concurrent edit/delete pair can deadlock
    # (PostgreSQL 40P01). Taken after every refusal above, so a request that
    # is going to be refused never acquires the lock. ``populate_existing``
    # matches the PUT's own lock: the row this transaction holds is the one
    # the deletion below acts on, not whatever the relationship read above
    # happened to see.
    #
    # This statement orders THIS repository's two tables and nothing else. A
    # connector team hook writes its own tables, which this lock does not
    # cover, and the two routes reach it in opposite orders relative to this
    # lock: the PUT takes the lock above and calls rename_team_connector
    # afterwards, while this route calls delete_team_connector before taking
    # the lock at all. So an installing application whose hooks lock a row of
    # its own can still deadlock against a concurrent edit/delete pair on the
    # same connector, and no ordering statement inside this repository can
    # prevent that -- the hook side has to take its rows in an order
    # compatible with this one, and only the application can arrange that.
    locked_api = (
        db.query(CustomApi)
        .filter(CustomApi.id == api_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if locked_api is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )
    api = locked_api

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
