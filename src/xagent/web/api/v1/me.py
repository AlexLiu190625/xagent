"""GET /v1/me -- personal management key identity probe."""

from typing import Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...models.user import User
from ...models.user_api_key import UserApiKey
from .deps import get_user_from_personal_key

router = APIRouter()


class MeResponse(BaseModel):
    """Response model for ``GET /v1/me``."""

    principal_type: str = Field(default="user", description="Authenticated principal.")
    user_id: int = Field(..., description="User bound to the presented personal key.")
    email: str = Field(..., description="User email or username.")
    name: str = Field(..., description="User display name.")
    key_prefix: str = Field(
        ...,
        description=(
            "Public-safe 6-char lookup handle of the presented key. "
            "Lets the SDK log which key is in use without exposing the secret."
        ),
    )


@router.get("/me", response_model=MeResponse)
async def get_me(
    authed: Tuple[User, UserApiKey] = Depends(get_user_from_personal_key),
) -> MeResponse:
    """Probe the user identity bound to the caller's personal key."""
    user, key = authed
    username = str(user.username)
    return MeResponse(
        principal_type="user",
        user_id=int(user.id),
        email=username,
        name=username,
        key_prefix=key.key_prefix,
    )
