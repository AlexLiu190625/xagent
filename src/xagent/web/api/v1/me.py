"""GET /v1/me -- SDK auth probe.

Zero side-effect, read-only endpoint that returns the agent identity
bound to the presented API key. SDK clients call this once at startup
to verify their key is valid and to discover which agent they're
addressing.
"""

from typing import Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...models.agent import Agent
from ...models.agent_api_key import AgentApiKey
from .deps import get_agent_from_api_key

router = APIRouter()


class MeResponse(BaseModel):
    """Response model for ``GET /v1/me``."""

    agent_id: int = Field(..., description="The agent bound to the presented key.")
    agent_name: str = Field(..., description="Display name of the bound agent.")
    key_prefix: str = Field(
        ...,
        description=(
            "Public-safe 6-char lookup handle of the presented key. "
            "Lets the SDK log which key is in use without exposing the secret."
        ),
    )


@router.get("/me", response_model=MeResponse)
async def get_me(
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
) -> MeResponse:
    """Probe the agent identity bound to the caller's API key.

    Zero side-effect. SDK clients typically call this once on startup
    to confirm their key is valid and to log which agent they're
    targeting. The response shape is intentionally minimal -- richer
    agent metadata (description, models, etc) is the owner's view and
    lives behind JWT on ``/api/agents/{id}``.

    Args:
        authed: Resolved by ``get_agent_from_api_key``; the auth gate
            handles every failure path uniformly as 401.

    Returns:
        :class:`MeResponse` with ``agent_id``, ``agent_name``, and
        ``key_prefix``.

    Raises:
        V1ApiError 401: missing / malformed / unknown / revoked key.
            Translated to ``{"error": {"code": "invalid_api_key", ...}}``
            by the global exception handler.
    """
    agent, key = authed
    return MeResponse(
        agent_id=agent.id,
        agent_name=agent.name,
        key_prefix=key.key_prefix,
    )
