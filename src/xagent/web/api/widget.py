"""Web Widget API route handlers."""

from datetime import timedelta
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from ..models.agent import Agent, is_workforce_generated_manager_agent
from ..models.database import get_db
from ..models.user import User
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from .auth import create_access_token
from .public_chat_access import (
    PublicChatAccessContext,
    PublicChatAuthResponse,
    build_public_chat_dependency,
    create_public_chat_access_token,
    create_public_chat_task,
    public_chat_websocket_endpoint,
    upload_public_chat_files,
)

widget_router = APIRouter(prefix="/api/widget", tags=["widget"])

EMBED_TICKET_TYPE = "widget_embed_ticket"
EMBED_TICKET_TTL_SECONDS = 60


class WidgetAuthRequest(BaseModel):
    guest_id: str = Field(max_length=256)
    agent_id: Optional[int] = None
    # A signed embed ticket is a compact JWT; cap it to reject pathological
    # payloads, matching the length limits on sibling request models.
    embed_ticket: Optional[str] = Field(default=None, max_length=4096)


class EmbedTicketRequest(BaseModel):
    agent_id: int


class EmbedTicketResponse(BaseModel):
    ticket: str


WidgetAuthResponse = PublicChatAuthResponse


def _origin_to_domain(origin: str) -> str:
    """Extract a lowercased host[:port] from an origin/referer value."""
    if not origin:
        return ""
    parsed = urlparse(origin)
    return (parsed.netloc or parsed.path).lower()


def _domain_allowed(origin_domain: str, allowed_domains: list[str]) -> bool:
    """Check a domain against the agent allowlist (case-insensitive,
    supports "*" and subdomain suffix matches)."""
    for domain in allowed_domains:
        normalized_domain = domain.strip().lower()
        if (
            normalized_domain == "*"
            or normalized_domain == origin_domain
            or (origin_domain and origin_domain.endswith("." + normalized_domain))
        ):
            return True
    return False


def _require_domain_allowed(origin_domain: str, allowed_domains: list[str]) -> None:
    """Raise 403 unless the domain passes the agent allowlist."""
    if not _domain_allowed(origin_domain, allowed_domains):
        raise HTTPException(
            status_code=403, detail=f"Domain not allowed: {origin_domain}"
        )


def _get_widget_enabled_agent(db: Session, agent_id: int) -> Agent:
    """Load a widget-enabled agent or raise the matching HTTP error."""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None or is_workforce_generated_manager_agent(agent):
        raise HTTPException(
            status_code=401, detail="Widget owner not found or invalid agent_id"
        )
    if not agent.widget_enabled:
        raise HTTPException(status_code=403, detail="Widget is disabled for this agent")
    return agent


@widget_router.post("/embed-ticket", response_model=EmbedTicketResponse)
async def issue_widget_embed_ticket(
    request: EmbedTicketRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> Any:
    """Issue a short-lived signed embed ticket to the embedding page.

    This endpoint is called by widget.js from the top-level embedding page,
    so the browser-enforced Origin header carries the real embedding site —
    unlike fetches from inside the widget iframe, whose Origin is the xagent
    host itself. The signed ticket is the only way that validated origin is
    trusted downstream; the widget never self-reports its parent's origin.
    """
    agent = _get_widget_enabled_agent(db, request.agent_id)

    allowed_domains: list[str] = agent.allowed_domains or []  # type: ignore
    origin = req.headers.get("origin") or req.headers.get("referer", "")
    origin_domain = _origin_to_domain(origin)
    _require_domain_allowed(origin_domain, allowed_domains)

    # The ticket has no jti/nonce and is intentionally replayable within its
    # short TTL: it only re-certifies "this origin is allowed", which /auth
    # independently re-checks against the live allowlist on every use, and the
    # guest tokens it mints are low-privilege. Replay-within-TTL is accepted.
    ticket = create_access_token(
        {
            "type": EMBED_TICKET_TYPE,
            "agent_id": int(agent.id),
            "embed_origin": origin_domain,
        },
        expires_delta=timedelta(seconds=EMBED_TICKET_TTL_SECONDS),
    )
    return EmbedTicketResponse(ticket=ticket)


@widget_router.post("/auth", response_model=WidgetAuthResponse)
async def authenticate_widget(
    request: WidgetAuthRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> Any:
    """Authenticate widget and issue a guest token"""
    agent_id = request.agent_id
    user = None
    target_channel = None
    agent = None

    # Authenticate via agent_id directly (since web widget channel is deprecated)
    origin_domain = ""
    if agent_id:
        agent = _get_widget_enabled_agent(db, agent_id)

        allowed_domains: list[str] = agent.allowed_domains or []  # type: ignore

        if request.embed_ticket:
            # The auth fetch runs inside the widget iframe, so its
            # Origin/Referer headers reflect the xagent host, not the
            # embedding site. The embedding page's origin is instead carried
            # by the backend-signed embed ticket issued by /embed-ticket,
            # where it was validated against the browser-enforced Origin
            # header. A client-supplied origin value is never trusted here.
            try:
                claims = jwt.decode(
                    request.embed_ticket,
                    JWT_SECRET_KEY,
                    algorithms=[JWT_ALGORITHM],
                )
            except JWTError:
                raise HTTPException(
                    status_code=403, detail="Invalid or expired embed ticket"
                )
            if (
                claims.get("type") != EMBED_TICKET_TYPE
                or claims.get("agent_id") != agent_id
            ):
                raise HTTPException(
                    status_code=403, detail="Invalid or expired embed ticket"
                )
            origin_domain = str(claims.get("embed_origin") or "")
            # Re-check so tickets die immediately if the allowlist shrinks.
            _require_domain_allowed(origin_domain, allowed_domains)
        else:
            # No ticket: direct page visits (not embedded via widget.js).
            # Fall back to the request's own Origin/Referer headers.
            origin = req.headers.get("origin") or req.headers.get("referer", "")
            origin_domain = _origin_to_domain(origin)
            _require_domain_allowed(origin_domain, allowed_domains)

        user = db.query(User).filter(User.id == agent.user_id).first()

    if not user:
        raise HTTPException(
            status_code=401, detail="Widget owner not found or invalid agent_id"
        )

    # Get agent name if available
    agent_name = None
    agent_logo = None
    if agent:
        agent_name = agent.name
        agent_logo = agent.logo_url

    channel_id = target_channel.id if target_channel else None

    access_token = create_public_chat_access_token(
        {
            "sub": user.username,
            "user_id": user.id,
            "channel_id": channel_id,
            "guest_id": request.guest_id,
            "auth_mode": "widget",
            "widget_agent_id": int(agent.id) if agent else None,
        }
    )

    return WidgetAuthResponse(
        access_token=access_token,
        agent_id=agent_id,
        agent_name=agent_name,
        agent_logo=agent_logo,
        agent_description=agent.description if agent else None,
        suggested_prompts=agent.suggested_prompts or [] if agent else [],
    )


get_current_widget_user_dep = build_public_chat_dependency("widget")


@widget_router.post("/files/upload")
async def upload_widget_file(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    task_type: str = Form(...),
    message: str = Form(""),
    task_id: str = Form(None),
    folder: str = Form(None),
    widget_info: PublicChatAccessContext = Depends(get_current_widget_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    return await upload_public_chat_files(
        file=file,
        files=files,
        task_type=task_type,
        message=message,
        task_id=task_id,
        folder=folder,
        access_context=widget_info,
        db=db,
    )


@widget_router.post("/chat/task/create", response_model=TaskCreateResponse)
async def create_widget_task(
    request: TaskCreateRequest,
    widget_info: PublicChatAccessContext = Depends(get_current_widget_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    """Create new chat task for widget guest."""
    return await create_public_chat_task(
        request=request,
        access_context=widget_info,
        db=db,
        default_channel_name="Web Widget",
    )


@widget_router.websocket("/chat/ws/{task_id}")
async def websocket_widget_chat_endpoint(
    websocket: WebSocket,
    task_id: int,
    token: str = Query(..., description="Authentication token"),
) -> None:
    """WebSocket unified endpoint for widget."""
    await public_chat_websocket_endpoint(
        websocket=websocket,
        task_id=task_id,
        token=token,
        expected_auth_mode="widget",
    )
