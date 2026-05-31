"""API key services for SDK runtime and management credentials."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import NamedTuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.utils.api_key import ApiKeyKind, generate_api_key
from ..models.agent_api_key import AgentApiKey
from ..models.user_api_key import UserApiKey
from ..schemas.agent_api_key import (
    APIKeyGenerateResponse,
    APIKeyMetadataResponse,
    APIKeyRevokeResponse,
)
from ..schemas.user_api_key import (
    PersonalAPIKeyCreateResponse,
    PersonalAPIKeyMetadata,
    PersonalAPIKeyRevokeResponse,
)

logger = logging.getLogger(__name__)


class KeyRotationConflict(RuntimeError):
    """Raised when a concurrent key rotation wins the active-key race."""


class AgentApiKeyService:
    """Owns agent runtime API key rotation and metadata."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def stage_rotated_key(self, agent_id: int) -> tuple[AgentApiKey, str]:
        """Revoke the active key (if any) and stage a new one, then flush.

        Does NOT commit -- the caller owns the transaction boundary. This
        is the composable building block (mirror of
        ``AgentStore.add_agent``): single-step callers go through
        :meth:`rotate_key` which commits, while multi-step workflows
        (create-agent-with-key) stage this plus other writes and commit
        once at the outer boundary.

        Returns the staged ORM row and the one-shot plaintext key. The
        row's ``created_at`` is only populated after the caller commits
        and refreshes.
        """
        now = datetime.now(timezone.utc)
        existing = (
            self.db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .first()
        )
        if existing is not None:
            existing.revoked_at = now  # type: ignore[assignment]
            existing.updated_at = now  # type: ignore[assignment]

        full_key, key_prefix, key_hash = generate_api_key(
            self.db, kind=ApiKeyKind.AGENT
        )
        new_row = AgentApiKey(
            agent_id=agent_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
        )
        self.db.add(new_row)
        self.db.flush()
        logger.info(
            "Staged runtime API key for agent %s (prefix=%s, rotated=%s)",
            agent_id,
            key_prefix,
            existing is not None,
        )
        return new_row, full_key

    def rotate_key(self, agent_id: int) -> APIKeyGenerateResponse:
        """Single-step rotate: stage + commit. Used by the JWT-gated
        ``/api/agents/{id}/api-key`` endpoint, which owns its own
        transaction: returns a one-shot key, commits itself, and maps a
        unique-index race to :class:`KeyRotationConflict`.

        Staging and commit share one ``try`` so the
        ``uq_agent_api_keys_agent_active`` / ``key_prefix`` conflict is
        translated whether it surfaces at the staging flush or at commit.
        """
        try:
            new_row, full_key = self.stage_rotated_key(agent_id)
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise KeyRotationConflict(str(exc)) from exc

        self.db.refresh(new_row)
        return APIKeyGenerateResponse(
            full_key=full_key,
            key_prefix=new_row.key_prefix,
            created_at=new_row.created_at,
        )

    def get_metadata(self, agent_id: int) -> APIKeyMetadataResponse | None:
        row = (
            self.db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .first()
        )
        if row is None:
            return None
        return APIKeyMetadataResponse(
            key_prefix=row.key_prefix,
            masked_key=f"xag_{row.key_prefix}_••••••••",
            created_at=row.created_at,
        )

    def revoke_key(self, agent_id: int) -> APIKeyRevokeResponse:
        now = datetime.now(timezone.utc)
        updated_count = (
            self.db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .update(
                {AgentApiKey.revoked_at: now, AgentApiKey.updated_at: now},
                synchronize_session=False,
            )
        )
        if updated_count == 0:
            return APIKeyRevokeResponse(revoked=False, revoked_at=None)

        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        logger.info("Revoked runtime API key for agent %s", agent_id)
        return APIKeyRevokeResponse(revoked=True, revoked_at=now)


class PersonalKeySecret(NamedTuple):
    full_key: str
    key_prefix: str
    key_hash: str


class UserApiKeyService:
    """Owns personal management API keys."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_key(self, user_id: int) -> PersonalAPIKeyCreateResponse:
        full_key, key_prefix, key_hash = generate_api_key(
            self.db, kind=ApiKeyKind.PERSONAL
        )
        row = UserApiKey(
            user_id=user_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
        )
        self.db.add(row)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(row)
        logger.info(
            "Created personal API key for user %s (prefix=%s)", user_id, key_prefix
        )
        return PersonalAPIKeyCreateResponse(
            id=int(row.id),
            full_key=full_key,
            key_prefix=row.key_prefix,
            created_at=row.created_at,
            expires_at=row.expires_at,
        )

    def list_keys(self, user_id: int) -> list[PersonalAPIKeyMetadata]:
        rows = (
            self.db.query(UserApiKey)
            .filter(UserApiKey.user_id == user_id)
            .order_by(UserApiKey.created_at.desc())
            .all()
        )
        return [
            PersonalAPIKeyMetadata(
                id=int(row.id),
                key_prefix=row.key_prefix,
                masked_key=f"xag_personal_{row.key_prefix}_••••••••",
                revoked_at=row.revoked_at,
                expires_at=row.expires_at,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def revoke_key(self, user_id: int, key_id: int) -> PersonalAPIKeyRevokeResponse:
        row = (
            self.db.query(UserApiKey)
            .filter(UserApiKey.id == key_id, UserApiKey.user_id == user_id)
            .first()
        )
        if row is None:
            return PersonalAPIKeyRevokeResponse(revoked=False, revoked_at=None)
        if row.revoked_at is not None:
            return PersonalAPIKeyRevokeResponse(
                revoked=False, revoked_at=row.revoked_at
            )

        now = datetime.now(timezone.utc)
        row.revoked_at = now
        row.updated_at = now
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(row)
        logger.info("Revoked personal API key %s for user %s", key_id, user_id)
        return PersonalAPIKeyRevokeResponse(revoked=True, revoked_at=row.revoked_at)
