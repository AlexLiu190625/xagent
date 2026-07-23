"""Access helpers for the shared ``deployments`` table.

One deployment row per ``(owner_type, owner_id)`` holds the external-channel
opt-ins and credentials (share link, widget) for exposing an agent or a
workforce outside Xagent. Workforce channels store their state here from day
one; Agent's legacy flattened columns are read elsewhere until they migrate.
"""

from __future__ import annotations

import secrets

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.deployment import Deployment, DeploymentOwnerType


def new_share_token() -> str:
    return secrets.token_urlsafe(24)


def get_deployment(
    db: Session, owner_type: DeploymentOwnerType, owner_id: int
) -> Deployment | None:
    return (
        db.query(Deployment)
        .filter(
            Deployment.owner_type == owner_type.value,
            Deployment.owner_id == int(owner_id),
        )
        .first()
    )


def get_or_create_deployment(
    db: Session, owner_type: DeploymentOwnerType, owner_id: int
) -> Deployment:
    """Return the owner's deployment row, inserting (flush, no commit) if absent.

    The insert races against ``uq_deployment_owner``: two near-simultaneous
    callers (double-click, two tabs) both see no row and both try to insert.
    The loser's flush raises ``IntegrityError``; recover by rolling back and
    re-reading the winner's row so the caller resolves idempotently instead
    of surfacing a 500. Callers invoke this as the first mutation of their
    transaction, so the rollback discards nothing else.
    """
    deployment = get_deployment(db, owner_type, owner_id)
    if deployment is not None:
        return deployment
    deployment = Deployment(owner_type=owner_type.value, owner_id=int(owner_id))
    db.add(deployment)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existing = get_deployment(db, owner_type, owner_id)
        if existing is None:
            raise
        return existing
    return deployment


def find_enabled_share_deployment(
    db: Session, share_token: str, owner_type: DeploymentOwnerType
) -> Deployment | None:
    """Resolve a raw share token to an enabled deployment of the given type."""
    if not share_token:
        return None
    return (
        db.query(Deployment)
        .filter(
            Deployment.owner_type == owner_type.value,
            Deployment.share_token == share_token,
            Deployment.share_enabled.is_(True),
        )
        .first()
    )
