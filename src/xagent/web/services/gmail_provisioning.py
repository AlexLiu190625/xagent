"""Per-mailbox Gmail Pub/Sub provisioning state machine.

Each connected Gmail mailbox that backs an enabled Gmail trigger gets its own
deterministic Pub/Sub topic and push subscription plus a Gmail watch. The
watch state row records an observable pending/active/failed status with a
clear last_error, converging through idempotent re-registration, periodic
sweeps, and reference-counted teardown.

Google credentials come from Application Default Credentials (ADC) or
GOOGLE_APPLICATION_CREDENTIALS; no xagent-specific credential store exists.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import (
    get_gmail_pubsub_project_id,
    get_gmail_pubsub_push_service_account,
    get_gmail_pubsub_subscription_prefix,
    get_gmail_pubsub_topic_prefix,
    get_gmail_registration_timeout_seconds,
    get_public_api_base_url,
)
from ..models.gmail_watch import GmailWatchState
from ..models.trigger import (
    AgentTrigger,
    TriggerProvisioningStatus,
    TriggerType,
)
from ..models.user_oauth import UserOAuth

logger = logging.getLogger(__name__)

GMAIL_PUSH_PUBLISHER = "gmail-api-push@system.gserviceaccount.com"
GMAIL_WATCH_LABEL_IDS = ["INBOX"]

PublisherFactory = Callable[[], Any]
SubscriberFactory = Callable[[], Any]


class GmailProvisioningError(RuntimeError):
    """Raised when per-mailbox Gmail provisioning cannot proceed."""


def _default_publisher() -> Any:
    from google.cloud import pubsub_v1  # type: ignore[import-untyped]

    return pubsub_v1.PublisherClient()


def _default_subscriber() -> Any:
    from google.cloud import pubsub_v1

    return pubsub_v1.SubscriberClient()


def _default_gmail_service(db: Session, oauth_account: UserOAuth) -> Any:
    from .gmail_triggers import build_gmail_service

    return build_gmail_service(db, oauth_account)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def mailbox_slug(email: str) -> str:
    """Deterministic, resource-name-safe identifier for one mailbox."""
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:16]


def gmail_topic_path(project_id: str, email: str) -> str:
    return f"projects/{project_id}/topics/{get_gmail_pubsub_topic_prefix()}-{mailbox_slug(email)}"


def gmail_subscription_path(project_id: str, email: str) -> str:
    return (
        f"projects/{project_id}/subscriptions/"
        f"{get_gmail_pubsub_subscription_prefix()}-{mailbox_slug(email)}"
    )


def _new_callback_id() -> str:
    return secrets.token_urlsafe(24)


def _is_already_exists(exc: Exception) -> bool:
    try:
        from google.api_core.exceptions import AlreadyExists

        return isinstance(exc, AlreadyExists)
    except ImportError:  # pragma: no cover - google libs are a core dep
        return type(exc).__name__ == "AlreadyExists"


def _is_not_found(exc: Exception) -> bool:
    try:
        from google.api_core.exceptions import NotFound

        return isinstance(exc, NotFound)
    except ImportError:  # pragma: no cover
        return type(exc).__name__ == "NotFound"


def _validate_provisioning_config() -> tuple[str, str, str]:
    """Return (project_id, public_base_url, push_service_account) or raise."""
    project_id = get_gmail_pubsub_project_id()
    if not project_id:
        raise GmailProvisioningError(
            "XAGENT_GMAIL_PUBSUB_PROJECT_ID is required for Gmail provisioning"
        )
    base_url = get_public_api_base_url()
    if not base_url:
        raise GmailProvisioningError(
            "XAGENT_PUBLIC_API_BASE_URL is required for Gmail push registration"
        )
    push_service_account = get_gmail_pubsub_push_service_account()
    if not push_service_account:
        raise GmailProvisioningError(
            "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT is required for "
            "OIDC-verified Gmail push delivery"
        )
    return project_id, base_url, push_service_account


def _get_or_create_watch_state(
    db: Session, oauth_account: UserOAuth, email: str
) -> GmailWatchState:
    # Locked so create/reconcile serializes against the FOR UPDATE taken by
    # release_gmail_mailbox_if_unused: without it, unregistering the last
    # trigger can delete the row a concurrent provisioning run is updating,
    # stranding the new trigger at PENDING until the sweep retries.
    def locked_state() -> GmailWatchState | None:
        return (
            db.query(GmailWatchState)
            .filter(GmailWatchState.oauth_account_id == int(oauth_account.id))
            .with_for_update()
            .first()
        )

    def mark_pending(target: GmailWatchState) -> None:
        if not target.callback_id:
            setattr(target, "callback_id", _new_callback_id())
        setattr(target, "email", email)
        setattr(target, "status", TriggerProvisioningStatus.PENDING.value)

    state = locked_state()
    if state is None:
        state = GmailWatchState(
            user_id=int(oauth_account.user_id),
            oauth_account_id=int(oauth_account.id),
            email=email,
            history_id="",
            topic_name="",
        )
        db.add(state)
    mark_pending(state)
    try:
        db.commit()
    except IntegrityError:
        # FOR UPDATE takes no lock when the row does not exist yet, so two
        # concurrent first-time enables for the same account can both reach
        # the insert path; the loser trips the oauth_account_id unique
        # constraint and adopts the winner's row instead of erroring out.
        db.rollback()
        adopted = locked_state()
        if adopted is None:  # pragma: no cover - row deleted between retries
            raise
        state = adopted
        mark_pending(state)
        db.commit()
    db.refresh(state)
    return state


def _ensure_topic(publisher: Any, topic_path: str) -> None:
    try:
        publisher.create_topic(request={"name": topic_path})
    except Exception as exc:
        if not _is_already_exists(exc):
            raise
    # Gmail publishes watch notifications as this Google-owned identity.
    try:
        policy = publisher.get_iam_policy(request={"resource": topic_path})
        member = f"serviceAccount:{GMAIL_PUSH_PUBLISHER}"
        for binding in policy.bindings:
            if binding.role == "roles/pubsub.publisher" and member in binding.members:
                return
        policy.bindings.add(role="roles/pubsub.publisher", members=[member])
        publisher.set_iam_policy(request={"resource": topic_path, "policy": policy})
    except Exception as exc:
        logger.warning(
            "Could not verify Gmail publish permission on %s: %s", topic_path, exc
        )


def _ensure_push_subscription(
    subscriber: Any,
    *,
    subscription_path: str,
    topic_path: str,
    push_audience: str,
    push_service_account: str,
) -> None:
    push_config = {
        "push_endpoint": push_audience,
        "oidc_token": {
            "service_account_email": push_service_account,
            "audience": push_audience,
        },
    }
    try:
        subscriber.create_subscription(
            request={
                "name": subscription_path,
                "topic": topic_path,
                "push_config": push_config,
            }
        )
    except Exception as exc:
        if not _is_already_exists(exc):
            raise
        # The deterministic name survives config changes; make sure an
        # existing subscription still pushes to the current audience
        # (e.g. after XAGENT_PUBLIC_API_BASE_URL was changed).
        _sync_push_endpoint(
            subscriber,
            subscription_path=subscription_path,
            push_config=push_config,
        )


def _sync_push_endpoint(
    subscriber: Any,
    *,
    subscription_path: str,
    push_config: dict[str, Any],
) -> None:
    try:
        existing = subscriber.get_subscription(
            request={"subscription": subscription_path}
        )
    except Exception as exc:
        logger.warning(
            "Could not inspect existing subscription %s: %s", subscription_path, exc
        )
        return
    current_endpoint = str(
        getattr(getattr(existing, "push_config", None), "push_endpoint", "") or ""
    )
    if current_endpoint == push_config["push_endpoint"]:
        return
    subscriber.modify_push_config(
        request={
            "subscription": subscription_path,
            "push_config": push_config,
        }
    )


def _register_gmail_watch(service: Any, topic_path: str) -> tuple[str, datetime | None]:
    response = (
        service.users()
        .watch(
            userId="me",
            body={"topicName": topic_path, "labelIds": GMAIL_WATCH_LABEL_IDS},
        )
        .execute()
    )
    history_id = response.get("historyId")
    if history_id is None:
        raise GmailProvisioningError("Gmail watch response did not include historyId")
    expiration = response.get("expiration")
    watch_expiration: datetime | None = None
    if expiration not in (None, ""):
        try:
            watch_expiration = datetime.fromtimestamp(
                int(str(expiration)) / 1000, tz=timezone.utc
            )
        except (TypeError, ValueError, OSError):
            watch_expiration = None
    return str(history_id), watch_expiration


def ensure_gmail_mailbox_provisioned(
    db: Session,
    oauth_account: UserOAuth,
    *,
    service_factory: Callable[[Session, UserOAuth], Any] | None = None,
    publisher_factory: PublisherFactory | None = None,
    subscriber_factory: SubscriberFactory | None = None,
) -> GmailWatchState:
    """Idempotently provision Pub/Sub resources and a Gmail watch for a mailbox.

    Never raises for provisioning failures: the watch state converges to
    failed with a clear last_error, and later reconcile attempts retry.
    """
    service_factory = service_factory or _default_gmail_service
    publisher_factory = publisher_factory or _default_publisher
    subscriber_factory = subscriber_factory or _default_subscriber
    email = str(oauth_account.email or "").strip().lower()
    if not email:
        raise GmailProvisioningError("Gmail account email is required")

    state = _get_or_create_watch_state(db, oauth_account, email)
    state_id = int(state.id)
    try:
        project_id, base_url, push_service_account = _validate_provisioning_config()
        topic_path = gmail_topic_path(project_id, email)
        subscription_path = gmail_subscription_path(project_id, email)
        push_audience = f"{base_url}/api/triggers/callback/gmail/{state.callback_id}"

        publisher = publisher_factory()
        _ensure_topic(publisher, topic_path)
        _ensure_push_subscription(
            subscriber_factory(),
            subscription_path=subscription_path,
            topic_path=topic_path,
            push_audience=push_audience,
            push_service_account=push_service_account,
        )
        service = service_factory(db, oauth_account)
        history_id, watch_expiration = _register_gmail_watch(service, topic_path)
    except Exception as exc:
        db.rollback()
        state = db.query(GmailWatchState).filter(GmailWatchState.id == state_id).one()
        setattr(state, "status", TriggerProvisioningStatus.FAILED.value)
        setattr(state, "last_error", str(exc))
        db.add(state)
        db.commit()
        db.refresh(state)
        logger.warning("Gmail provisioning failed for %s: %s", email, exc)
        return state

    setattr(state, "topic_name", topic_path)
    setattr(state, "subscription_name", subscription_path)
    setattr(state, "push_audience", push_audience)
    setattr(state, "history_id", history_id)
    setattr(state, "watch_expiration", watch_expiration)
    setattr(state, "status", TriggerProvisioningStatus.ACTIVE.value)
    setattr(state, "last_error", None)
    db.add(state)
    db.commit()
    db.refresh(state)
    return state


def _provision_in_fresh_session(oauth_account_id: int) -> None:
    from ..models.database import get_session_local

    db = get_session_local()()
    try:
        oauth_account = (
            db.query(UserOAuth).filter(UserOAuth.id == oauth_account_id).first()
        )
        if oauth_account is None:
            return
        ensure_gmail_mailbox_provisioned(db, oauth_account)
    except Exception:
        logger.exception(
            "Background Gmail provisioning failed for account %s", oauth_account_id
        )
    finally:
        db.close()


def provision_gmail_trigger(
    db: Session,
    trigger: AgentTrigger,
    *,
    timeout_seconds: int | None = None,
    run_in_thread: Callable[[int], threading.Thread] | None = None,
) -> str:
    """Provision the mailbox bound to a Gmail trigger; reflect status on it.

    Runs provisioning in a background thread and waits up to the configured
    registration timeout. When the cloud side is slow, the API observes a
    pending state while the thread converges to active or failed on its own.
    Returns the trigger provisioning status.
    """
    config: dict[str, Any] = trigger.config if isinstance(trigger.config, dict) else {}
    oauth_account_id = config.get("oauth_account_id")
    if oauth_account_id is None:
        status = TriggerProvisioningStatus.FAILED.value
        setattr(trigger, "provisioning_status", status)
        setattr(
            trigger, "provisioning_error", "Gmail trigger has no bound OAuth account"
        )
        db.add(trigger)
        db.commit()
        return status

    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else get_gmail_registration_timeout_seconds()
    )

    if run_in_thread is None:

        def run_in_thread(account_id: int) -> threading.Thread:
            thread = threading.Thread(
                target=_provision_in_fresh_session,
                args=(account_id,),
                daemon=True,
                name=f"gmail-provision-{account_id}",
            )
            thread.start()
            return thread

    thread = run_in_thread(int(oauth_account_id))
    thread.join(timeout)

    db.expire_all()
    state = (
        db.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(oauth_account_id))
        .first()
    )
    if thread.is_alive() or state is None:
        status = TriggerProvisioningStatus.PENDING.value
        error: str | None = None
    else:
        status = str(state.status or TriggerProvisioningStatus.PENDING.value)
        last_error = getattr(state, "last_error", None)
        error = str(last_error) if last_error else None
    setattr(trigger, "provisioning_status", status)
    setattr(trigger, "provisioning_error", error)
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return status


def reconcile_gmail_trigger_provisioning(
    db: Session,
    triggers: Sequence[AgentTrigger] | None = None,
    *,
    batch_size: int = 100,
) -> int:
    """Refresh Gmail triggers' provisioning status from their watch states.

    The background provisioning thread and the periodic sweep converge
    GmailWatchState without touching AgentTrigger, so the status the API
    reports would otherwise stay frozen at whatever the original synchronous
    create/update call observed. Each batch joins its enabled Gmail triggers
    to their mailboxes' watch states with one IN() lookup and copies
    status/last_error over when they diverge. Returns the number of triggers
    updated.

    Without an explicit trigger set (the sweep path), candidates are walked
    in id-keyset pages of ``batch_size`` so every query stays bounded no
    matter how many Gmail triggers exist system-wide.
    """
    if triggers is not None:
        return _reconcile_gmail_trigger_batch(
            db,
            [
                trigger
                for trigger in triggers
                if str(trigger.type) == TriggerType.GMAIL.value
                and bool(trigger.enabled)
                and trigger.resource_id
            ],
        )

    page_size = max(1, batch_size)
    updated = 0
    last_id = 0
    while True:
        page = (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.type == TriggerType.GMAIL.value,
                AgentTrigger.enabled.is_(True),
                AgentTrigger.resource_id.isnot(None),
                AgentTrigger.id > last_id,
            )
            .order_by(AgentTrigger.id.asc())
            .limit(page_size)
            .all()
        )
        if not page:
            return updated
        last_id = int(page[-1].id)
        updated += _reconcile_gmail_trigger_batch(db, page)
        if len(page) < page_size:
            return updated


def _reconcile_gmail_trigger_batch(
    db: Session, candidates: Sequence[AgentTrigger]
) -> int:
    """Copy diverged watch-state status onto one bounded candidate batch."""
    if not candidates:
        return 0

    emails = {str(trigger.resource_id).strip().lower() for trigger in candidates}
    states = (
        db.query(GmailWatchState)
        .filter(func.lower(GmailWatchState.email).in_(emails))
        .all()
    )
    states_by_key = {
        (int(state.user_id), str(state.email or "").strip().lower()): state
        for state in states
    }

    updated = 0
    for trigger in candidates:
        key = (int(trigger.user_id), str(trigger.resource_id).strip().lower())
        state = states_by_key.get(key)
        if state is None:
            continue
        status = str(state.status or TriggerProvisioningStatus.PENDING.value)
        last_error = getattr(state, "last_error", None)
        error = str(last_error) if last_error else None
        if (
            str(trigger.provisioning_status or "") == status
            and (trigger.provisioning_error or None) == error
        ):
            continue
        setattr(trigger, "provisioning_status", status)
        setattr(trigger, "provisioning_error", error)
        db.add(trigger)
        updated += 1
    if updated:
        db.commit()
    return updated


def release_gmail_mailbox_if_unused(
    db: Session,
    oauth_account_id: int,
    *,
    service_factory: Callable[[Session, UserOAuth], Any] | None = None,
    publisher_factory: PublisherFactory | None = None,
    subscriber_factory: SubscriberFactory | None = None,
) -> bool:
    """Reference-counted teardown of one mailbox's delivery resources.

    Locks the watch state row, counts remaining enabled Gmail triggers bound
    to the mailbox, and only when none remain stops the Gmail watch and
    deletes the per-mailbox subscription, topic, and watch state.
    Returns True when resources were released.
    """
    service_factory = service_factory or _default_gmail_service
    publisher_factory = publisher_factory or _default_publisher
    subscriber_factory = subscriber_factory or _default_subscriber
    state = (
        db.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(oauth_account_id))
        .with_for_update()
        .first()
    )
    if state is None:
        db.commit()
        return False

    email = str(state.email or "").strip().lower()
    remaining = (
        db.query(AgentTrigger.id)
        .filter(
            AgentTrigger.type == TriggerType.GMAIL.value,
            AgentTrigger.enabled.is_(True),
            func.lower(AgentTrigger.resource_id) == email,
        )
        .count()
    )
    if remaining > 0:
        db.commit()
        return False

    oauth_account = (
        db.query(UserOAuth).filter(UserOAuth.id == int(oauth_account_id)).first()
    )
    if oauth_account is not None:
        try:
            service = service_factory(db, oauth_account)
            service.users().stop(userId="me").execute()
        except Exception as exc:
            logger.warning("Failed to stop Gmail watch for %s: %s", email, exc)

    project_id = get_gmail_pubsub_project_id()
    if project_id:
        subscription_path = str(
            state.subscription_name or gmail_subscription_path(project_id, email)
        )
        topic_path = str(state.topic_name or gmail_topic_path(project_id, email))
        try:
            subscriber_factory().delete_subscription(
                request={"subscription": subscription_path}
            )
        except Exception as exc:
            if not _is_not_found(exc):
                logger.warning(
                    "Failed to delete subscription %s: %s", subscription_path, exc
                )
        try:
            publisher_factory().delete_topic(request={"topic": topic_path})
        except Exception as exc:
            if not _is_not_found(exc):
                logger.warning("Failed to delete topic %s: %s", topic_path, exc)

    db.delete(state)
    db.commit()
    return True


def sweep_gmail_provisioning(
    db: Session,
    *,
    now: datetime | None = None,
    stale_pending_seconds: int = 300,
    limit: int = 100,
    service_factory: Callable[[Session, UserOAuth], Any] | None = None,
    publisher_factory: PublisherFactory | None = None,
    subscriber_factory: SubscriberFactory | None = None,
) -> int:
    """Retry stale pending and failed Gmail registrations.

    Only mailboxes still referenced by an enabled Gmail trigger are retried.
    Returns the number of registration attempts.
    """
    scan_time = now or _now()
    stale_before = scan_time - timedelta(seconds=stale_pending_seconds)
    candidates = (
        db.query(GmailWatchState)
        .filter(
            (GmailWatchState.status == TriggerProvisioningStatus.FAILED.value)
            | (
                (GmailWatchState.status == TriggerProvisioningStatus.PENDING.value)
                & (GmailWatchState.updated_at <= stale_before)
            )
        )
        .order_by(GmailWatchState.updated_at.asc())
        .limit(max(1, min(limit, 500)))
        .all()
    )

    attempts = 0
    for state in candidates:
        email = str(state.email or "").strip().lower()
        referenced = (
            db.query(AgentTrigger.id)
            .filter(
                AgentTrigger.type == TriggerType.GMAIL.value,
                AgentTrigger.enabled.is_(True),
                func.lower(AgentTrigger.resource_id) == email,
            )
            .first()
            is not None
        )
        if not referenced:
            continue
        oauth_account = (
            db.query(UserOAuth)
            .filter(UserOAuth.id == int(state.oauth_account_id))
            .first()
        )
        if oauth_account is None:
            continue
        ensure_gmail_mailbox_provisioned(
            db,
            oauth_account,
            service_factory=service_factory,
            publisher_factory=publisher_factory,
            subscriber_factory=subscriber_factory,
        )
        attempts += 1
    # Watch states that converged in a background thread (pending -> active)
    # are not sweep candidates, so the trigger-facing status is reconciled
    # here unconditionally, paged by the sweep's own limit.
    reconcile_gmail_trigger_provisioning(db, batch_size=max(1, min(limit, 500)))
    return attempts


def best_effort_provision_gmail_watches_for_user(
    db: Session,
    *,
    user_id: int,
    context: str,
) -> None:
    """Provision watches for a user's Gmail accounts referenced by triggers.

    Used after OAuth (re)connects a Gmail account: any enabled Gmail trigger
    already bound to that mailbox gets its delivery resources re-ensured.
    Failures are recorded on the watch state, never raised.
    """
    accounts = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == int(user_id), UserOAuth.provider == "gmail")
        .all()
    )
    for account in accounts:
        email = str(account.email or "").strip().lower()
        if not email:
            continue
        referenced = (
            db.query(AgentTrigger.id)
            .filter(
                AgentTrigger.type == TriggerType.GMAIL.value,
                AgentTrigger.enabled.is_(True),
                func.lower(AgentTrigger.resource_id) == email,
            )
            .first()
            is not None
        )
        if not referenced:
            continue
        try:
            ensure_gmail_mailbox_provisioned(db, account)
        except Exception as exc:
            db.rollback()
            logger.warning(
                "Failed to provision Gmail watch for %s %s: %s",
                email,
                context,
                exc,
                exc_info=True,
            )
