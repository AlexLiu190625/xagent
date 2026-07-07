from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerProvisioningStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.gmail_provisioning import (
    ensure_gmail_mailbox_provisioned,
    gmail_subscription_path,
    gmail_topic_path,
    reconcile_gmail_trigger_provisioning,
    release_gmail_mailbox_if_unused,
    sweep_gmail_provisioning,
)


class FakeExecutable:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {}

    def execute(self) -> dict[str, Any]:
        return self.response


class FakeGmailUsers:
    def __init__(self, service: FakeGmailService) -> None:
        self.service = service

    def watch(self, *, userId: str, body: dict[str, Any]) -> FakeExecutable:
        self.service.watch_calls.append({"userId": userId, "body": body})
        return FakeExecutable(
            {
                "historyId": self.service.history_id,
                "expiration": self.service.expiration,
            }
        )

    def stop(self, *, userId: str) -> FakeExecutable:
        self.service.stop_calls.append({"userId": userId})
        return FakeExecutable()


class FakeGmailService:
    def __init__(
        self, *, history_id: str = "hist-1", expiration: str = "4102444800000"
    ) -> None:
        self.history_id = history_id
        self.expiration = expiration
        self.watch_calls: list[dict[str, Any]] = []
        self.stop_calls: list[dict[str, Any]] = []

    def users(self) -> FakeGmailUsers:
        return FakeGmailUsers(self)


class FakeBinding:
    def __init__(self, *, role: str, members: list[str]) -> None:
        self.role = role
        self.members = members


class FakeBindings(list[FakeBinding]):
    def add(self, *, role: str, members: list[str]) -> FakeBinding:
        binding = FakeBinding(role=role, members=members)
        self.append(binding)
        return binding


class FakePolicy:
    def __init__(self) -> None:
        self.bindings = FakeBindings()


class FakePublisher:
    def __init__(self) -> None:
        self.topics: set[str] = set()
        self.policies: dict[str, FakePolicy] = {}
        self.deleted_topics: list[str] = []

    def create_topic(self, *, request: dict[str, str]) -> None:
        self.topics.add(request["name"])

    def get_iam_policy(self, *, request: dict[str, str]) -> FakePolicy:
        return self.policies.setdefault(request["resource"], FakePolicy())

    def set_iam_policy(self, *, request: dict[str, Any]) -> None:
        self.policies[request["resource"]] = request["policy"]

    def delete_topic(self, *, request: dict[str, str]) -> None:
        self.deleted_topics.append(request["topic"])
        self.topics.discard(request["topic"])


class FakeSubscriber:
    def __init__(self) -> None:
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.deleted_subscriptions: list[str] = []

    def create_subscription(self, *, request: dict[str, Any]) -> None:
        self.subscriptions[request["name"]] = request

    def delete_subscription(self, *, request: dict[str, str]) -> None:
        self.deleted_subscriptions.append(request["subscription"])
        self.subscriptions.pop(request["subscription"], None)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'gmail_provisioning.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture(autouse=True)
def gmail_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", "demo-project")
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX", "xagent-gmail")
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX", "xagent-gmail-push")
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.com/")
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "pubsub-push@demo-project.iam.gserviceaccount.com",
    )


def _create_user(db: Session) -> User:
    user = User(
        username="owner",
        email="owner@example.com",
        password_hash="hash",
        is_admin=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_agent(db: Session, user: User) -> Agent:
    agent = Agent(
        user_id=int(user.id),
        name="Gmail agent",
        description="test",
        instructions="test",
        status=AgentStatus.DRAFT,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _create_oauth(db: Session, user: User, *, email: str = "Owner@Gmail.Example"):
    account = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        access_token="access-token",
        email=email,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def _create_gmail_trigger(
    db: Session,
    user: User,
    agent: Agent,
    account: UserOAuth,
    *,
    enabled: bool = True,
) -> AgentTrigger:
    trigger = AgentTrigger(
        user_id=int(user.id),
        agent_id=int(agent.id),
        type=TriggerType.GMAIL.value,
        name="Gmail inbox",
        enabled=enabled,
        provider=TriggerType.GMAIL.value,
        resource_id=str(account.email).lower(),
        config={"watch_label": "INBOX", "oauth_account_id": int(account.id)},
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def test_provisioning_creates_deterministic_resources_and_active_state(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    email = "owner@gmail.example"
    expected_topic = gmail_topic_path("demo-project", email)
    expected_subscription = gmail_subscription_path("demo-project", email)
    expected_audience = (
        f"https://api.example.com/api/triggers/callback/gmail/{state.callback_id}"
    )
    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    assert state.last_error is None
    assert state.topic_name == expected_topic
    assert state.subscription_name == expected_subscription
    assert state.push_audience == expected_audience
    assert state.history_id == "hist-1"
    assert publisher.topics == {expected_topic}
    assert set(subscriber.subscriptions) == {expected_subscription}
    assert subscriber.subscriptions[expected_subscription]["push_config"] == {
        "push_endpoint": expected_audience,
        "oidc_token": {
            "service_account_email": "pubsub-push@demo-project.iam.gserviceaccount.com",
            "audience": expected_audience,
        },
    }
    assert gmail.watch_calls == [
        {
            "userId": "me",
            "body": {"topicName": expected_topic, "labelIds": ["INBOX"]},
        }
    ]


def test_missing_public_api_base_records_failed_state_without_app_base_fallback(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://frontend.example.com")
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: FakeSubscriber(),
    )

    assert state.status == TriggerProvisioningStatus.FAILED.value
    assert "XAGENT_PUBLIC_API_BASE_URL" in str(state.last_error)
    assert state.push_audience is None


def test_sweep_retries_stale_failed_referenced_mailbox(db_session: Session) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    state = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(account.id),
        email="owner@gmail.example",
        history_id="",
        topic_name="",
        status=TriggerProvisioningStatus.FAILED.value,
        last_error="old failure",
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    db_session.add(state)
    db_session.commit()
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService(history_id="hist-retry")

    attempts = sweep_gmail_provisioning(
        db_session,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    refreshed = (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(account.id))
        .one()
    )
    assert attempts == 1
    assert refreshed.status == TriggerProvisioningStatus.ACTIVE.value
    assert refreshed.history_id == "hist-retry"
    assert refreshed.last_error is None


def test_unregister_releases_mailbox_only_after_last_enabled_trigger_is_deleted(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    first = _create_gmail_trigger(db_session, user, agent, account)
    second = _create_gmail_trigger(db_session, user, agent, account)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is False
    )
    db_session.delete(first)
    db_session.commit()
    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is False
    )
    db_session.delete(second)
    db_session.commit()

    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is True
    )
    assert gmail.stop_calls == [{"userId": "me"}]
    assert subscriber.deleted_subscriptions == [state.subscription_name]
    assert publisher.deleted_topics == [state.topic_name]
    assert db_session.query(GmailWatchState).count() == 0


async def test_gmail_provider_register_unregister_offload_sync_sdk_work(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.services.trigger_providers.gmail import GmailProvider

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)
    calls: list[str] = []

    def fake_provision(_db: Session, provisioned_trigger: AgentTrigger) -> str:
        calls.append(f"provision:{provisioned_trigger.id}")
        setattr(
            provisioned_trigger,
            "provisioning_status",
            TriggerProvisioningStatus.ACTIVE.value,
        )
        setattr(provisioned_trigger, "provisioning_error", None)
        _db.add(provisioned_trigger)
        _db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release(_db: Session, oauth_account_id: int) -> bool:
        calls.append(f"release:{oauth_account_id}")
        return True

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append(f"to_thread:{fn.__name__}")
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.release_gmail_mailbox_if_unused",
        fake_release,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.asyncio.to_thread",
        fake_to_thread,
    )

    provider = GmailProvider()
    result = await provider.register(db_session, trigger, object())
    # unregister resolves the binding from config alone; the trigger row may
    # already be rebound or deleted when CRUD dispatches it.
    await provider.unregister(
        db_session, trigger, {"oauth_account_id": int(account.id)}
    )

    assert result.status == TriggerProvisioningStatus.ACTIVE
    assert calls == [
        "to_thread:fake_provision",
        f"provision:{trigger.id}",
        "to_thread:fake_release",
        f"release:{account.id}",
    ]


def test_slow_registration_returns_pending_then_reconciles_to_active(
    db_session: Session,
) -> None:
    """Slow cloud provisioning yields pending; the thread converges later."""
    import threading

    from xagent.web.services.gmail_provisioning import provision_gmail_trigger

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)

    release = threading.Event()
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService(history_id="hist-slow")
    threads: list[threading.Thread] = []

    def slow_provision(account_id: int) -> None:
        release.wait(timeout=10)
        from xagent.web.models.database import get_session_local

        db2 = get_session_local()()
        try:
            slow_account = db2.query(UserOAuth).filter(UserOAuth.id == account_id).one()
            ensure_gmail_mailbox_provisioned(
                db2,
                slow_account,
                service_factory=lambda _db, _account: gmail,
                publisher_factory=lambda: publisher,
                subscriber_factory=lambda: subscriber,
            )
        finally:
            db2.close()

    def run_in_thread(account_id: int) -> threading.Thread:
        thread = threading.Thread(target=slow_provision, args=(account_id,))
        thread.start()
        threads.append(thread)
        return thread

    status = provision_gmail_trigger(
        db_session,
        trigger,
        timeout_seconds=0,
        run_in_thread=run_in_thread,
    )
    assert status == TriggerProvisioningStatus.PENDING.value
    assert trigger.provisioning_status == TriggerProvisioningStatus.PENDING.value

    release.set()
    threads[0].join(timeout=10)
    assert not threads[0].is_alive()

    db_session.expire_all()
    state = (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(account.id))
        .one()
    )
    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    assert state.history_id == "hist-slow"

    # The periodic sweep - not another user-initiated create/update - must
    # surface the converged state on the trigger the API serves.
    attempts = sweep_gmail_provisioning(
        db_session,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    assert attempts == 0  # state is active; nothing to re-register
    db_session.refresh(trigger)
    assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value
    assert trigger.provisioning_error is None


def test_reconcile_copies_watch_state_status_onto_triggers(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)
    disabled = _create_gmail_trigger(db_session, user, agent, account, enabled=False)
    setattr(trigger, "provisioning_status", TriggerProvisioningStatus.PENDING.value)
    setattr(disabled, "provisioning_status", TriggerProvisioningStatus.PENDING.value)
    db_session.add_all([trigger, disabled])
    state = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(account.id),
        email="owner@gmail.example",
        history_id="hist-1",
        topic_name="projects/demo-project/topics/xagent-gmail-abc",
        status=TriggerProvisioningStatus.ACTIVE.value,
    )
    db_session.add(state)
    db_session.commit()

    updated = reconcile_gmail_trigger_provisioning(db_session)

    assert updated == 1
    db_session.refresh(trigger)
    db_session.refresh(disabled)
    assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value
    assert trigger.provisioning_error is None
    # Disabled triggers hold no watch reference; their status is not touched.
    assert disabled.provisioning_status == TriggerProvisioningStatus.PENDING.value

    # Failures propagate too, including the error message.
    setattr(state, "status", TriggerProvisioningStatus.FAILED.value)
    setattr(state, "last_error", "watch registration denied")
    db_session.add(state)
    db_session.commit()

    assert reconcile_gmail_trigger_provisioning(db_session) == 1
    db_session.refresh(trigger)
    assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
    assert trigger.provisioning_error == "watch registration denied"

    # Idempotent: nothing to update on a second pass.
    assert reconcile_gmail_trigger_provisioning(db_session) == 0


@pytest.mark.parametrize(
    ("candidate_count", "expected_page_queries"),
    [
        # Short final page (5 = 2+2+1): the len(page) < page_size branch
        # terminates without an extra query.
        (5, 3),
        # Exact multiple of the page size (4 = 2+2): termination needs one
        # extra empty-page query, taking the `if not page` branch.
        (4, 3),
    ],
)
def test_reconcile_full_scan_pages_candidates_with_bounded_queries(
    db_session: Session,
    candidate_count: int,
    expected_page_queries: int,
) -> None:
    """The sweep-path reconcile (triggers=None) walks candidates in keyset
    pages: every diverged trigger still reconciles, and every candidate
    query carries the page bound instead of scanning system-wide."""
    from sqlalchemy import event as sa_event

    from xagent.web.models.database import get_engine

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    triggers = []
    for index in range(candidate_count):
        account = _create_oauth(db_session, user, email=f"owner{index}@gmail.example")
        trigger = _create_gmail_trigger(db_session, user, agent, account)
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.PENDING.value)
        db_session.add(trigger)
        db_session.add(
            GmailWatchState(
                user_id=int(user.id),
                oauth_account_id=int(account.id),
                email=str(account.email).lower(),
                history_id=f"hist-{index}",
                topic_name=f"projects/demo-project/topics/xagent-gmail-{index}",
                status=TriggerProvisioningStatus.ACTIVE.value,
            )
        )
        triggers.append(trigger)
    db_session.commit()

    statements: list[str] = []

    def _track(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    engine = get_engine()
    sa_event.listen(engine, "before_cursor_execute", _track)
    try:
        updated = reconcile_gmail_trigger_provisioning(db_session, batch_size=2)
    finally:
        sa_event.remove(engine, "before_cursor_execute", _track)

    assert updated == candidate_count
    for trigger in triggers:
        db_session.refresh(trigger)
        assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value

    candidate_queries = [
        s for s in statements if "FROM agent_triggers" in s and "SELECT" in s
    ]
    assert candidate_queries, "expected candidate page queries"
    assert all("LIMIT" in s for s in candidate_queries)
    assert len(candidate_queries) == expected_page_queries


class ResyncFakeSubscriber(FakeSubscriber):
    """FakeSubscriber that behaves like Pub/Sub for existing subscriptions."""

    def __init__(self) -> None:
        super().__init__()
        self.modify_calls: list[dict[str, Any]] = []

    def create_subscription(self, *, request: dict[str, Any]) -> None:
        if request["name"] in self.subscriptions:
            from google.api_core.exceptions import AlreadyExists

            raise AlreadyExists("subscription exists")
        super().create_subscription(request=request)

    def get_subscription(self, *, request: dict[str, str]) -> Any:
        from types import SimpleNamespace

        stored = self.subscriptions[request["subscription"]]
        return SimpleNamespace(
            push_config=SimpleNamespace(
                push_endpoint=stored["push_config"]["push_endpoint"]
            )
        )

    def modify_push_config(self, *, request: dict[str, Any]) -> None:
        self.modify_calls.append(request)
        self.subscriptions[request["subscription"]]["push_config"] = request[
            "push_config"
        ]


def test_existing_subscription_endpoint_resyncs_after_base_url_change(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = ResyncFakeSubscriber()
    gmail = FakeGmailService()

    first = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    old_audience = str(first.push_audience)
    assert subscriber.modify_calls == []

    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api-v2.example.com")
    second = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    new_audience = (
        f"https://api-v2.example.com/api/triggers/callback/gmail/{second.callback_id}"
    )
    assert second.push_audience == new_audience
    assert new_audience != old_audience
    assert len(subscriber.modify_calls) == 1
    stored = subscriber.subscriptions[str(second.subscription_name)]
    assert stored["push_config"]["push_endpoint"] == new_audience
    assert stored["push_config"]["oidc_token"]["audience"] == new_audience


def test_renewal_scan_uses_per_mailbox_provisioning_when_project_configured(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Watch renewal must not point per-mailbox states back at the global topic."""
    from xagent.web.services import gmail_provisioning
    from xagent.web.services.gmail_triggers import scan_due_gmail_watch_renewals

    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_TOPIC", raising=False)

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    stale = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(account.id),
        email="owner@gmail.example",
        history_id="old",
        topic_name="projects/demo-project/topics/legacy-global",
        watch_expiration=datetime.now(timezone.utc) - timedelta(hours=1),
        status=TriggerProvisioningStatus.ACTIVE.value,
    )
    db_session.add(stale)
    db_session.commit()

    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    monkeypatch.setattr(gmail_provisioning, "_default_publisher", lambda: publisher)
    monkeypatch.setattr(gmail_provisioning, "_default_subscriber", lambda: subscriber)
    gmail = FakeGmailService(history_id="hist-renewed")

    renewed = scan_due_gmail_watch_renewals(
        db_session,
        service_factory=lambda _db, _account: gmail,
    )

    assert renewed == 1
    refreshed = (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(account.id))
        .one()
    )
    expected_topic = gmail_topic_path("demo-project", "owner@gmail.example")
    assert refreshed.topic_name == expected_topic
    assert refreshed.status == TriggerProvisioningStatus.ACTIVE.value
    assert refreshed.history_id == "hist-renewed"
    assert gmail.watch_calls[-1]["body"]["topicName"] == expected_topic


@pytest.fixture()
def pg_session():
    """Session against a real Postgres, where SELECT ... FOR UPDATE locks.

    SQLite silently no-ops row locks, so the provisioning/release contention
    path can only be exercised here. Set XAGENT_TEST_POSTGRES_URL to run
    (CI provides it in the PostgreSQL job).
    """
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_release_and_reprovision_contend_on_the_watch_state_lock(
    pg_session: Session,
) -> None:
    """Unregister-of-last-trigger and a concurrent provision serialize.

    While release_gmail_mailbox_if_unused holds the watch-state row lock,
    _get_or_create_watch_state must block instead of updating a row that is
    about to be deleted (which strands the new trigger at PENDING via
    StaleDataError on the losing commit).
    """
    from xagent.web.models.database import get_session_local

    db = pg_session
    user = _create_user(db)
    account = _create_oauth(db, user)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()
    state = ensure_gmail_mailbox_provisioned(
        db,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    old_callback_id = str(state.callback_id)
    account_id = int(account.id)

    release_holds_lock = threading.Event()
    release_may_finish = threading.Event()
    results: dict[str, Any] = {}

    def blocking_service_factory(_db: Session, _account: UserOAuth) -> FakeGmailService:
        # Called by release after it has taken FOR UPDATE on the state row.
        release_holds_lock.set()
        release_may_finish.wait(timeout=30)
        return gmail

    def do_release() -> None:
        db_a = get_session_local()()
        try:
            results["released"] = release_gmail_mailbox_if_unused(
                db_a,
                account_id,
                service_factory=blocking_service_factory,
                publisher_factory=lambda: publisher,
                subscriber_factory=lambda: subscriber,
            )
        finally:
            db_a.close()

    def do_provision() -> None:
        db_b = get_session_local()()
        try:
            account_b = db_b.query(UserOAuth).filter(UserOAuth.id == account_id).one()
            fresh = ensure_gmail_mailbox_provisioned(
                db_b,
                account_b,
                service_factory=lambda _db, _account: gmail,
                publisher_factory=lambda: publisher,
                subscriber_factory=lambda: subscriber,
            )
            results["status"] = str(fresh.status)
            results["callback_id"] = str(fresh.callback_id)
        finally:
            db_b.close()

    releaser = threading.Thread(target=do_release)
    releaser.start()
    assert release_holds_lock.wait(timeout=30)

    provisioner = threading.Thread(target=do_provision)
    provisioner.start()
    provisioner.join(timeout=1.0)
    # Provisioning is parked on the row lock, not racing the delete.
    assert provisioner.is_alive()

    release_may_finish.set()
    releaser.join(timeout=30)
    provisioner.join(timeout=30)
    assert not releaser.is_alive() and not provisioner.is_alive()

    assert results["released"] is True
    assert results["status"] == TriggerProvisioningStatus.ACTIVE.value
    # The mailbox was fully released first, then provisioned from scratch.
    assert results["callback_id"] != old_callback_id
    rows = (
        db.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == account_id)
        .all()
    )
    assert len(rows) == 1
    assert rows[0].status == TriggerProvisioningStatus.ACTIVE.value


def test_first_time_creation_race_adopts_winner_row(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FOR UPDATE takes no lock when the watch-state row does not exist yet,
    so two concurrent first-time enables can both reach the insert path. The
    loser's IntegrityError must adopt the winner's committed row instead of
    propagating a spurious error into the background thread."""
    from sqlalchemy.exc import IntegrityError

    from xagent.web.services.gmail_provisioning import _get_or_create_watch_state

    user = _create_user(db_session)
    account = _create_oauth(db_session, user)

    real_commit = db_session.commit
    real_rollback = db_session.rollback
    calls = {"count": 0}

    def racing_commit() -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            # Simulate a concurrent winner committing between this session's
            # empty FOR UPDATE select and its own insert commit.
            real_rollback()
            winner = GmailWatchState(
                user_id=int(user.id),
                oauth_account_id=int(account.id),
                email="owner@gmail.example",
                history_id="hist-winner",
                topic_name="projects/demo-project/topics/winner",
                callback_id="winner-callback-id",
                status=TriggerProvisioningStatus.ACTIVE.value,
            )
            db_session.add(winner)
            real_commit()
            raise IntegrityError(
                "UNIQUE constraint failed: gmail_watch_states.oauth_account_id",
                params=None,
                orig=Exception("simulated concurrent insert"),
            )
        real_commit()

    monkeypatch.setattr(db_session, "commit", racing_commit)
    state = _get_or_create_watch_state(db_session, account, "owner@gmail.example")

    assert state.callback_id == "winner-callback-id"
    assert state.history_id == "hist-winner"
    assert state.status == TriggerProvisioningStatus.PENDING.value
    assert db_session.query(GmailWatchState).count() == 1


def test_concurrent_first_time_creations_race_on_the_unique_constraint(
    pg_session: Session,
) -> None:
    """Two sessions both pass the empty FOR UPDATE select and insert; the
    loser must recover from the genuine unique-constraint violation by
    rolling back and adopting the winner's committed row. Unlike the mocked
    variant above, the loser's session really is in a failed transaction, so
    this fails if _get_or_create_watch_state drops its rollback."""
    from xagent.web.models.database import get_session_local
    from xagent.web.services.gmail_provisioning import _get_or_create_watch_state

    db = pg_session
    user = _create_user(db)
    account = _create_oauth(db, user)
    account_id = int(account.id)

    both_past_the_empty_select = threading.Barrier(2)
    results: dict[str, str] = {}
    errors: dict[str, BaseException] = {}

    def do_enable(name: str) -> None:
        session = get_session_local()()
        real_commit = session.commit
        insert_commit_pending = True

        def synchronized_commit() -> None:
            # Hold the insert commit until both sessions have run the empty
            # FOR UPDATE select, so both take the insert path; the loser's
            # adoption retry commit passes straight through.
            nonlocal insert_commit_pending
            if insert_commit_pending:
                insert_commit_pending = False
                both_past_the_empty_select.wait(timeout=30)
            real_commit()

        session.commit = synchronized_commit  # type: ignore[method-assign]
        try:
            acct = session.query(UserOAuth).filter(UserOAuth.id == account_id).one()
            state = _get_or_create_watch_state(session, acct, "owner@gmail.example")
            results[name] = str(state.callback_id)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors[name] = exc
        finally:
            session.close()

    threads = [threading.Thread(target=do_enable, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)

    assert errors == {}
    # The loser adopted the winner's row, so both report the same identity.
    assert results["a"] == results["b"]
    rows = (
        db.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == account_id)
        .all()
    )
    assert len(rows) == 1
    assert str(rows[0].callback_id) == results["a"]
    assert rows[0].status == TriggerProvisioningStatus.PENDING.value
    assert rows[0].email == "owner@gmail.example"


def test_provisioning_requires_account_email(db_session: Session) -> None:
    from xagent.web.services.gmail_provisioning import GmailProvisioningError

    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    setattr(account, "email", None)
    db_session.add(account)
    db_session.commit()

    with pytest.raises(GmailProvisioningError, match="email is required"):
        ensure_gmail_mailbox_provisioned(
            db_session,
            account,
            service_factory=lambda _db, _account: FakeGmailService(),
            publisher_factory=lambda: FakePublisher(),
            subscriber_factory=lambda: FakeSubscriber(),
        )
    assert db_session.query(GmailWatchState).count() == 0
