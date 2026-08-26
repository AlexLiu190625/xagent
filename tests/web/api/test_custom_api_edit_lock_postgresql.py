"""Real-PostgreSQL coverage for the row lock ``update_custom_api`` takes on
the ``CustomApi`` definition row before propagating a rename.

``FOR UPDATE`` is a no-op on SQLite -- every other suite in this repo runs
against SQLite, so nothing there can tell a genuine second-writer block
from a lock statement that silently does nothing. This file is the one
place that runs the real statement against a real server and proves it
actually blocks a second writer, plus the companion path where the row
vanishes between the route's first read and this lock. Mirrors
test_mcp_server_edit_lock_postgresql.py's structure for the MCP side of
the same lock.

Obtains its database through ``tests/shared/postgres_disposable.py``
(``disposable_database_factory``), the same disposable-CREATE-DATABASE
helper the other ``*_postgresql.py`` suites in this repo use, rather than
opening a hand-rolled connection. That helper reads
``XAGENT_TEST_POSTGRES_URL`` and skips the whole module when it is unset.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)

pytestmark = pytest.mark.postgresql


@pytest.fixture()
def session_factory():
    with disposable_database_factory("xagent_custom_api_edit_lock") as make_database:
        engine = make_database("edit_lock")
        Base.metadata.create_all(bind=engine)
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def seeded(session_factory):
    """One owner, one owned Custom API, in their own committed rows."""
    with session_factory() as db:
        owner = User(
            username="custom-api-edit-lock-owner", password_hash="x", is_admin=False
        )
        db.add(owner)
        db.flush()
        api = CustomApi(
            name="edit-lock-target",
            url="https://example.com/api",
            method="GET",
        )
        db.add(api)
        db.flush()
        db.add(
            UserCustomApi(
                user_id=int(owner.id),
                custom_api_id=int(api.id),
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
        return int(owner.id), int(api.id)


def test_a_second_editor_blocks_until_the_first_editors_transaction_finishes(
    session_factory, seeded
) -> None:
    """Two real connections, barrier-synchronised: the second call's own
    lock statement must not return until the first call's transaction
    commits or rolls back -- the actual behavior ``FOR UPDATE`` exists to
    provide, and the one thing no SQLite-backed test can demonstrate.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    second_finished = threading.Event()
    first_call_claimed = threading.Event()
    first_call_lock = threading.Lock()

    real_validate = custom_api_api.validate_runtime_config_declaration

    def paced_validate(**kwargs):
        # Both threads run through this same patched function once each
        # gets past its own lock statement. Only the call that gets here
        # *first* pauses: that is the first editor, holding its row lock
        # open via this still-uncommitted transaction. A second call that
        # reaches this point too (rather than staying blocked earlier,
        # inside its own lock statement) is not made to wait a second
        # time here -- pausing it too would prove nothing about the
        # database lock, only about this Python-level barrier.
        with first_call_lock:
            is_first_call = not first_call_claimed.is_set()
            first_call_claimed.set()
        if is_first_call:
            lock_acquired.set()
            assert release_lock.wait(timeout=10), "the first editor was never released"
        return real_validate(**kwargs)

    custom_api_api.validate_runtime_config_declaration = paced_validate
    session_a = session_factory()
    session_b = session_factory()
    try:

        def run_first():
            return custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(name="renamed-by-first-editor"),
                current_user=current_user,
                db=session_a,
            )

        def run_second():
            result = custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(description="edited-by-second-editor"),
                current_user=current_user,
                db=session_b,
            )
            second_finished.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run_first)
            assert lock_acquired.wait(timeout=5), (
                "the first editor never reached the lock"
            )

            second = executor.submit(run_second)
            # The second call's own lock statement should still be blocked
            # on the database at this point. If the lock were not real (or
            # a no-op, as on SQLite), the second call would sail through
            # almost immediately and this would flip to True.
            assert not second_finished.wait(timeout=1.0), (
                "the second editor finished before the first one released "
                "the row -- the lock did not actually block it"
            )

            release_lock.set()
            first.result(timeout=10)
            second.result(timeout=10)

        assert second_finished.is_set()
    finally:
        custom_api_api.validate_runtime_config_declaration = real_validate
        session_a.close()
        session_b.close()


def test_the_second_editors_rename_reports_the_first_editors_committed_name_as_old(
    session_factory, seeded
) -> None:
    """``rename_team_connector``'s ``old`` argument must be the name this
    transaction's own lock actually holds once acquired, not whatever the
    pre-lock read saw.

    Interleaving under test: the first editor renames the connector and
    commits while the second editor is blocked on the lock. The second
    editor then acquires the lock, refreshed to the first editor's
    committed name, and renames again. If the second editor's ``old``
    argument were captured before its own lock instead, it would report
    the connector's *original* name -- not the name every team agent's
    selector was already rewritten to by the first editor's own call --
    and the second rewrite would search for a name nothing holds anymore,
    leaving the first rewrite's result permanently dangling with no error.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    second_finished = threading.Event()
    first_call_claimed = threading.Event()
    first_call_lock = threading.Lock()

    renamed_calls: list[tuple[str, str]] = []
    renamed_calls_lock = threading.Lock()

    def spy_renamed_hook(_db, _user_id, _connector_type, _connector_id, old, new):
        with renamed_calls_lock:
            renamed_calls.append((old, new))

    real_validate = custom_api_api.validate_runtime_config_declaration

    def paced_validate(**kwargs):
        with first_call_lock:
            is_first_call = not first_call_claimed.is_set()
            first_call_claimed.set()
        if is_first_call:
            lock_acquired.set()
            assert release_lock.wait(timeout=10), "the first editor was never released"
        return real_validate(**kwargs)

    custom_api_api.validate_runtime_config_declaration = paced_validate
    session_a = session_factory()
    session_b = session_factory()
    try:
        with snapshot_connector_team_hooks():
            set_connector_team_hooks(renamed=spy_renamed_hook)

            def run_first():
                return custom_api_api.update_custom_api(
                    api_id,
                    CustomApiUpdate(name="renamed-by-first-editor"),
                    current_user=current_user,
                    db=session_a,
                )

            def run_second():
                result = custom_api_api.update_custom_api(
                    api_id,
                    CustomApiUpdate(name="renamed-by-second-editor"),
                    current_user=current_user,
                    db=session_b,
                )
                second_finished.set()
                return result

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(run_first)
                assert lock_acquired.wait(timeout=5), (
                    "the first editor never reached the lock"
                )

                second = executor.submit(run_second)
                assert not second_finished.wait(timeout=1.0), (
                    "the second editor finished before the first one released the row"
                )

                release_lock.set()
                first.result(timeout=10)
                second.result(timeout=10)

        assert renamed_calls == [
            ("edit-lock-target", "renamed-by-first-editor"),
            ("renamed-by-first-editor", "renamed-by-second-editor"),
        ]
    finally:
        custom_api_api.validate_runtime_config_declaration = real_validate
        session_a.close()
        session_b.close()


def test_a_row_that_vanishes_after_the_gate_but_before_the_lock_is_a_404_not_a_500(
    session_factory, seeded
) -> None:
    """The gate helper's own read can find the row and still lose a race to
    a concurrent delete that commits before this route's own lock
    statement runs. The lock statement must see that as an ordinary
    "row not found" (``None``) and let the route's existing 404 handle
    it, not surface as an unrelated 500 out of the write path below.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    real_resolve = custom_api_api._resolve_custom_api_for_request

    def resolve_then_delete_concurrently(db_, user_id, aid, **kwargs):
        result = real_resolve(db_, user_id, aid, **kwargs)
        # A concurrent delete that actually commits, from a separate
        # connection, landing strictly between the gate helper's read
        # above and the route's own lock statement below.
        with session_factory() as other:
            other.execute(
                sa.delete(UserCustomApi).where(UserCustomApi.custom_api_id == aid)
            )
            other.execute(sa.delete(CustomApi).where(CustomApi.id == aid))
            other.commit()
        return result

    custom_api_api._resolve_custom_api_for_request = resolve_then_delete_concurrently
    db = session_factory()
    try:
        with pytest.raises(HTTPException) as exc:
            custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(name="renamed-after-vanish"),
                current_user=current_user,
                db=db,
            )
        assert exc.value.status_code == 404
    finally:
        custom_api_api._resolve_custom_api_for_request = real_resolve
        db.close()
