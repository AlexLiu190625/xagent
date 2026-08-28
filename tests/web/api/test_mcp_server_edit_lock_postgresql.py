"""Real-PostgreSQL coverage for the row lock ``update_mcp_server`` takes on
the ``MCPServer`` definition row before building the new config.

``FOR UPDATE`` is a no-op on SQLite -- every other suite in this repo runs
against SQLite, so nothing there can tell a genuine second-writer block
from a lock statement that silently does nothing. This file is the one
place that runs the real statement against a real server and proves it
actually blocks a second writer, plus the companion path where the row
vanishes between the route's first read and this lock.

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
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import set_connector_team_hooks

pytestmark = pytest.mark.postgresql


@pytest.fixture()
def session_factory():
    with disposable_database_factory("xagent_mcp_edit_lock") as make_database:
        engine = make_database("edit_lock")
        Base.metadata.create_all(bind=engine)
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def seeded(session_factory):
    """One owner, one owned MCP server, in their own committed rows."""
    with session_factory() as db:
        owner = User(username="mcp-edit-lock-owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.flush()
        server = MCPServer(
            name="edit-lock-target",
            transport="stdio",
            managed="external",
            command="true",
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=int(owner.id),
                mcpserver_id=int(server.id),
                is_owner=True,
                is_active=True,
            )
        )
        db.commit()
        return int(owner.id), int(server.id)


def test_a_second_editor_blocks_until_the_first_editors_transaction_finishes(
    session_factory, seeded
) -> None:
    """Two real connections, barrier-synchronised: the second call's own
    lock statement must not return until the first call's transaction
    commits or rolls back -- the actual behavior ``FOR UPDATE`` exists to
    provide, and the one thing no SQLite-backed test can demonstrate.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    second_finished = threading.Event()
    first_call_claimed = threading.Event()
    first_call_lock = threading.Lock()

    real_build_server_config = mcp_api._build_server_config

    def paced_build_server_config(update_data, server):
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
        return real_build_server_config(update_data, server)

    mcp_api._build_server_config = paced_build_server_config
    session_a = session_factory()
    session_b = session_factory()
    try:

        def run_first():
            return mcp_api.update_mcp_server(
                server_id,
                MCPServerUpdate(name="renamed-by-first-editor"),
                current_user=current_user,
                db=session_a,
            )

        def run_second():
            result = mcp_api.update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited-by-second-editor"),
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
        mcp_api._build_server_config = real_build_server_config
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
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
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

    real_build_server_config = mcp_api._build_server_config

    def paced_build_server_config(update_data, server):
        with first_call_lock:
            is_first_call = not first_call_claimed.is_set()
            first_call_claimed.set()
        if is_first_call:
            lock_acquired.set()
            assert release_lock.wait(timeout=10), "the first editor was never released"
        return real_build_server_config(update_data, server)

    mcp_api._build_server_config = paced_build_server_config
    session_a = session_factory()
    session_b = session_factory()
    set_connector_team_hooks(renamed=spy_renamed_hook)
    try:

        def run_first():
            return mcp_api.update_mcp_server(
                server_id,
                MCPServerUpdate(name="renamed-by-first-editor"),
                current_user=current_user,
                db=session_a,
            )

        def run_second():
            result = mcp_api.update_mcp_server(
                server_id,
                MCPServerUpdate(name="renamed-by-second-editor"),
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
        set_connector_team_hooks()
        mcp_api._build_server_config = real_build_server_config
        session_a.close()
        session_b.close()


def test_a_row_that_vanishes_after_the_gate_but_before_the_lock_is_a_404_not_a_500(
    session_factory, seeded
) -> None:
    """The route's own access read can find the row and still lose a race
    to a concurrent delete that commits before the lock statement runs. The
    lock statement must see that as an ordinary "row not found" (``None``)
    and let the route's existing 404 handle it, rather than leaving the
    write path below to fail on a row that is no longer there.

    The concurrent delete is fired from a wrapper around this session's own
    ``query``, which lands it strictly between the two reads: the lock is
    the only statement in this route that asks for ``MCPServer`` as its
    sole entity (the access read above asks for ``UserMCPServer`` joined to
    it, passing ``MCPServer`` as a second entity), so the wrapper can
    recognise the lock query and nothing else.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_query = db.query
    deleted_already = threading.Event()

    def delete_the_row_when_the_lock_query_starts(*entities, **kwargs):
        if entities == (MCPServer,) and not deleted_already.is_set():
            deleted_already.set()
            with session_factory() as other:
                other.execute(
                    sa.delete(UserMCPServer).where(
                        UserMCPServer.mcpserver_id == server_id
                    )
                )
                other.execute(sa.delete(MCPServer).where(MCPServer.id == server_id))
                other.commit()
        return real_query(*entities, **kwargs)

    db.query = delete_the_row_when_the_lock_query_starts
    try:
        with pytest.raises(HTTPException) as exc:
            mcp_api.update_mcp_server(
                server_id,
                MCPServerUpdate(name="renamed-after-vanish"),
                current_user=current_user,
                db=db,
            )
        assert deleted_already.is_set(), "the concurrent delete never ran"
        assert exc.value.status_code == 404
    finally:
        db.close()
