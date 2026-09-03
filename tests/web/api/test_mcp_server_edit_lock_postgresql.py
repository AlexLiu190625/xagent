"""Real-PostgreSQL coverage for the row lock ``update_mcp_server`` takes on
the ``MCPServer`` definition row before building the new config -- and, for
the payloads that must *not* take it: a PUT that sets only ``is_active``
and/or ``user_env`` writes the caller's own ``UserMCPServer`` link row and
never runs the config rebuild at all, regardless of what the row it reads
carries -- a server with a global ``env`` or ``auth`` gets no re-encrypted
secret written back on this path either. Under PostgreSQL REPEATABLE READ,
locking a row this payload never writes is the difference between HTTP 200
and a serialization failure surfacing as HTTP 500.

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

Also covers the caller's own ``UserMCPServer`` link row being revoked --
deleted, or stripped of ownership -- by a second connection while the
route holds the definition row locked.
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


def _seed_by_the_create_route(session_factory) -> tuple[int, int]:
    """One owner and one server created the way the API actually creates
    one -- through ``create_mcp_server`` itself, not hand-built rows.

    The shape matters for the activation-only tests below. ``MCPServer``
    rows the create route makes store ``concurrent_tools`` as an empty
    list (``MCPServerConfig`` declares it ``default_factory=list`` and
    ``MCPServer.from_config`` normalizes it), while a hand-built row
    leaves the column NULL. An update rebuilds the shared config on every
    payload and assigns ``[]`` back, which is a no-op against ``[]`` and a
    real ``UPDATE`` against NULL -- so a hand-built row would make the
    activation-only request write the definition row for a reason that
    exists nowhere in production, and the tests below would be measuring
    that instead of what they claim to measure.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerCreate

    with session_factory() as db:
        owner = User(username="mcp-activation-owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        owner_id = int(owner.id)

        mcp_api.create_mcp_server(
            MCPServerCreate(
                name="activation-target",
                transport="stdio",
                config={"command": "true"},
            ),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )

    with session_factory() as db:
        server = db.query(MCPServer).filter(MCPServer.name == "activation-target").one()
        # The anchor for the docstring above: if the create route ever stops
        # storing an empty list here, these tests must be re-derived rather
        # than silently start measuring a different row shape.
        assert server.concurrent_tools == [], (
            "the create route no longer stores concurrent_tools as an empty "
            f"list (saw {server.concurrent_tools!r}); the activation-only "
            "tests below depend on that shape"
        )
        return owner_id, int(server.id)


@pytest.fixture()
def repeatable_read_sessions():
    """Two session factories on one disposable database: the first at the
    server's default isolation level (used to seed and to read back what
    committed), the second at REPEATABLE READ -- the level the route runs
    at on a deployment configured that way, and the one under which the
    interleaving below used to fail.
    """
    with disposable_database_factory("xagent_mcp_activation_rr") as make_database:
        engine = make_database("rr")
        Base.metadata.create_all(bind=engine)
        yield (
            sessionmaker(autocommit=False, autoflush=False, bind=engine),
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine.execution_options(isolation_level="REPEATABLE READ"),
            ),
        )


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

    # Both writer sessions are closed above, so this reads what actually
    # committed rather than either session's own uncommitted view. The
    # block above proves the second editor waited; without this it would
    # still pass if neither editor's write survived.
    with session_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.name == "renamed-by-first-editor"
        assert row.description == "edited-by-second-editor"
        assert row.transport == "stdio"
        assert row.command == "true"
        assert row.managed == "external"
        assert (
            fresh.query(UserMCPServer)
            .filter(
                UserMCPServer.mcpserver_id == server_id,
                UserMCPServer.user_id == owner_id,
            )
            .one()
            .is_active
            is True
        )


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

    # The hook tuples above are in-process call records; this reads what
    # actually committed, so a rename that reported the right pair of names
    # and then failed to persist cannot pass.
    with session_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.name == "renamed-by-second-editor"
        assert row.transport == "stdio"
        assert row.command == "true"
        assert row.managed == "external"
        assert (
            fresh.query(UserMCPServer)
            .filter(
                UserMCPServer.mcpserver_id == server_id,
                UserMCPServer.user_id == owner_id,
            )
            .one()
            .is_active
            is True
        )


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
    queried_entities: list[tuple] = []

    def delete_the_row_when_the_lock_query_starts(*entities, **kwargs):
        queried_entities.append(entities)
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
        assert queried_entities[:2] == [(UserMCPServer, MCPServer), (MCPServer,)], (
            "the concurrent delete must land after the access read's own "
            "read and before this route's definition read; otherwise this "
            "404 could be the gate's rather than the lock query's own "
            f"empty result -- saw {queried_entities!r}"
        )
    finally:
        db.close()


def test_an_activation_only_edit_survives_a_concurrent_definition_commit_under_repeatable_read(
    repeatable_read_sessions,
) -> None:
    """A payload that sets only ``is_active`` writes this caller's own
    ``UserMCPServer`` link row. Under PostgreSQL REPEATABLE READ the
    route's own snapshot is fixed by its first read, so a definition edit
    another request commits after that read is invisible to this one --
    which is harmless for a request that does not write the definition
    row, and fatal for one that asks the database to lock it: the lock
    statement raises SQLSTATE 40001 and this route's generic handler turns
    that into HTTP 500 with the requested activation state unwritten.

    The concurrent commit is fired from a wrapper around this session's
    own ``query``, which lands it strictly between the access read (which
    asks for ``UserMCPServer`` joined to ``MCPServer``) and this route's
    first single-entity ``MCPServer`` read.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    default_factory, rr_factory = repeatable_read_sessions
    owner_id, server_id = _seed_by_the_create_route(default_factory)

    db = rr_factory()
    real_query = db.query
    committed = threading.Event()
    queried_entities: list[tuple] = []

    def commit_a_definition_edit_when_the_definition_query_starts(*entities, **kwargs):
        queried_entities.append(entities)
        if entities == (MCPServer,) and not committed.is_set():
            committed.set()
            with default_factory() as other:
                other.execute(
                    sa.update(MCPServer)
                    .where(MCPServer.id == server_id)
                    .values(description="edited-by-the-concurrent-definition-editor")
                )
                other.commit()
        return real_query(*entities, **kwargs)

    db.query = commit_a_definition_edit_when_the_definition_query_starts
    try:
        response = mcp_api.update_mcp_server(
            server_id,
            MCPServerUpdate(is_active=False),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )
    finally:
        db.close()

    assert committed.is_set(), "the concurrent definition edit never ran"
    assert queried_entities[:2] == [(UserMCPServer, MCPServer), (MCPServer,)], (
        "the concurrent commit must land after the access read's own read and "
        "before this route's definition read; otherwise this test is not "
        "exercising the window it claims to -- saw "
        f"{queried_entities!r}"
    )
    assert response.is_active is False

    with default_factory() as fresh:
        assert (
            fresh.query(UserMCPServer)
            .filter(
                UserMCPServer.mcpserver_id == server_id,
                UserMCPServer.user_id == owner_id,
            )
            .one()
            .is_active
            is False
        )
        assert (
            fresh.query(MCPServer).filter(MCPServer.id == server_id).one().description
            == "edited-by-the-concurrent-definition-editor"
        )


def test_a_user_env_only_edit_on_a_server_with_a_global_env_does_not_take_the_lock(
    repeatable_read_sessions,
) -> None:
    """A payload that sets only ``user_env`` writes this caller's own
    ``UserMCPServer`` link row, even when the shared server it targets
    carries a global ``env``. Before the fix this route rebuilds, this row
    shape is exactly the one the reviewer's finding was about: the config
    rebuild decrypts the definition row's ``env``, then encrypts it again
    on the way back out, and Fernet ciphertext differs on every call even
    when the plaintext does not -- so the lock-free path still wrote this
    row, unlocked. After the fix the rebuild does not run at all on this
    path, so the stored ciphertext must come back byte-for-byte identical.

    Proves the lock is not taken the same way the activation-only test
    above does: a concurrent definition-row commit lands strictly between
    this route's access read and its definition-row read, and under
    REPEATABLE READ that commit is invisible to a plain read but forces a
    locking read (``FOR ... UPDATE``) to fail with SQLSTATE 40001. This
    request surviving with HTTP 200 is what shows no locking read ran --
    the concurrent commit is itself a real definition-row write (to
    ``description``, a column this test does not otherwise touch), so it
    is expected to move the row's own ``updated_at``; the companion test
    below covers that assertion for a request with no concurrent editor
    to confuse it.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.core.utils.encryption import encrypt_env_dict
    from xagent.web.api.mcp import MCPServerUpdate

    default_factory, rr_factory = repeatable_read_sessions
    stored_env = encrypt_env_dict({"K": "v"})
    with default_factory() as db:
        owner = User(username="mcp-env-only-owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.flush()
        server = MCPServer(
            name="env-only-target",
            transport="stdio",
            managed="external",
            command="true",
            concurrent_tools=[],
            env=stored_env,
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
        owner_id, server_id = int(owner.id), int(server.id)

    db = rr_factory()
    real_query = db.query
    committed = threading.Event()

    def commit_a_definition_edit_when_the_definition_query_starts(*entities, **kwargs):
        if entities == (MCPServer,) and not committed.is_set():
            committed.set()
            with default_factory() as other:
                other.execute(
                    sa.update(MCPServer)
                    .where(MCPServer.id == server_id)
                    .values(description="edited-by-the-concurrent-definition-editor")
                )
                other.commit()
        return real_query(*entities, **kwargs)

    db.query = commit_a_definition_edit_when_the_definition_query_starts
    try:
        response = mcp_api.update_mcp_server(
            server_id,
            MCPServerUpdate(user_env={"MINE": "x"}),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )
    finally:
        db.close()

    assert committed.is_set(), "the concurrent definition edit never ran"
    assert response is not None

    with default_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.env == stored_env, (
            "the shared definition row's env must be untouched byte-for-byte "
            "by a request that only sets user_env -- a changed value here "
            "means the config rebuild ran (and re-encrypted it) on a path "
            "that must not run it at all"
        )
        assert row.description == "edited-by-the-concurrent-definition-editor", (
            "the concurrent definition edit must have actually landed, or "
            "this test proves nothing about interleaving"
        )


def test_a_user_env_only_edit_on_a_server_with_a_global_env_leaves_it_and_updated_at_untouched(
    session_factory, seeded
) -> None:
    """The single-request counterpart to the concurrency test above, with
    no concurrent editor to also move ``updated_at``: a payload that sets
    only ``user_env`` on a server carrying a global ``env`` must leave that
    ``env`` value AND the row's ``updated_at`` exactly as they were --
    nothing in this request has any business touching either.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.core.utils.encryption import encrypt_env_dict
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
    stored_env = encrypt_env_dict({"K": "v"})
    with session_factory() as db:
        db.execute(
            sa.update(MCPServer)
            .where(MCPServer.id == server_id)
            .values(env=stored_env, concurrent_tools=[])
        )
        db.commit()
        original_updated_at = (
            db.query(MCPServer).filter(MCPServer.id == server_id).one().updated_at
        )

    with session_factory() as db:
        response = mcp_api.update_mcp_server(
            server_id,
            MCPServerUpdate(user_env={"MINE": "x"}),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )
    assert response is not None

    with session_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.env == stored_env, (
            "env must be untouched byte-for-byte by a user_env-only request"
        )
        assert row.updated_at == original_updated_at, (
            "a request that writes only the caller's own link row must not "
            "move the shared definition row's updated_at"
        )


def test_an_is_active_only_edit_on_a_server_with_concurrent_tools_null_does_not_take_the_lock(
    repeatable_read_sessions,
) -> None:
    """Same shape as the global-``env`` case above, for the other row shape
    the reviewer's finding covers: a pre-migration row whose
    ``concurrent_tools`` column is still ``NULL`` (the
    ``20260624_add_mcp_concurrency_config`` migration added the column with
    no ``server_default`` and no backfill). Before the fix, the config
    rebuild normalizes a ``NULL`` ``concurrent_tools`` to ``[]`` and writes
    it back even on the lock-free path; after the fix the rebuild does not
    run, so the column must still read ``NULL``. See the companion test
    below for the ``updated_at``-unchanged assertion this test's concurrent
    editor (a real definition-row write of its own) would otherwise
    confuse.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    default_factory, rr_factory = repeatable_read_sessions
    with default_factory() as db:
        owner = User(
            username="mcp-ctnull-only-owner", password_hash="x", is_admin=False
        )
        db.add(owner)
        db.flush()
        server = MCPServer(
            name="ctnull-only-target",
            transport="stdio",
            managed="external",
            command="true",
        )
        db.add(server)
        db.flush()
        assert server.concurrent_tools is None, (
            "this test depends on a hand-built row leaving concurrent_tools "
            "NULL, the way a pre-migration row does"
        )
        db.add(
            UserMCPServer(
                user_id=int(owner.id),
                mcpserver_id=int(server.id),
                is_owner=True,
                is_active=True,
            )
        )
        db.commit()
        owner_id, server_id = int(owner.id), int(server.id)

    db = rr_factory()
    real_query = db.query
    committed = threading.Event()

    def commit_a_definition_edit_when_the_definition_query_starts(*entities, **kwargs):
        if entities == (MCPServer,) and not committed.is_set():
            committed.set()
            with default_factory() as other:
                other.execute(
                    sa.update(MCPServer)
                    .where(MCPServer.id == server_id)
                    .values(description="edited-by-the-concurrent-definition-editor")
                )
                other.commit()
        return real_query(*entities, **kwargs)

    db.query = commit_a_definition_edit_when_the_definition_query_starts
    try:
        response = mcp_api.update_mcp_server(
            server_id,
            MCPServerUpdate(is_active=False),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )
    finally:
        db.close()

    assert committed.is_set(), "the concurrent definition edit never ran"
    assert response.is_active is False

    with default_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.concurrent_tools is None, (
            "an is_active-only request must not normalize this row's NULL "
            "concurrent_tools to [] -- that write means the config rebuild "
            "ran on a path that must not run it at all"
        )
        assert row.description == "edited-by-the-concurrent-definition-editor", (
            "the concurrent definition edit must have actually landed, or "
            "this test proves nothing about interleaving"
        )


def test_an_is_active_only_edit_on_a_server_with_concurrent_tools_null_stays_null(
    session_factory, seeded
) -> None:
    """The single-request counterpart to the concurrency test above: an
    ``is_active``-only payload against a row whose ``concurrent_tools`` is
    still ``NULL`` must leave it ``NULL``, not normalize it to ``[]``."""
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
    with session_factory() as db:
        assert (
            db.query(MCPServer).filter(MCPServer.id == server_id).one().concurrent_tools
            is None
        ), "the seeded fixture's row must leave concurrent_tools NULL"

    with session_factory() as db:
        response = mcp_api.update_mcp_server(
            server_id,
            MCPServerUpdate(is_active=False),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )
    assert response.is_active is False

    with session_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.concurrent_tools is None, (
            "an is_active-only request must not normalize a NULL concurrent_tools to []"
        )


def test_a_user_env_only_edit_on_a_server_with_restart_policy_always_does_not_take_the_lock(
    repeatable_read_sessions,
) -> None:
    """Same shape again, for the third row form the reviewer's finding
    covers: a ``restart_policy`` other than the config default. Before the
    fix, ``_build_server_config``'s round trip through ``to_config_dict()``
    only emits ``restart_policy`` for a ``managed="internal"`` row, so an
    ``external`` row's real ``"always"`` reads back as the config default
    ``"no"`` and gets written onto the row on the lock-free path -- silently
    turning off a connector's configured auto-restart on an unrelated
    per-user env edit. After the fix the rebuild does not run, so the
    column must still read ``"always"``.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    default_factory, rr_factory = repeatable_read_sessions
    with default_factory() as db:
        owner = User(
            username="mcp-restart-only-owner", password_hash="x", is_admin=False
        )
        db.add(owner)
        db.flush()
        server = MCPServer(
            name="restart-only-target",
            transport="stdio",
            managed="external",
            command="true",
            concurrent_tools=[],
            restart_policy="always",
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
        owner_id, server_id = int(owner.id), int(server.id)

    db = rr_factory()
    real_query = db.query
    committed = threading.Event()

    def commit_a_definition_edit_when_the_definition_query_starts(*entities, **kwargs):
        if entities == (MCPServer,) and not committed.is_set():
            committed.set()
            with default_factory() as other:
                other.execute(
                    sa.update(MCPServer)
                    .where(MCPServer.id == server_id)
                    .values(description="edited-by-the-concurrent-definition-editor")
                )
                other.commit()
        return real_query(*entities, **kwargs)

    db.query = commit_a_definition_edit_when_the_definition_query_starts
    try:
        response = mcp_api.update_mcp_server(
            server_id,
            MCPServerUpdate(user_env={"MINE": "x"}),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )
    finally:
        db.close()

    assert committed.is_set(), "the concurrent definition edit never ran"
    assert response is not None

    with default_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.restart_policy == "always", (
            "a user_env-only request must not overwrite this row's "
            "restart_policy with the config default -- that write means "
            "the config rebuild ran on a path that must not run it at all"
        )
        assert row.description == "edited-by-the-concurrent-definition-editor", (
            "the concurrent definition edit must have actually landed, or "
            "this test proves nothing about interleaving"
        )


def test_a_user_env_only_edit_on_a_server_with_restart_policy_always_leaves_it_untouched(
    session_factory, seeded
) -> None:
    """The single-request counterpart to the concurrency test above: a
    ``user_env``-only payload against a row with ``restart_policy="always"``
    must leave that value alone, not overwrite it with the config
    default (``"no"``)."""
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
    with session_factory() as db:
        db.execute(
            sa.update(MCPServer)
            .where(MCPServer.id == server_id)
            .values(restart_policy="always", concurrent_tools=[])
        )
        db.commit()

    with session_factory() as db:
        response = mcp_api.update_mcp_server(
            server_id,
            MCPServerUpdate(user_env={"MINE": "x"}),
            current_user=SimpleNamespace(id=owner_id, is_admin=False),
            db=db,
        )
    assert response is not None

    with session_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.restart_policy == "always", (
            "a user_env-only request must not overwrite restart_policy "
            "with the config default"
        )


def test_an_activation_only_edit_whose_row_vanishes_before_its_read_is_a_404(
    session_factory, seeded
) -> None:
    """The activation-only path skips the lock but keeps the same
    vanished-definition handling: its own fresh read of ``MCPServer`` must
    return ``None`` and raise the route's 404, rather than leaving the
    write path below to fail on a row that is no longer there.

    Uses the plain hand-built ``seeded`` fixture rather than the
    create-route seeding helper: this request 404s before any write is
    attempted, so the row's shape (NULL vs. ``[]`` ``concurrent_tools``)
    plays no part in what this test measures.

    Same wrapper shape as
    ``test_a_row_that_vanishes_after_the_gate_but_before_the_lock_is_a_404_not_a_500``
    above: the concurrent delete fires from this session's own ``query``,
    which lands it strictly between the access read (which asks for
    ``UserMCPServer`` joined to ``MCPServer``) and this route's first
    single-entity ``MCPServer`` read.
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_query = db.query
    deleted_already = threading.Event()
    queried_entities: list[tuple] = []

    def delete_the_row_when_the_definition_query_starts(*entities, **kwargs):
        queried_entities.append(entities)
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

    db.query = delete_the_row_when_the_definition_query_starts
    try:
        with pytest.raises(HTTPException) as exc:
            mcp_api.update_mcp_server(
                server_id,
                MCPServerUpdate(is_active=False),
                current_user=current_user,
                db=db,
            )
        assert exc.value.status_code == 404
        assert deleted_already.is_set(), "the concurrent delete never ran"
        assert queried_entities[:2] == [(UserMCPServer, MCPServer), (MCPServer,)], (
            "the concurrent delete must land after the access read's own "
            "read and before this route's definition read; otherwise the "
            "404 under test could be the access read's -- saw "
            f"{queried_entities!r}"
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    ("revocation", "expected_status"),
    [("link-deleted", 404), ("ownership-cleared", 403)],
)
def test_a_put_whose_association_is_revoked_after_the_lock_is_refused_with_no_shared_write(
    session_factory, seeded, revocation, expected_status
) -> None:
    """A second connection can revoke the caller's own link -- delete it,
    or clear ``is_owner`` -- and commit while this route still holds the
    definition row locked, after the gate already let the request through.
    The re-read added after the lock, which re-derives ``can_edit_global``,
    must catch that: a gone link is the gate's own 404. A link that no
    longer owns the server is not an error by itself -- the gate does not
    refuse a non-owner either -- so a payload that changes the shared
    configuration is refused by the existing owner-only guard instead.

    The revocation fires on this route's first single-entity
    ``UserMCPServer`` read -- the gate's own read joins it to ``MCPServer``,
    so a bare ``(UserMCPServer,)`` can only be the re-read after the lock.
    The recorded sequence is filtered to the three statements under test
    (``DatabaseMCPServerManager`` may issue queries of its own).
    """
    import xagent.web.api.mcp as mcp_api
    from xagent.web.api.mcp import MCPServerUpdate

    owner_id, server_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_query = db.query
    real_commit = db.commit
    revoked_already = threading.Event()
    queried_entities: list[tuple] = []
    tracked_keys = {(UserMCPServer, MCPServer), (MCPServer,), (UserMCPServer,)}
    commits: list[str] = []

    def revoke_when_the_recheck_query_starts(*entities, **kwargs):
        if entities in tracked_keys:
            queried_entities.append(entities)
        if entities == (UserMCPServer,) and not revoked_already.is_set():
            revoked_already.set()
            with session_factory() as other:
                if revocation == "link-deleted":
                    other.execute(
                        sa.delete(UserMCPServer).where(
                            UserMCPServer.user_id == owner_id,
                            UserMCPServer.mcpserver_id == server_id,
                        )
                    )
                else:
                    other.execute(
                        sa.update(UserMCPServer)
                        .where(
                            UserMCPServer.user_id == owner_id,
                            UserMCPServer.mcpserver_id == server_id,
                        )
                        .values(is_owner=False)
                    )
                other.commit()
        return real_query(*entities, **kwargs)

    def record_commit():
        commits.append("commit")
        return real_commit()

    db.query = revoke_when_the_recheck_query_starts
    db.commit = record_commit
    try:
        with pytest.raises(HTTPException) as exc:
            mcp_api.update_mcp_server(
                server_id,
                MCPServerUpdate(name="renamed-after-revocation"),
                current_user=current_user,
                db=db,
            )
        assert exc.value.status_code == expected_status
        assert revoked_already.is_set(), "the concurrent revocation never ran"
        assert queried_entities[:3] == [
            (UserMCPServer, MCPServer),
            (MCPServer,),
            (UserMCPServer,),
        ], (
            "the concurrent revocation must land after the gate's own read "
            "and after the lock statement, and be caught by the re-read "
            "added after the lock -- otherwise the status under test could "
            f"be the gate's rather than the re-read's -- saw "
            f"{queried_entities!r}"
        )
        if revocation == "ownership-cleared":
            assert (
                exc.value.detail
                == "Only the server owner can change the shared configuration"
            ), (
                "the 403 for a link that no longer owns the server must come "
                "from the route's existing owner-only guard, not a new error "
                f"shape -- saw {exc.value.detail!r}"
            )
        assert commits == [], "the refused edit must commit nothing"
    finally:
        db.close()

    with session_factory() as fresh:
        row = fresh.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert row.name == "edit-lock-target", (
            "the shared definition row must be untouched by a refused edit"
        )
