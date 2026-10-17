"""The dispatch class each upstream ingress path gives the command it stages.

Every case drives the real ingress function, not ``stage_task_command``
directly, and reads the class back from a fresh session.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

from xagent.web.api import a2a as a2a_api
from xagent.web.api import websocket as websocket_api
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_resume_command
from xagent.web.services.task_command_transport import TaskCommandKind
from xagent.web.services.task_resume import TaskReplyInput
from xagent.web.services.task_start_protocol import (
    TaskStartPayload,
    stage_task_start_command,
)

INTERACTIVE = "interactive"
BATCH = "batch"
SRC = Path(__file__).resolve().parents[3] / "src" / "xagent"


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    init_db(db_url=f"sqlite:///{tmp_path / 'entrypoints.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        agent = Agent(user_id=user.id, name="entry")
        db.add(agent)
        db.commit()
        owner_id, agent_id = user.id, agent.id
    counter = itertools.count(1)

    class Host:
        owner = owner_id
        agent = agent_id

        def task(self, source, **fields):
            fields.setdefault("status", TaskStatus.PENDING)
            with get_session_local()() as db:
                task = Task(
                    user_id=owner_id,
                    agent_id=agent_id,
                    title="Entry",
                    source=source,
                    **fields,
                )
                db.add(task)
                db.commit()
                return task.id

        def next(self):
            return next(counter)

        def row(self, command_db_id):
            with get_session_local()() as db:
                row = db.get(TaskExecutionCommand, command_db_id)
                return row.kind, row.dispatch_class

    try:
        yield Host()
    finally:
        Base.metadata.drop_all(bind=get_engine())


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (TaskCommandKind.MESSAGE, INTERACTIVE),
        (TaskCommandKind.PAUSE, BATCH),
        (TaskCommandKind.CANCEL, BATCH),
    ],
)
def test_websocket_commands(host, kind, expected):
    task_id = host.task("internal")
    result = websocket_api._enqueue_websocket_task_command_sync(
        task_id=task_id,
        actor_user_id=host.owner,
        actor_is_admin=False,
        command_id=f"ws-{kind.value}-{host.next()}",
        kind=kind,
        payload={"message": "hello"} if kind is TaskCommandKind.MESSAGE else {},
        allow_missing_task=False,
    )
    assert host.row(result.command_id) == (kind.value, expected)


@pytest.mark.parametrize(
    ("source", "channel_name", "expected"),
    [
        pytest.param("internal", None, INTERACTIVE, id="chat-window"),
        pytest.param("internal", "slack", INTERACTIVE, id="channel"),
        pytest.param("widget", None, INTERACTIVE, id="upstream-widget"),
        pytest.param("shared_link", None, INTERACTIVE, id="shared-link"),
        pytest.param("sdk", None, BATCH, id="sdk-start"),
        pytest.param("a2a", None, BATCH, id="a2a-start"),
        pytest.param("trigger", None, BATCH, id="trigger"),
    ],
)
def test_start_command_by_task_source(host, source, channel_name, expected):
    task_id = host.task(source, channel_name=channel_name)
    turn = f"turn-{host.next()}"
    with get_session_local()() as db:
        staged = stage_task_start_command(
            db,
            task_id=task_id,
            actor_user_id=host.owner,
            start=TaskStartPayload(
                version=1,
                run_id=f"run-{turn}",
                state_version=0,
                turn_id=turn,
                kind="create",
                message="hello",
            ),
        )
        db.commit()
    assert host.row(staged.staged_db_id) == ("start", expected)


@pytest.mark.parametrize("source", ["sdk", "a2a"])
def test_reply_to_a_waiting_task_is_interactive(host, source):
    task_id = host.task(
        source,
        status=TaskStatus.WAITING_FOR_USER,
        control_state="waiting_for_user",
        run_id="run-1",
        state_version=1,
    )
    ctx = TaskReplyInput(
        task_id=task_id,
        agent_id=host.agent,
        task_owner_user_id=host.owner,
        actor_user_id=host.owner,
        run_id="run-1",
        status=TaskStatus.WAITING_FOR_USER,
        text="answer",
    )
    command_db_id = task_resume_command._admit_reply(ctx, source, "", "reply-1")
    assert host.row(command_db_id) == ("resume_input", INTERACTIVE)


def test_a2a_cancel_is_batch(host):
    task_id = host.task(
        "a2a",
        status=TaskStatus.RUNNING,
        control_state="running",
        run_id="run-1",
        state_version=3,
    )
    prepared = a2a_api._prepare_a2a_cancel_command_sync(
        task_id=task_id, agent_id=host.agent, task_owner_user_id=host.owner
    )
    assert host.row(prepared.command_db_id) == ("cancel", BATCH)


# --------------------------------------------------------------- construction sites


def _calls(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            callee = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if callee == name:
                yield node


def _enclosing_functions(tree: ast.AST):
    """Map each call node to the name of its innermost enclosing function."""
    owners: dict[int, str] = {}

    def visit(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else current
            )
            owners[id(child)] = name
            visit(child, name)

    visit(tree, "<module>")
    return owners


def _call_sites(function: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owners = _enclosing_functions(tree)
        for call in _calls(tree, function):
            found.add((str(path.relative_to(SRC)), owners.get(id(call), "<module>")))
    return found


def test_command_rows_are_built_in_exactly_one_place():
    # The class is decided where the row is built, so a second construction
    # site would stage commands the classifier never saw.
    assert _call_sites("TaskExecutionCommand") == {
        ("web/services/task_command_transport.py", "stage_task_command")
    }


# Every function that stages a command row, directly or through the start
# helper. A new or moved call site changes this set: decide the dispatch class
# of what it stages and cover it above (or state why a control command needs no
# case) before adding it here.
_INGRESS = {
    ("stage_task_command", "web/services/channel_stop.py", "_stop_once_no_commit"),
    ("stage_task_command", "web/services/shared_channel_execution.py", "enqueue"),
    (
        "stage_task_command",
        "web/services/task_auto_resume.py",
        "_process_due_candidate_no_commit",
    ),
    (
        "stage_task_command",
        "web/services/task_command_transport.py",
        "enqueue_task_command",
    ),
    ("stage_task_command", "web/services/task_interaction_service.py", "respond"),
    ("stage_task_command", "web/services/task_resume_command.py", "_admit_reply"),
    (
        "stage_task_command",
        "web/services/task_start_protocol.py",
        "stage_task_start_command",
    ),
    (
        "enqueue_task_command",
        "web/api/a2a.py",
        "_prepare_a2a_cancel_command_sync",
    ),
    (
        "enqueue_task_command",
        "web/api/websocket.py",
        "_enqueue_websocket_task_command_sync",
    ),
    (
        "enqueue_task_command",
        "web/services/workforce_runtime.py",
        "pause_workforce_tasks_after_archive",
    ),
    (
        "stage_task_start_command",
        "web/services/shared_channel_execution.py",
        "accept_channel_turn_no_commit",
    ),
    (
        "stage_task_start_command",
        "web/services/task_existing_command.py",
        "enqueue_existing_execution",
    ),
    (
        "stage_task_start_command",
        "web/services/task_orchestrator.py",
        "_claim_turn_no_commit",
    ),
}


def test_ingress_inventory_is_pinned():
    found = {
        (function, path, owner)
        for function in (
            "stage_task_command",
            "enqueue_task_command",
            "stage_task_start_command",
        )
        for path, owner in _call_sites(function)
    }
    assert found == _INGRESS
