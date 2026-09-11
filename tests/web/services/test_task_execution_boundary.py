"""The task runtime can load and publish events without a Web API host."""

import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock, Mock, call

import pytest

from xagent.web.services import task_events


def test_execution_services_and_tracer_load_without_api_routes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import sys

                class RejectRoutes(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "xagent.web.api" or fullname.startswith("xagent.web.api."):
                            raise AssertionError(f"Execution imported API route: {fullname}")

                sys.meta_path.insert(0, RejectRoutes())
                from xagent.web.services import agent_service_manager, task_execution, task_orchestrator
                from xagent.web.services.external_task_cancel import _broadcast_external_cancel_terminal_event
                import asyncio
                from unittest.mock import AsyncMock, patch
                from xagent.web.services import task_events
                sink = AsyncMock()
                task_events.set_task_event_sink(sink)
                event = {"type": "task_error", "task_id": 1}
                with patch.object(task_execution, "create_terminal_task_error_event", return_value=event):
                    asyncio.run(_broadcast_external_cancel_terminal_event(1))
                sink.assert_awaited_once_with(event, 1)
                from xagent.web.tracing import create_task_tracer
                create_task_tracer(1, user_id=1)
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_event_delivery_uses_the_host_sink(monkeypatch) -> None:
    monkeypatch.setattr(task_events, "_task_event_sink", None)
    monkeypatch.setattr(task_events, "_warned_missing_sink", False)
    counter = Mock()
    monkeypatch.setattr(task_events, "increment_counter", counter)
    event = {"type": "task_completed", "task_id": 42}

    sink = AsyncMock()
    task_events.set_task_event_sink(sink)
    await task_events.publish_task_event(event, 42)
    sink.assert_awaited_once_with(event, 42)

    sink.side_effect = RuntimeError("delivery failed")
    with pytest.raises(RuntimeError, match="delivery failed"):
        await task_events.publish_task_event(event, 42)
    counter.assert_not_called()


@pytest.mark.asyncio
async def test_missing_sink_counts_every_event_and_warns_once_until_registered(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(task_events, "_task_event_sink", None)
    monkeypatch.setattr(task_events, "_warned_missing_sink", False)
    counter = Mock()
    monkeypatch.setattr(task_events, "increment_counter", counter)
    event = {"type": "task_error", "message": "private event content"}

    await task_events.publish_task_event(event, 42)
    task_events.set_task_event_sink(None)
    await task_events.publish_task_event(event, 42)
    warnings = [r for r in caplog.records if r.name == task_events.__name__]
    assert len(warnings) == 1
    assert warnings[0].levelname == "WARNING"
    assert "no host event sink" in warnings[0].message
    assert event["message"] not in caplog.text

    sink = AsyncMock()
    task_events.set_task_event_sink(sink)
    await task_events.publish_task_event(event, 42)
    sink.assert_awaited_once_with(event, 42)
    assert counter.call_count == 2

    task_events.set_task_event_sink(None)
    await task_events.publish_task_event(event, 42)
    warnings = [r for r in caplog.records if r.name == task_events.__name__]
    assert len(warnings) == 2
    assert (
        counter.call_args_list
        == [call("xagent.task_events.dropped", attributes={"outcome": "no_sink"})] * 3
    )


def test_web_host_registers_event_delivery_on_import() -> None:
    # A fresh process exercises registration, not an adapter installed by a
    # previous test or an importlib.reload that leaves old module state behind.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import asyncio
            from unittest.mock import AsyncMock, patch
            from xagent.web.services import task_events
            assert task_events._task_event_sink is None
            from xagent.web.api import websocket
            event = {"type": "task_completed", "task_id": 42}
            sink = AsyncMock()
            with patch.object(websocket.manager, "broadcast_to_task", sink):
                asyncio.run(task_events.publish_task_event(event, 42))
            sink.assert_awaited_once_with(event, 42)
        """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
