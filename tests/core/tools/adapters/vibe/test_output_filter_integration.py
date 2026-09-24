"""
Integration tests for output filter with tool factory.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from xagent.core.agent.context import ExecutionContext
from xagent.core.tools import tool_result_spill
from xagent.core.tools.adapters.vibe import output_filter_wrapper
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.output_filter import DEFAULT_TRUNCATION_MESSAGE
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.tool_result_spill import (
    SPILL_PLACEHOLDER_TEXT,
    SPILL_RESERVED_RESULT_KEY,
    SpillRunBudget,
    SpillTarget,
    spill_record_shape_is_valid,
)
from xagent.core.tools.user_interaction import WAITING_FOR_USER_STATUS
from xagent.core.workspace import TaskWorkspace


@pytest.mark.asyncio
async def test_tool_factory_applies_filters():
    """Test that tools created by factory have output filtering."""
    config = ToolConfig(
        {
            "workspace": None,
            "max_output_length": 100,
        }
    )

    tools = await ToolFactory.create_all_tools(config)

    # Check that tools were created
    assert len(tools) > 0

    # Find any wrapped tool (all tools should be wrapped with _filter)
    wrapped_tools = [t for t in tools if hasattr(t, "_filter")]
    assert len(wrapped_tools) > 0, "No tools with output filter found"

    # Check that the filter has the correct configuration
    tool = wrapped_tools[0]
    assert hasattr(tool, "_filter")
    assert tool._filter.max_chars == 100


@pytest.mark.asyncio
async def test_filtered_tool_execution():
    """Test that filtered tools truncate output correctly when executed."""
    from langchain_core.tools.structured import StructuredTool
    from pydantic import BaseModel, Field

    from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolMetadata
    from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
        OutputFilteredToolWrapper,
    )

    # Create a simple test tool that returns predictable long output
    class TestInput(BaseModel):
        text: str = Field(description="Text to repeat")

    def test_long_output_func(text: str) -> str:
        """Return the input text repeated 100 times for testing output filtering."""
        return text * 100

    # Create a StructuredTool
    langchain_tool = StructuredTool.from_function(
        func=test_long_output_func,
        name="test_long_output",
        description="Test tool that returns long output",
        args_schema=TestInput,
    )

    # Create AbstractBaseTool wrapper
    class TestTool(AbstractBaseTool):
        @property
        def name(self) -> str:
            return "test_long_output"

        @property
        def description(self) -> str:
            return "Test tool that returns long output"

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="test_long_output",
                description="Test tool that returns long output",
                category="BASIC",
            )

        def args_type(self):
            return TestInput

        def return_type(self):
            return str

        def state_type(self):
            return None

        def is_async(self):
            return False

        def run_json_sync(self, args):
            result = langchain_tool.invoke(args)
            return result

        async def run_json_async(self, args):
            return self.run_json_sync(args)

    # Wrap it with the same wrapper used by ToolFactory
    wrapped = OutputFilteredToolWrapper(
        target_tool=TestTool(),
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
    )

    # Execute the tool and verify truncation
    result = wrapped.run_json_sync({"text": "abcdefghij" * 10})  # 100 chars

    # Result should be truncated to 50 chars + message
    assert len(result) <= 50 + len(DEFAULT_TRUNCATION_MESSAGE)
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)
    assert result.startswith("abcdefghij")


@pytest.mark.asyncio
async def test_default_max_output_length():
    """Test that default max output length is 50K characters."""
    config = ToolConfig({"workspace": None})

    tools = await ToolFactory.create_all_tools(config)

    # Check that at least one tool was created
    assert len(tools) > 0

    # Check that tools have the default limit
    for tool in tools:
        if hasattr(tool, "_filter"):
            assert tool._filter.max_chars == 50 * 1024


@pytest.mark.asyncio
async def test_hardcoded_truncation_message():
    """Test that truncation message uses the hardcoded default from output_filter.py."""
    config = ToolConfig(
        {
            "workspace": None,
            "max_output_length": 10,
        }
    )

    tools = await ToolFactory.create_all_tools(config)

    # Find a tool and verify truncation message is used
    for tool in tools:
        if hasattr(tool, "_filter"):
            # The filter uses the hardcoded message from output_filter.py
            assert tool._filter.max_chars == 10
            break


# --- wrapper integration (strip / spill / bypass branches) ----------------


def _wrapper(spill_target=None, max_chars=50, max_fields=1000, max_recursion=20):
    return OutputFilteredToolWrapper(
        target_tool=SimpleNamespace(name="acme"),
        max_chars=max_chars,
        max_fields=max_fields,
        max_recursion=max_recursion,
        spill_target=spill_target,
    )


def test_wrapper_spills_oversized_dict_result_instead_of_truncating(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    big_text = "x" * 100
    result = {
        "content": [{"type": "text", "text": big_text}],
        "structured_content": None,
        "is_error": False,
    }
    filtered = wrapper._filter_result(result)

    assert DEFAULT_TRUNCATION_MESSAGE not in str(filtered)
    assert filtered["content"][0]["text"] == SPILL_PLACEHOLDER_TEXT
    assert filtered["is_error"] is False
    records = filtered[SPILL_RESERVED_RESULT_KEY]
    assert len(records) == 1
    written = spill_dir / records[0]["relative_path"].split("/")[-1]
    assert written.read_text(encoding="utf-8") == big_text


def test_a_real_spill_is_invisible_to_get_output_files(tmp_path):
    """The file a real spill writes must not turn into a deliverable the
    model can hand back to the user. It lands inside the workspace's
    engine-owned tool-results directory, which get_output_files() already
    excludes from every listing -- this pins that the two mechanisms
    actually meet, using a real TaskWorkspace rather than a bare tmp_path
    and manually planted files the way test_workspace_engine_owned_dir.py
    does."""
    workspace = TaskWorkspace("task-spill", str(tmp_path))
    spill_dir = workspace.output_dir / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    big_text = "x" * 100
    result = {
        "content": [{"type": "text", "text": big_text}],
        "structured_content": None,
        "is_error": False,
    }
    filtered = wrapper._filter_result(result)
    records = filtered[SPILL_RESERVED_RESULT_KEY]
    written = spill_dir / records[0]["relative_path"].split("/")[-1]
    assert written.exists()  # the spill really happened

    listed_paths = {entry["file_path"] for entry in workspace.get_output_files()}
    assert str(written) not in listed_paths


def test_wrapper_without_spill_target_truncates_as_before(tmp_path):
    wrapper = _wrapper(spill_target=None)
    result = {"content": [{"type": "text", "text": "x" * 100}]}
    filtered = wrapper._filter_result(result)
    assert DEFAULT_TRUNCATION_MESSAGE in filtered["content"][0]["text"]
    assert SPILL_RESERVED_RESULT_KEY not in filtered


def test_wrapper_small_results_are_byte_identical_with_a_spill_target(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50))
    result = {"output": "small value", "count": 3}
    filtered = wrapper._filter_result(result)
    assert filtered == result
    assert not spill_dir.exists()


def test_a_waiting_for_user_result_is_never_spilled(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    result = {
        "status": "waiting_for_user",
        "interaction_id": "i1",
        "message_type": "question",
        "message": "please answer",
        "interactions": [{"prompt": "pick one"}],
        "records": "r" * 100,  # oversized sibling, not part of the card
    }
    filtered = wrapper._filter_result(result)
    assert SPILL_RESERVED_RESULT_KEY not in filtered
    assert not spill_dir.exists()
    assert DEFAULT_TRUNCATION_MESSAGE in filtered["records"]
    assert filtered["status"] == WAITING_FOR_USER_STATUS
    assert filtered["message"] == "please answer"
    assert filtered["interactions"] == [{"prompt": "pick one"}]


def test_a_classified_failure_result_is_never_spilled(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    result = {
        "success": False,
        "is_error": True,
        "status": "error",
        "error": "short error",
        "output": "o" * 100,  # oversized, not part of the failure signal
    }
    filtered = wrapper._filter_result(result)
    assert SPILL_RESERVED_RESULT_KEY not in filtered
    assert not spill_dir.exists()
    assert filtered["success"] is False
    assert filtered["is_error"] is True
    assert filtered["error"] == "short error"
    assert DEFAULT_TRUNCATION_MESSAGE in filtered["output"]


def test_the_bypass_branches_read_the_post_spill_object(tmp_path):
    """Neither the waiting-for-user nor the classified-failure envelope, but
    an oversized ``output`` sibling: the bypass branch's final `return
    filtered` still reads off the post-spill object, not the original one."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    result = {"output": "o" * 100}
    filtered = wrapper._filter_result(result)
    assert filtered["output"] == SPILL_PLACEHOLDER_TEXT
    assert filtered[SPILL_RESERVED_RESULT_KEY][0]["value_path"] == "output"


def test_wrapper_strips_a_forged_reserved_key_even_without_a_spill_target(caplog):
    wrapper = _wrapper(spill_target=None)
    forged = [{"relative_path": "tool-results/evil.json"}]
    result = {"output": "ok", SPILL_RESERVED_RESULT_KEY: forged}
    with caplog.at_level("WARNING"):
        filtered = wrapper._filter_result(result)
    assert filtered.get(SPILL_RESERVED_RESULT_KEY) != forged
    assert SPILL_RESERVED_RESULT_KEY not in filtered


@pytest.mark.asyncio
async def test_the_async_path_also_strips_a_forged_reserved_key():
    async def run_json_async(args):
        forged = [{"relative_path": "tool-results/evil.json"}]
        return {"output": "ok", SPILL_RESERVED_RESULT_KEY: forged}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=None,
    )
    filtered = await wrapper.run_json_async({})
    assert SPILL_RESERVED_RESULT_KEY not in filtered


def test_a_real_report_key_survives_while_a_forged_one_is_replaced(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    forged = [{"relative_path": "tool-results/evil.json"}]
    result = {
        "content": [{"type": "text", "text": "x" * 100}],
        "is_error": False,
        SPILL_RESERVED_RESULT_KEY: forged,
    }
    filtered = wrapper._filter_result(result)
    records = filtered[SPILL_RESERVED_RESULT_KEY]
    assert records != forged
    assert records[0]["value_path"] == "content[0].text"


def test_the_wrapper_uses_the_spill_module_s_only_failure_classifier():
    assert (
        output_filter_wrapper.is_classified_tool_failure
        is tool_result_spill.is_classified_tool_failure
    )


def test_the_wrapper_module_keeps_no_private_failure_classifier():
    assert not hasattr(output_filter_wrapper, "_is_classified_tool_failure")


# --- the spill entry point runs off the event loop -------------------------


def _thread_recording_stub(sink):
    """Stand in for spill_oversized_values, recording only which thread
    called it and the run_budget it was handed -- not doing any real spill
    work, so thread identity is the one thing these tests measure."""

    def _record(result, target, *, tool_name, max_recursion, run_budget):
        sink["thread"] = threading.get_ident()
        sink["run_budget"] = run_budget
        return result, []

    return _record


@pytest.mark.asyncio
async def test_run_json_async_offloads_the_spill_entry_point_to_a_worker_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    await wrapper.run_json_async({})
    assert sink["thread"] != caller_thread


@pytest.mark.asyncio
async def test_async_func_wrapper_offloads_the_spill_entry_point_to_a_worker_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    async def original(*args, **kwargs):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", func=original)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    await wrapper.func()
    assert sink["thread"] != caller_thread


def test_run_json_sync_keeps_the_spill_entry_point_on_the_calling_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    def run_json_sync(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_sync=run_json_sync)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    wrapper.run_json_sync({})
    assert sink["thread"] == caller_thread


def test_sync_func_wrapper_keeps_the_spill_entry_point_on_the_calling_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    def original(*args, **kwargs):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", func=original)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    wrapper.func()
    assert sink["thread"] == caller_thread


@pytest.mark.asyncio
async def test_a_wrapper_without_a_target_does_not_hop_to_a_worker_thread(
    monkeypatch,
):
    hop_calls = []
    real_to_thread = output_filter_wrapper.asyncio.to_thread

    async def _spy_to_thread(func, *args, **kwargs):
        hop_calls.append(True)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(output_filter_wrapper.asyncio, "to_thread", _spy_to_thread)

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=None,
    )
    await wrapper.run_json_async({})
    assert hop_calls == []


@pytest.mark.asyncio
async def test_the_run_budget_inside_the_worker_thread_is_the_same_object(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    await wrapper.run_json_async({})
    assert sink["run_budget"] is wrapper._spill_run_budget


@pytest.mark.asyncio
async def test_two_wrappers_accumulate_on_one_shared_budget_across_the_thread_hop(
    tmp_path,
):
    spill_dir = tmp_path / "output" / "tool-results"
    budget = SpillRunBudget()

    async def run_json_async(args):
        return {"content": [{"type": "text", "text": "x" * 100}]}

    wrapper_one = OutputFilteredToolWrapper(
        target_tool=SimpleNamespace(name="one", run_json_async=run_json_async),
        max_chars=80,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80),
        spill_run_budget=budget,
    )
    wrapper_two = OutputFilteredToolWrapper(
        target_tool=SimpleNamespace(name="two", run_json_async=run_json_async),
        max_chars=80,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80),
        spill_run_budget=budget,
    )
    await wrapper_one.run_json_async({})
    await wrapper_two.run_json_async({})
    assert budget.files_written == 2


# --- a failed spill degrades to plain filtering -----------------------------


class _SpillHostileValue:
    """Filtered without str(), but json.dumps can only render it with str()."""

    def model_dump(self) -> dict[str, str]:
        return {"kind": "unrenderable"}

    def __str__(self) -> str:
        raise RuntimeError("this value cannot be rendered")

    __repr__ = __str__


def test_a_value_that_cannot_be_rendered_falls_back_to_plain_filtering(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    no_target = _wrapper(spill_target=None, max_chars=80)
    with_target = _wrapper(
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80), max_chars=80
    )
    baseline = no_target._filter_result(
        {"payload": _SpillHostileValue(), "note": "n" * 200}
    )
    degraded = with_target._filter_result(
        {"payload": _SpillHostileValue(), "note": "n" * 200}
    )
    assert degraded == baseline


def test_a_value_that_cannot_be_rendered_leaves_no_spill_file(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80), max_chars=80
    )
    wrapper._filter_result({"payload": _SpillHostileValue(), "note": "n" * 200})
    assert not spill_dir.exists()


def test_a_failed_spill_logs_one_warning_naming_the_tool_and_the_exception_type(
    tmp_path, caplog
):
    """Scoped to output_filter_wrapper's own logger: rendering the same
    hostile value also trips an unrelated, pre-existing warning inside the
    output filter's own Pydantic-model fallback (it tries to reconstruct
    the value from its filtered model_dump() and that constructor call
    fails too) -- a fact about that filter, not about the spill boundary
    this test is pinning."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80), max_chars=80
    )
    with caplog.at_level("WARNING"):
        wrapper._filter_result({"payload": _SpillHostileValue(), "note": "n" * 200})
    warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and r.name == output_filter_wrapper.__name__
    ]
    assert len(warnings) == 1
    assert "acme" in warnings[0].getMessage()
    assert "RuntimeError" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_cancelling_an_async_tool_call_is_not_swallowed_by_the_spill_boundary(
    monkeypatch, tmp_path, caplog
):
    def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _raise_cancelled
    )

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    with caplog.at_level("WARNING"):
        with pytest.raises(asyncio.CancelledError):
            await wrapper.run_json_async({})
    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_a_keyboard_interrupt_is_not_swallowed_by_the_spill_boundary(
    monkeypatch, tmp_path, caplog
):
    def _raise_keyboard_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _raise_keyboard_interrupt
    )

    def run_json_sync(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_sync=run_json_sync)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    with caplog.at_level("WARNING"):
        with pytest.raises(KeyboardInterrupt):
            wrapper.run_json_sync({})
    assert not any(r.levelname == "WARNING" for r in caplog.records)


# --- wiring the spill path in changes nothing yet ---------------------------


@pytest.mark.asyncio
async def test_a_production_tool_set_wires_no_spill_target(tmp_path):
    """No production caller constructs a SpillTarget: factory.py is not
    part of this change, so a workspace-bound tool set -- the shape a later
    change resolves a real target from -- still wires every wrapper's
    spill target to None. This is the executable form of "this change is
    inert."."""
    config = ToolConfig(
        {
            "workspace": {"task_id": "spill-wiring-test", "base_dir": str(tmp_path)},
        }
    )
    tools = await ToolFactory.create_all_tools(config)
    checked = 0
    for tool in tools:
        if hasattr(tool, "_spill_target"):
            checked += 1
            assert tool._spill_target is None
    assert checked > 0
    assert not (tmp_path / "output" / "tool-results").exists()


# --- the engine's own spill report bypasses ordinary output filtering ------


def test_a_short_character_limit_leaves_the_spill_report_intact(tmp_path):
    """A generated relative_path (about 55 characters for a short tool name)
    is longer than max_chars=50. The engine's report is not subject to that
    cap: the record stays well-formed, ExecutionContext registers it and its
    notice is rendered."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=50, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50)
    )
    filtered = wrapper._filter_result({"content": "x" * 200})

    record = filtered[SPILL_RESERVED_RESULT_KEY][0]
    assert spill_record_shape_is_valid(record)
    written = spill_dir / record["relative_path"].split("/")[-1]
    assert written.exists()

    ctx = ExecutionContext()
    ctx.attach_workspace("ws-1", str(tmp_path))
    tool = ctx.add_tool_result("acme", filtered)
    assert len(ctx.spilled_results) == 1
    assert record["relative_path"] in tool.content


def test_a_field_count_limit_does_not_drop_the_spill_report(tmp_path):
    """Regression for the report key being appended after every tool key
    and then counted against max_fields: a root that already has
    max_fields keys pushed the report past the cutoff and the field-count
    truncation marker took its place, even though nothing about the
    result's own size called for it."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80,
        max_fields=2,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80),
    )
    filtered = wrapper._filter_result({"a": "x" * 200, "b": "small"})

    assert SPILL_RESERVED_RESULT_KEY in filtered
    assert spill_record_shape_is_valid(filtered[SPILL_RESERVED_RESULT_KEY][0])
    assert filtered["a"] != "x" * 200  # the oversized value was spilled
    assert filtered["b"] == "small"
    assert not any(str(key).endswith("more keys") for key in filtered)


@pytest.mark.asyncio
async def test_the_async_path_also_keeps_the_spill_report_intact(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"

    async def run_json_async(args):
        return {"content": "x" * 200}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50),
    )
    filtered = await wrapper.run_json_async({})

    record = filtered[SPILL_RESERVED_RESULT_KEY][0]
    assert spill_record_shape_is_valid(record)
    written = spill_dir / record["relative_path"].split("/")[-1]
    assert written.exists()

    ctx = ExecutionContext()
    ctx.attach_workspace("ws-1", str(tmp_path))
    tool = ctx.add_tool_result("acme", filtered)
    assert len(ctx.spilled_results) == 1
    assert record["relative_path"] in tool.content


def test_a_forged_report_key_is_stripped_with_a_target_when_nothing_spills(tmp_path):
    """With a real report bypassing the output filter, stripping a
    tool-supplied report key in _spill_only is the only guard against a
    forged key. It strips even when a spill target is configured and the
    result is too small to spill."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50))
    forged = [{"relative_path": "tool-results/evil.json"}]
    result = {"output": "small value", SPILL_RESERVED_RESULT_KEY: forged}
    filtered = wrapper._filter_result(result)

    assert SPILL_RESERVED_RESULT_KEY not in filtered
    assert not spill_dir.exists()
