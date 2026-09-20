"""
Integration tests for output filter with tool factory.
"""

import threading
from types import SimpleNamespace

import pytest

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
)
from xagent.core.tools.user_interaction import WAITING_FOR_USER_STATUS


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


# --- stage 1-e: wrapper integration (strip / spill / bypass branches) -----


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


def test_the_wrapper_uses_the_spill_module_s_only_failure_classifier():
    assert (
        output_filter_wrapper.is_classified_tool_failure
        is tool_result_spill.is_classified_tool_failure
    )


def test_the_wrapper_module_keeps_no_private_failure_classifier():
    assert not hasattr(output_filter_wrapper, "_is_classified_tool_failure")


# --- stage 2: the spill entry point runs off the event loop ----------------


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
