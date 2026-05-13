"""Test file information propagation from context to step execution."""

from unittest.mock import Mock

import pytest

from xagent.core.agent.pattern.dag_plan_execute.models import PlanStep
from xagent.core.agent.pattern.dag_plan_execute.plan_executor import PlanExecutor
from xagent.core.agent.utils.context_builder import ContextBuilder


@pytest.mark.asyncio
async def test_file_info_propagation_from_context():
    """Test that file information is properly propagated from context to step execution."""
    # Create a mock parent pattern with file information in context
    parent_pattern = Mock()
    parent_pattern._context = {
        "uploaded_files": ["file1.jpg", "file2.png"],
        "file_info": [
            {
                "name": "file1.jpg",
                "size": 1024,
                "type": "image/jpeg",
                "file_id": "11111111-1111-4111-8111-111111111111",
            },
            {
                "name": "file2.png",
                "size": 2048,
                "type": "image/png",
                "file_id": "22222222-2222-4222-8222-222222222222",
            },
        ],
    }

    # Create a ContextBuilder
    llm_mock = Mock()
    llm_mock.model_name = "test-model"
    context_builder = ContextBuilder(llm=llm_mock)

    # Build context for a step with file information
    messages = await context_builder.build_context_for_step(
        step_name="test_step",
        step_description="Test step with file information",
        dependencies=[],
        dependency_results={},
        file_info=parent_pattern._context["file_info"],
        uploaded_files=parent_pattern._context["uploaded_files"],
    )

    # Verify that file information is included in the messages
    assert len(messages) > 1  # System prompt + file information

    # Find the file information message
    file_info_msg = None
    for msg in messages:
        if "UPLOADED FILES" in msg.get("content", ""):
            file_info_msg = msg
            break

    assert file_info_msg is not None, "File information message not found"
    content = file_info_msg["content"]

    # Verify file information is present
    assert "2 files available for processing" in content
    assert "file1.jpg" in content
    assert "file2.png" in content
    assert "1024 bytes" in content
    assert "2048 bytes" in content
    assert "image/jpeg" in content
    assert "image/png" in content
    assert "File ID: 11111111-1111-4111-8111-111111111111" in content
    assert "File ID: 22222222-2222-4222-8222-222222222222" in content
    assert "Absolute Path: file1.jpg" in content


@pytest.mark.asyncio
async def test_plan_executor_retrieves_file_info_from_parent_context():
    """Test that PlanExecutor correctly retrieves file information from parent pattern context."""
    # Create a mock parent pattern with file information as a dict (as passed from websocket)
    parent_pattern = Mock()
    parent_pattern._context = {
        "uploaded_files": ["file1.jpg", "file2.png"],
        "file_info": [
            {
                "name": "file1.jpg",
                "size": 1024,
                "type": "image/jpeg",
                "file_id": "11111111-1111-4111-8111-111111111111",
            },
            {
                "name": "file2.png",
                "size": 2048,
                "type": "image/png",
                "file_id": "22222222-2222-4222-8222-222222222222",
            },
        ],
    }

    # Create a PlanExecutor with the parent pattern
    plan_executor = PlanExecutor(
        llm=Mock(),
        tracer=Mock(),
        workspace=Mock(),
        parent_pattern=parent_pattern,
    )

    # Access the file information through the parent pattern
    file_info = None
    uploaded_files = None

    if plan_executor.parent_pattern and hasattr(
        plan_executor.parent_pattern, "_context"
    ):
        parent_context = plan_executor.parent_pattern._context
        if parent_context:
            if isinstance(parent_context, dict):
                file_info = parent_context.get("file_info")
                uploaded_files = parent_context.get("uploaded_files")

    # Verify file information was retrieved correctly
    assert file_info is not None
    assert uploaded_files is not None
    assert len(file_info) == 2
    assert len(uploaded_files) == 2
    assert file_info[0]["name"] == "file1.jpg"
    assert uploaded_files[0] == "file1.jpg"


@pytest.mark.asyncio
async def test_context_builder_with_empty_file_info():
    """Test that ContextBuilder handles empty file information gracefully."""
    context_builder = ContextBuilder(llm=Mock())

    # Build context without file information
    messages = await context_builder.build_context_for_step(
        step_name="test_step",
        step_description="Test step without file information",
        dependencies=[],
        dependency_results={},
        file_info=None,
        uploaded_files=None,
    )

    # Verify that only system prompt is present
    assert len(messages) == 1  # Only system prompt
    assert messages[0]["role"] == "system"

    # Verify no file information message
    for msg in messages:
        assert "UPLOADED FILES" not in msg.get("content", "")


@pytest.mark.asyncio
async def test_context_builder_agent_builder_uses_file_ids_without_paths():
    """Agent-builder context should expose UUID file_ids and avoid path confusion."""
    context_builder = ContextBuilder(llm=Mock())

    messages = await context_builder.build_context_for_step(
        step_name="create_kb",
        step_description="Create KB from uploaded FAQ",
        dependencies=[],
        dependency_results={},
        file_info=[
            {
                "name": "Velvet_Enterprise_FAQ.docx",
                "size": 4096,
                "type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "file_id": "b341b1e0-e960-4158-8199-b11539025ced",
            }
        ],
        uploaded_files=["/tmp/Velvet_Enterprise_FAQ.docx"],
        is_builder_context=True,
    )

    content = "\n".join(msg.get("content", "") for msg in messages)
    assert "File ID: b341b1e0-e960-4158-8199-b11539025ced" in content
    assert 'file_ids = ["b341b1e0-e960-4158-8199-b11539025ced"]' in content
    assert "Absolute Path:" not in content


def test_plan_executor_extends_agent_builder_tool_closure():
    """Builder steps should keep their declared tools and gain the KB/agent closure."""
    tool_map = {
        "create_agent": Mock(),
        "update_agent": Mock(),
        "ask_user_question": Mock(),
        "list_knowledge_bases": Mock(),
        "create_knowledge_base_from_file": Mock(),
        "create_knowledge_base_from_url": Mock(),
    }

    extended = PlanExecutor._extend_builder_tool_names(["create_agent"], tool_map)

    assert extended == [
        "create_agent",
        "list_knowledge_bases",
        "ask_user_question",
        "create_knowledge_base_from_file",
        "create_knowledge_base_from_url",
        "update_agent",
    ]


def test_plan_executor_resolves_none_tool_names_as_all_tools():
    """Legacy tool_names=None should continue to mean all available tools."""
    tool_map = {
        "create_agent": Mock(),
        "ask_user_question": Mock(),
        "custom_tool": Mock(),
    }
    step = PlanStep(
        id="create_agent",
        name="Create agent",
        description="Create an agent from the uploaded FAQ",
    )
    step.tool_names = None

    resolved = PlanExecutor._resolve_step_tool_names(step, tool_map)

    assert resolved == ["create_agent", "ask_user_question", "custom_tool"]


def test_plan_executor_extends_builder_tool_closure_from_all_tools():
    """Builder closure should preserve all-tools semantics and add missing closure tools."""
    tool_map = {
        "create_agent": Mock(),
        "ask_user_question": Mock(),
        "custom_tool": Mock(),
        "list_knowledge_bases": Mock(),
        "create_knowledge_base_from_file": Mock(),
        "create_knowledge_base_from_url": Mock(),
        "update_agent": Mock(),
    }
    step = PlanStep(
        id="create_agent",
        name="Create agent",
        description="Create an agent from the uploaded FAQ",
    )
    step.tool_names = None

    tool_names = PlanExecutor._resolve_step_tool_names(step, tool_map)
    extended = PlanExecutor._extend_builder_tool_names(tool_names, tool_map)

    assert extended == [
        "create_agent",
        "ask_user_question",
        "custom_tool",
        "list_knowledge_bases",
        "create_knowledge_base_from_file",
        "create_knowledge_base_from_url",
        "update_agent",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
