"""
Test DAG plan generation retry mechanism for tool validation failures.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from xagent.core.agent.exceptions import DAGPlanGenerationError
from xagent.core.agent.pattern.dag_plan_execute.models import ExecutionPlan, PlanStep
from xagent.core.agent.pattern.dag_plan_execute.plan_generator import PlanGenerator


@pytest.fixture
def mock_llm():
    """Create a mock LLM with proper async generator support"""

    # Create the mock instance first
    mock_llm_instance = AsyncMock()
    mock_llm_instance.chat = AsyncMock()

    async def mock_stream_chat(**kwargs):
        """Mock stream_chat that yields a single chunk"""
        from xagent.core.model.chat.types import ChunkType, StreamChunk

        # Get the response from chat mock
        chat_result = mock_llm_instance.chat(**kwargs)
        # Handle both coroutines and direct values
        if hasattr(chat_result, "__await__"):
            response = await chat_result
        else:
            response = chat_result

        content = (
            response.get("content", "") if isinstance(response, dict) else response
        )

        yield StreamChunk(
            type=ChunkType.TOKEN,
            content=content,
            delta=content,
        )

    mock_llm_instance.stream_chat = mock_stream_chat
    return mock_llm_instance


@pytest.fixture
def mock_tracer():
    """Create a mock tracer"""
    tracer = MagicMock()
    tracer.trace_event = AsyncMock()
    return tracer


@pytest.fixture
def mock_tool():
    """Create a mock tool"""
    tool = MagicMock()
    tool.metadata.name = "write_file"
    tool.name = "write_file"
    tool.metadata.description = "Write content to a file"
    return tool


@pytest.mark.asyncio
async def test_plan_validation_retry_success(mock_llm, mock_tracer, mock_tool):
    """Test that plan generation retries successfully when tool validation fails initially"""

    # First response has invalid tool, second response has valid tool
    first_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step",
                    "tool_names": ["nonexistent_tool"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    second_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": first_response},
        {"content": second_response},
    ]

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should succeed after retry
    plan = await plan_generator.generate_plan(
        goal="Test goal",
        tools=tools,
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    # Verify the plan
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool_names == ["write_file"]
    assert step.name == "Test Step"
    assert step.description == "A test step with valid tool"

    # Verify LLM was called twice (initial + retry)
    assert mock_llm.chat.call_count == 2

    # Verify the retry call included error context
    retry_call = mock_llm.chat.call_args_list[1]
    retry_messages = retry_call[1]["messages"]
    retry_user_message = retry_messages[-1]["content"]

    assert "PREVIOUS ERROR INFORMATION" in retry_user_message
    assert "Error Type: DAGPlanGenerationError" in retry_user_message
    assert "nonexistent_tool" in retry_user_message
    assert "write_file" in retry_user_message
    assert "Available Tools" in retry_user_message


@pytest.mark.asyncio
async def test_plan_validation_max_retries_exhausted(mock_llm, mock_tracer, mock_tool):
    """Test that plan generation fails after max retries when validation keeps failing"""

    # Always return invalid tool
    invalid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step",
                    "tool_names": ["nonexistent_tool"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": invalid_response}

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should fail after max retries
    with pytest.raises(DAGPlanGenerationError) as exc_info:
        await plan_generator.generate_plan(
            goal="Test goal",
            tools=tools,
            iteration=1,
            history=[],
            tracer=mock_tracer,
            context=None,
        )

    # Verify error message
    assert "Generated plan references non-existent tools" in str(exc_info.value)

    # Verify LLM was called 3 times (initial + 2 retries)
    assert mock_llm.chat.call_count == 3

    # Verify all retry attempts included error context
    for i in range(1, 3):  # Retry calls at indices 1 and 2
        retry_call = mock_llm.chat.call_args_list[i]
        retry_messages = retry_call[1]["messages"]
        retry_user_message = retry_messages[-1]["content"]

        assert "PREVIOUS ERROR INFORMATION" in retry_user_message
        assert "Error Type: DAGPlanGenerationError" in retry_user_message


@pytest.mark.asyncio
async def test_plan_validation_no_retry_on_first_success(
    mock_llm, mock_tracer, mock_tool
):
    """Test that no retry happens when first plan validation succeeds"""

    valid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": valid_response}

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should succeed without retry
    plan = await plan_generator.generate_plan(
        goal="Test goal",
        tools=tools,
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    # Verify the plan
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_names == ["write_file"]

    # Verify LLM was called only once (no retry)
    assert mock_llm.chat.call_count == 1


@pytest.mark.asyncio
async def test_plan_extension_retry_success(mock_llm, mock_tracer, mock_tool):
    """Test retry mechanism for plan extension"""

    # Create current plan
    current_step = PlanStep(
        id="existing_step",
        name="Existing Step",
        description="An existing step",
        tool_names=[],
        dependencies=[],
        difficulty="easy",
    )
    current_plan = ExecutionPlan(
        id=str(uuid4()), goal="Existing goal", steps=[current_step], iteration=1
    )

    # First extension response has invalid tool, second has valid tool
    first_response = """{
        "plan": {
            "steps": [
                {
                    "id": "new_step",
                    "name": "New Step",
                    "description": "A new step with invalid tool",
                    "tool_names": ["nonexistent_tool"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    second_response = """{
        "plan": {
            "steps": [
                {
                    "id": "new_step",
                    "name": "New Step",
                    "description": "A new step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": first_response},
        {"content": second_response},
    ]

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should succeed after retry
    additional_steps = await plan_generator.extend_plan(
        goal="Test goal",
        tools=tools,
        iteration=2,
        history=[],
        current_plan=current_plan,
        tracer=mock_tracer,
        context=None,
    )

    # Verify the additional steps
    assert len(additional_steps) == 1
    step = additional_steps[0]
    assert step.tool_names == ["write_file"]
    assert step.name == "New Step"
    assert step.description == "A new step with valid tool"
    assert step.dependencies == ["existing_step"]

    # Verify LLM was called twice (initial + retry)
    assert mock_llm.chat.call_count == 2


@pytest.mark.asyncio
async def test_plan_validation_multiple_missing_tools(mock_llm, mock_tracer):
    """Test retry mechanism with multiple missing tools"""

    # Create multiple valid tools
    write_tool = MagicMock()
    write_tool.metadata.name = "write_file"
    write_tool.name = "write_file"

    read_tool = MagicMock()
    read_tool.metadata.name = "read_file"
    read_tool.name = "read_file"

    tools = [write_tool, read_tool]

    # Response with multiple missing tools
    first_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "description": "First step with missing tool",
                    "tool_names": ["missing_tool_1"],
                    "dependencies": [],
                    "difficulty": "easy"
                },
                {
                    "id": "step2",
                    "name": "Step 2",
                    "description": "Second step with missing tool",
                    "tool_names": ["missing_tool_2"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    # Valid response retry
    second_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "description": "First step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                },
                {
                    "id": "step2",
                    "name": "Step 2",
                    "description": "Second step with valid tool",
                    "tool_names": ["read_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": first_response},
        {"content": second_response},
    ]

    plan_generator = PlanGenerator(mock_llm)

    # Should succeed after retry
    plan = await plan_generator.generate_plan(
        goal="Test goal",
        tools=tools,
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    # Verify the plan
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_names == ["write_file"]
    assert plan.steps[1].tool_names == ["read_file"]

    # Verify LLM was called twice (initial + retry)
    assert mock_llm.chat.call_count == 2

    # Verify retry message includes both missing tools
    retry_call = mock_llm.chat.call_args_list[1]
    retry_messages = retry_call[1]["messages"]
    retry_user_message = retry_messages[-1]["content"]

    assert "missing_tool_1" in retry_user_message
    assert "missing_tool_2" in retry_user_message
    assert "write_file, read_file" in retry_user_message


@pytest.mark.asyncio
async def test_duplicate_step_ids_retry_success(mock_llm, mock_tracer, mock_tool):
    """Duplicate IDs in one generated plan should trigger retry before execution."""

    duplicate_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Repeated Name",
                    "description": "First duplicated id",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                },
                {
                    "id": "step1",
                    "name": "Repeated Name",
                    "description": "Second duplicated id",
                    "tool_names": ["write_file"],
                    "dependencies": ["step1"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    valid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Repeated Name",
                    "description": "First unique id",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                },
                {
                    "id": "step2",
                    "name": "Repeated Name",
                    "description": "Second unique id",
                    "tool_names": ["write_file"],
                    "dependencies": ["step1"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": duplicate_response},
        {"content": valid_response},
    ]

    plan_generator = PlanGenerator(mock_llm)
    plan = await plan_generator.generate_plan(
        goal="Test duplicate ids",
        tools=[mock_tool],
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    assert [step.id for step in plan.steps] == ["step1", "step2"]
    assert [step.name for step in plan.steps] == ["Repeated Name", "Repeated Name"]
    assert mock_llm.chat.call_count == 2

    retry_user_message = mock_llm.chat.call_args_list[1][1]["messages"][-1]["content"]
    assert "STEP ID VALIDATION ERRORS" in retry_user_message
    assert "step1" in retry_user_message


@pytest.mark.asyncio
async def test_missing_step_id_retries_instead_of_defaulting(
    mock_llm, mock_tracer, mock_tool
):
    """Missing step IDs should be rejected and surfaced to the retry prompt."""

    missing_id_response = """{
        "plan": {
            "steps": [
                {
                    "name": "Missing ID",
                    "description": "The model omitted the required id",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    valid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Valid ID",
                    "description": "The retry includes the required id",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": missing_id_response},
        {"content": valid_response},
    ]

    plan_generator = PlanGenerator(mock_llm)
    plan = await plan_generator.generate_plan(
        goal="Test missing ids",
        tools=[mock_tool],
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    assert [step.id for step in plan.steps] == ["step1"]
    assert mock_llm.chat.call_count == 2

    retry_user_message = mock_llm.chat.call_args_list[1][1]["messages"][-1]["content"]
    assert "Missing ID step indices: [0]" in retry_user_message


@pytest.mark.asyncio
async def test_step_references_are_normalized_before_dependency_validation(
    mock_llm, mock_tracer, mock_tool
):
    """Whitespace-padded step IDs, dependencies, and branch targets should match."""

    response = """{
        "plan": {
            "steps": [
                {
                    "id": " step1 ",
                    "name": "Branch",
                    "description": "Branches to the second step",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy",
                    "conditional_branches": {"continue": " step2 "}
                },
                {
                    "id": " step2 ",
                    "name": "Dependent",
                    "description": "Depends on the first step",
                    "tool_names": ["write_file"],
                    "dependencies": [" step1 "],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": response}

    plan_generator = PlanGenerator(mock_llm)
    plan = await plan_generator.generate_plan(
        goal="Test normalized refs",
        tools=[mock_tool],
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    assert [step.id for step in plan.steps] == ["step1", "step2"]
    assert plan.steps[0].conditional_branches == {"continue": "step2"}
    assert plan.steps[1].dependencies == ["step1"]
    assert mock_llm.chat.call_count == 1


@pytest.mark.asyncio
async def test_extension_reserved_step_id_retry_success(
    mock_llm, mock_tracer, mock_tool
):
    """Extension steps must not reuse IDs from the current task plan."""

    current_plan = ExecutionPlan(
        id=str(uuid4()),
        goal="Existing goal",
        steps=[
            PlanStep(
                id="existing_step",
                name="Existing Step",
                description="Already exposed step",
                tool_names=[],
                dependencies=[],
                difficulty="easy",
            )
        ],
        iteration=1,
    )

    reserved_response = """{
        "plan": {
            "steps": [
                {
                    "id": "existing_step",
                    "name": "New Work",
                    "description": "Incorrectly reuses an existing id",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    valid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "new_step",
                    "name": "New Work",
                    "description": "Uses a unique id",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": reserved_response},
        {"content": valid_response},
    ]

    plan_generator = PlanGenerator(mock_llm)
    additional_steps = await plan_generator.extend_plan(
        goal="Test extension",
        tools=[mock_tool],
        iteration=2,
        history=[],
        current_plan=current_plan,
        tracer=mock_tracer,
        context=None,
    )

    assert [step.id for step in additional_steps] == ["new_step"]
    assert additional_steps[0].dependencies == ["existing_step"]
    assert mock_llm.chat.call_count == 2

    initial_messages = mock_llm.chat.call_args_list[0][1]["messages"]
    initial_prompt = "\n".join(message["content"] for message in initial_messages)
    assert "FORBIDDEN_STEP_IDS" in initial_prompt
    assert "existing_step" in initial_prompt

    retry_user_message = mock_llm.chat.call_args_list[1][1]["messages"][-1]["content"]
    assert "Reserved step IDs used" in retry_user_message
    assert "existing_step" in retry_user_message


@pytest.mark.asyncio
async def test_extension_does_not_revalidate_historical_step_tools(
    mock_llm, mock_tracer, mock_tool
):
    """Extension uniqueness checks should not revalidate old step tool availability."""

    current_plan = ExecutionPlan(
        id=str(uuid4()),
        goal="Existing goal",
        steps=[
            PlanStep(
                id="existing_step",
                name="Existing Step",
                description="Already planned with a tool unavailable now",
                tool_names=["old_tool"],
                dependencies=[],
                difficulty="easy",
            )
        ],
        iteration=1,
    )

    valid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "new_step",
                    "name": "New Work",
                    "description": "Uses a currently available tool",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": valid_response}

    plan_generator = PlanGenerator(mock_llm)
    additional_steps = await plan_generator.extend_plan(
        goal="Test extension",
        tools=[mock_tool],
        iteration=2,
        history=[],
        current_plan=current_plan,
        tracer=mock_tracer,
        context=None,
    )

    assert [step.id for step in additional_steps] == ["new_step"]
    assert additional_steps[0].dependencies == ["existing_step"]
    assert mock_llm.chat.call_count == 1


@pytest.mark.asyncio
async def test_extension_invalid_conditional_branch_target_retries(
    mock_llm, mock_tracer, mock_tool
):
    """Extension branch targets must resolve against historical and new steps."""

    current_plan = ExecutionPlan(
        id=str(uuid4()),
        goal="Existing goal",
        steps=[
            PlanStep(
                id="existing_step",
                name="Existing Step",
                description="Already exposed step",
                tool_names=[],
                dependencies=[],
                difficulty="easy",
            )
        ],
        iteration=1,
    )

    invalid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "branch_step",
                    "name": "Branch Work",
                    "description": "Branches to a missing step",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy",
                    "conditional_branches": {"continue": "missing_step"}
                }
            ]
        }
    }"""

    valid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "branch_step",
                    "name": "Branch Work",
                    "description": "Branches to a valid new step",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy",
                    "conditional_branches": {"continue": "followup_step"}
                },
                {
                    "id": "followup_step",
                    "name": "Follow Up",
                    "description": "Handles the branch",
                    "tool_names": ["write_file"],
                    "dependencies": ["branch_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": invalid_response},
        {"content": valid_response},
    ]

    plan_generator = PlanGenerator(mock_llm)
    additional_steps = await plan_generator.extend_plan(
        goal="Test extension",
        tools=[mock_tool],
        iteration=2,
        history=[],
        current_plan=current_plan,
        tracer=mock_tracer,
        context=None,
    )

    assert [step.id for step in additional_steps] == ["branch_step", "followup_step"]
    assert additional_steps[0].conditional_branches == {"continue": "followup_step"}
    assert mock_llm.chat.call_count == 2

    retry_user_message = mock_llm.chat.call_args_list[1][1]["messages"][-1]["content"]
    assert "conditional branches pointing to non-existent steps" in retry_user_message
    assert "missing_step" in retry_user_message


@pytest.mark.asyncio
async def test_step_id_conflicts_fallback_rewrite_after_retries(
    mock_llm, mock_tracer, mock_tool
):
    """After retries are exhausted, duplicate step IDs are deterministically rewritten."""

    conflicting_response = """{
        "plan": {
            "steps": [
                {
                    "id": "analyze",
                    "name": "Analyze",
                    "description": "First analyze step",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy",
                    "conditional_branches": {"continue": "analyze"}
                },
                {
                    "id": "analyze",
                    "name": "Analyze",
                    "description": "Second analyze step",
                    "tool_names": ["write_file"],
                    "dependencies": ["analyze"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": conflicting_response}

    plan_generator = PlanGenerator(mock_llm)
    plan = await plan_generator.generate_plan(
        goal="Test fallback",
        tools=[mock_tool],
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    assert [step.id for step in plan.steps] == ["analyze", "analyze_2"]
    assert plan.steps[0].dependencies == []
    assert plan.steps[0].conditional_branches == {"continue": "analyze"}
    assert plan.steps[1].dependencies == ["analyze"]
    assert mock_llm.chat.call_count == 3


@pytest.mark.asyncio
async def test_reserved_step_id_conflicts_fail_after_retries(
    mock_llm, mock_tracer, mock_tool
):
    """Reserved step IDs are not rewritten because references are ambiguous."""

    conflicting_response = """{
        "plan": {
            "steps": [
                {
                    "id": "existing_step",
                    "name": "Existing Conflict",
                    "description": "Conflicts with historical id",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": conflicting_response}

    history = [
        {
            "role": "user",
            "content": '{"plan": {"steps": [{"id": "existing_step"}]}}',
        }
    ]
    plan_generator = PlanGenerator(mock_llm)
    with pytest.raises(DAGPlanGenerationError) as exc_info:
        await plan_generator.generate_plan(
            goal="Test reserved fallback failure",
            tools=[mock_tool],
            iteration=1,
            history=history,
            tracer=mock_tracer,
            context=None,
        )

    assert "reserved step IDs" in str(exc_info.value)
    assert exc_info.value.context["reserved_step_ids"] == [
        {"id": "existing_step", "index": 0}
    ]
    assert mock_llm.chat.call_count == 3


def test_collect_forbidden_step_ids_skips_non_json_history_content(mock_llm):
    """Plain conversation history should not be parsed as JSON while collecting IDs."""

    plan_generator = PlanGenerator(mock_llm)
    forbidden_step_ids = plan_generator._collect_forbidden_step_ids(
        [
            {"role": "user", "content": "please analyze the dataset"},
            {
                "role": "assistant",
                "content": '{"plan": {"steps": [{"id": "historical_step"}]}}',
            },
        ]
    )

    assert forbidden_step_ids == ["historical_step"]


def test_collect_forbidden_step_ids_reads_markdown_json_history_content(mock_llm):
    """Markdown-wrapped JSON history should still contribute forbidden step IDs."""

    plan_generator = PlanGenerator(mock_llm)
    forbidden_step_ids = plan_generator._collect_forbidden_step_ids(
        [
            {
                "role": "assistant",
                "content": """Here is the plan:
```json
{"plan": {"steps": [{"id": "markdown_step"}]}}
```
""",
            }
        ]
    )

    assert forbidden_step_ids == ["markdown_step"]


def test_validate_plan_rejects_duplicate_step_ids(mock_llm):
    """Plan validation should reject duplicate step IDs even when called directly."""

    plan = ExecutionPlan(
        id=str(uuid4()),
        goal="Duplicate validation",
        steps=[
            PlanStep(
                id="dup",
                name="First",
                description="First step",
                tool_names=[],
                dependencies=[],
            ),
            PlanStep(
                id="dup",
                name="Second",
                description="Second step",
                tool_names=[],
                dependencies=[],
            ),
        ],
    )

    plan_generator = PlanGenerator(mock_llm)
    with pytest.raises(DAGPlanGenerationError) as exc_info:
        plan_generator._validate_plan(plan, tools=[])

    assert "duplicate or empty step IDs" in str(exc_info.value)


def test_validate_plan_allows_duplicate_names(mock_llm):
    """Step names are display labels and may repeat when IDs are unique."""

    plan = ExecutionPlan(
        id=str(uuid4()),
        goal="Repeated names",
        steps=[
            PlanStep(
                id="step_a",
                name="Analyze",
                description="First analysis",
                tool_names=[],
                dependencies=[],
            ),
            PlanStep(
                id="step_b",
                name="Analyze",
                description="Second analysis",
                tool_names=[],
                dependencies=["step_a"],
            ),
        ],
    )

    plan_generator = PlanGenerator(mock_llm)
    plan_generator._validate_plan(plan, tools=[])
