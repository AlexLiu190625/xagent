from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import (
    ToolSelectionSpec,
    without_published_agent_tools,
)
from xagent.web.models.task import TaskStatus
from xagent.web.services.agent_service_manager import (
    AgentServiceManager,
    _spec_wants_mcp,
)
from xagent.web.services.channel_runtime import ChannelTaskMode
from xagent.web.services.mcp_runtime import (
    MCPBuiltinOAuthActorPolicy,
    MCPBuiltinOAuthActorPolicyMismatchError,
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
)
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskReconstructionSnapshot,
    TaskSetupSnapshot,
    _TaskFields,
)


def _snapshot(
    *,
    marker: Any = True,
    status: TaskStatus = TaskStatus.PENDING,
    has_reconstructable_history: bool = False,
    agent_config: dict[str, Any] | None = None,
    task_agent_id: int | None = None,
    task_agent_config: dict[str, Any] | None = None,
    runtime_agent: Any = None,
    task_llm: Any = None,
    run_id: str | None = None,
) -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=status,
            source="external",
            agent_id=task_agent_id,
            agent_config=(
                task_agent_config
                if task_agent_config is not None
                else {
                    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: marker,
                    **(
                        {"mcp_runtime_authorization_policy_identity": "actor:alice"}
                        if marker is True
                        else {}
                    ),
                }
            ),
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
            run_id=run_id,
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=has_reconstructable_history,
        task_pattern="single_call",
        task_llm=task_llm,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=runtime_agent,
        agent_config=agent_config,
        excluded_agent_id=None,
        reconstruction=TaskReconstructionSnapshot(
            tracer_events=(
                (
                    {
                        "id": "actor-event",
                        "event_type": "agent_step",
                        "task_id": "42",
                        "step_id": None,
                        "timestamp": None,
                        "data": {},
                        "parent_id": None,
                    },
                )
                if has_reconstructable_history
                else ()
            ),
            has_history=has_reconstructable_history,
        ),
    )


class _Agent:
    def __init__(self, tool_config: Any) -> None:
        self.tool_config = tool_config
        self.workspace = None
        self.invalidate_tools = MagicMock()

    def set_conversation_history(
        self,
        _messages: list[dict[str, Any]],
        *,
        watermark: int | None = None,
        event_watermark: dict[str, Any] | None = None,
    ) -> None: ...

    def set_execution_context_messages(self, _messages: list[Any]) -> None: ...

    def set_recovered_skill_context(self, _context: Any) -> None: ...

    async def reconstruct_from_history(self, *_args: Any) -> None: ...

    def cleanup_workspace(self) -> None: ...


@pytest.fixture
def actor_policy() -> MCPBuiltinOAuthActorPolicy:
    return MCPBuiltinOAuthActorPolicy(resource_owner_key="actor:alice")


@pytest.mark.asyncio
async def test_marked_task_requires_policy_before_tool_construction() -> None:
    manager = AgentServiceManager()
    create_tools = AsyncMock()

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        pytest.raises(
            MCPBuiltinOAuthActorPolicyRequiredError,
            match="requires an MCP runtime authorization policy",
        ),
    ):
        await manager.get_agent_for_task(
            42,
            task_setup_snapshot=_snapshot(),
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    create_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_marked_task_binds_policy_and_omits_published_agent_tools(
    actor_policy: MCPBuiltinOAuthActorPolicy,
) -> None:
    manager = AgentServiceManager()
    tool_config = MagicMock()
    tool_config.set_execution_scope.return_value = False
    create_tools = AsyncMock(return_value=([], tool_config))
    agent = _Agent(tool_config)

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch(
            "xagent.web.services.agent_service_manager.AgentService", return_value=agent
        ),
    ):
        built = await manager.get_agent_for_task(
            42,
            task_setup_snapshot=_snapshot(
                agent_config={
                    "knowledge_bases": [],
                    "skills": [],
                    "tool_categories": ["web_search", "file"],
                }
            ),
            task_owner_user_id=1,
            mcp_runtime_authorization_policy=actor_policy,
            resolved_execution_scope=None,
        )

    assert built is agent
    kwargs = create_tools.await_args.kwargs
    assert kwargs["mcp_runtime_authorization_policy"] is actor_policy
    assert _spec_wants_mcp(kwargs["tool_selection_spec"])
    assert not kwargs["tool_selection_spec"].includes_published_agent()
    assert manager._mcp_actor_policies[42] is actor_policy


@pytest.mark.asyncio
async def test_actor_interaction_rejects_different_persisted_policy(
    actor_policy: MCPBuiltinOAuthActorPolicy,
) -> None:
    manager = AgentServiceManager()
    create_tools = AsyncMock()

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        pytest.raises(
            MCPBuiltinOAuthActorPolicyMismatchError,
            match="durable identity",
        ),
    ):
        await manager.get_agent_for_task(
            42,
            task_setup_snapshot=_snapshot(
                status=TaskStatus.RUNNING,
                has_reconstructable_history=True,
                task_agent_id=7,
                runtime_agent=MagicMock(id=7),
                task_agent_config={
                    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True,
                    "mcp_runtime_authorization_policy_identity": "actor:bob",
                },
            ),
            task_owner_user_id=1,
            mcp_runtime_authorization_policy=actor_policy,
            task_mode=ChannelTaskMode.ACTOR_INTERACTION,
            resolved_execution_scope=None,
        )

    create_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_actor_interaction_rejects_deleted_claimed_agent(
    actor_policy: MCPBuiltinOAuthActorPolicy,
) -> None:
    manager = AgentServiceManager()
    create_tools = AsyncMock()

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        pytest.raises(
            MCPBuiltinOAuthActorPolicyRequiredError,
            match="claimed agent is unavailable",
        ),
    ):
        await manager.get_agent_for_task(
            42,
            task_setup_snapshot=_snapshot(
                status=TaskStatus.RUNNING,
                has_reconstructable_history=True,
                task_agent_id=7,
                runtime_agent=None,
            ),
            task_owner_user_id=1,
            mcp_runtime_authorization_policy=actor_policy,
            task_mode=ChannelTaskMode.ACTOR_INTERACTION,
            resolved_execution_scope=None,
        )

    create_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_published_agent_factory_is_never_reached_for_actor_selection() -> None:
    published_factory = AsyncMock(return_value=[])
    ordinary_factory = AsyncMock(return_value=[])
    spec = without_published_agent_tools(
        ToolSelectionSpec.from_raw(tool_categories=None)
    )
    config = MagicMock()
    config.get_tool_selection_spec.return_value = spec

    with (
        patch.object(ToolRegistry, "_import_tool_modules"),
        patch.object(
            ToolRegistry,
            "_tool_creators",
            [
                (published_factory, frozenset({"agent"}), "published_agent"),
                (ordinary_factory, frozenset({"basic"}), None),
            ],
        ),
    ):
        assert await ToolRegistry.create_registered_tools(config) == []

    published_factory.assert_not_awaited()
    ordinary_factory.assert_awaited_once_with(config)


@pytest.mark.asyncio
async def test_marked_warm_reuse_without_policy_and_policy_mismatch_fail_closed(
    actor_policy: MCPBuiltinOAuthActorPolicy,
) -> None:
    manager = AgentServiceManager()
    cached = MagicMock()
    manager._agents[42] = cached
    manager._agent_owner_ids[42] = 1
    manager._agent_scope_fingerprints[42] = None
    manager._mcp_actor_policies[42] = actor_policy

    with pytest.raises(MCPBuiltinOAuthActorPolicyRequiredError):
        await manager.get_agent_for_task(
            42,
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    different_policy = MCPBuiltinOAuthActorPolicy(resource_owner_key="actor:bob")
    with pytest.raises(MCPBuiltinOAuthActorPolicyMismatchError):
        await manager.get_agent_for_task(
            42,
            task_owner_user_id=1,
            mcp_runtime_authorization_policy=different_policy,
            resolved_execution_scope=None,
        )

    assert manager._agents[42] is cached
    cached.invalidate_tools.assert_not_called()


@pytest.mark.asyncio
async def test_marked_task_reconstruction_is_rejected_even_with_policy(
    actor_policy: MCPBuiltinOAuthActorPolicy,
) -> None:
    manager = AgentServiceManager()
    reconstruct = AsyncMock()

    with (
        patch.object(manager, "_reconstruct_agent_from_history", new=reconstruct),
        pytest.raises(MCPBuiltinOAuthActorPolicyRequiredError, match="reconstruction"),
    ):
        await manager.get_agent_for_task(
            42,
            task_setup_snapshot=_snapshot(
                status=TaskStatus.PAUSED,
                has_reconstructable_history=True,
            ),
            task_owner_user_id=1,
            mcp_runtime_authorization_policy=actor_policy,
            resolved_execution_scope=None,
        )

    reconstruct.assert_not_awaited()


@pytest.mark.asyncio
async def test_actor_interaction_reconstruction_preserves_tool_context(
    actor_policy: MCPBuiltinOAuthActorPolicy,
) -> None:
    manager = AgentServiceManager()
    tool_config = MagicMock()
    tool_config.set_execution_scope.return_value = False
    create_tools = AsyncMock(return_value=([], tool_config))
    reconstructed = _Agent(tool_config)

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch(
            "xagent.web.services.agent_service_manager.AgentService",
            return_value=reconstructed,
        ),
    ):
        result = await manager.get_agent_for_task(
            42,
            task_setup_snapshot=_snapshot(
                status=TaskStatus.RUNNING,
                has_reconstructable_history=True,
                agent_config={
                    "knowledge_bases": [],
                    "skills": [],
                    "tool_categories": ["web_search", "file"],
                },
                task_llm=MagicMock(),
            ),
            task_owner_user_id=1,
            connector_runtime_turn_id="approval-turn",
            mcp_runtime_authorization_policy=actor_policy,
            task_mode=ChannelTaskMode.ACTOR_INTERACTION,
            resolved_execution_scope=None,
        )

    assert result is reconstructed
    kwargs = create_tools.await_args.kwargs
    assert kwargs["connector_runtime_turn_id"] == "approval-turn"
    assert kwargs["mcp_runtime_authorization_policy"] is actor_policy
    assert kwargs["force_mcp_tools"] is True
    assert _spec_wants_mcp(kwargs["tool_selection_spec"])
    assert not kwargs["tool_selection_spec"].includes_published_agent()


@pytest.mark.asyncio
async def test_actor_interaction_reconstruction_requires_running_claim(
    actor_policy: MCPBuiltinOAuthActorPolicy,
) -> None:
    manager = AgentServiceManager()
    reconstruct = AsyncMock()

    with (
        patch.object(manager, "_reconstruct_agent_from_history", new=reconstruct),
        pytest.raises(MCPBuiltinOAuthActorPolicyRequiredError, match="reconstruction"),
    ):
        await manager.get_agent_for_task(
            42,
            task_setup_snapshot=_snapshot(
                status=TaskStatus.WAITING_FOR_USER,
                has_reconstructable_history=True,
            ),
            task_owner_user_id=1,
            mcp_runtime_authorization_policy=actor_policy,
            task_mode=ChannelTaskMode.ACTOR_INTERACTION,
            resolved_execution_scope=None,
        )

    reconstruct.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [False, None, "true", 1, {}, []])
async def test_non_literal_true_marker_preserves_ordinary_task_behavior(
    marker: Any,
) -> None:
    manager = AgentServiceManager()
    tool_config = MagicMock()
    tool_config.set_execution_scope.return_value = False
    create_tools = AsyncMock(return_value=([], tool_config))
    agent = _Agent(tool_config)

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch(
            "xagent.web.services.agent_service_manager.AgentService", return_value=agent
        ),
    ):
        assert (
            await manager.get_agent_for_task(
                42,
                task_setup_snapshot=_snapshot(marker=marker),
                task_owner_user_id=1,
                resolved_execution_scope=None,
            )
            is agent
        )

    assert create_tools.await_args.kwargs["mcp_runtime_authorization_policy"] is None
    assert create_tools.await_args.kwargs[
        "tool_selection_spec"
    ].includes_published_agent()


def _seed_manager_maps(manager: AgentServiceManager, agent: Any) -> None:
    manager._agents[42] = agent
    manager._agent_owner_ids[42] = 7
    manager._agent_run_ids[42] = "current-run"
    manager._agent_sandbox_keys[42] = "user:7"
    manager._agent_sandbox_providers[42] = object()
    manager._agent_scope_fingerprints[42] = None
    manager._agent_evicted_scope_fingerprints[42] = MagicMock()
    manager._mcp_actor_policies[42] = MCPBuiltinOAuthActorPolicy(
        resource_owner_key="actor:alice"
    )


def _acquire_stubbed_agent(manager: AgentServiceManager, agent: Any) -> int:
    async def build(*_args: Any, **_kwargs: Any) -> Any:
        _seed_manager_maps(manager, agent)
        return agent

    async def acquire() -> int:
        with patch.object(manager, "_get_agent_for_task_unlocked", side_effect=build):
            await manager.get_agent_for_task(
                42,
                task_setup_snapshot=_snapshot(marker=object(), run_id="current-run"),
                task_owner_user_id=1,
                resolved_execution_scope=None,
            )
        generation = manager.current_run_generation(42)
        assert generation is not None
        return generation

    return asyncio.run(acquire())


def test_remove_agent_ignores_stale_cleanup_within_the_same_run() -> None:
    """A resume re-claims its own run, so the run id cannot date a cleanup.

    An approval continuation deliberately claims the run the waiting
    checkpoint was written under -- that is what keeps the checkpoint
    readable. Both the previous turn's runtime and the continuation's then
    carry the same run id, so a late cleanup scheduled by the earlier one
    would match and evict the runtime the continuation is executing in.
    The generation is what tells the two acquisitions apart.
    """
    manager = AgentServiceManager()
    agent = MagicMock()
    _seed_manager_maps(manager, agent)
    # The continuation's acquisition; the stale cleanup below belongs to the
    # previous one, under the very same run id.
    manager._agent_run_generations[42] = 2

    manager.remove_agent(42, expected_run_id="current-run", expected_run_generation=1)

    assert manager._agents[42] is agent
    assert manager._agent_run_generations[42] == 2
    agent.cleanup_workspace.assert_not_called()


def test_remove_agent_evicts_when_the_generation_is_its_own() -> None:
    """The guard must not become fail-stuck: the owning cleanup still runs."""
    manager = AgentServiceManager()
    agent = MagicMock()
    _seed_manager_maps(manager, agent)
    manager._agent_run_generations[42] = 2

    manager.remove_agent(42, expected_run_id="current-run", expected_run_generation=2)

    assert 42 not in manager._agents
    assert 42 not in manager._agent_run_generations
    agent.cleanup_workspace.assert_called_once()


def test_get_agent_for_task_bumps_the_run_generation() -> None:
    """Each acquisition is a new generation, including a same-run resume."""
    manager = AgentServiceManager()
    agent = MagicMock()
    marker = object()

    async def acquire() -> None:
        with (
            patch.object(manager, "_get_agent_for_task_unlocked", return_value=agent),
        ):
            await manager.get_agent_for_task(
                42,
                task_setup_snapshot=_snapshot(marker=marker),
                task_owner_user_id=1,
                resolved_execution_scope=None,
            )

    assert manager.current_run_generation(42) is None
    asyncio.run(acquire())
    first = manager.current_run_generation(42)
    asyncio.run(acquire())
    second = manager.current_run_generation(42)

    assert first is not None
    # The snapshot carries one run id, so only the generation distinguishes
    # these two acquisitions -- which is the whole point of having it.
    assert second == first + 1


def test_run_generation_is_not_reused_after_runtime_eviction() -> None:
    """An old cleanup token cannot become current again after eviction."""
    manager = AgentServiceManager()
    first_agent = MagicMock()
    replacement_agent = MagicMock()

    first = _acquire_stubbed_agent(manager, first_agent)
    second = _acquire_stubbed_agent(manager, first_agent)
    manager.remove_agent(
        42,
        expected_run_id="current-run",
        expected_run_generation=second,
    )
    third = _acquire_stubbed_agent(manager, replacement_agent)

    assert len({first, second, third}) == 3

    manager.remove_agent(
        42,
        expected_run_id="current-run",
        expected_run_generation=first,
    )

    assert manager._agents[42] is replacement_agent
    assert manager.current_run_generation(42) == third
    replacement_agent.cleanup_workspace.assert_not_called()

    manager.remove_agent(
        42,
        expected_run_id="current-run",
        expected_run_generation=third,
    )

    assert 42 not in manager._agents
    assert manager.current_run_generation(42) is None
    replacement_agent.cleanup_workspace.assert_called_once()


def test_run_generation_is_not_reused_after_manager_replacement() -> None:
    """A delayed cleanup stays stale if the global manager is replaced."""
    original_manager = AgentServiceManager()
    replacement_manager = AgentServiceManager()
    original_agent = MagicMock()
    replacement_agent = MagicMock()

    original = _acquire_stubbed_agent(original_manager, original_agent)
    replacement = _acquire_stubbed_agent(replacement_manager, replacement_agent)

    assert replacement != original

    replacement_manager.remove_agent(
        42,
        expected_run_id="current-run",
        expected_run_generation=original,
    )

    assert replacement_manager._agents[42] is replacement_agent
    replacement_agent.cleanup_workspace.assert_not_called()

    replacement_manager.remove_agent(
        42,
        expected_run_id="current-run",
        expected_run_generation=replacement,
    )

    assert 42 not in replacement_manager._agents
    replacement_agent.cleanup_workspace.assert_called_once()


def test_remove_agent_ignores_stale_run_cleanup() -> None:
    manager = AgentServiceManager()
    agent = MagicMock()
    _seed_manager_maps(manager, agent)

    manager.remove_agent(42, expected_run_id="finished-run")

    assert manager._agents[42] is agent
    assert manager._agent_run_ids[42] == "current-run"
    agent.cleanup_workspace.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_remove_agent_evicts_runtime_and_retains_failed_cleanup_owner(
    cleanup_fails: bool,
) -> None:
    manager = AgentServiceManager()
    agent = MagicMock()
    if cleanup_fails:
        agent.cleanup_workspace.side_effect = OSError("directory busy")
    _seed_manager_maps(manager, agent)
    manager._cleanup_workspace_directory = MagicMock()

    if cleanup_fails:
        with pytest.raises(OSError, match="directory busy"):
            manager.remove_agent(42)
    else:
        manager.remove_agent(42)

    for runtime_map in (
        manager._agents,
        manager._agent_owner_ids,
        manager._agent_run_ids,
        manager._agent_sandbox_keys,
        manager._agent_sandbox_providers,
        manager._agent_scope_fingerprints,
        manager._agent_evicted_scope_fingerprints,
        manager._mcp_actor_policies,
    ):
        assert 42 not in runtime_map

    assert manager._agent_cleanup_owner_ids == ({42: 7} if cleanup_fails else {})

    manager.remove_agent(42)
    manager._cleanup_workspace_directory.assert_called_once_with(
        42, 7 if cleanup_fails else None, target=None
    )
    assert 42 not in manager._agent_cleanup_owner_ids
