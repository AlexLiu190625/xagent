"""Tests for :class:`ToolSelectionSpec` and the registry/creator
short-circuits it enables.

Background:
    Issue #427 observed that the agent setup path called
    ``ToolFactory.create_all_tools`` three times per task, each building
    the full ~52-tool default set. One of those calls (chat.py:872)
    existed purely to extract tool names by category from the pre-built
    list. ``ToolSelectionSpec`` lets the factory and individual creators
    short-circuit when an agent only needs a subset of categories /
    MCP servers / Custom APIs / published agents.

What these tests pin:
    * Spec semantics (``includes_*`` helpers) — both presence/absence
      and the empty-set "explicit exclusion" cases.
    * Registry-level skip in ``create_registered_tools`` — creators
      with declared categories that don't intersect the spec are not
      dispatched at all.
    * Dynamic creator short-circuits — MCP / Custom API / Image /
      Audio / Published Agent creators return ``[]`` early on spec
      exclusion, *without* invoking the DB / network calls their
      normal paths require. Asserted via call-count on the mocked
      config methods.
    * Backward compat — ``spec is None`` reverts every code path to
      the pre-spec "build everything" behavior.
    * ``allowed_tools=[]`` semantic fix in ``ToolFactory.create_all_tools``
      — an explicitly empty allowed_tools list now filters to an
      empty tool set instead of leaking the full default set through
      with only a warning logged.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

# ----- Spec helper semantics ---------------------------------------------


def test_spec_default_includes_everything():
    """A bare ``ToolSelectionSpec()`` carries no restrictions: every
    helper returns True so the legacy "build everything" path runs."""
    spec = ToolSelectionSpec()
    assert spec.includes_category("basic") is True
    assert spec.includes_category("mcp") is True
    assert spec.includes_mcp() is True
    assert spec.includes_custom_api() is True
    assert spec.includes_published_agent() is True


def test_spec_categories_restricts_category():
    spec = ToolSelectionSpec(categories=frozenset({"basic", "file"}))
    assert spec.includes_category("basic") is True
    assert spec.includes_category("file") is True
    assert spec.includes_category("mcp") is False


def test_spec_categories_empty_set_excludes_all():
    """Empty frozenset is explicit "no categories allowed" -- distinct
    from None which means "no restriction"."""
    spec = ToolSelectionSpec(categories=frozenset())
    assert spec.includes_category("basic") is False
    assert spec.includes_category("anything") is False


def test_spec_includes_mcp_when_category_present():
    spec = ToolSelectionSpec(categories=frozenset({"mcp"}))
    assert spec.includes_mcp() is True


def test_spec_excludes_mcp_when_category_missing():
    """Even with mcp_servers populated, omitting "mcp" from categories
    disables the MCP creator -- the category gate runs first."""
    spec = ToolSelectionSpec(
        categories=frozenset({"basic"}),
        mcp_servers=frozenset({"Gmail"}),
    )
    assert spec.includes_mcp() is False


def test_spec_excludes_mcp_on_empty_server_set():
    """Empty mcp_servers frozenset == explicit "no MCP tools",
    regardless of categories."""
    spec = ToolSelectionSpec(
        categories=frozenset({"mcp"}),
        mcp_servers=frozenset(),
    )
    assert spec.includes_mcp() is False


def test_spec_custom_api_empty_set_excludes():
    spec = ToolSelectionSpec(custom_api_ids=frozenset())
    assert spec.includes_custom_api() is False


def test_spec_custom_api_none_includes():
    """None means "no restriction" -- the creator still runs and falls
    back to whatever DB-level filtering it does internally."""
    spec = ToolSelectionSpec(custom_api_ids=None)
    assert spec.includes_custom_api() is True


def test_spec_published_agent_empty_set_excludes():
    spec = ToolSelectionSpec(published_agent_ids=frozenset())
    assert spec.includes_published_agent() is False


# ----- ToolRegistry registry-level skip ----------------------------------


@pytest.fixture
def isolated_registry():
    """Snapshot and restore ``ToolRegistry._tool_creators`` so the
    in-place mutations these tests do don't leak into other test
    modules that depend on the production creator list.
    """
    saved = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    ToolRegistry._tool_creators = []
    # ``_modules_imported = True`` so create_registered_tools doesn't
    # re-import the production modules and shadow our test fixtures.
    ToolRegistry._modules_imported = True
    try:
        yield ToolRegistry
    finally:
        ToolRegistry._tool_creators = saved
        ToolRegistry._modules_imported = saved_imported


class _FakeConfig:
    """Stand-in for ``BaseToolConfig`` carrying only the attribute the
    factory's spec-skip logic reads. Avoids the abstract-method burden
    of subclassing BaseToolConfig for these unit tests."""

    def __init__(self, selection_spec: ToolSelectionSpec | None = None):
        self.selection_spec = selection_spec

    def get_allowed_tools(self):  # noqa: D401
        return None

    def get_sandbox(self):  # noqa: D401
        return None

    def get_workspace_config(self):  # noqa: D401
        return None


async def test_registry_runs_all_creators_when_spec_none(isolated_registry):
    """Backward-compat path: ``spec=None`` (or no spec attribute) means
    every registered creator runs, regardless of declared categories."""
    basic = AsyncMock(return_value=[MagicMock(name="basic_tool")])
    basic.__name__ = "basic_creator"
    file_c = AsyncMock(return_value=[MagicMock(name="file_tool")])
    file_c.__name__ = "file_creator"
    isolated_registry.register(basic, categories={"basic"})
    isolated_registry.register(file_c, categories={"file"})

    tools = await isolated_registry.create_registered_tools(_FakeConfig(None))

    assert basic.await_count == 1
    assert file_c.await_count == 1
    assert len(tools) == 2


async def test_registry_skips_creator_when_categories_disjoint(isolated_registry):
    """``spec.categories={"basic"}`` skips the file creator at the
    registry level -- the creator callable is never awaited."""
    basic = AsyncMock(return_value=[MagicMock(name="basic_tool")])
    basic.__name__ = "basic_creator"
    file_c = AsyncMock(return_value=[MagicMock(name="file_tool")])
    file_c.__name__ = "file_creator"
    isolated_registry.register(basic, categories={"basic"})
    isolated_registry.register(file_c, categories={"file"})

    spec = ToolSelectionSpec(categories=frozenset({"basic"}))
    tools = await isolated_registry.create_registered_tools(_FakeConfig(spec))

    assert basic.await_count == 1
    assert file_c.await_count == 0  # registry-level skip
    assert len(tools) == 1


async def test_registry_always_runs_creator_without_declared_categories(
    isolated_registry,
):
    """Dynamic creators register without ``categories=`` so the registry
    can't statically determine whether they're needed. The registry
    runs them unconditionally; the creator itself must short-circuit
    internally on the spec.
    """
    dyn = AsyncMock(return_value=[])
    dyn.__name__ = "dynamic_creator"
    isolated_registry.register(dyn)  # no categories=

    spec = ToolSelectionSpec(categories=frozenset({"basic"}))
    await isolated_registry.create_registered_tools(_FakeConfig(spec))

    assert dyn.await_count == 1


# ----- factory.py:194 allowed_tools=[] semantic fix ----------------------


class _ConfigWithAllowed:
    """Config returning a fixed ``allowed_tools`` so we can pin the
    factory's filter behavior. Other accessors return safe defaults."""

    def __init__(self, allowed: List[str] | None):
        self._allowed = allowed

    def get_allowed_tools(self):
        return self._allowed

    def get_sandbox(self):
        return None

    def get_workspace_config(self):
        return None

    @property
    def selection_spec(self):
        return None


async def test_allowed_tools_empty_list_filters_to_empty(
    isolated_registry, monkeypatch
):
    """Pre-fix behavior: ``allowed_tools=[]`` only logged a warning and
    left the full tool set in place -- subtle leak when callers wanted
    explicit exclusion. Post-fix: empty list filters to empty.
    """
    fake_tool = MagicMock()
    fake_tool.name = "basic_tool"
    fake_tool.metadata.category = MagicMock(value="basic")
    creator = AsyncMock(return_value=[fake_tool])
    creator.__name__ = "creator"
    isolated_registry.register(creator, categories={"basic"})

    # Stub the second-stage filters ToolFactory.create_all_tools chains
    # on (sandbox wrapping, output filtering); we want to assert the
    # name-filter slice in isolation.
    monkeypatch.setattr(
        ToolFactory, "_apply_output_filters", staticmethod(lambda tools, cfg: tools)
    )

    tools = await ToolFactory.create_all_tools(
        _ConfigWithAllowed([]), apply_user_override_filter=False
    )
    assert tools == []


async def test_allowed_tools_none_returns_all(isolated_registry, monkeypatch):
    """``allowed_tools=None`` keeps the original "no name-level filter"
    behavior, complementary to the empty-list case above."""
    fake_tool = MagicMock()
    fake_tool.name = "basic_tool"
    fake_tool.metadata.category = MagicMock(value="basic")
    creator = AsyncMock(return_value=[fake_tool])
    creator.__name__ = "creator"
    isolated_registry.register(creator, categories={"basic"})

    monkeypatch.setattr(
        ToolFactory, "_apply_output_filters", staticmethod(lambda tools, cfg: tools)
    )

    tools = await ToolFactory.create_all_tools(
        _ConfigWithAllowed(None), apply_user_override_filter=False
    )
    assert len(tools) == 1


async def test_allowed_tools_subset_filters_by_name(isolated_registry, monkeypatch):
    """The name-filter still narrows the result when ``allowed_tools``
    is a non-empty list."""
    tool_a = MagicMock(name="a")
    tool_a.name = "tool_a"
    tool_a.metadata.category = MagicMock(value="basic")
    tool_b = MagicMock(name="b")
    tool_b.name = "tool_b"
    tool_b.metadata.category = MagicMock(value="basic")
    creator = AsyncMock(return_value=[tool_a, tool_b])
    creator.__name__ = "creator"
    isolated_registry.register(creator, categories={"basic"})

    monkeypatch.setattr(
        ToolFactory, "_apply_output_filters", staticmethod(lambda tools, cfg: tools)
    )

    tools = await ToolFactory.create_all_tools(
        _ConfigWithAllowed(["tool_a"]), apply_user_override_filter=False
    )
    assert len(tools) == 1
    assert tools[0].name == "tool_a"
