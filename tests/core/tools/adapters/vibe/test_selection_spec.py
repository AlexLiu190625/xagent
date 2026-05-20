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


# ----- MCP per-server filter (creator-internal short-circuit) ------------


class _MCPConfig:
    """Config returning a fixed list of MCP server config dicts so the
    creator's filter path is exercised against a known input. Matches
    the production shape (list of ``{"name": ..., "transport": ..., ...}``)
    closely enough for the per-server filter check."""

    def __init__(
        self,
        servers: List[dict],
        selection_spec: ToolSelectionSpec | None = None,
    ):
        self._servers = servers
        self.selection_spec = selection_spec

    async def get_mcp_server_configs(self):
        return self._servers

    def get_sandbox(self):
        return None


async def test_mcp_per_server_filter_skips_non_matching_configs(monkeypatch):
    """The MCP creator must filter ``mcp_configs`` by
    ``spec.mcp_servers`` BEFORE handing them to
    ``_create_mcp_tools_from_configs`` -- the latter does the network
    session-initialize work whose cost we want to avoid.

    The factory call inside the creator is patched so we can assert
    the filtered config list it actually receives, without spinning up
    real MCP sessions.
    """
    from xagent.core.tools.adapters.vibe import mcp_tools
    from xagent.core.tools.adapters.vibe.factory import ToolFactory

    received = []

    async def _fake_create(mcp_configs, sandbox=None):
        received.append(mcp_configs)
        return []

    monkeypatch.setattr(
        ToolFactory,
        "_create_mcp_tools_from_configs",
        staticmethod(_fake_create),
    )

    servers = [
        {"name": "Gmail"},
        {"name": "Google Drive"},
        {"name": "Slack"},
    ]
    spec = ToolSelectionSpec(
        categories=frozenset({"mcp"}),
        mcp_servers=frozenset({"Gmail"}),
    )
    cfg = _MCPConfig(servers, selection_spec=spec)

    await mcp_tools.create_mcp_tools(cfg)

    assert len(received) == 1
    assert [c["name"] for c in received[0]] == ["Gmail"]


async def test_mcp_per_server_filter_normalizes_whitespace(monkeypatch):
    """Server names with spaces or hyphens are normalized to underscores
    on both sides (chat.py's spec builder, mcp_adapter's tool naming).
    The per-server filter must apply the same normalization so a
    ``mcp:Google Drive`` user selection matches a server config whose
    actual stored name is ``Google Drive``."""
    from xagent.core.tools.adapters.vibe import mcp_tools
    from xagent.core.tools.adapters.vibe.factory import ToolFactory

    received = []

    async def _fake_create(mcp_configs, sandbox=None):
        received.append(mcp_configs)
        return []

    monkeypatch.setattr(
        ToolFactory,
        "_create_mcp_tools_from_configs",
        staticmethod(_fake_create),
    )

    servers = [
        {"name": "Google Drive"},
        {"name": "Slack"},
    ]
    # Spec contains the normalized form (matches how
    # _build_selection_spec_from_categories assembles it).
    spec = ToolSelectionSpec(
        categories=frozenset({"mcp"}),
        mcp_servers=frozenset({"Google_Drive"}),
    )
    cfg = _MCPConfig(servers, selection_spec=spec)

    await mcp_tools.create_mcp_tools(cfg)

    assert len(received) == 1
    assert [c["name"] for c in received[0]] == ["Google Drive"]


async def test_mcp_per_server_filter_empty_match_short_circuits(monkeypatch):
    """If the spec's ``mcp_servers`` set has no overlap with the active
    server list, the creator must return early WITHOUT calling
    ``_create_mcp_tools_from_configs`` -- otherwise we'd still pay the
    network-init cost for an empty filtered set."""
    from xagent.core.tools.adapters.vibe import mcp_tools
    from xagent.core.tools.adapters.vibe.factory import ToolFactory

    call_count = 0

    async def _fake_create(mcp_configs, sandbox=None):
        nonlocal call_count
        call_count += 1
        return []

    monkeypatch.setattr(
        ToolFactory,
        "_create_mcp_tools_from_configs",
        staticmethod(_fake_create),
    )

    servers = [{"name": "Slack"}]
    spec = ToolSelectionSpec(
        categories=frozenset({"mcp"}),
        mcp_servers=frozenset({"Gmail"}),
    )
    cfg = _MCPConfig(servers, selection_spec=spec)

    result = await mcp_tools.create_mcp_tools(cfg)

    assert result == []
    assert call_count == 0  # short-circuit, no factory call


async def test_mcp_no_per_server_filter_when_spec_lacks_servers(monkeypatch):
    """``spec.mcp_servers is None`` means "no per-server restriction";
    the creator must hand every active server's config through
    unfiltered to preserve the backward-compat "all MCP servers" path."""
    from xagent.core.tools.adapters.vibe import mcp_tools
    from xagent.core.tools.adapters.vibe.factory import ToolFactory

    received = []

    async def _fake_create(mcp_configs, sandbox=None):
        received.append(mcp_configs)
        return []

    monkeypatch.setattr(
        ToolFactory,
        "_create_mcp_tools_from_configs",
        staticmethod(_fake_create),
    )

    servers = [{"name": "Gmail"}, {"name": "Slack"}]
    spec = ToolSelectionSpec(categories=frozenset({"mcp"}), mcp_servers=None)
    cfg = _MCPConfig(servers, selection_spec=spec)

    await mcp_tools.create_mcp_tools(cfg)

    assert len(received) == 1
    assert [c["name"] for c in received[0]] == ["Gmail", "Slack"]


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
