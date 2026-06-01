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

import re
from pathlib import Path
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
    """Stand-in for ``BaseToolConfig`` carrying only the attributes /
    methods the factory's spec-skip logic reads. Avoids the
    abstract-method burden of subclassing BaseToolConfig for these
    unit tests."""

    def __init__(self, selection_spec: ToolSelectionSpec | None = None):
        self._tool_selection_spec = selection_spec

    def get_tool_selection_spec(self):  # noqa: D401
        return self._tool_selection_spec

    def get_sandbox(self):  # noqa: D401
        return None

    def get_workspace_config(self):  # noqa: D401
        return None

    def get_max_output_length(self):  # noqa: D401
        return None

    def get_max_field_count(self):  # noqa: D401
        return None

    def get_max_recursion_depth(self):  # noqa: D401
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


async def test_registry_runs_published_agent_creator_for_workforce_extras(
    isolated_registry,
):
    """Workforce injects specific worker agent tools by name, without
    selecting the whole ``agent`` category. The registry must still dispatch
    only the published-agent creator so the final name filter can keep the
    injected worker tool.
    """
    basic = AsyncMock(return_value=[_mock_tool("calc", "basic")])
    basic.__name__ = "basic_creator"
    published_agents = AsyncMock(return_value=[_mock_tool("agent_42", "agent")])
    published_agents.__name__ = "create_agent_tools"
    agent_management = AsyncMock(return_value=[_mock_tool("create_agent", "agent")])
    agent_management.__name__ = "create_create_agent_tool"
    isolated_registry.register(basic, categories={"basic"})
    isolated_registry.register(
        published_agents,
        categories={"agent"},
        selection_gate="published_agent",
    )
    isolated_registry.register(agent_management, categories={"agent"})

    spec = ToolSelectionSpec.from_raw(
        tool_categories=["basic"],
        workforce_extra_names={"agent_42"},
    )
    tools = await isolated_registry.create_registered_tools(_FakeConfig(spec))

    assert basic.await_count == 1
    assert published_agents.await_count == 1
    assert agent_management.await_count == 0
    assert [tool.name for tool in tools] == ["calc", "agent_42"]


async def test_factory_worker_only_mode_keeps_only_injected_agent_tools(
    isolated_registry,
):
    """Workforce managers with no ordinary categories should only expose
    injected worker agent tools, not the full default tool set.
    """
    basic = AsyncMock(return_value=[_mock_tool("exa_web_search", "basic")])
    basic.__name__ = "basic_creator"
    published_agents = AsyncMock(
        return_value=[
            _mock_tool("agent_42", "agent"),
            _mock_tool("agent_99", "agent"),
        ]
    )
    published_agents.__name__ = "create_agent_tools"
    agent_management = AsyncMock(return_value=[_mock_tool("create_agent", "agent")])
    agent_management.__name__ = "create_create_agent_tool"
    isolated_registry.register(basic, categories={"basic"})
    isolated_registry.register(
        published_agents,
        categories={"agent"},
        selection_gate="published_agent",
    )
    isolated_registry.register(agent_management, categories={"agent"})

    spec = ToolSelectionSpec.from_raw(
        tool_categories=[],
        workforce_extra_names={"agent_42"},
        extras_only_when_unconfigured=True,
    )
    tools = await ToolFactory.create_all_tools(
        _FakeConfig(spec),
        apply_user_override_filter=False,
    )

    assert basic.await_count == 0
    assert published_agents.await_count == 1
    assert agent_management.await_count == 0
    assert [tool.name for tool in tools] == ["agent_42"]


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
        self._tool_selection_spec = selection_spec

    def get_tool_selection_spec(self):
        return self._tool_selection_spec

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


# ``_ConfigWithAllowed`` + the 3 raw-allowed_tools tests it backed are
# OBSOLETE: the factory no longer reads ``config.get_allowed_tools()``
# directly. Same behaviors are now pinned by ``test_factory_*_mode_*``
# above (ALL → keep all, NONE → []  via ``_SpecNone()``, BY_CATEGORIES
# → filter via ``_SpecByCategories.compute_allowed_names``).


# ----- End-to-end: tool_categories → spec → factory dispatch -------------
#
# Reproduces the exact flow real Web/SDK chat traffic uses:
#
#   agents.tool_categories (DB column, list of strings written by the
#   agent builder UI) → chat._build_selection_spec_from_categories →
#   WebToolConfig.selection_spec → ToolFactory.create_all_tools →
#   ToolRegistry registry-level skip + per-creator short-circuit.
#
# The unit tests above pin each layer in isolation; these tests pin the
# composition. The string shapes exercised below match what real
# production agents carry: a small set of plain category names plus
# the ``mcp:<server>`` form for selecting specific MCP servers.


def _make_static_creator(name: str):
    """Build a uniquely-named AsyncMock so post-hoc assertions can tell
    them apart by ``mock.await_count``."""
    fn = AsyncMock(return_value=[])
    fn.__name__ = name
    return fn


@pytest.fixture
def static_creators(isolated_registry, monkeypatch):
    """Register one fake creator per static category that production
    actually uses, with the same categories= annotations the real
    creators carry. Returns the dict so individual tests can assert
    on per-creator dispatch counts.

    Also stubs ``ToolFactory._apply_output_filters`` to a passthrough,
    matching the pattern the per-allowed_tools tests use -- the
    fake creators return empty tool lists which the real output-
    filter pass would attempt to read accessors from the test config
    that the minimal ``_E2EConfig`` doesn't carry.
    """
    creators = {
        "basic": _make_static_creator("basic_creator"),
        "file": _make_static_creator("file_creator"),
        "knowledge": _make_static_creator("knowledge_creator"),
        "browser": _make_static_creator("browser_creator"),
        "image": _make_static_creator("image_creator"),
        "ppt": _make_static_creator("ppt_creator"),
        "vision": _make_static_creator("vision_creator"),
        "database": _make_static_creator("database_creator"),
    }
    for category, creator in creators.items():
        isolated_registry.register(creator, categories={category})
    monkeypatch.setattr(
        ToolFactory, "_apply_output_filters", staticmethod(lambda tools, cfg: tools)
    )
    return creators


class _E2EConfig:
    """Mimics WebToolConfig's surface that the factory + creators read.
    Carries the spec produced by chat.py's helper plus the minimal
    accessors ToolFactory.create_all_tools touches."""

    def __init__(self, selection_spec):
        self._tool_selection_spec = selection_spec

    def get_tool_selection_spec(self):
        return self._tool_selection_spec

    def get_sandbox(self):
        return None

    def get_workspace_config(self):
        return None


async def test_e2e_single_basic_category_skips_all_others(static_creators):
    """The simplest real-prod shape (e.g. agent "Velvet Assistant" =
    ['knowledge', 'basic']): with ``tool_categories=["basic"]`` the
    chat helper produces a spec restricted to {"basic"}, and the
    factory must dispatch *only* the basic creator. All seven other
    static creators stay un-called.
    """
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    spec = ToolSelectionSpec.from_raw(tool_categories=["basic"])
    assert spec is not None
    assert spec.categories == frozenset({"basic"})
    assert spec.mcp_servers is None

    await ToolFactory.create_all_tools(
        _E2EConfig(spec), apply_user_override_filter=False
    )

    assert static_creators["basic"].await_count == 1
    for cat in ("file", "knowledge", "browser", "image", "ppt", "vision", "database"):
        assert static_creators[cat].await_count == 0, (
            f"{cat} creator unexpectedly dispatched"
        )


async def test_e2e_multi_category_dispatches_matching_creators(static_creators):
    """A multi-category prod shape (e.g. agent 258 "Testing" =
    ['basic', 'browser', 'file', 'database', 'image', 'knowledge',
    'vision']): the spec includes all of them, the factory dispatches
    exactly those creators and skips the only category absent from
    the agent's selection (``ppt``)."""
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    spec = ToolSelectionSpec.from_raw(
        tool_categories=[
            "basic",
            "browser",
            "file",
            "database",
            "image",
            "knowledge",
            "vision",
        ]
    )

    await ToolFactory.create_all_tools(
        _E2EConfig(spec), apply_user_override_filter=False
    )

    for cat in ("basic", "browser", "file", "database", "image", "knowledge", "vision"):
        assert static_creators[cat].await_count == 1, f"{cat} creator should have run"
    assert static_creators["ppt"].await_count == 0, (
        "ppt creator should have been skipped"
    )


async def test_e2e_mcp_server_form_extracts_servers_and_includes_mcp(static_creators):
    """The ``mcp:<ServerName>`` form is dual-purposed: it both adds
    ``"mcp"`` to ``spec.categories`` (so the MCP creator runs) and
    populates ``spec.mcp_servers`` with the normalized server name
    (so the MCP creator's per-server filter narrows the work). Mimics
    agent 252 "Email Agent (Sales)_V2" = ['basic', 'file', 'knowledge',
    'mcp:Gmail']."""
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    spec = ToolSelectionSpec.from_raw(
        tool_categories=["basic", "file", "knowledge", "mcp:Gmail"]
    )

    # ``"mcp"`` and ``"other"`` are added implicitly by the builder so the
    # MCP creator AND the Custom-API-via-"other" legacy match path both
    # remain reachable; "Gmail" lands normalized in mcp_servers.
    assert "basic" in spec.categories
    assert "file" in spec.categories
    assert "knowledge" in spec.categories
    assert "mcp" in spec.categories
    assert "other" in spec.categories
    assert spec.mcp_servers == frozenset({"Gmail"})
    assert spec.includes_mcp() is True

    # Static fakes only — actually dispatching the real MCP creator is
    # covered by the per-server filter tests above.
    await ToolFactory.create_all_tools(
        _E2EConfig(spec), apply_user_override_filter=False
    )
    assert static_creators["basic"].await_count == 1
    assert static_creators["file"].await_count == 1
    assert static_creators["knowledge"].await_count == 1
    assert static_creators["browser"].await_count == 0
    assert static_creators["image"].await_count == 0


async def test_e2e_mcp_server_name_normalization_matches_prod_shape(static_creators):
    """Production has agents with multi-word and hyphenated MCP server
    names — agent 260 "Inbound Agent" carries 'mcp:Google Calendar'
    and 'mcp:Google Drive' simultaneously. The helper normalizes the
    space-separated names to underscore-separated so the downstream
    per-server filter in mcp_tools.create_mcp_tools (which applies the
    same normalization to the server's stored ``name``) matches."""
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    spec = ToolSelectionSpec.from_raw(
        tool_categories=["mcp:Google Calendar", "mcp:Google Drive", "mcp:HubSpot"]
    )

    # All three server names normalized identically to the way
    # mcp_tools.create_mcp_tools normalizes the prod ``mcp_configs[i]["name"]``
    # field when applying the per-server filter.
    assert spec.mcp_servers == frozenset({"Google_Calendar", "Google_Drive", "HubSpot"})


async def test_e2e_empty_categories_yields_none_spec(static_creators):
    """An agent with no ``tool_categories`` (or an empty list) is the
    backward-compat path: the chat helper returns ``None``, the
    factory falls through to "build everything", and every registered
    creator is dispatched.

    This is the property production code relies on to never
    accidentally suppress tools for legacy agents that pre-date the
    tool_categories field."""
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    # Empty / None → _SpecAll (mode predicate is_all()), not None.
    # "No restriction" is represented as an explicit mode subclass
    # rather than a sentinel value on a single dataclass.
    assert ToolSelectionSpec.from_raw(tool_categories=None).is_all()
    assert ToolSelectionSpec.from_raw(tool_categories=[]).is_all()

    await ToolFactory.create_all_tools(
        _E2EConfig(None), apply_user_override_filter=False
    )
    # Every static creator runs.
    for cat in static_creators:
        assert static_creators[cat].await_count == 1, (
            f"{cat} should run on the spec-less backward-compat path"
        )


# ---------------------------------------------------------------------------
# select_allowed_tool_names_from_categories — the SSOT helper that replaces
# inline implementations in chat.py + websocket.py. Pins the
# "empty/None tool_categories → return None (ALL)" contract so legacy
# default agents (whose Agent.tool_categories defaults to []) are not
# silently stripped of every tool.
# ---------------------------------------------------------------------------


def _mock_tool(name: str, category: str):
    """Build a minimal mock tool with the ``.metadata.category.value``
    shape the helper inspects. Using ``MagicMock`` here would
    silently match anything; explicit class keeps the contract tight.
    """
    from unittest.mock import MagicMock

    tool = MagicMock()
    tool.name = name
    tool.metadata = MagicMock()
    tool.metadata.category = MagicMock()
    tool.metadata.category.value = category
    return tool


def test_select_allowed_tool_names_none_input_returns_none() -> None:
    """``tool_categories=None`` is the "未配置" sentinel and must map
    to ``None`` (factory's "no name-level restriction" short-circuit).
    """
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    result = ToolSelectionSpec.from_raw(tool_categories=None).compute_allowed_names(
        [_mock_tool("calculator", "basic")],
    )
    assert result is None, (
        "tool_categories=None must yield None (ALL semantics); a non-None "
        "result would inadvertently filter the full default tool set."
    )


def test_select_allowed_tool_names_empty_input_returns_none() -> None:
    """Legacy-default invariant. ``Agent.tool_categories`` defaults
    to ``[]`` for legacy / default agents. Inline implementations
    that treat ``[]`` as "explicit no tools" would strip every tool
    from those agents. The SSOT helper normalizes ``[]`` to the same
    "未配置 → ALL" semantics as ``None``.
    """
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    result = ToolSelectionSpec.from_raw(tool_categories=[]).compute_allowed_names(
        [
            _mock_tool("calculator", "basic"),
            _mock_tool("file_read", "file"),
        ],
    )
    assert result is None, (
        "tool_categories=[] must yield None (legacy 'unconfigured' = "
        "ALL); a non-None result lets the factory strip every tool "
        "from default agents."
    )


def test_select_allowed_tool_names_plain_category_match() -> None:
    """Plain category entry matches tools whose
    ``metadata.category.value`` equals the entry."""
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    result = ToolSelectionSpec.from_raw(
        tool_categories=["basic"]
    ).compute_allowed_names(
        [
            _mock_tool("calculator", "basic"),
            _mock_tool("python_executor", "basic"),
            _mock_tool("file_read", "file"),
        ],
    )
    assert sorted(result or []) == ["calculator", "python_executor"]


def test_select_allowed_tool_names_mcp_server_form() -> None:
    """``mcp:<server>`` entry matches tools named ``mcp_<server>_*``
    (case-insensitive, with spaces / dashes folded to underscores).
    """
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    result = ToolSelectionSpec.from_raw(
        tool_categories=["mcp:Gmail"]
    ).compute_allowed_names(
        [
            _mock_tool("mcp_gmail_send_message", "mcp"),
            _mock_tool("mcp_gmail_list_messages", "mcp"),
            _mock_tool("mcp_slack_send", "mcp"),  # different server, excluded
            _mock_tool("calculator", "basic"),  # different category, excluded
        ],
    )
    assert sorted(result or []) == [
        "mcp_gmail_list_messages",
        "mcp_gmail_send_message",
    ]


def test_select_allowed_tool_names_unknown_mcp_server_yields_empty() -> None:
    """User selected an MCP server whose tools aren't registered (e.g.
    server config exists but no tools loaded). The result is an
    empty allow-list, NOT None -- the user did pick a category, so
    "0 tools" is the correct intent (the factory's ``allowed_tools=[]``
    short-circuit then produces zero tools).

    This case validates that the helper preserves the
    "non-empty input → possibly empty output" branch that distinguishes
    a legitimate 0 tools intent from the unconfigured "build all"
    semantic.
    """
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    result = ToolSelectionSpec.from_raw(
        tool_categories=["mcp:UnknownServer"]
    ).compute_allowed_names(
        [
            _mock_tool("calculator", "basic"),
            _mock_tool("mcp_gmail_send", "mcp"),
        ],
    )
    # Non-empty input with no matches: ``compute_allowed_names`` returns
    # an empty frozenset (BY_CATEGORIES filtered to nothing matched),
    # NOT None (which is the ALL-mode sentinel). Distinct return shapes
    # are load-bearing -- factory L252 differentiates "filter to []" vs
    # "no filter, keep all".
    assert result == frozenset(), (
        "Non-empty input with no matches must return frozenset() "
        "(legitimate 0 tools), not None (ALL); the latter would silently "
        "allow every tool when the user specifically picked an unknown "
        "MCP server."
    )


# ----- ABC sealed-type strict invariants ---------------------------------
#
# These pin the type-level enforcement that replaces the older
# runtime truthiness check on "None vs frozenset() vs frozenset({...})".
# Adding a new abstract method on ToolSelectionSpec forces every
# subclass to implement it -- a missing implementation is caught at
# instantiation time (not silently with a default).


def test_abc_base_cannot_be_instantiated_via_subclass_constructor():
    """Direct ``_SpecAll() / _SpecNone() / _SpecByCategories()``
    construction works -- they are concrete. Only the base ABC
    rejects instantiation, and we verify that via the missing-
    implementation test below (subclass that fails to override an
    abstract method)."""
    from xagent.core.tools.adapters.vibe.selection_spec import (
        _SpecAll,
        _SpecByCategories,
        _SpecNone,
    )

    # Concrete subclasses work.
    assert _SpecAll() is not None
    assert _SpecNone() is not None
    assert _SpecByCategories(categories=frozenset({"basic"})) is not None


def test_abc_subclass_missing_abstract_method_raises_on_instantiation():
    """``@abstractmethod`` enforces mode-dispatch completeness:
    a subclass that fails to implement an abstract method cannot
    be instantiated. This is the type-system replacement for the
    grep test that used to police mode dispatch correctness."""
    from dataclasses import dataclass

    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    # Define a subclass that misses ``compute_allowed_names``.
    @dataclass(frozen=True)
    class _BadSubclass(ToolSelectionSpec):
        def is_all(self):
            return True

        def is_none(self):
            return False

        def is_by_categories(self):
            return False

        def includes_mcp(self):
            return True

        def includes_custom_api(self):
            return True

        def includes_published_agent(self):
            return True

        # compute_allowed_names deliberately missing -- ABC should
        # reject instantiation.

    with pytest.raises(TypeError, match=r"abstract method.*compute_allowed_names"):
        _BadSubclass()


def test_by_categories_rejects_empty_categories():
    """``_SpecByCategories(categories=frozenset())`` would express
    "BY_CATEGORIES with zero categories" which is semantically
    indistinguishable from NONE mode; ``__post_init__`` rejects it
    to keep modes mutually exclusive and force callers to ``_SpecNone()``.
    """
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    with pytest.raises(ValueError, match="non-empty categories"):
        _SpecByCategories(categories=frozenset())


def test_subclasses_are_frozen():
    """Frozen dataclasses; mutation raises ``FrozenInstanceError``."""
    from dataclasses import FrozenInstanceError

    from xagent.core.tools.adapters.vibe.selection_spec import (
        _SpecAll,
        _SpecByCategories,
        _SpecNone,
    )

    sa = _SpecAll()
    sn = _SpecNone()
    sc = _SpecByCategories(categories=frozenset({"basic"}))

    with pytest.raises(FrozenInstanceError):
        sa.categories = frozenset({"x"})  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sn.categories = frozenset({"x"})  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sc.categories = frozenset({"x"})  # type: ignore[misc]


# ----- Mode predicates (explicit replaces implicit) ----------------------


def test_spec_all_mode_predicates():
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecAll

    s = _SpecAll()
    assert s.is_all() is True
    assert s.is_none() is False
    assert s.is_by_categories() is False


def test_spec_none_mode_predicates():
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecNone

    s = _SpecNone()
    assert s.is_all() is False
    assert s.is_none() is True
    assert s.is_by_categories() is False


def test_spec_by_categories_mode_predicates():
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    s = _SpecByCategories(categories=frozenset({"basic"}))
    assert s.is_all() is False
    assert s.is_none() is False
    assert s.is_by_categories() is True


# ----- from_raw single normalizer (4 entry cases) ------------------------


def test_from_raw_none_categories_yields_all_mode():
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecAll

    spec = ToolSelectionSpec.from_raw(tool_categories=None)
    assert isinstance(spec, _SpecAll)


def test_from_raw_empty_categories_yields_all_mode():
    """Legacy "未配置" semantics: empty list is NOT zero tools, it's
    ALL — every callsite must go through ``from_raw`` so this holds
    consistently."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecAll

    spec = ToolSelectionSpec.from_raw(tool_categories=[])
    assert isinstance(spec, _SpecAll)


def test_from_raw_categories_yields_by_categories_mode():
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    spec = ToolSelectionSpec.from_raw(tool_categories=["basic", "file"])
    assert isinstance(spec, _SpecByCategories)
    assert spec.categories == frozenset({"basic", "file"})


def test_from_raw_explicit_none_yields_none_mode():
    """``explicit_none=True`` wins over a non-empty ``tool_categories`` --
    reserved entry for a future product UI for "zero tools"."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecNone

    spec = ToolSelectionSpec.from_raw(
        tool_categories=["basic"],  # would normally yield BY_CATEGORIES
        explicit_none=True,
    )
    assert isinstance(spec, _SpecNone)


def test_from_raw_workforce_extras_ignored_in_all_mode():
    """``workforce_extra_names`` is only meaningful in BY_CATEGORIES;
    silently ignored in ALL (full set already includes everything)."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecAll

    spec = ToolSelectionSpec.from_raw(
        tool_categories=None,
        workforce_extra_names={"some_worker_tool"},
    )
    # ALL mode: no name_extras field on _SpecAll, callsite that
    # asks for extras must have categories set.
    assert isinstance(spec, _SpecAll)


def test_from_raw_can_restrict_unconfigured_agent_to_workforce_extras_only():
    """Workforce manager runtime can opt out of legacy ALL semantics for
    unconfigured categories, yielding only injected worker names.
    """
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    spec = ToolSelectionSpec.from_raw(
        tool_categories=[],
        workforce_extra_names={"agent_42"},
        extras_only_when_unconfigured=True,
    )

    assert isinstance(spec, _SpecByCategories)
    assert spec.categories == frozenset()
    assert spec.name_extras == frozenset({"agent_42"})
    assert spec.includes_category("basic") is False
    assert spec.includes_published_agent() is True
    assert spec.compute_allowed_names(
        [
            _mock_tool("exa_web_search", "basic"),
            _mock_tool("agent_42", "agent"),
            _mock_tool("agent_99", "agent"),
        ]
    ) == frozenset({"agent_42"})


def test_from_raw_unconfigured_extras_only_without_extras_yields_none_mode():
    """A workforce manager with no configured categories and no worker
    extras should get zero tools, not all ordinary tools.
    """
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecNone

    spec = ToolSelectionSpec.from_raw(
        tool_categories=None,
        workforce_extra_names=set(),
        extras_only_when_unconfigured=True,
    )

    assert isinstance(spec, _SpecNone)


def test_from_raw_workforce_extras_carried_in_by_categories():
    """In BY_CATEGORIES, ``workforce_extra_names`` lands on
    :attr:`_SpecByCategories.name_extras` for ``compute_allowed_names``
    injection."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    spec = ToolSelectionSpec.from_raw(
        tool_categories=["basic"],
        workforce_extra_names={"worker_tool_a", "worker_tool_b"},
    )
    assert isinstance(spec, _SpecByCategories)
    assert spec.name_extras == frozenset({"worker_tool_a", "worker_tool_b"})
    assert spec.includes_published_agent() is True


# ----- P2 fix: includes_custom_api with category restriction -------------


def test_by_categories_excludes_custom_api_when_other_missing():
    """P2 fix: ``categories={"basic"}`` excludes ``"other"``, so Custom
    API tools cannot survive the post-build category filter. The
    ``create_db_custom_api_tools`` creator's ``get_custom_api_configs()``
    DB lookup is wasted I/O in this case -- ``includes_custom_api``
    must short-circuit."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    spec = _SpecByCategories(categories=frozenset({"basic"}))
    assert spec.includes_custom_api() is False


def test_by_categories_includes_custom_api_when_other_present():
    """Mirror of the P2 fix: ``"other"`` IN categories means custom
    API tools survive the filter; creator must run."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    spec = _SpecByCategories(categories=frozenset({"other"}))
    assert spec.includes_custom_api() is True


def test_by_categories_excludes_custom_api_when_other_present_but_ids_empty():
    """Even with ``"other"`` in categories, an explicit empty
    ``custom_api_ids=frozenset()`` skips the creator (legacy
    'explicit exclude' shape preserved)."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    spec = _SpecByCategories(
        categories=frozenset({"other"}),
        custom_api_ids=frozenset(),
    )
    assert spec.includes_custom_api() is False


# ----- compute_allowed_names mode dispatch -------------------------------


def test_compute_allowed_names_all_returns_none():
    """ALL mode: factory should keep every tool, signalled by None."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecAll

    spec = _SpecAll()
    result = spec.compute_allowed_names(
        [_mock_tool("calc", "basic"), _mock_tool("img", "image")]
    )
    assert result is None


def test_compute_allowed_names_none_returns_empty_frozenset():
    """NONE mode: factory should drop every tool, signalled by
    empty frozenset. Distinct from None (ALL) -- the
    ``if allowed_names is not None`` check in factory differentiates."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecNone

    spec = _SpecNone()
    result = spec.compute_allowed_names(
        [_mock_tool("calc", "basic"), _mock_tool("img", "image")]
    )
    assert result == frozenset()


def test_compute_allowed_names_by_categories_filters_correctly():
    """BY_CATEGORIES mode: only tools whose category matches survive,
    plus any ``name_extras`` injection."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    spec = _SpecByCategories(
        categories=frozenset({"basic"}),
        name_extras=frozenset({"injected_worker_tool"}),
    )
    result = spec.compute_allowed_names(
        [
            _mock_tool("calc", "basic"),
            _mock_tool("img", "image"),  # not in categories
            _mock_tool("file", "file"),  # not in categories
        ]
    )
    assert result == frozenset({"calc", "injected_worker_tool"})


def test_compute_allowed_names_workforce_extra_does_not_admit_agent_category():
    """Injected workforce worker names must not broaden the user's category
    selection to every agent-category tool.
    """
    spec = ToolSelectionSpec.from_raw(
        tool_categories=["basic"],
        workforce_extra_names={"agent_42"},
    )
    result = spec.compute_allowed_names(
        [
            _mock_tool("calc", "basic"),
            _mock_tool("agent_42", "agent"),
            _mock_tool("agent_99", "agent"),
        ]
    )

    assert spec.categories == frozenset({"basic"})
    assert result == frozenset({"calc", "agent_42"})


# ----- Factory L252 dispatch through spec.compute_allowed_names ---------


async def test_factory_all_mode_keeps_all_tools(isolated_registry):
    """ALL mode: factory must NOT name-filter; all tools from the
    registry survive into the returned list."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecAll

    basic = AsyncMock(return_value=[_mock_tool("calc", "basic")])
    basic.__name__ = "basic_creator"
    isolated_registry.register(basic, categories={"basic"})

    cfg = MagicMock()
    cfg.get_tool_selection_spec.return_value = _SpecAll()
    cfg.get_allowed_tools.return_value = None
    cfg.get_sandbox.return_value = None
    cfg.get_workspace_config.return_value = None
    cfg.get_user_tool_overrides.return_value = {}
    cfg.get_max_output_length.return_value = None
    cfg.get_max_field_count.return_value = None
    cfg.get_max_recursion_depth.return_value = None

    tools = await ToolFactory.create_all_tools(cfg)
    assert [t.name for t in tools] == ["calc"]


async def test_factory_none_mode_filters_to_empty(isolated_registry):
    """NONE mode: factory must drop every tool (compute_allowed_names
    returns frozenset() → caller filters to [])."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecNone

    basic = AsyncMock(return_value=[_mock_tool("calc", "basic")])
    basic.__name__ = "basic_creator"
    isolated_registry.register(basic, categories={"basic"})

    cfg = MagicMock()
    cfg.get_tool_selection_spec.return_value = _SpecNone()
    cfg.get_allowed_tools.return_value = None
    cfg.get_sandbox.return_value = None
    cfg.get_workspace_config.return_value = None
    cfg.get_user_tool_overrides.return_value = {}
    cfg.get_max_output_length.return_value = None
    cfg.get_max_field_count.return_value = None
    cfg.get_max_recursion_depth.return_value = None

    tools = await ToolFactory.create_all_tools(cfg)
    assert tools == []


async def test_factory_by_categories_filters_by_compute_allowed_names(
    isolated_registry,
):
    """BY_CATEGORIES mode: factory must filter the registry's output to
    only the names returned by ``spec.compute_allowed_names``."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecByCategories

    basic_creator = AsyncMock(
        return_value=[
            _mock_tool("calc", "basic"),
            _mock_tool("search", "basic"),
        ]
    )
    basic_creator.__name__ = "basic_creator"
    file_creator = AsyncMock(return_value=[_mock_tool("read", "file")])
    file_creator.__name__ = "file_creator"
    isolated_registry.register(basic_creator, categories={"basic"})
    isolated_registry.register(file_creator, categories={"file"})

    cfg = MagicMock()
    cfg.get_tool_selection_spec.return_value = _SpecByCategories(
        categories=frozenset({"basic"})
    )
    cfg.get_allowed_tools.return_value = None
    cfg.get_sandbox.return_value = None
    cfg.get_workspace_config.return_value = None
    cfg.get_user_tool_overrides.return_value = {}
    cfg.get_max_output_length.return_value = None
    cfg.get_max_field_count.return_value = None
    cfg.get_max_recursion_depth.return_value = None

    tools = await ToolFactory.create_all_tools(cfg)
    # ``file_creator`` is registry-skipped (declared "file", spec wants
    # only "basic"); ``basic_creator`` runs and both names survive
    # the post-build filter.
    assert sorted(t.name for t in tools) == ["calc", "search"]


def test_compute_allowed_names_by_categories_mcp_subcategory_match():
    """``mcp:Gmail`` sub-category in categories matches ``mcp_gmail_*``
    tool names (case-insensitive, spaces normalized)."""
    # Use from_raw so ``mcp_servers`` is derived consistently with
    # ``categories`` -- direct ``_SpecByCategories`` construction with
    # ``mcp:<server>`` in categories but ``mcp_servers=None`` is
    # rejected by ``__post_init__`` (defense against bypass).
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:Gmail"])
    result = spec.compute_allowed_names(
        [
            _mock_tool("mcp_gmail_send", "mcp"),
            _mock_tool("mcp_gmail_read", "mcp"),
            _mock_tool("mcp_slack_post", "mcp"),  # different server
            _mock_tool("calc", "basic"),
        ]
    )
    assert result == frozenset({"mcp_gmail_send", "mcp_gmail_read"})


# ----- Mechanical SSOT pins(防未来同类回归)-----------------------------
#
# Defense-in-depth source-level grep tests. ABC + abstractmethod already
# polices mode-dispatch completeness; these tests catch the kind of
# pattern that the SSOT extraction was meant to eliminate — new
# callpaths that bypass the normalizer with inline category matching,
# new ``@register_tool`` creators forgetting the ``categories=``
# annotation. If one fails, the message points the reader at the fix;
# do NOT silence the test, fix the violation.

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_ROOT = _REPO_ROOT / "src"


def _iter_production_python_files():
    """Yield every .py under src/ EXCEPT the helper module(s) that
    legitimately implement the inlined match logic. Tests live under
    tests/ so we skip src/ → tests/ is naturally excluded."""
    helper_modules = {"selection_spec.py"}
    for path in _SRC_ROOT.rglob("*.py"):
        if path.name in helper_modules:
            continue
        yield path


def test_no_inline_tool_categories_matching_in_production():
    """Mechanical SSOT pin: only ``selection_spec.py`` may iterate
    ``all_tools`` and match against ``tool.metadata.category`` to
    derive a name allow-list. Any other file doing this bypasses the
    ``empty list = ALL`` invariant — the exact failure mode the
    ``AgentTool`` delegation path previously had.

    If you legitimately need this mapping in a new file, call
    ``ToolSelectionSpec.from_raw(...)`` and let
    ``spec.compute_allowed_names`` do the matching instead.
    """
    inline_pattern = re.compile(
        r"for\s+\w+\s+in\s+all_tools.*?metadata\.category", re.DOTALL
    )
    offenders: list[str] = []
    for path in _iter_production_python_files():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if inline_pattern.search(content):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"Inline ``for tool in all_tools: ...metadata.category`` found in "
        f"{offenders}. This bypasses the ToolSelectionSpec SSOT and risks "
        f"the P1 regression where ``Agent.tool_categories=[]`` is misread "
        f"as 'zero tools'. Replace with "
        f"``ToolSelectionSpec.from_raw(tool_categories=...)`` and let "
        f"``spec.compute_allowed_names`` do the matching."
    )


def test_register_tool_must_declare_categories_or_be_allowlisted():
    """Every ``@register_tool`` creator MUST declare ``categories=``
    (so the registry can skip it on category mismatch) OR be in the
    explicit allowlist below. ``create_db_custom_api_tools`` is the
    one legitimate unannotated creator because its tools have a
    dynamic category that the spec's ``includes_custom_api`` checks
    at runtime (the P2 fix). Any NEW unannotated creator must be
    added here with rationale -- silently adding one risks the same
    P2 perf-leak pattern (running the creator's DB lookup when the
    spec would have skipped a static creator)."""
    unannotated_allowlist = {"create_db_custom_api_tools"}
    bare_register_pattern = re.compile(
        r"^@register_tool\s*$",  # @register_tool with no args
        re.MULTILINE,
    )
    function_def_after = re.compile(
        r"^@register_tool\s*\n(?:async\s+)?def\s+(\w+)\b", re.MULTILINE
    )
    offenders: list[str] = []
    for path in _iter_production_python_files():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not bare_register_pattern.search(content):
            continue
        # File has a bare ``@register_tool``; find the function name(s)
        for match in function_def_after.finditer(content):
            func_name = match.group(1)
            if func_name not in unannotated_allowlist:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{func_name}")
    assert not offenders, (
        f"Unannotated ``@register_tool`` creators found: {offenders}. "
        f"Every creator MUST declare its categories statically "
        f"(``@register_tool(categories={{'basic'}})``) so the registry "
        f"can skip it without invoking the creator (and any DB / network "
        f"I/O the creator does). The only legitimate exception is "
        f"``create_db_custom_api_tools`` whose tools have dynamic "
        f"categories — the spec's ``includes_custom_api`` checks that at "
        f"runtime. If you have a new dynamic-category creator, add it to "
        f"the allowlist with rationale AND make sure "
        f"``ToolSelectionSpec`` has the corresponding ``includes_*`` "
        f"short-circuit method."
    )


# ----- Follow-up review regressions(防回归)-----------------------------


def test_factory_falls_back_to_get_allowed_tools_when_spec_none():
    """When the caller doesn't supply a ``ToolSelectionSpec``,
    ``BaseToolConfig.get_allowed_tools()`` is still the public
    name-allow-list contract (standalone ``ToolConfig`` callers).
    Factory must honour the raw list rather than silently keeping
    every tool.
    """
    import asyncio
    from unittest.mock import MagicMock as _MagicMock

    tool_allowed = _mock_tool("allowed", "basic")
    tool_leaked = _mock_tool("leaked", "basic")

    cfg = _MagicMock()
    cfg.get_tool_selection_spec.return_value = None
    cfg.get_allowed_tools.return_value = ["allowed"]
    cfg.get_sandbox.return_value = None
    cfg.get_workspace_config.return_value = None
    cfg.get_user_tool_overrides.return_value = {}
    cfg.get_max_output_length.return_value = None
    cfg.get_max_field_count.return_value = None
    cfg.get_max_recursion_depth.return_value = None

    saved_creators = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    ToolRegistry._tool_creators = []
    ToolRegistry._modules_imported = True
    try:

        async def creator(_cfg):
            return [tool_allowed, tool_leaked]

        creator.__name__ = "test_creator"
        ToolRegistry.register(creator, categories={"basic"})

        tools = asyncio.run(ToolFactory.create_all_tools(cfg))
        assert [t.name for t in tools] == ["allowed"], (
            "spec=None must still apply config.get_allowed_tools() as a "
            "name allow-list; leaving every tool through breaks the "
            "legacy ToolConfig contract."
        )
    finally:
        ToolRegistry._tool_creators = saved_creators
        ToolRegistry._modules_imported = saved_imported


def test_compute_allowed_names_plain_mcp_admits_all_mcp_tools():
    """User picked plain ``["mcp"]`` (no server qualifier) — MUST
    admit every mcp-category tool, not 0. Previously broken because
    the name-filter step routed all mcp/other tools to sub-category
    matching, which found no ``mcp:<server>`` entry to match against."""
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp"])
    result = spec.compute_allowed_names(
        [
            _mock_tool("mcp_gmail_send", "mcp"),
            _mock_tool("mcp_slack_post", "mcp"),
            _mock_tool("calc", "basic"),
        ]
    )
    assert result == frozenset({"mcp_gmail_send", "mcp_slack_post"})


def test_compute_allowed_names_plain_other_admits_all_other_tools():
    """User picked plain ``["other"]`` — MUST admit every other-
    category tool."""
    spec = ToolSelectionSpec.from_raw(tool_categories=["other"])
    result = spec.compute_allowed_names(
        [
            _mock_tool("api_custom_call", "other"),
            _mock_tool("api_legacy_call", "other"),
            _mock_tool("calc", "basic"),
        ]
    )
    assert result == frozenset({"api_custom_call", "api_legacy_call"})


def test_compute_allowed_names_mcp_server_does_not_broaden_to_all_mcp():
    """User picked only ``["mcp:Gmail"]`` — MUST stay narrow to
    Gmail's mcp tools, NOT broaden to every mcp tool."""
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:Gmail"])
    result = spec.compute_allowed_names(
        [
            _mock_tool("mcp_gmail_send", "mcp"),
            _mock_tool("mcp_slack_post", "mcp"),
        ]
    )
    assert result == frozenset({"mcp_gmail_send"})


def test_compute_allowed_names_mixed_plain_and_server_picks():
    """``["mcp:Gmail", "other"]`` should pick only Gmail's mcp tools
    plus every other-category tool."""
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:Gmail", "other"])
    result = spec.compute_allowed_names(
        [
            _mock_tool("mcp_gmail_send", "mcp"),
            _mock_tool("mcp_slack_post", "mcp"),
            _mock_tool("api_custom_call", "other"),
            _mock_tool("calc", "basic"),
        ]
    )
    assert result == frozenset({"mcp_gmail_send", "api_custom_call"})


def test_spec_wants_mcp_only_for_explicit_mcp_selection():
    """``_spec_wants_mcp`` (chat.py) must NOT trigger MCP DB query for
    default / no-MCP agents. Pin the contract here to keep the
    derivation honest."""
    from xagent.core.tools.adapters.vibe.selection_spec import _SpecAll, _SpecNone
    from xagent.web.api.chat import _spec_wants_mcp

    assert _spec_wants_mcp(None) is False  # legacy / no-spec caller
    assert _spec_wants_mcp(_SpecAll()) is False  # default agent
    assert _spec_wants_mcp(_SpecNone()) is False  # explicit zero tools
    assert (
        _spec_wants_mcp(ToolSelectionSpec.from_raw(tool_categories=["basic"])) is False
    )  # no mcp picked
    assert (
        _spec_wants_mcp(ToolSelectionSpec.from_raw(tool_categories=["mcp"])) is True
    )  # plain mcp
    assert (
        _spec_wants_mcp(ToolSelectionSpec.from_raw(tool_categories=["mcp:Gmail"]))
        is True
    )  # mcp:<server>
    assert (
        _spec_wants_mcp(
            ToolSelectionSpec.from_raw(tool_categories=["basic", "mcp:Gmail"])
        )
        is True
    )  # mixed
