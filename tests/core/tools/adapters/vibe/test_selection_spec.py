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
# composition. Real production agents (75 published, avg 3.9 categories,
# 43% with mcp:<server> entries — see PR #461 discussion) carry exactly
# the string shapes the cases below exercise.


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

    def __init__(self, selection_spec, allowed_tools=None):
        self.selection_spec = selection_spec
        self._allowed_tools = allowed_tools

    def get_allowed_tools(self):
        return self._allowed_tools

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
    from xagent.web.api.chat import _build_selection_spec_from_categories

    spec = _build_selection_spec_from_categories(["basic"])
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
    from xagent.web.api.chat import _build_selection_spec_from_categories

    spec = _build_selection_spec_from_categories(
        ["basic", "browser", "file", "database", "image", "knowledge", "vision"]
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
    from xagent.web.api.chat import _build_selection_spec_from_categories

    spec = _build_selection_spec_from_categories(
        ["basic", "file", "knowledge", "mcp:Gmail"]
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
    from xagent.web.api.chat import _build_selection_spec_from_categories

    spec = _build_selection_spec_from_categories(
        ["mcp:Google Calendar", "mcp:Google Drive", "mcp:HubSpot"]
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
    from xagent.web.api.chat import _build_selection_spec_from_categories

    assert _build_selection_spec_from_categories(None) is None
    assert _build_selection_spec_from_categories([]) is None

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
# 2 inline implementations in chat.py + 1 in websocket.py. Pins the
# "empty/None tool_categories → return None (ALL)" contract that
# review C1 (factory.py:253) flagged as missing.
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
    from xagent.web.api.chat import select_allowed_tool_names_from_categories

    result = select_allowed_tool_names_from_categories(
        tool_categories=None,
        all_tools=[_mock_tool("calculator", "basic")],
    )
    assert result is None, (
        "tool_categories=None must yield None (ALL semantics); a non-None "
        "result would inadvertently filter the full default tool set."
    )


def test_select_allowed_tool_names_empty_input_returns_none() -> None:
    """**Review C1 core invariant.** ``Agent.tool_categories`` defaults
    to ``[]`` for legacy / default agents. Before this fix, the inline
    implementations in chat.py treated ``[]`` as "explicit no tools"
    and stripped every tool from the default agent. The SSOT helper
    must normalize ``[]`` to the same "未配置 → ALL" semantics as
    ``None``.
    """
    from xagent.web.api.chat import select_allowed_tool_names_from_categories

    result = select_allowed_tool_names_from_categories(
        tool_categories=[],
        all_tools=[
            _mock_tool("calculator", "basic"),
            _mock_tool("file_read", "file"),
        ],
    )
    assert result is None, (
        "tool_categories=[] must yield None (legacy 'unconfigured' = "
        "ALL); a non-None result reintroduces the C1 regression where "
        "default agents lose every tool."
    )


def test_select_allowed_tool_names_plain_category_match() -> None:
    """Plain category entry matches tools whose
    ``metadata.category.value`` equals the entry."""
    from xagent.web.api.chat import select_allowed_tool_names_from_categories

    result = select_allowed_tool_names_from_categories(
        tool_categories=["basic"],
        all_tools=[
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
    from xagent.web.api.chat import select_allowed_tool_names_from_categories

    result = select_allowed_tool_names_from_categories(
        tool_categories=["mcp:Gmail"],
        all_tools=[
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
    legitimate 0 tools from the C1 regression.
    """
    from xagent.web.api.chat import select_allowed_tool_names_from_categories

    result = select_allowed_tool_names_from_categories(
        tool_categories=["mcp:UnknownServer"],
        all_tools=[
            _mock_tool("calculator", "basic"),
            _mock_tool("mcp_gmail_send", "mcp"),
        ],
    )
    assert result == [], (
        "Non-empty input with no matches must return [] (legitimate 0 "
        "tools), not None (ALL); the latter would silently allow every "
        "tool when the user specifically picked an unknown MCP server."
    )
