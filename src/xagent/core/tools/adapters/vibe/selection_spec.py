"""Declarative tool-selection spec for :class:`ToolFactory`.

Background:
    Before this spec, :func:`ToolFactory.create_all_tools` built the
    full set of registered tools and filtered them by name afterwards
    via ``config.get_allowed_tools()``. Callers that only needed a
    category-level filter (e.g. the WS chat path) had to pre-build the
    entire ~52 tool list just to read each tool's metadata.category
    and assemble a name list -- a redundant build that dominated
    per-task setup time (see issue #427).

    ``ToolSelectionSpec`` lets callers declare which categories /
    MCP servers / Custom API IDs / Published Agent IDs the agent
    actually needs, so the factory can skip both the registry-level
    creator dispatch AND the creator-internal I/O (DB queries, MCP
    server initialization) for selectors not in the spec.

Backward compat:
    Every field defaults to ``None``, which means "no restriction" --
    callers that don't supply a spec get the original "build everything"
    behavior. Empty sets (``frozenset()``) are explicit exclusion:
    ``categories=frozenset()`` returns no tools.

This module deliberately has no dependencies on the rest of the
codebase so the spec can be imported by both the factory and the
individual tool creators without circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSelectionSpec:
    """Specification of which tools the caller wants the factory to build.

    Fields:
        categories: When set, only creators whose declared categories
            intersect this set are dispatched. ``None`` = no category
            filter.
        mcp_servers: When set, only MCP tools for these server names
            are built. ``None`` = build MCP tools for every active
            server (subject to category gating). Empty set is "no MCP
            tools" -- the MCP creator must short-circuit and skip its
            ``list_active_servers`` call.
        custom_api_ids: When set, only these Custom API IDs produce
            tools. ``None`` = build for every active Custom API.
            Empty set is "no Custom API tools" -- the creator must
            skip its DB lookup.
        published_agent_ids: When set, only these Published Agent IDs
            produce delegation tools. ``None`` = include every visible
            published agent. Empty set is "no agent delegation" --
            the creator must skip its DB lookup.
    """

    categories: frozenset[str] | None = None
    mcp_servers: frozenset[str] | None = None
    custom_api_ids: frozenset[int] | None = None
    published_agent_ids: frozenset[int] | None = None

    def includes_category(self, cat: str) -> bool:
        """Whether the given category passes the spec.

        ``categories is None`` means "no restriction" so every category
        passes. Otherwise membership is required.
        """
        if self.categories is None:
            return True
        return cat in self.categories

    def includes_mcp(self) -> bool:
        """Whether the MCP creator should run at all.

        Returns ``False`` if either:
          - ``categories`` is set and doesn't contain ``"mcp"`` -- no
            MCP work allowed regardless of server selection
          - ``mcp_servers`` is an explicit empty set -- caller said
            "no MCP tools"

        Otherwise returns ``True``. The creator is then responsible for
        consulting ``mcp_servers`` to filter at the server level.
        """
        if self.categories is not None and "mcp" not in self.categories:
            return False
        if self.mcp_servers is not None and len(self.mcp_servers) == 0:
            return False
        return True

    def includes_custom_api(self) -> bool:
        """Whether the Custom API creator should run.

        Mirrors :meth:`includes_mcp` but for custom APIs.
        """
        if self.custom_api_ids is not None and len(self.custom_api_ids) == 0:
            return False
        return True

    def includes_published_agent(self) -> bool:
        """Whether the Published Agent delegation creators should run.

        Mirrors :meth:`includes_mcp` but for published agents.
        """
        if self.published_agent_ids is not None and len(self.published_agent_ids) == 0:
            return False
        return True
