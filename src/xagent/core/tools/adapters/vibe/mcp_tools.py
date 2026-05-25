"""MCP tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool(categories={"mcp"})
async def create_mcp_tools(config: "BaseToolConfig") -> List[Any]:
    """Create MCP tools from configuration.

    Internal short-circuit via ``ToolSelectionSpec.includes_mcp()``:
    when the spec explicitly excludes MCP (either by omitting ``"mcp"``
    from ``categories`` or by setting ``mcp_servers`` to an empty
    frozenset), this creator returns early WITHOUT calling
    ``config.get_mcp_server_configs()`` — that call goes through the
    MCP server scan / DB lookup / per-server session-initialize path
    which dominates the 25-30s setup window for tasks that don't
    actually want MCP tools (see issue #427).

    Registry-level skip (``categories={"mcp"}``) handles the case
    where the spec's ``categories`` set doesn't include ``"mcp"`` at
    all; the internal check covers the finer "include MCP category
    but no servers" case and the legacy spec=None backward-compat
    path.
    """
    spec = (
        config.get_tool_selection_spec()
        if hasattr(config, "get_tool_selection_spec")
        else None
    )
    if spec is not None and not spec.includes_mcp():
        return []
    mcp_configs = await config.get_mcp_server_configs()
    if not mcp_configs:
        return []

    # Per-server filter: when the spec restricts which MCP servers the
    # agent wants, drop configs for any server outside that set BEFORE
    # ``_create_mcp_tools_from_configs`` runs -- the latter performs the
    # actual session initialization (network I/O), which is the real
    # cost we want to avoid. Server names are normalized the same way
    # ``chat.py._build_selection_spec_from_categories`` and
    # ``mcp_adapter.py`` normalize them (spaces and hyphens -> underscore)
    # so the comparison matches across both sides.
    if spec is not None and spec.mcp_servers is not None:
        mcp_configs = [
            cfg
            for cfg in mcp_configs
            if cfg.get("name", "").replace(" ", "_").replace("-", "_")
            in spec.mcp_servers
        ]
        if not mcp_configs:
            return []

    try:
        from .factory import ToolFactory

        return await ToolFactory._create_mcp_tools_from_configs(
            mcp_configs,
            sandbox=config.get_sandbox(),
        )
    except Exception as e:
        logger.warning(f"Failed to create MCP tools: {e}")
        return []
