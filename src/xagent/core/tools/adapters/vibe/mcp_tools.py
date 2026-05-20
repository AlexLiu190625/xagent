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
    spec = getattr(config, "selection_spec", None)
    if spec is not None and not spec.includes_mcp():
        return []
    mcp_configs = await config.get_mcp_server_configs()
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
