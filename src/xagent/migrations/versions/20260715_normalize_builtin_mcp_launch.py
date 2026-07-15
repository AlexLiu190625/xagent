"""normalize built-in Python MCP launch configurations

Revision ID: 20260715_normalize_builtin_mcp_launch
Revises: 20260713_add_agent_visibility
Create Date: 2026-07-15

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260715_normalize_builtin_mcp_launch"
down_revision: Union[str, None] = "20260713_add_agent_visibility"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("launch_config", sa.JSON),
)

# This is an immutable migration snapshot. Runtime registry changes must not alter
# migrations that have already been applied to deployed databases.
CANONICAL_LAUNCH_CONFIGS: dict[str, dict[str, object]] = {
    "linkedin": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.linkedin"],
        "env_mapping": {"LINKEDIN_ACCESS_TOKEN": "access_token"},
    },
    "gmail": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.gmail"],
        "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
    },
    "google-drive": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.google_drive"],
        "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
    },
    "google-calendar": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.calendar"],
        "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
    },
    "teams": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.teams"],
        "env_mapping": {"AUTH_TOKEN": "access_token"},
    },
    "outlook": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.outlook"],
        "env_mapping": {"AUTH_TOKEN": "access_token"},
    },
    "onedrive": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.onedrive"],
        "env_mapping": {"AUTH_TOKEN": "access_token"},
    },
    "facebook": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.facebook"],
        "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
    },
    "instagram": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.instagram"],
        "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
    },
}


def _update_statement(app_id: str, launch_config: object) -> sa.sql.dml.Update:
    return (
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == app_id)
        .values(launch_config=launch_config)
    )


def _upgrade_offline() -> None:
    for app_id, launch_config in CANONICAL_LAUNCH_CONFIGS.items():
        serialized_config = json.dumps(launch_config, sort_keys=True)
        statement = (
            sa.update(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.app_id == op.inline_literal(app_id))
            .values(
                launch_config=sa.cast(op.inline_literal(serialized_config), sa.JSON())
            )
        )
        op.execute(statement)


def upgrade() -> None:
    if op.get_context().as_sql:
        _upgrade_offline()
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "launch_config"}.issubset(columns):
        return

    for app_id, launch_config in CANONICAL_LAUNCH_CONFIGS.items():
        bind.execute(_update_statement(app_id, launch_config))


def downgrade() -> None:
    # Invalid launcher values are data defects, so a downgrade intentionally
    # keeps the corrected rows instead of restoring environment-specific drift.
    pass
