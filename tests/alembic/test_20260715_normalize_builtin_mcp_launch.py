"""Tests for normalizing built-in Python MCP launch configurations."""

import importlib.util
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260715_normalize_builtin_mcp_launch.py"
)

EXPECTED_LAUNCH_CONFIGS = {
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

GOOGLE_MAPS_CONFIG = {
    "command": "npx",
    "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
    "required_env": ["GOOGLE_MAPS_API_KEY"],
}
CUSTOM_UV_CONFIG = {
    "command": "uv",
    "args": ["run", "custom_server.py"],
    "env": {"CUSTOM_SETTING": "preserve-me"},
}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "normalize_builtin_mcp_launch_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _public_mcp_apps(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("launch_config", sa.JSON),
    )


def _uv_config(canonical: dict[str, object]) -> dict[str, object]:
    return {
        **canonical,
        "command": "uv",
        "args": ["run", "python", *canonical["args"]],
    }


def _seed_environment(
    connection,
    table: sa.Table,
    *,
    uv_app_ids: set[str],
    missing_app_ids: set[str] | None = None,
) -> None:
    missing_app_ids = missing_app_ids or set()
    rows = [
        {
            "app_id": app_id,
            "name": app_id,
            "launch_config": (_uv_config(config) if app_id in uv_app_ids else config),
        }
        for app_id, config in EXPECTED_LAUNCH_CONFIGS.items()
        if app_id not in missing_app_ids
    ]
    rows.extend(
        [
            {
                "app_id": "google-maps",
                "name": "Google Maps",
                "launch_config": GOOGLE_MAPS_CONFIG,
            },
            {
                "app_id": "custom-uv",
                "name": "Custom uv MCP",
                "launch_config": CUSTOM_UV_CONFIG,
            },
        ]
    )
    connection.execute(sa.insert(table), rows)


def _configs_by_app_id(connection, table: sa.Table) -> dict[str, object]:
    return dict(
        connection.execute(
            sa.select(table.c.app_id, table.c.launch_config).order_by(table.c.app_id)
        ).all()
    )


@pytest.mark.parametrize(
    "uv_app_ids",
    [
        {
            "linkedin",
            "gmail",
            "google-drive",
            "google-calendar",
            "instagram",
        },
        {"instagram"},
    ],
    ids=["au-shaped-data", "sg-shaped-data"],
)
def test_upgrade_converges_environment_data_without_touching_other_apps(
    tmp_path, uv_app_ids
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _seed_environment(connection, table, uv_app_ids=uv_app_ids)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        configs = _configs_by_app_id(connection, table)

    for app_id, expected_config in EXPECTED_LAUNCH_CONFIGS.items():
        assert configs[app_id] == expected_config
    assert configs["google-maps"] == GOOGLE_MAPS_CONFIG
    assert configs["custom-uv"] == CUSTOM_UV_CONFIG


def test_upgrade_is_update_only_and_idempotent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _seed_environment(
            connection,
            table,
            uv_app_ids=set(EXPECTED_LAUNCH_CONFIGS),
            missing_app_ids={"facebook"},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            first_upgrade = _configs_by_app_id(connection, table)
            migration.upgrade()
            second_upgrade = _configs_by_app_id(connection, table)

    assert "facebook" not in first_upgrade
    assert first_upgrade == second_upgrade
    assert first_upgrade["google-maps"] == GOOGLE_MAPS_CONFIG
    assert first_upgrade["custom-uv"] == CUSTOM_UV_CONFIG


def test_downgrade_does_not_restore_invalid_launch_configs(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        _seed_environment(connection, table, uv_app_ids=set(EXPECTED_LAUNCH_CONFIGS))

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        configs = _configs_by_app_id(connection, table)

    for app_id, expected_config in EXPECTED_LAUNCH_CONFIGS.items():
        assert configs[app_id] == expected_config


def test_upgrade_skips_when_catalog_table_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()


def test_offline_postgresql_upgrade_emits_literal_update_only_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET launch_config=") == len(
        EXPECTED_LAUNCH_CONFIGS
    )
    assert "INSERT INTO public_mcp_apps" not in sql
    assert "DELETE FROM public_mcp_apps" not in sql
    assert "%(" not in sql
    assert '"command": "python"' in sql
    for app_id, config in EXPECTED_LAUNCH_CONFIGS.items():
        assert f"public_mcp_apps.app_id = '{app_id}'" in sql
        assert config["args"][1] in sql


def test_offline_postgresql_downgrade_emits_no_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.downgrade()

    assert output.getvalue() == ""


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260715_normalize_builtin_mcp_launch"
    assert migration.down_revision == "20260713_add_agent_visibility"
