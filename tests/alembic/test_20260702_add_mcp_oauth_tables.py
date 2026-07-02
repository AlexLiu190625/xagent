"""Tests for the MCP OAuth authorization tables migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260702_add_mcp_oauth_authorization_tables.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mcp_oauth_tables_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    context = MigrationContext.configure(connection)
    return Operations(context)


def _create_parent_tables(connection) -> None:
    connection.execute(
        text(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                password_hash VARCHAR(255) NOT NULL
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL
            )
            """
        )
    )


def test_upgrade_creates_mcp_oauth_tables_with_constraints(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_parent_tables(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        inspector = inspect(connection)
        tables = set(inspector.get_table_names())

        assert "mcp_oauth_clients" in tables
        assert "mcp_oauth_grants" in tables
        assert "mcp_oauth_flow_states" in tables

        client_columns = {
            column["name"] for column in inspector.get_columns("mcp_oauth_clients")
        }
        grant_columns = {
            column["name"] for column in inspector.get_columns("mcp_oauth_grants")
        }
        state_columns = {
            column["name"] for column in inspector.get_columns("mcp_oauth_flow_states")
        }

        assert {
            "mcp_server_id",
            "issuer",
            "authorization_endpoint",
            "token_endpoint",
            "client_id",
            "client_secret",
            "metadata",
        }.issubset(client_columns)
        assert {
            "mcp_server_id",
            "user_id",
            "mcp_oauth_client_id",
            "resource_owner_key",
            "issuer",
            "resource",
            "scope",
            "access_token",
            "refresh_token",
            "metadata",
        }.issubset(grant_columns)
        assert {
            "state",
            "mcp_server_id",
            "user_id",
            "mcp_oauth_client_id",
            "resource_owner_key",
            "code_verifier",
            "expires_at",
        }.issubset(state_columns)

        grant_indexes = {
            index["name"] for index in inspector.get_indexes("mcp_oauth_grants")
        }
        assert "ix_mcp_oauth_grants_user_id" in grant_indexes
        assert "ix_mcp_oauth_grants_mcp_server_id" in grant_indexes
        assert "ix_mcp_oauth_grants_mcp_oauth_client_id" in grant_indexes
        assert "ix_mcp_oauth_grants_expires_at" in grant_indexes

        grant_unique_constraints = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("mcp_oauth_grants")
        }
        assert "uq_mcp_oauth_grants_lookup" in grant_unique_constraints

        grant_foreign_keys = inspector.get_foreign_keys("mcp_oauth_grants")
        constrained_columns = {
            tuple(foreign_key["constrained_columns"])
            for foreign_key in grant_foreign_keys
        }
        assert ("mcp_server_id",) in constrained_columns
        assert ("user_id",) in constrained_columns
        assert ("mcp_oauth_client_id",) in constrained_columns


def test_downgrade_removes_mcp_oauth_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_parent_tables(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        tables = set(inspect(connection).get_table_names())
        assert "mcp_oauth_clients" not in tables
        assert "mcp_oauth_grants" not in tables
        assert "mcp_oauth_flow_states" not in tables
