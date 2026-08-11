from xagent.web.services import agent_team_scope as agent_scope
from xagent.web.services import connector_team_scope as connector_scope
from xagent.web.services import knowledge_base_team_scope as kb_scope


def test_agent_team_hooks_install_as_one_group():
    scope = agent_scope.AgentTeamScope(team_id=42, is_team_admin=True)
    agent_scope.set_agent_team_hooks(
        scope=lambda db, user_id: scope,
        connector_validator=lambda db, user_id, team_id, tools: [{"id": 1}],
        knowledge_base_validator=lambda db, user_id, team_id, names: [{"name": "kb"}],
    )
    try:
        assert agent_scope.get_agent_team_scope(None, 7) == scope
        assert agent_scope.validate_team_agent_connectors(None, 7, 42, []) == [
            {"id": 1}
        ]
        assert agent_scope.validate_team_agent_knowledge_bases(None, 7, 42, []) == [
            {"name": "kb"}
        ]
    finally:
        agent_scope.set_agent_team_hooks()


def test_connector_team_hooks_delegate_and_reset():
    deleted_calls = []
    renamed_calls = []

    connector_scope.set_connector_team_hooks(
        visibility=lambda db, user_id: {"mcp": {11}, "custom_api": {22}},
        team_visibility=lambda db, *, team_id: {"mcp": {33}, "custom_api": {44}},
        deleted=lambda db, user_id, kind, connector_id: (
            deleted_calls.append((db, user_id, kind, connector_id))
            or connector_scope.ConnectorDeleteDecision(
                team_owned=True, authorized=True, delete_definition=False
            )
        ),
        renamed=lambda db, user_id, kind, connector_id, old, new: renamed_calls.append(
            (db, user_id, kind, connector_id, old, new)
        ),
    )
    try:
        assert connector_scope.visible_team_connector_ids(None, 7) == {
            "mcp": {11},
            "custom_api": {22},
        }
        assert connector_scope.team_connector_ids(None, team_id=9) == {
            "mcp": {33},
            "custom_api": {44},
        }
        decision = connector_scope.delete_team_connector(None, 7, "mcp", 11)
        assert decision.team_owned and decision.authorized
        connector_scope.rename_team_connector(None, 7, "mcp", 11, "old", "new")
        assert deleted_calls == [(None, 7, "mcp", 11)]
        assert renamed_calls == [(None, 7, "mcp", 11, "old", "new")]
    finally:
        connector_scope.set_connector_team_hooks()
    assert connector_scope.team_connector_hook_installed() is False


def test_knowledge_base_team_hooks_delegate_with_none_session():
    lifecycle_calls = []
    access = kb_scope.KnowledgeBaseAccess(
        name="shared", storage_user_id=42, team_owned=True
    )
    kb_scope.set_knowledge_base_team_hooks(
        visibility=lambda db, user_id: [access],
        access=lambda db, user_id, name, action: access,
        renamed=lambda db, user_id, old, new: lifecycle_calls.append(
            ("rename", db, user_id, old, new)
        ),
        deleted=lambda db, user_id, name, new: lifecycle_calls.append(
            ("delete", db, user_id, name, new)
        ),
    )
    try:
        assert kb_scope.visible_team_knowledge_bases(None, 7) == [access]
        assert kb_scope.resolve_knowledge_base_access(None, 7, "shared") == access
        kb_scope.notify_knowledge_base_renamed(None, 42, "old", "new")
        kb_scope.notify_knowledge_base_deleted(None, 42, "new")
        assert lifecycle_calls == [
            ("rename", None, 42, "old", "new"),
            ("delete", None, 42, "new", None),
        ]
    finally:
        kb_scope.set_knowledge_base_team_hooks()
