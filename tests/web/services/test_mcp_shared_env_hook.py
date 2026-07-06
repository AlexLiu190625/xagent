"""Tests for the shared (app-injected) MCP env hook and layered merge."""

from xagent.web.services import mcp_runtime


def test_merge_stdio_env_layers_later_wins():
    """global -> shared -> user, each layer overriding the previous."""
    merged = mcp_runtime.merge_stdio_env(
        {"A": "g", "B": "g", "C": "g"},
        {"B": "s", "C": "s"},
        {"C": "u"},
    )
    assert merged == {"A": "g", "B": "s", "C": "u"}


def test_merge_stdio_env_two_arg_unchanged():
    """Existing two-arg callers keep working (user over global)."""
    assert mcp_runtime.merge_stdio_env({"A": "g"}, {"A": "u"}) == {"A": "u"}
    assert mcp_runtime.merge_stdio_env({"A": "g"}, None) == {"A": "g"}


def test_resolve_stdio_env_by_source():
    """env_source selects which layer to apply over the global env."""
    g = {"A": "g", "B": "g"}
    shared = {"B": "team"}
    user = {"B": "own"}
    assert mcp_runtime.resolve_stdio_env("platform", g, shared, user) == g
    assert mcp_runtime.resolve_stdio_env("shared", g, shared, user) == {
        "A": "g",
        "B": "team",
    }
    assert mcp_runtime.resolve_stdio_env("own", g, shared, user) == {
        "A": "g",
        "B": "own",
    }
    # None keeps the legacy fallback: global < shared < user (most specific wins).
    assert mcp_runtime.resolve_stdio_env(None, g, shared, user) == {
        "A": "g",
        "B": "own",
    }


def test_resolve_stdio_env_missing_chosen_layer_falls_to_global():
    """A pick whose layer is empty falls back to global, never a blank key."""
    g = {"A": "g"}
    assert mcp_runtime.resolve_stdio_env("shared", g, None, {"A": "own"}) == g
    assert mcp_runtime.resolve_stdio_env("own", g, {"A": "team"}, None) == g


def test_shared_env_hook_default_is_noop():
    mcp_runtime.set_mcp_shared_env_hook(None)
    assert mcp_runtime.load_shared_env_overrides(object(), 1) == {}


def test_shared_env_hook_is_invoked():
    calls = {}

    def hook(db, user_id):
        calls["args"] = (db, user_id)
        return {5: {"KEY": "shared-value"}}

    mcp_runtime.set_mcp_shared_env_hook(hook)
    try:
        assert mcp_runtime.load_shared_env_overrides("db", 7) == {
            5: {"KEY": "shared-value"}
        }
        assert calls["args"] == ("db", 7)
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)


def test_shared_env_hook_none_user_returns_empty():
    mcp_runtime.set_mcp_shared_env_hook(lambda db, uid: {1: {"K": "v"}})
    try:
        assert mcp_runtime.load_shared_env_overrides("db", None) == {}
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)


def test_shared_env_hook_failure_degrades_to_empty():
    def boom(db, user_id):
        raise RuntimeError("hook exploded")

    mcp_runtime.set_mcp_shared_env_hook(boom)
    try:
        assert mcp_runtime.load_shared_env_overrides("db", 7) == {}
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)
