"""Tests for the team-keyed MCP env hook slot: loader, validator, installed
predicate, and the hook-snapshot primitive.

This slot is deliberately separate from the pre-existing user-keyed shared
env hook (set_mcp_shared_env_hook / load_shared_env_overrides, pinned in
test_mcp_shared_env_hook.py): the two resolve unrelated grouping keys and
must never be interchangeable. Wiring-level pins -- how the two layers
combine inside WebToolConfig, the cross-team-borrowing prohibition, and the
env-source degrade -- live in tests/web/tools/test_mcp_team_env_wiring.py.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.services import mcp_runtime

T1 = 101


@pytest.fixture(autouse=True)
def _reset_hooks() -> Iterator[None]:
    yield
    mcp_runtime.set_mcp_team_env_hook(None)
    mcp_runtime.set_mcp_shared_env_hook(None)


# ---------------------------------------------------------------------------
# Default (no hook installed) is closed, and free: no call, no query.
# ---------------------------------------------------------------------------


def test_team_env_hook_default_not_installed():
    assert mcp_runtime.team_env_hook_installed() is False


def test_load_team_env_overrides_default_is_empty_and_uncalled():
    # No hook installed at all -- the loader must not reach for one.
    assert mcp_runtime.load_team_env_overrides("db", T1) == {}


def test_load_team_env_overrides_none_team_id_never_calls_installed_hook():
    calls: list[tuple] = []

    def hook(db, *, team_id):
        calls.append((db, team_id))
        return {}

    mcp_runtime.set_mcp_team_env_hook(hook)
    assert mcp_runtime.load_team_env_overrides("db", None) == {}
    assert calls == []


# ---------------------------------------------------------------------------
# Installed predicate and the positive path.
# ---------------------------------------------------------------------------


def test_team_env_hook_installed_true_once_set():
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})
    assert mcp_runtime.team_env_hook_installed() is True


def test_team_env_hook_installed_selects_on_predicate_not_empty_answer():
    """An installed hook that legitimately answers empty for a team that
    stored nothing is still 'installed' -- callers must select on the
    predicate, never infer absence from an empty return."""
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})
    assert mcp_runtime.team_env_hook_installed() is True
    assert mcp_runtime.load_team_env_overrides("db", T1) == {}


def test_load_team_env_overrides_returns_hook_answer():
    calls = {}

    def hook(db, *, team_id):
        calls["args"] = (db, team_id)
        return {5: {"KEY": "team-value"}}

    mcp_runtime.set_mcp_team_env_hook(hook)
    assert mcp_runtime.load_team_env_overrides("db", T1) == {5: {"KEY": "team-value"}}
    assert calls["args"] == ("db", T1)


# The hook is always invoked with team_id passed as a keyword. A hook
# declared keyword-only (as the Protocol requires) would raise TypeError if
# the call site ever regressed to passing team_id positionally, and that
# TypeError would surface here as a ConnectorRuntimeError instead of the
# expected return value -- failing this test.
def test_team_env_hook_call_uses_keyword_team_id():
    def strict_hook(db, *, team_id):
        assert team_id == T1
        return {}

    mcp_runtime.set_mcp_team_env_hook(strict_hook)
    assert mcp_runtime.load_team_env_overrides("db", T1) == {}


# ---------------------------------------------------------------------------
# Loader level: hook failure raises the typed error, never degrades.
# ---------------------------------------------------------------------------


def test_load_team_env_overrides_hook_failure_raises_connector_runtime_error():
    def boom(db, *, team_id):
        raise RuntimeError("hook exploded")

    mcp_runtime.set_mcp_team_env_hook(boom)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        mcp_runtime.load_team_env_overrides("db", T1)
    assert excinfo.value.status_code == 503
    assert excinfo.value.details["reason"] == "team_env_resolution_failed"
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_load_team_env_overrides_passes_through_planted_connector_runtime_error():
    """A hook that already raises the typed error reaches the caller as
    that exact object -- not re-wrapped, not given a new cause."""
    planted = ConnectorRuntimeError(
        "planted_code", "planted", details={"reason": "planted_reason"}
    )

    def raising_hook(db, *, team_id):
        raise planted

    mcp_runtime.set_mcp_team_env_hook(raising_hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        mcp_runtime.load_team_env_overrides("db", T1)
    assert excinfo.value is planted


# ---------------------------------------------------------------------------
# Loader level: malformed answer shapes raise, one case each.
# There exists no input for which an internal failure returns {}.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed_answer",
    [
        None,
        "not-a-dict",
        {"5": {"K": "v"}},  # string key, not int
        {True: {"K": "v"}},  # bool key, not accepted as int
        {5: "not-a-dict"},  # value is not a dict
    ],
    ids=[
        "none",
        "non-dict-answer",
        "string-key",
        "bool-key",
        "non-dict-value",
    ],
)
def test_load_team_env_overrides_raises_on_malformed_answer(malformed_answer):
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: malformed_answer)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        mcp_runtime.load_team_env_overrides("db", T1)
    assert excinfo.value.status_code == 503


@pytest.mark.parametrize(
    "malformed_answer",
    [
        None,
        "not-a-dict",
        {"5": {"K": "v"}},
        {True: {"K": "v"}},
        {5: "not-a-dict"},
    ],
    ids=[
        "none",
        "non-dict-answer",
        "string-key",
        "bool-key",
        "non-dict-value",
    ],
)
def test_validate_team_mcp_env_answer_raises_value_error(malformed_answer):
    with pytest.raises(ValueError):
        mcp_runtime._validate_team_mcp_env_answer(malformed_answer)


def test_validate_team_mcp_env_answer_accepts_empty_dict():
    assert mcp_runtime._validate_team_mcp_env_answer({}) == {}


# ---------------------------------------------------------------------------
# Loader level: the answer object is never handed back unwrapped.
# ---------------------------------------------------------------------------


def test_load_team_env_overrides_defensive_copy_of_outer_dict():
    hook_owned = {5: {"KEY": "v"}}
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: hook_owned)

    result = mcp_runtime.load_team_env_overrides("db", T1)
    result[999] = {"INJECTED": "x"}

    assert 999 not in hook_owned


# ---------------------------------------------------------------------------
# The two hook slots are independent: installing/using one never reads,
# calls, or is affected by the other, in either direction.
# ---------------------------------------------------------------------------


def test_team_and_shared_hooks_are_independent_slots():
    shared_calls = []
    team_calls = []

    mcp_runtime.set_mcp_shared_env_hook(
        lambda db, user_id: (shared_calls.append((db, user_id)), {})[1]
    )
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (team_calls.append((db, team_id)), {})[1]
    )

    mcp_runtime.load_team_env_overrides("db", T1)
    assert team_calls == [("db", T1)]
    assert shared_calls == []

    mcp_runtime.load_shared_env_overrides("db", 7)
    assert shared_calls == [("db", 7)]
    assert team_calls == [("db", T1)]  # unchanged


def test_setting_team_hook_does_not_install_shared_hook():
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})
    assert mcp_runtime.load_shared_env_overrides("db", 7) == {}


def test_setting_shared_hook_does_not_install_team_hook():
    mcp_runtime.set_mcp_shared_env_hook(lambda db, user_id: {})
    assert mcp_runtime.team_env_hook_installed() is False


# ---------------------------------------------------------------------------
# The snapshot/restore primitive covers every hook slot in the module,
# not just the two known today. Enumerating slots by name (rather than
# hard-coding "the shared one and the team one") is what makes a slot added
# later, and not added to mcp_env_hooks_snapshot, fail this test instead of
# silently escaping coverage.
# ---------------------------------------------------------------------------


def _hook_slot_names() -> list[str]:
    return [
        name
        for name in vars(mcp_runtime)
        if name.startswith("_get_mcp_") and name.endswith("_hook")
    ]


def test_hook_slot_names_are_discoverable():
    # Sanity check the enumeration itself finds both known slots, so the
    # coverage test below is not vacuously true.
    names = _hook_slot_names()
    assert "_get_mcp_shared_env_hook" in names
    assert "_get_mcp_team_env_hook" in names


def test_snapshot_restores_every_hook_slot_by_identity():
    names = _hook_slot_names()
    originals = {name: getattr(mcp_runtime, name) for name in names}

    with mcp_runtime.mcp_env_hooks_snapshot():
        for name in names:
            setattr(mcp_runtime, name, lambda *a, **k: None)
        for name in names:
            assert getattr(mcp_runtime, name) is not originals[name]

    for name in names:
        assert getattr(mcp_runtime, name) is originals[name]


def test_snapshot_restores_on_exception():
    names = _hook_slot_names()
    originals = {name: getattr(mcp_runtime, name) for name in names}

    with pytest.raises(RuntimeError):
        with mcp_runtime.mcp_env_hooks_snapshot():
            for name in names:
                setattr(mcp_runtime, name, lambda *a, **k: None)
            raise RuntimeError("boom inside the block")

    for name in names:
        assert getattr(mcp_runtime, name) is originals[name]


def test_snapshot_leaves_untouched_state_alone_when_nothing_changes_inside():
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})
    installed_hook = mcp_runtime._get_mcp_team_env_hook

    with mcp_runtime.mcp_env_hooks_snapshot():
        pass

    assert mcp_runtime._get_mcp_team_env_hook is installed_hook
