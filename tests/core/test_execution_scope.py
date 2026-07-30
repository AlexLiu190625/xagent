"""Unit tests for core/execution_scope.py.

Covers the value type, the contextvar helpers, the resolver hook, and the
per-turn activation contract (including restart/resume re-resolution). The
resolver is authoritative over a persisted snapshot rather than always
losing to it. The resolver fixtures below register through
``acknowledges_snapshot_candidate_contract=True`` -- see
``TestSnapshotCandidateAuthority`` for the full resolver/snapshot precedence
matrix and ``tests/web/test_execution_scope_delegation.py`` for the
web-layer snapshot loader wiring. The resolver/loader globals themselves are
reset by the root-level ``isolate_execution_scope_hooks`` autouse fixture in
``tests/conftest.py``, not by a fixture in this module.
"""

import asyncio
import contextvars
import dataclasses
import logging
from contextlib import contextmanager

import pytest

from xagent.core.execution_scope import (
    EXECUTION_SCOPE_SHAPE_VERSION,
    DeferToSnapshot,
    ExecutionScope,
    ExecutionScopeAuthorityError,
    ExecutionScopeContext,
    ExecutionScopeResolverContractError,
    InvalidScopeComponentError,
    defer_to_snapshot,
    execution_scope_resolver_registered,
    get_execution_scope,
    reset_execution_scope,
    resolve_execution_scope,
    resolve_execution_scope_off_turn,
    scope_fingerprint,
    set_execution_scope,
    set_execution_scope_resolver,
    set_execution_scope_snapshot_loader,
    turn_execution_scope,
    validate_scope_component,
)


@contextmanager
def scope_log_records(level: int = logging.WARNING):
    """Collect this module's log records independently of global logging config.

    ``caplog`` installs its handler on the root logger, so an assertion built on
    it only holds while records propagate there. Whether they do is decided by
    whatever else configured logging in the same process -- which under parallel
    test execution is whatever module happens to share the worker. Attaching to
    the logger that emits makes the assertion depend on the code under test
    instead.
    """
    from xagent.core import execution_scope as scope_module

    # The logger object the module logs through, not a name this test assumes
    # resolves to it: under parallel execution the module can be imported under
    # more than one name, and a handler bound by name would then watch a logger
    # nothing writes to.
    logger = scope_module.logger

    class _Records(list):
        """A list that can carry the diagnostics a failure needs."""

        diagnostics: dict[str, object] = {}

    records: list[logging.LogRecord] = _Records()

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level)
    previous_level = logger.level
    # ``logging.disable`` suppresses record *creation* process-wide, below any
    # handler or level this context sets: a library that calls it at import
    # time -- which happens once the full suite is loaded -- would otherwise
    # make this capture silently empty while the code under test behaves
    # correctly. Lift it for the duration and put it back.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        # Snapshot what decided whether a record could exist at all, so a
        # failing assertion reports the reason instead of an empty list.
        records.diagnostics = {  # type: ignore[attr-defined]
            "logger": logger.name,
            "logger_id": id(logger),
            "effective_level": logger.getEffectiveLevel(),
            "manager_disable": logging.root.manager.disable,
            "handlers": [type(h).__name__ for h in logger.handlers],
            "propagate": logger.propagate,
            "captured": [r.getMessage() for r in records],
        }
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logging.disable(previous_disable)


class TestValidateScopeComponent:
    def test_accepts_valid_components(self):
        for value in ["a", "A-b_9", "x" * 63, "0", "_", "-"]:
            assert validate_scope_component(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "x" * 64,
            "a:b",
            "a/b",
            "..",
            "a b",
            "a\n",
            "café",
            "a.b",
            None,
            123,
            ["a"],
        ],
    )
    def test_rejects_invalid_components(self, value):
        with pytest.raises(InvalidScopeComponentError):
            validate_scope_component(value)

    def test_rejects_without_sanitizing(self):
        """Invalid input raises and logs; it is never rewritten to a valid form."""
        with scope_log_records(logging.ERROR) as records:
            with pytest.raises(InvalidScopeComponentError):
                validate_scope_component("bad:name", field_name="sandbox_key_suffix")
        assert any("sandbox_key_suffix" in r.getMessage() for r in records)


class TestExecutionScope:
    def test_defaults_are_unscoped_behavior(self):
        scope = ExecutionScope()
        assert scope.sandbox_key_suffix is None
        assert scope.workspace_segments == ()
        assert dict(scope.memory_dimensions) == {}
        assert scope.strict_memory_isolation is False
        assert scope.isolate_external_dirs is False

    def test_frozen(self):
        scope = ExecutionScope()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.sandbox_key_suffix = "x"

    def test_memory_dimensions_are_read_only(self):
        scope = ExecutionScope(memory_dimensions={"tenant": "acme"})
        with pytest.raises(TypeError):
            scope.memory_dimensions["tenant"] = "other"

    def test_workspace_segments_normalized_to_tuple(self):
        scope = ExecutionScope(workspace_segments=["proj", "env"])
        assert scope.workspace_segments == ("proj", "env")

    def test_equality(self):
        a = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("w",),
            memory_dimensions={"k": "v"},
        )
        b = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=["w"],
            memory_dimensions={"k": "v"},
        )
        assert a == b
        assert a != ExecutionScope(sandbox_key_suffix="other")

    def test_rejects_invalid_sandbox_key_suffix(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(sandbox_key_suffix="a:b")

    def test_rejects_invalid_workspace_segment(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(workspace_segments=("ok", "../escape"))

    def test_rejects_invalid_memory_dimension_key(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(memory_dimensions={"bad key": "v"})

    @pytest.mark.parametrize("value", ["", None, 3])
    def test_rejects_invalid_memory_dimension_value(self, value):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(memory_dimensions={"k": value})

    def test_none_containers_raise_descriptive_value_error(self):
        """None for the collection fields raises a descriptive ValueError
        instead of an opaque TypeError from tuple()/dict() conversion."""
        with pytest.raises(ValueError, match="workspace_segments cannot be None"):
            ExecutionScope(workspace_segments=None)
        with pytest.raises(ValueError, match="memory_dimensions cannot be None"):
            ExecutionScope(memory_dimensions=None)

    def test_boolean_flags_independent_of_other_fields(self):
        """Flags are consumable with an otherwise-empty scope (independent fields)."""
        scope = ExecutionScope(strict_memory_isolation=True)
        assert scope.strict_memory_isolation is True
        assert scope.sandbox_key_suffix is None
        assert scope.workspace_segments == ()


class TestSandboxMountSegments:
    """The mount-prefix field (#79-01): decouples the sandbox mount root from
    the full workspace_segments so scopes sharing a suffix + prefix share one
    container while deeper segments stay in disjoint subtrees."""

    def test_default_mount_covers_full_workspace_segments(self):
        """Unset prefix => mount root == workspace root (byte-identical)."""
        scope = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "7"),
        )
        assert scope.sandbox_mount_segments is None
        assert scope.effective_mount_segments == ("clients", "3", "end_users", "7")

    def test_unscoped_scope_has_empty_effective_mount(self):
        assert ExecutionScope().effective_mount_segments == ()

    def test_prefix_mount_shared_across_deeper_segments(self):
        """Two end users of one CA share a mount prefix; only deeper differs."""
        a = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "7"),
            sandbox_mount_segments=("clients", "3"),
        )
        b = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "8"),
            sandbox_mount_segments=("clients", "3"),
        )
        assert a.effective_mount_segments == ("clients", "3")
        assert b.effective_mount_segments == ("clients", "3")
        assert a.workspace_segments != b.workspace_segments

    def test_mount_segments_normalized_to_tuple(self):
        scope = ExecutionScope(
            workspace_segments=["clients", "3", "end_users", "7"],
            sandbox_mount_segments=["clients", "3"],
        )
        assert scope.sandbox_mount_segments == ("clients", "3")

    def test_empty_prefix_mounts_at_user_root(self):
        """() is a valid prefix of any segments — mount at the user root."""
        scope = ExecutionScope(
            workspace_segments=("clients", "3"),
            sandbox_mount_segments=(),
        )
        assert scope.effective_mount_segments == ()

    def test_rejects_non_prefix_mount_segments(self):
        with pytest.raises(InvalidScopeComponentError, match="must be a prefix"):
            ExecutionScope(
                workspace_segments=("clients", "3", "end_users", "7"),
                sandbox_mount_segments=("clients", "4"),
            )

    def test_rejects_mount_longer_than_workspace_segments(self):
        with pytest.raises(InvalidScopeComponentError, match="must be a prefix"):
            ExecutionScope(
                workspace_segments=("clients", "3"),
                sandbox_mount_segments=("clients", "3", "end_users", "7"),
            )

    def test_rejects_invalid_mount_segment_component(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(
                workspace_segments=("clients", "3"),
                sandbox_mount_segments=("clients", "../escape"),
            )

    def test_to_dict_from_dict_round_trips_prefix(self):
        scope = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "7"),
            sandbox_mount_segments=("clients", "3"),
        )
        assert ExecutionScope.from_dict(scope.to_dict()) == scope
        assert scope.to_dict()["sandbox_mount_segments"] == ["clients", "3"]

    def test_to_dict_preserves_none_vs_empty_distinction(self):
        """None (mount == full segments) must not collapse into () (mount at
        user root) across a serialization round-trip."""
        default = ExecutionScope(workspace_segments=("clients", "3"))
        assert default.to_dict()["sandbox_mount_segments"] is None
        restored_default = ExecutionScope.from_dict(default.to_dict())
        assert restored_default.sandbox_mount_segments is None

        rooted = ExecutionScope(
            workspace_segments=("clients", "3"), sandbox_mount_segments=()
        )
        assert rooted.to_dict()["sandbox_mount_segments"] == []
        restored_rooted = ExecutionScope.from_dict(rooted.to_dict())
        assert restored_rooted.sandbox_mount_segments == ()


class TestDurableStorageSegments:
    """The durable-storage-handle field (#828): mirrors the filesystem
    external-dir allowlist — narrow the object-storage handle to the scope
    subtree only under ``isolate_external_dirs``."""

    def test_isolated_scope_yields_workspace_segments(self):
        scope = ExecutionScope(
            workspace_segments=("clients", "3", "end_users", "7"),
            isolate_external_dirs=True,
        )
        assert scope.durable_storage_segments == ("clients", "3", "end_users", "7")

    def test_non_isolated_scope_yields_empty(self):
        # Segments present but not isolated => owner-root handle (shared reads).
        scope = ExecutionScope(
            workspace_segments=("clients", "3", "end_users", "7"),
            isolate_external_dirs=False,
        )
        assert scope.durable_storage_segments == ()

    def test_unscoped_scope_yields_empty(self):
        assert ExecutionScope().durable_storage_segments == ()

    def test_isolated_without_segments_yields_empty(self):
        assert ExecutionScope(isolate_external_dirs=True).durable_storage_segments == ()


class TestContextvarHelpers:
    def test_not_provided_sentinel_is_a_shared_typed_value(self):
        from xagent.core.execution_scope import (
            EXECUTION_SCOPE_NOT_PROVIDED,
            ExecutionScopeNotProvided,
        )

        assert EXECUTION_SCOPE_NOT_PROVIDED is not None
        assert type(EXECUTION_SCOPE_NOT_PROVIDED) is ExecutionScopeNotProvided

    def test_default_is_none(self):
        assert get_execution_scope() is None

    def test_set_and_reset(self):
        scope = ExecutionScope(sandbox_key_suffix="s1")
        token = set_execution_scope(scope)
        try:
            assert get_execution_scope() is scope
        finally:
            reset_execution_scope(token)
        assert get_execution_scope() is None

    def test_context_manager_restores_previous(self):
        outer = ExecutionScope(sandbox_key_suffix="outer")
        inner = ExecutionScope(sandbox_key_suffix="inner")
        with ExecutionScopeContext(outer):
            assert get_execution_scope() is outer
            with ExecutionScopeContext(inner):
                assert get_execution_scope() is inner
            assert get_execution_scope() is outer
        assert get_execution_scope() is None

    def test_context_manager_restores_on_exception(self):
        scope = ExecutionScope(sandbox_key_suffix="s1")
        with pytest.raises(RuntimeError):
            with ExecutionScopeContext(scope):
                raise RuntimeError("boom")
        assert get_execution_scope() is None

    def test_explicit_none_overrides_outer_scope(self):
        """Setting None is explicitly-unscoped, shadowing any outer scope."""
        outer = ExecutionScope(sandbox_key_suffix="outer")
        with ExecutionScopeContext(outer):
            with ExecutionScopeContext(None):
                assert get_execution_scope() is None
            assert get_execution_scope() is outer


class TestResolverHook:
    def test_no_resolver_resolves_unscoped(self):
        assert resolve_execution_scope("42") is None

    def test_resolver_receives_task_id_as_str(self):
        seen = []

        def resolver(task_id):
            seen.append(task_id)
            return None

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        resolve_execution_scope(42)
        assert seen == ["42"]

    def test_resolver_result_is_returned(self):
        scope = ExecutionScope(sandbox_key_suffix="s1")
        set_execution_scope_resolver(
            lambda task_id: scope, acknowledges_snapshot_candidate_contract=True
        )
        assert resolve_execution_scope("42") is scope

    def test_resolver_exception_propagates(self):
        """A resolver error fails the turn instead of silently running unscoped."""

        def resolver(task_id):
            raise RuntimeError("resolver down")

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        with pytest.raises(RuntimeError, match="resolver down"):
            resolve_execution_scope("42")

    def test_none_task_id_raises_instead_of_resolving_the_string_none(self):
        """str(None) would silently query the resolver for "None"; a caller
        with no task identity must treat the execution as unscoped itself."""
        seen = []
        set_execution_scope_resolver(
            lambda task_id: seen.append(task_id),
            acknowledges_snapshot_candidate_contract=True,
        )
        with pytest.raises(ValueError, match="task_id cannot be None"):
            resolve_execution_scope(None)
        assert seen == []

    def test_resolver_can_be_cleared(self):
        set_execution_scope_resolver(
            lambda task_id: ExecutionScope(),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_resolver(None)
        assert resolve_execution_scope("42") is None


class TestTurnExecutionScope:
    def test_activates_resolved_scope_for_the_turn(self):
        scope = ExecutionScope(workspace_segments=("proj",))
        set_execution_scope_resolver(
            lambda task_id: scope if task_id == "7" else None,
            acknowledges_snapshot_candidate_contract=True,
        )
        with turn_execution_scope(7) as active:
            assert active is scope
            assert get_execution_scope() is scope
        assert get_execution_scope() is None

    def test_unscoped_turn_activates_none(self):
        with turn_execution_scope("7") as active:
            assert active is None
            assert get_execution_scope() is None

    def test_resolver_called_once_per_turn(self):
        calls = []

        def resolver(task_id):
            calls.append(task_id)
            return ExecutionScope(sandbox_key_suffix=f"t{task_id}")

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        with turn_execution_scope("7"):
            pass
        with turn_execution_scope("7"):
            pass
        assert calls == ["7", "7"]

    def test_scope_restored_on_exception(self):
        set_execution_scope_resolver(
            lambda task_id: ExecutionScope(),
            acknowledges_snapshot_candidate_contract=True,
        )
        with pytest.raises(RuntimeError):
            with turn_execution_scope("7"):
                raise RuntimeError("turn failed")
        assert get_execution_scope() is None

    def test_resume_after_restart_re_resolves_identical_scope(self):
        """A resumed task re-resolves and re-applies the identical scope.

        Simulates a process restart between turns: the contextvar starts
        empty in a fresh context and the embedder re-registers its resolver
        at startup; the scope is re-derived from the resolver's persistent
        mapping keyed by task_id, not from any prior in-process state.
        """
        resolver_calls = []

        def make_resolver():
            # The embedder derives the scope from its own persistent data;
            # a fresh resolver instance (new process) yields an equal scope.
            def resolver(task_id):
                resolver_calls.append(task_id)
                return ExecutionScope(
                    sandbox_key_suffix="tenant-a",
                    workspace_segments=("tenant-a",),
                    memory_dimensions={"tenant": "a"},
                )

            return resolver

        def run_turn():
            set_execution_scope_resolver(
                make_resolver(), acknowledges_snapshot_candidate_contract=True
            )
            with turn_execution_scope("99") as scope:
                return scope, get_execution_scope()

        # Turn 1 and turn 2 run in independent contexts (as after a restart).
        first_scope, first_active = contextvars.copy_context().run(run_turn)
        set_execution_scope_resolver(None)
        second_scope, second_active = contextvars.copy_context().run(run_turn)

        assert resolver_calls == ["99", "99"]
        assert first_active is first_scope
        assert second_active is second_scope
        assert first_scope == second_scope

    def test_scope_visible_inside_async_turn(self):
        """The activated scope propagates into the turn's async execution."""
        scope = ExecutionScope(sandbox_key_suffix="s1")
        set_execution_scope_resolver(
            lambda task_id: scope, acknowledges_snapshot_candidate_contract=True
        )

        async def fake_agent_execution():
            await asyncio.sleep(0)
            return get_execution_scope()

        async def turn():
            with turn_execution_scope("7"):
                return await fake_agent_execution()

        assert asyncio.run(turn()) is scope


class TestDeferToSnapshot:
    """The carrier for a resolver's explicit abstention."""

    def test_requires_an_execution_scope_fallback(self):
        with pytest.raises(TypeError):
            defer_to_snapshot("not-a-scope")

    def test_carrier_is_not_an_execution_scope(self):
        carrier = defer_to_snapshot(ExecutionScope())
        assert isinstance(carrier, DeferToSnapshot)
        assert isinstance(carrier, ExecutionScope) is False

    def test_carrier_has_no_public_bare_singleton(self):
        """Only the fallback-carrying factory is exported -- a bare
        module-level sentinel would let "defer" mean "unscoped on miss"
        implicitly instead of requiring an explicit fallback."""
        import xagent.core.execution_scope as scope_module

        assert not hasattr(scope_module, "DEFER_TO_SNAPSHOT")

    def test_carrier_exposes_the_fallback(self):
        fallback = ExecutionScope(sandbox_key_suffix="fallback")
        assert defer_to_snapshot(fallback).fallback == fallback


class TestSetExecutionScopeResolverAckToken:
    """The confirmation-token contract on ``set_execution_scope_resolver``."""

    def test_registering_a_resolver_without_ack_raises(self):
        with pytest.raises(TypeError, match="acknowledges_snapshot_candidate_contract"):
            set_execution_scope_resolver(lambda task_id: None)

    def test_clearing_the_resolver_never_needs_ack(self):
        # None never triggers the check: the ~20 cleanup call sites across
        # the test suite that reset the resolver to None stay unchanged.
        set_execution_scope_resolver(None)
        assert execution_scope_resolver_registered() is False

    def test_ack_true_registers_successfully(self):
        set_execution_scope_resolver(
            lambda task_id: None, acknowledges_snapshot_candidate_contract=True
        )
        assert execution_scope_resolver_registered() is True

    def test_ack_false_explicitly_still_raises(self):
        with pytest.raises(TypeError):
            set_execution_scope_resolver(
                lambda task_id: None,
                acknowledges_snapshot_candidate_contract=False,
            )


class TestExecutionScopeAuthorityErrorInheritance:
    """Pin: distinguishable from the two exceptions this module
    already raises, so no existing ``except RuntimeError``/``except
    ValueError`` clause silently folds an authority conflict into them."""

    def test_not_a_runtime_error(self):
        assert not issubclass(ExecutionScopeAuthorityError, RuntimeError)

    def test_not_a_value_error(self):
        assert not issubclass(ExecutionScopeAuthorityError, ValueError)

    def test_is_a_plain_exception(self):
        assert issubclass(ExecutionScopeAuthorityError, Exception)

    def test_str_carries_task_id_and_field_names_never_values(self):
        """``str()`` on this exception ends up in ``task.error_message`` and
        in the client's terminal error event (see this class's docstring for
        the exact surfacing sites), so it must never leak the scope values
        -- e.g. ``sandbox_key_suffix``/``workspace_segments``/
        ``memory_dimensions`` -- which can carry end-user/client
        identifiers. Only the task id and the mismatched field *names* are
        safe to include."""
        resolver_scope = ExecutionScope(
            sandbox_key_suffix="resolver-secret-suffix",
            workspace_segments=("resolver-secret-segment",),
        )
        snapshot_scope = ExecutionScope(
            sandbox_key_suffix="snapshot-secret-suffix",
            workspace_segments=("snapshot-secret-segment",),
        )
        exc = ExecutionScopeAuthorityError(
            "task-123",
            resolver_scope=resolver_scope,
            snapshot_scope=snapshot_scope,
            mismatched_fields={
                "sandbox_key_suffix": (
                    snapshot_scope.sandbox_key_suffix,
                    resolver_scope.sandbox_key_suffix,
                ),
                "workspace_segments": (
                    snapshot_scope.workspace_segments,
                    resolver_scope.workspace_segments,
                ),
            },
        )
        message = str(exc)
        assert "task-123" in message
        assert "sandbox_key_suffix" in message
        assert "workspace_segments" in message
        assert "resolver-secret-suffix" not in message
        assert "snapshot-secret-suffix" not in message
        assert "resolver-secret-segment" not in message
        assert "snapshot-secret-segment" not in message


class TestExecutionScopeResolverContractErrorInheritance:
    """Pin: the resolve-boundary "unknown return type" error
    must not be catchable by the ``except (ValueError, KeyError, TypeError)``
    clauses several websocket handlers wrap around the turn-execution path
    -- those fold anything in that tuple into a generic "client message
    format error" response, which would misreport a resolver author's bug
    as a malformed client message."""

    def test_not_a_runtime_error(self):
        assert not issubclass(ExecutionScopeResolverContractError, RuntimeError)

    def test_not_a_value_error(self):
        assert not issubclass(ExecutionScopeResolverContractError, ValueError)

    def test_not_a_type_error(self):
        assert not issubclass(ExecutionScopeResolverContractError, TypeError)

    def test_is_a_plain_exception(self):
        assert issubclass(ExecutionScopeResolverContractError, Exception)

    def test_not_folded_by_the_websocket_validation_except_tuple(self):
        """Reproduces the exact tuple websocket.py catches around the
        turn-execution path (e.g. lines ~5802/5864/7029/7257): the new
        error must fall through it and propagate."""
        folded = False
        try:
            raise ExecutionScopeResolverContractError("boom")
        except (ValueError, KeyError, TypeError):
            folded = True
        except ExecutionScopeResolverContractError:
            pass
        assert not folded


class TestExecutionScopeShapeVersionAlignment:
    """``to_dict``'s key set must track every dataclass field (precedent:
    test_runtime_spec.py:315) so a newly added field cannot silently miss
    persistence -- which would make a historical snapshot indistinguishable
    from a current one that simply left the field at its default."""

    def test_to_dict_keys_match_dataclass_fields(self):
        field_names = {f.name for f in dataclasses.fields(ExecutionScope)}
        assert set(ExecutionScope().to_dict().keys()) == field_names

    def test_fresh_scope_is_current_version(self):
        assert ExecutionScope().version == EXECUTION_SCOPE_SHAPE_VERSION

    def test_from_dict_missing_version_defaults_to_legacy_zero(self):
        data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        del data["version"]
        assert ExecutionScope.from_dict(data).version == 0

    def test_from_dict_string_version_is_coerced_to_int(self):
        """A JSON-decoded snapshot carries ``version`` as whatever type the
        wire format gave it; ``from_dict`` must coerce it to ``int`` so a
        stringly-typed ``"1"`` still compares equal to
        ``EXECUTION_SCOPE_SHAPE_VERSION`` in ``resolve_execution_scope``
        instead of being silently treated as a shape mismatch."""
        data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        data["version"] = str(EXECUTION_SCOPE_SHAPE_VERSION)
        decoded = ExecutionScope.from_dict(data)
        assert decoded.version == EXECUTION_SCOPE_SHAPE_VERSION
        assert isinstance(decoded.version, int)

    def test_version_is_excluded_from_equality(self):
        current = ExecutionScope(sandbox_key_suffix="x")
        legacy_data = current.to_dict()
        legacy_data["version"] = 0
        legacy = ExecutionScope.from_dict(legacy_data)
        assert legacy.version == 0
        assert legacy == current

    def test_to_dict_stamps_current_version_even_from_a_stale_scope(self):
        """``to_dict()`` always stamps :data:`EXECUTION_SCOPE_SHAPE_VERSION`,
        never ``self.version``: the dict it returns is being built *now*, in
        the current shape, regardless of whether ``self`` was itself decoded
        from an older snapshot (stale ``.version``). Propagating that stale
        value would let a decoded-then-re-persisted scope masquerade as
        pre-dating fields it actually has, permanently ignoring it as a
        candidate in ``resolve_execution_scope``."""
        stale_data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        stale_data["version"] = 0
        stale_scope = ExecutionScope.from_dict(stale_data)
        assert stale_scope.version == 0
        assert stale_scope.to_dict()["version"] == EXECUTION_SCOPE_SHAPE_VERSION

    def test_shape_version_bump_requires_touching_this_pin(self):
        """Pins the exact field set ``EXECUTION_SCOPE_SHAPE_VERSION`` == 1
        was cut for. A namespace-affecting field can be added --
        wired into ``to_dict``/``from_dict``/
        ``_EXECUTION_SCOPE_NAMESPACE_FIELDS``/``scope_fingerprint`` -- and
        pass the rest of the suite without bumping the shape version, which
        would make every already-persisted snapshot compare against a wider
        shape it never had a chance to opt into, and
        ``resolve_execution_scope`` would fail every turn touching one until
        the version is bumped. This test forces the field set to be
        re-affirmed (and the version bumped) on the next field addition
        instead of drifting unnoticed."""
        field_names = {f.name for f in dataclasses.fields(ExecutionScope)}
        assert field_names == {
            "sandbox_key_suffix",
            "workspace_segments",
            "sandbox_mount_segments",
            "memory_dimensions",
            "strict_memory_isolation",
            "isolate_external_dirs",
            "version",
        }
        assert EXECUTION_SCOPE_SHAPE_VERSION == 1


class TestSnapshotCandidateAuthority:
    """Full resolver x snapshot precedence matrix (#296).

    Axes: resolver registered x (ExecutionScope / None / DeferToSnapshot) x
    (snapshot equal / namespace-differing / policy-only-differing / absent /
    stale-version / loader-broken). Plus the "no resolver registered" golden
    (unchanged from the resolver-less contract) and the resolver-exception
    short-circuit.
    """

    RESOLVER_SCOPE = ExecutionScope(sandbox_key_suffix="from-resolver")

    def _register(self, resolver, loader=None):
        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        if loader is not None:
            set_execution_scope_snapshot_loader(loader)

    # --- resolver returns None: authoritative unscoped -------------------
    def test_resolver_none_is_authoritative_unscoped_even_with_snapshot(self):
        loader_calls = []

        def loader(task_id):
            loader_calls.append(task_id)
            return self.RESOLVER_SCOPE

        self._register(lambda task_id: None, loader)
        assert resolve_execution_scope("1") is None
        assert loader_calls == []  # None short-circuits before the snapshot

    # --- resolver returns ExecutionScope -----------------------------------
    def test_resolver_scope_with_no_snapshot(self):
        self._register(lambda task_id: self.RESOLVER_SCOPE, lambda task_id: None)
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    def test_resolver_scope_with_equal_snapshot_corroborates_silently(self):
        self._register(
            lambda task_id: self.RESOLVER_SCOPE, lambda task_id: self.RESOLVER_SCOPE
        )
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    @pytest.mark.parametrize(
        "field_name,snapshot_kwargs",
        [
            ("sandbox_key_suffix", {"sandbox_key_suffix": "other"}),
            ("workspace_segments", {"workspace_segments": ("other",)}),
            (
                "sandbox_mount_segments",
                {
                    "workspace_segments": ("a", "b"),
                    "sandbox_mount_segments": ("a",),
                },
            ),
            ("memory_dimensions", {"memory_dimensions": {"k": "v"}}),
            ("isolate_external_dirs", {"isolate_external_dirs": True}),
        ],
    )
    def test_namespace_field_mismatch_fails_the_turn(self, field_name, snapshot_kwargs):
        resolver_scope = ExecutionScope()
        snapshot_scope = ExecutionScope(**snapshot_kwargs)
        self._register(lambda task_id: resolver_scope, lambda task_id: snapshot_scope)

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert field_name in exc_info.value.mismatched_fields
        assert exc_info.value.resolver_scope == resolver_scope
        assert exc_info.value.snapshot_scope == snapshot_scope

    def test_policy_only_mismatch_does_not_fail_the_turn(self):
        """strict_memory_isolation is a policy field: a disagreement there
        does not change which key/path is touched, so the resolver's value
        wins without raising. The disagreement must still be observable --
        a silently-won policy mismatch would otherwise be undebuggable."""
        resolver_scope = ExecutionScope(strict_memory_isolation=False)
        snapshot_scope = ExecutionScope(strict_memory_isolation=True)
        self._register(lambda task_id: resolver_scope, lambda task_id: snapshot_scope)

        with scope_log_records() as records:
            assert resolve_execution_scope("1") == resolver_scope
        assert any(
            "policy-only mismatch" in r.getMessage()
            and "strict_memory_isolation" in r.getMessage()
            for r in records
        )

    def test_stale_version_snapshot_is_ignored_even_if_it_would_mismatch(self):
        resolver_scope = ExecutionScope()
        stale_data = ExecutionScope(sandbox_key_suffix="stale").to_dict()
        del stale_data["version"]  # predates EXECUTION_SCOPE_SHAPE_VERSION
        stale_snapshot = ExecutionScope.from_dict(stale_data)
        assert stale_snapshot.version == 0

        self._register(lambda task_id: resolver_scope, lambda task_id: stale_snapshot)
        with scope_log_records() as records:
            assert resolve_execution_scope("1") == resolver_scope
        assert any(
            "shape" in r.getMessage() and "sandbox_key_suffix" in r.getMessage()
            for r in records
        )

    def test_newer_version_snapshot_is_ignored_too(self):
        """A mixed-version rollout can also see a snapshot stamped by a
        *newer* process than this one (e.g. during a rolling deploy where
        some workers already run the next shape). ``!=`` (not ``<``) covers
        this direction too: the snapshot's shape can't be safely compared
        field-by-field against a scope built under a different shape,
        regardless of which side is newer."""
        resolver_scope = ExecutionScope()
        newer_data = ExecutionScope(sandbox_key_suffix="from-the-future").to_dict()
        newer_data["version"] = EXECUTION_SCOPE_SHAPE_VERSION + 1
        newer_snapshot = ExecutionScope.from_dict(newer_data)

        self._register(lambda task_id: resolver_scope, lambda task_id: newer_snapshot)
        assert resolve_execution_scope("1") == resolver_scope

    def test_broken_snapshot_loader_does_not_veto_resolver_scope(self):
        def boom(task_id):
            raise RuntimeError("db down")

        self._register(lambda task_id: self.RESOLVER_SCOPE, boom)
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    def test_snapshot_wrong_type_raises_contract_error_not_attribute_error(self):
        """A snapshot loader is held to the same return-type discipline as
        the resolver. Without the shared validation funnel, a non-scope
        candidate would reach the ``.version`` comparison below and raise a
        bare, undiagnosable ``AttributeError`` instead."""
        self._register(
            lambda task_id: self.RESOLVER_SCOPE,
            lambda task_id: {"not": "a scope"},
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    # --- resolver returns DeferToSnapshot -----------------------------------
    def test_defer_uses_snapshot_when_present(self):
        snapshot_scope = ExecutionScope(sandbox_key_suffix="from-snapshot")
        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(
            lambda task_id: defer_to_snapshot(fallback),
            lambda task_id: snapshot_scope,
        )
        assert resolve_execution_scope("1") == snapshot_scope

    def test_defer_uses_fallback_when_snapshot_absent(self):
        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(
            lambda task_id: defer_to_snapshot(fallback), lambda task_id: None
        )
        assert resolve_execution_scope("1") == fallback

    def test_defer_loader_exception_propagates(self):
        """Unlike the authoritative branch (``test_broken_snapshot_loader_
        does_not_veto_resolver_scope`` above), a broken loader here must fail
        the turn: the resolver has just said it does not know this task's
        scope, so a broken candidate means nobody knows, and turn faces that
        reach this branch must fail closed rather than silently proceed
        under a possibly-wrong fallback."""

        def boom(task_id):
            raise RuntimeError("db down")

        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(lambda task_id: defer_to_snapshot(fallback), boom)
        with pytest.raises(RuntimeError, match="db down"):
            resolve_execution_scope("1")

    def test_defer_snapshot_wrong_type_raises_contract_error(self):
        """A snapshot loader is held to the same return-type discipline as
        the resolver: it must return an ExecutionScope or None."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(
            lambda task_id: defer_to_snapshot(fallback),
            lambda task_id: {"not": "a scope"},
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    def test_defer_snapshot_carrier_from_loader_raises_contract_error(self):
        """A loader returning a DeferToSnapshot (instead of an actual
        snapshot) is exactly as invalid as a raw dict -- both fail the same
        isinstance(ExecutionScope) gate."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(
            lambda task_id: defer_to_snapshot(fallback),
            lambda task_id: defer_to_snapshot(ExecutionScope()),
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    def test_defer_stale_version_snapshot_is_ignored_falls_back(self):
        fallback = ExecutionScope(sandbox_key_suffix="fallback")
        stale_data = ExecutionScope(sandbox_key_suffix="stale").to_dict()
        del stale_data["version"]
        stale_snapshot = ExecutionScope.from_dict(stale_data)
        assert stale_snapshot.version == 0

        self._register(
            lambda task_id: defer_to_snapshot(fallback),
            lambda task_id: stale_snapshot,
        )
        with scope_log_records() as records:
            assert resolve_execution_scope("1") == fallback
        assert any(
            "shape" in r.getMessage() and "sandbox_key_suffix" in r.getMessage()
            for r in records
        )

    @pytest.mark.parametrize(
        "field_name,fallback_kwargs,snapshot_kwargs",
        [
            (
                "sandbox_key_suffix",
                {"sandbox_key_suffix": "fallback-suffix"},
                {"sandbox_key_suffix": "different-suffix"},
            ),
            (
                "workspace_segments",
                {"workspace_segments": ("tenant-a", "b")},
                {"workspace_segments": ("tenant-a",)},
            ),
            (
                "sandbox_mount_segments",
                {
                    "workspace_segments": ("tenant-a", "b", "c"),
                    "sandbox_mount_segments": ("tenant-a", "b", "c"),
                },
                {
                    "workspace_segments": ("tenant-a", "b", "c"),
                    "sandbox_mount_segments": ("tenant-a",),
                },
            ),
            (
                "isolate_external_dirs",
                {"isolate_external_dirs": True},
                {"isolate_external_dirs": False},
            ),
            (
                "memory_dimensions",
                {"memory_dimensions": {"k": "v"}},
                {"memory_dimensions": {}},
            ),
        ],
    )
    def test_defer_snapshot_widening_fallback_fails_the_turn(
        self, field_name, fallback_kwargs, snapshot_kwargs
    ):
        """A snapshot that is *wider* than the resolver's mandatory fallback
        on any namespace field must fail the turn: the snapshot is
        client-influenceable (an ``execution_scope`` key inside a
        client-supplied ``agent_config``), so an unchecked snapshot here
        would let a caller widen its own namespace past the resolver's own
        most conservative answer."""
        fallback = ExecutionScope(**fallback_kwargs)
        snapshot = ExecutionScope(**snapshot_kwargs)
        self._register(
            lambda task_id: defer_to_snapshot(fallback),
            lambda task_id: snapshot,
        )

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert field_name in exc_info.value.mismatched_fields
        assert exc_info.value.resolver_scope == fallback
        assert exc_info.value.snapshot_scope == snapshot

    @pytest.mark.parametrize(
        "fallback_kwargs,snapshot_kwargs",
        [
            ({}, {"sandbox_key_suffix": "narrower"}),
            (
                {"workspace_segments": ("tenant-a",)},
                {"workspace_segments": ("tenant-a", "sub")},
            ),
            (
                {"isolate_external_dirs": False},
                {"isolate_external_dirs": True},
            ),
            (
                {"memory_dimensions": {"k": "v"}},
                {"memory_dimensions": {"k": "v", "k2": "v2"}},
            ),
            (
                {"sandbox_key_suffix": "same"},
                {"sandbox_key_suffix": "same"},
            ),
        ],
    )
    def test_defer_snapshot_narrowing_fallback_is_accepted(
        self, fallback_kwargs, snapshot_kwargs
    ):
        """The mirror of the widening cases above: a snapshot that only
        narrows the fallback (equal, a deeper workspace path, external dirs
        going shared->scope-local, or extra memory dimensions) is accepted
        and used as the resolved scope."""
        fallback = ExecutionScope(**fallback_kwargs)
        snapshot = ExecutionScope(**snapshot_kwargs)
        self._register(
            lambda task_id: defer_to_snapshot(fallback),
            lambda task_id: snapshot,
        )
        assert resolve_execution_scope("1") == snapshot

    # --- boundary judgment / resolver misbehavior ---------------------------
    def test_resolver_returning_unexpected_type_raises_contract_error(self):
        self._register(lambda task_id: "not-a-scope")
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    def test_resolver_exception_short_circuits_before_the_snapshot(self):
        loader_calls = []

        def boom(task_id):
            raise RuntimeError("resolver down")

        self._register(boom, lambda task_id: loader_calls.append(task_id))
        with pytest.raises(RuntimeError, match="resolver down"):
            resolve_execution_scope("1")
        assert loader_calls == []

    # --- no resolver registered: golden: unchanged resolver-less contract --------
    def test_no_resolver_registered_snapshot_alone_drives(self):
        set_execution_scope_snapshot_loader(lambda task_id: self.RESOLVER_SCOPE)
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    def test_no_resolver_registered_no_snapshot_is_unscoped(self):
        assert resolve_execution_scope("1") is None

    def test_no_resolver_registered_loader_exception_propagates(self):
        def boom(task_id):
            raise RuntimeError("db down")

        set_execution_scope_snapshot_loader(boom)
        with pytest.raises(RuntimeError, match="db down"):
            resolve_execution_scope("1")


class TestResolveExecutionScopeOffTurn:
    """Off-turn consumers (websocket ``_scope_segments_for_task``,
    ``ManagedFileRef``) downgrade a namespace mismatch instead of failing."""

    def test_passthrough_when_no_mismatch(self):
        scope = ExecutionScope(sandbox_key_suffix="s")
        set_execution_scope_resolver(
            lambda task_id: scope, acknowledges_snapshot_candidate_contract=True
        )
        assert resolve_execution_scope_off_turn("1") == scope

    def test_namespace_mismatch_downgrades_to_resolver_value_with_warning(self):
        resolver_scope = ExecutionScope(sandbox_key_suffix="from-resolver")
        snapshot_scope = ExecutionScope(sandbox_key_suffix="from-snapshot")
        set_execution_scope_resolver(
            lambda task_id: resolver_scope,
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot_scope)

        # A real disagreement: the fail-closed entry point must reject it, or
        # the downgrade below would be indistinguishable from "nothing
        # disagreed" -- which also returns the resolver's scope.
        with pytest.raises(ExecutionScopeAuthorityError):
            resolve_execution_scope("1")

        with scope_log_records() as records:
            result = resolve_execution_scope_off_turn("1")

        assert result == resolver_scope
        assert any("authority mismatch" in r.getMessage().lower() for r in records), (
            f"no authority-mismatch record; capture state: "
            f"{getattr(records, 'diagnostics', None)}"
        )

    def test_other_exceptions_still_propagate(self):
        def boom(task_id):
            raise RuntimeError("resolver down")

        set_execution_scope_resolver(
            boom, acknowledges_snapshot_candidate_contract=True
        )
        with pytest.raises(RuntimeError, match="resolver down"):
            resolve_execution_scope_off_turn("1")


class TestDeferCarrierNeverActivated:
    """The carrier is never mistaken for a real scope, and the turn
    contextvar only ever holds the resolved fallback/snapshot, never the
    carrier itself."""

    def test_carrier_is_not_an_execution_scope_instance(self):
        carrier = defer_to_snapshot(ExecutionScope())
        assert isinstance(carrier, ExecutionScope) is False

    def test_turn_activates_the_fallback_not_the_carrier(self):
        fallback = ExecutionScope(sandbox_key_suffix="fallback")
        set_execution_scope_resolver(
            lambda task_id: defer_to_snapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        with turn_execution_scope("1") as active:
            assert active == fallback
            current = get_execution_scope()
            assert isinstance(current, ExecutionScope)
            assert isinstance(current, DeferToSnapshot) is False


class TestScopeFingerprintCoversIsolateExternalDirs:
    """(#296) ``isolate_external_dirs`` is baked into the
    cached ``AgentService``'s ``Workspace.allowed_external_dirs`` at build
    time (see ``AgentServiceManager.get_agent_for_task`` ->
    ``_build_allowed_external_dirs``), so an isolate_external_dirs-only
    change must change the fingerprint or the cache never evicts and the
    stale allowed-dirs list keeps being enforced."""

    def test_isolate_external_dirs_only_change_changes_the_fingerprint(self):
        base = ExecutionScope(workspace_segments=("tenant-a",))
        isolated = ExecutionScope(
            workspace_segments=("tenant-a",), isolate_external_dirs=True
        )
        assert scope_fingerprint(base) != scope_fingerprint(isolated)

    def test_strict_memory_isolation_only_change_does_not_change_the_fingerprint(
        self,
    ):
        """Read fresh from the contextvar on every memory operation
        (``UserIsolatedMemoryStore``); nothing cached here goes stale."""
        relaxed = ExecutionScope(strict_memory_isolation=False)
        strict = ExecutionScope(strict_memory_isolation=True)
        assert scope_fingerprint(relaxed) == scope_fingerprint(strict)


class TestExecutionScopeFieldClassificationCompleteness:
    """Every dataclass field must be
    explicitly bucketed as namespace-affecting or policy-only, so a newly
    added field can't silently miss ``resolve_execution_scope``'s
    authority-mismatch classification. A field landing in neither bucket
    would never be compared at all (not even logged), which is worse than
    being misclassified into either one.
    """

    def test_every_field_is_classified_as_namespace_policy_or_version(self):
        from xagent.core import execution_scope as scope_module

        field_names = {f.name for f in dataclasses.fields(ExecutionScope)}
        classified = (
            set(scope_module._EXECUTION_SCOPE_NAMESPACE_FIELDS)
            | set(scope_module._EXECUTION_SCOPE_POLICY_FIELDS)
            | {"version"}
        )
        assert classified == field_names

    def test_fingerprint_tracks_exactly_the_namespace_fields(self):
        """``scope_fingerprint``'s tuple corresponds 1:1 to the
        namespace-field bucket -- ``sandbox_mount_segments`` is represented
        via the derived ``effective_mount_segments`` rather than the raw
        field (two scopes with an unset vs. full-length
        ``sandbox_mount_segments`` that select the identical mount must
        still fingerprint identically), and ``version`` is excluded from
        both (bookkeeping, not a namespace/policy field; see
        ``ExecutionScope.version``'s docstring).

        No namespace field is currently exempted from the fingerprint. If
        one ever needs to be, it must be added to
        ``exempted_from_fingerprint`` below with a comment explaining why
        the cache doesn't need to evict on that field changing -- not
        silently dropped from the set comparison.
        """
        from xagent.core import execution_scope as scope_module

        namespace_fields = set(scope_module._EXECUTION_SCOPE_NAMESPACE_FIELDS)
        fingerprint_represented_fields = {
            "sandbox_key_suffix",
            "workspace_segments",
            "sandbox_mount_segments",  # via effective_mount_segments
            "memory_dimensions",
            "isolate_external_dirs",
        }
        exempted_from_fingerprint: set[str] = set()  # none today
        assert (
            namespace_fields - exempted_from_fingerprint
            == fingerprint_represented_fields
        )
