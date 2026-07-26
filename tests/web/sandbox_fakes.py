"""Shared fake ``SandboxService`` for ``SandboxManager`` unit tests.

Before this module existed, manager tests each rolled their own stand-in for
the injected ``SandboxService``: a bare ``AsyncMock()``, a bare
``MagicMock()``, or a hand-written ``_FakeService`` class that implemented
only the legacy methods a given test file happened to touch. None of those
three shapes are safe once ``SandboxManager`` starts routing on
``await service.supports_runtime_spec()``: an unconfigured ``AsyncMock()``
returns a truthy child mock for that call (never the real default ``False``),
a bare ``MagicMock()`` cannot be awaited at all, and a hand-written fake
without the method simply raises ``AttributeError``. All three would
silently misroute or crash under that gate.

``FakeSandboxService`` fixes this by actually inheriting ``SandboxService``.
The four spec-reconciliation methods (``supports_runtime_spec``/``inspect``/
``create``/``start_existing``/``stop_existing``) are deliberately left
un-overridden here, so a plain ``FakeSandboxService()`` carries the exact
same production defaults a real legacy-only backend would: awaiting
``supports_runtime_spec()`` returns ``False`` (a real ``bool``, not a mock),
and the other four raise ``SandboxReconcileUnsupportedError``. Constructor
keyword arguments let an individual test opt a specific instance into the
reconciliation surface without touching the class-level defaults that every
other test relies on.

The seven legacy lifecycle methods (``get_or_create``/``list_sandboxes``/
``delete``/``supports_snapshots``/``create_snapshot``/``list_snapshots``/
``delete_snapshot``) are abstract on ``SandboxService`` and therefore must be
concretely defined on this class for it to be instantiable at all. Each is
also wrapped in ``AsyncMock(wraps=...)`` on the instance, so tests keep full
``unittest.mock`` call-tracking (``assert_awaited_once_with``,
``await_args_list``, ``.side_effect =``, ``.return_value =``) exactly as they
did against the ad hoc ``AsyncMock()``/``_FakeService`` stand-ins they
replace, while the default behavior (when a test does not override
return_value/side_effect) is a small in-memory container registry mirroring
what the various ad hoc fakes already did.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional
from unittest.mock import AsyncMock, MagicMock

from xagent.sandbox.base import (
    Sandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxInspection,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
)


class FakeSandboxService(SandboxService):
    """In-memory ``SandboxService`` stand-in for manager-level tests."""

    def __init__(
        self,
        initial: Iterable[str] = (),
        *,
        runtime_spec_supported: bool = False,
        inspect_result: Optional[SandboxInspection] = None,
        create_result: Any = None,
        start_existing_result: Any = None,
        stop_existing_result: Any = None,
    ) -> None:
        self.containers: set[str] = set(initial)
        self.peak = len(self.containers)
        self.deleted: list[str] = []
        self.snapshots: dict[str, SandboxSnapshot] = {}

        # Wrap the concrete bodies below in AsyncMock so tests get the full
        # unittest.mock spy/override API on the instance while default
        # behavior still calls through to this class's own implementation.
        self.get_or_create = AsyncMock(wraps=self.get_or_create)
        self.list_sandboxes = AsyncMock(wraps=self.list_sandboxes)
        self.delete = AsyncMock(wraps=self.delete)
        self.supports_snapshots = AsyncMock(wraps=self.supports_snapshots)
        self.create_snapshot = AsyncMock(wraps=self.create_snapshot)
        self.list_snapshots = AsyncMock(wraps=self.list_snapshots)
        self.delete_snapshot = AsyncMock(wraps=self.delete_snapshot)

        # Reserved for PR-1b stage 2 reconciliation tests: left untouched,
        # this instance keeps the SandboxService base class's own defaults
        # (supports_runtime_spec() -> False; inspect/create/start_existing/
        # stop_existing -> SandboxReconcileUnsupportedError). Passing any of
        # these keyword arguments opts this one instance into the
        # reconciliation surface without changing the class-level defaults
        # every other (legacy-only) test relies on.
        if runtime_spec_supported:
            self.supports_runtime_spec = AsyncMock(return_value=True)
        if inspect_result is not None:
            self.inspect = AsyncMock(return_value=inspect_result)
        if create_result is not None:
            self.create = AsyncMock(return_value=create_result)
        if start_existing_result is not None:
            self.start_existing = AsyncMock(return_value=start_existing_result)
        if stop_existing_result is not None:
            self.stop_existing = AsyncMock(return_value=stop_existing_result)

    # --- legacy lifecycle: concrete bodies (also satisfies SandboxService's
    # abstract methods so this class is instantiable) ---

    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> Sandbox:
        self.containers.add(name)
        self.peak = max(self.peak, len(self.containers))
        sandbox = MagicMock()
        sandbox.name = name
        return sandbox

    async def list_sandboxes(self) -> list[SandboxInfo]:
        return [
            SandboxInfo(
                name=name,
                state="stopped",
                template=SandboxTemplate(type="image", image="img:v1"),
                config=SandboxConfig(),
            )
            for name in sorted(self.containers)
        ]

    async def delete(self, name: str) -> None:
        self.containers.discard(name)
        self.deleted.append(name)

    async def supports_snapshots(self) -> bool:
        return False

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        snapshot = SandboxSnapshot(snapshot_id=snapshot_id)
        self.snapshots[snapshot_id] = snapshot
        return snapshot

    async def list_snapshots(self) -> list[SandboxSnapshot]:
        return list(self.snapshots.values())

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.snapshots.pop(snapshot_id, None)
