"""
Bridge tests between the legacy get_or_create()/create() lifecycle paths and
spec_matches_inspection(): the three seams the reconciliation matcher must
handle correctly once both paths can produce containers for the same
service.

Real Docker required (label/attestation behavior is only meaningful against
real container attrs). Parallel discipline: uuid-suffixed names, membership-
only list_sandboxes() assertions (none used here), try/finally cleanup per
test.
"""

from __future__ import annotations

import uuid

import pytest

from xagent.sandbox import DEFAULT_SANDBOX_IMAGE
from xagent.sandbox.base import (
    ResolvedSandboxRuntimeSpec,
    SandboxConfig,
    SandboxTemplate,
    SpecVerdict,
    spec_matches_inspection,
)
from xagent.sandbox.docker_sandbox import (
    DockerSandboxService,
    MemDockerStore,
    is_docker_available,
)

requires_docker = pytest.mark.skipif(
    not is_docker_available(), reason="Requires reachable Docker daemon"
)


@pytest.fixture(scope="module")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def docker_service():
    return DockerSandboxService(MemDockerStore())


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _spec_for(config: SandboxConfig, image: str = DEFAULT_SANDBOX_IMAGE):
    return ResolvedSandboxRuntimeSpec.from_parts(
        template_type="image",
        image=image,
        working_dir=config.working_dir,
        cpus=config.cpus,
        memory=config.memory,
        env=config.env,
        volumes=config.volumes,
        network_isolated=bool(config.network_isolated),
        ports=config.ports,
    )


@requires_docker
class TestBridgeSeams:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_legacy_get_or_create_container_is_unverified(self, docker_service):
        """A container created by the old get_or_create() path carries no
        spec-attestation label, so the matcher must report UNVERIFIED rather
        than MISMATCH (which would force-rebuild every pre-existing
        container the moment reconciliation is turned on)."""
        service = docker_service
        name = _unique_name("bridge-legacy")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=config,
            )

            inspection = await service.inspect(name)
            assert inspection is not None
            desired = _spec_for(config)

            assert (
                spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED
            )
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_new_create_container_is_match(self, docker_service):
        """A container created by the new create() lifecycle carries a
        verified fingerprint/version label pair, so the same desired spec
        used to create it must compare as MATCH."""
        service = docker_service
        name = _unique_name("bridge-new")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )

            inspection = await service.inspect(name)
            assert inspection is not None
            desired = _spec_for(config)

            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_label_present_but_store_row_missing_still_matches(
        self, docker_service
    ):
        """The matcher itself is blind to the store: it only ever looks at
        the live label/facts, so a container with a verified label but no
        store row still reports MATCH here. Recognizing that this
        specific combination (label present, store row absent) is the one
        case reconciliation must always treat as needing a rebuild is a
        consumer-side contract documented on spec_matches_inspection() in
        base.py, not something this matcher call enforces on its own.
        """
        service = docker_service
        name = _unique_name("bridge-no-store-row")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            # Simulate the store row having been lost independently of the
            # container (the label is immutable and unaffected).
            service._store.delete_info(name)
            assert service._store.get_info(name) is None

            inspection = await service.inspect(name)
            assert inspection is not None
            desired = _spec_for(config)

            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass
