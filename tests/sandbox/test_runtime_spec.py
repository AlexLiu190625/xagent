"""
Unit tests for the sandbox runtime spec types, matcher, and lifecycle
contract additions in ``xagent.sandbox.base``.

These are pure unit tests with no Docker/Boxlite dependency: SandboxInspection
instances are hand-constructed to drive the matcher through every verdict
path.
"""

from __future__ import annotations

import dataclasses

import pytest

from xagent.sandbox.base import (
    SPEC_CONTRACT_VERSION,
    ObservedRuntimeFacts,
    ResolvedSandboxRuntimeSpec,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxContractError,
    SandboxInspection,
    SandboxLeaseSpec,
    SandboxReconcileUnsupportedError,
    SandboxRecoveryRequiredError,
    SandboxRuntimeConflictError,
    SandboxService,
    SandboxTemplate,
    SpecVerdict,
    spec_matches_inspection,
)


def _make_spec(**overrides) -> ResolvedSandboxRuntimeSpec:
    kwargs = dict(
        template_type="image",
        image="python:3.12-slim",
        working_dir="/home",
        cpus=1,
        memory=512,
        env={"FOO": "bar", "BAZ": "qux"},
        volumes=[("/host/a", "/guest/a", "ro")],
        network_isolated=False,
        ports=[(8080, 80)],
    )
    kwargs.update(overrides)
    return ResolvedSandboxRuntimeSpec.from_parts(**kwargs)


def _make_facts(**overrides) -> ObservedRuntimeFacts:
    kwargs = dict(
        raw_status="running",
        image_ref="python:3.12-slim",
        image_digest="sha256:deadbeef",
        raw_nano_cpus=1_000_000_000,
        raw_memory_bytes=512 * 1024 * 1024,
        env={"FOO": "bar"},
        volumes=(("/host/a", "/guest/a", "ro"),),
        ports=((8080, 80),),
        network_isolated=False,
        runtime_networks=("bridge",),
        labels={},
        created_at="2026-01-01T00:00:00Z",
        working_dir="/home",
    )
    kwargs.update(overrides)
    return ObservedRuntimeFacts(**kwargs)


def _make_inspection(**overrides) -> SandboxInspection:
    kwargs = dict(
        state="running",
        facts=_make_facts(),
        fingerprint_label=None,
        version_label=str(SPEC_CONTRACT_VERSION),
    )
    kwargs.update(overrides)
    return SandboxInspection(**kwargs)


# --- ResolvedSandboxRuntimeSpec: field sensitivity ---


class TestSpecFieldSensitivity:
    def test_baseline_fingerprint_is_stable(self):
        a = _make_spec()
        b = _make_spec()
        assert a == b
        assert a.fingerprint() == b.fingerprint()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"working_dir": "/other"},
            {"cpus": 2},
            {"memory": 1024},
            {"env": {"FOO": "different", "BAZ": "qux"}},
            {"volumes": [("/host/a", "/guest/a", "rw")]},
            {"network_isolated": True},
            {"ports": [(9090, 90)]},
        ],
    )
    def test_perturbing_any_field_changes_fingerprint(self, overrides):
        baseline = _make_spec()
        perturbed = _make_spec(**overrides)
        assert perturbed != baseline
        assert perturbed.fingerprint() != baseline.fingerprint()

    def test_perturbing_image_changes_fingerprint(self):
        baseline = _make_spec()
        perturbed = _make_spec(image="python:3.11-slim")
        assert perturbed != baseline
        assert perturbed.fingerprint() != baseline.fingerprint()


# --- ResolvedSandboxRuntimeSpec: order insensitivity ---


class TestSpecOrderInsensitivity:
    def test_env_key_order_does_not_affect_equality_or_fingerprint(self):
        a = _make_spec(env={"FOO": "bar", "BAZ": "qux"})
        b = _make_spec(env={"BAZ": "qux", "FOO": "bar"})
        assert a == b
        assert a.fingerprint() == b.fingerprint()

    def test_volume_order_and_duplicates_do_not_affect_equality(self):
        a = _make_spec(
            volumes=[("/host/a", "/guest/a", "ro"), ("/host/b", "/guest/b", "rw")]
        )
        b = _make_spec(
            volumes=[
                ("/host/b", "/guest/b", "rw"),
                ("/host/a", "/guest/a", "ro"),
                ("/host/a", "/guest/a", "ro"),
            ]
        )
        assert a == b
        assert a.fingerprint() == b.fingerprint()
        assert len(a.volumes) == 2

    def test_port_order_and_duplicates_do_not_affect_equality(self):
        a = _make_spec(ports=[(8080, 80), (9090, 90)])
        b = _make_spec(ports=[(9090, 90), (9090, 90), (8080, 80)])
        assert a == b
        assert a.fingerprint() == b.fingerprint()
        assert len(a.ports) == 2

    def test_volume_paths_are_normalized(self):
        a = _make_spec(volumes=[("/host/a", "/guest/a", "ro")])
        b = _make_spec(volumes=[("/host/x/../a", "/guest/./a", "ro")])
        assert a == b
        assert a.fingerprint() == b.fingerprint()


# --- ResolvedSandboxRuntimeSpec: repr redaction ---


class TestSpecReprRedaction:
    def test_repr_does_not_leak_env_values_or_paths(self):
        spec = _make_spec(
            env={"SECRET_TOKEN": "sup3r-s3cr3t-value"},
            volumes=[("/very/private/host/path", "/guest/a", "ro")],
        )
        text = repr(spec)
        assert "sup3r-s3cr3t-value" not in text
        assert "SECRET_TOKEN" not in text
        assert "/very/private/host/path" not in text
        assert "/guest/a" not in text
        assert spec.fingerprint()[:12] in text

    def test_repr_reports_field_counts(self):
        spec = _make_spec(
            env={"A": "1", "B": "2"},
            volumes=[("/host/a", "/guest/a", "ro")],
            ports=[(1, 1), (2, 2)],
        )
        text = repr(spec)
        assert "env_count=2" in text
        assert "volumes_count=1" in text
        assert "ports_count=2" in text


# --- ResolvedSandboxRuntimeSpec: image/snapshot mutual exclusion ---


class TestSpecTemplateMutualExclusion:
    def test_image_type_requires_image(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=None,
                snapshot_id=None,
            )

    def test_image_type_forbids_snapshot_id(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image="python:3.12-slim",
                snapshot_id="snap-1",
            )

    def test_snapshot_type_requires_snapshot_id(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="snapshot",
                image=None,
                snapshot_id=None,
            )

    def test_snapshot_type_forbids_image(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="snapshot",
                image="python:3.12-slim",
                snapshot_id="snap-1",
            )

    def test_snapshot_type_is_valid_on_its_own(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="snapshot",
            image=None,
            snapshot_id="snap-1",
        )
        assert spec.snapshot_id == "snap-1"


# --- ResolvedSandboxRuntimeSpec: from_parts sentinel defaults ---


class TestFromPartsSentinelDefaults:
    """A resolved spec must never carry a 0/None cpus/memory or an
    un-normalized working_dir: from_parts applies the same defaults the
    Docker backend applies at container creation, so both sides of a later
    publish-verification comparison are computed from one canonical form."""

    def test_cpus_none_defaults_to_one(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", cpus=None
        )
        assert spec.cpus == 1

    def test_memory_none_defaults_to_512(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", memory=None
        )
        assert spec.memory == 512

    def test_working_dir_none_defaults_to_home(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", working_dir=None
        )
        assert spec.working_dir == "/home"

    def test_working_dir_trailing_slash_is_normalized(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", working_dir="/home/"
        )
        assert spec.working_dir == "/home"


# --- ResolvedSandboxRuntimeSpec: contract version is out-of-band ---


class TestContractVersionNotInFingerprint:
    def test_contract_version_is_not_a_spec_field(self):
        field_names = {f.name for f in dataclasses.fields(ResolvedSandboxRuntimeSpec)}
        assert "contract_version" not in field_names

    def test_module_constant_exists(self):
        assert SPEC_CONTRACT_VERSION == 1


# --- ResolvedSandboxRuntimeSpec: to_backend_config round trip ---


class TestToBackendConfig:
    def test_round_trip_image_template(self):
        spec = _make_spec()
        template, config = spec.to_backend_config()
        assert isinstance(template, SandboxTemplate)
        assert isinstance(config, SandboxConfig)
        assert template.type == "image"
        assert template.image == "python:3.12-slim"
        assert template.snapshot_id is None
        assert config.working_dir == "/home"
        assert config.cpus == 1
        assert config.memory == 512
        assert config.env == {"FOO": "bar", "BAZ": "qux"}
        assert config.volumes == [("/host/a", "/guest/a", "ro")]
        assert config.network_isolated is False
        assert config.ports == [(8080, 80)]

    def test_round_trip_snapshot_template(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="snapshot",
            image=None,
            snapshot_id="snap-1",
        )
        template, _config = spec.to_backend_config()
        assert template.type == "snapshot"
        assert template.snapshot_id == "snap-1"
        assert template.image is None


# --- SandboxLeaseSpec ---


class TestSandboxLeaseSpec:
    def test_holds_runtime_and_max_concurrency(self):
        runtime = _make_spec()
        lease = SandboxLeaseSpec(runtime=runtime, max_concurrency=3)
        assert lease.runtime == runtime
        assert lease.max_concurrency == 3

    def test_is_frozen(self):
        lease = SandboxLeaseSpec(runtime=_make_spec(), max_concurrency=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            lease.max_concurrency = 5


# --- Exceptions: never RuntimeError ---


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            SandboxContractError,
            SandboxAlreadyExistsError,
            SandboxRuntimeConflictError,
            SandboxRecoveryRequiredError,
            SandboxReconcileUnsupportedError,
        ],
    )
    def test_not_a_runtime_error(self, exc_cls):
        assert not issubclass(exc_cls, RuntimeError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SandboxAlreadyExistsError,
            SandboxRuntimeConflictError,
            SandboxRecoveryRequiredError,
            SandboxReconcileUnsupportedError,
        ],
    )
    def test_subclasses_of_contract_error(self, exc_cls):
        assert issubclass(exc_cls, SandboxContractError)


# --- SandboxService: default implementations ---


class _MinimalSandboxService(SandboxService):
    """Smallest possible concrete SandboxService for pinning default bodies."""

    async def get_or_create(self, name, template=None, config=None):
        raise NotImplementedError

    async def list_sandboxes(self):
        raise NotImplementedError

    async def delete(self, name):
        raise NotImplementedError

    async def supports_snapshots(self):
        return False

    async def create_snapshot(self, name, snapshot_id):
        raise NotImplementedError

    async def list_snapshots(self):
        raise NotImplementedError

    async def delete_snapshot(self, snapshot_id):
        raise NotImplementedError


class TestSandboxServiceDefaults:
    async def test_supports_runtime_spec_defaults_false(self):
        service = _MinimalSandboxService()
        assert await service.supports_runtime_spec() is False

    async def test_inspect_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.inspect("some-name")

    async def test_create_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.create("some-name", SandboxTemplate(), SandboxConfig())

    async def test_start_existing_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.start_existing("some-name")

    async def test_stop_existing_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.stop_existing("some-name")


# --- Matcher: spec_matches_inspection ---


class TestSpecMatchesInspection:
    def test_missing_fingerprint_label_is_unverified(self):
        desired = _make_spec()
        inspection = _make_inspection(fingerprint_label=None)
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED

    def test_missing_version_label_is_unverified(self):
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(), version_label=None
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED

    def test_older_version_label_is_unverified(self):
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            version_label=str(SPEC_CONTRACT_VERSION - 1),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED

    def test_matching_label_and_matching_raw_units_is_match(self):
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000, raw_memory_bytes=512 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH

    def test_docker_update_cpus_drift_is_mismatch(self):
        # `docker update --cpus 0.5` after creation with desired cpus=1.
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=500_000_000, raw_memory_bytes=512 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MISMATCH

    def test_memory_drift_is_mismatch(self):
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000, raw_memory_bytes=1024 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MISMATCH

    def test_label_mismatch_is_mismatch_regardless_of_raw_units(self):
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label="not-the-real-fingerprint",
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000, raw_memory_bytes=512 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MISMATCH

    def test_env_volumes_ports_are_not_directly_compared(self):
        # Label + raw cpu/memory agree; env/volumes/ports on facts differ from
        # desired entirely. Verdict must still be MATCH because these fields
        # are covered by label attestation, not live re-comparison.
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000,
                raw_memory_bytes=512 * 1024 * 1024,
                env={"UNRELATED": "value"},
                volumes=(("/totally/different", "/x", "rw"),),
                ports=((1, 1),),
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH

    def test_current_contract_version_override(self):
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(), version_label="5"
        )
        assert (
            spec_matches_inspection(desired, inspection, current_contract_version=5)
            is SpecVerdict.MATCH
        )
        assert (
            spec_matches_inspection(desired, inspection, current_contract_version=6)
            is SpecVerdict.UNVERIFIED
        )

    def test_newer_version_label_is_unverified(self):
        # A label written by a newer contract version than this code
        # currently implements: the old `<` comparison would fall through
        # to a fingerprint/raw-unit comparison this code cannot safely make
        # sense of; `!=` correctly reports UNVERIFIED for either direction
        # of version mismatch.
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(), version_label="2"
        )
        assert (
            spec_matches_inspection(desired, inspection, current_contract_version=1)
            is SpecVerdict.UNVERIFIED
        )


# --- ObservedRuntimeFacts / SandboxInspection: identity equality only ---


class TestObservedFactsAndInspectionIdentityEquality:
    def test_facts_equality_is_identity_not_structural(self):
        facts = _make_facts()
        assert (facts == dataclasses.replace(facts)) is False

    def test_inspection_equality_is_identity_not_structural(self):
        inspection = _make_inspection()
        assert (inspection == dataclasses.replace(inspection)) is False
