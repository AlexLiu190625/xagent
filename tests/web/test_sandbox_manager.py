"""Test sandbox manager functionality."""

import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.sandbox_manager import (
    SandboxManager,
    _create_boxlite_service,
    _create_docker_service,
    _create_sandbox_service,
    get_sandbox_manager,
)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global singleton state before each test."""
    import xagent.web.sandbox_manager as mod

    mod._sandbox_manager = None
    mod._sandbox_manager_initialized = False
    yield
    mod._sandbox_manager = None
    mod._sandbox_manager_initialized = False


class TestCreateSandboxService:
    """Test _create_sandbox_service function."""

    def test_disabled_returns_none(self):
        """Test sandbox disabled via env returns None."""
        with patch.dict("os.environ", {"SANDBOX_ENABLED": ""}):
            result = _create_sandbox_service()
        assert result is None

    def test_docker_default(self):
        """Test default implementation is docker."""
        with (
            patch.dict("os.environ", {"SANDBOX_ENABLED": "true"}, clear=False),
            patch("xagent.web.sandbox_manager._create_docker_service") as mock_create,
        ):
            os.environ.pop("SANDBOX_IMPLEMENTATION", None)
            mock_create.return_value = MagicMock()
            result = _create_sandbox_service()
        assert result is not None
        mock_create.assert_called_once()

    def test_unknown_implementation_falls_back_to_docker(self):
        """Test unknown implementation falls back to docker."""
        with (
            patch.dict(
                "os.environ",
                {"SANDBOX_ENABLED": "true", "SANDBOX_IMPLEMENTATION": "unknown"},
                clear=False,
            ),
            patch("xagent.web.sandbox_manager._create_docker_service") as mock_create,
        ):
            mock_create.return_value = MagicMock()
            _create_sandbox_service()
        mock_create.assert_called_once()

    def test_docker_selected(self):
        """Test docker implementation selection."""
        with (
            patch.dict(
                "os.environ",
                {"SANDBOX_ENABLED": "true", "SANDBOX_IMPLEMENTATION": "docker"},
                clear=False,
            ),
            patch("xagent.web.sandbox_manager._create_docker_service") as mock_create,
        ):
            mock_create.return_value = MagicMock()
            result = _create_sandbox_service()
        assert result is not None
        mock_create.assert_called_once()


class TestGetSandboxManager:
    """Test get_sandbox_manager singleton."""

    def test_returns_none_when_service_none(self):
        """Test returns None when sandbox service creation fails."""
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service", return_value=None
        ):
            result = get_sandbox_manager()
        assert result is None

    def test_returns_manager_when_service_available(self):
        """Test returns SandboxManager when service is available."""
        mock_service = MagicMock()
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service",
            return_value=mock_service,
        ):
            result = get_sandbox_manager()
        assert isinstance(result, SandboxManager)

    def test_singleton_returns_same_instance(self):
        """Test singleton pattern returns same instance."""
        mock_service = MagicMock()
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service",
            return_value=mock_service,
        ):
            first = get_sandbox_manager()
            second = get_sandbox_manager()
        assert first is second

    def test_initialized_flag_prevents_retry_on_none(self):
        """Test that once initialized with None, it doesn't retry."""
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service", return_value=None
        ) as mock_create:
            get_sandbox_manager()
            get_sandbox_manager()
            get_sandbox_manager()
        # Should only be called once due to _initialized flag
        mock_create.assert_called_once()

    def test_thread_safety(self):
        """Test concurrent access returns same instance."""
        mock_service = MagicMock()
        results = []
        barrier = threading.Barrier(5)

        def worker():
            barrier.wait()
            with patch(
                "xagent.web.sandbox_manager._create_sandbox_service",
                return_value=mock_service,
            ):
                results.append(get_sandbox_manager())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same instance
        assert all(r is results[0] for r in results)


try:
    from xagent.sandbox import BoxliteSandboxService  # noqa: F401

    _has_boxlite = True
except ImportError:
    _has_boxlite = False


@pytest.mark.skipif(not _has_boxlite, reason="boxlite not installed")
class TestCreateBoxliteService:
    """Test _create_boxlite_service function."""

    def test_custom_home_dir(self):
        """Test creating service with custom home directory."""
        with (
            patch.dict(
                "os.environ",
                {"BOXLITE_HOME_DIR": "/tmp/sandbox"},
                clear=False,
            ),
            patch(
                "xagent.sandbox.BoxliteSandboxService", return_value=MagicMock()
            ) as mock_cls,
            patch("xagent.sandbox.MemBoxliteStore", return_value=MagicMock()),
        ):
            _create_boxlite_service()

        assert mock_cls.call_args[1]["home_dir"] == "/tmp/sandbox"

    def test_creation_failure_returns_none(self):
        """Test that BoxliteSandboxService construction failure returns None."""
        with (
            patch(
                "xagent.sandbox.BoxliteSandboxService",
                side_effect=RuntimeError("docker not available"),
            ),
            patch("xagent.sandbox.MemBoxliteStore", return_value=MagicMock()),
        ):
            result = _create_boxlite_service()

        assert result is None


class TestCreateDockerService:
    """Test _create_docker_service function."""

    def test_uses_db_store(self):
        """Test Docker sandbox service is created with persistent store."""
        with (
            patch("xagent.web.sandbox_store.DBDockerStore") as mock_store_cls,
            patch(
                "xagent.sandbox.DockerSandboxService", return_value=MagicMock()
            ) as mock_service_cls,
        ):
            _create_docker_service()

        mock_store_cls.assert_called_once_with()
        assert mock_service_cls.call_args[1]["store"] is mock_store_cls.return_value

    def test_creation_failure_returns_none(self):
        """Test that DockerSandboxService construction failure returns None."""
        with (
            patch("xagent.web.sandbox_store.DBDockerStore", return_value=MagicMock()),
            patch(
                "xagent.sandbox.DockerSandboxService",
                side_effect=RuntimeError("docker not available"),
            ),
        ):
            result = _create_docker_service()

        assert result is None


class TestSandboxConfigParsing:
    """Test sandbox config parsing from environment variables."""

    def test_default_config_when_no_env_set(self):
        """Test default config when no env vars are set."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {}, clear=True):
            image, config = manager._get_sandbox_image_and_config()

        # Should use defaults - SandboxConfig has cpus=1, memory=512 as defaults
        assert image  # Should have some image value
        assert config.cpus == 1  # SandboxConfig default
        assert config.memory == 512  # SandboxConfig default
        assert config.env is None
        assert config.volumes is None

    def test_cpu_parsing_valid(self):
        """Test valid CPU value is parsed correctly."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_CPUS": "4"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.cpus == 4

    def test_cpu_parsing_invalid(self):
        """Test invalid CPU value uses SandboxConfig default (1)."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_CPUS": "invalid"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        # Invalid value is skipped, SandboxConfig default (1) is used
        assert config.cpus == 1

    def test_memory_parsing_valid(self):
        """Test valid memory value is parsed correctly."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_MEMORY": "2048"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.memory == 2048

    def test_env_parsing_single_var(self):
        """Test parsing single environment variable."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_ENV": "KEY=value"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY": "value"}

    def test_env_parsing_multiple_vars(self):
        """Test parsing multiple environment variables."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {"SANDBOX_ENV": "KEY1=value1;KEY2=value2;KEY3=value3"},
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY1": "value1", "KEY2": "value2", "KEY3": "value3"}

    def test_env_parsing_empty(self):
        """Test empty env string results in None."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_ENV": ""}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env is None

    def test_env_parsing_invalid_format(self):
        """Test invalid env format is skipped with warning."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_ENV": "VALID=1;INVALID;VALID2=2"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        # Should skip invalid entry
        assert config.env == {"VALID": "1", "VALID2": "2"}

    def test_env_parsing_with_spaces(self):
        """Test env vars with spaces are trimmed."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_ENV": " KEY = value ; ANOTHER = test "}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY": "value", "ANOTHER": "test"}

    def test_volume_parsing_single_volume(self):
        """Test parsing single volume mount."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container:ro"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "ro")]

    def test_volume_parsing_multiple_volumes(self):
        """Test parsing multiple volume mounts."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/host1:/container1:ro;/host2:/container2:rw"},
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [
            ("/host1", "/container1", "ro"),
            ("/host2", "/container2", "rw"),
        ]

    def test_volume_parsing_default_mode(self):
        """Test volume defaults to 'ro' mode when not specified."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "ro")]

    def test_volume_parsing_with_tilde_expansion(self):
        """Test volume path with tilde expansion."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "~/data:/data:ro"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        # Should expand tilde to absolute path
        src_path = config.volumes[0][0]
        expected_path = os.path.abspath(os.path.expanduser("~/data"))
        assert "~" not in src_path
        assert src_path == expected_path
        assert config.volumes[0][1] == "/data"
        assert config.volumes[0][2] == "ro"

    def test_sibling_mode_rejects_tilde_sandbox_volume_source(self):
        """Docker sibling mode must not expand backend-container home paths."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": "/host/.xagent",
                "SANDBOX_VOLUMES": "~/data:/data:ro",
            },
            clear=True,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes is None

    def test_sibling_mode_rejects_relative_sandbox_volume_source(self):
        """Docker sibling mode requires host-side absolute volume sources."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": "/host/.xagent",
                "SANDBOX_VOLUMES": "relative:/data:ro",
            },
            clear=True,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes is None

    def test_volume_parsing_empty(self):
        """Test empty volumes string results in None."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_VOLUMES": ""}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes is None

    def test_volume_parsing_invalid_format(self):
        """Test invalid volume format is skipped with warning."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/valid:/valid:ro;invalid;/another:/another:rw"},
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        # Should skip invalid entries
        assert config.volumes == [
            ("/valid", "/valid", "ro"),
            ("/another", "/another", "rw"),
        ]

    def test_volume_parsing_invalid_mode_defaults_to_ro(self):
        """Test invalid volume mode defaults to 'ro'."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container:xyz"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "ro")]

    def test_volume_parsing_mode_case_insensitive(self):
        """Test volume mode is case-insensitive."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container:RW"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "rw")]

    def test_combined_config(self):
        """Test parsing all config options together."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "SANDBOX_IMAGE": "custom/image:latest",
                "SANDBOX_CPUS": "2",
                "SANDBOX_MEMORY": "1024",
                "SANDBOX_ENV": "KEY1=val1;KEY2=val2",
                "SANDBOX_VOLUMES": "/host:/container:ro",
            },
            clear=False,
        ):
            image, config = manager._get_sandbox_image_and_config()

        assert image == "custom/image:latest"
        assert config.cpus == 2
        assert config.memory == 1024
        assert config.env == {"KEY1": "val1", "KEY2": "val2"}
        assert config.volumes == [("/host", "/container", "ro")]

    def test_volumes_with_semicolon_in_env(self):
        """Test env vars and volumes both use semicolon separator."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "SANDBOX_ENV": "KEY=val",
                "SANDBOX_VOLUMES": "/host:/container:ro",
            },
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY": "val"}
        assert config.volumes == [("/host", "/container", "ro")]


class TestSandboxLifecycleConfig:
    """Test sandbox lifecycle identity and runtime config consistency."""

    @pytest.mark.asyncio
    async def test_same_lifecycle_rejects_different_workspace_config(self, tmp_path):
        """Same sandbox name should not be deleted/recreated for another workspace."""
        service = AsyncMock()
        service.get_or_create = AsyncMock(return_value=MagicMock())
        service.delete = AsyncMock()
        manager = SandboxManager(service)

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            await manager.get_or_create_sandbox(
                "user",
                "42",
                workspace_config={"base_dir": str(tmp_path / "user_42")},
            )
            with pytest.raises(RuntimeError, match="different runtime configuration"):
                await manager.get_or_create_sandbox(
                    "user",
                    "42",
                    workspace_config={"base_dir": str(tmp_path / "build_preview")},
                )

        service.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_distinct_lifecycles_allow_distinct_workspace_configs(self, tmp_path):
        """Build preview and chat should not compete for the same sandbox name."""
        service = AsyncMock()
        service.get_or_create = AsyncMock(side_effect=[MagicMock(), MagicMock()])
        manager = SandboxManager(service)

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            await manager.get_or_create_sandbox(
                "user",
                "42",
                workspace_config={"base_dir": str(tmp_path / "user_42")},
            )
            await manager.get_or_create_sandbox(
                "build_preview",
                "42",
                workspace_config={"base_dir": str(tmp_path / "build_preview")},
            )

        assert service.get_or_create.await_count == 2
        assert service.get_or_create.await_args_list[0].args[0] == "user::42"
        assert service.get_or_create.await_args_list[1].args[0] == "build_preview::42"


class TestSandboxManagerWarmup:
    """Test sandbox warmup functionality."""

    @pytest.mark.asyncio
    async def test_warmup_uses_empty_config(self):
        """Test warmup uses empty config to avoid unnecessary mounts."""
        mock_service = MagicMock()
        manager = SandboxManager(mock_service)

        # Mock the service methods
        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = MagicMock(return_value=None)
        mock_service.get_or_create = MagicMock(return_value=mock_sandbox)
        mock_service.delete = MagicMock(return_value=None)

        # Set environment vars that would normally trigger mounts
        with patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/nonexistent:/path:ro", "SANDBOX_ENV": "TEST=value"},
            clear=False,
        ):
            await manager.warmup()

        # Verify get_or_create was called with empty config (no volumes/env)
        mock_service.get_or_create.assert_called_once()
        call_args = mock_service.get_or_create.call_args
        config = call_args[1]["config"]

        # Verify warmup config is empty (no volumes/env)
        assert config.volumes is None
        assert config.env is None
        # Should have default cpus/memory from SandboxConfig
        assert config.cpus == 1
        assert config.memory == 512
