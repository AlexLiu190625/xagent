from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text()


def read_repo_file(path: str) -> str:
    return (ROOT / path).read_text()


def pyproject_section(pyproject: str, section_name: str) -> str:
    marker = f"[{section_name}]"
    start = pyproject.index(marker)
    next_section = pyproject.find("\n[", start + len(marker))
    if next_section == -1:
        return pyproject[start:]
    return pyproject[start:next_section]


def test_nightly_build_uses_pep440_package_version() -> None:
    workflow = read_workflow("nightly-build.yml")

    assert 'NIGHTLY_VERSION="nightly-$NIGHTLY_DATE"' in workflow
    assert 'PACKAGE_VERSION="0.0.dev$NIGHTLY_DATE"' in workflow
    assert (
        "XAGENT_VERSION=${{ steps.version-meta.outputs.nightly_version }}" in workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in workflow
    )
    assert 'python scripts/write_package_version.py "$PACKAGE_VERSION"' not in workflow
    assert 'echo "package_version=$PACKAGE_VERSION" >> "$GITHUB_OUTPUT"' in workflow
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.nightly_version }}"
        not in workflow
    )


def test_release_build_sanitizes_package_version_for_manual_runs() -> None:
    workflow = read_workflow("docker-publish.yml")

    assert 'PACKAGE_VERSION="${RELEASE_VERSION#v}"' in workflow
    assert 'PACKAGE_VERSION="0.0.0+${GITHUB_SHA::12}"' in workflow
    assert (
        "XAGENT_VERSION=${{ steps.version-meta.outputs.release_version }}" in workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in workflow
    )
    assert 'python scripts/write_package_version.py "$PACKAGE_VERSION"' not in workflow
    assert 'echo "package_version=$PACKAGE_VERSION" >> "$GITHUB_OUTPUT"' in workflow
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.release_version }}"
        not in workflow
    )


def test_backend_dockerfile_applies_package_specific_vcs_version() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")

    assert dockerfile.count('ARG XAGENT_PACKAGE_VERSION="0.0.0+docker"') == 2
    assert (
        dockerfile.count('SETUPTOOLS_SCM_PRETEND_VERSION="${XAGENT_PACKAGE_VERSION}"')
        == 2
    )
    dependency_sync = (
        'SETUPTOOLS_SCM_PRETEND_VERSION="${XAGENT_PACKAGE_VERSION}" \\\n'
        "    VIRTUAL_ENV=/opt/venv uv sync --active --locked --no-dev "
        "--no-install-project --no-editable"
    )
    build_sync = (
        'SETUPTOOLS_SCM_PRETEND_VERSION="${XAGENT_PACKAGE_VERSION}" \\\n'
        "    VIRTUAL_ENV=/opt/venv uv sync --active --locked --no-dev --no-editable"
    )
    assert dependency_sync in dockerfile
    assert build_sync in dockerfile
    assert "COPY .git .git" not in dockerfile


def test_backend_dockerfile_uses_uv_deployment_sync() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")

    assert "uv pip compile" not in dockerfile
    assert "uv pip sync" not in dockerfile
    assert "--no-emit-package xagent" not in dockerfile
    assert dockerfile.count("uv sync") == 2
    assert dockerfile.count("--active") == 2
    assert dockerfile.count("--locked") == 2
    assert dockerfile.count("--no-dev") == 2
    assert dockerfile.count("--no-editable") == 2
    assert dockerfile.count("--group backend-image") == 2
    assert "--torch-backend" not in dockerfile
    assert "--no-install-project" in dockerfile
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "ENV UV_COMPILE_BYTECODE=1" in dockerfile
    assert "ENV UV_LINK_MODE=copy" in dockerfile
    assert "ENV UV_PYTHON_DOWNLOADS=0" in dockerfile


def test_backend_image_dependencies_are_deployment_group() -> None:
    pyproject = read_repo_file("pyproject.toml")
    optional_dependencies = pyproject_section(
        pyproject, "project.optional-dependencies"
    )
    dependency_groups = pyproject_section(pyproject, "dependency-groups")

    assert "backend-image = [" not in optional_dependencies
    assert "backend-image = [" in dependency_groups
    assert '"torch"' in dependency_groups
    assert '"torchvision"' in dependency_groups


def test_pytorch_cpu_index_is_project_configured_for_uv_sync() -> None:
    pyproject = read_repo_file("pyproject.toml")

    assert 'torch = [{ index = "pytorch-cpu" }]' in pyproject
    assert 'torchvision = [{ index = "pytorch-cpu" }]' in pyproject
    assert 'name = "pytorch-cpu"' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cpu"' in pyproject
    assert "explicit = true" in pyproject


def test_boxlite_is_not_declared_for_linux_aarch64() -> None:
    pyproject = read_repo_file("pyproject.toml")

    assert (
        "\"boxlite>=0.6.0; sys_platform == 'linux' and platform_machine == 'x86_64'\""
        in pyproject
    )
    assert (
        "\"boxlite>=0.6.0; sys_platform == 'darwin' and platform_machine == 'arm64'\""
        in pyproject
    )
    assert (
        "boxlite>=0.6.0; sys_platform == 'linux' and platform_machine == 'aarch64'"
        not in pyproject
    )


def test_publish_script_derives_package_version_from_valid_tags() -> None:
    publish_script = read_repo_file("docker/publish.sh")

    assert 'DEFAULT_PACKAGE_VERSION="${TAG#v}"' in publish_script
    assert (
        'PACKAGE_VERSION="${XAGENT_PACKAGE_VERSION:-${DEFAULT_PACKAGE_VERSION}}"'
        in publish_script
    )
    assert 'DEFAULT_PACKAGE_VERSION="0.0.0+${GIT_COMMIT::12}"' in publish_script
    assert 'XAGENT_VERSION="${XAGENT_VERSION:-${TAG}}"' in publish_script
    assert (
        'python "${REPO_ROOT}/scripts/write_package_version.py"' not in publish_script
    )
    assert '--build-arg "XAGENT_PACKAGE_VERSION=${PACKAGE_VERSION}"' in publish_script


def test_backend_dockerfile_uses_frontend_managed_pptxgenjs() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")
    package_json = read_repo_file("frontend/package.json")

    assert '"pptxgenjs": "4.0.1"' in package_json
    assert "npm install -g pptxgenjs" not in dockerfile
    assert "/usr/lib/node_modules/pptxgenjs" not in dockerfile
    assert 'ENV NODE_PATH="/opt/xagent/frontend/node_modules"' in dockerfile


def test_backend_runtime_keeps_uv_binaries() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")

    assert dockerfile.count("COPY --from=uv /uv /uvx /usr/local/bin/") == 2


def test_backend_package_version_is_vcs_based_for_normal_builds() -> None:
    pyproject = read_repo_file("pyproject.toml")

    assert 'dynamic = ["version"]' in pyproject
    assert 'requires = ["hatchling", "hatch-vcs"]' in pyproject
    assert 'source = "vcs"' in pyproject
    assert 'path = "src/xagent/_version.py"' not in pyproject
    assert not (ROOT / "src" / "xagent" / "_version.py").exists()
    assert not (ROOT / "scripts" / "write_package_version.py").exists()


def test_docker_workflows_pass_package_version_to_backend_build() -> None:
    release_workflow = read_workflow("docker-publish.yml")
    nightly_workflow = read_workflow("nightly-build.yml")

    assert (
        'python scripts/write_package_version.py "$PACKAGE_VERSION"'
        not in release_workflow
    )
    assert (
        'python scripts/write_package_version.py "$PACKAGE_VERSION"'
        not in nightly_workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in release_workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in nightly_workflow
    )


def test_docker_readme_documents_lockfile_requirement() -> None:
    readme = read_repo_file("docker/README.md")

    assert "`uv.lock` during the Docker build" in readme
    assert "uv sync --locked" in readme
    assert "uv.lock` is not copied" not in readme
