"""
Tests for CommandExecutor tool
"""

import os
import shlex
import sys
from collections.abc import Set as AbstractSet
from typing import TypeVar
from unittest.mock import Mock

import pytest

from xagent.core.tools.adapters.vibe.command_executor import (
    CommandExecutorArgs,
    CommandExecutorResult,
    CommandExecutorTool,
)
from xagent.core.tools.core import command_path_guard as command_path_guard_module
from xagent.core.tools.core.command_executor import (
    CommandExecutorCore,
    execute_command,
    execute_script,
)
from xagent.core.tools.core.command_path_guard import (
    CommandPathViolation,
    WorkspaceCommandPathGuard,
)
from xagent.core.workspace import TaskWorkspace

_SetValue = TypeVar("_SetValue")


@pytest.fixture
def command_executor():
    """Create CommandExecutorTool instance for testing"""
    return CommandExecutorTool()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for file operations"""
    return str(tmp_path)


@pytest.fixture
def scoped_command_workspace(tmp_path):
    """Workspace plus same-mount sibling and read-only external roots."""
    alice_base = tmp_path / "clients" / "1" / "end_users" / "7"
    external = tmp_path / "external" / "7"
    external.mkdir(parents=True, exist_ok=True)
    workspace = TaskWorkspace(
        "task",
        str(alice_base),
        allowed_external_dirs=[str(external)],
    )
    external_file = external / "reference.txt"
    external_file.write_text("external reference", encoding="utf-8")

    sibling = tmp_path / "clients" / "1" / "end_users" / "8"
    sibling.mkdir(parents=True, exist_ok=True)
    sibling_file = sibling / "secret.txt"
    sibling_file.write_text("sibling secret", encoding="utf-8")

    return workspace, external_file, sibling_file


class TestCommandExecutorTool:
    """Test cases for CommandExecutorTool"""

    def test_tool_properties(self, command_executor):
        """Test basic tool properties"""
        assert command_executor.name == "command_executor"
        assert "shell" in command_executor.tags or "command" in command_executor.tags
        assert command_executor.args_type() == CommandExecutorArgs
        assert command_executor.return_type() == CommandExecutorResult

    def test_description_includes_workspace_cwd_and_search_scope(self, tmp_path):
        """Test that shell guidance exposes cwd and discourages broad searches."""
        workspace = Mock()
        workspace.resolve_path.return_value = tmp_path
        tool = CommandExecutorTool(workspace=workspace)

        description = tool.description

        assert f"current working directory: {tmp_path}" in description
        assert "Use concrete paths" in description
        assert "Only search for files when no usable path was provided" in description
        assert "Do not run broad recursive searches from `/`" in description

    def test_simple_echo_command(self, command_executor):
        """Test simple echo command"""
        result = command_executor.run_json_sync({"command": "echo Hello World"})

        assert result["success"] is True
        assert "Hello World" in result["output"]
        assert result["error"] == ""
        assert result["return_code"] == 0

    def test_command_with_pipe(self, command_executor):
        """Test command with pipe operation"""
        result = command_executor.run_json_sync(
            {"command": 'echo "apple\\nbanana\\ncherry" | grep banana'}
        )

        assert result["success"] is True
        assert "banana" in result["output"]
        assert result["return_code"] == 0

    def test_list_directory(self, command_executor, temp_dir):
        """Test listing directory contents"""
        result = command_executor.run_json_sync(
            {"command": f"ls -la {shlex.quote(temp_dir)}"}
        )

        assert result["success"] is True
        assert len(result["output"]) > 0
        assert result["return_code"] == 0

    def test_command_with_timeout(self, command_executor):
        """Test command execution with timeout"""
        # Sleep command that should complete within timeout
        result = command_executor.run_json_sync({"command": "sleep 0.1", "timeout": 5})

        assert result["success"] is True
        assert result["return_code"] == 0

    def test_command_timeout_exceeded(self, command_executor):
        """Test command that exceeds timeout"""
        # Sleep longer than timeout
        result = command_executor.run_json_sync({"command": "sleep 5", "timeout": 1})

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert result["return_code"] == -999  # TIMEOUT_EXIT_CODE

    def test_invalid_command(self, command_executor):
        """Test handling of invalid command"""
        result = command_executor.run_json_sync({"command": "nonexistentcommand12345"})

        assert result["success"] is False
        assert (
            "not found" in result["error"].lower()
            or "command not found" in result["error"].lower()
        )
        assert result["return_code"] != 0

    def test_command_with_stderr(self, command_executor):
        """Test that stderr is captured"""
        result = command_executor.run_json_sync({"command": 'echo "error message" >&2'})

        # Command succeeds but stderr is captured
        assert result["success"] is True
        assert "error message" in result["error"]

    def test_command_failure_nonzero_exit(self, command_executor):
        """Test command that fails with non-zero exit code"""
        result = command_executor.run_json_sync(
            {"command": "ls /nonexistent_directory_12345"}
        )

        assert result["success"] is False
        assert result["return_code"] != 0

    def test_command_with_redirection(self, command_executor, temp_dir):
        """Test command with output redirection"""
        output_file = os.path.join(temp_dir, "output.txt")
        result = command_executor.run_json_sync(
            {"command": f'echo "test content" > {output_file}'}
        )

        assert result["success"] is True
        assert os.path.exists(output_file)
        with open(output_file) as f:
            assert "test content" in f.read()

    def test_command_chain(self, command_executor):
        """Test chaining multiple commands with &&"""
        result = command_executor.run_json_sync(
            {"command": 'echo "first" && echo "second"'}
        )

        assert result["success"] is True
        assert "first" in result["output"]
        assert "second" in result["output"]

    def test_command_with_quotes(self, command_executor):
        """Test command with quoted arguments"""
        result = command_executor.run_json_sync({"command": 'echo "hello world"'})

        assert result["success"] is True
        assert "hello world" in result["output"]

    def test_grep_command(self, command_executor):
        """Test grep command for text search"""
        result = command_executor.run_json_sync(
            {"command": 'echo -e "apple\\nbanana\\ncherry" | grep banana'}
        )

        assert result["success"] is True
        assert "banana" in result["output"]

    def test_wc_command(self, command_executor):
        """Test wc command for word count"""
        result = command_executor.run_json_sync(
            {"command": 'echo "test content here" | wc -w'}
        )

        assert result["success"] is True
        assert len(result["output"].strip()) > 0

    def test_cat_command(self, command_executor, temp_dir):
        """Test cat command to read file"""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        result = command_executor.run_json_sync({"command": f"cat {test_file}"})

        assert result["success"] is True
        assert "test content" in result["output"]

    def test_head_command(self, command_executor):
        """Test head command for limiting output"""
        result = command_executor.run_json_sync({"command": "seq 1 100 | head -5"})

        assert result["success"] is True
        assert "1" in result["output"]
        assert "5" in result["output"]

    def test_tail_command(self, command_executor):
        """Test tail command for showing end of file"""
        result = command_executor.run_json_sync({"command": "seq 1 10 | tail -3"})

        assert result["success"] is True
        assert "8" in result["output"]
        assert "10" in result["output"]

    @pytest.mark.asyncio
    async def test_async_execution_same_as_sync(self, command_executor):
        """Test that async execution produces same results as sync"""
        command = "echo test"

        sync_result = command_executor.run_json_sync({"command": command})
        async_result = await command_executor.run_json_async({"command": command})

        assert sync_result == async_result

    def test_args_validation(self):
        """Test CommandExecutorArgs validation"""
        # Valid args with defaults
        args = CommandExecutorArgs(command="ls")
        assert args.command == "ls"
        assert args.timeout is None  # default

        # Custom args
        args = CommandExecutorArgs(command="sleep 1", timeout=5)
        assert args.command == "sleep 1"
        assert args.timeout == 5

    def test_result_model(self):
        """Test CommandExecutorResult model"""
        # Success result
        result = CommandExecutorResult(
            success=True, output="test output", error="", return_code=0
        )
        assert result.success is True
        assert result.output == "test output"
        assert result.error == ""
        assert result.return_code == 0

        # Error result
        result = CommandExecutorResult(
            success=False, output="", error="Some error", return_code=1
        )
        assert result.success is False
        assert result.output == ""
        assert result.error == "Some error"
        assert result.return_code == 1


class TestScopedCommandPathGuard:
    """Cooperative command path checks enabled only for scoped executions."""

    def test_rejects_shell_false_argv_read_from_sibling(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        executor = CommandExecutorCore(
            str(workspace.resolve_path("")),
            path_guard=WorkspaceCommandPathGuard(workspace),
        )

        result = executor.execute_command(
            ["cat", str(sibling_file)],
            shell=False,
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "outside allowed read paths" in result["error"]
        assert "sibling secret" not in result["output"]

    def test_rejects_shell_true_list_when_path_guard_is_enabled(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        executor = CommandExecutorCore(
            str(workspace.resolve_path("")),
            path_guard=WorkspaceCommandPathGuard(workspace),
        )

        result = executor.execute_command(
            [f"cat {shlex.quote(str(sibling_file))}"],
            shell=True,
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "requires a string command" in result["error"]
        assert "sibling secret" not in result["output"]

    def test_allows_shell_false_argv_inside_workspace(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("own content", encoding="utf-8")
        executor = CommandExecutorCore(
            str(workspace.resolve_path("")),
            path_guard=WorkspaceCommandPathGuard(workspace),
        )

        result = executor.execute_command(
            ["cat", str(own_file)],
            shell=False,
        )

        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["output"] == "own content"

    def test_creator_behavior_remains_unrestricted(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace)

        result = tool.run_json_sync(
            {"command": f"cat {shlex.quote(str(sibling_file))}"}
        )

        assert result["success"] is True
        assert "sibling secret" in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat {path}",
            "head -n 1 {path}",
            "tail -n 1 {path}",
            "grep sibling {path}",
            "grep --regexp=sibling {path}",
            "sed -n '1p' {path}",
            "sed -f {path} own.txt",
            "sed -f{path} own.txt",
            "awk '{{print}}' {path}",
            "awk -f {path} own.txt",
            "awk -f{path} own.txt",
            "find {parent} -type f -exec cat {{}} \\;",
            "find -L {parent} -type f -exec cat {{}} \\;",
        ],
    )
    def test_rejects_common_reads_from_sibling(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        command = command_template.format(
            path=shlex.quote(str(sibling_file)),
            parent=shlex.quote(str(sibling_file.parent)),
        )

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "outside allowed read paths" in result["error"]
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "rm {path}",
            "cp own.txt {path}",
            "cp --target-directory={parent} own.txt",
            "mv own.txt {path}",
            "mv --target-directory={parent} own.txt",
            "sed -i.bak 's/secret/changed/' {path}",
            "sort -o {path} own.txt",
            "uniq own.txt {path}",
            "diff --output={path} own.txt own.txt",
            "echo changed > {path}",
            "(echo changed) > {path}",
        ],
    )
    def test_rejects_common_writes_to_sibling_without_partial_execution(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("own", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        command = command_template.format(
            path=shlex.quote(str(sibling_file)),
            parent=shlex.quote(str(sibling_file.parent)),
        )

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "outside allowed write paths" in result["error"]
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"
        assert own_file.exists()

    def test_external_directory_is_read_only(self, scoped_command_workspace):
        workspace, external_file, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        read_result = tool.run_json_sync(
            {"command": f"cat {shlex.quote(str(external_file))}"}
        )
        write_result = tool.run_json_sync(
            {"command": f"echo changed > {shlex.quote(str(external_file))}"}
        )

        assert read_result["success"] is True
        assert "external reference" in read_result["output"]
        assert write_result["success"] is False
        assert "outside allowed write paths" in write_result["error"]
        assert external_file.read_text(encoding="utf-8") == "external reference"

    def test_rejects_cd_and_symlink_escapes(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        link = workspace.output_dir / "sibling-link"
        link.symlink_to(sibling_file)
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        cd_result = tool.run_json_sync(
            {
                "command": (
                    f"cd {shlex.quote(str(sibling_file.parent))} && cat secret.txt"
                )
            }
        )
        link_result = tool.run_json_sync({"command": "cat sibling-link"})

        assert cd_result["success"] is False
        assert link_result["success"] is False
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    @pytest.mark.parametrize(
        "command_template",
        [
            "sh -c 'cat {path}'",
            "sh -lc 'cat {path}'",
            "xargs cat {path}",
            "xargs -n 1 cat {path}",
            "xargs -I {{}} cat {path}",
            "wc -c < {path}",
            "(cat {path})",
            "{{ cd {parent}; cat secret.txt; }}",
            "if true; then cat {path}; fi",
            "find . -exec sh -c 'cat {path}' \\;",
            "find . -exec grep sibling {path} \\;",
            "xargs -n 1 grep sibling {path}",
            "find . -exec find {parent} -type f \\;",
        ],
    )
    def test_rejects_nested_shell_xargs_and_input_redirection(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        command = command_template.format(
            path=shlex.quote(str(sibling_file)),
            parent=shlex.quote(str(sibling_file.parent)),
        )

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126

    @pytest.mark.parametrize("redirect", ["2>&1", "1>&2", "3<&0"])
    def test_descriptor_duplication_is_not_treated_as_path(
        self, scoped_command_workspace, redirect
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"cd {shlex.quote(str(external_file.parent))} && printf ok {redirect}"
        )

    @pytest.mark.parametrize(
        "command_template",
        [
            "bash --rcfile {path} -i",
            "bash --rcfile={path} -i",
            "bash --init-file {path} -i -c exit",
            "bash --rcfile {path} -i -c exit",
        ],
    )
    def test_rejects_bash_file_options_outside_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    def test_shell_c_positional_arguments_are_not_treated_as_file_paths(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"bash -c 'printf %s \"$1\"' ignored {shlex.quote(str(sibling_file))}"
        )

    def test_malformed_top_level_shell_input_remains_cooperative(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "echo '"})

        assert result["success"] is False
        assert result["return_code"] != 126

    def test_malformed_nested_shell_input_remains_cooperative(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": 'sh -c "echo \'"'})

        assert result["success"] is False
        assert result["return_code"] != 126

    @pytest.mark.parametrize(
        "command_template",
        [
            "sed 'w {path}' own.txt",
            "sed '/own/r {path}' own.txt",
            "sed 's#.*#cat {path}#e' own.txt",
            """awk 'BEGIN {{print "x" > "{path}"}}'""",
            """awk 'BEGIN {{system("cat {path}")}}'""",
        ],
    )
    def test_rejects_static_embedded_sed_awk_file_io(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {"command": command_template.format(path=str(sibling_file))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    def test_read_path_option_union_is_computed_once(self, scoped_command_workspace):
        class CountingSet(set[str]):
            union_count = 0

            def __or__(self, other: AbstractSet[_SetValue]) -> set[str | _SetValue]:
                self.union_count += 1
                return super().__or__(other)

        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        short_options = CountingSet({"-f"})

        remaining = guard._check_read_path_options(
            ("first.txt", "second.txt"),
            workspace.output_dir,
            short_options=short_options,
            long_options={"--files-from"},
        )

        assert remaining == ["first.txt", "second.txt"]
        assert short_options.union_count == 1

    def test_embedded_io_patterns_are_precompiled(
        self, scoped_command_workspace, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        monkeypatch.setattr(command_path_guard_module, "re", None)

        guard._reject_embedded_io("awk", "BEGIN { print 1 }")
        assert guard._sed_has_unsafe_io("s/a/b/") is False

        with pytest.raises(CommandPathViolation):
            guard._reject_embedded_io("awk", 'BEGIN { system("cat secret") }')
        assert guard._sed_has_unsafe_io("w secret.txt") is True

    @pytest.mark.parametrize(
        ("script_name", "script_template", "invocation"),
        [
            ("dynamic.sed", "w {path}", "sed -f dynamic.sed own.txt"),
            (
                "dynamic.awk",
                'BEGIN {{print "changed" > "{path}"}}',
                "awk -f dynamic.awk own.txt",
            ),
        ],
    )
    def test_rejects_dynamically_created_sed_awk_scripts_with_embedded_io(
        self,
        scoped_command_workspace,
        script_name,
        script_template,
        invocation,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        script = script_template.format(path=sibling_file)

        result = tool.run_json_sync(
            {
                "command": (
                    f"printf '%s\\n' {shlex.quote(script)} > {script_name} "
                    f"&& {invocation}"
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert not (workspace.output_dir / script_name).exists()
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    @pytest.mark.parametrize("kind", ["device", "fifo"])
    def test_script_inspection_rejects_non_regular_files_without_reading(
        self, scoped_command_workspace, monkeypatch, kind
    ):
        workspace, _, _ = scoped_command_workspace
        script_path = workspace.output_dir / "script.sed"
        if kind == "device":
            script_path = type(script_path)("/dev/null")
            if not script_path.exists():
                pytest.skip("/dev/null is unavailable")
        else:
            os.mkfifo(script_path)

        read_paths = []

        def record_read(path, *args, **kwargs):
            read_paths.append(path)
            return ""

        monkeypatch.setattr(type(script_path), "read_text", record_read)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard._inspect_script_file("sed", str(script_path), workspace.output_dir)

        assert read_paths == []

    def test_script_inspection_rejects_oversized_regular_file(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        script_path = workspace.output_dir / "large.sed"
        script_path.write_bytes(b"#" * (1024 * 1024 + 1))
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard._inspect_script_file("sed", str(script_path), workspace.output_dir)

    @pytest.mark.parametrize(
        "nested_command",
        [
            "cp own.txt {}",
            "cp --target-directory={} own.txt",
            "sort -o {} own.txt",
            "uniq own.txt {}",
            "diff --output={} own.txt own.txt",
            "sh -c 'printf changed > {}'",
        ],
    )
    def test_find_exec_placeholder_write_requires_writable_root(
        self, scoped_command_workspace, nested_command
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        command = (
            f"find {shlex.quote(str(external_file.parent))} "
            f"-type f -exec {nested_command} \\;"
        )

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command)

        assert exc_info.value.access == "write"

    def test_find_exec_read_placeholder_keeps_read_only_root_allowed(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"find {shlex.quote(str(external_file.parent))} "
            "-type f -exec cp {} copy.txt \\;"
        )

    def test_path_resolution_failure_returns_policy_rejection(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        loop = workspace.output_dir / "loop"
        loop.symlink_to("loop")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "cat loop"})

        assert result["success"] is False
        assert result["return_code"] == 126

    def test_rejects_bare_cd_before_relative_file_access(
        self, scoped_command_workspace, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        outside_home = workspace.base_dir.parent / "outside-home"
        outside_home.mkdir()
        (outside_home / "secret.txt").write_text("outside", encoding="utf-8")
        monkeypatch.setenv("HOME", str(outside_home))
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "cd; cat secret.txt"})

        assert result["success"] is False
        assert result["return_code"] == 126

    def test_find_exec_write_treats_read_only_root_as_write_target(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    f"find {shlex.quote(str(external_file.parent))} "
                    "-type f -exec rm {} \\;"
                )
            }
        )

        assert result["success"] is False
        assert "outside allowed write paths" in result["error"]
        assert external_file.exists()

    @pytest.mark.parametrize("marker", ["-exec", "-execdir"])
    def test_find_later_exec_write_treats_read_only_root_as_write_target(
        self, scoped_command_workspace, marker
    ):
        workspace, external_file, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    f"find {shlex.quote(str(external_file.parent))} -type f "
                    f"{marker} cat {{}} \\; {marker} rm {{}} \\;"
                )
            }
        )

        assert result["success"] is False
        assert "outside allowed write paths" in result["error"]
        assert external_file.exists()

    def test_find_exec_copy_rejects_out_of_scope_destination(
        self, scoped_command_workspace
    ):
        workspace, external_file, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    f"find {shlex.quote(str(external_file.parent))} "
                    f"-type f -exec cp {{}} {shlex.quote(str(sibling_file.parent))} \\;"
                )
            }
        )

        assert result["success"] is False
        assert "outside allowed write paths" in result["error"]
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    def test_unknown_command_is_not_blanket_disabled(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "printf allowed"})

        assert result["success"] is True
        assert result["output"] == "allowed"

    def test_allows_supported_commands_inside_workspace(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("alpha", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    "cp own.txt copy.txt "
                    "&& sed -i.bak 's/alpha/beta/' copy.txt "
                    "&& find . -name copy.txt -exec cat {} \\; "
                    "&& printf 'left/right\\n' | cut -d / -f 2"
                )
            }
        )

        assert result["success"] is True
        assert "beta" in result["output"]
        assert "right" in result["output"]
        assert (workspace.output_dir / "copy.txt").read_text(encoding="utf-8") == "beta"


class TestCommandExecutorCore:
    """Test cases for CommandExecutorCore"""

    def test_basic_execution(self):
        """Test basic command execution"""
        executor = CommandExecutorCore()
        result = executor.execute_command("echo test")

        assert result["success"] is True
        assert "test" in result["output"]
        assert result["return_code"] == 0

    def test_working_directory_change(self, tmp_path):
        """Test execution in specific working directory"""
        test_dir = str(tmp_path)
        executor = CommandExecutorCore(working_directory=test_dir)

        result = executor.execute_command("pwd")

        assert result["success"] is True
        assert test_dir in result["output"]

    def test_custom_timeout(self):
        """Test custom timeout setting"""
        executor = CommandExecutorCore()

        # Should complete within default timeout
        result = executor.execute_command("sleep 0.1")

        assert result["success"] is True

        # Test with custom timeout parameter
        result = executor.execute_command("sleep 0.1", timeout=5)
        assert result["success"] is True

    def test_shell_parameter(self):
        """Test shell parameter"""
        executor = CommandExecutorCore()

        # With shell=True (default)
        result = executor.execute_command("echo test", shell=True)
        assert result["success"] is True

        # With shell=False
        result = executor.execute_command(["echo", "test"], shell=False)
        assert result["success"] is True


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_execute_command_function(self):
        """Test execute_command convenience function"""
        result = execute_command("echo convenience test")

        assert result["success"] is True
        assert "convenience test" in result["output"]

    def test_execute_command_with_working_directory(self, tmp_path):
        """Test execute_command with working directory"""
        test_dir = str(tmp_path)
        result = execute_command("pwd", working_directory=test_dir)

        assert result["success"] is True
        assert test_dir in result["output"]

    def test_execute_command_with_timeout(self):
        """Test execute_command with timeout"""
        result = execute_command("sleep 0.1", timeout=5)

        assert result["success"] is True

    def test_execute_script_function(self):
        """Test execute_script convenience function"""
        script = """
echo "Script line 1"
echo "Script line 2"
"""
        result = execute_script(script, interpreter="bash")

        assert result["success"] is True
        assert "Script line 1" in result["output"]
        assert "Script line 2" in result["output"]

    def test_execute_script_with_working_directory(self, tmp_path):
        """Test execute_script with working directory"""
        test_dir = str(tmp_path)
        script = "pwd"
        result = execute_script(script, interpreter="bash", working_directory=test_dir)

        assert result["success"] is True
        assert test_dir in result["output"]

    def test_execute_script_with_timeout(self):
        """Test execute_script with timeout"""
        script = "#!/bin/bash\nsleep 0.1"
        result = execute_script(script, interpreter="bash", timeout=5)

        assert result["success"] is True


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_command(self, command_executor):
        """Test handling of empty command"""
        result = command_executor.run_json_sync({"command": ""})

        # Empty command actually succeeds in shell (returns exit code 0)
        # but produces no output
        assert result["success"] is True
        assert result["output"] == ""
        assert result["return_code"] == 0

    def test_very_long_command(self, command_executor):
        """Test handling of very long command"""
        long_command = "echo " + "x" * 10000
        result = command_executor.run_json_sync({"command": long_command})

        # Should handle long commands
        assert result["success"] is True

    def test_command_with_special_characters(self, command_executor):
        """Test command with special characters"""
        result = command_executor.run_json_sync({"command": 'echo "test@#$%^&*()"'})
        assert result["success"] is True
        assert "test@#$%^&*()" in result["output"]

    def test_command_with_newlines(self, command_executor):
        """Test command with embedded newlines"""
        result = command_executor.run_json_sync(
            {"command": 'echo "line1\\nline2\\nline3"'}
        )

        assert result["success"] is True
        assert "line1" in result["output"]
        assert "line2" in result["output"]
        assert "line3" in result["output"]

    def test_zero_timeout(self, command_executor):
        """Test command with zero timeout"""
        # Zero timeout should now raise ValueError
        with pytest.raises(ValueError, match="timeout must be positive"):
            command_executor.run_json_sync({"command": "echo test", "timeout": 0})

    def test_negative_timeout(self, command_executor):
        """Test command with negative timeout"""
        # Negative timeout should now raise ValueError
        with pytest.raises(ValueError, match="timeout must be positive"):
            command_executor.run_json_sync({"command": "echo test", "timeout": -1})


class TestPlatformSpecific:
    """Platform-specific tests"""

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_macos_specific_command(self, command_executor):
        """Test macOS-specific command"""
        result = command_executor.run_json_sync({"command": "sw_vers"})

        assert result["success"] is True
        assert "macOS" in result["output"] or "Product" in result["output"]

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_linux_specific_command(self, command_executor):
        """Test Linux-specific command"""
        result = command_executor.run_json_sync({"command": "uname -a"})

        assert result["success"] is True
        assert "Linux" in result["output"]

    def test_uname_command(self, command_executor):
        """Test uname command (works on most Unix-like systems)"""
        result = command_executor.run_json_sync({"command": "uname"})

        assert result["success"] is True
        assert len(result["output"].strip()) > 0


class TestExecuteScriptFunction:
    """Test cases for the execute_script convenience function"""

    def test_execute_script_function(self):
        """Test execute_script convenience function"""
        script = "#!/bin/bash\necho 'script output'"
        result = execute_script(script, interpreter="bash")

        assert result["success"] is True
        assert "script output" in result["output"]

    def test_execute_script_with_python(self):
        """Test execute_script with Python interpreter"""
        script = "print('python script output')"
        result = execute_script(script, interpreter="python")

        assert result["success"] is True
        assert "python script output" in result["output"]

    def test_execute_script_with_timeout(self):
        """Test execute_script with timeout"""
        script = "#!/bin/bash\nsleep 0.1"
        result = execute_script(script, interpreter="bash", timeout=5)

        assert result["success"] is True


class TestConcurrentExecution:
    """Test cases for concurrent command execution"""

    def test_concurrent_execution(self):
        """Test that concurrent executions don't interfere"""
        import threading

        results = []

        def run_cmd(work_dir, thread_id):
            try:
                executor = CommandExecutorCore(working_directory=work_dir)
                result = executor.execute_command("pwd")
                results.append((thread_id, result["output"].strip(), result["success"]))
            except Exception as e:
                results.append((thread_id, str(e), False))

        threads = [
            threading.Thread(target=run_cmd, args=("/tmp", 1)),
            threading.Thread(target=run_cmd, args=("/home", 2)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Both threads should complete successfully
        assert len(results) == 2
        for tid, output, success in results:
            assert success is True
            assert len(output) > 0


class TestTimeoutValidation:
    """Test cases for timeout validation"""

    def test_negative_timeout_raises_error(self):
        """Test that negative timeout raises ValueError"""
        executor = CommandExecutorCore()

        with pytest.raises(ValueError, match="timeout must be positive"):
            executor.execute_command("echo test", timeout=-1)

    def test_zero_timeout_raises_error(self):
        """Test that zero timeout raises ValueError"""
        executor = CommandExecutorCore()

        with pytest.raises(ValueError, match="timeout must be positive"):
            executor.execute_command("echo test", timeout=0)


class TestWorkingDirectoryValidation:
    """Test cases for working directory validation"""

    def test_nonexistent_working_directory(self):
        """Test that nonexistent working directory raises FileNotFoundError"""
        executor = CommandExecutorCore(working_directory="/nonexistent/path/xyz")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            executor.execute_command("echo test")

    def test_file_as_working_directory(self, tmp_path):
        """Test that using a file (not directory) as working directory raises error"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        executor = CommandExecutorCore(working_directory=str(test_file))

        with pytest.raises(NotADirectoryError, match="not a directory"):
            executor.execute_command("echo test")


class TestOutputSizeLimit:
    """Test cases for output size limiting"""

    def test_large_output_truncation(self, command_executor):
        """Test that very large output is truncated"""
        # Generate a command that produces lots of output (more than 10MB)
        # Use Python to generate large output
        result = command_executor.run_json_sync(
            {"command": "python -c \"print('x' * 11_000_000)\""}
        )

        assert result["success"] is True
        # Output should be truncated
        assert "[OUTPUT TRUNCATED]" in result["output"]
        # Output should be truncated to MAX_OUTPUT_SIZE + suffix
        assert len(result["output"]) <= 10 * 1024 * 1024 + 100
