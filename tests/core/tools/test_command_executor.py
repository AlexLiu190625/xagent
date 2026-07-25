"""
Tests for CommandExecutor tool
"""

import os
import shlex
import sys
from unittest.mock import Mock

import pytest

from xagent.core.tools.adapters.vibe.command_executor import (
    CommandExecutorArgs,
    CommandExecutorResult,
    CommandExecutorTool,
    CommandExecutorToolForBasic,
    create_command_executor_tool,
)
from xagent.core.tools.core.command_executor import (
    CommandExecutorCore,
    execute_command,
    execute_script,
)
from xagent.core.tools.core.command_path_guard import (
    CommandPathViolation,
    CommandPolicyViolation,
    WorkspaceCommandPathGuard,
)
from xagent.core.workspace import TaskWorkspace


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

    def test_restricted_description_states_guard_boundary(self, tmp_path):
        workspace = Mock()
        workspace.resolve_path.return_value = tmp_path
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        description = tool.description

        assert "best-effort" in description
        assert "not an operating-system security boundary" in description
        assert "unknown commands" in description
        assert "xargs" in description
        assert "active globs" in description
        assert "./deploy.sh" not in description

    @pytest.mark.asyncio
    @pytest.mark.parametrize("restricted", [False, True])
    async def test_basic_tools_project_policy_from_execution_scope(
        self,
        tmp_path,
        restricted,
    ):
        from xagent.core.execution_scope import ExecutionScope
        from xagent.core.tools.adapters.vibe.basic_tools import create_basic_tools
        from xagent.core.tools.adapters.vibe.config import ToolConfig

        config = ToolConfig(
            {
                "workspace": {
                    "task_id": "42",
                    "base_dir": str(tmp_path),
                },
                "tool_credentials": {},
            }
        )
        scope = ExecutionScope(restrict_command_paths=restricted)
        config.get_execution_scope = lambda: scope

        tools = await create_basic_tools(config)

        command_tool = next(
            tool for tool in tools if isinstance(tool, CommandExecutorToolForBasic)
        )
        sibling = tmp_path.parent / f"{tmp_path.name}-sibling-secret"
        sibling.write_text("sibling secret", encoding="utf-8")
        result = command_tool.run_json_sync(
            {"command": f"cat {shlex.quote(str(sibling))}"}
        )

        assert result["success"] is (not restricted)
        assert ("sibling secret" in result["output"]) is (not restricted)

    def test_execution_scope_factory_projects_command_path_policy(self, tmp_path):
        from xagent.core.execution_scope import ExecutionScope

        workspace = TaskWorkspace("task", str(tmp_path))
        tool = CommandExecutorToolForBasic.from_execution_scope(
            workspace,
            ExecutionScope(restrict_command_paths=True),
        )
        sibling = tmp_path.parent / f"{tmp_path.name}-factory-secret"
        sibling.write_text("secret", encoding="utf-8")

        result = tool.run_json_sync({"command": f"cat {shlex.quote(str(sibling))}"})

        assert result["return_code"] == 126
        assert "secret" not in result["output"]

    def test_public_factory_uses_execution_scope_policy(self, tmp_path):
        from xagent.core.execution_scope import ExecutionScope

        workspace = TaskWorkspace("task", str(tmp_path))
        tool = create_command_executor_tool(
            workspace,
            execution_scope=ExecutionScope(restrict_command_paths=True),
        )
        sibling = tmp_path.parent / f"{tmp_path.name}-public-factory-secret"
        sibling.write_text("secret", encoding="utf-8")

        assert isinstance(tool, CommandExecutorTool)
        result = tool.run_json_sync({"command": f"cat {shlex.quote(str(sibling))}"})
        assert result["return_code"] == 126
        assert "secret" not in result["output"]

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

    def test_rejects_shell_true_list_without_path_guard(self, tmp_path):
        executor = CommandExecutorCore(str(tmp_path))

        result = executor.execute_command(["printf", "dropped-argument"], shell=True)

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "shell=True requires a string command" in result["error"]

    def test_guarded_execute_script_uses_same_shell_policy(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        executor = CommandExecutorCore.for_workspace(
            workspace,
            restrict_paths=True,
        )

        result = executor.execute_script(
            f"cat {shlex.quote(str(sibling_file))}",
            interpreter="bash",
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_convenience_functions_project_execution_scope(
        self,
        scoped_command_workspace,
    ):
        from xagent.core.execution_scope import ExecutionScope

        workspace, _, sibling_file = scoped_command_workspace
        scope = ExecutionScope(restrict_command_paths=True)

        command_result = execute_command(
            f"cat {shlex.quote(str(sibling_file))}",
            workspace=workspace,
            execution_scope=scope,
        )
        script_result = execute_script(
            f"cat {shlex.quote(str(sibling_file))}",
            workspace=workspace,
            execution_scope=scope,
        )

        assert command_result["return_code"] == 126
        assert script_result["return_code"] == 126
        assert "sibling secret" not in command_result["output"]
        assert "sibling secret" not in script_result["output"]

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

    def test_shell_false_argv_treats_braces_as_literal(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        literal_file = workspace.output_dir / "{a,b}"
        literal_file.write_text("literal content", encoding="utf-8")
        executor = CommandExecutorCore(
            str(workspace.resolve_path("")),
            path_guard=WorkspaceCommandPathGuard(workspace),
        )

        result = executor.execute_command(["cat", "{a,b}"], shell=False)

        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["output"] == "literal content"

    @pytest.mark.parametrize("literal_name", ["*", "?", "[ab]"])
    def test_shell_false_argv_treats_globs_as_literal(
        self, scoped_command_workspace, literal_name
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / literal_name).write_text(
            "literal content", encoding="utf-8"
        )
        executor = CommandExecutorCore(
            str(workspace.resolve_path("")),
            path_guard=WorkspaceCommandPathGuard(workspace),
        )

        result = executor.execute_command(["cat", literal_name], shell=False)

        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["output"] == "literal content"

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
            "sort -o{path} own.txt",
            "sort -ro{path} own.txt",
            "sort -T{parent} own.txt",
            "sort -rT{parent} own.txt",
            "sort --out={path} own.txt",
            "sort --temp={parent} own.txt",
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

    @pytest.mark.parametrize(
        "command",
        [
            'sort -t "$SEPARATOR" own.txt',
            'sort -t"$SEPARATOR" own.txt',
            'sort -k "$KEY" own.txt',
            'sort -k"$KEY" own.txt',
            'sort --field-separator="$SEPARATOR" own.txt',
        ],
    )
    def test_sort_dynamic_scalar_option_values_are_not_paths(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            'sort -o"$TARGET" own.txt',
            'sort -ro"$TARGET" own.txt',
            'sort -T"$TARGET" own.txt',
            'sort -rT"$TARGET" own.txt',
            'sort --output="$TARGET" own.txt',
            'sort --temporary-directory="$TARGET" own.txt',
        ],
    )
    def test_rejects_dynamic_sort_path_option_values(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command_template)

    @pytest.mark.parametrize(
        "command_template",
        [
            "sort --files0-from={path}",
            "sort --files0={path}",
            "sort --random-source={path} own.txt",
            "sort --random-sour={path} own.txt",
        ],
    )
    def test_rejects_sort_read_control_files_outside_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "sort --not-a-sort-option own.txt",
            "sort --random-s=/dev/zero own.txt",
            "sort --compress-program=cat own.txt",
            "sort -q own.txt",
        ],
    )
    def test_rejects_unrecognized_or_ambiguous_sort_options(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "sort --human-numeric-sort own.txt",
            "sort --check=quiet own.txt",
            "sort -bdfgiMhnRrVcmsuz own.txt",
        ],
    )
    def test_allows_recognized_sort_flag_options(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat {{{path},missing}}",
            "{{cat,printf}} {path}",
            "rm -f {{{path},missing}}",
            "tee {{{path},own.txt}} < /dev/null",
            "printf changed > {{{path},own.txt}}",
            "bash -c 'cat {{{path},missing}}'",
        ],
    )
    def test_rejects_brace_expansion_before_read_write_or_nested_execution(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        command = command_template.format(path=str(sibling_file))

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    @pytest.mark.parametrize("operand", ["{a,b}", "{1..3}", "{a..z..2}"])
    def test_rejects_unmodeled_brace_expansion_forms(
        self, scoped_command_workspace, operand
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="dynamic path operand"):
            guard.validate(f"cat {operand}")

    @pytest.mark.parametrize("operand", ["'{a,b}'", r"\{1..3\}"])
    def test_allows_quoted_or_escaped_literal_braces(
        self, scoped_command_workspace, operand
    ):
        workspace, _, _ = scoped_command_workspace
        literal_name = operand.replace("'", "").replace("\\", "")
        (workspace.output_dir / literal_name).write_text("literal", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": f"cat {operand}"})

        assert result["success"] is True
        assert result["output"] == "literal"

    @pytest.mark.parametrize("operand", ["*", "file?.txt", "[ab].txt"])
    def test_rejects_unmodeled_glob_expansion_forms(
        self, scoped_command_workspace, operand
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="dynamic path operand"):
            guard.validate(f"cat {operand}")

    @pytest.mark.parametrize(
        ("operand", "literal_name"),
        [
            ("'*'", "*"),
            ('"file?.txt"', "file?.txt"),
            (r"\[ab\].txt", "[ab].txt"),
        ],
    )
    def test_allows_quoted_or_escaped_literal_globs(
        self, scoped_command_workspace, operand, literal_name
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / literal_name).write_text("literal", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": f"cat {operand}"})

        assert result["success"] is True
        assert result["output"] == "literal"

    def test_rejects_glob_that_expands_to_out_of_scope_symlink(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        glob_dir = workspace.output_dir / "glob-dir"
        glob_dir.mkdir()
        (glob_dir / "leak").symlink_to(sibling_file)
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "cat glob-dir/*"})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

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

    @pytest.mark.parametrize(
        "options",
        ["-ni", "-ri", "-si", "-zi", "-Ei", "-nri", "-ni.backup"],
    )
    def test_rejects_bundled_sed_in_place_write_to_external_directory(
        self, scoped_command_workspace, options
    ):
        workspace, external_file, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    f"sed {options} 's/external/changed/' "
                    f"{shlex.quote(str(external_file))}"
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "outside allowed write paths" in result["error"]
        assert external_file.read_text(encoding="utf-8") == "external reference"

    def test_sed_short_option_values_are_not_reparsed_as_in_place_flags(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        script_file = workspace.output_dir / "inline.sed"
        script_file.write_text("s/external/reference/", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"sed -nes/external/reference/ {shlex.quote(str(external_file))}"
        )
        guard.validate(
            f"sed -nf{shlex.quote(str(script_file))} {shlex.quote(str(external_file))}"
        )

    def test_rejects_bundled_sed_script_file_outside_workspace(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"sed -nf{shlex.quote(str(sibling_file))} own.txt")

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command_template",
        [
            "file -f {path}",
            "file -m {path} own.txt",
            "file --files-from={path}",
            "file --magic-file={path} own.txt",
            "wc --files0-from={path}",
        ],
    )
    def test_rejects_read_control_file_options_outside_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

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

    @pytest.mark.parametrize(
        "command_template",
        [
            "exec cat {path}",
            "env cat {path}",
            "env -i NAME=value cat {path}",
            "timeout --signal=TERM 5 cat {path}",
            "nohup cat {path}",
            "nice -n 5 cat {path}",
            "stdbuf -oL cat {path}",
            "command -p cat {path}",
            "setsid -f cat {path}",
            "ionice -c 2 -n 7 cat {path}",
        ],
    )
    def test_rejects_supported_wrapper_commands_accessing_sibling(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command",
        [
            "exec cat own.txt",
            "env cat own.txt",
            "env -i NAME=value cat own.txt",
            "timeout --signal=TERM 5 cat own.txt",
            "nohup cat own.txt",
            "nice -n 5 cat own.txt",
            "stdbuf -oL cat own.txt",
            "command -p cat own.txt",
            "setsid -f cat own.txt",
            "ionice -c 2 -n 7 cat own.txt",
        ],
    )
    def test_supported_wrapper_commands_preserve_nested_command_classification(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        ("variable", "value", "command_template"),
        [
            ("COMMAND", "cat", 'env "$COMMAND" own.txt'),
            ("ENV_ARGS", "UNUSED cat", "env -u $ENV_ARGS {path}"),
            ("TIMEOUT_ARGS", "1 cat", "timeout $TIMEOUT_ARGS {path}"),
        ],
    )
    def test_rejects_dynamic_wrapper_argv_shape(
        self,
        scoped_command_workspace,
        monkeypatch,
        variable,
        value,
        command_template,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        monkeypatch.setenv(variable, value)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    def test_wrapper_nesting_depth_is_bounded(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation, match="wrapper nesting depth exceeded"
        ):
            guard.validate(f"{'env ' * 33}cat own.txt")

    def test_wrapper_nesting_depth_survives_nested_dispatch(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation, match="wrapper nesting depth exceeded"
        ):
            guard.validate(f"{'command xargs ' * 33}cat own.txt")

    def test_env_chdir_applies_only_to_nested_command(self, scoped_command_workspace):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"env --chdir={shlex.quote(str(external_file.parent))} "
            f"cat {shlex.quote(external_file.name)}"
        )

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"env -C {shlex.quote(str(external_file.parent))} "
                f"tee {shlex.quote(external_file.name)}"
            )
        assert exc_info.value.access == "write"

    def test_wrapper_recursion_preserves_find_write_classification(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"find {shlex.quote(str(external_file.parent))} "
                "-type f -exec env cp own.txt {} \\;"
            )
        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("directory_command", ["cd", "pushd"])
    def test_command_wrapper_propagates_shell_builtin_directory_state(
        self, scoped_command_workspace, directory_command
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"command {directory_command} "
                f"{shlex.quote(str(external_file.parent))} && tee leak.txt"
            )
        assert exc_info.value.access == "write"

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

    @pytest.mark.parametrize(
        "command_template",
        [
            'cat "$TARGET"',
            'cat "$(printf %s {path})"',
            "cat `printf %s {path}`",
        ],
    )
    def test_rejects_unresolved_expansion_in_supported_path_operand(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        command = command_template.format(path=shlex.quote(str(sibling_file)))
        if "$TARGET" in command:
            command = f"TARGET={shlex.quote(str(sibling_file))}; {command}"

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_unresolved_expansion_in_redirect_path(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    f"TARGET={shlex.quote(str(sibling_file))}; "
                    'printf changed > "$TARGET"'
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    def test_allows_unresolved_expansion_in_non_path_operand(
        self, scoped_command_workspace, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("needle", encoding="utf-8")
        monkeypatch.setenv("PATTERN", "needle")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": 'grep "$PATTERN" own.txt'})

        assert result["success"] is True
        assert result["output"] == "needle\n"

    def test_tilde_path_is_resolved_as_a_static_path(
        self, scoped_command_workspace, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setenv("HOME", str(workspace.output_dir))
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cat ~/own.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "echo '",
            "for ((i=0;i<1;i++)); do cat sibling.txt; done",
            "coproc cat sibling.txt",
            "select x in a b; do cat sibling.txt; done",
            "cat $'sibling.txt'",
        ],
    )
    def test_unparsed_top_level_shell_input_fails_closed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126

    def test_unparsed_nested_shell_input_fails_closed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": 'sh -c "echo \'"'})

        assert result["success"] is False
        assert result["return_code"] == 126

    @pytest.mark.parametrize(
        "command_template",
        [
            "eval 'cat {path}'",
            "trap 'cat {path}' EXIT",
            "builtin eval 'cat {path}'",
            "command eval 'cat {path}'",
        ],
    )
    def test_rejects_shell_text_reentry_builtins(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "BASH_ENV=hook.sh bash -c true",
            "env BASH_ENV=hook.sh bash -c true",
            "export BASH_ENV=hook.sh; bash -c true",
        ],
    )
    def test_rejects_implicit_shell_initialization_files(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "hook.sh").write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": command_template})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_ambient_shell_initialization_file(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        hook = workspace.output_dir / "hook.sh"
        hook.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("BASH_ENV", str(hook))
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "bash -c true"})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat <<EOF\n$(cat {path})\nEOF",
            "cat <<EOF\n`cat {path}`\nEOF",
            'cat <<<"$(cat {path})"',
        ],
    )
    def test_rejects_shell_execution_from_here_input(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_allows_literal_here_document(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "cat <<EOF\nliteral\nEOF"})

        assert result["success"] is True
        assert result["output"] == "literal\n"

    @pytest.mark.parametrize("command", ["", "   ", "\n", "# comment"])
    def test_guard_allows_noop_shell_input(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["output"] == ""

    def test_guard_rejects_oversized_shell_input(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "printf " + "x" * (64 * 1024)})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "command is too large to inspect" in result["error"]

    def test_guard_rejects_null_byte_input(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": "printf 'before\x00after'"})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "null byte" in result["error"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat <(cat {path})",
            "cat >(tee {path})",
        ],
    )
    def test_process_substitution_uses_nested_command_policy(
        self,
        scoped_command_workspace,
        command_template,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize("absolute", [False, True])
    def test_rejects_direct_shell_script_with_out_of_scope_access(
        self, scoped_command_workspace, absolute
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "deploy.sh"
        script.write_text(
            f"#!/bin/sh\ncat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        command = str(script) if absolute else "./deploy.sh"
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": shlex.quote(command)})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_direct_shell_script_in_literal_argv(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "deploy.sh"
        script.write_text(
            f"#!/bin/sh\ncat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        executor = CommandExecutorCore.for_workspace(
            workspace,
            restrict_paths=True,
        )

        result = executor.execute_command(["./deploy.sh"], shell=False)

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "action",
        [
            "-fprint {path}",
            "-fprint0 {path}",
            "-fls {path}",
            "-fprintf {path} '%p\\n'",
        ],
    )
    def test_rejects_find_output_actions_outside_workspace(
        self,
        scoped_command_workspace,
        action,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"find . -type f {action.format(path=shlex.quote(str(sibling_file)))}"
            )

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "predicate",
        [
            "-newer {path}",
            "-anewer {path}",
            "-cnewer {path}",
            "-samefile {path}",
            "-newermt '2026-01-01'",
        ],
    )
    def test_find_reference_predicates_distinguish_paths_from_timestamps(
        self,
        scoped_command_workspace,
        predicate,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        command = f"find . {predicate.format(path=shlex.quote(str(sibling_file)))}"

        if predicate.startswith("-newermt"):
            guard.validate(command)
        else:
            with pytest.raises(CommandPathViolation) as exc_info:
                guard.validate(command)
            assert exc_info.value.access == "read"

    @pytest.mark.parametrize("marker", ["-ok", "-okdir"])
    def test_find_interactive_exec_actions_use_nested_command_policy(
        self,
        scoped_command_workspace,
        marker,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"find {shlex.quote(str(external_file.parent))} "
                f"-type f {marker} rm {{}} \\;"
            )

        assert exc_info.value.access == "write"

    def test_rejects_find_runtime_root_list(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        root_list = workspace.output_dir / "roots"
        root_list.write_text(str(sibling_file.parent), encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="runtime root list"):
            guard.validate("find -files0-from roots -type f -exec cat {} \\;")

    @pytest.mark.parametrize("global_options", ["-O2", "-D tree"])
    def test_find_global_options_preserve_explicit_roots(
        self,
        scoped_command_workspace,
        global_options,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"find {global_options} {shlex.quote(str(sibling_file.parent))} -type f"
            )

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "printf '%s\\0' ../sibling/secret.txt | xargs -0 cat",
            "xargs -a own.txt cat",
            "xargs cat own.txt",
        ],
    )
    def test_rejects_xargs_runtime_arguments(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="xargs"):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "sh",
            "sh -s",
            "printf 'cat own.txt' | sh",
            "sh < run.sh",
        ],
    )
    def test_rejects_shell_input_that_cannot_be_inspected(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126

    @pytest.mark.parametrize(
        "invocation", ["bash {script}", "sh {script}", ". {script}", "source {script}"]
    )
    def test_rejects_shell_script_that_accesses_sibling(
        self, scoped_command_workspace, invocation
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "read-sibling.sh"
        script.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {"command": invocation.format(script=shlex.quote(str(script)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_dynamically_created_shell_script_before_partial_execution(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = f"cat {shlex.quote(str(sibling_file))}"
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    f"printf '%s\\n' {shlex.quote(script)} > run.sh && bash run.sh"
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert not (workspace.output_dir / "run.sh").exists()

    def test_shell_script_arguments_are_not_treated_as_paths(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "print-arg.sh"
        script.write_text("printf '%s\\n' \"$1\"\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"bash {shlex.quote(str(script))} {shlex.quote(str(sibling_file))}"
        )

    def test_recursive_shell_script_inspection_is_bounded(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "loop.sh"
        script.write_text("source loop.sh\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="depth exceeded"):
            guard.validate("bash loop.sh")

    def test_rejects_unsafe_bash_initialization_file(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "unsafe.rc"
        script.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(f"bash --rcfile {shlex.quote(str(script))} -i -c exit")

    def test_directory_stack_keeps_relative_paths_bound_to_real_cwd(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        subdirectory = workspace.output_dir / "sub"
        subdirectory.mkdir()
        forbidden = workspace.base_dir / "forbidden" / "secret.txt"
        forbidden.parent.mkdir()
        forbidden.write_text("outside workspace", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate("pushd sub && popd && cat ../../forbidden/secret.txt")

        guard.validate("pushd sub && popd && cat own.txt")

    def test_cd_dash_uses_tracked_previous_directory(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cd sub && cd - && cat own.txt")

    def test_rejects_background_directory_state(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("cd sub & cat own.txt")

    @pytest.mark.parametrize("operator", [";", "||"])
    def test_validates_each_possible_directory_state_across_shell_operator(
        self,
        scoped_command_workspace,
        operator,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"cd sub {operator} cat own.txt")

        with pytest.raises(CommandPathViolation):
            guard.validate(f"cd sub {operator} cat ../../forbidden/secret.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "cd sub; ls",
            "cd sub && ls; echo done",
            "cd sub && ls || echo fail",
        ],
    )
    def test_allows_common_directory_chains(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_tracks_conditional_directory_state_at_unconditional_join(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cd missing && true; touch ../outside.txt")
        guard.validate("cd sub && printf reached-only-after-success")

    def test_source_state_propagates_but_child_shell_state_does_not(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        script = workspace.output_dir / "change-directory.sh"
        script.write_text("cd sub\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f". {shlex.quote(str(script))} && cat ../../forbidden/secret.txt"
        )
        with pytest.raises(CommandPathViolation):
            guard.validate(
                f"bash {shlex.quote(str(script))} && cat ../../forbidden/secret.txt"
            )

    def test_guard_internal_error_returns_stable_rejection(
        self, scoped_command_workspace, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        executor = CommandExecutorCore(
            str(workspace.resolve_path("")),
            path_guard=guard,
        )

        def crash(_command):
            raise RecursionError("secret parser detail")

        monkeypatch.setattr(guard, "validate", crash)

        result = executor.execute_command("printf should-not-run")

        assert result == {
            "success": False,
            "output": "",
            "error": "Command rejected by workspace path policy: command validation failed",
            "return_code": 126,
        }

    def test_deeply_nested_shell_returns_stable_rejection(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        command = "(" * 2000 + "true" + ")" * 2000

        result = tool.run_json_sync({"command": command})

        assert result == {
            "success": False,
            "output": "",
            "error": (
                "Command rejected by workspace path policy: "
                "cannot safely parse shell command"
            ),
            "return_code": 126,
        }

    @pytest.mark.parametrize(
        "command_template",
        [
            "sed 'w {path}' own.txt",
            "sed 'w{path}' own.txt",
            "sed 'R{path}' own.txt",
            "sed '/own/r {path}' own.txt",
            "sed '1!w{path}' own.txt",
            "sed '1!!w{path}' own.txt",
            "sed '1! !w{path}' own.txt",
            "sed '1~2w{path}' own.txt",
            "sed '1,+2w{path}' own.txt",
            "sed '1,~2w{path}' own.txt",
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

    @pytest.mark.parametrize(
        "script_template",
        [
            "1{{w {path}\n}}",
            "1{{w{path}\n}}",
            "1{{2{{w {path}\n}}\n}}",
            "1{{p;w {path}\n}}",
            "1{{s/own/changed/w{path}\n}}",
            "/own/{{s/own/changed/w{path}\n}}",
            "1,2!s/own/changed/w{path}",
        ],
    )
    def test_rejects_sed_block_embedded_file_io(
        self, scoped_command_workspace, script_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("own\n", encoding="utf-8")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)
        script = script_template.format(path=sibling_file)

        result = tool.run_json_sync(
            {"command": f"sed {shlex.quote(script)} {shlex.quote(str(own_file))}"}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    @pytest.mark.parametrize(
        "script",
        [
            r"s/a/{w literal/",
            r"/{w literal}/p",
            r"y/{w/xy/",
        ],
    )
    def test_sed_braces_inside_data_are_not_treated_as_block_commands(
        self, scoped_command_workspace, script
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"sed {shlex.quote(script)} own.txt")

    @pytest.mark.parametrize(
        "program",
        [
            'BEGIN { print "changed" > target }',
            'BEGIN { print "changed" > (target) }',
        ],
    )
    def test_rejects_dynamic_awk_output_target(self, scoped_command_workspace, program):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    f"awk -v target={shlex.quote(str(sibling_file))} "
                    f"{shlex.quote(program)}"
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    def test_allows_awk_comparison_without_file_io(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("awk 'BEGIN { print (2 > 1) }'")

    @pytest.mark.parametrize(
        "command",
        [
            'PROGRAM=\'BEGIN { system("cat own.txt") }\' awk "$PROGRAM"',
            "PROGRAM='e cat own.txt' sed -e \"$PROGRAM\" own.txt",
        ],
    )
    def test_rejects_dynamic_sed_and_awk_programs(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

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

    def test_script_inspection_reports_symlink_race_as_policy_error(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        import errno

        workspace, _, _ = scoped_command_workspace
        script_path = workspace.output_dir / "script.sed"
        script_path.write_text("s/a/b/", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        def refuse_symlink(*_args, **_kwargs):
            raise OSError(errno.ELOOP, "symbolic link changed during inspection")

        monkeypatch.setattr(
            "xagent.core.tools.core.command_path_guard.os.open",
            refuse_symlink,
        )

        with pytest.raises(
            CommandPolicyViolation,
            match="symbolic link changed during inspection",
        ) as exc_info:
            guard._inspect_script_file("sed", str(script_path), workspace.output_dir)

        assert not isinstance(exc_info.value, CommandPathViolation)

    @pytest.mark.parametrize(
        "nested_command",
        [
            "cp own.txt {}",
            "cp --target-directory={} own.txt",
            "sort -o {} own.txt",
            "sort -o{} own.txt",
            "sort -ro{} own.txt",
            "sort -T{} own.txt",
            "sort -rT{} own.txt",
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

    def test_unknown_command_path_access_is_explicitly_fail_open(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": (
                    "python3 -c "
                    + shlex.quote(
                        "from pathlib import Path; "
                        f"print(Path({str(sibling_file)!r}).read_text())"
                    )
                )
            }
        )

        assert result["success"] is True
        assert "sibling secret" in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "tac {path}",
            "base64 {path}",
            "base64 -i {path}",
            "tar -tf {path}",
            "tar --list --file={path}",
            "tar -cf archive.tar {path}",
            "dd if={path} of=copy.bin",
        ],
    )
    def test_rejects_added_command_reads_from_sibling(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "base64 -i own.txt -o {path}",
            "tar -cf {path} own.txt",
            "tar --create --file={path} own.txt",
            "tar -xf own.tar -C {parent}",
            "dd if=own.txt of={path}",
        ],
    )
    def test_rejects_added_command_writes_to_sibling(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "own.tar").write_bytes(b"not-an-archive")
        tool = CommandExecutorTool(workspace=workspace, restrict_paths=True)

        result = tool.run_json_sync(
            {
                "command": command_template.format(
                    path=shlex.quote(str(sibling_file)),
                    parent=shlex.quote(str(sibling_file.parent)),
                )
            }
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    @pytest.mark.parametrize(
        "command",
        [
            "tac own.txt",
            "base64 own.txt",
            "base64 -i own.txt -o encoded.txt",
            "tar -cf archive.tar own.txt",
            "tar -tf archive.tar",
            "tar -xf archive.tar -C extracted",
            "dd if=own.txt of=copy.bin",
        ],
    )
    def test_added_command_paths_inside_workspace_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "tac -s / own.txt",
            "tac -s/ own.txt",
            "tac --separator / own.txt",
            "tac --separator=/ own.txt",
            "base64 -w 20 own.txt",
            "base64 --wrap=20 own.txt",
        ],
    )
    def test_added_command_scalar_options_are_not_paths(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            'SEPARATOR=:; tac -s"$SEPARATOR" own.txt',
            'WIDTH=20; base64 -w"$WIDTH" own.txt',
            'PAIR="if=own.txt of=copy.bin"; dd $PAIR',
            'OPTION=farchive.tar; tar -c"$OPTION" own.txt',
        ],
    )
    def test_rejects_dynamic_added_command_grammar(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            'COMMAND=rm; find {root} -type f -exec "$COMMAND" {{}} \\;',
            'COMMAND=rm; printf "%s\\n" own.txt | xargs "$COMMAND"',
        ],
    )
    def test_rejects_dynamic_nested_command_names(
        self, scoped_command_workspace, command_template
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(
                command_template.format(root=shlex.quote(str(external_file.parent)))
            )

    @pytest.mark.parametrize(
        "command_template",
        [
            "base64 -i{path}",
            "base64 --input {path}",
            "base64 --input={path}",
            "base64 -di{path}",
            "base64 -o{path}",
            "base64 --output {path}",
            "base64 --output={path}",
            "base64 -do{path}",
        ],
    )
    def test_rejects_base64_path_option_variants(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    def test_dd_flag_assignments_are_not_paths(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("dd iflag=fullblock oflag=sync")

    @pytest.mark.parametrize(
        "argv_template",
        [
            ["tac", "{path}"],
            ["base64", "-i", "{path}"],
            ["sort", "-o{path}", "own.txt"],
            ["sort", "-ro{path}", "own.txt"],
            ["sort", "-T{path}", "own.txt"],
            ["sort", "-rT{path}", "own.txt"],
            ["tar", "-tf", "{path}"],
            ["dd", "if={path}", "of=copy.bin"],
        ],
    )
    def test_added_command_argv_uses_same_path_policy(
        self, scoped_command_workspace, argv_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        argv = [value.format(path=str(sibling_file)) for value in argv_template]

        with pytest.raises(CommandPathViolation):
            guard.validate_argv(argv)

    @pytest.mark.parametrize(
        "command_template",
        [
            "tar -f{path} -t",
            "tar --file {path} --list",
            "tar tf {path}",
            "tar cf archive.tar {path}",
            "tar --add-file={path} -cf archive.tar",
            "tar -cf archive.tar @{path}",
        ],
    )
    def test_rejects_tar_read_path_variants(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    def test_tar_archive_path_remains_anchored_to_process_cwd(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate("tar -c -C sub -f ../../escape.tar file")

    @pytest.mark.parametrize(
        "command",
        [
            "tar -cf archive.tar -T file-list.txt",
            "tar --create --files-from=file-list.txt -f archive.tar",
            "tar -xPf archive.tar",
            "tar -xf archive.tar --absolute-names",
            "tar -cf archive.tar --checkpoint-action=exec=sh own.txt",
            "tar -cf archive.tar --checkpoint-action exec=sh own.txt",
            "tar -cf archive.tar --to-command=sh own.txt",
            "tar -cf archive.tar -I sh own.txt",
            "tar -cf archive.tar -F hook.sh own.txt",
            "tar -tf host:archive.tar",
        ],
    )
    def test_rejects_tar_indirect_or_executable_path_sources(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_tar_delete_and_compare_keep_distinct_archive_access(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"tar -df {shlex.quote(str(external_file))}")
        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"tar --delete -f {shlex.quote(str(external_file))} member")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "tar -tf own.tar ../../member",
            "tar -xf own.tar ../../member -C extracted",
        ],
    )
    def test_tar_member_selectors_are_not_local_paths(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "nested_command",
        [
            "base64 -o {} own.txt",
            "tar -cf {} own.txt",
            "dd if=own.txt of={}",
            "sed -ni s/external/changed/ {}",
        ],
    )
    def test_added_find_exec_writes_make_root_write_sensitive(
        self, scoped_command_workspace, nested_command
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"find {shlex.quote(str(external_file.parent))} "
                f"-type f -exec {nested_command} \\;"
            )

        assert exc_info.value.access == "write"

    def test_find_execdir_relative_write_makes_root_write_sensitive(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"find {shlex.quote(str(external_file.parent))} "
                "-type f -execdir sh -c 'printf changed > marker' \\;"
            )

        assert exc_info.value.access == "write"

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
