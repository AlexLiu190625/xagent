"""Bash-semantics regressions for the scoped command path guard."""

import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from xagent.core.tools.core import command_path_guard as command_path_guard_module
from xagent.core.tools.core.command_executor import CommandExecutorCore
from xagent.core.tools.core.command_path_guard import WorkspaceCommandPathGuard
from xagent.core.tools.core.command_policy import (
    CommandPathViolation,
    CommandPolicyViolation,
    resolve_trusted_executable,
)
from xagent.core.workspace import TaskWorkspace


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


def _guarded_executor(workspace):
    return CommandExecutorCore(
        str(workspace.resolve_path("")),
        path_guard=WorkspaceCommandPathGuard(workspace),
    )


class _GuardedCommandTool:
    def __init__(self, workspace):
        self._executor = _guarded_executor(workspace)

    def run_json_sync(self, args):
        return self._executor.execute_command(
            args["command"],
            timeout=args.get("timeout"),
        )


def _guarded_tool(workspace):
    return _GuardedCommandTool(workspace)


def _write_comment_script_bytes(path, size):
    prefix = b"#"
    path.write_bytes(prefix + (b"x" * (size - len(prefix))))


@pytest.fixture(scope="module")
def trusted_bash_executable():
    try:
        return resolve_trusted_executable("bash")
    except CommandPolicyViolation:
        pytest.skip("trusted Bash is unavailable")


class TestScopedCommandPathGuardBash:
    """Bash parsing and shell-state policy checks."""

    @pytest.mark.parametrize(
        ("command_name", "access"),
        [("cat", "read"), ("rm", "write")],
    )
    def test_minimal_file_command_set_uses_workspace_authorization(
        self,
        scoped_command_workspace,
        command_name,
        access,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"{command_name} {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == access

    def test_unknown_commands_keep_the_explicit_fail_open_contract(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"python {shlex.quote(str(sibling_file))}")

    def test_recognized_name_workspace_script_is_inspected_before_dispatch(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "cat"
        script.write_text(
            f"#!/usr/bin/env bash\ncat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{workspace.output_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": str(script)})

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_bare_recognized_name_uses_validated_executable_identity(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "cat"
        script.write_text(
            f"#!/usr/bin/env bash\n/bin/cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{workspace.output_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "cat"})

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize(
        "command",
        [
            "PATH=. cat own.txt",
            "env PATH=. cat own.txt",
            "export PATH=.; cat own.txt",
            "hash -p ./cat cat; cat own.txt",
        ],
    )
    def test_rejects_runtime_command_identity_changes(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_rejects_alias_definition_before_shell_execution(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": (f"alias leak='cat {shlex.quote(str(sibling_file))}'\nleak")}
        )

        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]
        assert "sibling secret" not in result["output"]

    def test_allows_literal_quoted_here_document(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": "cat <<'EOF'\n$HOME `printf not-executed`\nEOF\n"}
        )

        assert result["success"] is True
        assert result["output"] == "$HOME `printf not-executed`\n"

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<'EOF'\r\nbody\r\nEOF\r\nprintf AFTER\r\n",
            "cat <<'EOF'\nbody\r\nEOF\r\nprintf AFTER\n",
        ],
    )
    def test_crlf_heredoc_syntax_fails_closed(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely parse shell command",
        ):
            guard.validate(command)

    def test_quoted_heredoc_normalization_does_not_mask_later_commands(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(
                f"printf %s \"<<'EOF'\"\ncat {shlex.quote(str(sibling_file))}\nEOF\n"
            )

    def test_nested_quoted_heredoc_body_is_masked_once(self):
        source = "cat <<'OUTER'\nreport <<'INNER'\nOUTER\nINNER\nprintf live\n"

        normalized = command_path_guard_module._normalize_policy_shell_source(source)

        assert len(normalized) == len(source)
        assert normalized.splitlines()[2] == "OUTER"
        assert normalized.endswith("printf live\n")

    def test_multiple_quoted_heredocs_are_consumed_in_declaration_order(self):
        source = (
            "cat <<'FIRST' <<-'SECOND'\n"
            "first <<'NESTED'\n"
            "FIRST\n"
            "\tsecond time\n"
            "\tSECOND\n"
            "printf live\n"
        )

        normalized = command_path_guard_module._normalize_policy_shell_source(source)

        assert len(normalized) == len(source)
        assert normalized.splitlines()[2] == "FIRST"
        assert normalized.splitlines()[4] == "\tSECOND"
        assert normalized.endswith("printf live\n")

    def test_time_keyword_normalization_ignores_multiline_quoted_literal(self):
        source = 'printf "%s" "literal\ntime cat own.txt"\ntime cat own.txt\n'

        normalized = command_path_guard_module._normalize_policy_shell_source(source)

        assert normalized == (
            'printf "%s" "literal\ntime cat own.txt"\n__t_ cat own.txt\n'
        )

    def test_allows_time_keyword_with_validated_nested_command(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("own", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "time cat own.txt"})

        assert result["success"] is True
        assert result["output"] == "own"

    def test_time_keyword_propagates_builtin_directory_state(
        self,
        scoped_command_workspace,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"time cd {shlex.quote(str(external_file.parent))} "
                "&& printf leaked > leaked.txt"
            )

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("shell_name", ["sh", "dash", "zsh"])
    def test_rejects_shell_dialects_not_owned_by_policy_parser(
        self,
        scoped_command_workspace,
        shell_name,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="shell dialect"):
            guard.validate(f"{shell_name} -c 'cat own.txt'")

    def test_guarded_shell_execution_uses_bash_dialect(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        captured = {}

        def fake_run(command, **kwargs):
            captured.update(kwargs)
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )()

        monkeypatch.setattr(
            "xagent.core.tools.core.command_executor.subprocess.run",
            fake_run,
        )
        executor = _guarded_executor(workspace)

        result = executor.execute_command("printf ok")

        assert result["success"] is True
        assert os.path.basename(captured["executable"]) == "bash"

    def test_unsupported_operator_error_is_actionable(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="unsupported shell operator.*separate command calls",
        ):
            guard.validate("printf one & printf two")

    @pytest.mark.parametrize(
        "command_template",
        [
            "printf changed >> {path}",
            "cat missing 2> {path}",
        ],
    )
    def test_append_and_stderr_redirects_require_write_authorization(
        self,
        scoped_command_workspace,
        command_template,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                command_template.format(path=shlex.quote(str(external_file)))
            )

        assert exc_info.value.access == "write"

    def test_possible_directory_state_growth_is_bounded(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        for index in range(17):
            (workspace.output_dir / f"d{index}").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)
        command = " || ".join(f"cd d{index}" for index in range(17))

        with pytest.raises(CommandPolicyViolation, match="too many possible"):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat {{{path},missing}}",
            "{{cat,printf}} {path}",
            "rm -f {{{path},missing}}",
            "printf changed > {{{path},own.txt}}",
            "bash -c 'cat {{{path},missing}}'",
        ],
    )
    def test_rejects_brace_expansion_before_read_write_or_nested_execution(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)
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
        tool = _guarded_tool(workspace)

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
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": f"cat {operand}"})

        assert result["success"] is True
        assert result["output"] == "literal"

    def test_rejects_active_glob_before_execution(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "cat *.txt"})

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
            "/usr/bin/time -p cat {path}",
        ],
    )
    def test_rejects_supported_wrapper_commands_accessing_sibling(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        tool = _guarded_tool(workspace)

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
            "/usr/bin/time -p cat own.txt",
        ],
    )
    def test_supported_wrapper_commands_preserve_nested_command_classification(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -n cat own.txt",
            "env sudo -n cat own.txt",
            "command sudo -n cat own.txt",
        ],
    )
    def test_rejects_sudo_privilege_wrapper(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect privilege escalation via sudo",
        ):
            guard.validate(command)

    def test_rejects_trusted_absolute_sudo_path_when_available(
        self,
        scoped_command_workspace,
    ):
        sudo = shutil.which("sudo")
        if sudo is None:
            pytest.skip("sudo is unavailable")
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect privilege escalation via sudo",
        ):
            guard.validate(f"{shlex.quote(sudo)} -n cat own.txt")

    def test_rejects_path_shadowed_sudo_before_script_inspection(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        fake_sudo = workspace.output_dir / "sudo"
        fake_sudo.write_text("#!/usr/bin/env bash\n:\n", encoding="utf-8")
        fake_sudo.chmod(0o755)
        monkeypatch.setenv(
            "PATH",
            f"{workspace.output_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect privilege escalation via sudo",
        ):
            guard.validate("sudo -n cat own.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "chroot / cat own.txt",
            "sudo --chroot=/ cat own.txt",
            "sudo --shell",
            "/usr/bin/time --output=timing.txt cat own.txt",
        ],
    )
    def test_rejects_wrapper_modes_that_change_unmodeled_execution_context(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
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
                f"rm -f {shlex.quote(external_file.name)}"
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
                f"{shlex.quote(str(external_file.parent))} "
                "&& printf leak > leak.txt"
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

    @pytest.mark.parametrize("option", ["--rcfile", "--init-file"])
    def test_bash_file_option_consumes_dash_prefixed_argument(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(f"bash {option} -c 'printf safe'")

    @pytest.mark.parametrize("option", ["--rcfile", "--init-file"])
    def test_rejects_attached_bash_long_file_option(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        init_file = workspace.output_dir / "safe.rc"
        init_file.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect bash option",
        ):
            guard.validate(f"bash {option}={shlex.quote(str(init_file))} -c exit")

    @pytest.mark.parametrize("option", ["-o", "+o", "-O", "+O"])
    def test_bash_named_option_consumes_required_value(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        option_value = "posix" if option in {"-o", "+o"} else "extglob"
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"bash {option} {option_value} -c 'printf safe'")

    @pytest.mark.parametrize(
        ("case", "expected_markers"),
        [
            ("minus-o", {"command"}),
            ("plus-o", {"command"}),
            ("minus-O", {"command"}),
            ("plus-O", {"command"}),
            ("short-cluster", {"command"}),
            ("stdin", {"stdin"}),
            ("file-option", {"init", "command"}),
            ("terminator", {"script"}),
            ("command-text", {"command"}),
            ("bare-minus-o", {"listing", "stdin"}),
            ("bare-plus-o", {"listing", "stdin"}),
            ("bare-minus-O", {"listing", "stdin"}),
            ("bare-plus-O", {"listing", "stdin"}),
            ("bare-minus-xo", {"listing", "stdin"}),
            ("bare-plus-xO", {"listing", "stdin"}),
        ],
    )
    def test_trusted_bash_invocation_outcome_matrix(
        self,
        scoped_command_workspace,
        trusted_bash_executable,
        case,
        expected_markers,
    ):
        workspace, _, _ = scoped_command_workspace
        markers = {
            "command": "__COMMAND_MARKER__",
            "init": "__INIT_MARKER__",
            "script": "__SCRIPT_MARKER__",
            "stdin": "__STDIN_MARKER__",
        }
        script = workspace.output_dir / "-policy-script"
        script.write_text(
            f"printf %s {markers['script']}\n",
            encoding="utf-8",
        )
        init_file = workspace.output_dir / "policy.rc"
        init_file.write_text(
            f"printf %s {markers['init']}\n",
            encoding="utf-8",
        )
        named_options = {
            "minus-o": ["-o", "posix"],
            "plus-o": ["+o", "posix"],
            "minus-O": ["-O", "extglob"],
            "plus-O": ["+O", "extglob"],
        }
        if case in named_options:
            arguments = [
                *named_options[case],
                "-c",
                f"printf %s {markers['command']}",
            ]
            stdin = ""
        elif case == "short-cluster":
            arguments = ["-xc", f"printf %s {markers['command']}"]
            stdin = ""
        elif case == "stdin":
            arguments = ["-s"]
            stdin = f"printf %s {markers['stdin']}"
        elif case == "file-option":
            arguments = [
                "--rcfile",
                str(init_file),
                "-i",
                "-c",
                f"printf %s {markers['command']}",
            ]
            stdin = ""
        elif case == "terminator":
            arguments = ["--", script.name]
            stdin = ""
        elif case == "command-text":
            arguments = ["-c", f"printf %s {markers['command']}"]
            stdin = ""
        else:
            arguments = [
                {
                    "bare-minus-o": "-o",
                    "bare-plus-o": "+o",
                    "bare-minus-O": "-O",
                    "bare-plus-O": "+O",
                    "bare-minus-xo": "-xo",
                    "bare-plus-xO": "+xO",
                }[case]
            ]
            stdin = f"printf %s {markers['stdin']}"

        environment = os.environ.copy()
        for name in command_path_guard_module._IMPLICIT_SHELL_ENVIRONMENT:
            environment.pop(name, None)
        completed = subprocess.run(
            [str(trusted_bash_executable), *arguments],
            cwd=workspace.output_dir,
            env=environment,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert completed.returncode == 0
        for outcome, marker in markers.items():
            if outcome in expected_markers:
                assert marker in completed.stdout
            else:
                assert marker not in completed.stdout
        if "listing" in expected_markers:
            assert completed.stdout.index(markers["stdin"]) > 0

        guard = WorkspaceCommandPathGuard(workspace)
        if expected_markers == {"stdin"} or "listing" in expected_markers:
            with pytest.raises(
                CommandPolicyViolation,
                match="without command text or a script",
            ):
                guard.validate_argv(["bash", *arguments])
        else:
            guard.validate_argv(["bash", *arguments])

    @pytest.mark.parametrize("option", ["-o", "+o", "-O", "+O", "-xo", "+xO"])
    def test_bare_bash_named_option_listing_remains_stdin_fed_and_rejected(
        self,
        scoped_command_workspace,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="without command text or a script",
        ):
            guard.validate(f"bash {option}")

    def test_guarded_executor_runs_named_bash_option_value_once(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        executor = _guarded_executor(workspace)

        result = executor.execute_command(
            ["bash", "-o", "posix", "-c", "printf %s __COMMAND_MARKER__"],
            shell=False,
        )

        assert result["return_code"] == 0
        assert result["output"] == "__COMMAND_MARKER__"

    @pytest.mark.parametrize("option", ["-o", "+o", "-O", "+O"])
    def test_guarded_executor_rejects_bare_bash_named_option_as_stdin_fed(
        self,
        scoped_command_workspace,
        caplog,
        option,
    ):
        workspace, _, _ = scoped_command_workspace
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["bash", option], shell=False)

        assert result["return_code"] == 126
        assert result["error"].endswith("command denied by policy")
        assert "without command text or a script" in caplog.text
        assert "missing bash argument" not in caplog.text

    @pytest.mark.parametrize(
        "command",
        [
            "bash --rcfile",
            "bash --init-file",
            "bash -c",
        ],
    )
    def test_rejects_missing_bash_option_argument(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="missing bash argument"):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "bash $BASH_OPTIONS -c 'printf safe'",
            "bash -o $BASH_OPTION -c 'printf safe'",
            "bash --norc $BASH_OPTIONS -c 'printf safe'",
        ],
    )
    def test_rejects_dynamic_bash_option_region(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot safely inspect dynamic bash option region",
        ):
            guard.validate(command)

    @pytest.mark.parametrize("cluster", ["-xc", "-cx", "-sc", "-cs"])
    def test_bash_short_clusters_consume_command_text(
        self,
        scoped_command_workspace,
        cluster,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"bash {cluster} 'printf %s \"$1\"' ignored "
            f"{shlex.quote(str(sibling_file))}"
        )

    @pytest.mark.parametrize("command", ["bash", "bash -s", "bash --", "bash -"])
    def test_rejects_stdin_fed_or_missing_bash_input(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation, match="without command text or a script"
        ):
            guard.validate(command)

    def test_dynamic_bash_command_positionals_are_not_option_region(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("bash -c 'printf safe' ignored \"$DYNAMIC_POSITIONAL\"")

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
        tool = _guarded_tool(workspace)
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
        tool = _guarded_tool(workspace)

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
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": ('bash -c \'printf "%s\\n" "$1"\' _ "$PATTERN"')}
        )

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
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is False
        assert result["return_code"] == 126

    def test_unparsed_nested_shell_input_fails_closed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

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
        tool = _guarded_tool(workspace)

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
        tool = _guarded_tool(workspace)

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
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "printf safe"})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_rejects_ambient_cdpath_that_changes_directory_resolution(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        monkeypatch.setenv("CDPATH", str(sibling_file.parent.parent))
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": f"cd {shlex.quote(sibling_file.parent.name)}; cat secret.txt"}
        )

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
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": command_template.format(path=shlex.quote(str(sibling_file)))}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    def test_allows_literal_here_document(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "cat <<EOF\nliteral\nEOF"})

        assert result["success"] is True
        assert result["output"] == "literal\n"

    @pytest.mark.parametrize("command", ["", "   ", "\n", "# comment"])
    def test_guard_allows_noop_shell_input(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": command})

        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["output"] == ""

    def test_guard_rejects_oversized_shell_input(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "printf " + "x" * (64 * 1024)})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]

    def test_guard_rejects_null_byte_input(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "printf 'before\x00after'"})

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]

    @pytest.mark.parametrize(
        "command_template",
        [
            "cat <(cat {path})",
            "cat >(rm -f {path})",
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
        tool = _guarded_tool(workspace)

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
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["./deploy.sh"], shell=False)

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

    @pytest.mark.parametrize("shebang", ["#!", "#!   "])
    def test_rejects_direct_script_without_shebang_interpreter(
        self,
        scoped_command_workspace,
        shebang,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "malformed-shebang"
        script.write_text(f"{shebang}\nprintf safe\n", encoding="utf-8")
        script.chmod(0o755)
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["./malformed-shebang"], shell=False)

        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]
        assert "command validation failed" not in result["error"]

    @pytest.mark.parametrize("missing_flag", ["O_NONBLOCK", "O_NOFOLLOW"])
    def test_direct_script_inspection_requires_secure_open_flags(
        self,
        scoped_command_workspace,
        monkeypatch,
        missing_flag,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "direct-script"
        script.write_text("#!/usr/bin/env bash\nprintf safe\n", encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.delattr(os, missing_flag)
        executor = _guarded_executor(workspace)

        result = executor.execute_command(["./direct-script"], shell=False)

        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]
        assert "command validation failed" not in result["error"]

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
        tool = _guarded_tool(workspace)

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
        tool = _guarded_tool(workspace)

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
        tool = _guarded_tool(workspace)

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

    @pytest.mark.parametrize("script_name", ["-c", "--rcfile"])
    def test_shell_option_terminator_treats_dash_prefixed_name_as_script(
        self,
        scoped_command_workspace,
        script_name,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / script_name
        script.write_text("printf 'safe script\\n'\n", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {
                "command": (
                    f"bash -- {shlex.quote(script_name)} "
                    f"{shlex.quote(str(sibling_file))}"
                )
            }
        )

        assert result["return_code"] == 0
        assert result["output"] == "safe script\n"

    def test_rejects_unsafe_dash_prefixed_script_after_option_terminator(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        script = workspace.output_dir / "-c"
        script.write_text(
            f"cat {shlex.quote(str(sibling_file))}\n",
            encoding="utf-8",
        )
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": "bash -- -c safe-argument"})

        assert result["return_code"] == 126
        assert "sibling secret" not in result["output"]

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
            "cd sub; cat own.txt",
            "cd sub && cat own.txt; echo done",
            "cd sub && cat own.txt || echo fail",
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

        guard.validate("cd missing && true; rm -f ../outside.txt")
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

    @pytest.mark.parametrize(
        "command_template",
        [
            'COMMAND=rm; printf "%s\\n" own.txt | xargs "$COMMAND"',
        ],
    )
    def test_rejects_dynamic_nested_command_names(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command_template)

    @pytest.mark.parametrize("effect_command", ["rm -f safe.sh", "python -c pass"])
    def test_effect_before_script_rejects_but_script_before_effect_is_allowed(
        self,
        scoped_command_workspace,
        effect_command,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "safe.sh"
        script.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(f"{effect_command}; bash safe.sh")

        guard.validate(f"bash safe.sh; {effect_command}")

    @pytest.mark.parametrize("effect_command", ["rm -f safe.sh", "python -c pass"])
    @pytest.mark.parametrize("script_first", [False, True])
    def test_pipeline_rejects_concurrent_script_effects_in_both_directions(
        self,
        scoped_command_workspace,
        effect_command,
        script_first,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "safe.sh"
        script.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        members = ["bash safe.sh", effect_command]
        if not script_first:
            members.reverse()

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by a concurrent command",
        ):
            guard.validate(" | ".join(members))

    def test_effect_ledger_resets_for_fresh_top_level_validation(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        script = workspace.output_dir / "safe.sh"
        script.write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("python -c pass")
        guard.validate("bash safe.sh")

    def test_parse_attempt_budget_is_shared_across_nested_scripts(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "outer.sh").write_text(
            "bash inner.sh\n",
            encoding="utf-8",
        )
        (workspace.output_dir / "inner.sh").write_text(":\n", encoding="utf-8")
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_PARSE_ATTEMPTS",
            2,
            raising=False,
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy parse attempt budget exceeded",
        ):
            guard.validate("bash outer.sh")

    def test_node_state_evaluation_budget_is_deterministic(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_NODE_STATE_EVALUATIONS",
            1,
            raising=False,
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy node-state evaluation budget exceeded",
        ):
            guard.validate("true; true")

    def test_argv_token_budget_accepts_boundary_and_rejects_exhaustion(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        limit = 8192

        guard.validate_argv(["unknown", *(["value"] * (limit - 1))])
        with pytest.raises(
            CommandPolicyViolation,
            match="command policy argv token budget exceeded",
        ):
            guard.validate_argv(["unknown", *(["value"] * limit)])

    def test_nested_public_reentry_shares_session_and_exception_resets_it(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        original = guard._validate_command_values
        reentered = False

        def reenter(*args, **kwargs):
            nonlocal reentered
            if not reentered:
                reentered = True
                guard.validate_argv(["nested", "value"])
            return original(*args, **kwargs)

        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 3)
        monkeypatch.setattr(guard, "_validate_command_values", reenter)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy argv token budget exceeded",
        ):
            guard.validate_argv(["outer", "value"])

        monkeypatch.setattr(guard, "_validate_command_values", original)
        guard.validate_argv(["fresh", "value", "control"])

    def test_validation_sessions_are_isolated_across_concurrent_contexts(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 1)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(guard.validate_argv, (["one"], ["two"])))

        assert results == [None, None]

    def test_cumulative_source_character_budget_is_shared(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        command = "bash -c ':'"
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_SOURCE_CHARS",
            len(command),
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy source character budget exceeded",
        ):
            guard.validate(command)

    def test_cumulative_script_byte_budget_is_shared(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        script_source = "# comment\n"
        (workspace.output_dir / "one.sh").write_text(script_source, encoding="utf-8")
        (workspace.output_dir / "two.sh").write_text(script_source, encoding="utf-8")
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_SCRIPT_BYTES",
            len(script_source.encode("utf-8")) * 2 - 1,
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy script byte budget exceeded",
        ):
            guard.validate("bash one.sh; bash two.sh")

    @pytest.mark.parametrize("invocation", ["bash {script}", "source {script}"])
    @pytest.mark.parametrize("offset", [-1, 0, 1])
    def test_policy_script_byte_limit_boundary(
        self,
        scoped_command_workspace,
        invocation,
        offset,
    ):
        workspace, _, _ = scoped_command_workspace
        limit = command_path_guard_module._MAX_INSPECTED_SCRIPT_BYTES
        script = workspace.output_dir / f"boundary-{offset}.sh"
        _write_comment_script_bytes(script, limit + offset)
        guard = WorkspaceCommandPathGuard(workspace)
        command = invocation.format(script=shlex.quote(str(script)))

        if offset <= 0:
            guard.validate(command)
        else:
            with pytest.raises(
                CommandPolicyViolation,
                match=rf"shell policy script exceeds the {limit}-byte inspection limit",
            ):
                guard.validate(command)

    @pytest.mark.parametrize("offset", [0, 1])
    def test_policy_script_byte_limit_counts_multibyte_utf8(
        self,
        scoped_command_workspace,
        offset,
    ):
        workspace, _, _ = scoped_command_workspace
        limit = command_path_guard_module._MAX_INSPECTED_SCRIPT_BYTES
        script = workspace.output_dir / f"multibyte-{offset}.sh"
        payload = b"#" + ("é".encode("utf-8") * ((limit - 1) // 2))
        payload += b"x" * (limit + offset - len(payload))
        script.write_bytes(payload)
        guard = WorkspaceCommandPathGuard(workspace)

        if offset == 0:
            guard.validate(f"bash {shlex.quote(str(script))}")
        else:
            with pytest.raises(
                CommandPolicyViolation,
                match=rf"shell policy script exceeds the {limit}-byte inspection limit",
            ):
                guard.validate(f"bash {shlex.quote(str(script))}")

    def test_script_size_rejection_keeps_executor_generic_error(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        limit = command_path_guard_module._MAX_INSPECTED_SCRIPT_BYTES
        script = workspace.output_dir / "oversized.sh"
        _write_comment_script_bytes(script, limit + 1)
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync({"command": f"bash {shlex.quote(str(script))}"})

        assert result["return_code"] == 126
        assert result["error"].endswith("command denied by policy")
        assert "inspection limit" not in result["error"]

    @pytest.mark.parametrize(
        "command",
        [
            "env python -c pass; bash safe.sh",
            "(python -c pass); bash safe.sh",
            "source effect.sh; bash safe.sh",
            "bash -c 'python -c pass'; bash safe.sh",
        ],
    )
    def test_unknown_effect_propagates_through_nested_shell_regions(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text("# safe\n", encoding="utf-8")
        (workspace.output_dir / "effect.sh").write_text(
            "python -c pass\n",
            encoding="utf-8",
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(command)

    @pytest.mark.parametrize("command", ["env true", "xargs true"])
    def test_nested_argv_dispatch_charges_shared_token_budget(
        self,
        scoped_command_workspace,
        monkeypatch,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 2)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy argv token budget exceeded",
        ):
            guard.validate(command)

    def test_deterministic_budgets_allow_ordinary_nested_validation(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_PARSE_ATTEMPTS", 2)
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_SOURCE_CHARS", 32)
        monkeypatch.setattr(
            command_path_guard_module,
            "_MAX_POLICY_NODE_STATE_EVALUATIONS",
            2,
        )
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_ARGV_TOKENS", 4)
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("bash -c ':'")

    def test_comment_only_input_consumes_parse_attempt_budget(
        self,
        scoped_command_workspace,
        monkeypatch,
    ):
        workspace, _, _ = scoped_command_workspace
        monkeypatch.setattr(command_path_guard_module, "_MAX_POLICY_PARSE_ATTEMPTS", 0)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="command policy parse attempt budget exceeded",
        ):
            guard.validate("# comment only\n")

    def test_new_effect_rejection_keeps_executor_exit_126_contract(
        self,
        scoped_command_workspace,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text("# safe\n", encoding="utf-8")
        tool = _guarded_tool(workspace)

        result = tool.run_json_sync(
            {"command": "python -c pass; bash safe.sh"},
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "command denied by policy" in result["error"]


class TestReadCommandFamily:
    """Pure readers newly classified alongside `cat`: I1 (out-of-ws reject) and

    I2 (a value-consuming option's separate argument is skipped, not
    misread as a path, while the real operand is still checked).
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "cmp own.txt {path}",
            "file {path}",
            "head {path}",
            "less {path}",
            "ls {path}",
            "more {path}",
            "stat {path}",
            "tac {path}",
            "tail {path}",
            "wc {path}",
            "cut -f1 {path}",
        ],
    )
    def test_read_family_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "cmp own.txt own.txt",
            "file own.txt",
            "head -n 2 -c 10 own.txt",
            "less own.txt",
            "ls own.txt",
            "more own.txt",
            "stat own.txt",
            "tac -s , own.txt",
            "tail -n 2 -c 10 own.txt",
            "wc own.txt",
            "cut -d , -f 1 own.txt",
        ],
    )
    def test_read_family_workspace_paths_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "head -n {path} own.txt",
            "head -c {path} own.txt",
            "tail -n {path} own.txt",
            "tail -c {path} own.txt",
            "cut -d {path} -f 1 own.txt",
            "tac -s {path} own.txt",
        ],
    )
    def test_value_option_scalar_argument_is_not_treated_as_a_path(
        self, scoped_command_workspace, command_template
    ):
        # A value-consuming option's argument (line count, delimiter, etc.) is
        # never a path, so an out-of-workspace-looking value does not cause a
        # spurious rejection.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize(
        "command_template",
        [
            "head -n 1 {path}",
            "tail -n 1 {path}",
            "cut -d , -f 1 {path}",
            "tac -s , {path}",
        ],
    )
    def test_real_operand_outside_workspace_still_rejected_with_value_option(
        self, scoped_command_workspace, command_template
    ):
        # Skipping the option's own value must not skip the real file operand.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

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
    def test_read_control_file_options_reject_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "file -f control.txt own.txt",
            "file -m control.txt own.txt",
            "file --files-from=control.txt",
            "file --magic-file=control.txt own.txt",
            "wc --files0-from=control.txt",
        ],
    )
    def test_read_control_file_options_workspace_paths_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "control.txt").write_text("c", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "wc -l own.txt",
            "wc -c own.txt",
            "wc -w own.txt",
            "wc -m own.txt",
            "wc -L own.txt",
            "file -b own.txt",
            "file -i own.txt",
            "file -h own.txt",
            "file -L own.txt",
            "file -s own.txt",
            "file -z own.txt",
            "file -k own.txt",
            "file -n own.txt",
            "file -p own.txt",
            "file -r own.txt",
            "file -v own.txt",
            "file -0 own.txt",
        ],
    )
    def test_file_and_wc_common_flags_are_allowed(
        self, scoped_command_workspace, command
    ):
        # R11: the fail-closed short-option flip left `file`/`wc` with an
        # empty `flag_short_options`, rejecting their own ordinary GNU flags.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_file_rejects_unrecognized_short_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("file -@ own.txt")

    def test_wc_rejects_unrecognized_short_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("wc -@ own.txt")

    @pytest.mark.parametrize("command_name", ["file", "wc"])
    def test_file_and_wc_post_terminator_operand_is_still_read_checked(
        self, scoped_command_workspace, command_name
    ):
        # R1: `_partition_path_options` consumes `--` and returns a plain
        # operand list; routing that list back through `_check_operands`/
        # `_operands` re-applied `_operands`'s own `-`-prefix filter with no
        # memory of the `--` it already passed, silently dropping (and never
        # read-checking) a post-`--` operand that happens to start with `-`.
        workspace, _, sibling_file = scoped_command_workspace
        cwd = workspace.resolve_path("")
        relative_escape = os.path.relpath(str(sibling_file), start=str(cwd))
        disguised = f"-x/{relative_escape}"
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"{command_name} -- {shlex.quote(disguised)}")

        assert exc_info.value.access == "read"

    def test_file_post_terminator_workspace_operand_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "-x").write_text("dashed name", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("file -- -x")


class TestSortUniqDiffGrepHandlers:
    """Dedicated handlers for commands that own a write or read-control slot.

    Each owns at least one slot that a flat read/write classification cannot
    express: `sort`'s `-o`/`-T` write options, `uniq`'s second positional
    operand, `diff`'s `--output`, and `grep`'s `-f` pattern file.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "sort -o {path} own.txt",
            "sort -o{path} own.txt",
            "sort -ro{path} own.txt",
            "sort -T{path} own.txt",
            "sort -rT{path} own.txt",
            "sort --output={path} own.txt",
            "sort --temporary-directory={path} own.txt",
            "sort --temp={path} own.txt",
        ],
    )
    def test_sort_write_options_reject_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "sort -o out.txt own.txt",
            "sort -oout.txt own.txt",
            "sort -roout.txt own.txt",
            "sort -Tout.txt own.txt",
            "sort --output=out.txt own.txt",
            "sort --temporary-directory=out.txt own.txt",
            "sort --temp=out.txt own.txt",
            "sort own.txt",
        ],
    )
    def test_sort_write_options_workspace_paths_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "sort --files0-from={path}",
            "sort --files0={path}",
            "sort --random-source={path} own.txt",
            "sort --random-sour={path} own.txt",
        ],
    )
    def test_sort_read_control_options_reject_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

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
        # A dynamic (unexpanded) value is only rejected when it feeds a path
        # check; sort's scalar options never do, so it is simply skipped.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            'sort -o"$TARGET" own.txt',
            'sort -ro"$TARGET" own.txt',
            'sort -T"$TARGET" own.txt',
            'sort --output="$TARGET" own.txt',
            'sort --temporary-directory="$TARGET" own.txt',
        ],
    )
    def test_sort_rejects_dynamic_write_option_values(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "sort --not-a-sort-option own.txt",
            "sort --random-s=/dev/zero own.txt",
            "sort --compress-program=cat own.txt",
            "sort -q own.txt",
        ],
    )
    def test_sort_rejects_unrecognized_or_ambiguous_options(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
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
    def test_sort_allows_recognized_flag_options(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_uniq_second_operand_is_write_checked(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"uniq own.txt {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    def test_uniq_first_operand_out_of_workspace_is_read_checked(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"uniq {shlex.quote(str(sibling_file))} out.txt")

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        ["uniq own.txt", "uniq own.txt out.txt", "uniq -f 1 own.txt out.txt"],
    )
    def test_uniq_workspace_paths_are_allowed(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_uniq_skip_fields_value_is_not_treated_as_a_path(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        # The skip-fields count is not a path, so an out-of-workspace-looking
        # value does not cause a spurious rejection; the real operand does.
        guard.validate(f"uniq -f {shlex.quote(str(sibling_file))} own.txt")

    def test_uniq_rejects_unrecognized_long_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("uniq --not-a-uniq-option own.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "uniq --count own.txt",
            "uniq --repeated own.txt",
            "uniq --all-repeated own.txt",
            "uniq --ignore-case own.txt",
            "uniq --unique own.txt",
            "uniq --zero-terminated own.txt",
            "uniq --group own.txt",
        ],
    )
    def test_uniq_allows_recognized_flag_options(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "uniq -c own.txt",
            "uniq -i own.txt",
            "uniq -u own.txt",
            "uniq -d own.txt",
            "uniq -z own.txt",
        ],
    )
    def test_uniq_allows_recognized_short_flag_options(
        self, scoped_command_workspace, command
    ):
        # R11: the fail-closed short-option flip left `uniq` with an empty
        # `flag_short_options`, rejecting its own ordinary GNU short flags.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_uniq_rejects_unrecognized_short_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("uniq -@ own.txt")

    def test_diff_output_option_registers_write(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"diff --output={shlex.quote(str(sibling_file))} own.txt own.txt"
            )

        assert exc_info.value.access == "write"

    def test_diff_workspace_output_is_allowed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("diff --output=out.txt own.txt own.txt")
        guard.validate("diff -N own.txt own.txt")

    def test_diff_rejects_unrecognized_long_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("diff --not-a-diff-option own.txt own.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "diff --brief own.txt own.txt",
            "diff --unified own.txt own.txt",
            "diff --recursive own.txt own.txt",
            "diff --ignore-case own.txt own.txt",
            "diff --color own.txt own.txt",
            "diff --color=always own.txt own.txt",
            "diff --side-by-side own.txt own.txt",
            "diff --new-file own.txt own.txt",
        ],
    )
    def test_diff_allows_recognized_flag_options(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "diff -u own.txt own.txt",
            "diff -r own.txt own.txt",
            "diff -q own.txt own.txt",
            "diff -i own.txt own.txt",
            "diff -w own.txt own.txt",
            "diff -b own.txt own.txt",
            "diff -B own.txt own.txt",
            "diff -c own.txt own.txt",
            "diff -y own.txt own.txt",
            "diff -a own.txt own.txt",
            "diff -t own.txt own.txt",
            "diff -T own.txt own.txt",
            "diff -p own.txt own.txt",
            "diff -s own.txt own.txt",
            "diff -e own.txt own.txt",
            "diff -n own.txt own.txt",
        ],
    )
    def test_diff_allows_recognized_short_flag_options(
        self, scoped_command_workspace, command
    ):
        # R11: the fail-closed short-option flip left `diff` with only `-N`
        # in `flag_short_options`, rejecting its other ordinary GNU flags.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_diff_rejects_unrecognized_short_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("diff -@ own.txt own.txt")

    @pytest.mark.parametrize(
        "command_template",
        [
            "grep sibling {path}",
            "grep --regexp=sibling {path}",
            "grep -f {path} own.txt",
            "grep -f{path} own.txt",
        ],
    )
    def test_grep_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "grep needle own.txt",
            "grep -e needle own.txt",
            "grep -f pattern.txt own.txt",
            "grep -r needle .",
        ],
    )
    def test_grep_workspace_paths_are_allowed(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "pattern.txt").write_text("needle", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_grep_recursive_directory_operand_is_read_checked(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"grep -r needle {shlex.quote(str(sibling_file.parent))}")

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command_template",
        [
            "grep -if {path} own.txt",
            "grep -vf {path} own.txt",
            "grep -nf {path} own.txt",
        ],
    )
    def test_grep_bundled_pattern_file_option_is_read_checked(
        self, scoped_command_workspace, command_template
    ):
        # `-f`'s pattern-file argument must still be read-checked when it is
        # not the leading character of a bundled short-option cluster.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        ["grep -if pattern.txt own.txt", "grep -vf pattern.txt own.txt"],
    )
    def test_grep_bundled_pattern_file_workspace_path_is_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "pattern.txt").write_text("needle", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "grep --fil={path} own.txt",
            "grep --reg=sibling {path}",
            "grep --exclude-fr={path} own.txt",
        ],
    )
    def test_grep_long_option_abbreviation_still_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        # `--fil=`/`--reg=`/`--exclude-fr=` are GNU unambiguous-prefix
        # abbreviations of `--file`/`--regexp`/`--exclude-from`. An
        # unrecognized long option is skipped whole (this family's own
        # documented pure-read permissiveness), so leaving these
        # unresolved either drops the pattern-file read check entirely or
        # misclassifies the real file operand as the (excluded) pattern
        # positional — a silent read-containment bypass, not merely a
        # missed classification.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "grep --fil=pattern.txt own.txt",
            "grep --reg=needle own.txt",
        ],
    )
    def test_grep_long_option_abbreviation_workspace_paths_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "pattern.txt").write_text("needle", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_grep_rejects_unmodeled_short_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("grep -@ needle own.txt")

    @pytest.mark.parametrize(
        "command_template",
        [
            "grep --label {path} needle own.txt",
            "grep --label={path} needle own.txt",
            "grep --include={path} needle own.txt",
            "grep --context 3 needle own.txt",
        ],
    )
    def test_grep_long_value_option_argument_is_not_treated_as_a_path(
        self, scoped_command_workspace, command_template
    ):
        # R6: `--label`'s (and `--include`'s/`--context`'s) argument is
        # never a filesystem path (a label, glob pattern, or count); an
        # out-of-workspace-looking value must not cause a spurious
        # rejection, and it must not shift the real file operand's
        # classification either.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    def test_grep_label_value_does_not_spuriously_reject_with_explicit_pattern(
        self, scoped_command_workspace
    ):
        # Before `--label` was modeled, an unrecognized long option was
        # skipped whole, leaving its own argument in the token stream; once
        # an explicit `-e` pattern is already present, that stray argument
        # was misclassified as a file operand and spuriously rejected even
        # though grep never reads it from disk.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"grep -e needle --label {shlex.quote(str(sibling_file))} own.txt"
        )

    def test_grep_real_file_operand_after_long_value_option_is_still_read_checked(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"grep --label harmless needle {shlex.quote(str(sibling_file))}"
            )

        assert exc_info.value.access == "read"


def _trust_locally_shadowed_ownership_commands(monkeypatch):
    """Treat `chown` as a trusted system command regardless of its host path.

    macOS ships `chown` under `/usr/sbin`, outside this guard's fixed trusted
    executable roots (`/bin`, `/usr/bin`, `/usr/local/bin`,
    `/opt/homebrew/bin`, owned by `command_policy.py`); most Linux
    distributions ship it under `/usr/bin`, already trusted there. This keeps
    the write-classification tests independent of that host difference
    without touching the trusted-roots list itself.
    """
    original = WorkspaceCommandPathGuard._is_trusted_system_command

    def _is_trusted(command_word: str) -> bool:
        if os.path.basename(command_word) == "chown":
            return True
        return original(command_word)

    monkeypatch.setattr(
        WorkspaceCommandPathGuard,
        "_is_trusted_system_command",
        staticmethod(_is_trusted),
    )


class TestWriteCreateFamily:
    """Write/create commands newly classified alongside `rm`/`mkdir`: I3 (the

    destination registers a write) plus I2 (a non-path positional or option
    value is not misread as a path while the real operand is still checked).
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "chmod 755 {path}",
            "chown owner {path}",
            "chgrp staff {path}",
            "rmdir {path}",
            "tee {path}",
            "touch {path}",
            "truncate -s 10 {path}",
        ],
    )
    def test_write_create_family_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template, monkeypatch
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "chmod 755 own.txt",
            "chown owner own.txt",
            "chgrp staff own.txt",
            "rmdir own_dir",
            "tee own.txt",
            "touch own.txt",
            "truncate -s 10 own.txt",
        ],
    )
    def test_write_create_family_workspace_paths_are_allowed(
        self, scoped_command_workspace, command, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "chmod {path} own.txt",
            "chown {path} own.txt",
            "chgrp {path} own.txt",
            "truncate -s {path} own.txt",
        ],
    )
    def test_write_create_leading_scalar_is_not_treated_as_a_path(
        self, scoped_command_workspace, command_template, monkeypatch
    ):
        # chmod's mode, chown/chgrp's owner/group spec, and truncate's -s size
        # are never paths, so an out-of-workspace-looking value does not
        # cause a spurious rejection.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize(
        "command_template",
        [
            "chmod 755 {path}",
            "chown owner {path}",
            "chgrp staff {path}",
            "truncate -s 100 {path}",
        ],
    )
    def test_write_create_real_operand_outside_workspace_still_rejected(
        self, scoped_command_workspace, command_template, monkeypatch
    ):
        # Skipping the leading scalar must not skip the real path operand.
        workspace, _, sibling_file = scoped_command_workspace
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize(
        "command_template",
        [
            "chmod -X{path} 644 f",
            "chown -X{path} owner f",
            "chgrp -X{path} staff f",
            "touch -X{path} f",
        ],
    )
    def test_ownership_and_touch_reject_unrecognized_short_option(
        self, scoped_command_workspace, command_template, monkeypatch
    ):
        # D1: an unrecognized short option must fail closed instead of being
        # silently skipped — `_partition_path_options` used to skip any
        # short option it didn't model by exact match, so a path embedded in
        # an unmodeled option's own token (e.g. `-X<path>`) went completely
        # unchecked. `f` itself need not exist for this: the invocation must
        # never get far enough to check it.
        workspace, _, sibling_file = scoped_command_workspace
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize(
        "command",
        [
            "chmod -R 644 own.txt",
            "chmod -v 644 own.txt",
            "chmod -c 644 own.txt",
            "chmod -f 644 own.txt",
            "chown -R owner own.txt",
            "chown -H owner own.txt",
            "chown -L owner own.txt",
            "chown -P owner own.txt",
            "chgrp -R staff own.txt",
            "touch -a own.txt",
            "touch -c own.txt",
            "touch -m own.txt",
        ],
    )
    def test_ownership_and_touch_common_short_flags_are_allowed(
        self, scoped_command_workspace, command, monkeypatch
    ):
        # R11: the fail-closed short-option flip left `chmod`/`chown`/
        # `chgrp`/`touch` with an empty (or narrower) `flag_short_options`,
        # rejecting their own ordinary GNU flags.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "truncate -c -s 10 own.txt",
            "truncate -o -s 10 own.txt",
        ],
    )
    def test_truncate_common_short_flags_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_touch_date_option_value_is_not_treated_as_a_path(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        # `-t`/`-d`/`--date`'s value is a timestamp/date string, never a
        # path, so an out-of-workspace-looking value does not cause a
        # spurious rejection; the real operand is still write-checked.
        guard.validate("touch -t 202601010000 own.txt")
        guard.validate('touch -d "yesterday" own.txt')
        guard.validate(f"touch -d {shlex.quote(str(sibling_file))} own.txt")

    def test_touch_date_option_real_operand_still_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"touch -c {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    def test_chmod_reference_file_is_read_checked(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"chmod --reference={shlex.quote(str(sibling_file))} own.txt"
            )

        assert exc_info.value.access == "read"

    def test_chmod_reference_file_workspace_path_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "mode.ref").write_text("", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("chmod --reference=mode.ref own.txt")

    def test_chmod_reference_abbreviation_still_write_checks_the_target(
        self, scoped_command_workspace
    ):
        # `--ref=` is the same option as `--reference=` via GNU unambiguous-
        # prefix abbreviation. The presence check deciding whether the
        # leading operand is the (excluded) MODE positional must use the
        # SAME resolved option set `_partition_path_options` itself uses,
        # not a second raw-string pass: otherwise an abbreviated
        # `--reference` looks absent, the real target gets misread as MODE
        # and stripped from the operand list, and the write check on it is
        # silently skipped entirely.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "mode.ref").write_text("", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"chmod --ref=mode.ref {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    def test_chmod_bare_terminator_does_not_spuriously_match_reference(
        self, scoped_command_workspace
    ):
        # R7: `_resolve_long_option` would previously prefix-match a bare
        # `--` (the operand terminator, not an abbreviation of anything)
        # against the sole `--reference` candidate, so the `has_reference`
        # presence check spuriously believed `--reference` was given. That
        # left the leading MODE positional un-stripped and write-checked as
        # if it were a path, spuriously rejecting an ordinary
        # `chmod -- MODE file` invocation whose MODE happens to resolve
        # outside the workspace.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"chmod -- {shlex.quote(str(sibling_file))} own.txt")

    @pytest.mark.parametrize(
        "command_template",
        ["touch --reference={path} own.txt", "touch -r {path} own.txt"],
    )
    def test_touch_reference_file_is_read_checked(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    def test_touch_reference_file_workspace_path_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "time.ref").write_text("", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("touch --reference=time.ref own.txt")
        guard.validate("touch -r time.ref own.txt")

    def test_touch_reference_external_read_only_file_is_allowed(
        self, scoped_command_workspace
    ):
        # `-r`/`--reference` only reads the reference file's timestamps, so
        # a read-only external directory authorizes it even though the same
        # command's own operand is a write.
        workspace, external_file, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"touch -r {shlex.quote(str(external_file))} own.txt")

    def test_truncate_reference_file_is_read_checked(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"truncate --reference={shlex.quote(str(sibling_file))} own.txt"
            )

        assert exc_info.value.access == "read"

    def test_truncate_reference_file_workspace_path_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "size.ref").write_text("", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("truncate --reference=size.ref own.txt")
        guard.validate("truncate -r size.ref own.txt")


class TestCopyInstallMoveLinkDestructive:
    """Dedicated handlers for cp/install/mv/ln/unlink/shred.

    Each owns a slot a flat write classification cannot express: `cp`/
    `install`'s source-vs-destination split (and `-t/--target-directory`),
    `ln`'s every-operand-write-sensitivity (I4), and `shred`'s scalar/read
    options.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "cp own.txt {path}",
            "install own.txt {path}",
            "mv own.txt {path}",
            "ln own.txt {path}",
            "unlink {path}",
            "shred {path}",
        ],
    )
    def test_copy_move_link_family_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "cp own.txt dest.txt",
            "install own.txt dest.txt",
            "mv own.txt dest.txt",
            "ln own.txt dest.txt",
            "unlink own.txt",
            "shred own.txt",
        ],
    )
    def test_copy_move_link_family_workspace_paths_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_copy_source_outside_workspace_is_read_checked(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"cp {shlex.quote(str(sibling_file))} dest.txt")

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize("command_template", ["cp {path}", "install {path}"])
    def test_single_operand_is_still_read_checked(
        self, scoped_command_workspace, command_template
    ):
        # R9: `len(operands) < 2` used to skip ALL containment for a
        # single-operand invocation; a lone operand is still a real source
        # and must still be read-checked, matching the fix `rsync` already
        # applies to its own single-operand invocation (N1).
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize("command", ["cp own.txt", "install own.txt"])
    def test_single_operand_workspace_path_is_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "cp own.txt -t {path}",
            "install own.txt -t {path}",
        ],
    )
    def test_target_directory_option_outside_workspace_is_rejected(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                command_template.format(path=shlex.quote(str(sibling_file.parent)))
            )

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "cp own.txt -t out",
            "install own.txt -t out",
        ],
    )
    def test_target_directory_option_workspace_path_is_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "out").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_install_directory_mode_marks_every_operand_write(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"install -d {shlex.quote(str(sibling_file.parent / 'x'))}")

        assert exc_info.value.access == "write"

    def test_install_bundled_flag_and_mode_short_options_are_allowed(
        self, scoped_command_workspace
    ):
        # `-Dm755` bundles the no-value `-D` flag with `-m`'s attached mode
        # value; only the trailing character may carry a value, matching the
        # `sort`/`grep` short-option-cluster contract.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("install -Dm755 own.txt dest.txt")

    def test_install_bundled_mode_value_is_not_treated_as_a_path(
        self, scoped_command_workspace
    ):
        # The `-m` mode value is a scalar even when it looks like a path and
        # even when bundled behind `-D`; an out-of-workspace-looking mode
        # value does not cause a spurious rejection.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"install -Dm{shlex.quote(str(sibling_file))} own.txt dest.txt")

    def test_install_bundled_flag_and_mode_real_destination_still_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"install -Dm755 own.txt {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    def test_install_bundled_target_directory_short_option_is_allowed(
        self, scoped_command_workspace
    ):
        # `-Dt` bundles the no-value `-D` flag with `-t`'s target-directory
        # write destination, the same bundling contract `-Dm755` already
        # covers for the scalar `-m` value.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "destdir").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("install -Dt destdir own.txt")

    def test_install_bundled_target_directory_short_option_outside_workspace_is_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"install -Dt {shlex.quote(str(sibling_file.parent))} own.txt"
            )

        assert exc_info.value.access == "write"

    def test_install_target_directory_long_option_abbreviation_is_allowed(
        self, scoped_command_workspace
    ):
        # R9: `_check_install`'s target-directory extraction used to match
        # `--target-directory=` as a literal prefix, so a valid GNU
        # unambiguous-prefix abbreviation like `--target-dir=` fell through
        # to the unrecognized-option fail-closed path instead of resolving
        # like `cp`'s `--targ=` does.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "out").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("install own.txt --target-dir=out")

    def test_install_target_directory_long_option_abbreviation_rejects_out_of_workspace(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"install own.txt --target-dir={shlex.quote(str(sibling_file))}"
            )

        assert exc_info.value.access == "write"

    def test_ln_source_inside_workspace_aliasing_external_read_only_dir_is_rejected(
        self, scoped_command_workspace
    ):
        # I4: a hard link inside the workspace aliases the external source
        # inode, so the source is write-checked, not merely read-checked;
        # the external directory is only approved for read.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"ln {shlex.quote(str(external_file))} alias.txt")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "cp -rL . copied",
            "cp --recursive --dereference . copied",
            "cp -rH . copied",
            "cp --derefe -r . copied",
            "cp -r --derefe . copied",
            "cp --recurs --dereference . copied",
            "cp -r --follow-comm . copied",
            "cp -aL . copied",
            "cp -a -L . copied",
            "cp --archive --dereference . copied",
            "cp --archive -L . copied",
        ],
    )
    def test_copy_recursive_symlink_dereference_is_rejected(
        self, scoped_command_workspace, command
    ):
        # `--derefe`/`--recurs`/`--follow-comm` are GNU unambiguous-prefix
        # abbreviations of `--dereference`/`--recursive`/
        # `--follow-command-line-symlink`; an abbreviated spelling of
        # either flag must be classified identically to the full name so
        # this hard denial still fires. `-a`/`--archive` implies `-r`
        # (GNU cp documents `-a` as `-dR --preserve=all`), so it must be
        # treated as recursive for this denial too, not only the literal
        # `-r`/`-R`/`--recursive` spellings (M1).
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="symbolic links"):
            guard.validate(command)

    def test_copy_dereference_abbreviation_without_recursive_is_allowed(
        self, scoped_command_workspace
    ):
        # The hard denial only fires for the recursive+dereference
        # combination; an abbreviated `--dereference` alone must not be
        # misclassified as triggering it.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cp --derefe own.txt copied.txt")

    def test_copy_archive_alone_without_dereference_is_allowed(
        self, scoped_command_workspace
    ):
        # `-a`/`--archive` implies recursive, but the hard denial only fires
        # for recursive+dereference together; `-a` alone (no `-L`/`-H`/
        # `--dereference`) must not be misclassified as triggering it.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cp -a own.txt copied.txt")
        guard.validate("cp --archive own.txt copied.txt")

    def test_cp_target_directory_bundled_behind_recursive_is_write_checked(
        self, scoped_command_workspace
    ):
        # `-t` bundled behind `-r` (`-rt`) must be recognized as
        # `--target-directory` through the same bundled-cluster contract
        # every other family uses, so the directory argument is
        # write-checked as the actual destination — not misread as an
        # ordinary source operand and merely read-checked (an inverted,
        # under-strict classification of the real write target).
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"cp -rt {shlex.quote(str(sibling_file))} own.txt")

        assert exc_info.value.access == "write"

    def test_cp_target_directory_bundled_behind_recursive_workspace_path_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "out").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cp -rt out own.txt")

    @pytest.mark.parametrize(
        "command_template",
        [
            "cp own.txt --targ={path}",
            "mv own.txt --targ={path}",
            "ln own.txt --targ={path}",
        ],
    )
    def test_target_directory_long_option_abbreviation_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        # `--targ=` is the GNU unambiguous-prefix abbreviation of
        # `--target-directory`.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        ["cp own.txt --targ=out", "mv own.txt --targ=out", "ln own.txt --targ=out"],
    )
    def test_target_directory_long_option_abbreviation_workspace_path_is_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        (workspace.output_dir / "out").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_mv_target_directory_bundled_behind_verbose_is_write_checked(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"mv -vt {shlex.quote(str(sibling_file))} own.txt")

        assert exc_info.value.access == "write"

    def test_shred_scalar_options_are_not_treated_as_paths(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"shred -n {shlex.quote(str(sibling_file))} -s10 own.txt")

    def test_shred_random_source_is_read_checked(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"shred --random-source={shlex.quote(str(sibling_file))} own.txt"
            )

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "shred -f own.txt",
            "shred -u own.txt",
            "shred -v own.txt",
            "shred -x own.txt",
            "shred -z own.txt",
        ],
    )
    def test_shred_common_short_flags_are_allowed(
        self, scoped_command_workspace, command
    ):
        # R11: the fail-closed short-option flip left `shred` with an empty
        # `flag_short_options`, rejecting its own ordinary GNU flags.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_unlink_rejects_unrecognized_short_option(self, scoped_command_workspace):
        # R3: `unlink` used to route through the permissive `_operands`
        # helper (unlike `shred`'s fail-closed `_partition_path_options`),
        # so an unrecognized short option carrying a path (e.g. `-X<path>`)
        # was silently skipped as an ordinary token instead of failing
        # closed — the same class of bypass the fail-closed flip already
        # closed for `shred`.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(f"unlink -X{shlex.quote(str(sibling_file))} own.txt")


class TestPathClassificationSubstrate:
    """Unit tests for the shared option-classification substrate itself.

    Exercised directly (not only through a specific command family) so a
    regression in the abbreviation resolver, the access-map partitioner, or
    the short-option-cluster parser is caught at the substrate boundary.
    """

    def test_resolve_long_option_exact_match(self):
        known = frozenset({"--output", "--other-flag"})

        assert (
            WorkspaceCommandPathGuard._resolve_long_option("--output", known)
            == "--output"
        )

    def test_resolve_long_option_unambiguous_abbreviation(self):
        known = frozenset({"--output", "--other-flag"})

        assert (
            WorkspaceCommandPathGuard._resolve_long_option("--out", known) == "--output"
        )

    def test_resolve_long_option_ambiguous_abbreviation_returns_none(self):
        known = frozenset({"--output", "--outline"})

        assert WorkspaceCommandPathGuard._resolve_long_option("--out", known) is None

    def test_resolve_long_option_unmatched_returns_none(self):
        known = frozenset({"--output"})

        assert (
            WorkspaceCommandPathGuard._resolve_long_option("--nonexistent", known)
            is None
        )

    def test_resolve_long_option_bare_terminator_returns_none(self):
        # R7: every known long option starts with `--`, so a bare `--` (the
        # argument-list terminator, never an abbreviation of anything) would
        # otherwise "unambiguously" prefix-match a known set containing
        # exactly one candidate.
        known = frozenset({"--reference"})

        assert WorkspaceCommandPathGuard._resolve_long_option("--", known) is None

    def test_partition_path_options_drives_the_access_map(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        cwd = guard.execution_cwd

        with command_path_guard_module._validation_session_scope():
            remaining = guard._partition_path_options(
                ["--in", "own.txt", "positional"],
                cwd,
                option_access={"--in": "read"},
            )

        assert remaining == ["positional"]

        with command_path_guard_module._validation_session_scope():
            with pytest.raises(CommandPathViolation) as exc_info:
                guard._partition_path_options(
                    ["--in", str(sibling_file)],
                    cwd,
                    option_access={"--in": "read"},
                )
        assert exc_info.value.access == "read"

    def test_partition_path_options_resolves_abbreviation_to_write(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        cwd = guard.execution_cwd

        with command_path_guard_module._validation_session_scope():
            with pytest.raises(CommandPathViolation) as exc_info:
                guard._partition_path_options(
                    [f"--out={sibling_file}"],
                    cwd,
                    option_access={"--output": "write"},
                )
        assert exc_info.value.access == "write"

        with command_path_guard_module._validation_session_scope():
            remaining = guard._partition_path_options(
                ["--out=out.txt"],
                cwd,
                option_access={"--output": "write"},
            )
        assert remaining == []

    def test_partition_path_options_fails_closed_on_unknown_long_option_for_write_family(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        cwd = guard.execution_cwd

        with command_path_guard_module._validation_session_scope():
            with pytest.raises(CommandPolicyViolation):
                guard._partition_path_options(
                    ["--unmodeled-option", "value"],
                    cwd,
                    option_access={"--output": "write"},
                    fail_closed_on_unknown_long_option=True,
                )

    def test_short_option_cluster_carries_a_path(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        cwd = guard.execution_cwd

        with command_path_guard_module._validation_session_scope():
            next_index, matched_option, argument = guard._parse_short_option_cluster(
                ["-nofile.txt"],
                0,
                cwd,
                flag_options=frozenset({"n"}),
                value_options={"o": "write"},
            )
        assert next_index == 1
        assert matched_option == "o"
        assert argument == "file.txt"

        with command_path_guard_module._validation_session_scope():
            with pytest.raises(CommandPathViolation) as exc_info:
                guard._parse_short_option_cluster(
                    [f"-no{sibling_file}"],
                    0,
                    cwd,
                    flag_options=frozenset({"n"}),
                    value_options={"o": "write"},
                )
        assert exc_info.value.access == "write"

    def test_short_option_cluster_fails_closed_on_unknown_character(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        cwd = guard.execution_cwd

        with command_path_guard_module._validation_session_scope():
            with pytest.raises(CommandPolicyViolation):
                guard._parse_short_option_cluster(
                    ["-z"],
                    0,
                    cwd,
                    flag_options=frozenset({"n"}),
                    value_options={"o": "write"},
                )


class TestCommandCoverage:
    """Mutation-sensitive coverage gate for every classified command.

    `_ALL_COMMANDS` is derived from the guard's OWN live classification sets
    (`_READ_COMMANDS`, `_WRITE_COMMANDS`, `_DEDICATED_HANDLER_COMMANDS`),
    read through the module reference (not a top-level `from ... import`,
    which would bind an independent copy and stop reflecting a later
    monkeypatch) so a command added to any of them without a registry entry
    fails `test_registry_covers_every_classified_command` instead of being
    silently untested. `test_registry_coverage_is_mutation_sensitive_to_live_classification`
    proves that sensitivity directly.
    """

    # Maps each dedicated-handler command to the guard method that classifies
    # it; a method may serve more than one command name (`mv`/`ln`,
    # `unlink`/`shred`). The key set must equal the live
    # `_DEDICATED_HANDLER_COMMANDS` (checked below), not be hand-duplicated
    # against it.
    _DEDICATED_HANDLER_METHODS: dict[str, str] = {
        "sort": "_check_sort",
        "uniq": "_check_uniq",
        "diff": "_check_diff",
        "grep": "_check_grep",
        "cp": "_check_copy",
        "install": "_check_install",
        "mv": "_check_move_or_link",
        "ln": "_check_move_or_link",
        "unlink": "_check_destructive_file_command",
        "shred": "_check_destructive_file_command",
        "find": "_check_find",
        "tar": "_check_tar",
        "sed": "_check_script_command",
        "awk": "_check_script_command",
        "dd": "_check_dd",
        "base64": "_check_base64",
        "gzip": "_check_gzip",
        "rsync": "_check_rsync",
        "curl": "_check_curl",
        "wget": "_check_wget",
    }

    # Each entry: (positive command, negative command template with
    # `{outside}`, expected `CommandPathViolation.access` the negative
    # template's out-of-workspace target is checked as). The expected access
    # is asserted by both negative-case tests below so a write silently
    # downgraded to a read (or vice versa) fails the gate mechanically
    # instead of only being caught by exception type.
    _REGISTRY: dict[str, tuple[str, str, str]] = {
        "cat": ("cat own.txt", "cat {outside}", "read"),
        "cmp": ("cmp own.txt own.txt", "cmp own.txt {outside}", "read"),
        "file": ("file own.txt", "file {outside}", "read"),
        "head": ("head -n 1 own.txt", "head -n 1 {outside}", "read"),
        "less": ("less own.txt", "less {outside}", "read"),
        "ls": ("ls own.txt", "ls {outside}", "read"),
        "more": ("more own.txt", "more {outside}", "read"),
        "stat": ("stat own.txt", "stat {outside}", "read"),
        "tac": ("tac -s , own.txt", "tac -s , {outside}", "read"),
        "tail": ("tail -n 1 own.txt", "tail -n 1 {outside}", "read"),
        "wc": ("wc own.txt", "wc {outside}", "read"),
        "cut": ("cut -d , -f 1 own.txt", "cut -d , -f 1 {outside}", "read"),
        "sort": ("sort own.txt", "sort {outside}", "read"),
        "uniq": ("uniq own.txt", "uniq {outside}", "read"),
        "diff": ("diff own.txt own.txt", "diff own.txt {outside}", "read"),
        "grep": ("grep pattern own.txt", "grep pattern {outside}", "read"),
        "chmod": ("chmod 755 own.txt", "chmod 755 {outside}", "write"),
        "chown": ("chown owner own.txt", "chown owner {outside}", "write"),
        "chgrp": ("chgrp staff own.txt", "chgrp staff {outside}", "write"),
        "mkdir": ("mkdir own_new_dir", "mkdir {outside}", "write"),
        "rm": ("rm own.txt", "rm {outside}", "write"),
        "rmdir": ("rmdir own_dir", "rmdir {outside}", "write"),
        "tee": ("tee own.txt", "tee {outside}", "write"),
        "touch": ("touch own.txt", "touch {outside}", "write"),
        "truncate": ("truncate -s 1 own.txt", "truncate -s 1 {outside}", "write"),
        "cp": ("cp own.txt dest.txt", "cp own.txt {outside}", "write"),
        "install": (
            "install own.txt dest.txt",
            "install own.txt {outside}",
            "write",
        ),
        "mv": ("mv own.txt dest.txt", "mv own.txt {outside}", "write"),
        "ln": ("ln own.txt dest.txt", "ln own.txt {outside}", "write"),
        "unlink": ("unlink own.txt", "unlink {outside}", "write"),
        "shred": ("shred own.txt", "shred {outside}", "write"),
        "find": ("find own.txt -print", "find {outside}", "read"),
        "tar": (
            "tar -cf archive.tar own.txt",
            "tar -cf archive.tar {outside}",
            "read",
        ),
        "sed": ("sed 's/a/b/' own.txt", "sed 'w {outside}' own.txt", "write"),
        "awk": (
            "awk '{print $0}' own.txt",
            "awk '{{print $0 > \"{outside}\"}}' own.txt",
            "write",
        ),
        "dd": ("dd if=own.txt of=dest.bin", "dd if=own.txt of={outside}", "write"),
        "base64": (
            "base64 -i own.txt -o own.b64",
            "base64 -i own.txt -o {outside}",
            "write",
        ),
        "gzip": ("gzip own.txt", "gzip {outside}", "write"),
        "rsync": ("rsync own.txt dest.txt", "rsync own.txt {outside}", "write"),
        "curl": (
            "curl -o out.bin https://example.invalid/file",
            "curl -o {outside} https://example.invalid/file",
            "write",
        ),
        "wget": (
            "wget -O out.bin https://example.invalid/file",
            "wget -O {outside} https://example.invalid/file",
            "write",
        ),
    }

    # Registry entries whose negative case exercises a write-owning slot;
    # these are also probed against a read-ALLOWED external directory (not
    # only a fully-out-of-workspace sibling) so a write silently downgraded
    # to read cannot hide behind a target that both classifications would
    # reject anyway.
    _WRITE_ACCESS_COMMANDS = frozenset(
        command_name
        for command_name, (_, _, access) in _REGISTRY.items()
        if access == "write"
    )

    @staticmethod
    def _all_commands() -> frozenset:
        """Recompute the live-classified command set from the module.

        Re-reads the module's sets on every call (not a cached class
        constant) so a test can mutate them first (via `monkeypatch`) and
        observe the effect, proving the coverage gate actually depends on
        live classification rather than a frozen snapshot taken at import.
        """
        return (
            frozenset(command_path_guard_module._READ_COMMANDS)
            | frozenset(command_path_guard_module._WRITE_COMMANDS)
            | frozenset(command_path_guard_module._DEDICATED_HANDLER_COMMANDS)
        )

    # A fixed snapshot for parametrization, which pytest evaluates at
    # collection time; the mutation-sensitivity test below re-derives the
    # live set independently rather than relying on this constant.
    _ALL_COMMANDS = _all_commands()

    def test_registry_covers_every_classified_command(self):
        assert set(self._REGISTRY) == self._all_commands()

    def test_dedicated_handler_registry_matches_live_classification(self):
        assert (
            set(self._DEDICATED_HANDLER_METHODS)
            == command_path_guard_module._DEDICATED_HANDLER_COMMANDS
        )

    def test_dedicated_handlers_are_not_flat_read_or_write(self):
        for command_name, method_name in self._DEDICATED_HANDLER_METHODS.items():
            assert command_name not in command_path_guard_module._READ_COMMANDS
            assert command_name not in command_path_guard_module._WRITE_COMMANDS
            assert hasattr(WorkspaceCommandPathGuard, method_name)

    def test_every_command_is_shadow_script_classified(self):
        for command_name in self._all_commands():
            assert (
                command_name
                in command_path_guard_module._CLASSIFIED_EXECUTABLE_COMMANDS
            )

    def test_registry_coverage_is_mutation_sensitive_to_live_classification(
        self, monkeypatch
    ):
        # Prove the coverage gate actually depends on the live module sets:
        # injecting a fake command into `_READ_COMMANDS` must desynchronize
        # it from the (unchanged) registry, which is exactly the condition
        # `test_registry_covers_every_classified_command` asserts does not
        # hold. If this stayed equal, that test would stay silently green
        # after a real command were added without a registry entry.
        monkeypatch.setattr(
            command_path_guard_module,
            "_READ_COMMANDS",
            command_path_guard_module._READ_COMMANDS | {"__not_a_real_command__"},
        )

        assert set(self._REGISTRY) != self._all_commands()

    @pytest.mark.parametrize("command_name", sorted(_ALL_COMMANDS))
    def test_registry_positive_case_is_accepted(
        self, scoped_command_workspace, command_name, monkeypatch
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("data\n", encoding="utf-8")
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)
        positive_command, _, _ = self._REGISTRY[command_name]

        guard.validate(positive_command)

    @pytest.mark.parametrize("command_name", sorted(_ALL_COMMANDS))
    def test_registry_negative_case_is_rejected(
        self, scoped_command_workspace, command_name, monkeypatch
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("data\n", encoding="utf-8")
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)
        _, negative_template, expected_access = self._REGISTRY[command_name]

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                negative_template.format(outside=shlex.quote(str(sibling_file)))
            )

        # A write silently downgraded to a read (or vice versa) must not
        # pass this gate merely because SOME violation was raised.
        assert exc_info.value.access == expected_access

    @pytest.mark.parametrize("command_name", sorted(_WRITE_ACCESS_COMMANDS))
    def test_registry_negative_case_rejects_read_allowed_external_dir_as_write(
        self, scoped_command_workspace, command_name, monkeypatch
    ):
        # A write-owning option's target must be rejected even when pointed
        # at a directory the workspace approves for READ (`external_file`),
        # not only a fully-out-of-workspace sibling both read and write
        # already reject alike. This is the read/write-confusion case a
        # target that is universally denied can never exercise: a write
        # silently downgraded to a read would otherwise still raise here
        # (external reads are allowed) and the gate would stay green.
        workspace, external_file, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("data\n", encoding="utf-8")
        _trust_locally_shadowed_ownership_commands(monkeypatch)
        guard = WorkspaceCommandPathGuard(workspace)
        _, negative_template, _ = self._REGISTRY[command_name]

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                negative_template.format(outside=shlex.quote(str(external_file)))
            )

        assert exc_info.value.access == "write"


class TestCdSymlinkTraversal:
    """`cd`/`pushd` logical-vs-physical divergence must not escape the workspace.

    Bash's ``cd`` (without ``-P``) is logical: it collapses ``..`` textually
    against the pre-symlink path. The guard resolves physically. When a target
    crosses a symlink and carries enough ``..`` to round-trip past its real
    depth, the two disagree and a fully-classified ``cd`` + ``cat``/``rm``
    sequence could reach a sibling tenant's files. The guard fails closed on any
    directory change that traverses a symlink.
    """

    def test_cd_through_symlink_with_dotdot_is_rejected(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "a" / "b" / "c" / "d" / "e" / "f").mkdir(parents=True)
        (ws / "s").symlink_to(ws / "a" / "b" / "c" / "d" / "e" / "f")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="traverses a symlink"):
            guard.validate("cd s/../../../../../..")

    def test_pushd_through_symlink_is_rejected(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "deep" / "nested").mkdir(parents=True)
        (ws / "link").symlink_to(ws / "deep" / "nested")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="traverses a symlink"):
            guard.validate("pushd link/../..")

    def test_symlink_free_cd_still_resolves(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "sub" / "inner").mkdir(parents=True)
        guard = WorkspaceCommandPathGuard(workspace)

        # Real directories with a textual ``..`` cross no symlink and stay in
        # the workspace, so navigation and a subsequent read are allowed.
        guard.validate("cd sub/inner/.. && cat placeholder.txt")

    def test_guarded_executor_blocks_cd_symlink_escape(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "a" / "b" / "c" / "d" / "e" / "f").mkdir(parents=True)
        (ws / "s").symlink_to(ws / "a" / "b" / "c" / "d" / "e" / "f")
        tool = _guarded_tool(workspace)
        rel_secret = os.path.relpath(sibling_file, ws)

        result = tool.run_json_sync(
            {"command": f"cd s/../../../../../.. && cat {rel_secret}"}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert "sibling secret" not in (result.get("output") or "")
        assert sibling_file.read_text(encoding="utf-8") == "sibling secret"

    def test_guarded_executor_blocks_cd_symlink_escape_rm_variant(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "a" / "b" / "c" / "d" / "e" / "f").mkdir(parents=True)
        (ws / "s").symlink_to(ws / "a" / "b" / "c" / "d" / "e" / "f")
        tool = _guarded_tool(workspace)
        rel_secret = os.path.relpath(sibling_file, ws)

        result = tool.run_json_sync(
            {"command": f"cd s/../../../../../.. && rm {rel_secret}"}
        )

        assert result["success"] is False
        assert result["return_code"] == 126
        assert sibling_file.exists()

    def test_quoted_brace_literal_is_checked_as_a_file_path(
        self, scoped_command_workspace
    ):
        # ``{}`` is no longer a cwd sentinel: a quoted literal is authorized as
        # the file itself and still resolves inside the workspace.
        workspace, _, _ = scoped_command_workspace
        ws = Path(workspace.resolve_path(""))
        (ws / "{}").write_text("brace", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("cat '{}'")


class TestNoEffectCommandClassification:
    """Commands with no filesystem write effect must not poison script inspection.

    The unknown-effect flag stays session-wide (an *unknown* command has an
    unknowable target set), but commands that cannot write the filesystem, and
    path-scoped writers whose writes are recorded per-path, no longer trip it.
    """

    @pytest.mark.parametrize(
        "setup_command",
        [
            "echo hi",
            "printf hi",
            "pwd",
            "true",
            "test -f safe.sh",
            "export FOO=bar",
            "declare BAR=baz",
        ],
    )
    def test_no_effect_command_before_script_is_allowed(
        self, scoped_command_workspace, setup_command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"{setup_command} && bash safe.sh")

    def test_mkdir_before_script_is_allowed_and_scopes_write(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        # mkdir writes only its operand (recorded per-path), so a later,
        # unrelated script inspection is not blocked.
        guard.validate("mkdir out && bash safe.sh")

    def test_mkdir_outside_workspace_is_rejected(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(f"mkdir {shlex.quote(str(sibling_file.parent / 'x'))}")

    @pytest.mark.parametrize(
        "mkdir_command",
        [
            "mkdir -m ../../../../../../etc newdir",
            "mkdir --mode ../../../../../../etc newdir",
            "mkdir -m0755 newdir",
            "mkdir --mode=0755 newdir",
        ],
    )
    def test_mkdir_mode_value_is_not_treated_as_a_path(
        self, scoped_command_workspace, mkdir_command
    ):
        # The `-m`/`--mode` value is consumed as the mode, not validated as a
        # path operand, so an out-of-workspace-looking mode token does not cause
        # a spurious rejection. Only the real directory operand is checked.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(mkdir_command)

    def test_mkdir_real_directory_outside_workspace_still_rejected_with_mode(
        self, scoped_command_workspace
    ):
        # Skipping the mode value must not skip the real path operand.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        target = shlex.quote(str(sibling_file.parent / "x"))
        with pytest.raises(CommandPathViolation):
            guard.validate(f"mkdir -m 0755 {target}")

    def test_redirect_write_from_no_effect_command_still_blocks_inspection(
        self, scoped_command_workspace
    ):
        # A no-effect command with a redirect still registers the redirect's
        # write, so inspecting that script afterwards is rejected. Declassifying
        # the command does not exempt its redirections.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("echo hi > safe.sh; bash safe.sh")

    def test_unknown_command_still_poisons_script_inspection(
        self, scoped_command_workspace
    ):
        # Genuinely unknown commands keep the session-wide fail-closed contract.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("python -c pass && bash safe.sh")


class TestFindCommand:
    """`find`: frozen single-pass, observer-scoped `-exec`/`-execdir` algorithm.

    The searched root's read/write classification comes from a clause-scoped
    observer that inspects every raw operand of an exec/execdir/ok/okdir
    clause's real enforcement pass, not from a second throwaway pass and not
    from inspecting `written_paths` after the fact.
    """

    def test_exec_write_via_placeholder_marks_root_write(
        self, scoped_command_workspace
    ):
        # The observer fires on the raw "{}" operand before any short-circuit
        # and marks the searched root write-sensitive; the external directory
        # is only approved for read, so the write-checked root is rejected.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        external_dir = shlex.quote(str(external_file.parent))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate("find {ext} -exec rm {{}} \\;".format(ext=external_dir))

        assert exc_info.value.access == "write"

    def test_exec_placeholder_is_arity_preserving_destination_still_checked(
        self, scoped_command_workspace
    ):
        # "{}" stays a real operand (never pre-stripped or substituted before
        # dispatch), so `cp`'s source/destination split is unaffected and the
        # fixed destination is still write-checked by the same enforcement
        # pass that classified "{}".
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("data", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate("find . -exec cp {{}} {out} \\;".format(out=outside))

        assert exc_info.value.access == "write"

    def test_exec_write_root_poisons_later_script_inspection(
        self, scoped_command_workspace
    ):
        # A write-root find sets unknown_effect: find's per-match children are
        # not exact-registerable, so the poison must be session-wide.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "build.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("find . -exec rm {} \\; ; bash build.sh")

    def test_exec_then_execdir_write_or_aggregates_over_all_clauses(
        self, scoped_command_workspace
    ):
        # writes_root is an OR over every clause: a read clause first must not
        # shadow a write clause that follows it.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        external_dir = shlex.quote(str(external_file.parent))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                "find {ext} -exec cat {{}} \\; -execdir rm {{}} \\;".format(
                    ext=external_dir
                )
            )

        assert exc_info.value.access == "write"

    def test_execdir_then_exec_write_or_aggregates_regardless_of_order(
        self, scoped_command_workspace
    ):
        # Mutation pin for the same invariant in the opposite clause order: a
        # last-clause-wins bug (assignment instead of OR-accumulate) would let
        # the later read clause erase the earlier write classification.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        external_dir = shlex.quote(str(external_file.parent))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                "find {ext} -execdir rm {{}} \\; -exec cat {{}} \\;".format(
                    ext=external_dir
                )
            )

        assert exc_info.value.access == "write"

    def test_fprintf_out_of_workspace_destination_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate("find . -fprintf {out} '%p'".format(out=outside))

        assert exc_info.value.access == "write"

    def test_fprintf_in_workspace_registers_on_the_real_effects(
        self, scoped_command_workspace
    ):
        # Fixed path events are checked on the real session effects, not a
        # throwaway classification pass, so the write is visible to a later
        # script inspection in the same command chain.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "report.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("find . -fprintf report.sh '%p' && bash report.sh")

    def test_files0_from_fails_closed(self, scoped_command_workspace):
        # A runtime root list cannot be inspected statically.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("find -files0-from roots.txt")

    @pytest.mark.parametrize("global_option", ["-O2", "-D tree"])
    def test_global_options_do_not_swallow_the_following_root(
        self, scoped_command_workspace, global_option
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"find {global_option} {outside}")

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize("global_option", ["-O2", "-D tree"])
    def test_global_options_root_inside_workspace_is_allowed(
        self, scoped_command_workspace, global_option
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own_dir").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"find {global_option} own_dir")

    def test_newermt_timestamp_argument_is_not_treated_as_a_path(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("find . -newermt '2026-01-01'")

    def test_samefile_reference_argument_is_read_checked(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate("find . -samefile {out}".format(out=outside))

        assert exc_info.value.access == "read"

    def test_delete_marks_root_write(self, scoped_command_workspace):
        # `-delete` also participates in the writes_root OR, not only
        # exec/execdir/ok/okdir clauses.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        external_dir = shlex.quote(str(external_file.parent))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate("find {ext} -delete".format(ext=external_dir))

        assert exc_info.value.access == "write"

    def test_nested_find_sharing_the_outer_terminator_fails_closed(
        self, scoped_command_workspace
    ):
        # `-exec`/`-execdir` consume tokens up to the first bare `;`/`+`
        # regardless of nesting (matching find's own runtime parser), so an
        # inner find embedded in an outer clause loses its own terminator to
        # the outer's. The inner find then raises for its own unterminated
        # `-execdir`, which is a sound fail-closed outcome, not a corrupted
        # outer classification.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "a").mkdir()
        (workspace.output_dir / "a" / "b").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("find a -exec find b -execdir rm {} \\; \\;")

    def test_nested_find_without_further_exec_nesting_leaves_outer_intact(
        self, scoped_command_workspace
    ):
        # A cleanly-terminated nested find (no further -exec inside it) is a
        # positive control: the inner find's own write classification (via
        # `-delete`) must reach the real session effects exactly as an outer
        # command's would, and the outer find's own clause-loop bookkeeping
        # must not be corrupted by the recursion (the outer clause here is a
        # plain read of the inner find's stdout, never itself write-flagged).
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "a").mkdir()
        (workspace.output_dir / "b").mkdir()
        (workspace.output_dir / "build.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("find a -exec find b -delete \\; ; bash build.sh")

    def test_positive_find_inside_workspace_with_exec_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("data", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("find . -exec cat {} \\;")


class TestTarCommand:
    """`tar`: archive/control paths anchored to cwd; `-C` scopes only members;

    extract poisons `unknown_effect` (M2 additive) in addition to a real
    containment check on its extraction root; remote/stdin archives and
    executable hooks fail closed.
    """

    def test_extract_change_directory_outside_workspace_rejected(
        self, scoped_command_workspace
    ):
        # M2 containment: `-C`'s extraction root is still write-checked even
        # though extract also poisons `unknown_effect` for its members.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(external_file.parent))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"tar -x -C {outside} -f a.tar")

        assert exc_info.value.access == "write"

    def test_extract_write_root_poisons_later_script_inspection(
        self, scoped_command_workspace
    ):
        # Extraction targets are non-enumerable per-match children, so the
        # `-C` write-check alone cannot capture them; `unknown_effect` must
        # additionally poison later script inspection in the same chain.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own_dir").mkdir()
        (workspace.output_dir / "own_dir" / "x.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("tar -x -C own_dir -f a.tar ; bash own_dir/x.sh")

    def test_create_archive_write_outside_workspace_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own_dir").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)
        outside_archive = shlex.quote(f"{sibling_file}.tar")

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"tar -c -f {outside_archive} own_dir")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "tar -xf host:a.tar",
            "tar --file=host:a.tar --list",
            "tar -xf user@host:a.tar",
        ],
    )
    def test_remote_archive_fails_closed(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "tar -xf -",
            "tar --file - --list",
        ],
    )
    def test_stdin_archive_fails_closed(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "tar --to-command=sh -xf a.tar",
            "tar -I sh -cf archive.tar own.txt",
            "tar --use-compress-program=sh -xf a.tar",
        ],
    )
    def test_executable_hooks_rejected(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("data", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "tar -x --absolute-names -f a.tar",
            "tar -xPf a.tar",
        ],
    )
    def test_rejects_absolute_extraction(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_list_with_change_directory_inside_workspace_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own_dir").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("tar -C own_dir -tf a.tar")

    def test_change_directory_scope_applies_to_extract_destination(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own_dir").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("tar -xzf a.tgz -C own_dir")

    def test_change_directory_scope_does_not_reanchor_the_archive_path(
        self, scoped_command_workspace
    ):
        # `-C sub` shifts member resolution to `own_dir/sub`, but the archive
        # path is anchored to the unshifted process cwd: from `sub`, "../.."
        # would still land inside the workspace, but from the real cwd it
        # escapes. A wrong implementation that resolves the archive against
        # `-C`'s directory would let this pass instead of rejecting it.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "sub").mkdir()
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate("tar -c -C sub -f ../../escape.tar file")

    def test_delete_and_compare_keep_distinct_archive_access(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(external_file))

        # Compare only reads the archive, so an approved external read-only
        # path is fine.
        guard.validate(f"tar -df {outside}")
        # Delete rewrites the archive in place, so the same path must now be
        # rejected as a write, not silently reused as a read.
        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"tar --delete -f {outside} member")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "tar -tf own.tar ../../member",
            "tar -xf own.tar ../../member -C extracted",
        ],
    )
    def test_member_selectors_are_not_local_paths(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

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
    def test_rejects_read_path_variants(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    @pytest.mark.parametrize(
        "command",
        [
            "tar -cf archive.tar -T file-list.txt",
            "tar --create --files-from=file-list.txt -f archive.tar",
            "tar --create --files-fr=own.txt -f archive.tar",
            "tar --create --files-fr own.txt -f archive.tar",
            "tar -cf archive.tar --checkpoint-action=exec=sh own.txt",
            "tar -cf archive.tar --checkpoint-action exec=sh own.txt",
            "tar -cf archive.tar -F hook.sh own.txt",
        ],
    )
    def test_rejects_indirect_or_executable_path_sources(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("data", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        "option_spelling",
        ["--listed-incr", "--listed-incremental"],
    )
    @pytest.mark.parametrize("attach_form", ["=", " "])
    def test_create_mode_listed_incremental_write_checks_out_of_workspace_path(
        self, scoped_command_workspace, option_spelling, attach_form
    ):
        # `--listed-incr` is the GNU unambiguous-prefix abbreviation of
        # `--listed-incremental`; both spellings, in both the `=`-attached
        # and space-separated forms, must resolve to the same
        # write-checked snapshot-file option, not fall through as an
        # unchecked (read-checked) source positional.
        workspace, external_file, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(external_file))
        option = (
            f"{option_spelling}={outside}"
            if attach_form == "="
            else f"{option_spelling} {outside}"
        )

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"tar --create {option} -f a.tar own.txt")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "option_spelling",
        ["--listed-incr", "--listed-incremental"],
    )
    def test_create_mode_listed_incremental_workspace_path_is_allowed(
        self, scoped_command_workspace, option_spelling
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"tar --create {option_spelling}=snapshot -f a.tar own.txt")

    def test_short_option_bundle_consumes_archive_value(self, scoped_command_workspace):
        # `-xzf archive.tgz`: `x`=extract, `z`=unrecognized flag (skipped),
        # `f`=archive (consumes the next token), all in one bundled token.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation):
            guard.validate(f"tar -xzf {outside}")

    def test_traditional_bundle_gives_each_argument_letter_its_own_token(
        self, scoped_command_workspace
    ):
        # `xfC archive.tar <dir>`: traditional (no-dash) bundled tar syntax
        # gives each argument-taking letter its own subsequent whitespace
        # token, in the order the letters appear — `f` takes the archive,
        # `C` takes the directory. Misreading `C` as `f`'s attached value
        # would leave the archive and the directory as unchecked positionals.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside_dir = shlex.quote(str(sibling_file.parent))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"tar xfC archive.tar {outside_dir}")

        assert exc_info.value.access == "write"

    def test_dash_prefixed_bundle_gives_each_argument_letter_its_own_token(
        self, scoped_command_workspace
    ):
        # `-xfC archive.tar <dir>`: the GNU dash-prefixed cluster must give
        # `C` the same own-token treatment the no-dash form gets above.
        # Misreading `C` as `f`'s attached value would leave the archive and
        # the directory as unchecked positionals.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside_dir = shlex.quote(str(sibling_file.parent))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"tar -xfC archive.tar {outside_dir}")

        assert exc_info.value.access == "write"

    def test_rejects_dynamic_short_option_bundle(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate('OPTION=farchive.tar; tar -c"$OPTION" own.txt')

    def test_added_command_argv_uses_same_path_policy(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate_argv(["tar", "-tf", str(sibling_file)])

    def test_index_file_option_writes_the_verbose_listing(
        self, scoped_command_workspace
    ):
        # `--index-file`'s separated argument must be write-checked, not
        # returned as an unchecked in-archive member selector.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"tar --index-file {shlex.quote(str(sibling_file))} -xf a.tar"
            )

        assert exc_info.value.access == "write"

    def test_index_file_option_workspace_path_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("tar --index-file listing.txt -xf a.tar")

    def test_exclude_pattern_is_not_misresolved_as_exclude_from_path(
        self, scoped_command_workspace
    ):
        # N2: `--exclude` is a real, distinct tar option (a glob PATTERN
        # matched against member names, never a local path) that this
        # family did not model; before it was added, its unambiguous-prefix
        # match against the modeled `--exclude-from` (a real path option)
        # made `_resolve_long_option` snap `--exclude` to `--exclude-from`
        # and read-check the pattern as if it were a file. An out-of-
        # workspace-looking pattern must not be rejected as a path, since
        # it is never opened as one.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"tar -cf archive.tar --exclude={shlex.quote(str(sibling_file))} own.txt"
        )

    def test_exclude_from_still_read_checked_after_exclude_is_modeled(
        self, scoped_command_workspace
    ):
        # Modeling the exact `--exclude` spelling must not regress
        # `--exclude-from`'s own (real path) read check, including its own
        # unambiguous-prefix abbreviation.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"tar -cf archive.tar --exclude-from={shlex.quote(str(sibling_file))} own.txt"
            )
        assert exc_info.value.access == "read"

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"tar -cf archive.tar --exclude-fr={shlex.quote(str(sibling_file))} own.txt"
            )
        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command_template",
        [
            "tar -x -N {path} -f a.tar",
            "tar --newer-mtime {path} -xf a.tar",
            "tar --after-date={path} -xf a.tar",
        ],
    )
    def test_newer_mtime_option_is_read_checked(
        self, scoped_command_workspace, command_template
    ):
        # R2: `-N`/`--newer-mtime`/`--after-date`'s DATE-OR-FILE argument can
        # name a reference file; before this letter was modeled,
        # `_parse_tar_short_events` silently treated an unmodeled short
        # letter as a bare flag, so `-N`'s own argument fell through as an
        # unchecked positional (extract mode never path-checks a
        # positional).
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "a.tar").write_text("tar", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    def test_newer_mtime_option_workspace_path_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "a.tar").write_text("tar", encoding="utf-8")
        (workspace.output_dir / "reference.txt").write_text("", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("tar -x -N reference.txt -f a.tar")

    @pytest.mark.parametrize(
        "mode_flag", ["-xf", "-tf", "-cf", "-rf", "-uf", "-Af", "-df"]
    )
    def test_unrecognized_bare_long_option_fails_closed_in_every_mode(
        self, scoped_command_workspace, mode_flag
    ):
        # A bare (argument-free) unrecognized long option is ambiguous: it
        # may take a separated argument this parser cannot identify, which
        # would otherwise become an unchecked member-selector positional in
        # extract/list mode, or an under-strict source/destination
        # positional in create/append/update/concatenate/compare mode
        # instead of the write-checked or hard-denied classification the
        # real option would receive (e.g. `--listed-incremental`/
        # `--files-from`). It fails closed in every mode, not only
        # extract/list.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(f"tar --not-a-tar-option {mode_flag} a.tar own.txt")

    def test_unrecognized_long_option_with_value_fails_closed_regardless_of_mode(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("tar --not-a-tar-option=value -cf archive.tar own.txt")


class TestSedAwkCommand:
    """`sed`/`awk`: command-slot lexer over the embedded program (I8/I9).

    sed's `r`/`R`/`w`/`W`/`s///w` file commands and awk's `print`/`printf`
    redirection and `getline` are path-checked through the same containment
    every other family uses; `e`/`s///e` and any awk pipe I/O execute an
    arbitrary shell command and always fail closed. `-f`/`--file` script
    reads route through `_read_policy_script` (M4), the same unknown_effect
    gate, non-regular-file rejection, and bounded read every other script
    read uses.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "sed 'r {path}' own.txt",
            "sed 'R {path}' own.txt",
            "sed '/own/r {path}' own.txt",
        ],
    )
    def test_sed_read_commands_target_out_of_workspace_rejected(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command_template",
        [
            "sed 'w {path}' own.txt",
            "sed 'W {path}' own.txt",
            "sed 's/a/b/w {path}' own.txt",
        ],
    )
    def test_sed_write_commands_target_out_of_workspace_rejected(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    def test_sed_write_commands_inside_workspace_are_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("sed 'w own_out.txt' own.txt")
        guard.validate("sed 's/a/b/w own_out.txt' own.txt")

    @pytest.mark.parametrize(
        "command_template",
        [
            "sed -n '/x/Iw {path}' own.txt",
            "sed '/x/M w {path}' own.txt",
            "sed -n '/x/IMw {path}' own.txt",
            "sed -n '/x/MIw {path}' own.txt",
            "sed -n '1,/x/Iw {path}' own.txt",
        ],
    )
    def test_sed_regex_address_modifier_does_not_hide_write_command(
        self, scoped_command_workspace, command_template
    ):
        # C3: a trailing `I`/`M` regex-address modifier (in either order or
        # combination, on either address of a range) must be consumed as
        # part of the address, not misread as the command letter itself --
        # otherwise the real `w` command and its filename are swallowed
        # whole as opaque trailing text and never path-checked.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    def test_sed_regex_address_modifier_write_command_inside_workspace_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("sed -n '/x/Iw own_out.txt' own.txt")
        guard.validate("sed '/x/M w own_out.txt' own.txt")

    def test_sed_braces_in_pattern_are_data(self, scoped_command_workspace):
        # `{`/`}`/`/` inside a delimited pattern or replacement are DATA, not
        # block/command structure; a benign program using them must be
        # allowed, not misparsed as an unterminated or unsafe command.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(r"sed 's/a\/b{}/x/' own.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "sed 'e cmd' own.txt",
            "sed '1e' own.txt",
        ],
    )
    def test_sed_execute_command_fails_closed(self, scoped_command_workspace, command):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_sed_substitution_execute_flag_fails_closed(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPolicyViolation):
            guard.validate(f"sed 's#.*#cat {outside}#e' own.txt")

    def test_sed_script_file_read_out_of_workspace_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(f"{sibling_file}.sed")

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"sed -f {outside} own.txt")

        assert exc_info.value.access == "read"

    def test_sed_script_file_inside_workspace_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "safe.sed").write_text(
            "s/own/safe/\n", encoding="utf-8"
        )
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("sed -f safe.sed own.txt")

    @pytest.mark.parametrize("command_name", ["sed", "awk"])
    def test_unknown_effect_poisons_script_file_inspection(
        self, scoped_command_workspace, command_name
    ):
        # M4: `-f`/`--file` script reads route through the same shared
        # `_read_policy_script` every other family uses, so an unclassified
        # command earlier in the same chain (poisoning `unknown_effect`)
        # blocks the later script read exactly like it would for `bash`.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "prog").write_text(
            "s/own/safe/\n" if command_name == "sed" else "{ print $0 }\n",
            encoding="utf-8",
        )
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(f"python -c pass; {command_name} -f prog own.txt")

    def test_sed_in_place_switches_file_operand_to_write(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("sed -i 's/a/b/' own.txt")
        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"sed -i 's/a/b/' {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "in_place_option",
        ["-i.bak", "-i.orig", "--in-place=.bak", "--in-place=bak"],
    )
    def test_sed_in_place_backup_suffix_poisons_unknown_effect(
        self, scoped_command_workspace, in_place_option
    ):
        # N3: `-i`/`--in-place` with a backup suffix derives a second write
        # target (the backup file) whose exact name this guard never
        # computes and never path-checks, unlike find/tar/gzip's other
        # poisoning sites for a non-enumerable derived write. It must
        # poison `unknown_effect` the same way, so a later, unrelated
        # script inspection in the same chain fails closed instead of
        # silently ignoring the untracked backup write.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "other_safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(
                f"sed {in_place_option} 's/a/b/' own.txt; bash other_safe.sh"
            )

    def test_sed_in_place_without_backup_suffix_does_not_poison_unknown_effect(
        self, scoped_command_workspace
    ):
        # The no-suffix form (`-i`/`--in-place` alone) only ever writes the
        # operand itself, which is already write-checked per-path; it must
        # not poison unrelated later script inspection.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "other_safe.sh").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("sed -i 's/a/b/' own.txt; bash other_safe.sh")
        guard.validate("sed --in-place 's/a/b/' own.txt; bash other_safe.sh")

    def test_sed_long_option_abbreviation_is_resolved(self, scoped_command_workspace):
        # GNU unambiguous-prefix abbreviation must resolve `--expr=` to
        # `--expression=` and still write-check its `w` file command, not
        # silently skip the option as unrecognized.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("sed --expr='w own_out.txt' own.txt")
        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"sed --expr='w {sibling_file}' own.txt")

        assert exc_info.value.access == "write"

    def test_sed_in_place_long_option_abbreviation_switches_to_write(
        self, scoped_command_workspace
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"sed --in-pl -e 's/a/b/' {shlex.quote(str(external_file))}")

        assert exc_info.value.access == "write"

    def test_sed_script_file_long_option_abbreviation_is_read_checked(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"sed --fil={shlex.quote(str(sibling_file))} own.txt")

        assert exc_info.value.access == "read"

    def test_sed_rejects_unrecognized_long_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("sed --not-a-sed-option 's/a/b/' own.txt")

    @pytest.mark.parametrize(
        "flag",
        [
            "--quiet",
            "--silent",
            "--posix",
            "--sandbox",
            "--separate",
            "--null-data",
            "--zero-terminated",
            "--unbuffered",
            "--regexp-extended",
            "--debug",
            "--follow-symlinks",
            "--help",
            "--version",
        ],
    )
    def test_sed_flag_only_long_options_are_allowed(
        self, scoped_command_workspace, flag
    ):
        # These GNU sed long options never consume a value and never name a
        # path; before `flag_long_options` existed they fell through to the
        # unrecognized-long-option fail-closed path (`_check_script_command`
        # raised on every one of them).
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"sed {flag} 's/x/y/' own.txt")

    def test_sed_unrecognized_long_option_still_fails_closed_ahead_of_a_path(
        self, scoped_command_workspace
    ):
        # `flag_long_options` must only resolve the specific GNU flags it
        # lists; a genuinely unmodeled long option must still fail closed
        # instead of the addition making the unknown-long-option path
        # permissive and letting the following out-of-workspace operand
        # slip through unclassified.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(f"sed --bogus {shlex.quote(str(sibling_file))} own.txt")

    @pytest.mark.parametrize(
        "command_template",
        [
            "awk '{{print $0 > \"{path}\"}}' own.txt",
            'awk \'{{printf "%s", $0 > "{path}"}}\' own.txt',
            "awk '{{print $0 >> \"{path}\"}}' own.txt",
        ],
    )
    def test_awk_print_redirect_out_of_workspace_rejected(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=str(sibling_file)))

        assert exc_info.value.access == "write"

    def test_awk_print_redirect_inside_workspace_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("awk '{print $0 > \"own_out.txt\"}' own.txt")

    def test_awk_system_call_fails_closed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("awk '{system(\"rm x\")}' own.txt")

    def test_awk_getline_redirect_out_of_workspace_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"awk '{{getline < \"{sibling_file}\"}}' own.txt")

        assert exc_info.value.access == "read"

    def test_awk_getline_inside_workspace_is_allowed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "feed.txt").write_text("data\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("awk '{getline < \"feed.txt\"}' own.txt")

    @pytest.mark.parametrize(
        "command_template",
        [
            "awk '{{getline $0 < \"{path}\"}}' own.txt",
            "awk '{{getline $1 < \"{path}\"}}' own.txt",
            "awk '{{getline arr[1] < \"{path}\"}}' own.txt",
        ],
    )
    def test_awk_getline_field_and_subscript_targets_are_read_checked(
        self, scoped_command_workspace, command_template
    ):
        # A field reference (`$0`/`$1`) or an array subscript (`arr[1]`)
        # receiving target must not make the following `< FILE` source
        # invisible to the read check.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=str(sibling_file)))

        assert exc_info.value.access == "read"

    @pytest.mark.parametrize(
        "command",
        [
            "awk '{getline $0 < \"feed.txt\"}' own.txt",
            "awk '{getline $1 < \"feed.txt\"}' own.txt",
            "awk '{getline arr[1] < \"feed.txt\"}' own.txt",
        ],
    )
    def test_awk_getline_field_and_subscript_targets_workspace_path_is_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "feed.txt").write_text("data\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "awk '\"cmd\" | getline' own.txt",
            "awk '{print | \"cmd\"}' own.txt",
            "awk '\"cmd\" |& getline' own.txt",
            "awk 'BEGIN{ \"cat /etc/passwd\" |& getline l }' own.txt",
        ],
    )
    def test_awk_pipe_io_fails_closed(self, scoped_command_workspace, command):
        # M5: a two-way coprocess pipe (`|&`) reads from and writes to an
        # arbitrary shell command exactly like a plain `|` pipe and must
        # fail closed the same way, not be silently missed because the
        # detection only recognized a prefix ending in a bare `|`.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_awk_comparison_operator_is_not_misread_as_redirect(
        self, scoped_command_workspace
    ):
        # `print (2 > 1)`: the `>` is a comparison inside parens, not a
        # redirect, so paren-depth tracking must keep this from being
        # misread as file I/O.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("awk 'BEGIN { print (2 > 1) }'")

    @pytest.mark.parametrize(
        "command",
        [
            "awk '{print > $1}' own.txt",
            "awk -v target=out.txt '{print $0 > target}' own.txt",
            "awk -v target=out.txt '{print $0 > (target)}' own.txt",
        ],
    )
    def test_awk_dynamic_redirect_target_rejected(
        self, scoped_command_workspace, command
    ):
        # A redirect target is only statically resolvable as a double-quoted
        # string literal; a field reference (`$1`), a bareword variable, or
        # a parenthesized expression is dynamic from the lexer's point of
        # view and must fail closed rather than being silently skipped.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_awk_script_file_inside_workspace_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "safe.awk").write_text(
            "{ print $0 }\n", encoding="utf-8"
        )
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("awk -f safe.awk own.txt")

    def test_awk_long_option_abbreviation_is_resolved(self, scoped_command_workspace):
        # GNU unambiguous-prefix abbreviation must resolve `--sour=` to
        # `--source=` and still classify its print-redirect write, not
        # silently skip the option as unrecognized. The whole `--sour=...`
        # argument is quoted as one shell word (matching how a caller would
        # protect an embedded double-quoted redirect target).
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("awk '--sour={print $0 > \"own_out.txt\"}' own.txt")
        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"awk '--sour={{print $0 > \"{sibling_file}\"}}' own.txt")

        assert exc_info.value.access == "write"

    def test_awk_rejects_unrecognized_long_option(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("awk --not-an-awk-option '{print}' own.txt")

    def test_awk_rejects_unmodeled_short_option(self, scoped_command_workspace):
        # `-Z` is not a real gawk option; an unmodeled short option must fail
        # closed rather than being silently skipped.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(
                f"awk -Z {shlex.quote(str(sibling_file))} '{{print}}' own.txt"
            )

    @pytest.mark.parametrize(
        "command_template",
        [
            "awk -o{path} '{{print}}' own.txt",
            "awk --pretty-print={path} '{{print}}' own.txt",
            "awk -p{path} '{{print}}' own.txt",
            "awk --profile={path} '{{print}}' own.txt",
            "awk -d{path} '{{print}}' own.txt",
            "awk --dump-variables={path} '{{print}}' own.txt",
        ],
    )
    def test_awk_optional_write_options_reject_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "awk -oprof.out '{print}' own.txt",
            "awk --pretty-print=prof.out '{print}' own.txt",
            "awk -pprof.out '{print}' own.txt",
            "awk --profile=prof.out '{print}' own.txt",
            "awk -dvars.out '{print}' own.txt",
            "awk --dump-variables=vars.out '{print}' own.txt",
        ],
    )
    def test_awk_optional_write_options_workspace_paths_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command",
        [
            "awk -o '{print}' own.txt",
            "awk --pretty-print '{print}' own.txt",
        ],
    )
    def test_awk_optional_write_option_default_filename_is_write_checked(
        self, scoped_command_workspace, command
    ):
        # With no attached argument, gawk writes the fixed default filename
        # (`awkprof.out`) in the current directory; that default is a
        # workspace-relative path, so it is allowed without needing an
        # explicit operand for it.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_awk_program_containing_assignment_syntax_is_still_inspected(
        self, scoped_command_workspace
    ):
        # The CLI `var=value` operand skip (`ignores_assignment_arguments`)
        # must never apply to the PROGRAM token itself: dropping it here
        # (because the program text happens to contain "=") would silently
        # disable all program inspection, letting `system()` through
        # unchecked.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("awk 'BEGIN{x=1; system(\"id\")}'")

    @pytest.mark.parametrize(
        "command_template",
        [
            "awk '{{x=1; print > \"{path}\"}}' own.txt",
            "awk '{{x=1; getline l < \"{path}\"}}' own.txt",
        ],
    )
    def test_awk_assignment_bearing_program_still_classifies_redirect_targets(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate(command_template.format(path=str(sibling_file)))

    def test_awk_positional_assignment_operand_after_program_is_not_a_path(
        self, scoped_command_workspace
    ):
        # A real `var=value` CLI operand, positioned AFTER the program, is
        # still a scalar assignment, not a file operand — the fix must not
        # turn this into a spurious path check.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("awk '{print}' x=1 own.txt")

    def test_awk_redirect_target_string_concatenation_fails_closed(
        self, scoped_command_workspace
    ):
        # `print > "sub/" "../../8/x"` is awk string concatenation (bare
        # juxtaposition); only resolving the first quoted literal and
        # ignoring what follows would validate `"sub/"` while the actual
        # runtime target escapes it. Anything after the closing quote other
        # than a statement terminator must fail closed.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate('awk \'{print > "sub/" "../../8/x"}\' own.txt')

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
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    @pytest.mark.parametrize(
        ("script_name", "script_template", "invocation"),
        [
            ("dynamic.sed", "w {path}", "sed -f dynamic.sed own.txt"),
            (
                "dynamic.awk",
                '{{print $0 > "{path}"}}',
                "awk -f dynamic.awk own.txt",
            ),
        ],
    )
    def test_rejects_scripts_created_earlier_in_the_same_chain(
        self,
        scoped_command_workspace,
        script_name,
        script_template,
        invocation,
    ):
        # A script written earlier in the same command chain registers a
        # write on its own path; M4's shared `_read_policy_script` then
        # refuses to inspect it, exactly like `bash`'s own script reads.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        script = script_template.format(path=sibling_file)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(
                f"printf '%s\\n' {shlex.quote(script)} > {script_name} && {invocation}"
            )

    @pytest.mark.parametrize(
        ("script_name", "safe_script", "unsafe_script", "invocation"),
        [
            (
                "replace.sed",
                "s/own/own/",
                "w {path}",
                "sed -f replace.sed own.txt",
            ),
            (
                "replace.awk",
                "{ print $0 }",
                '{{print $0 > "{path}"}}',
                "awk -f replace.awk own.txt",
            ),
        ],
    )
    def test_rejects_script_overwritten_before_inspection(
        self,
        scoped_command_workspace,
        script_name,
        safe_script,
        unsafe_script,
        invocation,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        script_path = workspace.output_dir / script_name
        script_path.write_text(safe_script, encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        replacement = unsafe_script.format(path=sibling_file)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate(
                f"printf '%s\\n' {shlex.quote(replacement)} > {script_name} "
                f"&& {invocation}"
            )

        assert script_path.read_text(encoding="utf-8") == safe_script

    def test_script_write_tracking_is_scoped_to_one_validation(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        (workspace.output_dir / "safe.sed").write_text("s/own/safe/", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("printf replacement > safe.sed")
        guard.validate("sed -f safe.sed own.txt")

    def test_script_inspection_rejects_oversized_regular_file(
        self, scoped_command_workspace
    ):
        # M4: routes through the shared `_read_policy_script`, whose byte
        # budget (`_MAX_INSPECTED_SCRIPT_BYTES`) and `CommandPolicyViolation`
        # (not `CommandPathViolation`, reserved for non-regular files) this
        # test confirms sed/awk inherit rather than reimplement.
        workspace, _, _ = scoped_command_workspace
        limit = command_path_guard_module._MAX_INSPECTED_SCRIPT_BYTES
        script_path = workspace.output_dir / "large.sed"
        script_path.write_bytes(b"#" * (limit + 1))
        guard = WorkspaceCommandPathGuard(workspace)

        with command_path_guard_module._validation_session_scope():
            with pytest.raises(
                CommandPolicyViolation,
                match=rf"shell policy script exceeds the {limit}-byte inspection limit",
            ):
                guard._inspect_script_file(
                    "sed", str(script_path), workspace.output_dir
                )


class TestDdBase64GzipRemoteTransferCommand:
    """`dd`/`base64`/`gzip` and the remote-transfer family (`rsync`/`curl`/
    `wget`): the final family stage of the file-command policy layer.

    `gzip`'s default mode is the family's M2-additive case (a real write
    check on the operand plus `unknown_effect` for the non-enumerable
    derived `.gz`); `rsync` is the family's all-or-nothing remote case
    (M5: any remote operand rejects the whole invocation, not just the
    remote side); `curl`/`wget` fail closed on their own uninspectable
    option grammar, independent of `_is_remote_transfer_operand`.
    """

    # -- dd --------------------------------------------------------------

    def test_dd_write_operand_outside_workspace_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"dd if=own.txt of={shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    def test_dd_read_operand_outside_workspace_rejected(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"dd if={shlex.quote(str(sibling_file))} of=own.txt")

        assert exc_info.value.access == "read"

    def test_dd_non_path_assignments_are_allowed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("dd if=own.txt of=own2.txt bs=512 count=1")

    def test_dd_flag_assignments_are_not_paths(self, scoped_command_workspace):
        # `iflag=`/`oflag=` share the `if=`/`of=` prefix character but are
        # not path-bearing; a substring match instead of an exact prefix
        # match would misclassify them.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("dd iflag=fullblock oflag=sync")

    def test_dd_argv_uses_same_path_policy(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate_argv(["dd", f"if={sibling_file}", "of=copy.bin"])

    # -- base64 ------------------------------------------------------------

    @pytest.mark.parametrize(
        "command_template",
        [
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

    @pytest.mark.parametrize(
        "command_template",
        [
            "base64 -i{path}",
            "base64 --input {path}",
            "base64 --input={path}",
            "base64 -di{path}",
        ],
    )
    def test_base64_ignore_garbage_is_boolean_not_a_path_option(
        self, scoped_command_workspace, command_template
    ):
        # `-i`/`--ignore-garbage` is a boolean flag (GNU coreutils), not a
        # value-taking `--input` option: this family no longer models a
        # "-i"-consumes-a-path grammar at all, so a value attempted against
        # it (attached, or `--input`, which base64 does not actually have)
        # fails closed as an unrecognized option rather than being
        # write/read-classified — still a REJECT, just not a path-specific
        # one, since `-i` never carries a path to classify in the first
        # place.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

    def test_base64_workspace_input_and_output_are_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("base64 -i own.txt -o own.b64")

    def test_base64_argv_uses_same_path_policy(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation):
            guard.validate_argv(["base64", "-i", str(sibling_file)])

    @pytest.mark.parametrize(
        "command_template",
        [
            "base64 --outp={path} own.txt",
            "base64 --output={path} own.txt",
        ],
    )
    def test_rejects_base64_output_abbreviation(
        self, scoped_command_workspace, command_template
    ):
        # `--outp=`/`--output=` must resolve through the same GNU
        # unambiguous-prefix matching `grep`/`curl`/`wget` use: an
        # unresolved abbreviation must not silently skip the write check.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    def test_base64_benign_invocations_are_allowed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("base64 own.txt")
        guard.validate("base64 -w0 own.txt")

    # -- gzip ----------------------------------------------------------------

    def test_gzip_default_mode_write_checks_the_operand(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"gzip {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    def test_gzip_stdout_mode_keeps_external_operand_read_only(
        self,
        scoped_command_workspace,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"gzip -c {shlex.quote(str(external_file))}")
        guard.validate(f"gzip -lt {shlex.quote(str(external_file))}")

    def test_gzip_suffix_value_is_not_misread_as_read_only_mode(
        self, scoped_command_workspace
    ):
        # M3: `-S`'s own (attached-or-separate) suffix argument must not be
        # misread as the `-t`/`-l`/`-c` read-only-mode flag merely because
        # it contains that character. `-S -t <path>` here is "-S" with a
        # SEPARATE suffix argument of literally "-t" — the read-only-mode
        # classification is option-aware (which characters were parsed as
        # flags), not a position-independent scan of the raw token text, so
        # this stays the DEFAULT write-checked mode, not read-only.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"gzip -S -t {shlex.quote(str(external_file))}")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("command", ["gzip --best own.txt", "gzip --fast own.txt"])
    def test_gzip_allows_recognized_compression_level_flags(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_gzip_default_mode_poisons_later_script_inspection(
        self, scoped_command_workspace
    ):
        # The `.gz` gzip actually creates is a distinct, non-enumerable path
        # this guard never computes; the operand's own write check cannot
        # capture that derived file, so `unknown_effect` (M2 additive) must
        # poison later script inspection in the same chain.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text(":\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(
            CommandPolicyViolation,
            match="cannot inspect a script affected by an earlier command",
        ):
            guard.validate("gzip own.txt ; bash own.txt.gz")

    # -- rsync -----------------------------------------------------------

    @pytest.mark.parametrize(
        "command",
        [
            "rsync own.txt host:/remote",
            "rsync own.txt user@host:/remote",
            "rsync own.txt rsync://host/remote",
        ],
    )
    def test_rsync_remote_operand_rejects_the_whole_invocation(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_rsync_local_destination_outside_workspace_rejected(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"rsync own.txt {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "write"

    def test_rsync_link_dest_requires_write_authorization(
        self,
        scoped_command_workspace,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"rsync -a --link-dest={shlex.quote(str(external_file.parent))} "
                "source copied"
            )

        assert exc_info.value.access == "write"

    def test_rsync_backup_dir_requires_write_authorization(
        self,
        scoped_command_workspace,
    ):
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"rsync -a --backup-dir={shlex.quote(str(external_file.parent))} "
                "source copied"
            )

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("option_spelling", ["--link-des", "--link-dest"])
    def test_rsync_link_dest_space_separated_form_requires_write_authorization(
        self, scoped_command_workspace, option_spelling
    ):
        # `--link-dest` may hard-link unchanged files straight into the
        # destination, so a read-only external root (read-allowed, not
        # write-allowed) is not sufficient authorization for it. Both the
        # GNU unambiguous-prefix abbreviation and the full spelling must
        # resolve through `_resolve_long_option` and write-check the
        # space-separated argument instead of leaking it into the generic
        # read/write operand list.
        workspace, external_file, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"rsync -a {option_spelling} {shlex.quote(str(external_file))} "
                "own.txt d"
            )

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize("option_spelling", ["--backup-di", "--backup-dir"])
    def test_rsync_backup_dir_space_separated_form_requires_write_authorization(
        self, scoped_command_workspace, option_spelling
    ):
        workspace, external_file, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"rsync -a {option_spelling} {shlex.quote(str(external_file))} "
                "own.txt d"
            )

        assert exc_info.value.access == "write"

    def test_rsync_benign_local_invocation_is_allowed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "src.txt").write_text("src\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("rsync -a src.txt dst.txt")

    def test_rsync_remote_source_still_rejects_the_whole_invocation(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("rsync src host:/x")

    def test_rsync_log_file_option_is_write_checked(self, scoped_command_workspace):
        # `--log-file` is a modeled write-owning option: its argument must
        # be write-checked directly, not leaked into the generic
        # read/write operand list (the prior, coincidental behavior this
        # test used to pin).
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"rsync --log-file {shlex.quote(str(sibling_file))} own.txt dest.txt"
            )

        assert exc_info.value.access == "write"

    def test_rsync_short_temp_dir_option_requires_write_authorization(
        self, scoped_command_workspace
    ):
        # C2/M4: `-T`/`--temp-dir` had no short-option grammar at all, so
        # `-T` silently fell through as an unmodeled (ignored) short flag
        # instead of write-checking its directory argument.
        workspace, external_file, _ = scoped_command_workspace
        (workspace.output_dir / "src.txt").write_text("src\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"rsync -T {shlex.quote(str(external_file.parent))} src.txt dst.txt"
            )

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command", ["rsync -K src.txt dst.txt", "rsync -k src.txt dst.txt"]
    )
    def test_rsync_keep_or_copy_dirlinks_short_option_fails_closed(
        self, scoped_command_workspace, command
    ):
        # M4: `-K`/`--keep-dirlinks` and `-k`/`--copy-dirlinks` let rsync
        # write through (or read through) an existing symlink at the
        # destination/source instead of the literal directory entry, which
        # can escape the intended root; both must deny outright, matching
        # `--keep-dirlinks`'s existing long-option denial, instead of
        # silently falling through as an unmodeled short flag.
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "src.txt").write_text("src\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_rsync_single_operand_is_still_read_checked(self, scoped_command_workspace):
        # N1: `len(operands) < 2` used to skip ALL containment (including
        # the remote-operand check) for a single-operand invocation; a lone
        # local operand must still be read-checked, not silently exempted
        # for lack of a second (destination) operand.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"rsync {shlex.quote(str(sibling_file))}")

        assert exc_info.value.access == "read"

    # -- curl/wget/rsync shared uninspectable-option pins -----------------

    @pytest.mark.parametrize(
        "command",
        [
            "curl -O https://example.invalid/file",
            "wget --config config",
            "rsync host:/secret copied",
            "rsync --files-from=list.txt source copied",
            "rsync --files-fr list.txt source copied",
        ],
    )
    def test_rejects_uninspectable_new_tool_path_sources(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(command)

    def test_wget_config_option_fails_closed_regardless_of_path(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(f"wget --config {shlex.quote(str(sibling_file))}")

    # -- curl ----------------------------------------------------------------

    def test_curl_output_outside_workspace_rejected(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"curl -o {outside} https://example.invalid/file")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command_template",
        [
            "curl -D {path} https://example.invalid/file",
            "curl --dump-header {path} https://example.invalid/file",
            "curl -c {path} https://example.invalid/file",
            "curl --cookie-jar {path} https://example.invalid/file",
            "curl --trace {path} https://example.invalid/file",
            "curl --trace-ascii {path} https://example.invalid/file",
            "curl --stderr {path} https://example.invalid/file",
        ],
    )
    def test_curl_write_output_options_reject_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "curl -D headers.txt https://example.invalid/file",
            "curl --dump-header headers.txt https://example.invalid/file",
            "curl -c cookies.txt https://example.invalid/file",
            "curl --cookie-jar cookies.txt https://example.invalid/file",
            "curl --trace trace.txt https://example.invalid/file",
            "curl --trace-ascii trace.txt https://example.invalid/file",
            "curl --stderr err.txt https://example.invalid/file",
        ],
    )
    def test_curl_write_output_options_workspace_paths_are_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_curl_cookie_option_is_read_checked_not_write(
        self, scoped_command_workspace
    ):
        # N2: `--cookie` is a real, distinct curl option (a file curl
        # *reads* cookies from to send, or an inline `name=value` cookie
        # list) that this family did not model; before it was added, its
        # unambiguous-prefix match against the modeled `--cookie-jar` (a
        # write path) made `_resolve_long_option` snap `--cookie` to
        # `--cookie-jar` and write-check the argument. A read-only cookie
        # file the invocation may only read (an approved external
        # directory) must be allowed, not rejected as a write.
        workspace, external_file, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            f"curl --cookie {shlex.quote(str(external_file))} "
            "https://example.invalid/file"
        )

    def test_curl_cookie_abbreviation_is_ambiguous_and_fails_closed(
        self, scoped_command_workspace
    ):
        # Every strict abbreviation of `--cookie` is also a prefix of the
        # longer `--cookie-jar`, so once both are modeled, only the exact
        # `--cookie` spelling is unambiguous; a truncated spelling must fail
        # closed rather than silently snapping to either candidate.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("curl --cooki somefile https://example.invalid/file")

    def test_curl_cookie_option_rejects_out_of_workspace_read(
        self, scoped_command_workspace
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"curl --cookie {shlex.quote(str(sibling_file))} "
                "https://example.invalid/file"
            )

        assert exc_info.value.access == "read"

    def test_curl_cookie_jar_still_write_checked_after_cookie_is_modeled(
        self, scoped_command_workspace
    ):
        # Modeling the exact `--cookie` spelling must not regress
        # `--cookie-jar`'s own (real write path) check, including its own
        # unambiguous-prefix abbreviation.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"curl --cookie-jar {shlex.quote(str(sibling_file))} "
                "https://example.invalid/file"
            )
        assert exc_info.value.access == "write"

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"curl --cookie-j={shlex.quote(str(sibling_file))} "
                "https://example.invalid/file"
            )
        assert exc_info.value.access == "write"

    def test_curl_rejects_unrecognized_long_option_with_value(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(
                "curl --not-a-curl-option=value https://example.invalid/file"
            )

    def test_curl_allows_unrecognized_bare_long_flag(self, scoped_command_workspace):
        # A long flag with no attached value cannot be told apart from a
        # real curl flag this family does not model, so it stays permissive
        # (over-checked, never under-checked) — only an unmodeled option that
        # visibly carries a value fails closed.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("curl --location https://example.invalid/file")

    @pytest.mark.parametrize(
        "command_template",
        [
            "curl -sD {path} https://example.invalid/file",
            "curl -sLD {path} https://example.invalid/file",
            "curl -sc {path} https://example.invalid/file",
        ],
    )
    def test_curl_bundled_write_option_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        # `-D`/`-c` must still be recognized (and write-checked) when they
        # are not the leading character of a bundled short-option cluster
        # (e.g. `-sD` combines silent mode with dump-header); a hand-rolled
        # leading-character-only match silently skips the whole token.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "curl -sD headers.txt https://example.invalid/file",
            "curl -sLD headers.txt https://example.invalid/file",
            "curl -sc cookies.txt https://example.invalid/file",
        ],
    )
    def test_curl_bundled_write_option_workspace_path_is_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "curl --libcurl {path} https://example.invalid/file",
            "curl --hsts {path} https://example.invalid/file",
            "curl --etag-save {path} https://example.invalid/file",
        ],
    )
    def test_curl_new_write_options_reject_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command_template",
        [
            "curl --dump-hea={path} https://example.invalid/file",
            "curl --cookie-j={path} https://example.invalid/file",
        ],
    )
    def test_curl_long_option_abbreviation_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    def test_curl_bundled_remote_output_flag_fails_closed(
        self, scoped_command_workspace
    ):
        # `-sO` (silent + remote-name) is a common curl idiom; `-O`'s output
        # filename is derived from the remote response and is never
        # statically knowable, so this must fail closed even bundled behind
        # another flag, not silently fall through as permissive.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("curl -sO https://example.invalid/file")

    def test_curl_file_url_output_source_is_read_checked(
        self, scoped_command_workspace
    ):
        # T3/C1: a `file:` URL is a real filesystem channel, not a network
        # transfer; the positional URL operand was never inspected at all
        # (a bare `index += 1`), so this used to ALLOW unconditionally.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"curl -o out.txt file://{sibling_file}")

        assert exc_info.value.access == "read"

    def test_curl_upload_to_file_url_is_write_checked(self, scoped_command_workspace):
        # T3/C1: `-T`/`--upload-file` turns the request into an upload whose
        # DESTINATION is the URL operand, so a `file:` URL there is a write
        # target, not a read source.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"curl -T own.txt file://{sibling_file}")

        assert exc_info.value.access == "write"

    def test_curl_upload_target_classification_is_order_independent(
        self, scoped_command_workspace
    ):
        # `-T` need not precede the URL operand for curl's own option
        # parsing; the guard's write-vs-read classification of the URL must
        # not depend on argv order either.
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"curl file://{sibling_file} -T own.txt")

        assert exc_info.value.access == "write"

    def test_curl_file_url_workspace_path_is_allowed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"curl -o out.txt file://{own_file}")

    @pytest.mark.parametrize(
        "scheme", ["gopher", "smb", "dict", "ldap", "unknown-scheme"]
    )
    def test_curl_unrecognized_url_scheme_fails_closed(
        self, scoped_command_workspace, scheme
    ):
        # Any scheme this family does not explicitly recognize as network
        # transfer fails closed rather than being assumed non-local.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(f"curl -o out.txt {scheme}://example.invalid/file")

    def test_curl_end_of_options_marker_stops_option_parsing(
        self, scoped_command_workspace
    ):
        # A literal `--` ends option parsing; a URL-shaped token after it is
        # still a plain operand, not silently reinterpreted as an option.
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("curl -o out.txt -- https://example.invalid/file")

    # -- wget ------------------------------------------------------------

    def test_wget_output_outside_workspace_rejected(self, scoped_command_workspace):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"wget -O {outside} https://example.invalid/file")

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command_template",
        [
            "wget -O keep -o {path} https://example.invalid/file",
            "wget -O keep --output-file {path} https://example.invalid/file",
        ],
    )
    def test_wget_log_file_option_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    @pytest.mark.parametrize(
        "command",
        [
            "wget -O keep -o log.txt https://example.invalid/file",
            "wget -O keep --output-file log.txt https://example.invalid/file",
        ],
    )
    def test_wget_log_file_option_workspace_path_is_allowed(
        self, scoped_command_workspace, command
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    def test_wget_rejects_unrecognized_long_option_with_value(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(
                "wget --not-a-wget-option=value https://example.invalid/file"
            )

    def test_wget_save_cookies_still_rejected(self, scoped_command_workspace):
        # `--save-cookies` is not modeled as a write option; it must not be
        # "fixed" into an ALLOW. It rejects today because the whole
        # invocation has no explicit `-O` output, so the remote-derived
        # output filename fails closed — that reason must not regress into
        # a silent ALLOW even once `--save-cookies` itself is modeled.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate(
                f"wget --save-cookies {shlex.quote(str(sibling_file))} "
                "https://example.invalid/file"
            )

    def test_wget_save_cookies_write_checked_when_output_is_given(
        self, scoped_command_workspace
    ):
        # With an explicit `-O` the "no output filename" fail-closed reason
        # above no longer applies; `--save-cookies` must be write-checked
        # in its own right rather than silently falling through as an
        # unmodeled option.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(
                f"wget -O keep --save-cookies {shlex.quote(str(sibling_file))} "
                "https://example.invalid/file"
            )

        assert exc_info.value.access == "write"

    def test_wget_save_cookies_workspace_path_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(
            "wget -O keep --save-cookies cookies.txt https://example.invalid/file"
        )

    @pytest.mark.parametrize(
        "command_template",
        [
            "wget --output-docu={path} https://example.invalid/file",
            "wget --directory-p={path} https://example.invalid/file",
            "wget -O keep --output-fi={path} https://example.invalid/file",
        ],
    )
    def test_wget_long_option_abbreviation_rejects_out_of_workspace(
        self, scoped_command_workspace, command_template
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    def test_wget_bundled_output_option_rejects_out_of_workspace(
        self, scoped_command_workspace
    ):
        # `-qO` (quiet + output-document) is a common wget idiom; `-O`'s
        # write-path argument must still be recognized (and write-checked)
        # when it is not the leading character of a bundled short-option
        # cluster.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"wget -qO {outside} https://example.invalid/file")

        assert exc_info.value.access == "write"

    def test_wget_bundled_output_option_workspace_path_is_allowed(
        self, scoped_command_workspace
    ):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate("wget -qO out.txt https://example.invalid/file")

    def test_wget_bundled_input_file_flag_fails_closed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("wget -qi urls.txt")

    def test_wget_file_url_source_is_read_checked(self, scoped_command_workspace):
        # T3/C1: a `file:` URL is a real filesystem channel; wget's URL
        # operand was never inspected at all, so this used to ALLOW
        # unconditionally.
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"wget -O out.txt file://{sibling_file}")

        assert exc_info.value.access == "read"

    def test_wget_file_url_workspace_path_is_allowed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        own_file = workspace.output_dir / "own.txt"
        own_file.write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(f"wget -O out.txt file://{own_file}")

    def test_wget_unrecognized_url_scheme_fails_closed(self, scoped_command_workspace):
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation):
            guard.validate("wget -O out.txt gopher://example.invalid/file")

    # -- path-classification bypass regressions -------------------------

    @pytest.mark.parametrize(
        "command",
        [
            "gzip -c own.txt",
            "rsync -av own.txt copied.txt",
            "curl -oout.txt https://example.invalid/file",
            "wget -Oout.txt https://example.invalid/file",
        ],
    )
    def test_allows_newly_classified_tool_workspace_paths(
        self,
        scoped_command_workspace,
        command,
    ):
        workspace, _, _ = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        guard.validate(command)

    @pytest.mark.parametrize(
        "command_template",
        [
            "gzip {path}",
            "rsync own.txt {path}",
            "curl -o {path} https://example.invalid/file",
            "wget -O {path} https://example.invalid/file",
        ],
    )
    def test_rejects_newly_classified_tool_writes_to_sibling(
        self,
        scoped_command_workspace,
        command_template,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own\n", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(command_template.format(path=shlex.quote(str(sibling_file))))

        assert exc_info.value.access == "write"

    def test_rejects_newly_classified_tool_reads_from_sibling(
        self,
        scoped_command_workspace,
    ):
        workspace, _, sibling_file = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPathViolation) as exc_info:
            guard.validate(f"rsync {shlex.quote(str(sibling_file))} copied.txt")

        assert exc_info.value.access == "read"


class TestBundledAndAbbreviatedPathOptionCoverage:
    """Recurrence guard for the bug classes this module's hand-rolled option
    parsers kept reintroducing: a value-taking option not recognized when
    bundled behind another short flag (tar's `xfC`, curl's `-sD`, wget's
    `-qO`, cp/mv/ln's `-rt`), and a GNU-abbreviated long option not resolved
    to the option it stands for — in EITHER the `=`-attached form (grep's
    `--fil=`, cp/mv/ln's `--targ=`, curl's `--dump-hea=`, wget's
    `--output-docu=`) or the space-separated form (rsync's `--link-des
    <path>`, which is what let the abbreviated-`--link-dest`/`--backup-dir`
    hole through: the `=`-form was covered here, but the space-separated
    form was not, so `_check_rsync` fell back to its unmodeled-flag path
    and swept the value into the generic read/write operand list instead
    of write-checking it). tar's `--listed-incr`/`--files-fr` rows cover the
    same abbreviation gap in create/append/update/concatenate/compare mode
    specifically, not only extract/list, which is where an earlier revision
    of this parser left it unguarded.

    Each row names one path-bearing option a command owns and drives a
    bundled-short-cluster form (when the option has one; `None` skips it)
    plus BOTH the `--abbr=value` and `--abbr value` spellings of its
    GNU-abbreviated long-option form, against an out-of-workspace path.
    `expected_access` of `None` marks an option this guard denies
    unconditionally (rsync's `--files-from`, which delegates to an external
    file list this guard cannot inspect) rather than access-classifies: both
    spellings must still raise, just without a specific access to assert.
    A command added later that owns a path-bearing option without an entry
    here has no guarantee any of these bug classes was ever checked for it
    — that gap is the point of this table, not an oversight to silently
    fill in.

    `TestModeFlagAbbreviationCoverage` below extends the same guard to a
    different shape of the same bug class: a GNU-abbreviated long option
    that is a bare mode *flag* rather than a path-bearing option (cp's
    `--derefe`/`--recurs`), where the recurrence risk is a flag-combination
    check missing the abbreviated spelling entirely rather than
    misclassifying a value's read/write access.
    """

    # (label, bundled-short-form template or None, GNU-abbreviated-long-form
    # `=`-attached template, expected access or None for unconditional deny)
    _BUNDLED_AND_ABBREVIATED_PATH_OPTIONS: list[
        tuple[str, str | None, str, str | None]
    ] = [
        ("sort -o", "sort -ro{path} own.txt", "sort --out={path} own.txt", "write"),
        ("sort -T", "sort -rT{path} own.txt", "sort --temp={path} own.txt", "write"),
        ("grep -f", "grep -vf {path} own.txt", "grep --fil={path} own.txt", "read"),
        ("cp -t", "cp -rt {path} own.txt", "cp own.txt --targ={path}", "write"),
        ("mv -t", "mv -vt {path} own.txt", "mv own.txt --targ={path}", "write"),
        ("ln -t", "ln -st {path} own.txt", "ln own.txt --targ={path}", "write"),
        (
            "curl -D",
            "curl -sD {path} https://example.invalid/file",
            "curl --dump-hea={path} https://example.invalid/file",
            "write",
        ),
        (
            "curl -c",
            "curl -sc {path} https://example.invalid/file",
            "curl --cookie-j={path} https://example.invalid/file",
            "write",
        ),
        (
            "wget -O",
            "wget -qO {path} https://example.invalid/file",
            "wget --output-docu={path} https://example.invalid/file",
            "write",
        ),
        (
            "wget -P",
            "wget -qP {path} https://example.invalid/file",
            "wget --directory-p={path} https://example.invalid/file",
            "write",
        ),
        (
            "wget -o",
            "wget -O keep -qo {path} https://example.invalid/file",
            "wget -O keep --output-fi={path} https://example.invalid/file",
            "write",
        ),
        (
            "rsync --link-dest",
            None,
            "rsync -a --link-des={path} own.txt d",
            "write",
        ),
        (
            "rsync --backup-dir",
            None,
            "rsync -a --backup-di={path} own.txt d",
            "write",
        ),
        (
            "rsync --files-from",
            None,
            "rsync -a --files-fr={path} own.txt d",
            None,
        ),
        (
            "tar --listed-incremental (create mode)",
            None,
            "tar --create --listed-incr={path} -f a.tar own.txt",
            "write",
        ),
        (
            "tar --files-from (create mode)",
            None,
            "tar --create --files-fr={path} -f a.tar",
            None,
        ),
    ]

    @pytest.mark.parametrize(
        ("label", "bundled_template", "abbreviated_template", "expected_access"),
        _BUNDLED_AND_ABBREVIATED_PATH_OPTIONS,
        ids=[row[0] for row in _BUNDLED_AND_ABBREVIATED_PATH_OPTIONS],
    )
    def test_bundled_and_abbreviated_forms_both_reject_out_of_workspace(
        self,
        scoped_command_workspace,
        label,
        bundled_template,
        abbreviated_template,
        expected_access,
    ):
        del label  # only used as the parametrize id
        workspace, _, sibling_file = scoped_command_workspace
        (workspace.output_dir / "own.txt").write_text("own", encoding="utf-8")
        guard = WorkspaceCommandPathGuard(workspace)
        outside = shlex.quote(str(sibling_file))

        if bundled_template is not None:
            with pytest.raises(CommandPathViolation) as bundled_exc:
                guard.validate(bundled_template.format(path=outside))
            assert bundled_exc.value.access == expected_access

        assert "={path}" in abbreviated_template
        equals_command = abbreviated_template.format(path=outside)
        space_command = abbreviated_template.replace("={path}", " {path}").format(
            path=outside
        )

        if expected_access is None:
            with pytest.raises(CommandPolicyViolation):
                guard.validate(equals_command)
            with pytest.raises(CommandPolicyViolation):
                guard.validate(space_command)
        else:
            with pytest.raises(CommandPathViolation) as equals_exc:
                guard.validate(equals_command)
            assert equals_exc.value.access == expected_access

            with pytest.raises(CommandPathViolation) as space_exc:
                guard.validate(space_command)
            assert space_exc.value.access == expected_access


class TestModeFlagAbbreviationCoverage:
    """Recurrence guard for a GNU-abbreviated long option that is a bare
    mode *flag* rather than a path-bearing option.

    `cp`'s `--dereference`/`--recursive`/`--follow-command-line-symlink`
    carry no value; the bug this covers is a flag-combination check
    (recursive copy through a dereferenced symlink) matching only the full
    spelling, so an abbreviated flag silently misses the combination
    entirely instead of triggering the hard denial. Unlike
    `TestBundledAndAbbreviatedPathOptionCoverage`'s table, there is no
    `--abbr=value`/`--abbr value` duality to drive (a bare flag takes no
    argument), so each row is a single, already fully-formed command.
    """

    # (label, command using a GNU-abbreviated mode flag that must still
    # trigger the recursive+dereference hard denial)
    _ABBREVIATED_MODE_FLAG_COMMANDS: list[tuple[str, str]] = [
        ("cp --derefe (long, abbreviated) + -r (short)", "cp --derefe -r . copied"),
        ("cp -r (short) + --derefe (long, abbreviated)", "cp -r --derefe . copied"),
        (
            "cp --recurs (long, abbreviated) + --dereference (long, full)",
            "cp --recurs --dereference . copied",
        ),
        (
            "cp -r (short) + --follow-comm (long, abbreviated)",
            "cp -r --follow-comm . copied",
        ),
    ]

    @pytest.mark.parametrize(
        ("label", "command"),
        _ABBREVIATED_MODE_FLAG_COMMANDS,
        ids=[row[0] for row in _ABBREVIATED_MODE_FLAG_COMMANDS],
    )
    def test_abbreviated_mode_flag_still_triggers_hard_denial(
        self, scoped_command_workspace, label, command
    ):
        del label  # only used as the parametrize id
        workspace, _, _ = scoped_command_workspace
        guard = WorkspaceCommandPathGuard(workspace)

        with pytest.raises(CommandPolicyViolation, match="symbolic links"):
            guard.validate(command)
