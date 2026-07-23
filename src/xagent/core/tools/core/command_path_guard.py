"""Best-effort workspace path guard for shell commands.

This module deliberately provides a cooperative guard, not an operating-system
security boundary. For supported commands, it rejects out-of-scope paths and
path-bearing arguments that cannot be resolved statically. Unknown commands
remain unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, SupportsIndex, cast

import bashlex

from ...workspace import TaskWorkspace

logger = logging.getLogger(__name__)

PathAccess = Literal["read", "write"]
_MAX_INSPECTED_SCRIPT_BYTES = 1024 * 1024
_MAX_INSPECTED_SCRIPT_DEPTH = 8
_MAX_COMMAND_WRAPPER_DEPTH = 32
_AWK_ALWAYS_UNSAFE_IO_PATTERN = re.compile(r"\bsystem\s*\(|\bgetline\b")
_AWK_PRINT_PATTERN = re.compile(r"\b(?:print|printf)\b")

_READ_COMMANDS = {
    "cat",
    "cmp",
    "cut",
    "diff",
    "file",
    "head",
    "less",
    "ls",
    "more",
    "sort",
    "stat",
    "tac",
    "tail",
    "uniq",
    "wc",
}
_WRITE_COMMANDS = {
    "chmod",
    "chown",
    "chgrp",
    "mkdir",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}
_SHELL_COMMANDS = {"bash", "dash", "sh", "zsh"}
_BASH_FILE_OPTIONS = {"--init-file", "--rcfile"}
_SAFE_DEVICE_PATHS = {
    Path("/dev/null"),
    Path("/dev/stdin"),
    Path("/dev/stdout"),
    Path("/dev/stderr"),
}


def _has_active_unmodeled_expansion(raw_word: str) -> bool:
    """Return whether Bash may expand syntax omitted from the bashlex AST."""
    brace_has_expander: list[bool] = []
    bracket_start: int | None = None
    in_single_quote = False
    in_double_quote = False
    escaped = False
    index = 0

    while index < len(raw_word):
        character = raw_word[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
            index += 1
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            index += 1
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            index += 1
            continue
        if in_single_quote or in_double_quote:
            index += 1
            continue

        if character in {"*", "?"}:
            return True
        if character == "[" and bracket_start is None:
            bracket_start = index
        elif character == "]" and bracket_start is not None:
            if index > bracket_start + 1:
                return True
        elif character == "{":
            brace_has_expander.append(False)
        elif character == "," and brace_has_expander:
            brace_has_expander[-1] = True
        elif (
            character == "."
            and index + 1 < len(raw_word)
            and raw_word[index + 1] == "."
            and brace_has_expander
        ):
            brace_has_expander[-1] = True
            index += 1
        elif character == "}" and brace_has_expander:
            if brace_has_expander.pop():
                return True
        index += 1

    return False


def _mark_unmodeled_expansions(nodes: Sequence[Any], source: str) -> None:
    """Attach source-aware expansion metadata missing from the Bash AST."""
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if getattr(node, "kind", None) == "word":
            start, end = cast(tuple[int, int], node.pos)
            node.xagent_has_unmodeled_expansion = _has_active_unmodeled_expansion(
                source[start:end]
            )
        for value in vars(node).values():
            if getattr(value, "kind", None) is not None:
                pending.append(value)
            elif isinstance(value, (list, tuple)):
                pending.extend(
                    item for item in value if getattr(item, "kind", None) is not None
                )


class CommandPolicyViolation(ValueError):
    """The guard cannot safely authorize a command under the active policy."""


class CommandPathViolation(CommandPolicyViolation):
    """A statically identified command path falls outside its allowed roots."""

    def __init__(self, *, access: PathAccess, path: Path) -> None:
        self.access = access
        self.path = path
        super().__init__(f"{path} is outside allowed {access} paths")


class _CommandValue(str):
    """A shell word plus whether bashlex proved it has no runtime expansion."""

    is_static: bool

    def __new__(cls, value: str, *, is_static: bool = True) -> _CommandValue:
        instance = super().__new__(cls, value)
        instance.is_static = is_static
        return instance

    def __getitem__(self, key: SupportsIndex | slice) -> str:
        value = super().__getitem__(key)
        if isinstance(key, slice):
            return _CommandValue(value, is_static=self.is_static)
        return value

    def split(
        self,
        sep: str | None = None,
        maxsplit: SupportsIndex = -1,
    ) -> list[str]:
        # Runtime values retain the marker while the public return type keeps
        # this str subclass substitutable for normal parser code.
        return [
            _CommandValue(value, is_static=self.is_static)
            for value in super().split(sep, maxsplit)
        ]


@dataclass(frozen=True)
class _ShellState:
    cwd: Path
    directory_stack: tuple[Path, ...] = ()
    previous_cwd: Path | None = None
    script_depth: int = 0
    wrapper_depth: int = 0
    cwd_is_conditional: bool = False


@dataclass(frozen=True)
class _FindExecClause:
    marker: Literal["-exec", "-execdir"]
    command: tuple[str, ...]


TarMode = Literal[
    "create",
    "append",
    "update",
    "concatenate",
    "delete",
    "extract",
    "list",
    "compare",
]


@dataclass(frozen=True)
class _ScriptCommandGrammar:
    language: Literal["sed", "awk"]
    expression_long_option: str
    value_options: frozenset[str] = frozenset()
    ignores_assignment_arguments: bool = False


@dataclass(frozen=True)
class _CommandWrapperGrammar:
    flag_options: frozenset[str] = frozenset()
    value_options: frozenset[str] = frozenset()
    attached_short_value_options: frozenset[str] = frozenset()
    leading_scalars: int = 0
    allows_assignments: bool = False
    cwd_options: frozenset[str] = frozenset()
    terminal_options: frozenset[str] = frozenset()
    rejected_options: frozenset[str] = frozenset()
    propagates_state: bool = False


_SED_GRAMMAR = _ScriptCommandGrammar(
    language="sed",
    expression_long_option="--expression",
)
_AWK_GRAMMAR = _ScriptCommandGrammar(
    language="awk",
    expression_long_option="--source",
    value_options=frozenset({"-F", "-v"}),
    ignores_assignment_arguments=True,
)
_COMMAND_WRAPPER_GRAMMARS = {
    "command": _CommandWrapperGrammar(
        flag_options=frozenset({"-p"}),
        terminal_options=frozenset({"-v", "-V"}),
        propagates_state=True,
    ),
    "env": _CommandWrapperGrammar(
        flag_options=frozenset({"-", "-0", "-i", "--ignore-environment", "--null"}),
        value_options=frozenset({"-a", "--argv0", "-C", "--chdir", "-u", "--unset"}),
        attached_short_value_options=frozenset({"-a", "-C", "-u"}),
        allows_assignments=True,
        cwd_options=frozenset({"-C", "--chdir"}),
        terminal_options=frozenset({"--help", "--version"}),
        rejected_options=frozenset({"-S", "--split-string"}),
    ),
    "exec": _CommandWrapperGrammar(
        flag_options=frozenset({"-c", "-l"}),
        value_options=frozenset({"-a"}),
        attached_short_value_options=frozenset({"-a"}),
    ),
    "ionice": _CommandWrapperGrammar(
        flag_options=frozenset({"-t", "--ignore"}),
        value_options=frozenset(
            {
                "-c",
                "--class",
                "-n",
                "--classdata",
                "-P",
                "--pgid",
                "-p",
                "--pid",
                "-u",
                "--uid",
            }
        ),
        attached_short_value_options=frozenset({"-c", "-n", "-P", "-p", "-u"}),
        terminal_options=frozenset(
            {
                "-P",
                "--pgid",
                "-p",
                "--pid",
                "-u",
                "--uid",
                "-h",
                "--help",
                "-V",
                "--version",
            }
        ),
    ),
    "nice": _CommandWrapperGrammar(
        value_options=frozenset({"-n", "--adjustment"}),
        attached_short_value_options=frozenset({"-n"}),
        terminal_options=frozenset({"--help", "--version"}),
    ),
    "nohup": _CommandWrapperGrammar(
        terminal_options=frozenset({"--help", "--version"}),
    ),
    "setsid": _CommandWrapperGrammar(
        flag_options=frozenset({"-c", "--ctty", "-f", "--fork", "-w", "--wait"}),
        terminal_options=frozenset({"-h", "--help", "-V", "--version"}),
    ),
    "stdbuf": _CommandWrapperGrammar(
        value_options=frozenset({"-e", "--error", "-i", "--input", "-o", "--output"}),
        attached_short_value_options=frozenset({"-e", "-i", "-o"}),
        terminal_options=frozenset({"--help", "--version"}),
    ),
    "timeout": _CommandWrapperGrammar(
        flag_options=frozenset(
            {"-f", "--foreground", "--preserve-status", "-v", "--verbose"}
        ),
        value_options=frozenset({"-k", "--kill-after", "-s", "--signal"}),
        attached_short_value_options=frozenset({"-k", "-s"}),
        leading_scalars=1,
        terminal_options=frozenset({"--help", "--version"}),
    ),
}
TarEventKind = Literal[
    "archive",
    "directory",
    "files_from",
    "exclude_from",
    "incremental",
    "add_file",
    "positional",
    "dangerous",
    "absolute_names",
]


@dataclass(frozen=True)
class _TarEvent:
    kind: TarEventKind
    value: str | None = None


@dataclass(frozen=True)
class _TarInvocation:
    mode: TarMode
    events: tuple[_TarEvent, ...]
    remove_files: bool = False


class WorkspaceCommandPathGuard:
    """Validate common shell file operands against one task workspace."""

    def __init__(
        self,
        workspace: TaskWorkspace,
        *,
        _path_access_observer: Callable[[str, PathAccess], None] | None = None,
    ) -> None:
        self._workspace = workspace
        self._initial_cwd = workspace.resolve_path("").resolve()
        self._path_access_observer = _path_access_observer

    @staticmethod
    def _parse_shell(command: str) -> list[Any]:
        try:
            nodes = cast(list[Any], bashlex.parse(command))
        except (bashlex.errors.ParsingError, NotImplementedError, ValueError) as exc:
            logger.debug("Command path guard rejected unparsed shell input")
            raise CommandPolicyViolation("cannot safely parse shell command") from exc
        _mark_unmodeled_expansions(nodes, command)
        return nodes

    def validate(self, command: str) -> None:
        """Reject unsupported syntax and unsafe paths in ``command``."""
        nodes = self._parse_shell(command)

        state = _ShellState(cwd=self._initial_cwd)
        for node in nodes:
            state = self._validate_node(node, state)

    def validate_argv(self, argv: Sequence[str]) -> None:
        """Reject out-of-policy paths in literal, pre-tokenized arguments.

        No shell interprets this form, so shell expansion syntax remains literal.
        """
        if not argv:
            return
        command_name = os.path.basename(argv[0])
        self._validate_command_values(
            command_name,
            argv[1:],
            _ShellState(cwd=self._initial_cwd),
        )

    def _validate_node(self, node: Any, state: _ShellState) -> _ShellState:
        kind = getattr(node, "kind", None)
        if kind == "list":
            current = state
            state_changed = False
            for part in node.parts:
                part_kind = getattr(part, "kind", None)
                if part_kind in {"operator", "pipe"}:
                    operator = getattr(part, "op", None)
                    if operator == "&&":
                        if state_changed:
                            current = replace(current, cwd_is_conditional=True)
                    elif state_changed or current.cwd_is_conditional:
                        raise CommandPolicyViolation(
                            "cannot safely resolve directory state across shell operator"
                        )
                    state_changed = False
                    continue
                next_state = self._validate_node(part, current)
                state_changed = next_state != current
                current = next_state
            return current

        if kind == "pipeline":
            for part in node.parts:
                if getattr(part, "kind", None) != "pipe":
                    self._validate_node(part, state)
            return state

        if kind == "compound":
            current = state
            reserved_words = [
                getattr(part, "word", None)
                for part in getattr(node, "list", ())
                if getattr(part, "kind", None) == "reservedword"
            ]
            for part in getattr(node, "list", ()):
                current = self._validate_node(part, current)
            for redirect in getattr(node, "redirects", ()):
                self._validate_redirect(redirect, current.cwd)
            return state if reserved_words[:1] == ["("] else current

        if kind == "command":
            return self._validate_command(node, state)

        self._validate_nested_nodes(node, state)
        return state

    def _validate_command(self, node: Any, state: _ShellState) -> _ShellState:
        words: list[Any] = []
        for part in node.parts:
            kind = getattr(part, "kind", None)
            if kind == "redirect":
                self._validate_redirect(part, state.cwd)
            elif kind == "word":
                words.append(part)
                self._validate_nested_nodes(part, state)
            else:
                self._validate_nested_nodes(part, state)

        if not words:
            return state

        command_word = self._command_value(words[0])
        if not command_word.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        command_name = os.path.basename(command_word)
        args = self._command_values(words[1:])

        return self._validate_command_values(command_name, args, state)

    def _validate_command_values(
        self,
        command_name: str,
        args: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        if command_name in {"cd", "pushd", "popd"}:
            return self._change_directory(command_name, args, state)

        if command_name in _READ_COMMANDS:
            self._check_read_command(command_name, args, state.cwd)
        elif command_name in _WRITE_COMMANDS:
            self._check_operands(args, state.cwd, "write")
        elif command_name == "grep":
            self._check_grep(args, state.cwd)
        elif command_name == "sed":
            self._check_script_command(
                _SED_GRAMMAR,
                args,
                state.cwd,
                file_access="write" if self._sed_requests_in_place(args) else "read",
            )
        elif command_name == "awk":
            self._check_script_command(
                _AWK_GRAMMAR, args, state.cwd, file_access="read"
            )
        elif command_name == "base64":
            self._check_base64(args, state.cwd)
        elif command_name == "dd":
            self._check_dd(args, state.cwd)
        elif command_name == "tar":
            self._check_tar(args, state.cwd)
        elif command_name == "cp":
            self._check_copy(args, state.cwd)
        elif command_name in {"mv", "ln"}:
            self._check_move_or_link(args, state.cwd)
        elif command_name == "find":
            self._check_find(args, state)
        elif command_name in _SHELL_COMMANDS:
            self._check_nested_shell(command_name, args, state)
        elif command_name in {".", "source"}:
            return self._check_sourced_shell(args, state)
        elif command_name == "xargs":
            self._check_xargs(args, state)
        elif command_name in _COMMAND_WRAPPER_GRAMMARS:
            return self._check_command_wrapper(command_name, args, state)

        return state

    def _validate_redirect(self, node: Any, cwd: Path) -> None:
        redirect_type = getattr(node, "type", "")
        if redirect_type in {"<<", "<<<"}:
            return
        output = getattr(node, "output", None)
        # bashlex emits descriptor-duplication targets such as 2>&1 as integers.
        if output is None or getattr(output, "kind", None) != "word":
            return
        raw_path = self._command_value(output)
        access: PathAccess = "read" if redirect_type == "<" else "write"
        self._check_path(raw_path, cwd, access)

    def _change_directory(
        self,
        command_name: str,
        values: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        operands = self._operands(values)
        if command_name == "cd":
            if len(operands) > 1:
                raise CommandPolicyViolation("cannot safely resolve cd arguments")
            target_word = operands[0] if operands else str(Path.home())
            if target_word == "-":
                if state.previous_cwd is None:
                    raise CommandPolicyViolation(
                        "cannot resolve cd - without a previous directory"
                    )
                target = state.previous_cwd
            else:
                target = self._check_path(target_word, state.cwd, "read")
            return replace(state, cwd=target, previous_cwd=state.cwd)

        if any(value.startswith("-") and value != "--" for value in values):
            raise CommandPolicyViolation(
                f"cannot safely resolve {command_name} options"
            )

        if command_name == "pushd":
            if len(operands) > 1:
                raise CommandPolicyViolation("cannot safely resolve pushd arguments")
            if operands:
                if operands[0] == "-":
                    raise CommandPolicyViolation("cannot safely resolve pushd target")
                target = self._check_path(operands[0], state.cwd, "read")
                stack = (state.cwd, *state.directory_stack)
            else:
                if not state.directory_stack:
                    raise CommandPolicyViolation(
                        "cannot resolve pushd without a directory stack"
                    )
                target = state.directory_stack[0]
                stack = (state.cwd, *state.directory_stack[1:])
            return replace(
                state,
                cwd=target,
                directory_stack=stack,
                previous_cwd=state.cwd,
            )

        if operands:
            raise CommandPolicyViolation("cannot safely resolve popd arguments")
        if not state.directory_stack:
            raise CommandPolicyViolation(
                "cannot resolve popd without a directory stack"
            )
        return replace(
            state,
            cwd=state.directory_stack[0],
            directory_stack=state.directory_stack[1:],
            previous_cwd=state.cwd,
        )

    def _check_command_wrapper(
        self,
        command_name: str,
        values: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        if state.wrapper_depth >= _MAX_COMMAND_WRAPPER_DEPTH:
            raise CommandPolicyViolation("wrapper nesting depth exceeded")
        grammar = _COMMAND_WRAPPER_GRAMMARS[command_name]
        self._reject_dynamic_values(f"{command_name} arguments", values)
        index = 0
        options_done = False
        terminal_mode = False
        nested_state = replace(state, wrapper_depth=state.wrapper_depth + 1)

        while index < len(values):
            value = values[index]
            if not options_done and value == "--":
                options_done = True
                index += 1
                continue
            if options_done or not value.startswith("-"):
                break
            if value in grammar.rejected_options or any(
                value.startswith(f"{option}=")
                for option in grammar.rejected_options
                if option.startswith("--")
            ):
                raise CommandPolicyViolation(
                    f"cannot safely inspect {command_name} option {value}"
                )
            if value in grammar.flag_options:
                index += 1
                continue
            if value in grammar.terminal_options and value not in grammar.value_options:
                terminal_mode = True
                index += 1
                continue
            if value in grammar.value_options:
                if index + 1 >= len(values):
                    raise CommandPolicyViolation(
                        f"missing {command_name} argument for {value}"
                    )
                argument = values[index + 1]
                if value in grammar.cwd_options:
                    target = self._check_path(argument, nested_state.cwd, "read")
                    nested_state = replace(nested_state, cwd=target)
                if value in grammar.terminal_options:
                    terminal_mode = True
                index += 2
                continue

            matching_long = next(
                (
                    option
                    for option in grammar.value_options
                    if option.startswith("--") and value.startswith(f"{option}=")
                ),
                None,
            )
            if matching_long is not None:
                argument = self._derived_value(value, value.split("=", 1)[1])
                if matching_long in grammar.cwd_options:
                    target = self._check_path(argument, nested_state.cwd, "read")
                    nested_state = replace(nested_state, cwd=target)
                if matching_long in grammar.terminal_options:
                    terminal_mode = True
                index += 1
                continue

            matching_short = next(
                (
                    option
                    for option in grammar.attached_short_value_options
                    if value.startswith(option) and len(value) > len(option)
                ),
                None,
            )
            if matching_short is not None:
                argument = self._derived_value(value, value[len(matching_short) :])
                if matching_short in grammar.cwd_options:
                    target = self._check_path(argument, nested_state.cwd, "read")
                    nested_state = replace(nested_state, cwd=target)
                if matching_short in grammar.terminal_options:
                    terminal_mode = True
                index += 1
                continue

            raise CommandPolicyViolation(
                f"cannot safely inspect {command_name} option {value}"
            )

        for _ in range(grammar.leading_scalars):
            if index >= len(values):
                return state
            index += 1

        if grammar.allows_assignments:
            while index < len(values) and "=" in values[index]:
                index += 1

        if terminal_mode or index >= len(values):
            return state

        command_word = values[index]
        if isinstance(command_word, _CommandValue) and not command_word.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        nested_name = os.path.basename(command_word)
        # Wrappers spawn a child command: validate its paths against the adjusted
        # child cwd, but never propagate child shell directory state to the parent.
        validated_state = self._validate_command_values(
            nested_name, values[index + 1 :], nested_state
        )
        if not grammar.propagates_state:
            return state
        return replace(validated_state, wrapper_depth=state.wrapper_depth)

    def _check_operands(
        self,
        values: Sequence[str],
        cwd: Path,
        access: PathAccess,
    ) -> None:
        for raw_path in self._operands(values):
            self._check_path(raw_path, cwd, access)

    def _check_read_command(
        self,
        command_name: str,
        values: Sequence[str],
        cwd: Path,
    ) -> None:
        if command_name == "cut":
            self._check_cut(values, cwd)
            return
        if command_name == "sort":
            self._check_sort(values, cwd)
            return
        if command_name == "uniq":
            self._check_uniq(values, cwd)
            return
        if command_name == "diff":
            self._check_diff(values, cwd)
            return
        if command_name == "tac":
            self._check_tac(values, cwd)
            return

        if command_name == "file":
            values = self._check_read_path_options(
                values,
                cwd,
                short_options={"-f", "-m"},
                long_options={"--files-from", "--magic-file"},
            )
        elif command_name == "wc":
            values = self._check_read_path_options(
                values,
                cwd,
                short_options=set(),
                long_options={"--files0-from"},
            )
        self._check_operands(values, cwd, "read")

    def _check_tac(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("tac arguments", values)
        operands: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                operands.extend(values[index + 1 :])
                break
            if value in {"-s", "--separator"}:
                index += 2
                continue
            if value.startswith("--separator=") or (
                value.startswith("-s") and len(value) > 2
            ):
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            operands.append(value)
            index += 1
        for raw_path in operands:
            self._check_path(raw_path, cwd, "read")

    def _check_cut(self, values: Sequence[str], cwd: Path) -> None:
        options_with_value = {
            "-b",
            "--bytes",
            "-c",
            "--characters",
            "-d",
            "--delimiter",
            "-f",
            "--fields",
            "--output-delimiter",
        }
        attached_short = ("-b", "-c", "-d", "-f")
        operands: list[str] = []
        options_done = False
        index = 0
        while index < len(values):
            value = values[index]
            if not options_done and value == "--":
                options_done = True
                index += 1
                continue
            if not options_done and value in options_with_value:
                index += 2
                continue
            if not options_done and value.startswith("--") and "=" in value:
                index += 1
                continue
            if not options_done and value.startswith(attached_short) and len(value) > 2:
                index += 1
                continue
            if not options_done and value.startswith("-") and value != "-":
                index += 1
                continue
            operands.append(value)
            index += 1
        for raw_path in operands:
            self._check_path(raw_path, cwd, "read")

    def _check_sort(self, values: Sequence[str], cwd: Path) -> None:
        positionals: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            if value in {"-o", "--output", "-T", "--temporary-directory"}:
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "write")
                index += 2
                continue
            if value.startswith("--output=") or value.startswith(
                "--temporary-directory="
            ):
                self._check_path(value.split("=", 1)[1], cwd, "write")
                index += 1
                continue
            if value == "--files0-from":
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "read")
                index += 2
                continue
            if value.startswith("--files0-from="):
                self._check_path(value.split("=", 1)[1], cwd, "read")
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            positionals.append(value)
            index += 1
        for raw_path in positionals:
            self._check_path(raw_path, cwd, "read")

    def _check_uniq(self, values: Sequence[str], cwd: Path) -> None:
        options_with_value = {
            "-f",
            "--skip-fields",
            "-s",
            "--skip-chars",
            "-w",
            "--check-chars",
        }
        operands: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                operands.extend(values[index + 1 :])
                break
            if value in options_with_value:
                index += 2
                continue
            if value.startswith("--") and "=" in value:
                index += 1
                continue
            if value.startswith(("-f", "-s", "-w")) and len(value) > 2:
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            operands.append(value)
            index += 1
        if operands:
            self._check_path(operands[0], cwd, "read")
        if len(operands) > 1:
            self._check_path(operands[1], cwd, "write")

    def _check_diff(self, values: Sequence[str], cwd: Path) -> None:
        positionals: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            if value == "--output":
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "write")
                index += 2
                continue
            if value.startswith("--output="):
                self._check_path(value.split("=", 1)[1], cwd, "write")
                index += 1
                continue
            if value in {"--from-file", "--to-file"}:
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "read")
                index += 2
                continue
            if value.startswith(("--from-file=", "--to-file=")):
                self._check_path(value.split("=", 1)[1], cwd, "read")
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            positionals.append(value)
            index += 1
        for raw_path in positionals:
            self._check_path(raw_path, cwd, "read")

    def _check_read_path_options(
        self,
        values: Sequence[str],
        cwd: Path,
        *,
        short_options: set[str],
        long_options: set[str],
    ) -> list[str]:
        remaining: list[str] = []
        path_options = short_options | long_options
        index = 0
        while index < len(values):
            value = values[index]
            if value in path_options:
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "read")
                index += 2
                continue
            matching_long = next(
                (option for option in long_options if value.startswith(f"{option}=")),
                None,
            )
            if matching_long is not None:
                self._check_path(value.split("=", 1)[1], cwd, "read")
                index += 1
                continue
            matching_short = next(
                (
                    option
                    for option in short_options
                    if value.startswith(option) and len(value) > len(option)
                ),
                None,
            )
            if matching_short is not None:
                self._check_path(value[len(matching_short) :], cwd, "read")
                index += 1
                continue
            remaining.append(value)
            index += 1
        return remaining

    def _check_grep(self, values: Sequence[str], cwd: Path) -> None:
        positionals: list[str] = []
        explicit_pattern = False
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            if value in {"-e", "--regexp"}:
                explicit_pattern = True
                index += 2
                continue
            if value.startswith("-e") or value.startswith("--regexp="):
                explicit_pattern = True
                index += 1
                continue
            if value in {"-f", "--file"}:
                explicit_pattern = True
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "read")
                index += 2
                continue
            if value.startswith("-f") or value.startswith("--file="):
                explicit_pattern = True
                pattern_file = value.split("=", 1)[1] if "=" in value else value[2:]
                if pattern_file:
                    self._check_path(pattern_file, cwd, "read")
                index += 1
                continue
            if value in {"--exclude-from"}:
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "read")
                index += 2
                continue
            if value.startswith("--exclude-from="):
                self._check_path(value.split("=", 1)[1], cwd, "read")
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            positionals.append(value)
            index += 1

        file_operands = positionals if explicit_pattern else positionals[1:]
        for raw_path in file_operands:
            self._check_path(raw_path, cwd, "read")

    def _check_script_command(
        self,
        grammar: _ScriptCommandGrammar,
        values: Sequence[str],
        cwd: Path,
        *,
        file_access: PathAccess,
    ) -> None:
        positionals: list[str] = []
        explicit_script = False
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            if value in {"-f", "--file"}:
                explicit_script = True
                if index + 1 < len(values):
                    self._inspect_script_file(grammar.language, values[index + 1], cwd)
                index += 2
                continue
            if value.startswith("--file="):
                explicit_script = True
                self._inspect_script_file(
                    grammar.language,
                    value.split("=", 1)[1],
                    cwd,
                )
                index += 1
                continue
            if value.startswith("-f") and len(value) > 2:
                explicit_script = True
                self._inspect_script_file(grammar.language, value[2:], cwd)
                index += 1
                continue
            if value in {"-e", grammar.expression_long_option}:
                explicit_script = True
                if index + 1 < len(values):
                    self._reject_embedded_io(grammar.language, values[index + 1])
                index += 2
                continue
            if value.startswith("-e") or value.startswith(
                f"{grammar.expression_long_option}="
            ):
                explicit_script = True
                script = value.split("=", 1)[1] if "=" in value else value[2:]
                self._reject_embedded_io(grammar.language, script)
                index += 1
                continue
            if value in grammar.value_options:
                index += 2
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            if not grammar.ignores_assignment_arguments or "=" not in value:
                positionals.append(value)
            index += 1

        if not explicit_script and positionals:
            self._reject_embedded_io(grammar.language, positionals[0])
        file_operands = positionals if explicit_script else positionals[1:]
        for raw_path in file_operands:
            self._check_path(raw_path, cwd, file_access)

    @staticmethod
    def _sed_requests_in_place(values: Sequence[str]) -> bool:
        return any(
            value == "-i" or value.startswith("-i") or value.startswith("--in-place")
            for value in values
        )

    def _check_base64(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("base64 arguments", values)
        inputs: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                inputs.extend(values[index + 1 :])
                break
            if value in {"--input", "--output"}:
                if index + 1 < len(values):
                    self._check_path(
                        values[index + 1],
                        cwd,
                        "read" if value == "--input" else "write",
                    )
                index += 2
                continue
            if value.startswith(("--input=", "--output=")):
                self._check_path(
                    value.split("=", 1)[1],
                    cwd,
                    "read" if value.startswith("--input=") else "write",
                )
                index += 1
                continue
            if value in {"--break", "--wrap"}:
                index += 2
                continue
            if value.startswith(("--break=", "--wrap=")):
                index += 1
                continue
            if value.startswith("-") and not value.startswith("--") and value != "-":
                index = self._check_base64_short_options(values, index, cwd)
                continue
            inputs.append(value)
            index += 1
        for raw_path in inputs:
            self._check_path(raw_path, cwd, "read")

    def _check_base64_short_options(
        self,
        values: Sequence[str],
        index: int,
        cwd: Path,
    ) -> int:
        value = values[index]
        options = value[1:]
        cursor = 0
        while cursor < len(options):
            option = options[cursor]
            if option in {"i", "o", "b", "w"}:
                attached = options[cursor + 1 :]
                argument: str | None = None
                next_index = index + 1
                if attached:
                    argument = self._derived_value(value, attached)
                elif next_index < len(values):
                    argument = values[next_index]
                    next_index += 1
                if argument is not None and option in {"i", "o"}:
                    self._check_path(
                        argument,
                        cwd,
                        "read" if option == "i" else "write",
                    )
                return next_index
            cursor += 1
        return index + 1

    def _check_dd(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("dd arguments", values)
        for value in values:
            if value.startswith("if="):
                self._check_path(value.split("=", 1)[1], cwd, "read")
            elif value.startswith("of="):
                self._check_path(value.split("=", 1)[1], cwd, "write")

    def _check_tar(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("tar arguments", values)
        invocation = self._parse_tar(values)
        archive_access: PathAccess = (
            "write"
            if invocation.mode
            in {"create", "append", "update", "concatenate", "delete"}
            else "read"
        )
        source_access: PathAccess = "write" if invocation.remove_files else "read"
        active_cwd = cwd

        for event in invocation.events:
            if event.kind == "dangerous":
                raise CommandPolicyViolation(
                    "tar command hooks cannot be inspected safely"
                )
            if event.kind == "absolute_names":
                if invocation.mode == "extract":
                    raise CommandPolicyViolation(
                        "tar absolute extraction paths cannot be inspected safely"
                    )
                continue
            if event.value is None:
                continue

            if event.kind == "archive":
                self._check_tar_archive_path(event.value, cwd, archive_access)
            elif event.kind == "directory":
                active_cwd = self._check_path(
                    event.value,
                    active_cwd,
                    "write" if invocation.mode == "extract" else "read",
                )
            elif event.kind == "files_from":
                raise CommandPolicyViolation(
                    "tar file lists cannot be inspected safely"
                )
            elif event.kind == "exclude_from":
                self._check_path(event.value, cwd, "read")
            elif event.kind == "incremental":
                # Snapshot files can be updated even when the archive is read.
                self._check_path(event.value, cwd, "write")
            elif event.kind == "add_file":
                self._check_path(event.value, active_cwd, source_access)
            elif event.kind == "positional" and invocation.mode in {
                "create",
                "append",
                "update",
                "concatenate",
                "compare",
            }:
                raw_path = (
                    event.value[1:] if event.value.startswith("@") else event.value
                )
                self._check_path(raw_path, active_cwd, source_access)

    def _parse_tar(self, values: Sequence[str]) -> _TarInvocation:
        long_modes: dict[str, TarMode] = {
            "--create": "create",
            "--append": "append",
            "--update": "update",
            "--concatenate": "concatenate",
            "--catenate": "concatenate",
            "--delete": "delete",
            "--extract": "extract",
            "--get": "extract",
            "--list": "list",
            "--compare": "compare",
            "--diff": "compare",
        }
        modes: list[TarMode] = []
        events: list[_TarEvent] = []
        remove_files = False
        options_done = False
        index = 0

        # Traditional tar syntax omits the leading dash (for example, ``cf``).
        if (
            values
            and re.fullmatch(r"[A-Za-z]+", values[0])
            and any(option in "cruAxtd" for option in values[0])
        ):
            index, short_modes, short_events = self._parse_tar_short_events(
                values,
                0,
                values[0],
            )
            modes.extend(short_modes)
            events.extend(short_events)

        while index < len(values):
            value = values[index]
            if not options_done and value == "--":
                options_done = True
                index += 1
                continue
            if options_done or not value.startswith("-") or value == "-":
                events.append(_TarEvent("positional", value))
                index += 1
                continue
            if value in long_modes:
                modes.append(long_modes[value])
                index += 1
                continue
            if value == "--remove-files":
                remove_files = True
                index += 1
                continue
            if value in {"--to-stdout", "-O"}:
                index += 1
                continue
            if value in {"--absolute-names", "--absolute-paths"}:
                events.append(_TarEvent("absolute_names"))
                index += 1
                continue
            if value.startswith("--"):
                index, event = self._parse_tar_long_event(values, index)
                if event is not None:
                    events.append(event)
                continue

            index, short_modes, short_events = self._parse_tar_short_events(
                values,
                index,
                value[1:],
            )
            modes.extend(short_modes)
            events.extend(short_events)

        unique_modes = set(modes)
        if len(unique_modes) != 1:
            raise CommandPolicyViolation("cannot safely resolve tar operation")
        return _TarInvocation(
            mode=unique_modes.pop(),
            events=tuple(events),
            remove_files=remove_files,
        )

    def _parse_tar_long_event(
        self,
        values: Sequence[str],
        index: int,
    ) -> tuple[int, _TarEvent | None]:
        value = values[index]
        path_options: dict[str, TarEventKind] = {
            "--file": "archive",
            "--directory": "directory",
            "--files-from": "files_from",
            "--exclude-from": "exclude_from",
            "--listed-incremental": "incremental",
            "--add-file": "add_file",
        }
        dangerous_options = {
            "--use-compress-program",
            "--to-command",
            "--checkpoint-action",
            "--info-script",
            "--new-volume-script",
            "--rsh-command",
        }
        option, separator, attached = value.partition("=")
        kind = path_options.get(option)
        is_dangerous = option in dangerous_options
        if kind is None and not is_dangerous:
            return index + 1, None

        if separator:
            argument: str | None = self._derived_value(value, attached)
            next_index = index + 1
        elif index + 1 < len(values):
            argument = values[index + 1]
            next_index = index + 2
        else:
            raise CommandPolicyViolation(f"missing tar argument for {option}")

        if is_dangerous:
            return next_index, _TarEvent("dangerous", argument)
        return next_index, _TarEvent(cast(TarEventKind, kind), argument)

    def _parse_tar_short_events(
        self,
        values: Sequence[str],
        index: int,
        options: str,
    ) -> tuple[int, list[TarMode], list[_TarEvent]]:
        short_modes: dict[str, TarMode] = {
            "c": "create",
            "r": "append",
            "u": "update",
            "A": "concatenate",
            "x": "extract",
            "t": "list",
            "d": "compare",
        }
        argument_events: dict[str, TarEventKind | None] = {
            "f": "archive",
            "C": "directory",
            "T": "files_from",
            "X": "exclude_from",
            "g": "incremental",
            "I": "dangerous",
            "F": "dangerous",
            "b": None,
        }
        modes: list[TarMode] = []
        events: list[_TarEvent] = []
        cursor = 0
        next_index = index + 1

        while cursor < len(options):
            option = options[cursor]
            if option in short_modes:
                modes.append(short_modes[option])
                cursor += 1
                continue
            if option == "P":
                events.append(_TarEvent("absolute_names"))
                cursor += 1
                continue
            if option == "O":
                cursor += 1
                continue
            if option not in argument_events:
                cursor += 1
                continue

            argument: str
            attached = options[cursor + 1 :]
            if attached:
                argument = self._derived_value(values[index], attached)
            elif next_index < len(values):
                argument = values[next_index]
                next_index += 1
            else:
                raise CommandPolicyViolation(f"missing tar argument for -{option}")
            kind = argument_events[option]
            if kind is not None:
                events.append(_TarEvent(kind, argument))
            break

        return next_index, modes, events

    def _check_tar_archive_path(
        self,
        raw_path: str,
        cwd: Path,
        access: PathAccess,
    ) -> None:
        if raw_path != "-" and ":" in raw_path:
            raise CommandPolicyViolation(
                "remote tar archives cannot be inspected safely"
            )
        self._check_path(raw_path, cwd, access)

    def _inspect_script_file(
        self,
        language: Literal["sed", "awk"],
        raw_path: str,
        cwd: Path,
    ) -> None:
        script = self._read_policy_script(raw_path, cwd)
        if script is not None:
            self._reject_embedded_io(language, script)

    def _read_policy_script(self, raw_path: str, cwd: Path) -> str | None:
        script_path = self._check_path(raw_path, cwd, "read")
        if self._path_access_observer is not None:
            return None

        descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(script_path, flags)
            with os.fdopen(descriptor, "rb") as script_file:
                descriptor = None
                metadata = os.fstat(script_file.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > _MAX_INSPECTED_SCRIPT_BYTES
                ):
                    raise CommandPathViolation(access="read", path=script_path)
                raw_script = script_file.read(_MAX_INSPECTED_SCRIPT_BYTES + 1)
            if len(raw_script) > _MAX_INSPECTED_SCRIPT_BYTES:
                raise CommandPathViolation(access="read", path=script_path)
            script = raw_script.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            # A missing script may be created earlier in the same shell string;
            # skipping inspection would allow its embedded file I/O unchecked.
            raise CommandPathViolation(access="read", path=script_path) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return script

    def _inspect_shell_script(
        self,
        raw_path: str,
        state: _ShellState,
        *,
        propagate_state: bool,
    ) -> _ShellState:
        if state.script_depth >= _MAX_INSPECTED_SCRIPT_DEPTH:
            raise CommandPolicyViolation("shell script inspection depth exceeded")
        script = self._read_policy_script(raw_path, state.cwd)
        if script is None:
            return state
        nested_state = replace(state, script_depth=state.script_depth + 1)
        validated_state = self._validate_shell_text(script, nested_state)
        if not propagate_state:
            return state
        return replace(validated_state, script_depth=state.script_depth)

    @staticmethod
    def _reject_embedded_io(language: Literal["sed", "awk"], script: str) -> None:
        if isinstance(script, _CommandValue) and not script.is_static:
            raise CommandPolicyViolation(f"cannot inspect dynamic {language} program")
        if language == "awk":
            unsafe = WorkspaceCommandPathGuard._awk_has_unsafe_io(script)
        else:
            unsafe = WorkspaceCommandPathGuard._sed_has_unsafe_io(script)
        if unsafe:
            raise CommandPathViolation(
                access="write",
                path=Path(f"<{language} embedded file I/O>"),
            )

    @staticmethod
    def _awk_has_unsafe_io(script: str) -> bool:
        if _AWK_ALWAYS_UNSAFE_IO_PATTERN.search(script):
            return True

        for match in _AWK_PRINT_PATTERN.finditer(script):
            index = match.end()
            depth = 0
            quote: str | None = None
            escaped = False
            while index < len(script):
                char = script[index]
                index += 1
                if quote is not None:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = None
                    continue
                if char in {"'", '"'}:
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")" and depth:
                    depth -= 1
                elif char in ";\n" and depth == 0:
                    break
                elif char in {">", "|"} and depth == 0:
                    return True
        return False

    @staticmethod
    def _sed_has_unsafe_io(script: str) -> bool:
        index = 0
        while index < len(script):
            while index < len(script) and script[index] in " \t;\n}":
                index += 1
            if index >= len(script):
                break

            index = WorkspaceCommandPathGuard._skip_sed_addresses(script, index)
            while index < len(script) and script[index] in " \t":
                index += 1
            # BSD sed accepts repeated negation operators, so command discovery
            # must consume the complete prefix before classifying the command.
            while index < len(script) and script[index] == "!":
                index += 1
                while index < len(script) and script[index] in " \t":
                    index += 1
            if index >= len(script):
                raise CommandPolicyViolation("cannot safely inspect sed program")

            command = script[index]
            index += 1
            if command in "rRwWe":
                return True
            if command == "{":
                continue
            if command == "#":
                index = WorkspaceCommandPathGuard._skip_sed_to_boundary(
                    script, index, boundaries="\n"
                )
                continue
            if command == "s":
                index, unsafe = WorkspaceCommandPathGuard._scan_sed_substitution(
                    script, index
                )
                if unsafe:
                    return True
                continue
            if command == "y":
                index = WorkspaceCommandPathGuard._scan_sed_transliteration(
                    script, index
                )
                continue
            if command in "aic":
                index = WorkspaceCommandPathGuard._skip_sed_to_boundary(
                    script, index, boundaries="\n"
                )
                continue
            if command in "bTt:":
                index = WorkspaceCommandPathGuard._skip_sed_to_boundary(
                    script, index, boundaries=";\n"
                )
                continue
            index = WorkspaceCommandPathGuard._skip_sed_to_boundary(
                script, index, boundaries=";\n}"
            )
        return False

    @staticmethod
    def _skip_sed_addresses(script: str, index: int) -> int:
        first_end = WorkspaceCommandPathGuard._skip_sed_address(
            script, index, allow_relative=False
        )
        if first_end is None:
            return index

        cursor = first_end
        while cursor < len(script) and script[cursor] in " \t":
            cursor += 1
        if cursor >= len(script) or script[cursor] != ",":
            return cursor

        cursor += 1
        while cursor < len(script) and script[cursor] in " \t":
            cursor += 1
        second_end = WorkspaceCommandPathGuard._skip_sed_address(
            script, cursor, allow_relative=True
        )
        if second_end is None:
            raise CommandPolicyViolation("cannot safely inspect sed address range")
        return second_end

    @staticmethod
    def _skip_sed_address(
        script: str,
        index: int,
        *,
        allow_relative: bool,
    ) -> int | None:
        if index >= len(script):
            return None
        character = script[index]
        if character.isdigit():
            cursor = index + 1
            while cursor < len(script) and script[cursor].isdigit():
                cursor += 1
            if cursor < len(script) and script[cursor] == "~":
                cursor += 1
                step_start = cursor
                while cursor < len(script) and script[cursor].isdigit():
                    cursor += 1
                if cursor == step_start:
                    raise CommandPolicyViolation(
                        "cannot safely inspect sed step address"
                    )
            return cursor
        if character == "$":
            return index + 1
        if character == "/":
            return WorkspaceCommandPathGuard._scan_sed_delimited(
                script, delimiter_index=index
            )
        if character == "\\" and index + 1 < len(script):
            return WorkspaceCommandPathGuard._scan_sed_delimited(
                script, delimiter_index=index + 1
            )
        if allow_relative and character in {"+", "~"}:
            cursor = index + 1
            number_start = cursor
            while cursor < len(script) and script[cursor].isdigit():
                cursor += 1
            if cursor == number_start:
                raise CommandPolicyViolation(
                    "cannot safely inspect sed relative address"
                )
            return cursor
        return None

    @staticmethod
    def _scan_sed_delimited(
        script: str,
        *,
        delimiter_index: int,
    ) -> int:
        delimiter = script[delimiter_index]
        if delimiter == "\n":
            raise CommandPolicyViolation("cannot safely inspect sed delimiter")
        cursor = delimiter_index + 1
        escaped = False
        while cursor < len(script):
            character = script[cursor]
            cursor += 1
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == delimiter:
                return cursor
            elif character == "\n":
                raise CommandPolicyViolation("cannot safely inspect sed expression")
        raise CommandPolicyViolation(
            "cannot safely inspect unterminated sed expression"
        )

    @staticmethod
    def _scan_sed_substitution(script: str, index: int) -> tuple[int, bool]:
        if index >= len(script):
            raise CommandPolicyViolation("cannot safely inspect sed substitution")
        delimiter = script[index]
        if delimiter.isalnum() or delimiter.isspace() or delimiter == "\\":
            raise CommandPolicyViolation("cannot safely inspect sed substitution")

        cursor = index
        for _ in range(2):
            cursor = WorkspaceCommandPathGuard._scan_sed_delimited(
                script, delimiter_index=cursor
            )
            # The next field uses the same delimiter but starts immediately
            # after the previous closing delimiter.
            cursor -= 1
        cursor += 1

        while cursor < len(script) and script[cursor] not in ";\n}":
            if script[cursor] in {"e", "w"}:
                return cursor, True
            cursor += 1
        return cursor, False

    @staticmethod
    def _scan_sed_transliteration(script: str, index: int) -> int:
        if index >= len(script):
            raise CommandPolicyViolation("cannot safely inspect sed transliteration")
        delimiter = script[index]
        if delimiter.isalnum() or delimiter.isspace() or delimiter == "\\":
            raise CommandPolicyViolation("cannot safely inspect sed transliteration")

        cursor = index
        for _ in range(2):
            cursor = WorkspaceCommandPathGuard._scan_sed_delimited(
                script, delimiter_index=cursor
            )
            cursor -= 1
        return WorkspaceCommandPathGuard._skip_sed_to_boundary(
            script, cursor + 1, boundaries=";\n}"
        )

    @staticmethod
    def _skip_sed_to_boundary(script: str, index: int, *, boundaries: str) -> int:
        escaped = False
        while index < len(script):
            character = script[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character in boundaries:
                return index
            index += 1
        return index

    def _check_copy(self, values: Sequence[str], cwd: Path) -> None:
        target_dir, operands = self._parse_target_directory(values)
        if target_dir is not None:
            for raw_path in operands:
                self._check_path(raw_path, cwd, "read")
            self._check_path(target_dir, cwd, "write")
            return
        if len(operands) < 2:
            return
        for raw_path in operands[:-1]:
            self._check_path(raw_path, cwd, "read")
        self._check_path(operands[-1], cwd, "write")

    def _check_move_or_link(self, values: Sequence[str], cwd: Path) -> None:
        target_dir, operands = self._parse_target_directory(values)
        # A hard link inside the workspace aliases its source inode, so an
        # external read-only source must remain write-protected here.
        for raw_path in operands:
            self._check_path(raw_path, cwd, "write")
        if target_dir is not None:
            self._check_path(target_dir, cwd, "write")

    def _check_find(self, literals: Sequence[str], state: _ShellState) -> None:
        cwd = state.cwd
        roots: list[str] = []
        expression_start = len(literals)
        root_start = 0
        while root_start < len(literals) and literals[root_start] in {"-H", "-L", "-P"}:
            root_start += 1
        for index, value in enumerate(literals[root_start:], start=root_start):
            if value.startswith("-") or value in {"!", "("}:
                expression_start = index
                break
            roots.append(value)

        if not roots:
            roots = ["."]
        expression = literals[expression_start:]
        exec_clauses = self._parse_find_exec_commands(expression)
        access: PathAccess = (
            "write"
            if "-delete" in expression
            or any(
                self._find_exec_command_writes(clause, state) for clause in exec_clauses
            )
            else "read"
        )
        for root in roots:
            self._check_path(root, cwd, access)

        for clause in exec_clauses:
            self._validate_nested_command_words(
                clause.command,
                state,
            )

    @staticmethod
    def _parse_find_exec_commands(
        expression: Sequence[str],
    ) -> list[_FindExecClause]:
        clauses: list[_FindExecClause] = []
        index = 0
        while index < len(expression):
            marker = expression[index]
            if marker not in {"-exec", "-execdir"}:
                index += 1
                continue
            nested: list[str] = []
            index += 1
            while index < len(expression) and expression[index] not in {";", "+"}:
                nested.append(expression[index])
                index += 1
            if nested:
                clauses.append(
                    _FindExecClause(
                        marker=cast(Literal["-exec", "-execdir"], marker),
                        command=tuple(nested),
                    )
                )
            index += 1
        return clauses

    def _find_exec_command_writes(
        self,
        clause: _FindExecClause,
        state: _ShellState,
    ) -> bool:
        command = clause.command
        if not command:
            return False

        writes_from_find_root = False

        def observe_path(raw_path: str, access: PathAccess) -> None:
            nonlocal writes_from_find_root
            if access != "write":
                return
            if "{}" in raw_path or (
                clause.marker == "-execdir" and self._is_relative_file_operand(raw_path)
            ):
                writes_from_find_root = True

        probe = WorkspaceCommandPathGuard(
            self._workspace,
            _path_access_observer=observe_path,
        )
        try:
            probe._validate_nested_command_words(command, state)
        except CommandPolicyViolation:
            # The regular validation pass reports statically unsafe embedded
            # programs. This probe only classifies placeholder path access.
            pass
        return writes_from_find_root

    @staticmethod
    def _is_relative_file_operand(raw_path: str) -> bool:
        return raw_path not in {"", "-", "{}"} and not Path(raw_path).is_absolute()

    def _check_nested_shell(
        self,
        command_name: str,
        literals: Sequence[str],
        state: _ShellState,
    ) -> None:
        command_option_index: int | None = None
        for index, value in enumerate(literals[:-1]):
            if (
                value.startswith("-")
                and not value.startswith("--")
                and "c" in value[1:]
            ):
                command_option_index = index
                break

        file_values = (
            literals
            if command_option_index is None
            else literals[:command_option_index]
        )
        if command_name == "bash":
            file_values, initialization_files = self._extract_bash_file_options(
                file_values
            )
            for initialization_file in initialization_files:
                self._inspect_shell_script(
                    initialization_file,
                    state,
                    propagate_state=False,
                )

        if command_option_index is not None:
            self._validate_shell_text(literals[command_option_index + 1], state)
            return

        script = self._shell_script_operand(file_values)
        if script is None:
            raise CommandPolicyViolation(
                f"cannot inspect {command_name} input without command text or a script"
            )
        self._inspect_shell_script(script, state, propagate_state=False)

    @staticmethod
    def _extract_bash_file_options(
        values: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        remaining: list[str] = []
        initialization_files: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value in _BASH_FILE_OPTIONS:
                if index + 1 < len(values):
                    initialization_files.append(values[index + 1])
                index += 2
                continue
            if any(value.startswith(f"{option}=") for option in _BASH_FILE_OPTIONS):
                initialization_files.append(value.split("=", 1)[1])
                index += 1
                continue
            remaining.append(value)
            index += 1
        return remaining, initialization_files

    def _check_sourced_shell(
        self,
        values: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        operands = self._operands(values)
        if not operands:
            raise CommandPolicyViolation("cannot inspect source without a script")
        return self._inspect_shell_script(
            operands[0],
            state,
            propagate_state=True,
        )

    @staticmethod
    def _shell_script_operand(values: Sequence[str]) -> str | None:
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                return values[index + 1] if index + 1 < len(values) else None
            if value in {"-o", "-O"}:
                index += 2
                continue
            if value == "-s" or (
                value.startswith("-")
                and not value.startswith("--")
                and "s" in value[1:]
            ):
                return None
            if value.startswith("-") and value != "-":
                index += 1
                continue
            return value
        return None

    def _check_xargs(self, values: Sequence[str], state: _ShellState) -> None:
        cwd = state.cwd
        command_start = len(values)
        index = 0
        options_with_value = {
            "-a",
            "--arg-file",
            "-d",
            "--delimiter",
            "-E",
            "--eof",
            "-I",
            "--replace",
            "-L",
            "--max-lines",
            "-n",
            "--max-args",
            "-P",
            "--max-procs",
            "-s",
            "--max-chars",
        }
        attached_prefixes = ("-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s")
        while index < len(values):
            value = values[index]
            if value == "--":
                command_start = index + 1
                break
            if value in options_with_value:
                if value in {"-a", "--arg-file"} and index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "read")
                index += 2
                continue
            if value.startswith("--arg-file="):
                self._check_path(value.split("=", 1)[1], cwd, "read")
                index += 1
                continue
            if value.startswith("--") and "=" in value:
                index += 1
                continue
            if value.startswith(attached_prefixes) and len(value) > 2:
                if value.startswith("-a"):
                    self._check_path(value[2:], cwd, "read")
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            command_start = index
            break

        if command_start < len(values):
            self._validate_nested_command_words(
                values[command_start:],
                state,
            )

    def _validate_nested_command_words(
        self,
        words: Sequence[str],
        state: _ShellState,
    ) -> None:
        if not words:
            return
        if isinstance(words[0], _CommandValue) and not words[0].is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        command_name = os.path.basename(words[0])
        self._validate_command_values(command_name, words[1:], state)

    @staticmethod
    def _parse_target_directory(
        values: Sequence[str],
    ) -> tuple[str | None, list[str]]:
        target_dir: str | None = None
        operands: list[str] = []
        options_done = False
        index = 0
        while index < len(values):
            value = values[index]
            if not options_done and value == "--":
                options_done = True
                index += 1
                continue
            if not options_done and value in {"-t", "--target-directory"}:
                if index + 1 < len(values):
                    target_dir = values[index + 1]
                index += 2
                continue
            if not options_done and value.startswith("--target-directory="):
                target_dir = value.split("=", 1)[1]
                index += 1
                continue
            if not options_done and value.startswith("-") and value != "-":
                index += 1
                continue
            operands.append(value)
            index += 1
        return target_dir, operands

    def _validate_shell_text(
        self,
        command: str,
        state: _ShellState,
    ) -> _ShellState:
        if isinstance(command, _CommandValue) and not command.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic shell command text")
        nodes = self._parse_shell(command)
        current = state
        for node in nodes:
            current = self._validate_node(node, current)
        return current

    def _validate_nested_nodes(self, node: Any, state: _ShellState) -> None:
        for attr_name in ("parts", "command", "list"):
            child = getattr(node, attr_name, None)
            if isinstance(child, list):
                for item in child:
                    if getattr(item, "kind", None) is not None:
                        self._validate_node(item, state)
            elif getattr(child, "kind", None) is not None:
                self._validate_node(child, state)

    @staticmethod
    def _command_value(word: Any) -> _CommandValue:
        parts = getattr(word, "parts", ())
        return _CommandValue(
            str(word.word),
            # Tilde expansion is deterministic at the policy boundary and is
            # resolved by Path.expanduser(); parser parts and source-marked
            # expansions that cannot be resolved here remain dynamic.
            is_static=not getattr(word, "xagent_has_unmodeled_expansion", False)
            and (
                not parts
                or all(getattr(part, "kind", None) == "tilde" for part in parts)
            ),
        )

    def _command_values(self, words: Sequence[Any]) -> list[_CommandValue]:
        return [self._command_value(word) for word in words]

    @staticmethod
    def _derived_value(source: str, value: str) -> _CommandValue:
        return _CommandValue(
            value,
            is_static=(not isinstance(source, _CommandValue) or source.is_static),
        )

    @staticmethod
    def _reject_dynamic_values(context: str, values: Sequence[str]) -> None:
        if any(
            isinstance(value, _CommandValue) and not value.is_static for value in values
        ):
            raise CommandPolicyViolation(f"cannot resolve dynamic {context}")

    @staticmethod
    def _operands(values: Sequence[str]) -> list[str]:
        operands: list[str] = []
        options_done = False
        for value in values:
            if not options_done and value == "--":
                options_done = True
                continue
            if not options_done and value.startswith("-") and value != "-":
                continue
            operands.append(value)
        return operands

    def _check_path(self, raw_path: str, cwd: Path, access: PathAccess) -> Path:
        if isinstance(raw_path, _CommandValue) and not raw_path.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic path operand")

        if self._path_access_observer is not None:
            self._path_access_observer(raw_path, access)
            return cwd

        if raw_path in {"", "-", "{}"}:
            return cwd

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate

        if candidate in _SAFE_DEVICE_PATHS:
            return candidate

        try:
            return self._workspace.resolve_authorized_path(
                candidate,
                base_dir=cwd,
                include_external_dirs=access == "read",
            )
        except ValueError as exc:
            raise CommandPathViolation(access=access, path=candidate) from exc
