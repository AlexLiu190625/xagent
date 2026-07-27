"""Best-effort workspace path guard for shell commands.

This module deliberately provides a cooperative guard, not an operating-system
security boundary. For supported commands, it rejects out-of-scope paths and
path-bearing arguments that cannot be resolved statically. Unknown commands
remain unchanged.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import shutil
import stat
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Sequence,
    SupportsIndex,
    cast,
)

import bashlex

from ...workspace import TaskWorkspace
from .command_policy import (
    CommandPathViolation,
    CommandPolicyViolation,
    PathAccess,
)

logger = logging.getLogger(__name__)

_MAX_INSPECTED_SCRIPT_BYTES = 1024 * 1024
_MAX_INSPECTED_SCRIPT_DEPTH = 8
_MAX_COMMAND_WRAPPER_DEPTH = 32
_MAX_COMMAND_POLICY_INPUT_CHARS = 64 * 1024
_MAX_POSSIBLE_SHELL_STATES = 16
_COMMAND_WRITTEN_PATHS: ContextVar[set[Path] | None] = ContextVar(
    "command_written_paths",
    default=None,
)

_READ_COMMANDS = {"cat"}
_WRITE_COMMANDS = {"rm"}
_SHELL_COMMANDS = {"bash"}
_UNSUPPORTED_SHELL_COMMANDS = {"dash", "sh", "zsh"}
_BASH_FILE_OPTIONS = {"--init-file", "--rcfile"}
_IMPLICIT_SHELL_ENVIRONMENT = {"BASH_ENV", "CDPATH", "ENV", "ZDOTDIR"}
_REJECTED_SHELL_ASSIGNMENTS = {
    *_IMPLICIT_SHELL_ENVIRONMENT,
    "BASHOPTS",
    "PATH",
    "SHELLOPTS",
}
_SAFE_DEVICE_PATHS = {
    Path("/dev/null"),
    Path("/dev/stdin"),
    Path("/dev/stdout"),
    Path("/dev/stderr"),
}
_TRUSTED_EXECUTABLE_ROOTS = tuple(
    path.resolve()
    for path in (
        Path("/bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    )
)
_QUOTED_HEREDOC_PATTERN = re.compile(
    r"(?m)(?P<operator><<-?)(?P<space>[ \t]*)(?P<quote>['\"])"
    r"(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)(?=[ \t]*$)"
)
_TIME_KEYWORD_PATTERN = re.compile(
    r"(?m)(?P<prefix>^|(?<=[;\n])|(?<=&&)|(?<=\|\|))"
    r"(?P<space>[ \t]*)time(?=[ \t]+)"
)
_POLICY_TIME_WRAPPER = "__t_"


def _is_unquoted_on_current_line(source: str, position: int) -> bool:
    """Return whether ``position`` is outside quotes on its physical line."""
    line_start = source.rfind("\n", 0, position) + 1
    in_single_quote = False
    in_double_quote = False
    escaped = False
    for character in source[line_start:position]:
        if escaped:
            escaped = False
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
        elif character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
    return not in_single_quote and not in_double_quote and not escaped


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


def _normalize_policy_shell_source(command: str) -> str:
    """Normalize narrowly supported Bash syntax that bashlex cannot parse.

    The normalized text is used only for policy parsing; the executor receives
    the original command. Replacements preserve source length so bashlex node
    positions still describe the original command. Quoted here-document bodies
    are literal by Bash definition, so masking them cannot hide executable
    expansion or nested commands.
    """
    normalized = list(command)
    for match in _QUOTED_HEREDOC_PATTERN.finditer(command):
        if not _is_unquoted_on_current_line(command, match.start()):
            continue
        quote_start = match.start("quote")
        quote_end = match.end("delimiter")
        normalized[quote_start] = " "
        normalized[quote_end] = " "

        body_start = command.find("\n", match.end())
        if body_start < 0:
            continue
        body_start += 1
        delimiter = match.group("delimiter")
        strip_tabs = match.group("operator") == "<<-"
        cursor = body_start
        while cursor <= len(command):
            line_end = command.find("\n", cursor)
            if line_end < 0:
                line_end = len(command)
            line = command[cursor:line_end]
            comparable = line.lstrip("\t") if strip_tabs else line
            if comparable == delimiter:
                break
            for index in range(cursor, line_end):
                normalized[index] = "x"
            if line_end == len(command):
                break
            cursor = line_end + 1

    source = "".join(normalized)
    return _TIME_KEYWORD_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('space')}{_POLICY_TIME_WRAPPER}"
        ),
        source,
    )


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


_COMMAND_WRAPPER_GRAMMARS = {
    _POLICY_TIME_WRAPPER: _CommandWrapperGrammar(
        flag_options=frozenset({"-p"}),
        propagates_state=True,
    ),
    "builtin": _CommandWrapperGrammar(
        propagates_state=True,
    ),
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
    "sudo": _CommandWrapperGrammar(
        flag_options=frozenset(
            {
                "-A",
                "--askpass",
                "-b",
                "--background",
                "-E",
                "--preserve-env",
                "-H",
                "--set-home",
                "-K",
                "--remove-timestamp",
                "-k",
                "--reset-timestamp",
                "-n",
                "--non-interactive",
                "-P",
                "--preserve-groups",
                "-S",
                "--stdin",
            }
        ),
        value_options=frozenset(
            {
                "-C",
                "--close-from",
                "-D",
                "--chdir",
                "-g",
                "--group",
                "-h",
                "--host",
                "-p",
                "--prompt",
                "-R",
                "--chroot",
                "-r",
                "--role",
                "-T",
                "--command-timeout",
                "-t",
                "--type",
                "-u",
                "--user",
            }
        ),
        attached_short_value_options=frozenset(
            {"-C", "-D", "-g", "-h", "-p", "-R", "-r", "-T", "-t", "-u"}
        ),
        cwd_options=frozenset({"-D", "--chdir"}),
        terminal_options=frozenset({"-V", "--version", "-v", "--validate"}),
        rejected_options=frozenset(
            {
                "-e",
                "--edit",
                "-i",
                "--login",
                "-l",
                "--list",
                "-R",
                "--chroot",
                "-s",
                "--shell",
            }
        ),
    ),
    "time": _CommandWrapperGrammar(
        flag_options=frozenset(
            {"-a", "--append", "-p", "--portability", "-v", "--verbose"}
        ),
        value_options=frozenset({"-f", "--format"}),
        attached_short_value_options=frozenset({"-f"}),
        terminal_options=frozenset({"--help", "-V", "--version"}),
        rejected_options=frozenset({"-o", "--output"}),
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

_CLASSIFIED_EXECUTABLE_COMMANDS = {
    *_READ_COMMANDS,
    *_WRITE_COMMANDS,
    *_SHELL_COMMANDS,
    *_UNSUPPORTED_SHELL_COMMANDS,
    "chroot",
    "xargs",
    *(
        name
        for name in _COMMAND_WRAPPER_GRAMMARS
        if name not in {"builtin", "command", "exec", _POLICY_TIME_WRAPPER}
    ),
}


class WorkspaceCommandPathGuard:
    """Validate Bash language boundaries against one task workspace."""

    def __init__(self, workspace: TaskWorkspace) -> None:
        self._workspace = workspace
        self._initial_cwd = workspace.resolve_path("").resolve()

    @staticmethod
    def _parse_shell(command: str) -> list[Any]:
        if len(command) > _MAX_COMMAND_POLICY_INPUT_CHARS:
            raise CommandPolicyViolation("command is too large to inspect")
        if "\x00" in command:
            raise CommandPolicyViolation("command contains a null byte")
        if not any(
            line.strip() and not line.lstrip().startswith("#")
            for line in command.splitlines()
        ):
            return []
        policy_source = _normalize_policy_shell_source(command)
        try:
            nodes = cast(list[Any], bashlex.parse(policy_source))
        except Exception as exc:
            # bashlex exposes several parser-internal exception types. Normalize
            # all of them at this boundary so malformed input never reaches the
            # executor through a parser-version-specific failure mode.
            logger.debug("Command path guard rejected unparsed shell input")
            raise CommandPolicyViolation("cannot safely parse shell command") from exc
        _mark_unmodeled_expansions(nodes, policy_source)
        return nodes

    def validate(self, command: str) -> None:
        """Reject unsupported syntax and unsafe paths in ``command``."""
        write_token = _COMMAND_WRITTEN_PATHS.set(set())
        try:
            active_environment = self._active_implicit_shell_environment()
            if active_environment is not None:
                raise CommandPolicyViolation(
                    "cannot safely inspect implicit shell initialization via "
                    f"{active_environment}"
                )
            nodes = self._parse_shell(command)
            states: tuple[_ShellState, ...] = (_ShellState(cwd=self._initial_cwd),)
            for node in nodes:
                states = self._validate_node_states(node, states)
        finally:
            _COMMAND_WRITTEN_PATHS.reset(write_token)

    def _validate_node_states(
        self,
        node: Any,
        states: Sequence[_ShellState],
    ) -> tuple[_ShellState, ...]:
        if getattr(node, "kind", None) == "list":
            return self._validate_list_states(node, states)
        return self._dedupe_shell_states(
            self._validate_node(node, state) for state in states
        )

    def _validate_list_states(
        self,
        node: Any,
        states: Sequence[_ShellState],
    ) -> tuple[_ShellState, ...]:
        active = self._dedupe_shell_states(states)
        deferred: tuple[_ShellState, ...] = ()
        successful = active
        failed = active

        for part in node.parts:
            if getattr(part, "kind", None) in {"operator", "pipe"}:
                operator = getattr(part, "op", None)
                if operator == "&&":
                    deferred = self._dedupe_shell_states((*deferred, *failed))
                    active = successful
                elif operator == "||":
                    deferred = self._dedupe_shell_states((*deferred, *successful))
                    active = failed
                elif operator == ";":
                    active = self._dedupe_shell_states(
                        (*deferred, *successful, *failed)
                    )
                    deferred = ()
                else:
                    raise CommandPolicyViolation(
                        f"unsupported shell operator {operator!r}; split the "
                        "operation into separate command calls"
                    )
                continue

            before = active
            successful = self._validate_node_states(part, before)
            # A state-changing shell builtin can fail without changing the
            # process cwd. Keep that failure state until the shell operator
            # determines whether the next command runs.
            failed = before

        return self._dedupe_shell_states((*deferred, *successful, *failed))

    @staticmethod
    def _dedupe_shell_states(
        states: Iterable[_ShellState],
    ) -> tuple[_ShellState, ...]:
        unique = tuple(dict.fromkeys(states))
        if len(unique) > _MAX_POSSIBLE_SHELL_STATES:
            raise CommandPolicyViolation("too many possible shell directory states")
        return unique

    def validate_argv(self, argv: Sequence[str]) -> None:
        """Reject out-of-policy paths in literal, pre-tokenized arguments.

        No shell interprets this form, so shell expansion syntax remains literal.
        """
        if not argv:
            return
        write_token = _COMMAND_WRITTEN_PATHS.set(set())
        try:
            self._validate_command_values(
                argv[0],
                argv[1:],
                _ShellState(cwd=self._initial_cwd),
            )
        finally:
            _COMMAND_WRITTEN_PATHS.reset(write_token)

    def _validate_node(self, node: Any, state: _ShellState) -> _ShellState:
        kind = getattr(node, "kind", None)
        if kind == "list":
            states = self._validate_list_states(node, (state,))
            if len(states) != 1:
                raise CommandPolicyViolation(
                    "cannot propagate multiple shell directory states"
                )
            return states[0]

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
                self._validate_redirect(redirect, current)
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
                self._validate_redirect(part, state)
            elif kind == "word":
                words.append(part)
                self._validate_nested_nodes(part, state)
            elif kind == "assignment":
                self._reject_implicit_shell_environment((str(part.word),))
                self._validate_nested_nodes(part, state)
            else:
                self._validate_nested_nodes(part, state)

        if not words:
            return state

        command_word = self._command_value(words[0])
        if not command_word.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        args = self._command_values(words[1:])

        return self._validate_command_values(command_word, args, state)

    def _validate_command_values(
        self,
        command_word: str,
        args: Sequence[str],
        state: _ShellState,
    ) -> _ShellState:
        command_name = os.path.basename(command_word)
        if self._is_direct_command_path(command_word):
            if not self._is_trusted_system_command(command_word, command_name):
                self._inspect_direct_shell_script(command_word, state)
                return state
        elif command_name in _CLASSIFIED_EXECUTABLE_COMMANDS:
            discovered = shutil.which(command_name)
            if discovered is not None and not self._is_trusted_system_command(
                discovered,
                command_name,
            ):
                self._inspect_direct_shell_script(discovered, state)
                return state

        if command_name in {"cd", "pushd", "popd"}:
            return self._change_directory(command_name, args, state)

        if command_name in {"alias", "eval", "hash", "trap"}:
            raise CommandPolicyViolation(
                f"cannot safely inspect shell text executed by {command_name}"
            )
        if command_name in {"declare", "export", "typeset"}:
            self._reject_implicit_shell_environment(args)
        if command_name in _READ_COMMANDS:
            self._check_operands(args, state.cwd, "read")
        elif command_name in _WRITE_COMMANDS:
            self._check_operands(args, state.cwd, "write")
        elif command_name in _SHELL_COMMANDS:
            self._check_nested_shell(command_name, args, state)
        elif command_name in _UNSUPPORTED_SHELL_COMMANDS:
            raise CommandPolicyViolation(
                f"shell dialect {command_name} does not match the Bash policy parser"
            )
        elif command_name in {".", "source"}:
            return self._check_sourced_shell(args, state)
        elif command_name == "xargs":
            self._check_xargs(args, state)
        elif command_name == "chroot":
            raise CommandPolicyViolation(
                "cannot safely inspect chroot filesystem remapping"
            )
        elif command_name in _COMMAND_WRAPPER_GRAMMARS:
            return self._check_command_wrapper(command_name, args, state)
        return state

    def _validate_redirect(self, node: Any, state: _ShellState) -> None:
        redirect_type = getattr(node, "type", "")
        if redirect_type in {"<<", "<<-"}:
            heredoc = getattr(node, "heredoc", None)
            body = str(getattr(heredoc, "value", ""))
            if "$" in body or "`" in body:
                raise CommandPolicyViolation(
                    "cannot safely inspect expanding here-document"
                )
            return
        output = getattr(node, "output", None)
        if redirect_type == "<<<":
            if getattr(output, "kind", None) is not None:
                self._validate_nested_nodes(output, state)
            return
        # bashlex emits descriptor-duplication targets such as 2>&1 as integers.
        if output is None or getattr(output, "kind", None) != "word":
            return
        raw_path = self._command_value(output)
        access: PathAccess = "read" if redirect_type == "<" else "write"
        self._check_path(raw_path, state.cwd, access)

    @staticmethod
    def _active_implicit_shell_environment() -> str | None:
        return next(
            (name for name in _IMPLICIT_SHELL_ENVIRONMENT if os.environ.get(name)),
            None,
        )

    @staticmethod
    def _policy_home_directory() -> str:
        home = os.environ.get("HOME")
        if not home:
            raise CommandPolicyViolation(
                "cannot resolve bare cd without HOME in the execution environment"
            )
        return home

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
            target_word = operands[0] if operands else self._policy_home_directory()
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
            option, argument, consumed = self._parse_wrapper_option(
                command_name,
                values,
                index,
                grammar,
            )
            if option in grammar.cwd_options and argument is not None:
                target = self._check_path(argument, nested_state.cwd, "read")
                nested_state = replace(nested_state, cwd=target)
            terminal_mode = terminal_mode or option in grammar.terminal_options
            index += consumed

        for _ in range(grammar.leading_scalars):
            if index >= len(values):
                return state
            index += 1

        if grammar.allows_assignments:
            assignments_start = index
            while index < len(values) and "=" in values[index]:
                index += 1
            self._reject_implicit_shell_environment(values[assignments_start:index])

        if terminal_mode or index >= len(values):
            return state

        command_word = values[index]
        if isinstance(command_word, _CommandValue) and not command_word.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic command name")
        # Wrappers spawn a child command: validate its paths against the adjusted
        # child cwd, but never propagate child shell directory state to the parent.
        validated_state = self._validate_command_values(
            command_word, values[index + 1 :], nested_state
        )
        if not grammar.propagates_state:
            return state
        return replace(validated_state, wrapper_depth=state.wrapper_depth)

    def _parse_wrapper_option(
        self,
        command_name: str,
        values: Sequence[str],
        index: int,
        grammar: _CommandWrapperGrammar,
    ) -> tuple[str, str | None, int]:
        value = values[index]
        if value in grammar.flag_options or (
            value in grammar.terminal_options and value not in grammar.value_options
        ):
            return value, None, 1
        if value in grammar.value_options:
            if index + 1 >= len(values):
                raise CommandPolicyViolation(
                    f"missing {command_name} argument for {value}"
                )
            return value, values[index + 1], 2

        matching_long = next(
            (
                option
                for option in grammar.value_options
                if option.startswith("--") and value.startswith(f"{option}=")
            ),
            None,
        )
        if matching_long is not None:
            return (
                matching_long,
                self._derived_value(value, value.split("=", 1)[1]),
                1,
            )

        matching_short = next(
            (
                option
                for option in grammar.attached_short_value_options
                if value.startswith(option) and len(value) > len(option)
            ),
            None,
        )
        if matching_short is not None:
            return (
                matching_short,
                self._derived_value(value, value[len(matching_short) :]),
                1,
            )

        raise CommandPolicyViolation(
            f"cannot safely inspect {command_name} option {value}"
        )

    def _check_operands(
        self,
        values: Sequence[str],
        cwd: Path,
        access: PathAccess,
    ) -> None:
        for raw_path in self._operands(values):
            self._check_path(raw_path, cwd, access)

    def _read_policy_script(self, raw_path: str, cwd: Path) -> str:
        script_path = self._check_path(raw_path, cwd, "read")
        written_paths = _COMMAND_WRITTEN_PATHS.get()
        if written_paths is not None and script_path in written_paths:
            raise CommandPolicyViolation(
                "cannot inspect a script modified earlier in the same command"
            )

        descriptor: int | None = None
        try:
            flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
            descriptor = os.open(script_path, flags)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > _MAX_INSPECTED_SCRIPT_BYTES
            ):
                raise CommandPathViolation(access="read", path=script_path)
            with os.fdopen(descriptor, "rb") as script_file:
                descriptor = None
                raw_script = script_file.read(_MAX_INSPECTED_SCRIPT_BYTES + 1)
            if len(raw_script) > _MAX_INSPECTED_SCRIPT_BYTES:
                raise CommandPathViolation(access="read", path=script_path)
            script = raw_script.decode("utf-8")
        except OSError as exc:
            # A missing script may be created earlier in the same shell string;
            # skipping inspection would allow its embedded file I/O unchecked.
            if exc.errno == errno.ELOOP:
                raise CommandPolicyViolation(
                    f"{script_path} symbolic link changed during inspection"
                ) from exc
            raise CommandPathViolation(access="read", path=script_path) from exc
        except UnicodeError as exc:
            raise CommandPolicyViolation(
                f"{script_path} is not a UTF-8 policy script"
            ) from exc
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
        nested_state = replace(state, script_depth=state.script_depth + 1)
        validated_states = self._validate_shell_states(script, nested_state)
        if not propagate_state:
            return state
        if len(validated_states) != 1:
            raise CommandPolicyViolation(
                "cannot propagate multiple shell directory states from source"
            )
        validated_state = validated_states[0]
        return replace(validated_state, script_depth=state.script_depth)

    @staticmethod
    def _is_direct_command_path(command_word: str) -> bool:
        return "/" in command_word

    @staticmethod
    def _is_trusted_system_command(command_word: str, command_name: str) -> bool:
        candidate = Path(command_word).expanduser()
        if not candidate.is_absolute():
            return False
        discovered = shutil.which(command_name)
        if discovered is None:
            return False
        try:
            resolved = candidate.resolve()
            return resolved == Path(discovered).resolve() and any(
                resolved.is_relative_to(root) for root in _TRUSTED_EXECUTABLE_ROOTS
            )
        except (OSError, RuntimeError):
            return False

    def _inspect_direct_shell_script(
        self,
        raw_path: str,
        state: _ShellState,
    ) -> None:
        if state.script_depth >= _MAX_INSPECTED_SCRIPT_DEPTH:
            raise CommandPolicyViolation("shell script inspection depth exceeded")
        script_path = self._check_workspace_path(raw_path, state.cwd, "read")
        script = self._read_policy_script(str(script_path), state.cwd)

        first_line = script.splitlines()[0] if script.splitlines() else ""
        if first_line.startswith("#!"):
            interpreter = os.path.basename(first_line[2:].strip().split()[0])
            if interpreter == "env":
                interpreter_parts = first_line[2:].strip().split()
                interpreter = (
                    os.path.basename(interpreter_parts[1])
                    if len(interpreter_parts) > 1
                    else ""
                )
            if interpreter not in _SHELL_COMMANDS:
                raise CommandPolicyViolation(
                    "direct scripts must use the Bash policy shell dialect"
                )

        nested_state = replace(state, script_depth=state.script_depth + 1)
        self._validate_shell_states(script, nested_state)

    @staticmethod
    def _reject_implicit_shell_environment(values: Sequence[str]) -> None:
        for value in values:
            name, _, _ = str(value).partition("=")
            if name in _REJECTED_SHELL_ASSIGNMENTS:
                raise CommandPolicyViolation(
                    f"cannot safely inspect shell execution environment via {name}"
                )

    def _check_nested_shell(
        self,
        command_name: str,
        literals: Sequence[str],
        state: _ShellState,
    ) -> None:
        active_environment = self._active_implicit_shell_environment()
        if active_environment is not None:
            raise CommandPolicyViolation(
                "cannot safely inspect implicit shell initialization via "
                f"{active_environment}"
            )
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
            self._validate_shell_states(literals[command_option_index + 1], state)
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
                index += 2
                continue
            if value.startswith("--") and "=" in value:
                index += 1
                continue
            if value.startswith(attached_prefixes) and len(value) > 2:
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            command_start = index
            break

        if command_start < len(values):
            # Validate the fixed command first so wrapper limits and any static
            # path violations keep their precise diagnostics. Runtime stdin
            # arguments remain uninspectable regardless of that result.
            self._validate_nested_command_words(values[command_start:], state)
        raise CommandPolicyViolation(
            "cannot safely inspect runtime file arguments produced by xargs"
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
        self._validate_command_values(words[0], words[1:], state)

    def _validate_shell_states(
        self,
        command: str,
        state: _ShellState,
    ) -> tuple[_ShellState, ...]:
        if isinstance(command, _CommandValue) and not command.is_static:
            raise CommandPolicyViolation("cannot resolve dynamic shell command text")
        nodes = self._parse_shell(command)
        current: tuple[_ShellState, ...] = (state,)
        for node in nodes:
            current = self._validate_node_states(node, current)
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
        return self._resolve_policy_path(
            raw_path,
            cwd,
            access,
            include_external_dirs=access == "read",
        )

    def _check_workspace_path(
        self,
        raw_path: str,
        cwd: Path,
        access: PathAccess,
    ) -> Path:
        return self._resolve_policy_path(
            raw_path,
            cwd,
            access,
            include_external_dirs=False,
        )

    def _resolve_policy_path(
        self,
        raw_path: str,
        cwd: Path,
        access: PathAccess,
        *,
        include_external_dirs: bool,
    ) -> Path:
        if isinstance(raw_path, _CommandValue) and not raw_path.is_static:
            raise CommandPolicyViolation(
                "cannot resolve dynamic path operand; enumerate concrete paths "
                "instead of using active globs or shell expansions"
            )

        if raw_path in {"", "-", "{}"}:
            return cwd

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate

        if candidate in _SAFE_DEVICE_PATHS:
            return candidate

        try:
            resolved = self._workspace.resolve_authorized_path(
                candidate,
                base_dir=cwd,
                include_external_dirs=include_external_dirs,
            )
        except ValueError as exc:
            raise CommandPathViolation(access=access, path=candidate) from exc
        written_paths = _COMMAND_WRITTEN_PATHS.get()
        if access == "write" and written_paths is not None:
            written_paths.add(resolved)
        return resolved
