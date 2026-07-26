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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
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
class _FindExecClause:
    marker: Literal["-exec", "-execdir", "-ok", "-okdir"]
    command: tuple[str, ...]


@dataclass(frozen=True)
class _PathEvent:
    value: str
    access: PathAccess


@dataclass(frozen=True)
class _FindInvocation:
    roots: tuple[str, ...]
    path_events: tuple[_PathEvent, ...]
    exec_clauses: tuple[_FindExecClause, ...]
    writes_roots: bool = False


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
    short_flag_options: frozenset[str] = frozenset()
    short_value_options: frozenset[str] = frozenset()
    in_place_short_option: str | None = None
    in_place_long_option: str | None = None


@dataclass(frozen=True)
class _ScriptShortOptionResult:
    next_index: int
    explicit_script: bool = False
    requests_in_place: bool = False


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
    value_options=frozenset({"-l", "--line-length"}),
    short_flag_options=frozenset({"E", "n", "r", "s", "u", "z"}),
    short_value_options=frozenset({"l"}),
    in_place_short_option="i",
    in_place_long_option="--in-place",
)
_AWK_GRAMMAR = _ScriptCommandGrammar(
    language="awk",
    expression_long_option="--source",
    value_options=frozenset({"-F", "-v"}),
    ignores_assignment_arguments=True,
)
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
    "awk",
    "base64",
    "cp",
    "curl",
    "dd",
    "find",
    "grep",
    "gzip",
    "install",
    "ln",
    "mv",
    "rsync",
    "sed",
    "shred",
    "tar",
    "unlink",
    "wget",
    "xargs",
    *(
        name
        for name in _COMMAND_WRAPPER_GRAMMARS
        if name not in {"builtin", "command", "exec", _POLICY_TIME_WRAPPER}
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
        active_environment = next(
            (name for name in _IMPLICIT_SHELL_ENVIRONMENT if os.environ.get(name)),
            None,
        )
        if active_environment is not None:
            raise CommandPolicyViolation(
                "cannot safely inspect implicit shell initialization via "
                f"{active_environment}"
            )
        nodes = self._parse_shell(command)

        states: tuple[_ShellState, ...] = (_ShellState(cwd=self._initial_cwd),)
        for node in nodes:
            states = self._validate_node_states(node, states)

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
        self._validate_command_values(
            argv[0],
            argv[1:],
            _ShellState(cwd=self._initial_cwd),
        )

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
            self._check_read_command(command_name, args, state.cwd)
        elif command_name in _WRITE_COMMANDS:
            self._check_operands(args, state.cwd, "write")
        elif command_name == "grep":
            self._check_grep(args, state.cwd)
        elif command_name == "sed":
            self._check_script_command(_SED_GRAMMAR, args, state.cwd)
        elif command_name == "awk":
            self._check_script_command(_AWK_GRAMMAR, args, state.cwd)
        elif command_name == "base64":
            self._check_base64(args, state.cwd)
        elif command_name == "dd":
            self._check_dd(args, state.cwd)
        elif command_name == "tar":
            self._check_tar(args, state.cwd)
        elif command_name == "cp":
            self._check_copy(args, state.cwd)
        elif command_name == "install":
            self._check_install(args, state.cwd)
        elif command_name in {"mv", "ln"}:
            self._check_move_or_link(args, state.cwd)
        elif command_name in {"unlink", "shred"}:
            self._check_destructive_file_command(command_name, args, state.cwd)
        elif command_name == "gzip":
            self._check_gzip(args, state.cwd)
        elif command_name == "rsync":
            self._check_rsync(args, state.cwd)
        elif command_name == "curl":
            self._check_curl(args, state.cwd)
        elif command_name == "wget":
            self._check_wget(args, state.cwd)
        elif command_name == "find":
            self._check_find(args, state)
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
            values = self._partition_path_options(
                values,
                cwd,
                option_access={
                    "-f": "read",
                    "-m": "read",
                    "--files-from": "read",
                    "--magic-file": "read",
                },
                attached_short_options={"-f", "-m"},
            )
        elif command_name == "wc":
            values = self._partition_path_options(
                values,
                cwd,
                option_access={"--files0-from": "read"},
            )
        self._check_operands(values, cwd, "read")

    def _check_tac(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("tac arguments", values)
        for raw_path in self._partition_path_options(
            values,
            cwd,
            option_access={"-s": None, "--separator": None},
            attached_short_options={"-s"},
        ):
            self._check_path(raw_path, cwd, "read")

    def _check_cut(self, values: Sequence[str], cwd: Path) -> None:
        scalar_options = {
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
        for raw_path in self._partition_path_options(
            values,
            cwd,
            option_access=dict.fromkeys(scalar_options),
            attached_short_options={"-b", "-c", "-d", "-f"},
        ):
            self._check_path(raw_path, cwd, "read")

    def _check_sort(self, values: Sequence[str], cwd: Path) -> None:
        for event in self._parse_sort_path_events(values):
            self._check_path(event.value, cwd, event.access)

    def _parse_sort_path_events(
        self,
        values: Sequence[str],
    ) -> tuple[_PathEvent, ...]:
        short_flag_options = frozenset("bdfgiMhnRrVcCmsuz")
        short_value_options: dict[str, PathAccess | None] = {
            "k": None,
            "o": "write",
            "S": None,
            "t": None,
            "T": "write",
        }
        long_path_options: dict[str, PathAccess] = {
            "--files0-from": "read",
            "--output": "write",
            "--random-source": "read",
            "--temporary-directory": "write",
        }
        long_scalar_options = {
            "--batch-size",
            "--buffer-size",
            "--field-separator",
            "--key",
            "--parallel",
            "--sort",
        }
        denied_long_options = {"--compress-program"}
        long_optional_value_options = {"--check"}
        long_flag_options = {
            "--debug",
            "--dictionary-order",
            "--general-numeric-sort",
            "--help",
            "--human-numeric-sort",
            "--ignore-case",
            "--ignore-leading-blanks",
            "--ignore-nonprinting",
            "--merge",
            "--month-sort",
            "--numeric-sort",
            "--random-sort",
            "--reverse",
            "--stable",
            "--unique",
            "--version",
            "--version-sort",
            "--zero-terminated",
        }
        known_long_options = (
            set(long_path_options)
            | long_scalar_options
            | denied_long_options
            | long_optional_value_options
            | long_flag_options
        )

        path_events: list[_PathEvent] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                path_events.extend(
                    _PathEvent(raw_path, "read") for raw_path in values[index + 1 :]
                )
                break

            value_text = str(value)
            if value_text.startswith("--"):
                raw_option, separator, _ = value_text.partition("=")
                option = self._resolve_sort_long_option(
                    raw_option,
                    known_long_options,
                )
                path_access = long_path_options.get(option)
                if path_access is not None:
                    argument: str
                    if separator:
                        argument = self._derived_value(
                            value,
                            value[len(raw_option) + 1 :],
                        )
                        next_index = index + 1
                    elif index + 1 < len(values):
                        argument = values[index + 1]
                        next_index = index + 2
                    else:
                        raise CommandPolicyViolation(
                            f"missing sort argument for {option}"
                        )
                    path_events.append(_PathEvent(argument, path_access))
                    index = next_index
                    continue

                if option in denied_long_options:
                    raise CommandPolicyViolation(
                        f"sort option {option} cannot safely delegate execution"
                    )

                if option in long_scalar_options:
                    if separator:
                        index += 1
                    elif index + 1 < len(values):
                        index += 2
                    else:
                        raise CommandPolicyViolation(
                            f"missing sort argument for {option}"
                        )
                    continue

                if option in long_optional_value_options:
                    index += 1
                    continue

                if separator:
                    raise CommandPolicyViolation(
                        f"sort option {option} does not accept an argument"
                    )

                # The remaining recognized long options are argument-free.
                index += 1
                continue

            if value_text.startswith("-") and value_text != "-":
                index = self._parse_sort_short_options(
                    values,
                    index,
                    short_flag_options,
                    short_value_options,
                    path_events,
                )
                continue

            path_events.append(_PathEvent(value, "read"))
            index += 1

        return tuple(path_events)

    @staticmethod
    def _resolve_sort_long_option(
        option: str,
        known_options: set[str],
    ) -> str:
        if option in known_options:
            return option
        matches = [
            candidate for candidate in known_options if candidate.startswith(option)
        ]
        if len(matches) != 1:
            raise CommandPolicyViolation(f"cannot safely resolve sort option {option}")
        return matches[0]

    def _parse_sort_short_options(
        self,
        values: Sequence[str],
        index: int,
        flag_options: frozenset[str],
        value_options: dict[str, PathAccess | None],
        path_events: list[_PathEvent],
    ) -> int:
        source = values[index]
        options = str(source)[1:]
        cursor = 0
        while cursor < len(options):
            option = options[cursor]
            if option in flag_options:
                cursor += 1
                continue
            if option not in value_options:
                raise CommandPolicyViolation(
                    f"cannot safely resolve sort option -{option}"
                )

            attached = options[cursor + 1 :]
            argument: str
            if attached:
                argument = self._derived_value(source, attached)
                next_index = index + 1
            elif index + 1 < len(values):
                argument = values[index + 1]
                next_index = index + 2
            else:
                raise CommandPolicyViolation(f"missing sort argument for -{option}")

            access = value_options[option]
            if access is not None:
                path_events.append(_PathEvent(argument, access))
            return next_index

        return index + 1

    def _check_uniq(self, values: Sequence[str], cwd: Path) -> None:
        scalar_options = {
            "-f",
            "--skip-fields",
            "-s",
            "--skip-chars",
            "-w",
            "--check-chars",
        }
        operands = self._partition_path_options(
            values,
            cwd,
            option_access=dict.fromkeys(scalar_options),
            attached_short_options={"-f", "-s", "-w"},
        )
        if operands:
            self._check_path(operands[0], cwd, "read")
        if len(operands) > 1:
            self._check_path(operands[1], cwd, "write")

    def _check_diff(self, values: Sequence[str], cwd: Path) -> None:
        for raw_path in self._partition_path_options(
            values,
            cwd,
            option_access={
                "--output": "write",
                "--from-file": "read",
                "--to-file": "read",
            },
        ):
            self._check_path(raw_path, cwd, "read")

    def _partition_path_options(
        self,
        values: Sequence[str],
        cwd: Path,
        *,
        option_access: Mapping[str, PathAccess | None],
        attached_short_options: set[str] | frozenset[str] = frozenset(),
    ) -> list[str]:
        remaining: list[str] = []
        options_done = False
        index = 0
        while index < len(values):
            value = values[index]
            if not options_done and value == "--":
                options_done = True
                index += 1
                continue
            if options_done or not value.startswith("-") or value == "-":
                remaining.append(value)
                index += 1
                continue
            if value in option_access:
                if index + 1 >= len(values):
                    raise CommandPolicyViolation(f"missing argument for {value}")
                access = option_access[value]
                if access is not None:
                    self._check_path(values[index + 1], cwd, access)
                index += 2
                continue
            matching_long = next(
                (
                    option
                    for option in option_access
                    if option.startswith("--") and value.startswith(f"{option}=")
                ),
                None,
            )
            if matching_long is not None:
                access = option_access[matching_long]
                if access is not None:
                    self._check_path(value.split("=", 1)[1], cwd, access)
                index += 1
                continue
            matching_short = next(
                (
                    option
                    for option in attached_short_options
                    if value.startswith(option) and len(value) > len(option)
                ),
                None,
            )
            if matching_short is not None:
                access = option_access[matching_short]
                if access is not None:
                    self._check_path(value[len(matching_short) :], cwd, access)
                index += 1
                continue
            # Unknown options are not paths for these simple read grammars.
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
    ) -> None:
        positionals: list[str] = []
        explicit_script = False
        file_access: PathAccess = "read"
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
            if grammar.in_place_long_option is not None and (
                value == grammar.in_place_long_option
                or value.startswith(f"{grammar.in_place_long_option}=")
            ):
                file_access = "write"
                index += 1
                continue
            if (
                grammar.short_flag_options
                and value.startswith("-")
                and not value.startswith("--")
                and value != "-"
            ):
                option_result = self._consume_script_short_options(
                    grammar,
                    values,
                    index,
                    cwd,
                )
                explicit_script = explicit_script or option_result.explicit_script
                if option_result.requests_in_place:
                    file_access = "write"
                index = option_result.next_index
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

    def _consume_script_short_options(
        self,
        grammar: _ScriptCommandGrammar,
        values: Sequence[str],
        index: int,
        cwd: Path,
    ) -> _ScriptShortOptionResult:
        token = values[index]
        if isinstance(token, _CommandValue) and not token.is_static:
            raise CommandPolicyViolation(
                f"cannot inspect dynamic {grammar.language} options"
            )

        cursor = 1
        while cursor < len(token):
            option = token[cursor]
            if option in grammar.short_flag_options:
                cursor += 1
                continue
            if option == grammar.in_place_short_option:
                # GNU sed treats the remainder of this token as the optional
                # backup suffix, so no later character is another option.
                return _ScriptShortOptionResult(
                    next_index=index + 1,
                    requests_in_place=True,
                )
            if option in {"e", "f"} or option in grammar.short_value_options:
                attached_argument = token[cursor + 1 :]
                argument, next_index = self._take_option_argument(
                    values,
                    index,
                    attached_argument=(
                        self._derived_value(token, attached_argument)
                        if attached_argument
                        else None
                    ),
                    context=f"{grammar.language} argument for -{option}",
                )
                assert argument is not None

                if option == "e":
                    self._reject_embedded_io(grammar.language, argument)
                elif option == "f":
                    self._inspect_script_file(grammar.language, argument, cwd)
                return _ScriptShortOptionResult(
                    next_index=next_index,
                    explicit_script=option in {"e", "f"},
                )
            raise CommandPolicyViolation(
                f"cannot safely inspect {grammar.language} option -{option}"
            )

        return _ScriptShortOptionResult(next_index=index + 1)

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
                argument, next_index = self._take_option_argument(
                    values,
                    index,
                    attached_argument=(
                        self._derived_value(value, attached) if attached else None
                    ),
                    context=f"base64 argument for -{option}",
                    required=False,
                )
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

        argument, next_index = self._take_option_argument(
            values,
            index,
            attached_argument=(
                self._derived_value(value, attached) if separator else None
            ),
            context=f"tar argument for {option}",
        )

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

            attached = options[cursor + 1 :]
            argument, next_index = self._take_option_argument(
                values,
                index,
                attached_argument=(
                    self._derived_value(values[index], attached) if attached else None
                ),
                context=f"tar argument for -{option}",
            )
            assert argument is not None
            kind = argument_events[option]
            if kind is not None:
                events.append(_TarEvent(kind, argument))
            break

        return next_index, modes, events

    @staticmethod
    def _take_option_argument(
        values: Sequence[str],
        index: int,
        *,
        attached_argument: str | None,
        context: str,
        required: bool = True,
    ) -> tuple[str | None, int]:
        if attached_argument is not None:
            return attached_argument, index + 1
        if index + 1 < len(values):
            return values[index + 1], index + 2
        if required:
            raise CommandPolicyViolation(f"missing {context}")
        return None, index + 1

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
        if script is None:
            return state
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
        if script is None:
            return

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

        cursor = WorkspaceCommandPathGuard._scan_sed_fields(
            script,
            index,
            field_count=2,
        )

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

        cursor = WorkspaceCommandPathGuard._scan_sed_fields(
            script,
            index,
            field_count=2,
        )
        return WorkspaceCommandPathGuard._skip_sed_to_boundary(
            script, cursor, boundaries=";\n}"
        )

    @staticmethod
    def _scan_sed_fields(
        script: str,
        delimiter_index: int,
        *,
        field_count: int,
    ) -> int:
        cursor = delimiter_index
        for _ in range(field_count):
            cursor = WorkspaceCommandPathGuard._scan_sed_delimited(
                script,
                delimiter_index=cursor,
            )
            # The next field reuses the delimiter that closed this one.
            cursor -= 1
        return cursor + 1

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
        recursive = False
        dereferences_links = False
        for value in values:
            text = str(value)
            if text == "--":
                break
            if text in {"--recursive"}:
                recursive = True
            elif text in {"--dereference", "--follow-command-line-symlink"}:
                dereferences_links = True
            elif text.startswith("-") and not text.startswith("--"):
                flags = text[1:]
                recursive = recursive or "r" in flags or "R" in flags
                dereferences_links = dereferences_links or "L" in flags or "H" in flags
        if recursive and dereferences_links:
            raise CommandPolicyViolation(
                "cannot safely inspect recursive copying that follows symbolic links"
            )

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

    def _check_install(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("install arguments", values)
        target_dir: str | None = None
        operands: list[str] = []
        directory_mode = False
        options_done = False
        scalar_options = {
            "-g",
            "--group",
            "-m",
            "--mode",
            "-o",
            "--owner",
            "-S",
            "--suffix",
            "--context",
        }
        flag_options = {
            "-b",
            "--backup",
            "-c",
            "-C",
            "--compare",
            "-D",
            "-p",
            "--preserve-timestamps",
            "-s",
            "--strip",
            "-T",
            "--no-target-directory",
            "-v",
            "--verbose",
            "--help",
            "--version",
        }
        index = 0
        while index < len(values):
            value = values[index]
            text = str(value)
            if not options_done and text == "--":
                options_done = True
                index += 1
                continue
            if options_done or not text.startswith("-") or text == "-":
                operands.append(value)
                index += 1
                continue
            if text in {"-d", "--directory"}:
                directory_mode = True
                index += 1
                continue
            if text in {"-t", "--target-directory"}:
                if index + 1 >= len(values):
                    raise CommandPolicyViolation(f"missing install argument for {text}")
                target_dir = values[index + 1]
                index += 2
                continue
            if text.startswith("--target-directory="):
                target_dir = self._derived_value(value, text.split("=", 1)[1])
                index += 1
                continue
            if text.startswith("-t") and len(text) > 2:
                target_dir = self._derived_value(value, text[2:])
                index += 1
                continue
            if text == "--strip-program" or text.startswith("--strip-program="):
                raise CommandPolicyViolation(
                    "cannot safely inspect install delegated strip program"
                )
            if text in scalar_options:
                if index + 1 >= len(values):
                    raise CommandPolicyViolation(f"missing install argument for {text}")
                index += 2
                continue
            if any(
                text.startswith(f"{option}=")
                for option in scalar_options
                if option.startswith("--")
            ):
                index += 1
                continue
            if any(
                text.startswith(option) and len(text) > len(option)
                for option in {"-g", "-m", "-o", "-S"}
            ):
                index += 1
                continue
            if text in flag_options:
                index += 1
                continue
            raise CommandPolicyViolation(f"cannot safely inspect install option {text}")

        if directory_mode:
            for operand in operands:
                self._check_path(operand, cwd, "write")
            return
        if target_dir is not None:
            for operand in operands:
                self._check_path(operand, cwd, "read")
            self._check_path(target_dir, cwd, "write")
            return
        if len(operands) < 2:
            return
        for operand in operands[:-1]:
            self._check_path(operand, cwd, "read")
        self._check_path(operands[-1], cwd, "write")

    def _check_destructive_file_command(
        self,
        command_name: str,
        values: Sequence[str],
        cwd: Path,
    ) -> None:
        self._reject_dynamic_values(f"{command_name} arguments", values)
        if command_name == "unlink":
            operands = self._strict_simple_operands(
                command_name,
                values,
                flag_options={"-f", "--force", "--help", "--version"},
            )
        else:
            operands = self._parse_shred_operands(values, cwd)
        for operand in operands:
            self._check_path(operand, cwd, "write")

    def _parse_shred_operands(
        self,
        values: Sequence[str],
        cwd: Path,
    ) -> list[str]:
        operands: list[str] = []
        options_done = False
        index = 0
        while index < len(values):
            value = values[index]
            text = str(value)
            if not options_done and text == "--":
                options_done = True
                index += 1
                continue
            if options_done or not text.startswith("-") or text == "-":
                operands.append(value)
                index += 1
                continue
            if text in {"--random-source"} or text.startswith("--random-source="):
                argument, index = self._option_argument(values, index, value)
                self._check_path(argument, cwd, "read")
                continue
            if text in {"-n", "--iterations", "-s", "--size"}:
                if index + 1 >= len(values):
                    raise CommandPolicyViolation(f"missing shred argument for {text}")
                index += 2
                continue
            if text.startswith(("-n", "-s")) and len(text) > 2:
                index += 1
                continue
            if text.startswith(("--iterations=", "--size=", "--remove=")):
                index += 1
                continue
            if text in {
                "-f",
                "--force",
                "-u",
                "--remove",
                "-v",
                "--verbose",
                "-x",
                "--exact",
                "-z",
                "--zero",
                "--help",
                "--version",
            }:
                index += 1
                continue
            if (
                text.startswith("-")
                and not text.startswith("--")
                and set(text[1:]).issubset({"f", "u", "v", "x", "z"})
            ):
                index += 1
                continue
            raise CommandPolicyViolation(f"cannot safely inspect shred option {text}")
        return operands

    def _check_gzip(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("gzip arguments", values)
        read_only_mode = any(
            value
            in {
                "-c",
                "--stdout",
                "--to-stdout",
                "-l",
                "--list",
                "-t",
                "--test",
            }
            or (
                str(value).startswith("-")
                and not str(value).startswith("--")
                and any(flag in str(value)[1:] for flag in {"c", "l", "t"})
            )
            for value in values
        )
        operands = self._strict_simple_operands(
            "gzip",
            values,
            flag_options={
                "-a",
                "--ascii",
                "-c",
                "--stdout",
                "--to-stdout",
                "-d",
                "--decompress",
                "--uncompress",
                "-f",
                "--force",
                "-h",
                "--help",
                "-k",
                "--keep",
                "-l",
                "--list",
                "-n",
                "--no-name",
                "-N",
                "--name",
                "-q",
                "--quiet",
                "-r",
                "--recursive",
                "-t",
                "--test",
                "-v",
                "--verbose",
                "-V",
                "--version",
                *{f"-{level}" for level in range(1, 10)},
            },
            scalar_options={"-S", "--suffix"},
            allow_short_bundles=True,
        )
        access: PathAccess = "read" if read_only_mode else "write"
        for operand in operands:
            self._check_path(operand, cwd, access)

    def _check_rsync(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("rsync arguments", values)
        denied_options = {
            "-e",
            "--rsh",
            "-f",
            "--filter",
            "--files-from",
            "--include-from",
            "--exclude-from",
            "--password-file",
            "--rsync-path",
            "--copy-links",
            "--copy-unsafe-links",
            "--keep-dirlinks",
        }
        path_options: dict[str, PathAccess] = {
            "--backup-dir": "write",
            "--partial-dir": "write",
            "--temp-dir": "write",
            "--compare-dest": "read",
            "--copy-dest": "read",
            # rsync may hard-link unchanged files from this tree into the
            # destination, so read-only external roots are not sufficient.
            "--link-dest": "write",
        }
        scalar_options = {
            "--block-size",
            "--bwlimit",
            "--checksum-choice",
            "--chmod",
            "--compress-choice",
            "--compress-level",
            "--contimeout",
            "--max-alloc",
            "--max-delete",
            "--max-size",
            "--min-size",
            "--out-format",
            "--port",
            "--sockopts",
            "--timeout",
            "--usermap",
            "--groupmap",
        }
        operands: list[str] = []
        options_done = False
        index = 0
        while index < len(values):
            value = values[index]
            text = str(value)
            if not options_done and text == "--":
                options_done = True
                index += 1
                continue
            if options_done or not text.startswith("-") or text == "-":
                operands.append(value)
                index += 1
                continue
            option = text.split("=", 1)[0]
            if option in denied_options or (
                text.startswith("-")
                and not text.startswith("--")
                and any(flag in text[1:] for flag in {"e", "f", "L", "H"})
            ):
                raise CommandPolicyViolation(
                    f"cannot safely inspect rsync option {option}"
                )
            if option in path_options:
                argument, index = self._option_argument(values, index, value)
                self._check_path(argument, cwd, path_options[option])
                continue
            if option in scalar_options:
                _, index = self._option_argument(values, index, value)
                continue
            if text.startswith("--"):
                # Long flag options are argument-free here. Options with
                # unmodeled values fail closed instead of shifting operands.
                if "=" in text:
                    raise CommandPolicyViolation(
                        f"cannot safely inspect rsync option {option}"
                    )
                index += 1
                continue
            # Common short flags may be bundled (for example -avz).
            index += 1

        if len(operands) < 2:
            return
        if any(self._is_remote_transfer_operand(operand) for operand in operands):
            raise CommandPolicyViolation("cannot safely inspect remote rsync operands")
        for operand in operands[:-1]:
            self._check_path(operand, cwd, "read")
        self._check_path(operands[-1], cwd, "write")

    def _check_curl(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("curl arguments", values)
        index = 0
        while index < len(values):
            value = values[index]
            text = str(value)
            if text in {"-O", "--remote-name", "--remote-name-all"} or (
                text.startswith("-O") and not text.startswith("--")
            ):
                raise CommandPolicyViolation(
                    "cannot safely resolve curl remote output filename"
                )
            if (
                text in {"-K", "--config"}
                or text.startswith("--config=")
                or (text.startswith("-K") and len(text) > 2)
            ):
                raise CommandPolicyViolation(
                    "cannot safely inspect curl runtime configuration"
                )
            if text in {"-o", "--output", "--output-dir"} or text.startswith(
                ("--output=", "--output-dir=")
            ):
                argument, index = self._option_argument(values, index, value)
                self._check_path(argument, cwd, "write")
                continue
            if text.startswith("-o") and len(text) > 2:
                self._check_path(self._derived_value(value, text[2:]), cwd, "write")
                index += 1
                continue
            if text in {"-T", "--upload-file", "--netrc-file"} or text.startswith(
                ("--upload-file=", "--netrc-file=")
            ):
                argument, index = self._option_argument(values, index, value)
                self._check_path(argument, cwd, "read")
                continue
            if text.startswith("-T") and len(text) > 2:
                self._check_path(self._derived_value(value, text[2:]), cwd, "read")
                index += 1
                continue
            if text in {
                "-d",
                "--data",
                "--data-binary",
                "--data-raw",
            } or text.startswith(("--data=", "--data-binary=", "--data-raw=")):
                argument, index = self._option_argument(values, index, value)
                if str(argument).startswith("@"):
                    self._check_path(
                        self._derived_value(argument, str(argument)[1:]),
                        cwd,
                        "read",
                    )
                continue
            if text.startswith("-d") and len(text) > 2:
                argument = self._derived_value(value, text[2:])
                if str(argument).startswith("@"):
                    self._check_path(
                        self._derived_value(argument, str(argument)[1:]),
                        cwd,
                        "read",
                    )
                index += 1
                continue
            index += 1

    def _check_wget(self, values: Sequence[str], cwd: Path) -> None:
        self._reject_dynamic_values("wget arguments", values)
        has_explicit_output = False
        spider_mode = False
        has_url_operand = False
        index = 0
        while index < len(values):
            value = values[index]
            text = str(value)
            if text in {"--config"} or text.startswith("--config="):
                raise CommandPolicyViolation(
                    "cannot safely inspect wget runtime configuration"
                )
            if text in {"-i", "--input-file"} or text.startswith(
                ("-i", "--input-file=")
            ):
                raise CommandPolicyViolation(
                    "cannot safely inspect wget runtime URL list"
                )
            if text == "--spider":
                spider_mode = True
                index += 1
                continue
            if text in {"-O", "--output-document", "-P", "--directory-prefix"} or (
                text.startswith("--output-document=")
                or text.startswith("--directory-prefix=")
            ):
                argument, index = self._option_argument(values, index, value)
                self._check_path(argument, cwd, "write")
                if text in {"-O", "--output-document"} or text.startswith(
                    "--output-document="
                ):
                    has_explicit_output = True
                continue
            if (text.startswith("-O") or text.startswith("-P")) and len(text) > 2:
                self._check_path(self._derived_value(value, text[2:]), cwd, "write")
                if text.startswith("-O"):
                    has_explicit_output = True
                index += 1
                continue
            if text in {"--post-file", "--body-file"} or text.startswith(
                ("--post-file=", "--body-file=")
            ):
                argument, index = self._option_argument(values, index, value)
                self._check_path(argument, cwd, "read")
                continue
            if not text.startswith("-"):
                has_url_operand = True
            index += 1
        if has_url_operand and not has_explicit_output and not spider_mode:
            raise CommandPolicyViolation(
                "cannot safely resolve wget remote output filename"
            )

    def _strict_simple_operands(
        self,
        command_name: str,
        values: Sequence[str],
        *,
        flag_options: set[str],
        scalar_options: set[str] | frozenset[str] = frozenset(),
        allow_short_bundles: bool = False,
    ) -> list[str]:
        operands: list[str] = []
        options_done = False
        index = 0
        short_flags = {
            option[1:]
            for option in flag_options
            if option.startswith("-") and not option.startswith("--")
        }
        while index < len(values):
            value = values[index]
            text = str(value)
            if not options_done and text == "--":
                options_done = True
                index += 1
                continue
            if options_done or not text.startswith("-") or text == "-":
                operands.append(value)
                index += 1
                continue
            if text in flag_options:
                index += 1
                continue
            if text in scalar_options:
                if index + 1 >= len(values):
                    raise CommandPolicyViolation(
                        f"missing {command_name} argument for {text}"
                    )
                index += 2
                continue
            if any(
                text.startswith(f"{option}=")
                for option in scalar_options
                if option.startswith("--")
            ):
                index += 1
                continue
            if (
                allow_short_bundles
                and text.startswith("-")
                and not text.startswith("--")
                and set(text[1:]).issubset(short_flags)
            ):
                index += 1
                continue
            raise CommandPolicyViolation(
                f"cannot safely inspect {command_name} option {text}"
            )
        return operands

    def _option_argument(
        self,
        values: Sequence[str],
        index: int,
        source: str,
    ) -> tuple[str, int]:
        text = str(source)
        if "=" in text and text.startswith("--"):
            return self._derived_value(source, text.split("=", 1)[1]), index + 1
        if index + 1 >= len(values):
            raise CommandPolicyViolation(f"missing argument for {text}")
        return values[index + 1], index + 2

    @staticmethod
    def _is_remote_transfer_operand(value: str) -> bool:
        text = str(value)
        return text.startswith("rsync://") or ":" in text

    def _check_move_or_link(self, values: Sequence[str], cwd: Path) -> None:
        target_dir, operands = self._parse_target_directory(values)
        # A hard link inside the workspace aliases its source inode, so an
        # external read-only source must remain write-protected here.
        for raw_path in operands:
            self._check_path(raw_path, cwd, "write")
        if target_dir is not None:
            self._check_path(target_dir, cwd, "write")

    def _check_find(self, literals: Sequence[str], state: _ShellState) -> None:
        invocation = self._parse_find_invocation(literals)
        for event in invocation.path_events:
            self._check_path(event.value, state.cwd, event.access)

        writes_roots = invocation.writes_roots or any(
            self._find_exec_command_writes(clause, state)
            for clause in invocation.exec_clauses
        )
        root_access: PathAccess = "write" if writes_roots else "read"
        for root in invocation.roots:
            self._check_path(root, state.cwd, root_access)

        for clause in invocation.exec_clauses:
            self._validate_nested_command_words(clause.command, state)

    def _parse_find_invocation(
        self,
        literals: Sequence[str],
    ) -> _FindInvocation:
        self._reject_dynamic_values("find arguments", literals)
        roots: list[str] = []
        expression_start = len(literals)
        root_start = 0
        while root_start < len(literals):
            option = literals[root_start]
            if option in {"-H", "-L"}:
                raise CommandPolicyViolation(
                    "cannot safely inspect find traversal that follows symbolic links"
                )
            if option == "-P" or option.startswith("-O"):
                root_start += 1
                continue
            if option == "-D":
                if root_start + 1 >= len(literals):
                    raise CommandPolicyViolation("missing find argument for -D")
                root_start += 2
                continue
            if option.startswith("-D"):
                root_start += 1
                continue
            break
        for index, value in enumerate(literals[root_start:], start=root_start):
            if value.startswith("-") or value in {"!", "("}:
                expression_start = index
                break
            roots.append(value)

        if not roots:
            roots = ["."]
        expression = literals[expression_start:]
        path_events: list[_PathEvent] = []
        clauses: list[_FindExecClause] = []
        writes_roots = False
        output_actions = {
            "-fprint": 1,
            "-fprint0": 1,
            "-fls": 1,
            "-fprintf": 2,
        }
        reference_predicates = {"-newer", "-anewer", "-cnewer", "-samefile"}
        index = 0
        while index < len(expression):
            marker = expression[index]
            if marker == "-files0-from" or marker.startswith("-files0-from="):
                raise CommandPolicyViolation(
                    "cannot safely inspect find runtime root list"
                )
            if marker == "-delete":
                writes_roots = True
                index += 1
                continue
            if marker in output_actions:
                argument_count = output_actions[marker]
                if index + argument_count >= len(expression):
                    raise CommandPolicyViolation(f"missing find argument for {marker}")
                path_events.append(_PathEvent(expression[index + 1], "write"))
                index += argument_count + 1
                continue
            if marker in reference_predicates or (
                marker.startswith("-newer")
                and marker != "-newer"
                and not marker.endswith("t")
            ):
                if index + 1 >= len(expression):
                    raise CommandPolicyViolation(f"missing find argument for {marker}")
                path_events.append(_PathEvent(expression[index + 1], "read"))
                index += 2
                continue
            if marker.startswith("-newer") and marker.endswith("t"):
                if index + 1 >= len(expression):
                    raise CommandPolicyViolation(f"missing find argument for {marker}")
                index += 2
                continue
            if marker not in {"-exec", "-execdir", "-ok", "-okdir"}:
                index += 1
                continue
            nested: list[str] = []
            index += 1
            while index < len(expression) and expression[index] not in {";", "+"}:
                nested.append(expression[index])
                index += 1
            if not nested or index >= len(expression):
                raise CommandPolicyViolation(
                    f"cannot safely inspect unterminated find action {marker}"
                )
            clauses.append(
                _FindExecClause(
                    marker=cast(
                        Literal["-exec", "-execdir", "-ok", "-okdir"],
                        marker,
                    ),
                    command=tuple(nested),
                )
            )
            index += 1
        return _FindInvocation(
            roots=tuple(roots),
            path_events=tuple(path_events),
            exec_clauses=tuple(clauses),
            writes_roots=writes_roots,
        )

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
                clause.marker in {"-execdir", "-okdir"}
                and self._is_relative_file_operand(raw_path)
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
        active_environment = next(
            (name for name in _IMPLICIT_SHELL_ENVIRONMENT if os.environ.get(name)),
            None,
        )
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
        states = self._validate_shell_states(command, state)
        if len(states) != 1:
            raise CommandPolicyViolation(
                "cannot propagate multiple shell directory states"
            )
        return states[0]

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
            notify_observer=True,
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
            notify_observer=False,
        )

    def _resolve_policy_path(
        self,
        raw_path: str,
        cwd: Path,
        access: PathAccess,
        *,
        include_external_dirs: bool,
        notify_observer: bool,
    ) -> Path:
        if isinstance(raw_path, _CommandValue) and not raw_path.is_static:
            raise CommandPolicyViolation(
                "cannot resolve dynamic path operand; enumerate concrete paths "
                "instead of using active globs or shell expansions"
            )

        if notify_observer and self._path_access_observer is not None:
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
                include_external_dirs=include_external_dirs,
            )
        except ValueError as exc:
            raise CommandPathViolation(access=access, path=candidate) from exc
