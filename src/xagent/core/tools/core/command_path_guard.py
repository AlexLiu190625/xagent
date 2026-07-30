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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    SupportsIndex,
    cast,
)

try:
    import bashlex
except ModuleNotFoundError as _bashlex_import_error:  # pragma: no cover
    # ``bashlex`` ships in the optional ``command-path-guard`` extra, not the
    # base dependencies. Import lazily so importing this module (and collecting
    # its tests) never fails on a plain install; construction fails closed with
    # an actionable hint when the parser is genuinely required.
    bashlex = None
    _BASHLEX_IMPORT_ERROR: ModuleNotFoundError | None = _bashlex_import_error
else:
    _BASHLEX_IMPORT_ERROR = None

from ...workspace import TaskWorkspace
from .command_policy import (
    CommandPathViolation,
    CommandPolicyViolation,
    PathAccess,
    resolve_trusted_executable,
)

logger = logging.getLogger(__name__)

_MAX_COMMAND_POLICY_INPUT_CHARS = 64 * 1024
# Bounded budget for reading script/config files off disk before inspection.
# Numerically equal to the character input cap but kept in its own byte domain
# (compared against ``os.fstat().st_size`` and ``len(raw_bytes)``) so the two
# limits can move independently instead of silently coupling chars to bytes.
_MAX_INSPECTED_SCRIPT_BYTES = 64 * 1024
_MAX_INSPECTED_SCRIPT_DEPTH = 8
_MAX_COMMAND_WRAPPER_DEPTH = 32
_MAX_POSSIBLE_SHELL_STATES = 16
_MAX_POLICY_PARSE_ATTEMPTS = 64
_MAX_POLICY_SOURCE_CHARS = 1024 * 1024
_MAX_POLICY_SCRIPT_BYTES = 1024 * 1024
_MAX_POLICY_NODE_STATE_EVALUATIONS = 4096
_MAX_POLICY_ARGV_TOKENS = 8192


@dataclass
class _EffectScope:
    prior_written_paths: frozenset[Path] = frozenset()
    prior_unknown_effect: bool = False
    written_paths: set[Path] = field(default_factory=set)
    unknown_effect: bool = False
    inspected_scripts: set[Path] = field(default_factory=set)

    @property
    def visible_written_paths(self) -> frozenset[Path]:
        return self.prior_written_paths | self.written_paths

    @property
    def has_visible_unknown_effect(self) -> bool:
        return self.prior_unknown_effect or self.unknown_effect

    def inspect_script(self, script_path: Path) -> None:
        if self.has_visible_unknown_effect or script_path in self.visible_written_paths:
            raise CommandPolicyViolation(
                "cannot inspect a script affected by an earlier command"
            )
        self.inspected_scripts.add(script_path)

    def merge(self, child: _EffectScope) -> None:
        self.written_paths.update(child.written_paths)
        self.unknown_effect = self.unknown_effect or child.unknown_effect
        self.inspected_scripts.update(child.inspected_scripts)


@dataclass
class _ValidationSession:
    parse_attempts: int = 0
    source_chars: int = 0
    script_bytes: int = 0
    node_state_evaluations: int = 0
    argv_tokens: int = 0
    wrapped_bash_dispatch: bool = False
    effects: _EffectScope = field(default_factory=_EffectScope)
    # `find`-scoped side channel: while set, every `_check_path` resolution
    # reports its (raw_path, access) pair here before any short-circuit, so a
    # `-exec`/`-execdir`/`-ok`/`-okdir` clause's write-ness can be classified
    # from the same single enforcement pass that validates it. `None` outside
    # find's own clause enforcement; `_check_find` saves and restores the
    # prior value around every entry (including nested `find`) so an inner
    # find's clause classification can never leak into an outer one.
    find_clause_observer: Callable[[str, PathAccess], None] | None = None

    def charge_parse(self, source_chars: int) -> None:
        self.parse_attempts += 1
        if self.parse_attempts > _MAX_POLICY_PARSE_ATTEMPTS:
            raise CommandPolicyViolation("command policy parse attempt budget exceeded")
        self.source_chars += source_chars
        if self.source_chars > _MAX_POLICY_SOURCE_CHARS:
            raise CommandPolicyViolation(
                "command policy source character budget exceeded"
            )

    def charge_script_bytes(self, script_bytes: int) -> None:
        self.script_bytes += script_bytes
        if self.script_bytes > _MAX_POLICY_SCRIPT_BYTES:
            raise CommandPolicyViolation("command policy script byte budget exceeded")

    def charge_node_state_evaluation(self) -> None:
        self.node_state_evaluations += 1
        if self.node_state_evaluations > _MAX_POLICY_NODE_STATE_EVALUATIONS:
            raise CommandPolicyViolation(
                "command policy node-state evaluation budget exceeded"
            )

    def charge_argv_tokens(self, argv_tokens: int) -> None:
        self.argv_tokens += argv_tokens
        if self.argv_tokens > _MAX_POLICY_ARGV_TOKENS:
            raise CommandPolicyViolation("command policy argv token budget exceeded")


_VALIDATION_SESSION: ContextVar[_ValidationSession | None] = ContextVar(
    "command_validation_session",
    default=None,
)


@contextmanager
def _validation_session_scope() -> Iterator[_ValidationSession]:
    session = _VALIDATION_SESSION.get()
    token = None
    if session is None:
        session = _ValidationSession()
        token = _VALIDATION_SESSION.set(session)
    try:
        yield session
    finally:
        if token is not None:
            _VALIDATION_SESSION.reset(token)


def _active_validation_session() -> _ValidationSession:
    session = _VALIDATION_SESSION.get()
    if session is None:
        raise RuntimeError("command validation requires an active session")
    return session


# Plain readers: every operand is validated as a workspace read. A command
# whose grammar carries a write slot or a read-control file argument (that
# argument is itself a path, not a value to skip) gets its own dedicated
# handler below instead and must not be added here.
_READ_COMMANDS = {
    "cat",
    "cmp",
    "cut",
    "file",
    "head",
    "less",
    "ls",
    "more",
    "stat",
    "tac",
    "tail",
    "wc",
}
# Path-scoped writers: every operand is validated as a workspace write and
# recorded per-path, so they never poison the session-wide unknown-effect flag.
# `chmod`/`chown`/`chgrp` own a leading non-path positional (mode or
# owner[:group]/group spec) that `_check_write_command` strips before the
# write check; every other member here has no non-path positional.
_WRITE_COMMANDS = {
    "chgrp",
    "chmod",
    "chown",
    "mkdir",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}
# Subset of `_WRITE_COMMANDS` whose first non-option operand is a mode or
# owner/group spec, not a path; `--reference=RFILE` supplies that role from a
# file instead, so no positional is stripped when it is present.
_OWNERSHIP_MODE_COMMANDS = frozenset({"chmod", "chown", "chgrp"})
_SHELL_COMMANDS = {"bash"}
_UNSUPPORTED_SHELL_COMMANDS = {"dash", "sh", "zsh"}
_UNSUPPORTED_PRIVILEGE_COMMANDS = {"sudo"}
# Commands with no filesystem write effect of their own. Redirections are
# validated separately, so these can never mutate a later-inspected script and
# must not poison the session-wide unknown-effect flag. Keep this conservative:
# only add a command here once it is known to be unable to write the filesystem.
_NO_FILESYSTEM_EFFECT_COMMANDS = {
    "echo",
    "printf",
    "pwd",
    "true",
    "false",
    ":",
    "test",
    "[",
    "export",
    "declare",
    "typeset",
}
# Options that consume a following token as their value, per classified command,
# so the value is not misread as a path operand (e.g. `mkdir -m 0755 dir`).
# Only options whose value is never a path belong here; a value option that is
# itself a path (e.g. `wc --files0-from`) needs a dedicated handler instead so
# the argument gets checked, not skipped.
_COMMAND_VALUE_OPTIONS = {
    "mkdir": frozenset({"-m", "--mode"}),
    "cut": frozenset(
        {
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
    ),
    "head": frozenset({"-c", "--bytes", "-n", "--lines"}),
    "tail": frozenset({"-c", "--bytes", "-n", "--lines"}),
    "tac": frozenset({"-s", "--separator"}),
}
_BASH_FILE_OPTIONS = {"--init-file", "--rcfile"}
_BASH_LONG_FLAG_OPTIONS = {
    "--debug",
    "--debugger",
    "--dump-po-strings",
    "--dump-strings",
    "--help",
    "--login",
    "--noediting",
    "--noprofile",
    "--norc",
    "--posix",
    "--pretty-print",
    "--protected",
    "--restricted",
    "--verbose",
    "--version",
    "--wordexp",
}
_BASH_SHORT_FLAG_OPTIONS = frozenset("abefhiklmnprstuvxBCDHP")
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
_POLICY_TIME_WRAPPER = "__t_"


def _secure_script_open_flags() -> int:
    """Return mandatory flags for race-resistant, non-blocking script reads."""
    nonblocking = getattr(os, "O_NONBLOCK", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nonblocking, int) or not isinstance(nofollow, int):
        raise CommandPolicyViolation(
            "secure direct-script inspection is unavailable on this platform"
        )
    return os.O_RDONLY | nonblocking | nofollow


@dataclass(frozen=True)
class _HeredocDeclaration:
    delimiter: str
    strip_tabs: bool
    quoted: bool


def _is_time_keyword_position(source: str, position: int) -> bool:
    if not source.startswith("time", position):
        return False
    after = position + len("time")
    if after >= len(source) or source[after] not in " \t":
        return False
    cursor = position - 1
    while cursor >= 0 and source[cursor] in " \t":
        cursor -= 1
    if cursor < 0 or source[cursor] in ";\n":
        return True
    return source[max(0, cursor - 1) : cursor + 1] in {"&&", "||"}


def _parse_heredoc_delimiter(
    source: str,
    start: int,
    normalized: list[str],
) -> tuple[_HeredocDeclaration | None, int]:
    cursor = start
    while cursor < len(source) and source[cursor] in " \t":
        cursor += 1
    if cursor >= len(source) or source[cursor] in "\r\n;|&()<>":
        return None, cursor

    token_start = cursor
    delimiter: list[str] = []
    quote: str | None = None
    quoted = False
    while cursor < len(source):
        character = source[cursor]
        if quote is None and character in " \t\r\n;|&()<>":
            break
        if quote is None and character in {"'", '"'}:
            quote = character
            quoted = True
            normalized[cursor] = " "
        elif quote == character:
            quote = None
            normalized[cursor] = " "
        elif quote is None and character == "\\":
            quoted = True
            normalized[cursor] = " "
            cursor += 1
            if cursor >= len(source):
                return None, cursor
            delimiter.append(source[cursor])
        else:
            delimiter.append(character)
        cursor += 1

    if quote is not None or not delimiter:
        return None, cursor
    delimiter_value = "".join(delimiter)
    if quoted:
        normalized[token_start:cursor] = delimiter_value.ljust(cursor - token_start)
    return (
        _HeredocDeclaration(
            delimiter=delimiter_value,
            strip_tabs=False,
            quoted=quoted,
        ),
        cursor,
    )


def _consume_heredoc_bodies(
    source: str,
    start: int,
    declarations: Sequence[_HeredocDeclaration],
    normalized: list[str],
) -> int:
    cursor = start
    for declaration in declarations:
        while cursor <= len(source):
            line_end = source.find("\n", cursor)
            if line_end < 0:
                line_end = len(source)
            line = source[cursor:line_end]
            comparable = line.lstrip("\t") if declaration.strip_tabs else line
            if comparable == declaration.delimiter:
                cursor = line_end + (line_end < len(source))
                break
            if declaration.quoted:
                for index in range(cursor, line_end):
                    normalized[index] = "x"
            if line_end == len(source):
                raise CommandPolicyViolation("cannot safely parse shell command")
            cursor = line_end + 1
        else:
            raise CommandPolicyViolation("cannot safely parse shell command")
    return cursor


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
    declarations: list[_HeredocDeclaration] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False
    in_comment = False
    cursor = 0

    while cursor < len(command):
        character = command[cursor]
        if in_comment:
            if character == "\n":
                in_comment = False
            else:
                cursor += 1
                continue
        if escaped:
            escaped = False
            cursor += 1
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
            cursor += 1
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            cursor += 1
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            cursor += 1
            continue
        if in_single_quote or in_double_quote:
            cursor += 1
            continue

        if character == "#" and (cursor == 0 or command[cursor - 1] in " \t\r\n;|&()"):
            in_comment = True
            cursor += 1
            continue
        if _is_time_keyword_position(command, cursor):
            normalized[cursor : cursor + len("time")] = _POLICY_TIME_WRAPPER
            cursor += len("time")
            continue
        if (
            command.startswith("<<", cursor)
            and not command.startswith("<<<", cursor)
            and (cursor == 0 or command[cursor - 1] != "<")
        ):
            strip_tabs = command.startswith("<<-", cursor)
            delimiter_start = cursor + (3 if strip_tabs else 2)
            declaration, delimiter_end = _parse_heredoc_delimiter(
                command,
                delimiter_start,
                normalized,
            )
            if declaration is None:
                raise CommandPolicyViolation("cannot safely parse shell command")
            declarations.append(replace(declaration, strip_tabs=strip_tabs))
            cursor = delimiter_end
            continue
        if character == "\n" and declarations:
            cursor = _consume_heredoc_bodies(
                command,
                cursor + 1,
                declarations,
                normalized,
            )
            declarations.clear()
            continue
        cursor += 1

    if declarations or in_single_quote or in_double_quote or escaped:
        raise CommandPolicyViolation("cannot safely parse shell command")
    return "".join(normalized)


class _CommandValue(str):
    """A shell word plus whether bashlex proved it has no runtime expansion."""

    is_static: bool
    # Arity-preserving no-op resolution target for this exact value, set only
    # on the specific `{}` word tagged by `find`'s exec/execdir/ok/okdir clause
    # enforcement (see `_tag_find_placeholder`). The tag rides on the value
    # itself rather than on any shared instance/session state, so it cannot
    # outlive the one call it was built for and cannot leak into unrelated
    # path resolution (e.g. a literal `{}` filename outside `find`).
    find_placeholder_cwd: Path | None

    def __new__(
        cls,
        value: str,
        *,
        is_static: bool = True,
        find_placeholder_cwd: Path | None = None,
    ) -> _CommandValue:
        instance = super().__new__(cls, value)
        instance.is_static = is_static
        instance.find_placeholder_cwd = find_placeholder_cwd
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
    consumes_leading_scalar: bool = False
    allows_assignments: bool = False
    cwd_options: frozenset[str] = frozenset()
    terminal_options: frozenset[str] = frozenset()
    rejected_options: frozenset[str] = frozenset()
    propagates_state: bool = False


@dataclass(frozen=True)
class _ScriptCommandGrammar:
    """`sed`/`awk` argv shape: script source options plus file operands.

    `value_options` holds long option names (`--...`) that take a mandatory
    scalar (non-path) value, such as sed's `--line-length`; the matching
    short spelling, if any, belongs in `short_value_options` instead.
    `optional_write_options` holds awk's gawk-style `-o`/`--pretty-print`,
    `-p`/`--profile`, `-d`/`--dump-variables` options, keyed by BOTH the
    short (`-x`) and long (`--xxx`) spelling, mapped to the fixed filename
    gawk uses when no explicit argument is given; unlike `value_options`,
    these never consume a separate token (GNU optional-argument
    convention: the value, if given, must be attached). `flag_long_options`
    holds long options that take NO value and name NO path at all (sed's
    `--quiet`/`--posix`/etc.); each is consumed as an inert flag, unlike
    `value_options`, which always consumes a following/attached token.
    """

    language: Literal["sed", "awk"]
    expression_long_option: str
    value_options: frozenset[str] = frozenset()
    ignores_assignment_arguments: bool = False
    short_flag_options: frozenset[str] = frozenset()
    short_value_options: frozenset[str] = frozenset()
    in_place_short_option: str | None = None
    in_place_long_option: str | None = None
    optional_write_options: Mapping[str, str] = field(default_factory=dict)
    flag_long_options: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _ScriptShortOptionResult:
    next_index: int
    explicit_script: bool = False
    requests_in_place: bool = False
    has_backup_suffix: bool = False


@dataclass(frozen=True)
class _BashInvocation:
    initialization_files: tuple[str, ...]
    command_text: str | None = None
    script: str | None = None
    reads_stdin: bool = False
    lists_options: bool = False


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
        consumes_leading_scalar=True,
        terminal_options=frozenset({"--help", "--version"}),
    ),
}

# `-l`/`--line-length` (sed) and `-F`/`-v` (awk) take a scalar value that is
# never a path, so they are skipped rather than routed through the file-
# operand check. Both families bundle short options (e.g. sed's `-ne`,
# `-i.bak`; awk's `-Fx`, gawk's `-o[file]`), so both list at least one
# `short_value_options`/`optional_write_options` entry. The long-only flags
# below take no value and name no path (verified against GNU sed's own
# option list); `--line-length` already has a value entry above and stays
# there — it is the one long option in this family that does carry a value.
_SED_GRAMMAR = _ScriptCommandGrammar(
    language="sed",
    expression_long_option="--expression",
    value_options=frozenset({"--line-length"}),
    short_flag_options=frozenset({"E", "n", "r", "s", "u", "z"}),
    short_value_options=frozenset({"l"}),
    in_place_short_option="i",
    in_place_long_option="--in-place",
    flag_long_options=frozenset(
        {
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
        }
    ),
)
_AWK_GRAMMAR = _ScriptCommandGrammar(
    language="awk",
    expression_long_option="--source",
    short_value_options=frozenset({"F", "v"}),
    ignores_assignment_arguments=True,
    # gawk's optional-argument write options: the value, if present, is
    # always attached (never a separate token); the default is the fixed
    # filename gawk itself writes when no argument is given.
    optional_write_options={
        "-o": "awkprof.out",
        "--pretty-print": "awkprof.out",
        "-p": "awkprof.out",
        "--profile": "awkprof.out",
        "-d": "awkvars.out",
        "--dump-variables": "awkvars.out",
    },
)
# `system(...)` always executes an arbitrary shell command; `print`/`printf`
# redirection and `getline` I/O are located structurally instead (see
# `_check_awk_program`), since their file targets must be classified as a
# read or write path rather than a blanket rejection.
_AWK_SYSTEM_CALL_PATTERN = re.compile(r"\bsystem\s*\(")
_AWK_PRINT_PATTERN = re.compile(r"\b(?:print|printf)\b")
_AWK_GETLINE_PATTERN = re.compile(r"\bgetline\b")
_AWK_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Commands routed to a dedicated per-family handler in
# `_validate_command_values` (own a slot — write/read split, target-directory,
# every-operand-write-sensitivity, script-source parsing, ... — a flat
# read/write classification cannot express). This is the single source of
# truth for that dispatch set: `_CLASSIFIED_EXECUTABLE_COMMANDS` below and the
# test suite's coverage registry both derive from it, so a command dispatched
# here without a matching test entry fails the coverage gate instead of
# silently going untested.
_DEDICATED_HANDLER_COMMANDS = frozenset(
    {
        "sort",
        "uniq",
        "diff",
        "grep",
        "cp",
        "install",
        "mv",
        "ln",
        "unlink",
        "shred",
        "find",
        "tar",
        "sed",
        "awk",
        "dd",
        "base64",
        "gzip",
        "rsync",
        "curl",
        "wget",
    }
)

_CLASSIFIED_EXECUTABLE_COMMANDS = {
    *_READ_COMMANDS,
    *_WRITE_COMMANDS,
    *_SHELL_COMMANDS,
    *_UNSUPPORTED_SHELL_COMMANDS,
    *_UNSUPPORTED_PRIVILEGE_COMMANDS,
    *_DEDICATED_HANDLER_COMMANDS,
    "chroot",
    "xargs",
    *(
        name
        for name in _COMMAND_WRAPPER_GRAMMARS
        if name not in {"builtin", "command", "exec", _POLICY_TIME_WRAPPER}
    ),
}

# `sort`'s short options bundle into one token (e.g. `-no file` combines the
# `-n` flag with the `-o` write-path option), so its grammar is split into a
# flag set (no value) and a value set (one following/attached argument, with
# a read/write/skip access) for the short-option-cluster parser.
_SORT_SHORT_FLAG_OPTIONS = frozenset("bdfgiMhnRrVcCmsuz")
_SORT_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {
    "k": None,
    "o": "write",
    "S": None,
    "t": None,
    "T": "write",
}
_SORT_LONG_PATH_OPTIONS: dict[str, PathAccess] = {
    "--files0-from": "read",
    "--output": "write",
    "--random-source": "read",
    "--temporary-directory": "write",
}
_SORT_LONG_SCALAR_OPTIONS = frozenset(
    {
        "--batch-size",
        "--buffer-size",
        "--field-separator",
        "--key",
        "--parallel",
        "--sort",
    }
)
_SORT_LONG_FLAG_OPTIONS = frozenset(
    {
        "--check",
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
)
# `--compress-program` delegates to an arbitrary external program; this is a
# hard denial regardless of abbreviation, not a plain unknown option.
_SORT_DENIED_LONG_OPTIONS = frozenset({"--compress-program"})
_SORT_KNOWN_LONG_OPTIONS = frozenset(
    {
        *_SORT_LONG_PATH_OPTIONS,
        *_SORT_LONG_SCALAR_OPTIONS,
        *_SORT_LONG_FLAG_OPTIONS,
        *_SORT_DENIED_LONG_OPTIONS,
    }
)

# `grep`'s short options bundle into one token (e.g. `-if file` combines the
# `-i` flag with the `-f` pattern-file option), so its grammar is split into a
# flag set (no value) and a value set (one following/attached argument, with
# a read/scalar access) for the short-option-cluster parser, the same shape
# `sort` uses. `-e`/`-f` also carry the (never-a-path) pattern text/pattern
# file, so they are not paths themselves in this map; only `-f`'s argument is.
_GREP_SHORT_FLAG_OPTIONS = frozenset("EFGPivwxcLloqsaIHhnTZzrRVb")
_GREP_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {
    "e": None,
    "f": "read",
    "m": None,
    "A": None,
    "B": None,
    "C": None,
    "D": None,
    "d": None,
}
# grep long options that consume a value which is never a path (a pattern,
# label, count, or action keyword), so the value must be skipped as a unit
# rather than left in the token stream for `--label`'s argument (e.g., a real
# path) to slide into the implicit-pattern positional slot (R6).
_GREP_LONG_VALUE_OPTIONS = frozenset(
    {
        "--label",
        "--include",
        "--exclude",
        "--exclude-dir",
        "--binary-files",
        "--context",
        "--after-context",
        "--before-context",
        "--max-count",
        "--directories",
        "--devices",
    }
)
_GREP_KNOWN_LONG_OPTIONS = (
    frozenset({"--regexp", "--file", "--exclude-from"}) | _GREP_LONG_VALUE_OPTIONS
)

# A `curl`/`wget` positional operand is a URL, not a bare filesystem path,
# EXCEPT when its scheme is `file:`, which is a real local filesystem
# channel. These are the network schemes this family recognizes as
# definitely non-local; any other or unparsable scheme fails closed rather
# than being assumed safe, since an exotic scheme this map does not model
# could still resolve to a local resource.
_CURL_WGET_NETWORK_URL_SCHEMES = frozenset(
    {"http", "https", "ftp", "ftps", "sftp", "scp"}
)
_URL_SCHEME_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):(//)?(.*)\Z", re.DOTALL)

# `curl`'s short options bundle into one token (e.g. `-sD file` combines the
# `-s` flag with the `-D` write-path option), so its grammar is split into a
# flag set and a value set for the short-option-cluster parser, the same
# shape `sort`/`grep`/`install` use. `-d`'s value is only conditionally a
# path (an `@file` payload), so it is modeled here as a plain (unchecked)
# value and given its own `@`-prefix classification by the caller.
_CURL_SHORT_FLAG_OPTIONS = frozenset("sSLkfiIvNgGq#")
_CURL_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {
    "o": "write",
    "D": "write",
    "c": "write",
    "T": "read",
    "d": None,
}
_CURL_HARD_DENY_LONG_OPTIONS = frozenset({"--remote-name", "--remote-name-all"})
_CURL_CONFIG_LONG_OPTIONS = frozenset({"--config"})
_CURL_WRITE_LONG_OPTIONS = frozenset(
    {
        "--output",
        "--output-dir",
        "--dump-header",
        "--cookie-jar",
        "--trace",
        "--trace-ascii",
        "--stderr",
        "--libcurl",
        "--hsts",
        "--etag-save",
    }
)
_CURL_READ_LONG_OPTIONS = frozenset({"--upload-file", "--netrc-file", "--cookie"})
_CURL_DATA_LONG_OPTIONS = frozenset({"--data", "--data-binary", "--data-raw"})
# Long-form spellings of `_CURL_SHORT_FLAG_OPTIONS`'s no-value flags, so the
# common long spelling of an already-modeled short flag is not rejected
# merely for being spelled out.
_CURL_KNOWN_LONG_FLAG_OPTIONS = frozenset(
    {
        "--silent",
        "--show-error",
        "--location",
        "--insecure",
        "--fail",
        "--include",
        "--head",
        "--verbose",
        "--no-buffer",
        "--globoff",
        "--get",
        "--progress-bar",
    }
)
_CURL_KNOWN_LONG_OPTIONS = frozenset(
    _CURL_HARD_DENY_LONG_OPTIONS
    | _CURL_CONFIG_LONG_OPTIONS
    | _CURL_WRITE_LONG_OPTIONS
    | _CURL_READ_LONG_OPTIONS
    | _CURL_DATA_LONG_OPTIONS
    | _CURL_KNOWN_LONG_FLAG_OPTIONS
)

# `wget`'s own short options never bundle a value-taking option behind a
# flag in this family's existing tests or documented usage, but the same
# split keeps its long-option handling consistent with curl/rsync/sort.
_WGET_HARD_DENY_LONG_OPTIONS = frozenset({"--input-file"})
_WGET_CONFIG_LONG_OPTIONS = frozenset({"--config"})
_WGET_WRITE_LONG_OPTIONS = frozenset(
    {
        "--output-document",
        "--directory-prefix",
        "--output-file",
        "--save-cookies",
    }
)
_WGET_READ_LONG_OPTIONS = frozenset({"--post-file", "--body-file"})
# Long-form spellings of `_WGET_SHORT_FLAG_OPTIONS`'s no-value flags, for the
# same reason `_CURL_KNOWN_LONG_FLAG_OPTIONS` exists.
_WGET_KNOWN_LONG_FLAG_OPTIONS = frozenset(
    {
        "--background",
        "--quiet",
        "--verbose",
        "--continue",
        "--timestamping",
        "--server-response",
        "--debug",
        "--force-html",
    }
)
_WGET_KNOWN_LONG_OPTIONS = frozenset(
    _WGET_HARD_DENY_LONG_OPTIONS
    | _WGET_CONFIG_LONG_OPTIONS
    | _WGET_WRITE_LONG_OPTIONS
    | _WGET_READ_LONG_OPTIONS
    | _WGET_KNOWN_LONG_FLAG_OPTIONS
    | {"--spider"}
)
_WGET_SHORT_FLAG_OPTIONS = frozenset("bqvcNSdF")
_WGET_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {
    "O": "write",
    "P": "write",
    "o": "write",
}

# `rsync` has no short-option grammar today beyond a crude bundled-character
# scan for the highest-risk options; this builds one. `-T`/`--temp-dir` is
# the only short option that carries a path. `-K`/`--keep-dirlinks` and
# `-k`/`--copy-dirlinks` let rsync write through (or read through) an
# existing destination/source-side symlink instead of the literal directory
# entry, which can escape the intended root, so both are denied outright —
# matching `--keep-dirlinks`'s existing long-option denial — rather than
# access-classified. `-e`/`-f`/`-L`/`-H` mirror the long options already
# denied below (`--rsh`, `--filter`, `--copy-links`; `-H`/`--hard-links` is
# conservatively denied too, matching this family's pre-existing posture).
_RSYNC_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {"T": "write"}
_RSYNC_SHORT_DENIED_OPTIONS = frozenset("efLHKk")
_RSYNC_SHORT_FLAG_OPTIONS = frozenset("vqarRbudlpAXogDtOJnWxzCIm8h4y6sP")

# `base64`'s short options bundle into one token (e.g. `-do<file>` combines
# the `-d` decode flag with the `-o` output-path option), so its grammar is
# split into a flag set and a value set for the short-option-cluster parser,
# the same shape `sort`/`grep`/`curl`/`wget` use. `-b`/`-w` take a scalar
# column-width value that is never a path. `-i`/`--ignore-garbage` is a
# BOOLEAN flag (GNU coreutils: discard non-alphabet input when decoding
# instead of erroring), never value-taking: modeling it as value-taking
# would let it silently swallow the NEXT token as its own (read-checked)
# argument even when that token is itself another option (e.g. `-o`'s own
# write-path flag), so `-o`'s real argument would fall through unclassified
# as a plain read operand instead of being write-checked.
_BASE64_SHORT_FLAG_OPTIONS = frozenset("dDhi")
_BASE64_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {
    "o": "write",
    "b": None,
    "w": None,
}
_BASE64_LONG_FLAG_OPTIONS = frozenset({"--ignore-garbage"})
_BASE64_KNOWN_LONG_OPTIONS = frozenset(
    {"--output", "--break", "--wrap"} | _BASE64_LONG_FLAG_OPTIONS
)

# `gzip`'s short options bundle into one token (e.g. `-lt`, `-9v`); unlike
# the other bundled-cluster families, more than one character in the bundle
# is independently meaningful for the read-only-mode classification, so
# gzip owns its own cluster parser (`_parse_gzip_short_cluster`) instead of
# `_parse_short_option_cluster`. `-S`/`--suffix` takes a scalar value that is
# never a path. `-c`/`--stdout`/`--to-stdout`, `-l`/`--list`, and
# `-t`/`--test` never create or remove a file, so they keep the operand a
# plain read instead of the default mode's write.
_GZIP_SHORT_FLAG_OPTIONS = frozenset("acdfhklnNqrtvV123456789")
_GZIP_READ_ONLY_SHORT_FLAGS = frozenset("clt")
_GZIP_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {"S": None}
_GZIP_LONG_FLAG_OPTIONS = frozenset(
    {
        "--ascii",
        "--decompress",
        "--uncompress",
        "--force",
        "--help",
        "--keep",
        "--no-name",
        "--name",
        "--quiet",
        "--recursive",
        "--verbose",
        "--version",
        "--fast",
        "--best",
    }
)
_GZIP_READ_ONLY_LONG_OPTIONS = frozenset(
    {"--stdout", "--to-stdout", "--list", "--test"}
)
_GZIP_SCALAR_LONG_OPTIONS = frozenset({"--suffix"})
_GZIP_KNOWN_LONG_OPTIONS = frozenset(
    _GZIP_LONG_FLAG_OPTIONS | _GZIP_READ_ONLY_LONG_OPTIONS | _GZIP_SCALAR_LONG_OPTIONS
)

# `cp`/`mv`/`ln` share `_parse_target_directory`'s `-t`/`--target-directory`
# extraction. Short options bundle into one token (e.g. `-rt` combines the
# `-r` flag with the `-t` write-path option), so the grammar is split into a
# flag set and a value set for the short-option-cluster parser, the union of
# the three commands' real GNU short options (over-permissive for any one of
# them is safe here: none of these carry a path this parser would otherwise
# miss). `-S`'s value is a scalar backup suffix, never a path.
_TARGET_DIR_SHORT_FLAG_OPTIONS = frozenset("abdfFHilLnPprRsuvxZ")
_TARGET_DIR_SHORT_VALUE_OPTIONS: dict[str, PathAccess | None] = {
    "t": "write",
    "S": None,
}
_TARGET_DIR_LONG_FLAG_OPTIONS = frozenset(
    {
        "--archive",
        "--attributes-only",
        "--backup",
        "--copy-contents",
        "--dereference",
        "--follow-command-line-symlink",
        "--force",
        "--help",
        "--interactive",
        "--link",
        "--logical",
        "--no-clobber",
        "--no-dereference",
        "--no-target-directory",
        "--one-file-system",
        "--parents",
        "--physical",
        "--preserve",
        "--recursive",
        "--relative",
        "--remove-destination",
        "--strip-trailing-slashes",
        "--symbolic-link",
        "--update",
        "--verbose",
        "--version",
    }
)
_TARGET_DIR_LONG_SCALAR_OPTIONS = frozenset({"--suffix", "--sparse", "--context"})
_TARGET_DIR_KNOWN_LONG_OPTIONS = frozenset(
    {"--target-directory"}
    | _TARGET_DIR_LONG_FLAG_OPTIONS
    | _TARGET_DIR_LONG_SCALAR_OPTIONS
)


@dataclass(frozen=True)
class _PathEvent:
    """One fixed (non-`{}`) find operand and the access it must be checked as."""

    value: str
    access: PathAccess


@dataclass(frozen=True)
class _FindExecClause:
    """One `-exec`/`-execdir`/`-ok`/`-okdir ... ;`/`+` clause, unparsed inside."""

    marker: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class _FindInvocation:
    roots: tuple[str, ...]
    path_events: tuple[_PathEvent, ...]
    exec_clauses: tuple[_FindExecClause, ...]
    writes_delete: bool = False


# `-fprint`/`-fprint0`/`-fls` write one filename argument; `-fprintf` writes
# the same filename argument followed by a separate format-string argument
# that is never itself a path.
_FIND_OUTPUT_ACTIONS: dict[str, int] = {
    "-fprint": 1,
    "-fprint0": 1,
    "-fls": 1,
    "-fprintf": 2,
}
_FIND_REFERENCE_PREDICATES = frozenset({"-newer", "-anewer", "-cnewer", "-samefile"})
_FIND_EXEC_MARKERS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})

_TarMode = Literal[
    "create",
    "append",
    "update",
    "concatenate",
    "delete",
    "extract",
    "list",
    "compare",
]

# Every parsed tar argument's role in path classification. `positional`
# members are only local paths in create/append/update/concatenate/compare
# mode (list/extract members are in-archive selectors, not filesystem paths).
_TarEventKind = Literal[
    "archive",
    "directory",
    "files_from",
    "exclude_from",
    "incremental",
    "add_file",
    "positional",
    "dangerous",
    "absolute_names",
    "verbose_output",
    "unresolved_option",
    "pattern",
]


@dataclass(frozen=True)
class _TarEvent:
    kind: _TarEventKind
    value: str | None = None


@dataclass(frozen=True)
class _TarInvocation:
    mode: _TarMode
    events: tuple[_TarEvent, ...]
    remove_files: bool = False


class WorkspaceCommandPathGuard:
    """Validate Bash language boundaries against one task workspace."""

    def __init__(self, workspace: TaskWorkspace) -> None:
        if bashlex is None:
            raise ModuleNotFoundError(
                "WorkspaceCommandPathGuard requires the 'bashlex' parser. Install "
                "the optional extra: pip install 'xagent[command-path-guard]'."
            ) from _BASHLEX_IMPORT_ERROR
        self._workspace = workspace
        self._initial_cwd = workspace.resolve_path("").resolve()

    @property
    def execution_cwd(self) -> Path:
        """Return the canonical directory whose paths this guard validates."""
        return self._initial_cwd

    @staticmethod
    def _parse_shell(command: str) -> list[Any]:
        if len(command) > _MAX_COMMAND_POLICY_INPUT_CHARS:
            raise CommandPolicyViolation("command is too large to inspect")
        if "\x00" in command:
            raise CommandPolicyViolation("command contains a null byte")
        _active_validation_session().charge_parse(len(command))
        if not any(
            line.strip() and not line.lstrip().startswith("#")
            for line in command.splitlines()
        ):
            return []
        policy_source = _normalize_policy_shell_source(command)
        if bashlex is None:  # pragma: no cover - guarded at construction
            raise CommandPolicyViolation("shell command parser is unavailable")
        try:
            nodes = cast(list[Any], bashlex.parse(policy_source))
        except Exception as exc:
            # bashlex exposes several parser-internal exception types. Normalize
            # all of them at this boundary so malformed input never reaches the
            # executor through a parser-version-specific failure mode.
            #
            # Invariant: input bashlex cannot model (``$(( ))``, ``[[ ]]``,
            # ``case``, ``select``, ``coproc``, array assignments, ...) MUST stay
            # fail-closed here. Never relax this into a fail-open path to reduce
            # false positives on unparsable input.
            logger.debug("Command path guard rejected unparsed shell input")
            raise CommandPolicyViolation("cannot safely parse shell command") from exc
        _mark_unmodeled_expansions(nodes, policy_source)
        return nodes

    def validate(self, command: str) -> None:
        """Reject unsupported syntax and unsafe paths in ``command``."""
        with _validation_session_scope():
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
        self._validate_and_prepare_argv(argv, prepare_execution=False)

    def prepare_argv_for_execution(self, argv: Sequence[str]) -> list[str]:
        """Validate argv and bind direct Bash to its trusted executable identity."""
        return self._validate_and_prepare_argv(argv, prepare_execution=True)

    def _validate_and_prepare_argv(
        self,
        argv: Sequence[str],
        *,
        prepare_execution: bool,
    ) -> list[str]:
        with _validation_session_scope() as session:
            session.charge_argv_tokens(len(argv))
            execution_argv = list(argv)
            if not argv:
                return execution_argv
            if (
                prepare_execution
                and os.path.basename(execution_argv[0]) in _SHELL_COMMANDS
            ):
                execution_argv[0] = os.fspath(
                    resolve_trusted_executable(execution_argv[0])
                )
            self._validate_command_values(
                execution_argv[0],
                execution_argv[1:],
                _ShellState(cwd=self._initial_cwd),
                charge_argv=False,
            )
            if prepare_execution and session.wrapped_bash_dispatch:
                raise CommandPolicyViolation(
                    "cannot safely execute Bash through a command wrapper"
                )
            return execution_argv

    def _validate_node(self, node: Any, state: _ShellState) -> _ShellState:
        _active_validation_session().charge_node_state_evaluation()
        kind = getattr(node, "kind", None)
        if kind == "list":
            states = self._validate_list_states(node, (state,))
            if len(states) != 1:
                raise CommandPolicyViolation(
                    "cannot propagate multiple shell directory states"
                )
            return states[0]

        if kind == "pipeline":
            self._validate_pipeline(node, state)
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

    def _validate_pipeline(self, node: Any, state: _ShellState) -> None:
        session = _active_validation_session()
        parent_scope = session.effects
        member_scopes: list[_EffectScope] = []
        try:
            for part in node.parts:
                if getattr(part, "kind", None) == "pipe":
                    continue
                member_scope = _EffectScope(
                    prior_written_paths=parent_scope.visible_written_paths,
                    prior_unknown_effect=parent_scope.has_visible_unknown_effect,
                )
                session.effects = member_scope
                self._validate_node(part, state)
                member_scopes.append(member_scope)
        finally:
            session.effects = parent_scope

        unknown_member_count = sum(scope.unknown_effect for scope in member_scopes)
        write_owner_counts: dict[Path, int] = {}
        for member_scope in member_scopes:
            for path in member_scope.written_paths:
                write_owner_counts[path] = write_owner_counts.get(path, 0) + 1

        for member_scope in member_scopes:
            concurrent_unknown = unknown_member_count > int(member_scope.unknown_effect)
            concurrent_write = any(
                write_owner_counts.get(script_path, 0)
                > int(script_path in member_scope.written_paths)
                for script_path in member_scope.inspected_scripts
            )
            if member_scope.inspected_scripts and (
                concurrent_unknown or concurrent_write
            ):
                raise CommandPolicyViolation(
                    "cannot inspect a script affected by a concurrent command"
                )

        for member_scope in member_scopes:
            parent_scope.merge(member_scope)

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
        *,
        charge_argv: bool = True,
    ) -> _ShellState:
        if charge_argv:
            _active_validation_session().charge_argv_tokens(1 + len(args))
        command_name = os.path.basename(command_word)
        if command_name in _UNSUPPORTED_PRIVILEGE_COMMANDS:
            raise CommandPolicyViolation(
                f"cannot safely inspect privilege escalation via {command_name}"
            )
        if self._is_direct_command_path(command_word):
            if not self._is_trusted_system_command(command_word):
                self._inspect_direct_shell_script(command_word, state)
                return state
        elif command_name in _CLASSIFIED_EXECUTABLE_COMMANDS:
            if not self._is_trusted_system_command(command_word):
                discovered = shutil.which(command_name)
                if discovered is not None:
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
            self._check_write_command(command_name, args, state.cwd)
        elif command_name == "sort":
            self._check_sort(args, state.cwd)
        elif command_name == "uniq":
            self._check_uniq(args, state.cwd)
        elif command_name == "diff":
            self._check_diff(args, state.cwd)
        elif command_name == "grep":
            self._check_grep(args, state.cwd)
        elif command_name == "cp":
            self._check_copy(args, state.cwd)
        elif command_name == "install":
            self._check_install(args, state.cwd)
        elif command_name in {"mv", "ln"}:
            self._check_move_or_link(args, state.cwd)
        elif command_name in {"unlink", "shred"}:
            self._check_destructive_file_command(command_name, args, state.cwd)
        elif command_name == "find":
            self._check_find(args, state)
        elif command_name == "tar":
            self._check_tar(args, state.cwd)
        elif command_name == "sed":
            self._check_script_command(_SED_GRAMMAR, args, state.cwd)
        elif command_name == "awk":
            self._check_script_command(_AWK_GRAMMAR, args, state.cwd)
        elif command_name == "dd":
            self._check_dd(args, state.cwd)
        elif command_name == "base64":
            self._check_base64(args, state.cwd)
        elif command_name == "gzip":
            self._check_gzip(args, state.cwd)
        elif command_name == "rsync":
            self._check_rsync(args, state.cwd)
        elif command_name == "curl":
            self._check_curl(args, state.cwd)
        elif command_name == "wget":
            self._check_wget(args, state.cwd)
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
        elif command_name in _NO_FILESYSTEM_EFFECT_COMMANDS:
            # No filesystem write effect of its own; any redirection is validated
            # separately, so this cannot have mutated a later-inspected script.
            pass
        else:
            _active_validation_session().effects.unknown_effect = True
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

    @staticmethod
    def _reject_symlink_traversing_chdir(target_word: str, cwd: Path) -> None:
        """Reject a directory change whose target path crosses a symlink.

        The guard tracks a *physically* resolved cwd, but Bash's ``cd``/``pushd``
        (without ``-P``) are *logical*: they collapse ``..`` textually against
        the pre-symlink path instead of following symlinks. When a target both
        crosses a symlink and carries ``..`` segments, the guard's resolved cwd
        diverges from where Bash actually lands, so a later relative operand can
        resolve inside the workspace while Bash reads/writes a sibling tenant's
        files. Physical and logical resolution agree whenever no symlink is
        traversed, so rejecting symlink-crossing directory changes keeps the
        tracked cwd equal to Bash's logical cwd. Fail closed rather than emulate
        logical ``cd``. ``cd -``/``popd``/stack rotation restore an
        already-validated, symlink-free cwd and do not reach this check.
        """
        if isinstance(target_word, _CommandValue) and not target_word.is_static:
            # Dynamic targets are rejected by the resolver below; do not probe
            # an unexpanded literal against the filesystem.
            return
        target = Path(target_word).expanduser()
        if target.is_absolute():
            probe = Path(target.anchor)
            parts = target.parts[1:]
        else:
            probe = cwd
            parts = target.parts
        for part in parts:
            if part == ".":
                continue
            if part == "..":
                probe = probe.parent
                continue
            probe = probe / part
            try:
                crosses_symlink = probe.is_symlink()
            except OSError as exc:
                raise CommandPolicyViolation(
                    "cannot inspect directory change target"
                ) from exc
            if crosses_symlink:
                raise CommandPolicyViolation(
                    "cannot safely resolve a directory change that traverses a "
                    "symlink; Bash resolves 'cd' logically while the guard "
                    "resolves physically"
                )

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
                self._reject_symlink_traversing_chdir(target_word, state.cwd)
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
                self._reject_symlink_traversing_chdir(operands[0], state.cwd)
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

        if grammar.consumes_leading_scalar:
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
        if os.path.basename(command_word) in _SHELL_COMMANDS:
            _active_validation_session().wrapped_bash_dispatch = True
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
        *,
        value_options: frozenset[str] = frozenset(),
    ) -> None:
        for raw_path in self._operands(values, value_options=value_options):
            self._check_path(raw_path, cwd, access)

    def _check_read_command(
        self,
        command_name: str,
        values: Sequence[str],
        cwd: Path,
    ) -> None:
        """Dispatch a plain read-family command to its operand check.

        `file` and `wc` carry a read-control option whose value is itself a
        path (a pattern/magic file, or a NUL-delimited file-of-filenames), so
        they are partitioned through the access-map substrate first; every
        other read command is a flat operand check with only skip-valued
        options (`_COMMAND_VALUE_OPTIONS`). `_partition_path_options` already
        consumes/discards a `--` operand separator and returns pure operands
        (some legitimately starting with `-` if they followed `--`); feeding
        that result back through `_check_operands`/`_operands` would re-apply
        `_operands`'s own `-`-prefix filter with no memory of the `--` it
        already passed, silently dropping (and never read-checking) such an
        operand a second time (R1). So `file`/`wc` check their partitioned
        operands directly here instead of routing through `_check_operands`.
        """
        if command_name == "file":
            operands = self._partition_path_options(
                values,
                cwd,
                option_access={
                    "-f": "read",
                    "-m": "read",
                    "--files-from": "read",
                    "--magic-file": "read",
                },
                attached_short_options=frozenset({"-f", "-m"}),
                flag_short_options=frozenset(
                    {
                        "-b",
                        "-i",
                        "-h",
                        "-L",
                        "-s",
                        "-z",
                        "-k",
                        "-n",
                        "-p",
                        "-r",
                        "-v",
                        "-0",
                    }
                ),
            )
            for raw_path in operands:
                self._check_path(raw_path, cwd, "read")
            return
        if command_name == "wc":
            operands = self._partition_path_options(
                values,
                cwd,
                option_access={"--files0-from": "read"},
                flag_short_options=frozenset({"-l", "-c", "-w", "-m", "-L"}),
            )
            for raw_path in operands:
                self._check_path(raw_path, cwd, "read")
            return
        self._check_operands(
            values,
            cwd,
            "read",
            value_options=_COMMAND_VALUE_OPTIONS.get(command_name, frozenset()),
        )

    def _check_sort(self, values: Sequence[str], cwd: Path) -> None:
        """Classify `sort`'s write (`-o`/`-T`) and read-control path options.

        `sort` owns two write slots (`-o`/`--output`, `-T`/`--temporary-directory`)
        and two read-control slots (`--files0-from`, `--random-source`), so it
        cannot be a flat read command: an unrecognized option must fail closed
        rather than silently let a write slot bypass containment.
        """
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                for raw_path in values[index + 1 :]:
                    self._check_path(raw_path, cwd, "read")
                return

            value_text = str(value)
            if value_text.startswith("--"):
                raw_option, separator, _ = value_text.partition("=")
                option = self._resolve_long_option(raw_option, _SORT_KNOWN_LONG_OPTIONS)
                if option is None:
                    raise CommandPolicyViolation(
                        f"cannot safely resolve sort option {raw_option}"
                    )
                if option in _SORT_DENIED_LONG_OPTIONS:
                    raise CommandPolicyViolation(
                        f"sort option {option} cannot safely delegate execution"
                    )
                path_access = _SORT_LONG_PATH_OPTIONS.get(option)
                if path_access is not None:
                    argument: str
                    if separator:
                        argument = self._derived_value(
                            value, value[len(raw_option) + 1 :]
                        )
                        index += 1
                    elif index + 1 < len(values):
                        argument = values[index + 1]
                        index += 2
                    else:
                        raise CommandPolicyViolation(
                            f"missing sort argument for {option}"
                        )
                    self._check_path(argument, cwd, path_access)
                    continue
                if option in _SORT_LONG_SCALAR_OPTIONS:
                    if separator:
                        index += 1
                    elif index + 1 < len(values):
                        index += 2
                    else:
                        raise CommandPolicyViolation(
                            f"missing sort argument for {option}"
                        )
                    continue
                # The remaining recognized long options (`_SORT_LONG_FLAG_OPTIONS`,
                # including the optional-value `--check[=WHEN]`) take no
                # separate argument.
                index += 1
                continue

            if value_text.startswith("-") and value_text != "-":
                index, _, _ = self._parse_short_option_cluster(
                    values,
                    index,
                    cwd,
                    flag_options=_SORT_SHORT_FLAG_OPTIONS,
                    value_options=_SORT_SHORT_VALUE_OPTIONS,
                )
                continue

            self._check_path(value, cwd, "read")
            index += 1

    def _check_uniq(self, values: Sequence[str], cwd: Path) -> None:
        """Classify `uniq`'s positional read/write slots.

        `uniq [OPTION]... [INPUT [OUTPUT]]`: the first operand is the input
        (read), the second, if present, is the output (write). Its own
        value-bearing options only take scalar (non-path) values, and its
        no-value long options (`--count`, `--repeated`, `--all-repeated`,
        `--ignore-case`, `--unique`, `--zero-terminated`, `--group`) never
        consume an argument, so listing them separately keeps an
        unrecognized `--`-option failing closed rather than silently
        shifting the positional write slot. `-c`/`-i`/`-u`/`-d`/`-z` are the
        short-spelling equivalents of those same no-value long options and
        are listed in `flag_short_options` for the same reason (R11): an
        unrecognized short option still fails closed.
        """
        scalar_options = frozenset(
            {"-f", "--skip-fields", "-s", "--skip-chars", "-w", "--check-chars"}
        )
        operands = self._partition_path_options(
            values,
            cwd,
            option_access=dict.fromkeys(scalar_options),
            attached_short_options=frozenset({"-f", "-s", "-w"}),
            flag_long_options=frozenset(
                {
                    "--count",
                    "--repeated",
                    "--all-repeated",
                    "--ignore-case",
                    "--unique",
                    "--zero-terminated",
                    "--group",
                }
            ),
            flag_short_options=frozenset({"-c", "-i", "-u", "-d", "-z"}),
            fail_closed_on_unknown_long_option=True,
        )
        if operands:
            self._check_path(operands[0], cwd, "read")
        if len(operands) > 1:
            self._check_path(operands[1], cwd, "write")

    def _check_diff(self, values: Sequence[str], cwd: Path) -> None:
        """Classify `diff`'s write (`--output`) and read-control options.

        `-N`/`--new-file` and diff's other common no-value long options
        (`--brief`, `--unified`, `--recursive`, `--ignore-case`, `--color`,
        `--side-by-side`) never consume an argument, so listing them
        separately keeps the module's fail-closed-on-unrecognized-option
        invariant from rejecting ordinary read-only usage. `flag_short_options`
        is the short-spelling analogue (R11): `-u`/`-c`/`-y`/`-e`/`-n` are
        alternate output-format flags, `-r`/`-q`/`-i`/`-w`/`-b`/`-B`/`-a`/
        `-t`/`-T`/`-p`/`-s` are the remaining common no-value short options,
        and `-N` is `--new-file`'s short spelling — none of them take a
        value, so listing them keeps an unrecognized short option failing
        closed instead of rejecting this ordinary usage.
        """
        for raw_path in self._partition_path_options(
            values,
            cwd,
            option_access={
                "--output": "write",
                "--from-file": "read",
                "--to-file": "read",
            },
            flag_long_options=frozenset(
                {
                    "--brief",
                    "--unified",
                    "--recursive",
                    "--ignore-case",
                    "--color",
                    "--side-by-side",
                    "--new-file",
                }
            ),
            flag_short_options=frozenset(
                {
                    "-u",
                    "-r",
                    "-q",
                    "-N",
                    "-i",
                    "-w",
                    "-b",
                    "-B",
                    "-c",
                    "-y",
                    "-a",
                    "-t",
                    "-T",
                    "-p",
                    "-s",
                    "-e",
                    "-n",
                }
            ),
        ):
            self._check_path(raw_path, cwd, "read")

    def _check_grep(self, values: Sequence[str], cwd: Path) -> None:
        """Classify `grep`'s pattern-file read option and file operands.

        `-r`/`-R` (recurse into a directory) take no argument, so the
        directory they precede already reaches the file-operand loop below
        unmodified; no dedicated handling is needed for them. `--regexp`/
        `--file`/`--exclude-from` resolve through `_resolve_long_option`
        (GNU unambiguous-prefix abbreviation, e.g. `--fil=` for `--file`),
        so an abbreviated spelling is classified identically to the full
        name instead of falling through as an unrecognized option: for
        `--file`/`--exclude-from` that would silently skip the read check
        on the pattern-file argument, and for `--regexp` it would leave
        `explicit_pattern` unset, misreading the real first file operand as
        the (excluded) pattern positional. `_GREP_LONG_VALUE_OPTIONS`
        (`--label`, `--include`, `--context`, etc.) resolve the same way and
        have their own value consumed as a unit for the same reason: leaving
        the value in the token stream (the old skip-the-flag-only treatment)
        makes its classification depend on unrelated context — with no
        explicit pattern elsewhere it lands in the excluded (never
        checked) implicit-pattern slot, but once an explicit `-e`/`-f`/
        `--regexp` pattern IS present it instead lands in the checked file
        operands, spuriously rejecting an ordinary out-of-workspace-looking
        label/glob/count value that is never actually read from disk (R6).
        Consuming the value up front makes the classification of every
        other positional deterministic regardless of that context. An
        option this family does not otherwise recognize stays permissive
        (an ordinary, over-checked-never-under-checked operand) rather than
        failing closed, matching grep's existing pure-read classification.
        """
        positionals: list[str] = []
        explicit_pattern = False
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            value_text = str(value)
            if value_text == "-e":
                explicit_pattern = True
                _, index = self._take_option_argument(
                    values,
                    index,
                    attached_argument=None,
                    context="grep argument for -e",
                )
                continue
            if value_text.startswith("-e") and not value_text.startswith("--"):
                explicit_pattern = True
                index += 1
                continue
            if value_text == "-f":
                explicit_pattern = True
                if index + 1 < len(values):
                    self._check_path(values[index + 1], cwd, "read")
                index += 2
                continue
            if value_text.startswith("-f") and not value_text.startswith("--"):
                explicit_pattern = True
                attached = value_text[2:]
                if attached:
                    self._check_path(self._derived_value(value, attached), cwd, "read")
                index += 1
                continue
            if value_text.startswith("--"):
                raw_option, separator, attached = value_text.partition("=")
                resolved = self._resolve_long_option(
                    raw_option, _GREP_KNOWN_LONG_OPTIONS
                )
                if resolved == "--regexp":
                    explicit_pattern = True
                    _, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached) if separator else None
                        ),
                        context=f"grep argument for {resolved}",
                    )
                    continue
                if resolved in {"--file", "--exclude-from"}:
                    if resolved == "--file":
                        explicit_pattern = True
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached) if separator else None
                        ),
                        context=f"grep argument for {resolved}",
                    )
                    assert argument is not None
                    self._check_path(argument, cwd, "read")
                    continue
                if resolved in _GREP_LONG_VALUE_OPTIONS:
                    # `--label`, `--include`, `--context`, etc. take a value
                    # that is never a path (a label, glob pattern, or count),
                    # but it must still be consumed as a unit: skipping only
                    # the option token would leave the argument in the
                    # stream as an ordinary positional, whose classification
                    # then depends on unrelated context (R6) — misread as
                    # the excluded implicit-pattern slot with no explicit
                    # pattern elsewhere, or spuriously checked (and
                    # rejected) as a file operand once one is present.
                    _, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached) if separator else None
                        ),
                        context=f"grep argument for {resolved}",
                    )
                    continue
                # Long flag options this family does not model (e.g.
                # `--color=auto`) carry no path, so they are skipped rather
                # than parsed further.
                index += 1
                continue
            if value_text.startswith("-") and value_text != "-":
                # A bundled short-option cluster (e.g. `-if`, `-vf`, `-nf`):
                # only the trailing character may carry a value, so `-f`'s
                # pattern-file argument is still read-checked even when it is
                # not the leading flag in the token.
                cluster_chars = value_text[1:]
                if "e" in cluster_chars or "f" in cluster_chars:
                    explicit_pattern = True
                index, _, _ = self._parse_short_option_cluster(
                    values,
                    index,
                    cwd,
                    flag_options=_GREP_SHORT_FLAG_OPTIONS,
                    value_options=_GREP_SHORT_VALUE_OPTIONS,
                )
                continue
            positionals.append(value)
            index += 1

        file_operands = positionals if explicit_pattern else positionals[1:]
        for raw_path in file_operands:
            self._check_path(raw_path, cwd, "read")

    def _check_write_command(
        self,
        command_name: str,
        values: Sequence[str],
        cwd: Path,
    ) -> None:
        """Dispatch a `_WRITE_COMMANDS` member to its operand write check.

        `chmod`'s MODE and `chown`/`chgrp`'s OWNER[:GROUP]/GROUP positional
        argument are not paths, so that leading operand is excluded from the
        write check. `--reference=RFILE` supplies the same role from a file
        instead (itself a read path, classified through the access-map
        substrate), so no positional is excluded when it is present.
        `touch`/`truncate` own their own `-r`/`--reference` read slot (the
        reference file's timestamps/size are read, never written) plus, for
        `truncate`, the `-s`/`--size` scalar; every other write command in
        this family has no non-path positional.
        """
        if command_name in _OWNERSHIP_MODE_COMMANDS:
            reference_options = frozenset({"--reference"})
            operands = self._partition_path_options(
                values,
                cwd,
                option_access={"--reference": "read"},
            )
            # Must resolve through the same `_resolve_long_option` set the
            # partitioning pass above uses for this option, not an
            # independent raw-string pass: otherwise a GNU-abbreviated
            # `--ref=` is consumed (and read-checked) above but looks absent
            # here, so the real target is misread as the excluded MODE
            # positional and its write check is silently skipped.
            has_reference = any(
                str(value).startswith("--")
                and self._resolve_long_option(
                    str(value).partition("=")[0], reference_options
                )
                is not None
                for value in values
            )
            if not has_reference and operands:
                operands = operands[1:]
            for raw_path in operands:
                self._check_path(raw_path, cwd, "write")
            return
        if command_name in {"touch", "truncate"}:
            option_access: dict[str, PathAccess | None] = {
                "-r": "read",
                "--reference": "read",
            }
            attached_short_options = {"-r"}
            if command_name == "truncate":
                option_access.update({"-s": None, "--size": None})
                attached_short_options.add("-s")
            operands = self._partition_path_options(
                values,
                cwd,
                option_access=option_access,
                attached_short_options=frozenset(attached_short_options),
            )
            for raw_path in operands:
                self._check_path(raw_path, cwd, "write")
            return
        self._check_operands(
            values,
            cwd,
            "write",
            value_options=_COMMAND_VALUE_OPTIONS.get(command_name, frozenset()),
        )

    def _parse_target_directory(
        self,
        values: Sequence[str],
        cwd: Path,
    ) -> tuple[bool, list[str]]:
        """Split a `-t`/`--target-directory VALUE` destination from operands.

        Shared by `cp` and `mv`/`ln`: when present, every remaining operand is
        a source and the target directory is the sole write destination;
        otherwise the caller treats the last operand as the destination.
        `-t`/`--target-directory` resolve through the same bundled-cluster
        (`_parse_short_option_cluster`) and GNU-abbreviation
        (`_resolve_long_option`) substrates every other family uses, so a
        bundled form (`-rt`, `-vt`) or an abbreviated long form (`--targ=`)
        is classified identically to the standalone/full-name form instead
        of falling through as an ordinary operand — for `cp` that would
        misclassify the real write destination as an under-strict
        read-checked source. The target directory is write-checked here
        (every caller treats it as the write destination), so the return
        value is only whether one was found, not the raw string. `cp`/`mv`/
        `ln` own a write slot, so an option this parser cannot resolve fails
        closed.
        """
        found_target_dir = False
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
            if text.startswith("--"):
                raw_option, separator, attached = text.partition("=")
                resolved = self._resolve_long_option(
                    raw_option, _TARGET_DIR_KNOWN_LONG_OPTIONS
                )
                if resolved == "--target-directory":
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached) if separator else None
                        ),
                        context=f"argument for {resolved}",
                    )
                    assert argument is not None
                    self._check_path(argument, cwd, "write")
                    found_target_dir = True
                    continue
                if resolved in _TARGET_DIR_LONG_SCALAR_OPTIONS:
                    _, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached) if separator else None
                        ),
                        context=f"argument for {resolved}",
                    )
                    continue
                if resolved in _TARGET_DIR_LONG_FLAG_OPTIONS:
                    index += 1
                    continue
                raise CommandPolicyViolation(
                    f"cannot safely inspect option {raw_option}"
                )
            index, matched_option, _ = self._parse_short_option_cluster(
                values,
                index,
                cwd,
                flag_options=_TARGET_DIR_SHORT_FLAG_OPTIONS,
                value_options=_TARGET_DIR_SHORT_VALUE_OPTIONS,
            )
            if matched_option == "t":
                found_target_dir = True
        return found_target_dir, operands

    def _check_copy(self, values: Sequence[str], cwd: Path) -> None:
        """`cp`: sources read, destination (or `-t` target directory) write.

        Recursively copying through a dereferenced symlink (`-L`/
        `--dereference`, `-H`) can read arbitrarily deep external content
        through a single approved read boundary, which cannot be bounded
        statically, so that combination is a hard denial regardless of the
        actual paths involved. `-a`/`--archive` implies `-r` (GNU documents
        it as `-dR --preserve=all`) and therefore counts as recursive for
        this denial too (M1), even though it also implies `-d`/no-dereference
        on its own — `-a` combined with an explicit `-L`/`--dereference`
        still overrides that default and follows symlinks, so the denial
        must still fire. `--recursive`/`--archive`/`--dereference`/
        `--follow-command-line-symlink` resolve through `_resolve_long_option`
        against the same `_TARGET_DIR_KNOWN_LONG_OPTIONS` set
        `_parse_target_directory` uses for this family, so a GNU-abbreviated
        spelling (`--derefe`) is classified identically to the full name
        instead of silently missing this detection.
        """
        recursive = False
        dereferences_links = False
        for value in values:
            text = str(value)
            if text == "--":
                break
            if text.startswith("--"):
                raw_option, _, _ = text.partition("=")
                resolved = self._resolve_long_option(
                    raw_option, _TARGET_DIR_KNOWN_LONG_OPTIONS
                )
                if resolved in {"--recursive", "--archive"}:
                    recursive = True
                elif resolved in {"--dereference", "--follow-command-line-symlink"}:
                    dereferences_links = True
            elif text.startswith("-") and not text.startswith("--"):
                flags = text[1:]
                recursive = recursive or "r" in flags or "R" in flags or "a" in flags
                dereferences_links = dereferences_links or "L" in flags or "H" in flags
        if recursive and dereferences_links:
            raise CommandPolicyViolation(
                "cannot safely inspect recursive copying that follows symbolic links"
            )

        found_target_dir, operands = self._parse_target_directory(values, cwd)
        if found_target_dir:
            for raw_path in operands:
                self._check_path(raw_path, cwd, "read")
            return
        if len(operands) < 2:
            return
        for raw_path in operands[:-1]:
            self._check_path(raw_path, cwd, "read")
        self._check_path(operands[-1], cwd, "write")

    def _check_install(self, values: Sequence[str], cwd: Path) -> None:
        """`install`: `-t/--target-directory` write; `-d` marks every operand write.

        Scalar options (`-g/-m/-o/-S/--context`) never carry a path, and the
        delegated `--strip-program` is a hard denial; an unrecognized option
        fails closed rather than silently letting one shift the destination
        slot. `-t` also resolves when bundled behind another short flag
        (`-Dt destdir`), sharing the bundled-cluster parser `-Dm755` already
        uses; its value is write-checked below alongside every other
        `target_dir` spelling.
        """
        target_dir: str | None = None
        operands: list[str] = []
        directory_mode = False
        options_done = False
        scalar_options = frozenset(
            {
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
        )
        flag_options = frozenset(
            {
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
        )
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
            if not text.startswith("--"):
                # A bundled cluster (e.g. `-Dm755`): only the trailing
                # value-taking character may carry an attached argument, so
                # this shares the same GNU bundling contract as `sort`. `-t`
                # bundles the same way (`-Dt destdir`); its value is deferred
                # to the same `target_dir` slot the standalone `-t`/
                # `--target-directory` forms populate above, so the single
                # write check at the bottom of this method covers every
                # spelling instead of a second inline check here.
                index, matched_option, argument = self._parse_short_option_cluster(
                    values,
                    index,
                    cwd,
                    flag_options=frozenset("bcCDpsTv"),
                    value_options={
                        "g": None,
                        "m": None,
                        "o": None,
                        "S": None,
                        "t": None,
                    },
                )
                if matched_option == "t":
                    assert argument is not None
                    target_dir = argument
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

    def _check_move_or_link(self, values: Sequence[str], cwd: Path) -> None:
        """`mv`/`ln`: every operand is write-sensitive, including `ln` sources.

        A hard link inside the workspace aliases its source inode; later path
        resolution sees only the workspace-side alias, so an external
        read-only source must be write-checked here rather than read-checked,
        or it would stay silently mutable through the link. `_parse_target_directory`
        already write-checks a `-t`/`--target-directory` destination inline.
        """
        _, operands = self._parse_target_directory(values, cwd)
        for raw_path in operands:
            self._check_path(raw_path, cwd, "write")

    def _check_destructive_file_command(
        self,
        command_name: str,
        values: Sequence[str],
        cwd: Path,
    ) -> None:
        """`unlink`/`shred`: every remaining operand is a write target.

        `shred` additionally accepts `--random-source=FILE` (itself a read
        path) and scalar `-n`/`--iterations`, `-s`/`--size` options whose
        value must not be misread as a path operand.
        """
        if command_name == "shred":
            operands = self._partition_path_options(
                values,
                cwd,
                option_access={
                    "--random-source": "read",
                    "-n": None,
                    "--iterations": None,
                    "-s": None,
                    "--size": None,
                },
                attached_short_options=frozenset({"-n", "-s"}),
            )
        else:
            operands = self._operands(values)
        for raw_path in operands:
            self._check_path(raw_path, cwd, "write")

    def _check_find(self, literals: Sequence[str], state: _ShellState) -> None:
        """Classify `find` per the frozen single-pass, observer-scoped algorithm.

        1. Parse once into fixed path events, exec/execdir/ok/okdir clauses,
           and a delete flag.
        2. Fixed path events are checked on the real session effects.
        3. Each clause's write-ness is classified through the SAME real
           enforcement pass (`_validate_nested_command_words`), via a
           clause-scoped observer that inspects every raw operand before that
           operand's own path resolution: an OR over `{}` presence and
           execdir/okdir-relative operands, aggregated with `-delete`.
        4. The root is then checked write if any clause (or `-delete`) writes,
           read otherwise; a write root additionally poisons the session
           (find's per-match children are not exact-registerable, so the
           write effect cannot be captured as a single path).
        """
        invocation = self._parse_find_invocation(literals)
        for event in invocation.path_events:
            self._check_path(event.value, state.cwd, event.access)

        session = _active_validation_session()
        writes_root = invocation.writes_delete
        saved_observer = session.find_clause_observer
        # Suspend whatever observer an enclosing find clause left active: this
        # find's own path events and root must never be misclassified as
        # belonging to an outer find's clause, and a nested find below must
        # not corrupt this find's own classification either.
        session.find_clause_observer = None
        try:
            for clause in invocation.exec_clauses:
                clause_writes_root = False

                def _observe(
                    raw_path: str,
                    access: PathAccess,
                    _clause: _FindExecClause = clause,
                ) -> None:
                    nonlocal clause_writes_root
                    if access != "write":
                        return
                    if "{}" in str(raw_path) or (
                        _clause.marker in {"-execdir", "-okdir"}
                        and self._is_relative_file_operand(raw_path)
                    ):
                        clause_writes_root = True

                session.find_clause_observer = _observe
                try:
                    self._validate_nested_command_words(
                        self._tag_find_placeholder(clause.command, state.cwd),
                        state,
                    )
                finally:
                    session.find_clause_observer = None
                writes_root = writes_root or clause_writes_root

            root_access: PathAccess = "write" if writes_root else "read"
            for root in invocation.roots:
                self._check_path(root, state.cwd, root_access)
            if writes_root:
                session.effects.unknown_effect = True
        finally:
            session.find_clause_observer = saved_observer

    def _parse_find_invocation(self, literals: Sequence[str]) -> _FindInvocation:
        """Parse `find`'s argv once into roots, fixed path events and clauses.

        Global traversal-mode options (`-H`/`-L`/`-P`, `-Olevel`, `-D debugopts`)
        precede the starting points and must not consume them. `-files0-from`
        supplies roots at runtime and is rejected outright rather than trusted
        blind. Reference predicates take a path argument except `-newerXt`,
        whose argument is a timestamp string, not a path.
        """
        self._reject_dynamic_values("find arguments", literals)

        root_start = 0
        while root_start < len(literals):
            option = str(literals[root_start])
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

        roots: list[str] = []
        expression_start = len(literals)
        for index, value in enumerate(literals[root_start:], start=root_start):
            text = str(value)
            if text.startswith("-") or text in {"!", "("}:
                expression_start = index
                break
            roots.append(value)
        if not roots:
            roots = ["."]

        expression = literals[expression_start:]
        path_events: list[_PathEvent] = []
        clauses: list[_FindExecClause] = []
        writes_delete = False
        index = 0
        while index < len(expression):
            marker = str(expression[index])
            if marker == "-files0-from" or marker.startswith("-files0-from="):
                raise CommandPolicyViolation(
                    "cannot safely inspect find runtime root list"
                )
            if marker == "-delete":
                writes_delete = True
                index += 1
                continue
            if marker in _FIND_OUTPUT_ACTIONS:
                argument_count = _FIND_OUTPUT_ACTIONS[marker]
                if index + argument_count >= len(expression):
                    raise CommandPolicyViolation(f"missing find argument for {marker}")
                path_events.append(_PathEvent(expression[index + 1], "write"))
                index += argument_count + 1
                continue
            if marker in _FIND_REFERENCE_PREDICATES or (
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
            if marker not in _FIND_EXEC_MARKERS:
                index += 1
                continue

            nested: list[str] = []
            index += 1
            while index < len(expression) and str(expression[index]) not in {";", "+"}:
                nested.append(expression[index])
                index += 1
            if not nested or index >= len(expression):
                raise CommandPolicyViolation(
                    f"cannot safely inspect unterminated find action {marker}"
                )
            clauses.append(_FindExecClause(marker=marker, command=tuple(nested)))
            index += 1

        return _FindInvocation(
            roots=tuple(roots),
            path_events=tuple(path_events),
            exec_clauses=tuple(clauses),
            writes_delete=writes_delete,
        )

    @staticmethod
    def _is_relative_file_operand(raw_path: str) -> bool:
        return raw_path not in {"", "-", "{}"} and not Path(str(raw_path)).is_absolute()

    @staticmethod
    def _tag_find_placeholder(
        words: Sequence[str],
        cwd: Path,
    ) -> tuple[_CommandValue, ...]:
        """Tag exact `{}` operands with find's arity-preserving cwd resolution.

        `{}` is never removed or replaced in the operand vector (removing it
        would shift positional operands, e.g. turning a 2-operand `cp` into a
        1-operand call that silently skips its destination check); only the
        specific `_CommandValue` instance representing a literal `{}` word
        carries the resolution target, so an unrelated word is never affected.
        """
        tagged: list[_CommandValue] = []
        for word in words:
            if str(word) != "{}":
                tagged.append(
                    word if isinstance(word, _CommandValue) else _CommandValue(word)
                )
                continue
            is_static = not isinstance(word, _CommandValue) or word.is_static
            tagged.append(
                _CommandValue(word, is_static=is_static, find_placeholder_cwd=cwd)
            )
        return tuple(tagged)

    def _check_tar(self, values: Sequence[str], cwd: Path) -> None:
        """Classify `tar`'s archive/control/member paths.

        The archive path (`-f`/`--file`) and the two control-file options
        (`-T`/`--files-from`, `-X`/`--exclude-from`) stay anchored to the
        process `cwd`; only `-C`/`--directory` shifts the active directory,
        and only for later source/destination member operands, never for the
        archive itself. Extract mode additionally poisons the session
        (`unknown_effect`): per-member extracted children are not
        individually enumerable, so the `-C`/default extraction root's own
        write-containment check (below, via the `directory` event) cannot by
        itself capture them — the poison is additive to that check, not a
        replacement for it (M2).
        """
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

        if invocation.mode == "extract":
            _active_validation_session().effects.unknown_effect = True

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
            if event.kind == "unresolved_option":
                # Fails closed in every mode, not only extract/list: the
                # same parser ambiguity (bare flag vs. a hidden argument)
                # would otherwise let an abbreviated write-checked
                # (`--listed-incremental`) or hard-denied (`--files-from`)
                # option's value leak through as an ordinary source/
                # destination positional in create/append/update/
                # concatenate/compare mode.
                raise CommandPolicyViolation(
                    f"cannot safely resolve tar option {event.value}"
                )
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
            elif event.kind == "pattern":
                # `--exclude`'s argument is a glob PATTERN matched against
                # in-archive member names, never a local filesystem path;
                # its argument is already consumed by the parser, so there
                # is nothing left to check here (N2).
                continue
            elif event.kind == "incremental":
                # The snapshot file can be updated even when the archive
                # itself is only read (e.g. listing against a snapshot).
                self._check_path(event.value, cwd, "write")
            elif event.kind == "verbose_output":
                # `--index-file` always writes the verbose listing, even
                # when the archive itself is only read (e.g. `--list`).
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
                # List/extract positionals are in-archive member selectors,
                # not local filesystem paths, and are intentionally not
                # checked here. `@file` includes another archive/file-list in
                # place of a literal member; the same source access applies.
                raw_path = (
                    event.value[1:] if event.value.startswith("@") else event.value
                )
                self._check_path(raw_path, active_cwd, source_access)

    def _check_tar_archive_path(
        self,
        raw_path: str,
        cwd: Path,
        access: PathAccess,
    ) -> None:
        """Reject stdin/stdout and remote archive forms; else check normally.

        `-f -` streams the archive through standard input/output, and a
        `host:path` (or `user@host:path`) form delegates to a remote `rsh`
        transfer; neither is a local path this guard can inspect statically.
        """
        if raw_path == "-":
            raise CommandPolicyViolation(
                "tar archive on standard input/output cannot be inspected safely"
            )
        if ":" in raw_path:
            raise CommandPolicyViolation(
                "remote tar archives cannot be inspected safely"
            )
        self._check_path(raw_path, cwd, access)

    def _parse_tar(self, values: Sequence[str]) -> _TarInvocation:
        """Parse tar's argv once into its operation mode and typed events.

        Traditional syntax omits the leading dash on the first argument
        (`tar cf archive.tar file`); GNU long/short syntax may follow.
        Exactly one mode letter must be present across the whole invocation
        (POSIX tar requires exactly one of create/append/update/concatenate/
        delete/extract/list/compare) or the operation cannot be resolved.
        """
        long_modes: dict[str, _TarMode] = {
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
        modes: list[_TarMode] = []
        events: list[_TarEvent] = []
        remove_files = False
        options_done = False
        index = 0

        if (
            values
            and re.fullmatch(r"[A-Za-z]+", str(values[0]))
            and any(option in "cruAxtd" for option in str(values[0]))
        ):
            index, short_modes, short_events = self._parse_tar_short_events(
                values, 0, values[0]
            )
            modes.extend(short_modes)
            events.extend(short_events)

        while index < len(values):
            value = values[index]
            text = str(value)
            if not options_done and text == "--":
                options_done = True
                index += 1
                continue
            if options_done or not text.startswith("-") or text == "-":
                events.append(_TarEvent("positional", value))
                index += 1
                continue
            if text in long_modes:
                modes.append(long_modes[text])
                index += 1
                continue
            if text == "--remove-files":
                remove_files = True
                index += 1
                continue
            if text in {"--to-stdout", "-O"}:
                index += 1
                continue
            if text in {"--absolute-names", "--absolute-paths"}:
                events.append(_TarEvent("absolute_names"))
                index += 1
                continue
            if text.startswith("--"):
                index, event = self._parse_tar_long_event(values, index)
                if event is not None:
                    events.append(event)
                continue

            index, short_modes, short_events = self._parse_tar_short_events(
                values, index, value[1:]
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
        """Parse one `--`-prefixed tar argument into a typed event.

        The option resolves through `_resolve_long_option` (GNU
        unambiguous-prefix abbreviation, e.g. `--listed-incr` for
        `--listed-incremental`, `--files-fr` for `--files-from`) against
        this family's known path-bearing, pattern, and dangerous long
        options, so an abbreviated spelling is classified identically to
        the full name instead of falling through as unmodeled. `--exclude`
        (kind `pattern`) consumes its glob-PATTERN argument without a path
        check: it is a real, distinct tar option from `--exclude-from`, and
        omitting it from this set previously let its own unambiguous-prefix
        match snap to `--exclude-from` (a real path option) and misclassify
        an ordinary pattern as a file to read-check (N2). An unrecognized
        long option with an attached `=value` unambiguously carries a value
        this family does not model, and fails closed immediately regardless
        of mode (mirroring `rsync`/`curl`/`wget`). An unrecognized option
        with NO attached value is ambiguous: it may be a bare flag (common;
        e.g. `--gzip`), or it may take a separate following token this
        parser cannot identify as its argument, which would otherwise be
        misclassified as an unchecked archive-member positional in
        extract/list mode (never path-checked there) or as an unchecked
        source/destination positional in the other modes (source_access-
        checked, not the write/hard-deny classification the real option
        would receive). Rather than guess its arity, it is returned as an
        `unresolved_option` event; `_check_tar` fails closed on that event
        in every mode, not only extract/list, since the same ambiguity is
        exploitable wherever a long option could silently misclassify a
        value it would otherwise write-check or hard-deny.
        """
        value = values[index]
        text = str(value)
        path_options: dict[str, _TarEventKind] = {
            "--file": "archive",
            "--directory": "directory",
            "--files-from": "files_from",
            "--exclude-from": "exclude_from",
            "--exclude": "pattern",
            "--listed-incremental": "incremental",
            "--add-file": "add_file",
            "--index-file": "verbose_output",
        }
        dangerous_options = {
            "--use-compress-program",
            "--to-command",
            "--checkpoint-action",
            "--info-script",
            "--new-volume-script",
            "--rsh-command",
        }
        option, separator, attached = text.partition("=")
        known_long_options = frozenset(path_options) | dangerous_options
        resolved = self._resolve_long_option(option, known_long_options)
        kind = path_options.get(resolved) if resolved is not None else None
        is_dangerous = resolved in dangerous_options if resolved is not None else False
        if kind is None and not is_dangerous:
            if separator:
                raise CommandPolicyViolation(
                    f"cannot safely resolve tar option {option}"
                )
            return index + 1, _TarEvent("unresolved_option", option)

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
        return next_index, _TarEvent(cast(_TarEventKind, kind), argument)

    def _parse_tar_short_events(
        self,
        values: Sequence[str],
        index: int,
        options: str,
    ) -> tuple[int, list[_TarMode], list[_TarEvent]]:
        """Parse one bundled short-option token (e.g. `-xzf`, `-xfC`, `xfC`).

        The GNU dash-prefixed cluster (`-xzf`, `-xfC`) and the historical
        no-dash traditional "key" form (`xfC archive.tar dir`) share this
        same walk. The no-dash form has no attached-value convention at
        all: every argument-taking letter takes its OWN separate
        subsequent whitespace token, in the order the letters appear, per
        POSIX tar. The dash-prefixed form additionally allows a single
        trailing argument-taking letter to carry its value attached to the
        same token (e.g. `-farchive.tar`); once such a letter is found,
        whatever follows it in the token is normally taken whole as that
        value. But when that remainder itself STARTS with another
        recognized tar option letter (e.g. `C` in `-xfC`, from tar's own
        fixed short-option alphabet), attaching it as `f`'s value would
        misread an active `-C` as an inert path suffix, leaving both the
        archive and `-C`'s own directory argument as unchecked
        positionals; in that ambiguous shape this letter instead takes its
        own separate next token too, and scanning continues into the
        remainder exactly as the no-dash form does. Unrecognized
        characters are silently skipped (compression/verbosity flags like
        `z`/`v` this guard does not need to model), consistent with the
        conservative flag-only treatment of unknown tar options elsewhere.
        """
        short_modes: dict[str, _TarMode] = {
            "c": "create",
            "r": "append",
            "u": "update",
            "A": "concatenate",
            "x": "extract",
            "t": "list",
            "d": "compare",
        }
        argument_events: dict[str, _TarEventKind | None] = {
            "f": "archive",
            "C": "directory",
            "T": "files_from",
            "X": "exclude_from",
            "g": "incremental",
            "I": "dangerous",
            "F": "dangerous",
            "b": None,
        }
        known_option_characters = (
            frozenset(short_modes) | frozenset(argument_events) | {"P", "O"}
        )
        modes: list[_TarMode] = []
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
            if not attached or attached[0] in known_option_characters:
                if next_index >= len(values):
                    raise CommandPolicyViolation(f"missing tar argument for -{option}")
                argument: str | None = values[next_index]
                next_index += 1
                kind = argument_events[option]
                if kind is not None:
                    events.append(_TarEvent(kind, argument))
                cursor += 1
                continue

            argument, next_index = self._take_option_argument(
                values,
                index,
                attached_argument=self._derived_value(values[index], attached),
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

    def _parse_short_option_cluster(
        self,
        values: Sequence[str],
        index: int,
        cwd: Path,
        *,
        flag_options: frozenset[str],
        value_options: Mapping[str, PathAccess | None],
    ) -> tuple[int, str | None, str | None]:
        """Parse one bundled short-option token (e.g. `-no<file>`).

        GNU short options bundle into a single token where only the last
        option in the cluster may carry a value (attached, e.g. `-ofile`, or
        as the following token). Any character not recognized as a flag or a
        value option fails closed rather than being silently skipped, so an
        unmodeled option can never hide a path argument. Returns the index of
        the next unconsumed token, plus the matched value option and its raw
        argument (both `None` when the cluster was flags only) for callers
        that need to act on which option fired — a `None` access here still
        skips the path check but the caller can still see the raw argument,
        e.g. to apply its own non-path classification to it.
        """
        source = values[index]
        options = str(source)[1:]
        cursor = 0
        while cursor < len(options):
            option = options[cursor]
            if option in flag_options:
                cursor += 1
                continue
            if option not in value_options:
                raise CommandPolicyViolation(f"cannot safely resolve option -{option}")
            attached = options[cursor + 1 :]
            argument: str
            if attached:
                argument = self._derived_value(source, attached)
                next_index = index + 1
            elif index + 1 < len(values):
                argument = values[index + 1]
                next_index = index + 2
            else:
                raise CommandPolicyViolation(f"missing argument for -{option}")
            access = value_options[option]
            if access is not None:
                self._check_path(argument, cwd, access)
            return next_index, option, argument
        return index + 1, None, None

    @staticmethod
    def _resolve_long_option(value: str, known: Iterable[str]) -> str | None:
        """Resolve a GNU unambiguous-prefix long option, or None if it can't be.

        `known` must contain the full `--...` option name. An exact match
        always wins; otherwise a prefix must match exactly one candidate.
        Ambiguous or unmatched prefixes return None so the caller decides
        whether that is a benign skip (pure-read families) or a fail-closed
        violation (families that own a write slot). A bare `--` is the
        argument-list terminator, never an abbreviation: every candidate
        starts with `--`, so it would otherwise "unambiguously" prefix-match
        a known set containing exactly one option (R7).
        """
        if value in known:
            return value
        if value == "--":
            return None
        matches = [candidate for candidate in known if candidate.startswith(value)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _partition_path_options(
        self,
        values: Sequence[str],
        cwd: Path,
        *,
        option_access: Mapping[str, PathAccess | None],
        attached_short_options: frozenset[str] = frozenset(),
        flag_long_options: frozenset[str] = frozenset(),
        flag_short_options: frozenset[str] = frozenset(),
        fail_closed_on_unknown_long_option: bool = True,
    ) -> list[str]:
        """Split path-bearing options from operands using a per-option access map.

        Module invariant: an option grammar this family does not explicitly
        model FAILS CLOSED rather than being silently skipped or treated as
        an ordinary operand, for BOTH short and long options. Over-rejecting
        an unmodeled legitimate option is acceptable; silently letting an
        unmodeled option hide a path argument is not. This applies uniformly
        — there is no pure-read exemption — so `fail_closed_on_unknown_long_option`
        defaults to `True` for every caller; a caller may only pass `False`
        when it has an independent, equally strict reason to trust an
        unresolved long option cannot carry a path (rare, and must say why).

        Each key in `option_access` consumes exactly one following or attached
        token; a `None` access consumes it without a path check (a scalar
        value), `"read"`/`"write"` checks it. `flag_long_options` is a
        separate set of long options that never take an argument (bare, or
        with an optional attached `=value` this family ignores); listing a
        long option there instead of in `option_access` is required for any
        option that owns no value slot, since `option_access` always consumes
        a following/attached token. `flag_short_options` is the short-option
        analogue: a bare (never bundled, never valued) single-dash token this
        family recognizes as a no-op flag, e.g. `-N`. Long options resolve
        through `_resolve_long_option` first, so an accepted abbreviation
        (`--out=` for `--output`) is classified identically to the full name.
        A single-dash token that is not an exact `option_access` key, not an
        `attached_short_options` prefix carrying a value, and not an exact
        `flag_short_options` member fails closed: this substrate has no
        bundled short-option-cluster grammar of its own (see
        `_parse_short_option_cluster` for families that need one), so it
        cannot tell a genuinely unknown flag apart from one hiding a path
        argument and must not guess by skipping it.
        """
        known_long_options = (
            frozenset(option for option in option_access if option.startswith("--"))
            | flag_long_options
        )
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

            value_text = str(value)
            if value_text.startswith("--"):
                raw_option, separator, _ = value_text.partition("=")
                resolved = raw_option
                if (
                    resolved not in option_access
                    and resolved not in flag_long_options
                    and known_long_options
                ):
                    candidate = self._resolve_long_option(
                        raw_option, known_long_options
                    )
                    if candidate is not None:
                        resolved = candidate
                if resolved in flag_long_options:
                    index += 1
                    continue
                if resolved in option_access:
                    access = option_access[resolved]
                    argument: str
                    if separator:
                        argument = self._derived_value(
                            value, value[len(raw_option) + 1 :]
                        )
                        index += 1
                    elif index + 1 < len(values):
                        argument = values[index + 1]
                        index += 2
                    else:
                        raise CommandPolicyViolation(f"missing argument for {resolved}")
                    if access is not None:
                        self._check_path(argument, cwd, access)
                    continue
                if fail_closed_on_unknown_long_option:
                    raise CommandPolicyViolation(
                        f"cannot safely resolve option {raw_option}"
                    )
                index += 1
                continue

            matching_short = next(
                (
                    option
                    for option in attached_short_options
                    if value_text.startswith(option) and len(value_text) > len(option)
                ),
                None,
            )
            if matching_short is not None:
                access = option_access[matching_short]
                if access is not None:
                    self._check_path(
                        self._derived_value(value, value_text[len(matching_short) :]),
                        cwd,
                        access,
                    )
                index += 1
                continue
            if value_text in flag_short_options:
                index += 1
                continue
            # Module invariant (see docstring): an unrecognized short option
            # fails closed rather than being silently skipped, so it can
            # never hide a path argument behind it.
            raise CommandPolicyViolation(f"cannot safely resolve option {value_text}")
        return remaining

    def _read_policy_script(self, raw_path: str, cwd: Path) -> str:
        script_path = self._check_path(raw_path, cwd, "read")
        session = _active_validation_session()
        session.effects.inspect_script(script_path)

        descriptor: int | None = None
        try:
            descriptor = os.open(script_path, _secure_script_open_flags())
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise CommandPathViolation(access="read", path=script_path)
            if metadata.st_size > _MAX_INSPECTED_SCRIPT_BYTES:
                raise CommandPolicyViolation(
                    "shell policy script exceeds the "
                    f"{_MAX_INSPECTED_SCRIPT_BYTES}-byte inspection limit"
                )
            session.charge_script_bytes(metadata.st_size)
            with os.fdopen(descriptor, "rb") as script_file:
                descriptor = None
                raw_script = script_file.read(_MAX_INSPECTED_SCRIPT_BYTES + 1)
            if len(raw_script) > _MAX_INSPECTED_SCRIPT_BYTES:
                raise CommandPolicyViolation(
                    "shell policy script exceeds the "
                    f"{_MAX_INSPECTED_SCRIPT_BYTES}-byte inspection limit"
                )
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

    def _check_script_command(
        self,
        grammar: _ScriptCommandGrammar,
        values: Sequence[str],
        cwd: Path,
    ) -> None:
        """Classify `sed`/`awk`'s script source(s) and file operand access.

        Every `-e`/`--expression` (sed) or `--source` (awk) argument and,
        absent an explicit script, the leading positional program are
        inspected through the family's command-slot lexer (`_check_sed_program`
        / `_check_awk_program`). `-f`/`--file` script files route through
        `_read_policy_script` (M4: the same `unknown_effect` gate, non-regular-
        file rejection, and bounded read every other script read uses) before
        the same lexer, rather than a parallel copy. sed's `-i`/`--in-place`
        switches the remaining file operands from read to write; a backup
        suffix (`-i.bak`, `--in-place=.bak`) additionally poisons
        `unknown_effect`, since the derived backup file it also writes is
        never itself path-checked (N3). Long options resolve through
        `_resolve_long_option` (GNU unambiguous-prefix
        abbreviation, e.g. `--expr=` for `--expression`); an option this
        family does not recognize fails closed instead of being silently
        skipped, since a write-owning spelling (`--file`, `--in-place`, awk's
        `--pretty-print`/`--profile`/`--dump-variables`) could otherwise hide
        behind an unrecognized abbreviation or typo. Single-dash tokens are
        always parsed as a short-option cluster (`_consume_script_short_options`),
        which fails closed the same way for an unmodeled short option.
        """
        positionals: list[str] = []
        explicit_script = False
        file_access: PathAccess = "read"
        known_long_options = frozenset(
            {"--file", grammar.expression_long_option}
            | {option for option in grammar.value_options if option.startswith("--")}
            | (
                {grammar.in_place_long_option}
                if grammar.in_place_long_option is not None
                else set()
            )
            | {
                option
                for option in grammar.optional_write_options
                if option.startswith("--")
            }
            | grammar.flag_long_options
        )
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            value_text = str(value)
            if value_text.startswith("--") and value_text != "--":
                raw_option, separator, attached = value_text.partition("=")
                resolved = self._resolve_long_option(raw_option, known_long_options)
                if resolved is None:
                    raise CommandPolicyViolation(
                        f"cannot safely inspect {grammar.language} option {raw_option}"
                    )
                if resolved == "--file":
                    explicit_script = True
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached) if separator else None
                        ),
                        context=f"{grammar.language} argument for {resolved}",
                    )
                    assert argument is not None
                    self._inspect_script_file(grammar.language, argument, cwd)
                    continue
                if resolved == grammar.expression_long_option:
                    explicit_script = True
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached) if separator else None
                        ),
                        context=f"{grammar.language} argument for {resolved}",
                    )
                    assert argument is not None
                    self._check_script_expression(grammar.language, argument, cwd)
                    continue
                if resolved == grammar.in_place_long_option:
                    file_access = "write"
                    if separator and attached:
                        # A backup suffix (`--in-place=.bak`) derives a
                        # second write target (the backup file) whose exact
                        # name this guard never computes; poison the same
                        # way find/tar/gzip's other non-enumerable derived
                        # writes do (N3), additive to the operand's own
                        # write check above, not a replacement for it.
                        _active_validation_session().effects.unknown_effect = True
                    index += 1
                    continue
                if resolved in grammar.optional_write_options:
                    argument = (
                        self._derived_value(value, attached)
                        if separator
                        else _CommandValue(grammar.optional_write_options[resolved])
                    )
                    self._check_path(argument, cwd, "write")
                    index += 1
                    continue
                if resolved in grammar.flag_long_options:
                    # Takes no value and names no path (e.g. sed's
                    # `--quiet`/`--posix`): consumed as an inert flag, never
                    # routed through `_take_option_argument`.
                    index += 1
                    continue
                # The remaining known long options are mandatory scalar
                # values (e.g. sed's `--line-length`), never a path.
                _, index = self._take_option_argument(
                    values,
                    index,
                    attached_argument=(
                        self._derived_value(value, attached) if separator else None
                    ),
                    context=f"{grammar.language} argument for {resolved}",
                )
                continue
            if (
                (
                    grammar.short_flag_options
                    or grammar.short_value_options
                    or grammar.optional_write_options
                )
                and value_text.startswith("-")
                and value_text != "-"
            ):
                option_result = self._consume_script_short_options(
                    grammar, values, index, cwd
                )
                explicit_script = explicit_script or option_result.explicit_script
                if option_result.requests_in_place:
                    file_access = "write"
                    if option_result.has_backup_suffix:
                        # See the `--in-place=SUFFIX` branch above (N3): the
                        # same non-enumerable derived backup write applies
                        # to the short-option form (`-i.bak`).
                        _active_validation_session().effects.unknown_effect = True
                index = option_result.next_index
                continue
            if value_text.startswith("-") and value_text != "-":
                raise CommandPolicyViolation(
                    f"cannot safely inspect {grammar.language} option {value_text}"
                )
            # The CLI `var=value` assignment-operand skip only applies once
            # the program itself has already been consumed (an explicit
            # script via `-f`/`--file`/`-e`, or a prior positional): it must
            # never swallow the program token, or an inline program whose
            # text happens to contain "=" would be silently dropped from
            # inspection entirely.
            already_have_program = explicit_script or bool(positionals)
            if (
                grammar.ignores_assignment_arguments
                and already_have_program
                and "=" in value_text
            ):
                index += 1
                continue
            positionals.append(value)
            index += 1

        if not explicit_script and positionals:
            self._check_script_expression(grammar.language, positionals[0], cwd)
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
        token_text = str(token)
        while cursor < len(token_text):
            option = token_text[cursor]
            if option in grammar.short_flag_options:
                cursor += 1
                continue
            if option == grammar.in_place_short_option:
                # GNU sed treats the remainder of this token as the optional
                # backup suffix, so no later character is another option.
                return _ScriptShortOptionResult(
                    next_index=index + 1,
                    requests_in_place=True,
                    has_backup_suffix=bool(token_text[cursor + 1 :]),
                )
            if option in {"e", "f"} or option in grammar.short_value_options:
                attached_text = token_text[cursor + 1 :]
                argument, next_index = self._take_option_argument(
                    values,
                    index,
                    attached_argument=(
                        self._derived_value(token, attached_text)
                        if attached_text
                        else None
                    ),
                    context=f"{grammar.language} argument for -{option}",
                )
                assert argument is not None

                if option == "e":
                    self._check_script_expression(grammar.language, argument, cwd)
                elif option == "f":
                    self._inspect_script_file(grammar.language, argument, cwd)
                return _ScriptShortOptionResult(
                    next_index=next_index,
                    explicit_script=option in {"e", "f"},
                )
            if f"-{option}" in grammar.optional_write_options:
                # gawk's optional-argument convention: an attached value is
                # this option's argument, but a separate following token
                # never is (unlike `-e`/`-f` above), so nothing beyond this
                # token is ever consumed.
                attached_text = token_text[cursor + 1 :]
                argument = (
                    self._derived_value(token, attached_text)
                    if attached_text
                    else _CommandValue(grammar.optional_write_options[f"-{option}"])
                )
                self._check_path(argument, cwd, "write")
                return _ScriptShortOptionResult(next_index=index + 1)
            raise CommandPolicyViolation(
                f"cannot safely inspect {grammar.language} option -{option}"
            )

        return _ScriptShortOptionResult(next_index=index + 1)

    def _inspect_script_file(
        self,
        language: Literal["sed", "awk"],
        raw_path: str,
        cwd: Path,
    ) -> None:
        script = self._read_policy_script(raw_path, cwd)
        self._check_script_program(language, script, cwd)

    def _check_script_expression(
        self,
        language: Literal["sed", "awk"],
        script: str,
        cwd: Path,
    ) -> None:
        """Classify an inline `-e`/positional sed|awk program (I8/I9/I13)."""
        if isinstance(script, _CommandValue) and not script.is_static:
            raise CommandPolicyViolation(f"cannot inspect dynamic {language} program")
        self._check_script_program(language, str(script), cwd)

    def _check_script_program(
        self,
        language: Literal["sed", "awk"],
        script: str,
        cwd: Path,
    ) -> None:
        if language == "awk":
            self._check_awk_program(script, cwd)
        else:
            self._check_sed_program(script, cwd)

    def _check_sed_program(self, script: str, cwd: Path) -> None:
        """Walk a sed program one command slot at a time (I8).

        Traverses blocks (`{`/`}`), addresses/ranges, and repeated `!`
        negation before classifying the command letter; braces/delimiters
        inside patterns, replacements, addresses, `y///` transliterations,
        `#` comments, and `a`/`i`/`c` text payloads are DATA and are never
        re-entered as structure. `r`/`R` read a filename that runs to the end
        of the line (read); `w`/`W` and the `s///w` flag write one (write);
        `e` (bare, or `s///e`) executes an arbitrary shell command and always
        fails closed, since its argument cannot be inspected safely.
        """
        index = 0
        while index < len(script):
            while index < len(script) and script[index] in " \t;\n}":
                index += 1
            if index >= len(script):
                break

            index = self._skip_sed_addresses(script, index)
            while index < len(script) and script[index] in " \t":
                index += 1
            # BSD sed accepts repeated negation operators, so command
            # discovery must consume the complete prefix before classifying.
            while index < len(script) and script[index] == "!":
                index += 1
                while index < len(script) and script[index] in " \t":
                    index += 1
            if index >= len(script):
                raise CommandPolicyViolation("cannot safely inspect sed program")

            command = script[index]
            index += 1
            if command in "rR":
                index, raw_path = self._scan_sed_filename(script, index)
                self._check_path(raw_path, cwd, "read")
                continue
            if command in "wW":
                index, raw_path = self._scan_sed_filename(script, index)
                self._check_path(raw_path, cwd, "write")
                continue
            if command == "e":
                raise CommandPolicyViolation(
                    "sed 'e' command executes an arbitrary shell command and "
                    "cannot be inspected safely"
                )
            if command == "{":
                continue
            if command == "#":
                index = self._skip_sed_to_boundary(script, index, boundaries="\n")
                continue
            if command == "s":
                index = self._scan_sed_substitution(script, index, cwd)
                continue
            if command == "y":
                index = self._scan_sed_transliteration(script, index)
                continue
            if command in "aic":
                index = self._skip_sed_to_boundary(script, index, boundaries="\n")
                continue
            if command in "bTt:":
                index = self._skip_sed_to_boundary(script, index, boundaries=";\n")
                continue
            index = self._skip_sed_to_boundary(script, index, boundaries=";\n}")

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
            return WorkspaceCommandPathGuard._skip_sed_address_modifiers(
                script,
                WorkspaceCommandPathGuard._scan_sed_delimited(
                    script, delimiter_index=index
                ),
            )
        if character == "\\" and index + 1 < len(script):
            return WorkspaceCommandPathGuard._skip_sed_address_modifiers(
                script,
                WorkspaceCommandPathGuard._scan_sed_delimited(
                    script, delimiter_index=index + 1
                ),
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
    def _skip_sed_address_modifiers(script: str, index: int) -> int:
        """Consume a regex address's trailing `I`/`M` modifiers (C3).

        GNU sed allows either modifier, in either order and repeated, right
        after a `/regexp/` or `\\%regexp%` address (`/re/I`, `/re/MI`, ...).
        Neither letter is ever a valid sed command name on its own, so
        consuming them here cannot swallow a real command; skipping this
        step left the command-letter scan land on the modifier itself,
        misreading it as the command and treating everything after it
        (including a following `w`/`W` write clause) as opaque trailing
        text of a fallback command it never was.
        """
        cursor = index
        while cursor < len(script) and script[cursor] in "IM":
            cursor += 1
        return cursor

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

    def _scan_sed_substitution(self, script: str, index: int, cwd: Path) -> int:
        if index >= len(script):
            raise CommandPolicyViolation("cannot safely inspect sed substitution")
        delimiter = script[index]
        if delimiter.isalnum() or delimiter.isspace() or delimiter == "\\":
            raise CommandPolicyViolation("cannot safely inspect sed substitution")

        cursor = self._scan_sed_fields(script, index, field_count=2)

        while cursor < len(script) and script[cursor] not in ";\n}":
            if script[cursor] == "e":
                raise CommandPolicyViolation(
                    "sed 's///e' executes an arbitrary shell command and "
                    "cannot be inspected safely"
                )
            if script[cursor] == "w":
                cursor, raw_path = self._scan_sed_filename(script, cursor + 1)
                self._check_path(raw_path, cwd, "write")
                return cursor
            cursor += 1
        return cursor

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

    @staticmethod
    def _scan_sed_filename(script: str, index: int) -> tuple[int, str]:
        """Read a `r`/`R`/`w`/`W` command's filename argument (I8).

        Sed's file-command filename is the remainder of the script line
        verbatim after any leading whitespace (an embedded `;` is part of
        the filename, not a command separator), so it runs to the next
        newline or the end of the script, never to `;`/`}`.
        """
        cursor = index
        while cursor < len(script) and script[cursor] in " \t":
            cursor += 1
        start = cursor
        end = WorkspaceCommandPathGuard._skip_sed_to_boundary(
            script, cursor, boundaries="\n"
        )
        filename = script[start:end]
        if not filename:
            raise CommandPolicyViolation("missing sed file command filename")
        return end, filename

    def _check_awk_program(self, script: str, cwd: Path) -> None:
        """Classify awk's unsafe I/O forms (I9).

        `system(...)` always executes an arbitrary shell command and fails
        closed unconditionally. `print`/`printf` redirection (`>`, `>>`) and
        `getline < FILE` are located structurally and path-checked instead;
        a redirect target that is not a double-quoted string literal cannot
        be resolved statically (e.g. a field/variable like `$1`) and is
        rejected rather than silently skipped. Piping to or from a command
        (`| "cmd"`, `"cmd" | getline`) always executes an arbitrary shell
        command and fails closed unconditionally.
        """
        if _AWK_SYSTEM_CALL_PATTERN.search(script):
            raise CommandPolicyViolation(
                "awk 'system' call executes an arbitrary shell command and "
                "cannot be inspected safely"
            )
        for match in _AWK_PRINT_PATTERN.finditer(script):
            self._scan_awk_print_redirect(script, match.end(), cwd)
        self._scan_awk_getline_io(script, cwd)

    def _scan_awk_print_redirect(self, script: str, start: int, cwd: Path) -> None:
        index = start
        depth = 0
        quote: str | None = None
        escaped = False
        while index < len(script):
            char = script[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                index += 1
                continue
            if char in {"'", '"'}:
                quote = char
                index += 1
                continue
            if char == "(":
                depth += 1
                index += 1
                continue
            if char == ")" and depth:
                depth -= 1
                index += 1
                continue
            if char in ";\n}" and depth == 0:
                return
            if char == "|" and depth == 0:
                raise CommandPolicyViolation(
                    "awk pipe output executes an arbitrary shell command and "
                    "cannot be inspected safely"
                )
            if char == ">" and depth == 0:
                index += 1
                if index < len(script) and script[index] == ">":
                    index += 1
                target = self._scan_awk_redirect_target(script, index)
                self._check_path(target, cwd, "write")
                return
            index += 1

    def _scan_awk_getline_io(self, script: str, cwd: Path) -> None:
        """Path-check `getline`'s `< FILE` source past any receiving target.

        `getline` may fill a plain identifier, a field reference (`$0`,
        `$1`, a dynamic `$(expr)`), or an array element (`arr[i]`) before
        the optional `< FILE` clause; `_skip_awk_getline_target` advances
        past whichever form is present so the `<` is still found instead of
        the read check being silently skipped when the target is not a bare
        identifier. A prefix ending in `|&` (gawk's two-way coprocess pipe)
        executes a shell command exactly like a plain `|` and fails closed
        the same way (M5): checking `endswith("|")` alone missed it, since
        `&` — not `|` — is the trailing character.
        """
        for match in _AWK_GETLINE_PATTERN.finditer(script):
            prefix = script[: match.start()].rstrip()
            if prefix.endswith("|") or prefix.endswith("|&"):
                raise CommandPolicyViolation(
                    "awk pipe input to 'getline' executes an arbitrary shell "
                    "command and cannot be inspected safely"
                )
            cursor = match.end()
            while cursor < len(script) and script[cursor] in " \t":
                cursor += 1
            cursor = self._skip_awk_getline_target(script, cursor)
            while cursor < len(script) and script[cursor] in " \t":
                cursor += 1
            if cursor < len(script) and script[cursor] == "<":
                target = self._scan_awk_redirect_target(script, cursor + 1)
                self._check_path(target, cwd, "read")

    def _skip_awk_getline_target(self, script: str, cursor: int) -> int:
        """Advance past a `getline` receiving target, if one is present.

        Recognizes a plain identifier (optionally subscripted, `arr[i]`), a
        field reference (`$0`, `$NF`, or a parenthesized `$(expr)`), or a
        parenthesized expression; a bare `getline` (no target) or any other
        following syntax is left untouched, so the caller still finds an
        immediately following `< FILE`.
        """
        if cursor >= len(script):
            return cursor
        character = script[cursor]
        if character == "$":
            cursor += 1
            if cursor < len(script) and script[cursor].isdigit():
                while cursor < len(script) and script[cursor].isdigit():
                    cursor += 1
                return self._skip_awk_getline_subscript(script, cursor)
            identifier = _AWK_IDENTIFIER_PATTERN.match(script, cursor)
            if identifier is not None:
                return self._skip_awk_getline_subscript(script, identifier.end())
            if cursor < len(script) and script[cursor] == "(":
                return self._skip_awk_balanced(script, cursor, "(", ")")
            raise CommandPolicyViolation(
                "cannot safely inspect awk getline field target"
            )
        if character == "(":
            return self._skip_awk_balanced(script, cursor, "(", ")")
        identifier = _AWK_IDENTIFIER_PATTERN.match(script, cursor)
        if identifier is not None:
            return self._skip_awk_getline_subscript(script, identifier.end())
        return cursor

    def _skip_awk_getline_subscript(self, script: str, cursor: int) -> int:
        while cursor < len(script) and script[cursor] in " \t":
            cursor += 1
        if cursor < len(script) and script[cursor] == "[":
            return self._skip_awk_balanced(script, cursor, "[", "]")
        return cursor

    @staticmethod
    def _skip_awk_balanced(
        script: str,
        index: int,
        open_char: str,
        close_char: str,
    ) -> int:
        """Advance past one balanced `open_char`/`close_char` region.

        Quoted content inside is skipped whole (escape-aware) so a string
        literal carrying a stray `<`, `(`, `)`, `[`, or `]` cannot desynchronize
        depth tracking or be misread as the following redirect operator.
        """
        depth = 0
        quote: str | None = None
        escaped = False
        cursor = index
        while cursor < len(script):
            character = script[cursor]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                cursor += 1
                continue
            if character in {"'", '"'}:
                quote = character
                cursor += 1
                continue
            if character == open_char:
                depth += 1
            elif character == close_char:
                depth -= 1
                if depth == 0:
                    return cursor + 1
            cursor += 1
        raise CommandPolicyViolation("cannot safely inspect awk getline target")

    @staticmethod
    def _scan_awk_redirect_target(script: str, index: int) -> str:
        """Read a quoted-string-literal redirect target (I9/I13).

        Only a double-quoted string literal can be resolved to a concrete
        path statically; any other form (a bareword, a field reference like
        `$1`, a variable, a parenthesized expression) is dynamic from this
        lexer's point of view and is rejected rather than silently skipped.
        Awk concatenates adjacent expressions by bare juxtaposition (no
        operator), so anything other than a statement terminator following
        the closing quote (another string literal, an identifier, `$`, `(`,
        ...) means the true runtime target is this literal PLUS more text
        this lexer cannot resolve; that must fail closed too, rather than
        validating only the leading literal while the concatenated
        remainder silently escapes containment.
        """
        cursor = index
        while cursor < len(script) and script[cursor] in " \t":
            cursor += 1
        if cursor >= len(script) or script[cursor] != '"':
            raise CommandPolicyViolation("cannot resolve dynamic awk redirect target")
        start = cursor + 1
        cursor = start
        escaped = False
        while cursor < len(script):
            character = script[cursor]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                literal = script[start:cursor]
                trailer = cursor + 1
                while trailer < len(script) and script[trailer] in " \t":
                    trailer += 1
                if trailer < len(script) and script[trailer] not in ";\n}":
                    raise CommandPolicyViolation(
                        "cannot resolve dynamic awk redirect target"
                    )
                return literal
            cursor += 1
        raise CommandPolicyViolation("cannot safely inspect awk redirect target")

    def _check_dd(self, values: Sequence[str], cwd: Path) -> None:
        """`dd`: `if=`/`of=` operands are paths; every other `key=value`
        operand (`bs=`, `count=`, `seek=`, `iflag=`, `oflag=`, ...) is a
        scalar and is never a path.
        """
        self._reject_dynamic_values("dd arguments", values)
        for value in values:
            text = str(value)
            if text.startswith("if="):
                self._check_path(value.split("=", 1)[1], cwd, "read")
            elif text.startswith("of="):
                self._check_path(value.split("=", 1)[1], cwd, "write")

    def _check_base64(self, values: Sequence[str], cwd: Path) -> None:
        """`base64`: `-o`/`--output` writes; every remaining operand reads.

        `-i`/`--ignore-garbage` is a boolean flag (see `_BASE64_SHORT_FLAG_OPTIONS`);
        `-b`/`--break` and `-w`/`--wrap` take a scalar column-width value
        that is never a path. Long options resolve through
        `_resolve_long_option` (GNU unambiguous-prefix abbreviation, e.g.
        `--outp=` for `--output`), so an abbreviated spelling is classified
        identically to the full name instead of falling through as an
        unrecognized option — which would silently skip the write check on
        `--output`'s argument. Module invariant: ANY long option this
        allowlist cannot resolve fails closed, bare or `=value`-attached
        alike — an unmodeled flag is not assumed value-free, since one could
        otherwise consume (and hide) a later option's real path argument.
        Short options bundle through the same `_parse_short_option_cluster`
        substrate `sort`/`grep`/`curl`/`wget` use, so `-o` still consumes its
        argument even when not the leading character of the cluster (e.g.
        `-do`); an unrecognized short option fails closed the same way. A
        literal `--` ends option parsing; every token after it is a plain
        (read) operand even if it looks like an option.
        """
        self._reject_dynamic_values("base64 arguments", values)
        inputs: list[str] = []
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
                inputs.append(value)
                index += 1
                continue
            if text.startswith("--"):
                raw_option, separator, attached_text = text.partition("=")
                resolved = self._resolve_long_option(
                    raw_option, _BASE64_KNOWN_LONG_OPTIONS
                )
                if resolved == "--output":
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached_text)
                            if separator
                            else None
                        ),
                        context=f"base64 argument for {resolved}",
                    )
                    assert argument is not None
                    self._check_path(argument, cwd, "write")
                    continue
                if resolved in {"--break", "--wrap"}:
                    _, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached_text)
                            if separator
                            else None
                        ),
                        context=f"base64 argument for {resolved}",
                    )
                    continue
                if resolved in _BASE64_LONG_FLAG_OPTIONS:
                    if separator:
                        raise CommandPolicyViolation(
                            f"cannot safely resolve base64 option {raw_option}"
                        )
                    index += 1
                    continue
                raise CommandPolicyViolation(
                    f"cannot safely resolve base64 option {raw_option}"
                )
            index, _, _ = self._parse_short_option_cluster(
                values,
                index,
                cwd,
                flag_options=_BASE64_SHORT_FLAG_OPTIONS,
                value_options=_BASE64_SHORT_VALUE_OPTIONS,
            )
        for raw_path in inputs:
            self._check_path(raw_path, cwd, "read")

    def _check_gzip(self, values: Sequence[str], cwd: Path) -> None:
        """`gzip`: default mode replaces its operand with a derived `.gz` file.

        The `.gz` suffix (and any `--suffix`-overridden variant) is a
        distinct, non-enumerable path this guard does not compute, so
        default mode both write-checks the operand itself (gzip removes the
        original after compressing it) and poisons `unknown_effect` for the
        derived compressed file — additive to, not a replacement for, the
        operand's own containment check (M2). `-c`/`--stdout`/`-l`/`--list`/
        `-t`/`--test` never create or remove a file, so the operand stays a
        plain read in those modes — but that classification MUST be
        option-aware, not a position-independent character scan: `-S` takes
        a value (its own suffix argument, e.g. the `-t` in `-S -t <path>`),
        so a bare `t`/`c`/`l` character occurring as another option's
        consumed value must not be misread as the read-only mode flag
        (`_parse_gzip_short_cluster` tracks which characters were actually
        parsed as flags, not merely present in the token stream). Long
        options resolve through `_resolve_long_option` (GNU unambiguous-
        prefix abbreviation); module invariant: any option (short or long,
        bare or `=value`-attached) this allowlist cannot resolve fails
        closed rather than being treated as an operand or skipped.
        """
        self._reject_dynamic_values("gzip arguments", values)
        operands: list[str] = []
        read_only_mode = False
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
            if text.startswith("--"):
                raw_option, separator, attached_text = text.partition("=")
                resolved = self._resolve_long_option(
                    raw_option, _GZIP_KNOWN_LONG_OPTIONS
                )
                if resolved in _GZIP_SCALAR_LONG_OPTIONS:
                    _, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached_text)
                            if separator
                            else None
                        ),
                        context=f"gzip argument for {resolved}",
                    )
                    continue
                if resolved in _GZIP_READ_ONLY_LONG_OPTIONS:
                    if separator:
                        raise CommandPolicyViolation(
                            f"cannot safely resolve gzip option {raw_option}"
                        )
                    read_only_mode = True
                    index += 1
                    continue
                if resolved in _GZIP_LONG_FLAG_OPTIONS:
                    if separator:
                        raise CommandPolicyViolation(
                            f"cannot safely resolve gzip option {raw_option}"
                        )
                    index += 1
                    continue
                raise CommandPolicyViolation(
                    f"cannot safely resolve gzip option {raw_option}"
                )
            index, matched_flags = self._parse_gzip_short_cluster(values, index)
            if matched_flags & _GZIP_READ_ONLY_SHORT_FLAGS:
                read_only_mode = True
        access: PathAccess = "read" if read_only_mode else "write"
        if not read_only_mode:
            _active_validation_session().effects.unknown_effect = True
        for operand in operands:
            self._check_path(operand, cwd, access)

    @staticmethod
    def _parse_gzip_short_cluster(
        values: Sequence[str], index: int
    ) -> tuple[int, frozenset[str]]:
        """Parse one bundled gzip short-option token, e.g. `-lt`/`-9v`.

        Unlike `_parse_short_option_cluster`, this returns EVERY flag
        character actually consumed as a flag (not just the last
        value-option match), since `-c`/`-l`/`-t`'s read-only-mode
        classification depends on which characters were parsed as flags —
        `-S`'s own attached/separate suffix argument (e.g. the `-t` in
        `-S -t <path>`) must not count, even though it contains the same
        character. An unrecognized character fails closed.
        """
        source = values[index]
        options = str(source)[1:]
        seen_flags: set[str] = set()
        cursor = 0
        while cursor < len(options):
            option = options[cursor]
            if option in _GZIP_SHORT_FLAG_OPTIONS:
                seen_flags.add(option)
                cursor += 1
                continue
            if option not in _GZIP_SHORT_VALUE_OPTIONS:
                raise CommandPolicyViolation(
                    f"cannot safely resolve gzip option -{option}"
                )
            attached = options[cursor + 1 :]
            if attached:
                next_index = index + 1
            elif index + 1 < len(values):
                next_index = index + 2
            else:
                raise CommandPolicyViolation(f"missing gzip argument for -{option}")
            return next_index, frozenset(seen_flags)
        return index + 1, frozenset(seen_flags)

    def _check_rsync(self, values: Sequence[str], cwd: Path) -> None:
        """`rsync`: any remote operand rejects the WHOLE invocation (M5).

        A local invocation classifies its source/dest positionals plus its
        write/read-control options. `--link-dest` is a write slot even
        though it only *reads* its argument today: rsync may hard-link
        unchanged files from that tree straight into the destination, so a
        read-only external root is not sufficient authorization for it.
        `--log-file`/`--write-batch`/`--only-write-batch` also write to a
        filename argument. `--files-from`/`--include-from`/`--exclude-from`/
        `--read-batch`/`-f`/`--filter`/`-e`/`--rsh`/... delegate to an
        external file list or shell command this guard cannot inspect safely
        and fail closed unconditionally rather than access-checking their
        own argument: the file's *contents* can name further paths this
        guard never sees (`--read-batch`'s replayed changes are not bounded
        by the invocation's own destination operand the way a plain file
        argument would be), so a plain read check of the file itself would
        not bound what rsync actually touches.

        Short options bundle into one token (e.g. `-avz`); `-T`/`--temp-dir`
        is the only short option that carries a path (write). `-K`/
        `-k` (`--keep-dirlinks`/`--copy-dirlinks`) let rsync write through
        (or read through) an existing symlink at the destination/source
        instead of the literal directory entry, which can escape the
        intended root, so both fail closed regardless of position in a
        bundle, matching `--keep-dirlinks`'s long-option denial (see
        `_RSYNC_SHORT_DENIED_OPTIONS`).

        Long options resolve through `_resolve_long_option` (GNU
        unambiguous-prefix abbreviation, e.g. `--link-des=`/`--link-des ` for
        `--link-dest`), so an abbreviation of any modeled option — write,
        read, scalar, or denied — is classified identically to its full
        name in both the `=`-attached and space-separated spelling. Module
        invariant: any option (short or long, bare or `=value`-attached)
        this allowlist cannot resolve fails closed rather than being treated
        as an operand or a value-free flag.
        """
        self._reject_dynamic_values("rsync arguments", values)
        denied_options = {
            "-e",
            "--rsh",
            "-f",
            "--filter",
            "--files-from",
            "--include-from",
            "--exclude-from",
            "--read-batch",
            "--password-file",
            "--rsync-path",
            "--copy-links",
            "--copy-unsafe-links",
            "--keep-dirlinks",
            "--copy-dirlinks",
        }
        path_options: dict[str, PathAccess] = {
            "--backup-dir": "write",
            "--partial-dir": "write",
            "--temp-dir": "write",
            "--compare-dest": "read",
            "--copy-dest": "read",
            "--link-dest": "write",
            "--log-file": "write",
            "--write-batch": "write",
            "--only-write-batch": "write",
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
        known_long_options = (
            frozenset(path_options)
            | frozenset(scalar_options)
            | frozenset(option for option in denied_options if option.startswith("--"))
        )
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
            if not text.startswith("--"):
                # Scan leading flag characters; stop at the first one that
                # is not a plain flag (a denied character, `-T`'s value
                # slot, or an unmodeled character `_parse_short_option_cluster`
                # itself will fail closed on) rather than scanning past it,
                # since any remainder past a value-taking option's letter is
                # that option's own value, not another flag character.
                for character in text[1:]:
                    if character in _RSYNC_SHORT_FLAG_OPTIONS:
                        continue
                    if character in _RSYNC_SHORT_DENIED_OPTIONS:
                        raise CommandPolicyViolation(
                            f"cannot safely inspect rsync option -{character}"
                        )
                    break
                index, _, _ = self._parse_short_option_cluster(
                    values,
                    index,
                    cwd,
                    flag_options=_RSYNC_SHORT_FLAG_OPTIONS,
                    value_options=_RSYNC_SHORT_VALUE_OPTIONS,
                )
                continue
            option, separator, attached_text = text.partition("=")
            resolved_option = option
            if (
                option not in path_options
                and option not in scalar_options
                and option not in denied_options
            ):
                candidate = self._resolve_long_option(option, known_long_options)
                if candidate is not None:
                    resolved_option = candidate
            if resolved_option in denied_options:
                raise CommandPolicyViolation(
                    f"cannot safely inspect rsync option {option}"
                )
            if resolved_option in path_options:
                argument, index = self._take_option_argument(
                    values,
                    index,
                    attached_argument=(
                        self._derived_value(value, attached_text) if separator else None
                    ),
                    context=f"rsync argument for {resolved_option}",
                )
                assert argument is not None
                self._check_path(argument, cwd, path_options[resolved_option])
                continue
            if resolved_option in scalar_options:
                _, index = self._take_option_argument(
                    values,
                    index,
                    attached_argument=(
                        self._derived_value(value, attached_text) if separator else None
                    ),
                    context=f"rsync argument for {resolved_option}",
                )
                continue
            # Module invariant: an unresolved long option fails closed, bare
            # or `=value`-attached alike.
            raise CommandPolicyViolation(f"cannot safely inspect rsync option {option}")

        # N1: the remote-operand check must run regardless of operand count
        # — a single-operand invocation is not exempt from it — and a sole
        # local operand still gets contained as an ordinary read instead of
        # skipping containment entirely for lack of a second operand.
        if any(self._is_remote_transfer_operand(operand) for operand in operands):
            raise CommandPolicyViolation("cannot safely inspect remote rsync operands")
        if not operands:
            return
        if len(operands) == 1:
            self._check_path(operands[0], cwd, "read")
            return
        for operand in operands[:-1]:
            self._check_path(operand, cwd, "read")
        self._check_path(operands[-1], cwd, "write")

    @staticmethod
    def _is_remote_transfer_operand(value: str) -> bool:
        """Return whether `value` is a remote address, not a local path.

        Covers `rsync://host/path`, `host:path`, and `user@host:path`.
        """
        text = str(value)
        return text.startswith("rsync://") or ":" in text

    def _classify_url_operand(self, value: str, cwd: Path, access: PathAccess) -> None:
        """Classify a curl/wget positional URL operand (T3).

        A `file:` scheme is a real local filesystem channel — not a network
        transfer — so it is resolved to a path and checked with `access`
        (the caller decides read vs. write: a plain source URL is a read;
        curl's upload target when `-T`/`--upload-file` is in play is a
        write). A recognized network scheme
        (`_CURL_WGET_NETWORK_URL_SCHEMES`) is not a local path and is
        allowed through unchecked. Module invariant: any other or
        unparsable scheme fails closed — this family cannot assume an
        exotic/unmodeled scheme (some of which, e.g. `smb:`/`scp:`-like
        variants, can themselves resolve to local or attacker-controlled
        resources) is safely non-local.
        """
        text = str(value)
        match = _URL_SCHEME_PATTERN.match(text)
        if match is None:
            raise CommandPolicyViolation(
                f"cannot safely resolve operand {text} as a URL"
            )
        scheme = match.group(1).lower()
        has_authority_separator = match.group(2) == "//"
        remainder = match.group(3)
        if scheme == "file":
            if has_authority_separator and not remainder.startswith("/"):
                raise CommandPolicyViolation(
                    "cannot safely resolve a file:// URL host component"
                )
            self._check_path(self._derived_value(value, remainder), cwd, access)
            return
        if scheme in _CURL_WGET_NETWORK_URL_SCHEMES:
            return
        raise CommandPolicyViolation(f"cannot safely resolve URL scheme {scheme}:")

    def _check_curl(self, values: Sequence[str], cwd: Path) -> None:
        """`curl`: `-o`/`--output`/`--output-dir` write; `-T`/`--upload-file`/
        `--netrc-file`/`--cookie` read; an `@file` payload to `-d`/`--data*`
        reads that file. `--cookie` is modeled explicitly (not left to
        prefix-abbreviation) because it is a real, distinct option from
        `--cookie-jar`: leaving it unresolved let its own unambiguous-prefix
        match snap to `--cookie-jar` (a write path) and write-check a file
        the invocation only ever reads (N2). `-D`/`--dump-header`,
        `-c`/`--cookie-jar`, `--trace`, `--trace-ascii`, `--stderr`,
        `--libcurl`, `--hsts`, and `--etag-save` also write to a filename
        argument. `-K`/`--config` (arbitrary runtime options) and
        `-O`/`--remote-name[-all]` (output filename derived from the remote
        URL or response, not statically knowable) cannot be inspected safely
        and fail closed, including bundled behind another short flag (e.g.
        `-sO`). Long options resolve through `_resolve_long_option` (GNU
        unambiguous-prefix abbreviation, e.g. `--dump-hea=` for
        `--dump-header`); every strict abbreviation of `--cookie` is also a
        prefix of `--cookie-jar`, so once both are modeled only the exact
        `--cookie` spelling resolves and a truncated one is ambiguous and
        fails closed, as this family's invariant requires. Module invariant:
        any option (short or long, bare or `=value`-attached) this
        allowlist cannot resolve fails closed. Short options bundle through
        the same `_parse_short_option_cluster` substrate `sort`/`grep`/
        `install` use, so a value-taking option (`-o`/`-D`/`-c`/`-T`) still
        consumes its argument even when it is not the leading character of
        the cluster (e.g. `-sD`). A literal `--` ends option parsing.

        The positional URL/operand (T3) is classified through
        `_classify_url_operand`: a `file:` URL is a real filesystem channel
        (read by default; write when `-T`/`--upload-file` is anywhere in the
        invocation, since that turns the request into an upload whose
        destination is the URL — `has_upload_target` is computed by a
        presence-only pre-scan so the classification does not depend on
        argv order, matching curl's own order-independent option parsing).
        """
        self._reject_dynamic_values("curl arguments", values)
        has_upload_target = self._curl_has_upload_target(values)
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
                self._classify_url_operand(
                    value, cwd, "write" if has_upload_target else "read"
                )
                index += 1
                continue
            if text.startswith("--"):
                raw_option, separator, attached_text = text.partition("=")
                resolved = self._resolve_long_option(
                    raw_option, _CURL_KNOWN_LONG_OPTIONS
                )
                if resolved in _CURL_HARD_DENY_LONG_OPTIONS:
                    raise CommandPolicyViolation(
                        "cannot safely resolve curl remote output filename"
                    )
                if resolved in _CURL_CONFIG_LONG_OPTIONS:
                    raise CommandPolicyViolation(
                        "cannot safely inspect curl runtime configuration"
                    )
                if resolved in _CURL_WRITE_LONG_OPTIONS or (
                    resolved in _CURL_READ_LONG_OPTIONS
                ):
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached_text)
                            if separator
                            else None
                        ),
                        context=f"curl argument for {resolved}",
                    )
                    assert argument is not None
                    access: PathAccess = (
                        "write" if resolved in _CURL_WRITE_LONG_OPTIONS else "read"
                    )
                    self._check_path(argument, cwd, access)
                    continue
                if resolved in _CURL_DATA_LONG_OPTIONS:
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached_text)
                            if separator
                            else None
                        ),
                        context=f"curl argument for {resolved}",
                    )
                    assert argument is not None
                    if str(argument).startswith("@"):
                        self._check_path(
                            self._derived_value(argument, str(argument)[1:]),
                            cwd,
                            "read",
                        )
                    continue
                if resolved in _CURL_KNOWN_LONG_FLAG_OPTIONS:
                    if separator:
                        raise CommandPolicyViolation(
                            f"cannot safely inspect curl option {raw_option}"
                        )
                    index += 1
                    continue
                raise CommandPolicyViolation(
                    f"cannot safely inspect curl option {raw_option}"
                )
            # `-O`/`-K` fail closed even bundled behind another flag
            # (e.g. `-sO`); scanning stops at the first character that
            # is not a known no-value flag, since anything past a
            # value-taking option is that option's value, not another
            # flag letter.
            for character in text[1:]:
                if character in _CURL_SHORT_FLAG_OPTIONS:
                    continue
                if character == "O":
                    raise CommandPolicyViolation(
                        "cannot safely resolve curl remote output filename"
                    )
                if character == "K":
                    raise CommandPolicyViolation(
                        "cannot safely inspect curl runtime configuration"
                    )
                break
            index, matched_option, argument = self._parse_short_option_cluster(
                values,
                index,
                cwd,
                flag_options=_CURL_SHORT_FLAG_OPTIONS,
                value_options=_CURL_SHORT_VALUE_OPTIONS,
            )
            if (
                matched_option == "d"
                and argument is not None
                and str(argument).startswith("@")
            ):
                self._check_path(
                    self._derived_value(argument, str(argument)[1:]),
                    cwd,
                    "read",
                )

    @staticmethod
    def _curl_has_upload_target(values: Sequence[str]) -> bool:
        """Detect `-T`/`--upload-file` anywhere in argv (presence only).

        curl's own option parsing does not require `-T` to precede the URL
        operand, so the URL's read-vs-write classification must not depend
        on argv order either: this mirrors the same short-cluster and
        long-option resolution `_check_curl`'s real pass uses, but only
        checks presence — it performs no path checks of its own.
        """
        options_done = False
        index = 0
        while index < len(values):
            text = str(values[index])
            if not options_done and text == "--":
                options_done = True
                index += 1
                continue
            if options_done or not text.startswith("-") or text == "-":
                index += 1
                continue
            if text.startswith("--"):
                raw_option = text.partition("=")[0]
                if (
                    WorkspaceCommandPathGuard._resolve_long_option(
                        raw_option, _CURL_KNOWN_LONG_OPTIONS
                    )
                    == "--upload-file"
                ):
                    return True
                index += 1
                continue
            for character in text[1:]:
                if character in _CURL_SHORT_FLAG_OPTIONS:
                    continue
                if character == "T":
                    return True
                break
            index += 1
        return False

    def _check_wget(self, values: Sequence[str], cwd: Path) -> None:
        """`wget`: `-O`/`--output-document` write; `-P`/`--directory-prefix`
        write directory; `-o`/`--output-file` (log file) write;
        `--save-cookies` write; `--post-file`/`--body-file` read. `--config`
        and `-i`/`--input-file` (a runtime URL list) cannot be inspected
        safely and fail closed, including bundled behind another short flag
        (e.g. `-qi`). A URL operand with no explicit `-O` output (and not
        `--spider`, which fetches nothing to disk) resolves its output
        filename from the remote response, which is not statically
        knowable, so that combination fails closed too. Long options resolve
        through `_resolve_long_option` (GNU unambiguous-prefix abbreviation,
        e.g. `--output-docu=` for `--output-document`). Module invariant:
        any option (short or long, bare or `=value`-attached) this
        allowlist cannot resolve fails closed. Short options bundle through
        the same `_parse_short_option_cluster` substrate `curl`/`sort`/
        `grep`/`install` use, so `-O`/`-P`/`-o` still consume their argument
        even when not the leading character of the cluster (e.g. `-qO`). A
        literal `--` ends option parsing.

        The positional URL/operand (T3) is classified through
        `_classify_url_operand` as a plain read: wget's URL is always the
        fetch source, never an upload target.
        """
        self._reject_dynamic_values("wget arguments", values)
        has_explicit_output = False
        spider_mode = False
        has_url_operand = False
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
                self._classify_url_operand(value, cwd, "read")
                has_url_operand = True
                index += 1
                continue
            if text.startswith("--"):
                raw_option, separator, attached_text = text.partition("=")
                resolved = self._resolve_long_option(
                    raw_option, _WGET_KNOWN_LONG_OPTIONS
                )
                if resolved in _WGET_CONFIG_LONG_OPTIONS:
                    raise CommandPolicyViolation(
                        "cannot safely inspect wget runtime configuration"
                    )
                if resolved in _WGET_HARD_DENY_LONG_OPTIONS:
                    raise CommandPolicyViolation(
                        "cannot safely inspect wget runtime URL list"
                    )
                if resolved == "--spider":
                    spider_mode = True
                    index += 1
                    continue
                if resolved in _WGET_WRITE_LONG_OPTIONS or (
                    resolved in _WGET_READ_LONG_OPTIONS
                ):
                    argument, index = self._take_option_argument(
                        values,
                        index,
                        attached_argument=(
                            self._derived_value(value, attached_text)
                            if separator
                            else None
                        ),
                        context=f"wget argument for {resolved}",
                    )
                    assert argument is not None
                    access: PathAccess = (
                        "write" if resolved in _WGET_WRITE_LONG_OPTIONS else "read"
                    )
                    self._check_path(argument, cwd, access)
                    if resolved == "--output-document":
                        has_explicit_output = True
                    continue
                if resolved in _WGET_KNOWN_LONG_FLAG_OPTIONS:
                    if separator:
                        raise CommandPolicyViolation(
                            f"cannot safely inspect wget option {raw_option}"
                        )
                    index += 1
                    continue
                raise CommandPolicyViolation(
                    f"cannot safely inspect wget option {raw_option}"
                )
            # `-i` fails closed even bundled behind another flag (e.g.
            # `-qi`); scanning stops at the first character that is not
            # a known no-value flag, since anything past a value-taking
            # option is that option's value, not another flag letter.
            for character in text[1:]:
                if character in _WGET_SHORT_FLAG_OPTIONS:
                    continue
                if character == "i":
                    raise CommandPolicyViolation(
                        "cannot safely inspect wget runtime URL list"
                    )
                break
            index, matched_option, _ = self._parse_short_option_cluster(
                values,
                index,
                cwd,
                flag_options=_WGET_SHORT_FLAG_OPTIONS,
                value_options=_WGET_SHORT_VALUE_OPTIONS,
            )
            if matched_option == "O":
                has_explicit_output = True
        if has_url_operand and not has_explicit_output and not spider_mode:
            raise CommandPolicyViolation(
                "cannot safely resolve wget remote output filename"
            )

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
    def _is_trusted_system_command(command_word: str) -> bool:
        try:
            resolve_trusted_executable(command_word)
            return True
        except CommandPolicyViolation:
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
            interpreter_parts = first_line[2:].strip().split()
            interpreter = (
                os.path.basename(interpreter_parts[0]) if interpreter_parts else ""
            )
            if interpreter == "env":
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
        invocation = self._parse_bash_invocation(literals)
        for initialization_file in invocation.initialization_files:
            self._inspect_shell_script(
                initialization_file,
                state,
                propagate_state=False,
            )
        if invocation.command_text is not None:
            self._validate_shell_states(invocation.command_text, state)
            return
        if invocation.reads_stdin:
            listing_context = (
                " after listing options" if invocation.lists_options else ""
            )
            raise CommandPolicyViolation(
                f"cannot inspect {command_name} input without command text or a script"
                f"{listing_context}"
            )
        if invocation.script is None:
            raise CommandPolicyViolation(
                f"cannot inspect {command_name} input without command text or a script"
            )
        self._inspect_shell_script(invocation.script, state, propagate_state=False)

    @staticmethod
    def _parse_bash_invocation(
        values: Sequence[str],
    ) -> _BashInvocation:
        initialization_files: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if isinstance(value, _CommandValue) and not value.is_static:
                raise CommandPolicyViolation(
                    "cannot safely inspect dynamic bash option region"
                )
            if value == "--":
                index += 1
                return _BashInvocation(
                    initialization_files=tuple(initialization_files),
                    script=values[index] if index < len(values) else None,
                    reads_stdin=index >= len(values),
                )
            if value in _BASH_FILE_OPTIONS:
                index += 1
                if index >= len(values):
                    raise CommandPolicyViolation(f"missing bash argument for {value}")
                argument = values[index]
                if isinstance(argument, _CommandValue) and not argument.is_static:
                    raise CommandPolicyViolation(
                        "cannot safely inspect dynamic bash option region"
                    )
                initialization_files.append(argument)
                index += 1
                continue
            if any(value.startswith(f"{option}=") for option in _BASH_FILE_OPTIONS):
                raise CommandPolicyViolation(
                    f"cannot safely inspect bash option {value}"
                )
            if value in _BASH_LONG_FLAG_OPTIONS:
                index += 1
                continue
            if value.startswith("--"):
                raise CommandPolicyViolation(
                    f"cannot safely inspect bash option {value}"
                )
            if value == "-":
                return _BashInvocation(
                    initialization_files=tuple(initialization_files),
                    reads_stdin=True,
                )
            if value.startswith(("-", "+")) and len(value) > 1:
                named_option_count = 0
                has_command_text = False
                stdin_mode = False
                for option in value[1:]:
                    if option in {"o", "O"}:
                        named_option_count += 1
                    elif option == "c":
                        has_command_text = True
                    elif option == "s":
                        stdin_mode = True
                    elif option not in _BASH_SHORT_FLAG_OPTIONS:
                        raise CommandPolicyViolation(
                            f"cannot safely inspect bash option {value}"
                        )

                index += 1
                for _ in range(named_option_count):
                    if index >= len(values):
                        return _BashInvocation(
                            initialization_files=tuple(initialization_files),
                            reads_stdin=True,
                            lists_options=True,
                        )
                    argument = values[index]
                    if isinstance(argument, _CommandValue) and not argument.is_static:
                        raise CommandPolicyViolation(
                            "cannot safely inspect dynamic bash option region"
                        )
                    index += 1
                if has_command_text:
                    if index >= len(values):
                        raise CommandPolicyViolation(
                            f"missing bash argument for {value}"
                        )
                    command_text = values[index]
                    if (
                        isinstance(command_text, _CommandValue)
                        and not command_text.is_static
                    ):
                        raise CommandPolicyViolation(
                            "cannot safely inspect dynamic bash option region"
                        )
                    return _BashInvocation(
                        initialization_files=tuple(initialization_files),
                        command_text=command_text,
                    )
                if stdin_mode:
                    return _BashInvocation(
                        initialization_files=tuple(initialization_files),
                        reads_stdin=True,
                    )
                continue
            return _BashInvocation(
                initialization_files=tuple(initialization_files),
                script=value,
            )
        return _BashInvocation(
            initialization_files=tuple(initialization_files),
            reads_stdin=True,
        )

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
    def _operands(
        values: Sequence[str],
        *,
        value_options: frozenset[str] = frozenset(),
    ) -> list[str]:
        operands: list[str] = []
        options_done = False
        skip_next_value = False
        for value in values:
            if skip_next_value:
                # Separated value of a preceding value-consuming option.
                skip_next_value = False
                continue
            if not options_done and value == "--":
                options_done = True
                continue
            if not options_done and value.startswith("-") and value != "-":
                # An option that consumes a following token must not let that
                # value be misread as a path operand. Attached forms (`-m0755`,
                # `--mode=0755`) are already dropped whole; only a separated
                # value needs to be skipped explicitly.
                if value in value_options:
                    skip_next_value = True
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

        # Invariant: the find-clause observer must fire before any sentinel
        # short-circuit below (including the `{}` no-op resolution). A `{}`
        # write operand that returns early first would never reach the
        # observer, so `find`'s write-root classification would silently miss
        # every `{}` exec argument.
        find_observer = _active_validation_session().find_clause_observer
        if find_observer is not None:
            find_observer(raw_path, access)

        if (
            isinstance(raw_path, _CommandValue)
            and raw_path.find_placeholder_cwd is not None
        ):
            return raw_path.find_placeholder_cwd

        if raw_path in {"", "-"}:
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
        if access == "write":
            _active_validation_session().effects.written_paths.add(resolved)
        return resolved
