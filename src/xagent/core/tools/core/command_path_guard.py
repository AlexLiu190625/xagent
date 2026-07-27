"""Best-effort workspace path guard for shell commands.

This module deliberately provides a cooperative guard, not an operating-system
security boundary. It rejects statically identifiable out-of-scope paths for a
supported set of common file commands while leaving unknown commands unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import bashlex

from ...workspace import TaskWorkspace
from .command_policy import CommandPathViolation, PathAccess

logger = logging.getLogger(__name__)

_MAX_INSPECTED_SCRIPT_BYTES = 1024 * 1024

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
_SAFE_DEVICE_PATHS = {
    Path("/dev/null"),
    Path("/dev/stdin"),
    Path("/dev/stdout"),
    Path("/dev/stderr"),
}


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
    def _parse_shell(command: str) -> list[Any] | None:
        try:
            return cast(list[Any], bashlex.parse(command))
        except (bashlex.errors.ParsingError, NotImplementedError, ValueError) as exc:
            logger.debug(
                "Skipping command path guard for unparsed shell input: %s", exc
            )
            return None

    def validate(self, command: str) -> None:
        """Reject statically identifiable out-of-scope paths in ``command``.

        Shell syntax that cannot be parsed or path operands containing runtime
        expansion are intentionally left to the shell. This preserves existing
        command behavior and keeps the contract honest: the guard reduces risk
        for supported commands but is not a complete sandbox.
        """
        nodes = self._parse_shell(command)
        if nodes is None:
            return

        cwd = self._initial_cwd
        for node in nodes:
            cwd = self._validate_node(node, cwd)

    def validate_argv(self, argv: Sequence[str]) -> None:
        """Reject out-of-policy paths in an argument-vector command."""
        if not argv:
            return
        command_name = os.path.basename(argv[0])
        self._validate_command_values(command_name, argv[1:], self._initial_cwd)

    def _validate_node(self, node: Any, cwd: Path) -> Path:
        kind = getattr(node, "kind", None)
        if kind == "list":
            current = cwd
            for part in node.parts:
                if getattr(part, "kind", None) in {"operator", "pipe"}:
                    continue
                current = self._validate_node(part, current)
            return current

        if kind == "pipeline":
            for part in node.parts:
                if getattr(part, "kind", None) != "pipe":
                    self._validate_node(part, cwd)
            return cwd

        if kind == "compound":
            current = cwd
            for part in getattr(node, "list", ()):
                current = self._validate_node(part, current)
            for redirect in getattr(node, "redirects", ()):
                self._validate_redirect(redirect, current)
            # Propagating the resulting CWD is conservative for subshells and
            # correct for brace groups.
            return current

        if kind == "command":
            return self._validate_command(node, cwd)

        self._validate_nested_nodes(node, cwd)
        return cwd

    def _validate_command(self, node: Any, cwd: Path) -> Path:
        words: list[Any] = []
        for part in node.parts:
            kind = getattr(part, "kind", None)
            if kind == "redirect":
                self._validate_redirect(part, cwd)
            elif kind == "word":
                words.append(part)
                self._validate_nested_nodes(part, cwd)
            else:
                self._validate_nested_nodes(part, cwd)

        if not words:
            return cwd

        command_word = self._literal_word(words[0])
        if command_word is None:
            return cwd
        command_name = os.path.basename(command_word)
        args = self._literal_values(words[1:])

        return self._validate_command_values(command_name, args, cwd)

    def _validate_command_values(
        self,
        command_name: str,
        args: Sequence[str],
        cwd: Path,
    ) -> Path:
        if command_name in {"cd", "pushd"}:
            target_word = self._first_operand(args)
            if target_word is None:
                return self._check_path(str(Path.home()), cwd, "read")
            if target_word == "-":
                raise CommandPathViolation(
                    access="read",
                    path=Path("<dynamic cd target>"),
                )
            return self._check_path(target_word, cwd, "read")

        if command_name in _READ_COMMANDS:
            self._check_read_command(command_name, args, cwd)
        elif command_name in _WRITE_COMMANDS:
            self._check_operands(args, cwd, "write")
        elif command_name == "grep":
            self._check_grep(args, cwd)
        elif command_name == "sed":
            self._check_sed(args, cwd)
        elif command_name == "awk":
            self._check_awk(args, cwd)
        elif command_name == "cp":
            self._check_copy(args, cwd)
        elif command_name in {"mv", "ln"}:
            self._check_move_or_link(args, cwd)
        elif command_name == "find":
            self._check_find(args, cwd)
        elif command_name in _SHELL_COMMANDS:
            self._check_nested_shell(args, cwd)
        elif command_name in {".", "source"}:
            self._check_operands(args, cwd, "read")
        elif command_name == "xargs":
            self._check_xargs(args, cwd)

        return cwd

    def _validate_redirect(self, node: Any, cwd: Path) -> None:
        redirect_type = getattr(node, "type", "")
        if redirect_type in {"<<", "<<<"}:
            return
        output = getattr(node, "output", None)
        raw_path = self._literal_word(output)
        if raw_path is None:
            return
        access: PathAccess = "read" if redirect_type == "<" else "write"
        self._check_path(raw_path, cwd, access)

    def _check_operands(
        self,
        values: Sequence[str],
        cwd: Path,
        access: PathAccess,
    ) -> None:
        operands = self._operands(values)
        for raw_path in operands:
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
        index = 0
        while index < len(values):
            value = values[index]
            if value in short_options | long_options:
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

    def _check_sed(self, values: Sequence[str], cwd: Path) -> None:
        self._check_sed_values(values, cwd)

    def _check_awk(self, values: Sequence[str], cwd: Path) -> None:
        self._check_awk_values(values, cwd)

    def _check_sed_values(self, values: Sequence[str], cwd: Path) -> None:
        positionals: list[str] = []
        explicit_script = False
        in_place = self._sed_requests_in_place(values)
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            if value in {"-f", "--file"}:
                explicit_script = True
                if index + 1 < len(values):
                    self._inspect_script_file("sed", values[index + 1], cwd)
                index += 2
                continue
            if value.startswith("--file="):
                explicit_script = True
                self._inspect_script_file("sed", value.split("=", 1)[1], cwd)
                index += 1
                continue
            if value.startswith("-f") and len(value) > 2:
                explicit_script = True
                self._inspect_script_file("sed", value[2:], cwd)
                index += 1
                continue
            if value in {"-e", "--expression"}:
                explicit_script = True
                if index + 1 < len(values):
                    self._reject_embedded_io("sed", values[index + 1])
                index += 2
                continue
            if value.startswith("-e") or value.startswith("--expression="):
                explicit_script = True
                script = value.split("=", 1)[1] if "=" in value else value[2:]
                self._reject_embedded_io("sed", script)
                index += 1
                continue
            if (
                value == "-i"
                or value.startswith("-i")
                or value.startswith("--in-place")
            ):
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            positionals.append(value)
            index += 1

        if not explicit_script and positionals:
            self._reject_embedded_io("sed", positionals[0])
        file_operands = positionals if explicit_script else positionals[1:]
        for raw_path in file_operands:
            self._check_path(raw_path, cwd, "write" if in_place else "read")

    @staticmethod
    def _sed_requests_in_place(values: Sequence[str]) -> bool:
        return any(
            value == "-i" or value.startswith("-i") or value.startswith("--in-place")
            for value in values
        )

    def _check_awk_values(self, values: Sequence[str], cwd: Path) -> None:
        positionals: list[str] = []
        explicit_program = False
        index = 0
        while index < len(values):
            value = values[index]
            if value == "--":
                positionals.extend(values[index + 1 :])
                break
            if value in {"-f", "--file"}:
                explicit_program = True
                if index + 1 < len(values):
                    self._inspect_script_file("awk", values[index + 1], cwd)
                index += 2
                continue
            if value.startswith("--file="):
                explicit_program = True
                self._inspect_script_file("awk", value.split("=", 1)[1], cwd)
                index += 1
                continue
            if value.startswith("-f") and len(value) > 2:
                explicit_program = True
                self._inspect_script_file("awk", value[2:], cwd)
                index += 1
                continue
            if value in {"-e", "--source"}:
                explicit_program = True
                if index + 1 < len(values):
                    self._reject_embedded_io("awk", values[index + 1])
                index += 2
                continue
            if value.startswith("-e") or value.startswith("--source="):
                explicit_program = True
                program = value.split("=", 1)[1] if "=" in value else value[2:]
                self._reject_embedded_io("awk", program)
                index += 1
                continue
            if value in {"-F", "-v"}:
                index += 2
                continue
            if value.startswith("-F") or value.startswith("-v"):
                index += 1
                continue
            if value.startswith("-") and value != "-":
                index += 1
                continue
            if "=" not in value:
                positionals.append(value)
            index += 1

        if not explicit_program and positionals:
            self._reject_embedded_io("awk", positionals[0])
        file_operands = positionals if explicit_program else positionals[1:]
        for raw_path in file_operands:
            self._check_path(raw_path, cwd, "read")

    def _inspect_script_file(
        self,
        language: Literal["sed", "awk"],
        raw_path: str,
        cwd: Path,
    ) -> None:
        script_path = self._check_path(raw_path, cwd, "read")
        if self._path_access_observer is not None:
            return

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
        except (OSError, UnicodeError) as exc:
            # A missing script may be created earlier in the same shell string;
            # skipping inspection would allow its embedded file I/O unchecked.
            raise CommandPathViolation(access="read", path=script_path) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        self._reject_embedded_io(language, script)

    @staticmethod
    def _reject_embedded_io(language: Literal["sed", "awk"], script: str) -> None:
        if language == "awk":
            unsafe = bool(
                re.search(r"\bsystem\s*\(", script)
                or re.search(r"\bgetline\b", script)
                or re.search(r">>?\s*[\"']", script)
                or re.search(r"\|\s*[\"']", script)
            )
        else:
            unsafe = WorkspaceCommandPathGuard._sed_has_unsafe_io(script)
        if unsafe:
            raise CommandPathViolation(
                access="write",
                path=Path(f"<{language} embedded file I/O>"),
            )

    @staticmethod
    def _sed_has_unsafe_io(script: str) -> bool:
        address = r"(?:\d+|\$|/(?:\\.|[^/\n])*/)"
        file_or_exec_command = re.compile(
            rf"(?:^|[;\n])\s*(?:{address}(?:\s*,\s*{address})?\s*)?"
            r"[rRwWe](?:\s|$)"
        )
        if file_or_exec_command.search(script):
            return True

        # Parse substitution flags sufficiently to catch GNU sed's ``e`` and
        # ``w file`` extensions without treating ordinary replacement text as
        # commands. Dynamic/constructed sed programs remain outside this soft
        # guard's contract.
        for index, char in enumerate(script[:-1]):
            if char != "s":
                continue
            if index > 0 and script[index - 1] not in ";\n/0123456789$ \t":
                continue
            delimiter = script[index + 1]
            if delimiter.isalnum() or delimiter.isspace() or delimiter == "\\":
                continue

            cursor = index + 2
            complete = True
            for _ in range(2):
                escaped = False
                while cursor < len(script):
                    current = script[cursor]
                    cursor += 1
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == delimiter:
                        break
                else:
                    complete = False
                    break
            if not complete:
                continue

            flags_end = cursor
            while flags_end < len(script) and script[flags_end] not in ";\n":
                flags_end += 1
            flags = script[cursor:flags_end].strip()
            if "e" in flags or "w" in flags:
                return True
        return False

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

    def _check_find(self, literals: Sequence[str], cwd: Path) -> None:
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
        exec_commands = self._parse_find_exec_commands(expression)
        access: PathAccess = (
            "write"
            if "-delete" in expression
            or any(
                self._find_exec_command_writes(command, cwd)
                for command in exec_commands
            )
            else "read"
        )
        for root in roots:
            self._check_path(root, cwd, access)

        for command in exec_commands:
            self._validate_nested_command_words(command, cwd)

    @staticmethod
    def _parse_find_exec_commands(
        expression: Sequence[str],
    ) -> list[tuple[str, ...]]:
        commands: list[tuple[str, ...]] = []
        index = 0
        while index < len(expression):
            if expression[index] not in {"-exec", "-execdir"}:
                index += 1
                continue
            nested: list[str] = []
            index += 1
            while index < len(expression) and expression[index] not in {";", "+"}:
                nested.append(expression[index])
                index += 1
            if nested:
                commands.append(tuple(nested))
            index += 1
        return commands

    def _find_exec_command_writes(
        self,
        command: Sequence[str],
        cwd: Path,
    ) -> bool:
        if not command:
            return False

        writes_placeholder = False

        def observe_path(raw_path: str, access: PathAccess) -> None:
            nonlocal writes_placeholder
            if access == "write" and "{}" in raw_path:
                writes_placeholder = True

        probe = WorkspaceCommandPathGuard(
            self._workspace,
            _path_access_observer=observe_path,
        )
        try:
            probe._validate_nested_command_words(command, cwd)
        except CommandPathViolation:
            # The regular validation pass reports statically unsafe embedded
            # programs. This probe only classifies placeholder path access.
            pass
        return writes_placeholder

    def _check_nested_shell(self, literals: Sequence[str], cwd: Path) -> None:
        for index, value in enumerate(literals[:-1]):
            if value.startswith("-") and "c" in value[1:]:
                self._validate_shell_text(literals[index + 1], cwd)
                return
        self._check_operands(literals, cwd, "read")

    def _check_xargs(self, values: Sequence[str], cwd: Path) -> None:
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
            self._validate_nested_command_words(values[command_start:], cwd)

    def _validate_nested_command_words(self, words: Sequence[str], cwd: Path) -> None:
        if not words:
            return
        command_name = os.path.basename(words[0])
        self._validate_command_values(command_name, words[1:], cwd)

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

    def _validate_shell_text(self, command: str, cwd: Path) -> None:
        nodes = self._parse_shell(command)
        if nodes is None:
            return
        current = cwd
        for node in nodes:
            current = self._validate_node(node, current)

    def _validate_nested_nodes(self, node: Any, cwd: Path) -> None:
        for attr_name in ("parts", "command", "list"):
            child = getattr(node, attr_name, None)
            if isinstance(child, list):
                for item in child:
                    if getattr(item, "kind", None) is not None:
                        self._validate_node(item, cwd)
            elif getattr(child, "kind", None) is not None:
                self._validate_node(child, cwd)

    @staticmethod
    def _literal_word(word: Any) -> str | None:
        if word is None or getattr(word, "kind", None) != "word":
            return None
        if getattr(word, "parts", None):
            return None
        return str(word.word)

    def _literal_values(self, words: Sequence[Any]) -> list[str]:
        return [
            value for word in words if (value := self._literal_word(word)) is not None
        ]

    def _first_operand(self, values: Sequence[str]) -> str | None:
        operands = self._operands(values)
        return operands[0] if operands else None

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
