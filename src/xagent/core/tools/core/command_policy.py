"""Lightweight contracts shared by command policy parsing and execution."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

PathAccess = Literal["read", "write"]


class CommandPolicyViolation(ValueError):
    """A command cannot be authorized under the active cooperative policy."""


class CommandPathViolation(CommandPolicyViolation):
    """A statically identified command path falls outside its allowed roots."""

    def __init__(self, *, access: PathAccess, path: Path) -> None:
        self.access = access
        self.path = path
        super().__init__(f"{path} is outside allowed {access} paths")


def resolve_policy_shell_executable() -> str:
    """Return the Bash executable whose grammar is owned by the policy parser."""
    executable = shutil.which("bash")
    if executable is None:
        raise CommandPolicyViolation(
            "restricted command execution requires a Bash executable"
        )
    return executable
