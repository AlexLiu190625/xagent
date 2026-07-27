"""Contracts shared by command policy implementations and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Sequence

PathAccess = Literal["read", "write"]


class CommandPolicyViolation(ValueError):
    """A command cannot be authorized under the active cooperative policy."""


class CommandPathViolation(CommandPolicyViolation):
    """A command path falls outside its allowed roots."""

    def __init__(self, *, access: PathAccess, path: Path) -> None:
        self.access = access
        self.path = path
        super().__init__(f"path is outside allowed {access} paths")


class CommandPolicyGuard(Protocol):
    """Validate command representations before process creation."""

    def validate(self, command: str) -> None: ...

    def validate_argv(self, argv: Sequence[str]) -> None: ...
