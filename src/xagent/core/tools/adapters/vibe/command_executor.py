"""
Command Line Execution Tool for xagent
Framework wrapper around the pure command executor tool
"""

import asyncio
import logging
from dataclasses import replace
from typing import Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ....execution_scope import ExecutionScope
from ....workspace import TaskWorkspace
from ...core.command_executor import (
    CommandExecutorCore,
    execution_scope_restricts_command_paths,
)
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .sandboxed_tool.sandbox_config import (
    SandboxConfig,
    get_sandbox_config,
    sandbox_config,
    set_instance_sandbox_config,
)

logger = logging.getLogger(__name__)


class CommandExecutorArgs(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout: Optional[int] = Field(
        default=None, description="Execution timeout in seconds (default: 300)"
    )


class CommandExecutorResult(BaseModel):
    success: bool = Field(description="Whether the command executed successfully")
    output: str = Field(description="Standard output from the command")
    error: str = Field(default="", description="Standard error from the command")
    return_code: int = Field(description="Process exit code")


class CommandExecutorTool(AbstractBaseTool):
    """Framework wrapper for the pure command executor tool"""

    def __init__(
        self,
        workspace: Optional[TaskWorkspace] = None,
        *,
        restrict_paths: bool = False,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace
        self._restrict_paths = restrict_paths

    @classmethod
    def from_execution_scope(
        cls,
        workspace: TaskWorkspace,
        execution_scope: ExecutionScope | None,
    ) -> "CommandExecutorTool":
        """Build a workspace tool with the scope-owned command path policy."""
        return cls(
            workspace=workspace,
            restrict_paths=execution_scope_restricts_command_paths(execution_scope),
        )

    @property
    def name(self) -> str:
        return "command_executor"

    @property
    def description(self) -> str:
        working_directory = self._get_working_directory()
        workspace_line = (
            f"Commands run with current working directory: {working_directory}."
            if working_directory
            else "Commands run in the current process working directory."
        )
        lines = [
            "Execute shell commands and scripts.",
            (
                "Supports shell commands including system commands, pipes, "
                "and redirects."
            ),
            workspace_line,
        ]
        if self._restrict_paths:
            lines.append(
                "Common shell file operations are checked against the current "
                "workspace; external allowed directories are read-only. This is "
                "a cooperative, best-effort check, not an operating-system "
                "security boundary; unknown commands are not classified. "
                "This initial policy recognizes only statically parsed Bash "
                "syntax and a documented set of command operands; runtime "
                "expansion, unsupported syntax, implicit shell inputs, and "
                "unknown commands may remain unclassified. Enumerate concrete "
                "paths and use recognized commands when path checks matter."
            )
        lines.extend(
            [
                (
                    "Use concrete paths, URLs, or file identifiers already "
                    "returned by previous tool results directly. If a tool "
                    "returned an absolute path or a path relative to the command "
                    "working directory, pass that path to the next command "
                    "instead of rediscovering it."
                ),
                (
                    "Only search for files when no usable path was provided, "
                    "and keep searches scoped to the command working directory "
                    "or another explicitly relevant directory. Do not run broad "
                    "recursive searches from `/` or the user's home directory "
                    "unless the user explicitly asks for that scope."
                ),
                (
                    "Examples: ls -la output, grep -r 'pattern' ./output, "
                    "cat file.txt | grep error"
                ),
            ]
        )
        return "\n".join(lines)

    @property
    def tags(self) -> list[str]:
        return ["shell", "command", "bash", "script", "terminal"]

    def args_type(self) -> Type[BaseModel]:
        return CommandExecutorArgs

    def return_type(self) -> Type[BaseModel]:
        return CommandExecutorResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        exec_args = CommandExecutorArgs.model_validate(args)

        executor = self._create_executor()

        # Execute command
        result = executor.execute_command(exec_args.command, timeout=exec_args.timeout)

        return CommandExecutorResult(**result).model_dump()

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self.run_json_sync, args)

    def _create_executor(self) -> CommandExecutorCore:
        if self._workspace is not None:
            return CommandExecutorCore.for_workspace(
                self._workspace,
                restrict_paths=self._restrict_paths,
            )
        return CommandExecutorCore()

    def _get_working_directory(self) -> Optional[str]:
        """Determine the working directory based on workspace settings"""
        if self._workspace:
            # Use workspace output directory as working directory
            return str(self._workspace.resolve_path(""))
        return None


@sandbox_config()
class CommandExecutorToolForBasic(CommandExecutorTool):
    """Command executor tool with BASIC category."""

    category = ToolCategory.BASIC

    def __init__(
        self,
        workspace: Optional[TaskWorkspace] = None,
        *,
        restrict_paths: bool = False,
    ) -> None:
        super().__init__(workspace=workspace, restrict_paths=restrict_paths)
        base_config = get_sandbox_config(self) or SandboxConfig()
        packages = base_config.packages
        if restrict_paths and "bashlex>=0.18" not in packages:
            packages = (*packages, "bashlex>=0.18")
        set_instance_sandbox_config(
            self,
            replace(base_config, packages=packages),
        )

    @property
    def name(self) -> str:
        return "execute_command"
