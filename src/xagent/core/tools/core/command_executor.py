"""
Command Line Executor Tool

Execute shell commands and scripts with proper controls.
"""

import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, cast

from ...execution_scope import ExecutionScope
from .command_policy import CommandPathViolation, CommandPolicyViolation


class _CommandPathGuard(Protocol):
    def validate(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None: ...

    def validate_argv(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None: ...


logger = logging.getLogger(__name__)

# Constants
# Maximum output size to prevent memory exhaustion (10 MB)
MAX_OUTPUT_SIZE = 10 * 1024 * 1024

# Timeout return code constant
TIMEOUT_EXIT_CODE = -999

# Conventional shell exit code for a command found but not permitted to run.
COMMAND_REJECTED_EXIT_CODE = 126


def execution_scope_restricts_command_paths(
    execution_scope: ExecutionScope | None,
) -> bool:
    """Return the command policy bit owned by an execution scope."""
    return bool(execution_scope is not None and execution_scope.restrict_command_paths)


def _command_rejected_result(reason: object) -> Dict[str, Any]:
    if isinstance(reason, CommandPathViolation):
        logger.warning(
            "CommandExecutor: Rejected %s path: %s",
            reason.access,
            reason.path,
        )
        public_reason = f"path is outside allowed {reason.access} paths"
    else:
        logger.warning("CommandExecutor: Rejected command path: %s", reason)
        public_reason = str(reason)
    return {
        "success": False,
        "output": "",
        "error": f"Command rejected by workspace path policy: {public_reason}",
        "return_code": COMMAND_REJECTED_EXIT_CODE,
    }


def _resolve_policy_shell_executable() -> str:
    """Return the Bash executable whose grammar is owned by the policy parser."""
    executable = shutil.which("bash")
    if executable is None:
        raise CommandPolicyViolation(
            "restricted command execution requires a Bash executable"
        )
    return executable


def _restricted_shell_environment() -> dict[str, str]:
    """Return the inherited environment without implicit Bash code hooks."""
    environment = os.environ.copy()
    environment.pop("BASH_ENV", None)
    for name in tuple(environment):
        if name.startswith("BASH_FUNC_"):
            environment.pop(name)
    return environment


def _validate_timeout(timeout: Optional[int], default_timeout: int) -> int:
    """
    Validate and normalize timeout value.

    Args:
        timeout: Timeout in seconds
        default_timeout: Default timeout to use if timeout is None

    Returns:
        Validated timeout value

    Raises:
        ValueError: If timeout is invalid
    """
    if timeout is not None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got: {timeout}")
        return timeout
    return default_timeout


def _sanitize_command_for_logging(command: Any, max_length: int = 200) -> str:
    """
    Sanitize command for logging to avoid exposing sensitive data.

    Args:
        command: The command to sanitize (str or list)
        max_length: Maximum length of command to log

    Returns:
        Sanitized command string
    """
    # Convert list command to string for logging
    if isinstance(command, list):
        command_str = " ".join(str(x) for x in command)
    else:
        command_str = str(command)

    # Truncate long commands
    if len(command_str) > max_length:
        return command_str[:max_length] + "... [TRUNCATED]"

    # Redact potential sensitive patterns
    sensitive_patterns = [
        (
            r"(Bearer|Authorization|Token|API[_-]?KEY|PASSWORD|PASSWD|SECRET)[=\s][^\s]+",
            "REDACTED",
        ),
        (r"--password[=\s][^\s]+", "--password=REDACTED"),
        (r"-p\s+[^\s]+", "-p REDACTED"),
    ]

    for pattern, replacement in sensitive_patterns:
        command_str = re.sub(pattern, replacement, command_str, flags=re.IGNORECASE)

    return command_str


def _validate_working_directory(working_directory: Optional[str]) -> None:
    """
    Validate working directory before use.

    Args:
        working_directory: Directory path to validate

    Raises:
        FileNotFoundError: If directory doesn't exist
        NotADirectoryError: If path is not a directory
        PermissionError: If directory is not accessible
    """
    if not working_directory:
        return

    work_dir = Path(working_directory)

    if not work_dir.exists():
        raise FileNotFoundError(
            f"Working directory does not exist: {working_directory}"
        )

    if not work_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {working_directory}")

    if not os.access(working_directory, os.X_OK):
        raise PermissionError(
            f"No execute permission for directory: {working_directory}"
        )


def _sanitize_interpreter_suffix(interpreter: str) -> str:
    """
    Sanitize interpreter name for use as temp file suffix.

    Args:
        interpreter: Interpreter name (e.g., 'bash', 'python3.11')

    Returns:
        Sanitized interpreter name suitable for file suffix
    """
    # Take first part (before any space) and remove dots
    safe_name = interpreter.split()[0].replace(".", "").replace("-", "_")
    return safe_name if safe_name else "tmp"


class CommandExecutorCore:
    """Shell command executor with execution controls"""

    def __init__(
        self,
        working_directory: Optional[str] = None,
        path_guard: Optional[_CommandPathGuard] = None,
    ):
        """
        Initialize the command executor.

        Args:
            working_directory: Directory to use as working directory during execution
            path_guard: Optional cooperative workspace path guard
        """
        self.working_directory = working_directory
        self.path_guard = path_guard
        self.timeout = 300  # 5 minutes default

    @classmethod
    def for_workspace(
        cls,
        workspace: Any,
        *,
        restrict_paths: bool = False,
    ) -> "CommandExecutorCore":
        """Build the canonical executor for a workspace-bound command tool."""
        working_directory = str(workspace.resolve_path(""))
        if restrict_paths:
            # bashlex is an opt-in policy dependency. Unguarded command tools
            # must remain importable without paying its host/runtime cost.
            try:
                from .command_path_guard import WorkspaceCommandPathGuard
            except ModuleNotFoundError as exc:
                if exc.name != "bashlex":
                    raise
                raise RuntimeError(
                    "restricted command execution requires bashlex>=0.18; "
                    "install the command-policy optional dependency"
                ) from exc

            path_guard = WorkspaceCommandPathGuard(workspace)
        else:
            path_guard = None
        return cls(working_directory, path_guard=path_guard)

    def execute_command(
        self,
        command: str | list[str],
        timeout: Optional[int] = None,
        capture_output: bool = True,
        shell: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute shell command and return result.

        Args:
            command: Shell command text, or an argument vector when shell=False
            timeout: Execution timeout in seconds (default: 300)
            capture_output: Whether to capture stdout/stderr
            shell: Whether to use shell (allows pipes, redirects, etc.)

        Returns:
            Dictionary with success status, output, and error information

        Raises:
            ValueError: If timeout is invalid
            FileNotFoundError: If working directory doesn't exist
            NotADirectoryError: If working directory path is not a directory
            PermissionError: If working directory is not accessible
        """
        timeout = _validate_timeout(timeout, self.timeout)
        _validate_working_directory(self.working_directory)

        if shell and not isinstance(command, str):
            return _command_rejected_result("shell=True requires a string command")

        policy_environment = (
            _restricted_shell_environment() if self.path_guard is not None else None
        )
        if self.path_guard is not None:
            try:
                if shell:
                    self.path_guard.validate(
                        cast(str, command),
                        environment=policy_environment,
                    )
                else:
                    # The Vibe adapter currently sends shell strings, but argv
                    # validation intentionally protects direct and future
                    # non-shell CommandExecutorCore callers.
                    argv = [command] if isinstance(command, str) else list(command)
                    self.path_guard.validate_argv(
                        argv,
                        environment=policy_environment,
                    )
                    command = argv
            except CommandPolicyViolation as exc:
                return _command_rejected_result(exc)
            except Exception as exc:
                logger.error(
                    "CommandExecutor: Command path validation failed (%s)",
                    type(exc).__name__,
                    exc_info=True,
                )
                return _command_rejected_result("command validation failed")

        # Sanitize command for logging
        safe_command = _sanitize_command_for_logging(command)
        logger.info(f"CommandExecutor: Executing: {safe_command}")

        if self.working_directory:
            logger.info(
                f"CommandExecutor: Using working directory: {self.working_directory}"
            )

        run_options: dict[str, Any] = {
            "shell": shell,
            "capture_output": capture_output,
            "text": True,
            "timeout": timeout,
            "cwd": self.working_directory,
        }
        if self.path_guard is not None and shell:
            try:
                run_options["executable"] = _resolve_policy_shell_executable()
            except CommandPolicyViolation as exc:
                return _command_rejected_result(exc)
        if policy_environment is not None:
            run_options["env"] = policy_environment

        try:
            result = subprocess.run(
                command,
                **run_options,
            )

            output = result.stdout if capture_output else ""
            error = result.stderr if capture_output else ""

            # Truncate output if it exceeds maximum size
            if capture_output:
                if len(output) > MAX_OUTPUT_SIZE:
                    output = output[:MAX_OUTPUT_SIZE] + "\n[OUTPUT TRUNCATED]"
                if len(error) > MAX_OUTPUT_SIZE:
                    error = error[:MAX_OUTPUT_SIZE] + "\n[ERROR TRUNCATED]"

            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            logger.warning(
                f"CommandExecutor: Command timed out after {timeout} seconds"
            )
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "return_code": TIMEOUT_EXIT_CODE,
            }
        except Exception as e:
            logger.error(f"CommandExecutor: Execution error: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "return_code": TIMEOUT_EXIT_CODE,
            }

    def execute_script(
        self,
        script_content: str,
        interpreter: str = "bash",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute script content with specified interpreter.

        Args:
            script_content: Script content to execute
            interpreter: Interpreter to use (bash, python, node, sh, etc.)
            timeout: Execution timeout in seconds

        Returns:
            Dictionary with execution result

        Raises:
            ValueError: If timeout is invalid

        Notes:
            A guarded executor accepts only the Bash interpreter and validates
            the supplied content before running it through ``bash -c``.
        """
        timeout = _validate_timeout(timeout, self.timeout)

        if self.path_guard is not None:
            interpreter_argv = shlex.split(interpreter)
            if not interpreter_argv:
                return _command_rejected_result("script interpreter is required")
            if os.path.basename(interpreter_argv[0]) != "bash":
                return _command_rejected_result(
                    "restricted execute_script supports the Bash policy shell only"
                )
            return self.execute_command(
                [*interpreter_argv, "-c", script_content],
                timeout=timeout,
                shell=False,
            )

        try:
            logger.info(
                f"CommandExecutor: Executing script with interpreter: {interpreter}"
            )

            # Sanitize interpreter for temp file suffix
            safe_suffix = _sanitize_interpreter_suffix(interpreter)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{safe_suffix}", delete=False
            ) as f:
                f.write(script_content)
                script_path = f.name

            try:
                os.chmod(script_path, 0o755)
                command = f"{interpreter} {script_path}"
                return self.execute_command(command, timeout=timeout)
            finally:
                try:
                    os.unlink(script_path)
                except OSError:
                    pass  # Temp file cleanup failed, but command already ran

        except Exception as e:
            logger.error(f"CommandExecutor: Script execution error: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": f"Script execution error: {str(e)}",
                "return_code": TIMEOUT_EXIT_CODE,
            }


# Convenience functions for direct usage
def execute_command(
    command: str,
    working_directory: Optional[str] = None,
    timeout: Optional[int] = None,
    *,
    workspace: Any | None = None,
    execution_scope: ExecutionScope | None = None,
) -> Dict[str, Any]:
    """
    Execute a shell command.

    Args:
        command: Shell command to execute
        working_directory: Directory to use as working directory
        timeout: Execution timeout in seconds
        workspace: Optional workspace that owns command path authorization
        execution_scope: Optional typed scope that enables the path policy

    Returns:
        Dictionary with execution result
    """
    executor = _build_executor(working_directory, workspace, execution_scope)
    return executor.execute_command(command, timeout=timeout)


def execute_script(
    script_content: str,
    interpreter: str = "bash",
    working_directory: Optional[str] = None,
    timeout: Optional[int] = None,
    *,
    workspace: Any | None = None,
    execution_scope: ExecutionScope | None = None,
) -> Dict[str, Any]:
    """
    Execute script content.

    Args:
        script_content: Script content to execute
        interpreter: Interpreter to use (bash, python, node, etc.)
        working_directory: Directory to use as working directory
        timeout: Execution timeout in seconds
        workspace: Optional workspace that owns command path authorization
        execution_scope: Optional typed scope that enables the path policy

    Returns:
        Dictionary with execution result
    """
    executor = _build_executor(working_directory, workspace, execution_scope)
    return executor.execute_script(script_content, interpreter, timeout)


def _build_executor(
    working_directory: Optional[str],
    workspace: Any | None,
    execution_scope: ExecutionScope | None,
) -> CommandExecutorCore:
    """Build the shared executor used by module-level convenience functions."""
    return (
        CommandExecutorCore.for_workspace(
            workspace,
            restrict_paths=execution_scope_restricts_command_paths(execution_scope),
        )
        if workspace is not None
        else CommandExecutorCore(working_directory)
    )
