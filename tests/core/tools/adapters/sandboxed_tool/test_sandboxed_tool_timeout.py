"""Tests for the per-tool exec timeout applied inside the sandbox."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.core.tools.adapters.vibe.command_executor import CommandExecutorToolForBasic
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorToolForBasic
from xagent.core.tools.adapters.vibe.sandboxed_tool import sandboxed_tool_wrapper
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)

_ENSURE_REQUIREMENTS = (
    "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper."
    "SandboxDependencyManager.ensure_requirements"
)


@dataclass
class FakeExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class RecordingSandbox:
    """Records exec calls and answers the tool command with a fixed exit code."""

    def __init__(self, tool_exit_code: int = 0, tool_delay: float = 0) -> None:
        self.name = "recording"
        self.tool_exit_code = tool_exit_code
        self.tool_delay = tool_delay
        self.commands: list[list[str]] = []

    async def exec(self, command: str, *args: str, env=None, **_: object):
        self.commands.append([command, *args])
        if command == "cat":
            return FakeExecResult(stdout='{"ok": true}')
        if command == "rm":
            return FakeExecResult()
        await asyncio.sleep(self.tool_delay)
        return FakeExecResult(exit_code=self.tool_exit_code)


@sandbox_config(exec_timeout=lambda: 30)
class BoundedTool(FakeBaseTool):
    @property
    def name(self) -> str:
        return "bounded_tool"


@sandbox_config(exec_timeout=lambda: 1)
class ShortBoundedTool(FakeBaseTool):
    @property
    def name(self) -> str:
        return "short_bounded_tool"


async def _run(wrapper: SandboxedToolWrapper, code: str = "print('ok')"):
    with patch(_ENSURE_REQUIREMENTS, new=AsyncMock()):
        return await wrapper.run_json_async({"code": code})


@pytest.mark.asyncio
async def test_python_executor_runs_under_in_sandbox_timeout(monkeypatch):
    monkeypatch.setenv("XAGENT_PYTHON_EXECUTOR_TIMEOUT_SECONDS", "45")
    sandbox = RecordingSandbox()
    wrapper = SandboxedToolWrapper(PythonExecutorToolForBasic(None), sandbox)

    await _run(wrapper)

    tool_command = sandbox.commands[0]
    assert tool_command[:4] == ["timeout", "-s", "KILL", "45"]
    assert tool_command[4] == "python"


@pytest.mark.asyncio
async def test_tools_without_exec_timeout_stay_unbounded():
    sandbox = RecordingSandbox()
    wrapper = SandboxedToolWrapper(CommandExecutorToolForBasic(None), sandbox)

    await _run(wrapper, code="")

    assert sandbox.commands[0][0] == "python"


# `timeout -s KILL` also kills itself with the process group, so the exec
# reports a SIGKILL death: 137 from docker, -9 from boxlite.
@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [137, -9])
async def test_sigkill_after_limit_is_reported_as_timeout(exit_code):
    sandbox = RecordingSandbox(tool_exit_code=exit_code, tool_delay=1.1)
    wrapper = SandboxedToolWrapper(ShortBoundedTool(), sandbox)

    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        await _run(wrapper)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [137, -9])
async def test_sigkill_before_limit_is_not_reported_as_timeout(exit_code):
    wrapper = SandboxedToolWrapper(
        BoundedTool(), RecordingSandbox(tool_exit_code=exit_code)
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run(wrapper)

    assert "timed out" not in str(exc_info.value)


class LocalProcessSandbox:
    """Runs sandbox commands as local processes against this checkout."""

    name = "local"

    async def exec(self, command: str, *args: str, env=None, **_: object):
        if command == "timeout":
            argv = [command, *args]
            argv[argv.index("python")] = sys.executable
        elif command == "python":
            argv = [sys.executable, *args]
        else:
            argv = [command, *args]
        process_env = {**os.environ, **(env or {})}
        process_env["PYTHONPATH"] = str(
            Path(sandboxed_tool_wrapper.__file__).parents[6]
        )
        process = await asyncio.create_subprocess_exec(
            *argv,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return FakeExecResult(
            exit_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("timeout") is None, reason="needs coreutils timeout")
# A surviving grandchild keeps the exec's output open, so a regression hangs
# instead of failing; fail it well inside the CI job.
@pytest.mark.timeout(90)
@pytest.mark.parametrize("ignore_sigterm", [False, True])
async def test_timeout_kills_runner_and_its_children(
    monkeypatch, tmp_path, ignore_sigterm
):
    # Leaves room for the runner's cold xagent and pandas imports on CI.
    monkeypatch.setenv("XAGENT_PYTHON_EXECUTOR_TIMEOUT_SECONDS", "15")
    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "_TOOL_RUNNER_PATH",
        str(Path(sandboxed_tool_wrapper.__file__).with_name("tool_runner.py")),
    )
    pid_file = tmp_path / "child.pid"
    # Written via rename so a kill mid-write never leaves an empty pid file.
    grandchild = (
        "import os, signal, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_sigterm else "")
        + f"open({str(pid_file) + '.tmp'!r}, 'w').write(str(os.getpid())); "
        f"os.replace({str(pid_file) + '.tmp'!r}, {str(pid_file)!r}); "
        "time.sleep(600)"
    )
    code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    wrapper = SandboxedToolWrapper(
        PythonExecutorToolForBasic(None), LocalProcessSandbox()
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out after 15 seconds"):
        await _run(wrapper, code=code)

    assert time.monotonic() - started < 40
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _process_alive(child_pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert not _process_alive(child_pid)


def test_tool_runner_returns_output_with_undecodable_filenames(tmp_path):
    wrapper = SandboxedToolWrapper(PythonExecutorToolForBasic(None), RecordingSandbox())
    result_file = tmp_path / "result.json"
    command = wrapper._build_execution_command(
        {"code": "print(b'\\xff'.decode('utf-8', 'surrogateescape'))"},
        str(result_file),
    )
    command[0] = sys.executable
    command[1] = str(Path(sandboxed_tool_wrapper.__file__).with_name("tool_runner.py"))
    env = {
        **os.environ,
        **wrapper._build_execution_env(),
        "PYTHONPATH": str(Path(sandboxed_tool_wrapper.__file__).parents[6]),
    }

    completed = subprocess.run(command, env=env, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(result_file.read_text())["output"] == "\udcff\n"
