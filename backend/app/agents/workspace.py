"""Workspace agent runs — file-first, for content-heavy work.

Verified with ``agy`` 1.2.x headless: print mode has no default workspace (register
one with ``--add-dir``), file tools are soft-denied unless
``--dangerously-skip-permissions`` is passed (pair with ``--sandbox``), and the CLI
can loop internally across many turns. So a translation job is ONE agent run that
reads inputs, writes its draft, reviews it against the source and writes the final
file — while the deterministic pipeline validates the artifacts FROM DISK, never
from model prose. Claude Code and Codex run the same way (their adapters own the
flags that allow file edits inside the workspace and resume its last session).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from backend.app.infrastructure.process import kill_tree

from .agy import format_timeout
from .claude_code import build_claude_argv
from .codex import build_codex_argv
from .config import AgentConfig, AgentError

# Provider-specific flags that turn a one-shot CLI into a workspace agent.
_WORKSPACE_ARGS: dict[str, list[str]] = {
    "agy": ["--dangerously-skip-permissions", "--sandbox"],
}


@dataclass(slots=True)
class WorkspaceRun:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False


def build_workspace_argv(prompt: str, config: AgentConfig, workspace: Path, *, resume: bool = False) -> list[str]:
    provider = (config.provider or "agy").strip().lower()
    if provider in {"command", "cli", "acp"}:
        template = config.command.strip()
        if not template:
            raise AgentError("provider=command requires agent_command (template with {prompt})")
        return [part.replace("{prompt}", prompt) for part in template.split()]
    # claude/codex: the workspace is the cwd (see run_agent_workspace), no flag needed
    if provider == "claude_code":
        return build_claude_argv(prompt, config, workspace=True, resume=resume)
    if provider == "codex":
        return build_codex_argv(prompt, config, workspace=True, resume=resume)
    argv = [
        config.bin,
        "-p",
        prompt,
        "--add-dir",
        str(workspace),
        "--print-timeout",
        format_timeout(config.timeout),
    ]
    argv += _WORKSPACE_ARGS.get(provider, [])
    if config.model:
        argv += ["--model", config.model]
    if config.effort:
        argv += ["--effort", config.effort]
    if resume:
        argv += ["--continue"]
    argv += [str(arg) for arg in config.extra_args]
    return argv


def run_agent_workspace(
    workspace: Path,
    prompt: str,
    config: AgentConfig,
    *,
    timeout: float | None = None,
    resume: bool = False,
    cancel: threading.Event | None = None,
) -> WorkspaceRun:
    """Run one agent turn (or loop) inside ``workspace``; never raises on timeout.

    Setting ``cancel`` kills the agent (and anything it spawned) within a second and
    raises AgentError; whatever it already wrote under ``out/`` stays on disk."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    limit = float(timeout if timeout is not None else config.timeout)
    argv = build_workspace_argv(prompt, config, workspace, resume=resume)
    env = os.environ.copy()
    if config.home:
        env["HOME"] = os.path.expanduser(config.home)
    started = time.monotonic()
    deadline = started + limit + 30
    try:
        proc = subprocess.Popen(
            argv, cwd=str(workspace), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise AgentError(f"agent binary not found: {config.bin}") from exc
    while True:
        try:
            # retrying communicate() after its timeout loses no output
            stdout, stderr = proc.communicate(timeout=1.0)
            return WorkspaceRun(proc.returncode, stdout, stderr, round(time.monotonic() - started, 1))
        except subprocess.TimeoutExpired:
            canceled = cancel is not None and cancel.is_set()
            if not canceled and time.monotonic() < deadline:
                continue
            kill_tree(proc)
            stdout, _ = proc.communicate()
            if canceled:
                raise AgentError("canceled", retryable=False)
            return WorkspaceRun(-1, stdout or "", "workspace agent timed out", round(time.monotonic() - started, 1), timed_out=True)



__all__ = ["WorkspaceRun", "build_workspace_argv", "run_agent_workspace"]
