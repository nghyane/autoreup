"""Codex CLI (``codex exec``) headless adapter — uses the local ChatGPT/Codex login.

``codex exec`` streams progress on stderr; the reply is read from the file given to
``--output-last-message`` instead of scraping stdout. The sandbox is set through
``-c sandbox_mode=…`` because ``codex exec resume`` has no ``--sandbox`` flag:
  - one-shot: read-only sandbox, nothing persisted (``--ephemeral``);
  - workspace: writes allowed inside the cwd, and ``resume --last`` picks up the
    cwd's most recent session for follow-up turns.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .config import AgentConfig, AgentError


def build_codex_argv(
    prompt: str,
    config: AgentConfig,
    *,
    last_message: Path | None = None,
    workspace: bool = False,
    resume: bool = False,
) -> list[str]:
    argv = [config.bin, "exec"]
    if resume:
        argv += ["resume", "--last"]
    sandbox = "workspace-write" if workspace else "read-only"
    argv += ["--skip-git-repo-check", "-c", f'sandbox_mode="{sandbox}"']
    if not workspace:
        argv += ["--ephemeral"]
    if last_message is not None:
        argv += ["--output-last-message", str(last_message)]
    if config.model:
        argv += ["--model", config.model]
    if config.effort:
        argv += ["-c", f'model_reasoning_effort="{config.effort}"']
    argv += [str(arg) for arg in config.extra_args]
    # "--": a prompt starting with "-" is still the prompt, never a flag
    return [*argv, "--", prompt]


def codex_error(stderr: str, returncode: int) -> str:
    """The most telling stderr line: codex prefixes real failures with ``ERROR:``."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("ERROR:")]
    detail = (errors or lines or ["no output"])[-1]
    return f"agent exited {returncode}: {detail}"


def run_codex(prompt: str, config: AgentConfig) -> str:
    env = os.environ.copy()
    if config.home:
        env["HOME"] = os.path.expanduser(config.home)
    # an empty cwd: a one-shot call must not pick up the repo's AGENTS.md
    with tempfile.TemporaryDirectory(prefix="autoreup-codex-") as cwd:
        reply = Path(cwd) / "last-message.txt"
        argv = build_codex_argv(prompt, config, last_message=reply)
        try:
            proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=config.timeout, env=env)
        except FileNotFoundError as exc:
            raise AgentError(f"agent binary not found: {config.bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"agent timed out after {config.timeout}s") from exc
        if proc.returncode != 0:
            raise AgentError(codex_error(proc.stderr, proc.returncode))
        if not reply.exists():
            raise AgentError("codex returned no last message")
        return reply.read_text(encoding="utf-8").strip()


__all__ = ["build_codex_argv", "codex_error", "run_codex"]
