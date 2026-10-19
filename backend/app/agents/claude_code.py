"""Claude Code CLI (``claude``) headless adapter — uses the local Claude subscription.

Verified with ``claude`` 2.1.x print mode (``claude -p … --output-format json`` prints
ONE result object). A bare call loads the user's whole setup (CLAUDE.md, MCP servers,
plugins, hooks): measured ~283k context tokens for a one-line translation, and the
extra instructions even derailed it. ``_LEAN`` cuts that to ~4k; auth still comes
from the logged-in account.

Two shapes share one argv builder:
  - one-shot: no tools, no saved session, a minimal system prompt;
  - workspace: file tools only (no Bash), edits auto-accepted inside the cwd, and
    ``--continue`` resumes the cwd's last session for follow-up turns.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

from .config import AgentConfig, AgentError

_LEAN = ["--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands"]
_ONESHOT_SYSTEM = "Làm đúng yêu cầu trong tin nhắn và chỉ trả về đúng nội dung được yêu cầu."
_WORKSPACE_TOOLS = "Read,Write,Edit,Glob,Grep"


def build_claude_argv(prompt: str, config: AgentConfig, *, workspace: bool = False, resume: bool = False) -> list[str]:
    argv = [config.bin, "-p", prompt, "--output-format", "json", *_LEAN]
    if workspace:
        argv += ["--permission-mode", "acceptEdits", "--tools", _WORKSPACE_TOOLS]
        if resume:
            argv += ["--continue"]
    else:
        argv += ["--tools", "", "--no-session-persistence", "--system-prompt", _ONESHOT_SYSTEM]
    if config.model:
        argv += ["--model", config.model]
    if config.effort:
        argv += ["--effort", config.effort]
    argv += [str(arg) for arg in config.extra_args]
    return argv


def parse_claude_output(stdout: str) -> str:
    """``result`` of the last ``{"type": "result"}`` object on stdout."""
    for raw in reversed(stdout.splitlines()):
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("type") != "result":
            continue
        if data.get("is_error") or data.get("subtype") != "success":
            raise AgentError(str(data.get("result") or f"claude {data.get('subtype') or 'error'}"))
        return str(data.get("result") or "")
    raise AgentError("claude returned no result object")


def run_claude(prompt: str, config: AgentConfig) -> str:
    env = os.environ.copy()
    if config.home:
        env["HOME"] = os.path.expanduser(config.home)
    argv = build_claude_argv(prompt, config)
    # an empty cwd: a one-shot call must not pick up the repo's CLAUDE.md / settings
    with tempfile.TemporaryDirectory(prefix="autoreup-claude-") as cwd:
        try:
            proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=config.timeout, env=env)
        except FileNotFoundError as exc:
            raise AgentError(f"agent binary not found: {config.bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"agent timed out after {config.timeout}s") from exc
    if proc.returncode != 0 and not proc.stdout.strip():
        detail = (proc.stderr or "").strip().splitlines()
        raise AgentError(f"agent exited {proc.returncode}: {detail[-1] if detail else 'no output'}")
    return parse_claude_output(proc.stdout)


__all__ = ["build_claude_argv", "parse_claude_output", "run_claude"]
