"""Agent runtime — the single text-generation seam.

Every text step (glossary extraction, translation, repair, naming) goes through here:
a headless agent CLI — Antigravity (``agy``), Claude Code (``claude``) or Codex
(``codex``) natively, or any ACP/headless command via a configurable template — or an
LLM API connection (``openai_compat`` / ``gemini``).

The deterministic pipeline still owns timing, segmentation and QA; the agent only
receives a bounded prompt and returns text.
"""

from __future__ import annotations

import time

from . import api_llm
from .agy import run_agy
from .claude_code import run_claude
from .codex import run_codex
from .command import run_command
from .config import AgentConfig, AgentError

_MAX_RETRY_WAIT = 30.0  # cap on a server-requested Retry-After between attempts


def compose_prompt(system: str, user: str) -> str:
    system = system.strip()
    if not system:
        return user
    return f"{system}\n\n---\n\n{user}"


def is_api(config: AgentConfig) -> bool:
    """True when the config talks to an LLM API rather than spawning an agent CLI."""
    return (config.provider or "").strip().lower() in api_llm.API_PROVIDERS


def _call(system: str, user: str, config: AgentConfig, images: list[bytes] | None = None) -> str:
    # APIs keep system and user apart; CLIs take one composed prompt.
    if is_api(config):
        return api_llm.chat(system, user, config, images=images)
    if images:
        raise AgentError(f"provider {config.provider!r} không nhận ảnh — cần kết nối openai_compat/gemini")
    return run_once(compose_prompt(system, user), config)


def run_once(prompt: str, config: AgentConfig) -> str:
    if is_api(config):
        return api_llm.chat("", prompt, config)
    provider = (config.provider or "agy").strip().lower()
    if provider in {"agy", "antigravity"}:
        return run_agy(prompt, config)
    if provider == "claude_code":
        return run_claude(prompt, config)
    if provider == "codex":
        return run_codex(prompt, config)
    if provider in {"command", "cli", "acp"}:
        if not config.command.strip():
            raise AgentError("provider=command requires agent_command (template with {prompt})")
        return run_command(prompt, config)
    raise AgentError(f"unknown agent provider: {config.provider!r}")


def complete(system: str, user: str, config: AgentConfig, images: list[bytes] | None = None) -> str:
    """Run one agent call with retries; raise AgentError after the last attempt.

    ``images`` (vision steps) are only accepted by LLM API connections."""
    last_error: Exception | None = None
    for attempt in range(max(1, config.retries)):
        try:
            return _call(system, user, config, images)
        except Exception as exc:  # noqa: BLE001 - spawn errors vary by adapter
            last_error = exc
            if isinstance(exc, AgentError) and not exc.retryable:
                raise AgentError(f"agent call failed: {exc}", retryable=False) from exc
            if attempt + 1 < config.retries:
                wait = getattr(exc, "retry_after", 0.0) or 0.0
                time.sleep(max(0.5 * (attempt + 1), min(wait, _MAX_RETRY_WAIT)))
    raise AgentError(f"agent call failed after {config.retries} attempts: {last_error}")


__all__ = ["AgentConfig", "AgentError", "complete", "compose_prompt", "is_api", "run_once"]
