"""Agent runtime primitives: config + errors shared by all adapters.

Kept dependency-free so adapters (agy, command) can import without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


class AgentError(RuntimeError):
    """Agent invocation failed (spawn, timeout, non-success status).

    ``retryable=False`` marks failures a retry cannot fix (bad key, unknown model, bad
    request); ``retry_after`` is the server's requested wait (HTTP 429) in seconds.
    """

    def __init__(self, message: str, *, retryable: bool = True, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass(slots=True)
class AgentConfig:
    """One headless agent call configuration.

    ``provider``:
      - ``"agy"``      — Antigravity CLI headless (``agy -p … --output-format json``)
      - ``"claude_code"`` — Claude Code CLI headless (``claude -p …``)
      - ``"codex"``    — Codex CLI headless (``codex exec …``)
      - ``"command"``  — any other headless CLI template (ACP clients, …)
      - ``"openai_compat"`` / ``"gemini"`` — LLM API over HTTP (``base_url`` + ``api_key``)

    ``home`` overrides ``HOME`` for the child process (account isolation).
    """

    provider: str = "agy"
    model: str = ""
    effort: str = ""
    bin: str = "agy"
    home: str | None = None
    command: str = ""
    timeout: float = 300.0
    retries: int = 3
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    # LLM API providers (openai_compat, gemini): endpoint root + key.
    base_url: str = ""
    api_key: str = field(default="", repr=False)

    def with_model(self, model: str) -> "AgentConfig":
        return replace(self, model=model)


__all__ = ["AgentConfig", "AgentError"]
