"""LLM API adapters — chat over HTTP for connections of type ``openai_compat`` / ``gemini``.

Same seam as the CLI adapters (text in, text out, ``AgentError`` on failure), plus
optional images for vision steps. ``openai_compat`` speaks ``/chat/completions`` (OpenAI,
OpenRouter, Ollama, vLLM, …); ``gemini`` speaks the native ``generateContent`` API.

The api_key travels only in a request header — never in the URL, never in an error.
"""

from __future__ import annotations

import base64

import httpx

from .config import AgentConfig, AgentError

GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"
API_PROVIDERS = frozenset({"openai_compat", "gemini"})


def _client(timeout: float) -> httpx.Client:
    """Factory seam: tests swap in an ``httpx.MockTransport``."""
    return httpx.Client(timeout=timeout)


def _provider(config: AgentConfig) -> str:
    provider = (config.provider or "").strip().lower()
    if provider not in API_PROVIDERS:
        raise AgentError(f"not an LLM API provider: {config.provider!r}")
    return provider


def _root(config: AgentConfig, provider: str) -> str:
    root = config.base_url.strip().rstrip("/")
    if provider == "gemini":
        root = root or GEMINI_ROOT
        if not _key(config):
            raise AgentError("gemini: thiếu api_key")
    elif not root:
        raise AgentError(f"{provider}: thiếu base_url")
    return root


def _key(config: AgentConfig) -> str:
    # pasted keys often carry a trailing newline/space, which httpx rejects (echoing it)
    return config.api_key.strip()


def _headers(config: AgentConfig, provider: str) -> dict[str, str]:
    key = _key(config)
    if not key:
        return {}
    if provider == "gemini":
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"}


def _redact(text: str, config: AgentConfig) -> str:
    """Errors end up in served job manifests and API responses: never let the key through."""
    for secret in {config.api_key, _key(config)}:
        if secret:
            text = text.replace(secret, "***")
    return text


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("retry-after", "0")))
    except ValueError:
        return 0.0


def _mime(image: bytes) -> str:
    if image.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _error_message(response: httpx.Response) -> str:
    """The provider's own error text (both APIs use ``{"error": {"message": …}}``)."""
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()[:300] or response.reason_phrase
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:300]
    if isinstance(error, str):
        return error[:300]
    return response.text.strip()[:300]


def _request(method: str, url: str, config: AgentConfig, provider: str, **kwargs) -> dict:
    try:
        with _client(config.timeout) as client:
            response = client.request(method, url, headers=_headers(config, provider), **kwargs)
    except (httpx.HTTPError, httpx.InvalidURL, UnicodeError) as exc:
        # transport errors may retry; a malformed URL/header never will
        retryable = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))
        raise AgentError(_redact(f"{provider}: {type(exc).__name__}: {exc}", config), retryable=retryable) from exc
    status = response.status_code
    if status >= 400:
        # 4xx is the request's fault (key, model, payload) except timeouts and rate limits
        retryable = status >= 500 or status in (408, 429)
        message = _redact(f"{provider} HTTP {status}: {_error_message(response)}", config)
        raise AgentError(message, retryable=retryable, retry_after=_retry_after(response))
    try:
        data = response.json()
    except ValueError as exc:
        raise AgentError(f"{provider}: phản hồi không phải JSON") from exc
    if not isinstance(data, dict):
        raise AgentError(f"{provider}: phản hồi JSON không hợp lệ")
    return data


def _openai_chat(system: str, user: str, config: AgentConfig, images: list[bytes]) -> str:
    content: str | list[dict] = user
    if images:
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:{_mime(img)};base64,{base64.b64encode(img).decode()}"}}
            for img in images
        ]
    messages = [{"role": "system", "content": system}] if system.strip() else []
    messages.append({"role": "user", "content": content})
    body: dict = {"model": config.model, "messages": messages}
    if config.effort:
        body["reasoning_effort"] = config.effort
    data = _request("POST", f"{_root(config, 'openai_compat')}/chat/completions", config, "openai_compat", json=body)
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentError("openai_compat: phản hồi thiếu choices[0].message.content") from exc
    if isinstance(reply, list):  # some servers return content parts
        reply = "".join(part.get("text", "") for part in reply if isinstance(part, dict))
    return str(reply or "")


def _gemini_chat(system: str, user: str, config: AgentConfig, images: list[bytes]) -> str:
    parts: list[dict] = [{"text": user}]
    parts += [{"inline_data": {"mime_type": _mime(img), "data": base64.b64encode(img).decode()}} for img in images]
    body: dict = {"contents": [{"role": "user", "parts": parts}]}
    if system.strip():
        body["systemInstruction"] = {"parts": [{"text": system}]}
    model = config.model.strip().removeprefix("models/")
    url = f"{_root(config, 'gemini')}/models/{model}:generateContent"
    data = _request("POST", url, config, "gemini", json=body)
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason") or "không có candidate"
        raise AgentError(f"gemini: không có câu trả lời ({reason})")
    reply_parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in reply_parts if not part.get("thought"))


def chat(system: str, user: str, config: AgentConfig, images: list[bytes] | None = None) -> str:
    """One chat turn; raise ``AgentError`` on HTTP/network failure or an empty reply."""
    provider = _provider(config)
    if not config.model.strip():
        raise AgentError(f"{provider}: chưa chọn model")
    if provider == "gemini":
        reply = _gemini_chat(system, user, config, list(images or []))
    else:
        reply = _openai_chat(system, user, config, list(images or []))
    if not reply.strip():
        raise AgentError(f"{provider}: câu trả lời rỗng")
    return reply


def list_models(config: AgentConfig) -> list[str]:
    """Model ids the connection offers (sorted) — also a cheap connectivity/auth probe."""
    provider = _provider(config)
    root = _root(config, provider)
    if provider == "gemini":
        data = _request("GET", f"{root}/models", config, provider, params={"pageSize": 1000})
        names = [
            str(m.get("name", "")).removeprefix("models/")
            for m in data.get("models") or []
            # embedding/imagen models can't chat; models without the field are kept
            if isinstance(m, dict) and "generateContent" in m.get("supportedGenerationMethods", ["generateContent"])
        ]
    else:
        data = _request("GET", f"{root}/models", config, provider)
        names = [str(m.get("id", "")) for m in data.get("data") or [] if isinstance(m, dict)]
    return sorted({name for name in names if name})


__all__ = ["API_PROVIDERS", "GEMINI_ROOT", "chat", "list_models"]
