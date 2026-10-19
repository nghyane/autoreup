"""LLM API adapters (openai_compat, gemini): wire format, parsing, errors, dispatch, translate."""

import base64
import json
from dataclasses import replace

import httpx
import pytest

import backend.app.agents.api_llm as api_llm
import backend.app.dubbing.translate as translate_mod
from backend.app.agents import AgentConfig, AgentError, complete, is_api, run_once
from backend.app.dubbing.models import Anchor
from backend.app.dubbing.translate import TranslateConfig, translate

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8

OPENAI = AgentConfig(provider="openai_compat", model="gpt-x", base_url="https://api.example/v1/", api_key="sk-secret-123456", retries=1)
GEMINI = AgentConfig(provider="gemini", model="gemini-x", api_key="AIza-secret-999", retries=1)


def _mock(monkeypatch, handler):
    """Route every adapter request through ``handler``; returns the captured requests."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(api_llm, "_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(record), timeout=timeout))
    return seen


def _openai_reply(text: str):
    return lambda request: httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})


def _gemini_reply(*texts: str):
    parts = [{"text": text} for text in texts]
    return lambda request: httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": parts}}]})


def test_is_api_matches_api_providers_only():
    assert is_api(OPENAI) and is_api(GEMINI)
    assert not is_api(AgentConfig(provider="agy")) and not is_api(AgentConfig(provider="command"))


def test_openai_chat_sends_system_user_effort_and_bearer(monkeypatch):
    seen = _mock(monkeypatch, _openai_reply("xin chào"))
    config = replace(OPENAI, effort="low")
    assert api_llm.chat("hệ thống", "câu hỏi", config) == "xin chào"
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-secret-123456"
    body = json.loads(request.content)
    assert body["model"] == "gpt-x"
    assert body["reasoning_effort"] == "low"
    assert body["messages"] == [{"role": "system", "content": "hệ thống"}, {"role": "user", "content": "câu hỏi"}]


def test_openai_chat_embeds_images_and_skips_empty_system(monkeypatch):
    seen = _mock(monkeypatch, _openai_reply("ok"))
    api_llm.chat("", "đọc chữ", OPENAI, images=[PNG, JPEG])
    body = json.loads(seen[0].content)
    assert "reasoning_effort" not in body
    assert len(body["messages"]) == 1
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "đọc chữ"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_without_key_sends_no_auth_header(monkeypatch):
    seen = _mock(monkeypatch, _openai_reply("ok"))
    api_llm.chat("", "hi", AgentConfig(provider="openai_compat", model="m", base_url="http://localhost:11434/v1"))
    assert "authorization" not in seen[0].headers


def test_gemini_chat_wire_format_and_joins_text_parts(monkeypatch):
    seen = _mock(monkeypatch, _gemini_reply("xin ", "chào"))
    assert api_llm.chat("hệ thống", "câu hỏi", GEMINI, images=[PNG]) == "xin chào"
    request = seen[0]
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
    assert request.headers["x-goog-api-key"] == "AIza-secret-999"
    assert "AIza-secret-999" not in str(request.url)
    body = json.loads(request.content)
    assert body["systemInstruction"] == {"parts": [{"text": "hệ thống"}]}
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "câu hỏi"}
    assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(PNG).decode()}}


def test_gemini_skips_thought_parts_and_reports_blocked_prompt(monkeypatch):
    _mock(monkeypatch, lambda request: httpx.Response(200, json={"candidates": [{"content": {"parts": [
        {"text": "nghĩ…", "thought": True}, {"text": "trả lời"}]}}]}))
    assert api_llm.chat("", "hi", GEMINI) == "trả lời"
    _mock(monkeypatch, lambda request: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(AgentError, match="SAFETY"):
        api_llm.chat("", "hi", GEMINI)


def test_http_error_raises_agent_error_with_status_and_message_but_no_key(monkeypatch):
    _mock(monkeypatch, lambda request: httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}}))
    with pytest.raises(AgentError) as info:
        api_llm.chat("", "hi", OPENAI)
    assert "401" in str(info.value) and "Incorrect API key" in str(info.value)
    assert "sk-secret-123456" not in str(info.value)


def test_network_error_and_missing_config_are_agent_errors(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("connection refused", request=request)

    _mock(monkeypatch, boom)
    with pytest.raises(AgentError, match="connection refused"):
        api_llm.chat("", "hi", OPENAI)
    with pytest.raises(AgentError, match="base_url"):
        api_llm.chat("", "hi", AgentConfig(provider="openai_compat", model="m"))
    with pytest.raises(AgentError, match="model"):
        api_llm.chat("", "hi", AgentConfig(provider="gemini", api_key="k"))
    with pytest.raises(AgentError, match="api_key"):
        api_llm.chat("", "hi", AgentConfig(provider="gemini", model="m"))


def test_empty_reply_is_an_error(monkeypatch):
    _mock(monkeypatch, _openai_reply(""))
    with pytest.raises(AgentError):
        api_llm.chat("", "hi", OPENAI)


def test_list_models_for_both_providers(monkeypatch):
    seen = _mock(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "b-model"}, {"id": "a-model"}]}))
    assert api_llm.list_models(OPENAI) == ["a-model", "b-model"]
    assert str(seen[0].url) == "https://api.example/v1/models"
    assert seen[0].method == "GET"

    seen = _mock(monkeypatch, lambda request: httpx.Response(200, json={"models": [
        {"name": "models/gemini-x"}, {"name": "models/gemini-a"}]}))
    assert api_llm.list_models(GEMINI) == ["gemini-a", "gemini-x"]
    assert seen[0].url.path == "/v1beta/models"
    assert seen[0].headers["x-goog-api-key"] == "AIza-secret-999"


def test_complete_dispatches_api_with_system_and_user_apart(monkeypatch):
    calls = []

    def fake_chat(system, user, config, images=None):
        calls.append((system, user, config.provider))
        return "kết quả"

    monkeypatch.setattr(api_llm, "chat", fake_chat)
    assert complete("hệ thống", "người dùng", GEMINI) == "kết quả"
    assert calls == [("hệ thống", "người dùng", "gemini")]
    assert run_once("một lời nhắc", OPENAI) == "kết quả"
    assert calls[-1] == ("", "một lời nhắc", "openai_compat")


def test_complete_retries_api_errors_then_raises(monkeypatch):
    attempts = []

    def failing(system, user, config, images=None):
        attempts.append(1)
        raise AgentError("HTTP 500")

    monkeypatch.setattr(api_llm, "chat", failing)
    monkeypatch.setattr("backend.app.agents.runtime.time.sleep", lambda seconds: None)
    with pytest.raises(AgentError, match="HTTP 500"):
        complete("", "hi", replace(OPENAI, retries=2))
    assert len(attempts) == 2


def _anchors(n: int) -> list[Anchor]:
    return [Anchor(id=i, start=i * 2.0, end=i * 2.0 + 1.5, source_zh=f"句子{i}", source_ids=str(i)) for i in range(n)]


def test_translate_over_api_inlines_workspace_and_fills_missing_ids(monkeypatch, tmp_path):
    prompts = []

    def fake_chat(system, user, config, images=None):
        prompts.append((system, user))
        if len(prompts) == 1:  # first answer forgets id 2
            return "Đây là bản dịch:\n```tsv\n0\tLời 0\n1\tLời 1\n```"
        return "```\n2\tLời 2\n```"

    monkeypatch.setattr(api_llm, "chat", fake_chat)
    monkeypatch.setattr(translate_mod, "run_agent_workspace", _never)
    monkeypatch.setattr(translate_mod, "gemini_headroom", _never)
    workspace = tmp_path / "translate"
    result = translate(_anchors(3), [], OPENAI, workspace, TranslateConfig(retries=1))

    assert [row.text for row in result.rows] == ["Lời 0", "Lời 1", "Lời 2"]
    assert len(prompts) == 2 and len(result.runs) == 2
    assert all(run.returncode == 0 for run in result.runs)
    system, user = prompts[0]
    assert "Skill: vietsub thuyết minh" in system      # SKILL.md is the instruction
    assert "*0\t0.00\t1.50\t句子0" in user               # inputs travel inline
    assert "2" in prompts[1][1] and "thiếu" in prompts[1][1]
    final = (workspace / "out" / "final.tsv").read_text(encoding="utf-8")
    assert final.splitlines() == ["0\tLời 0", "1\tLời 1", "2\tLời 2"]
    assert (workspace / "out" / "draft.tsv").exists()


def test_translate_over_api_raises_when_ids_stay_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api_llm, "chat", lambda system, user, config, images=None: "0\tLời 0")
    with pytest.raises(translate_mod.TranslateError):
        translate(_anchors(2), [], OPENAI, tmp_path / "translate", TranslateConfig(retries=1, min_part_units=5))


def test_translate_over_api_skips_agy_quota_oracle(monkeypatch, tmp_path):
    monkeypatch.setattr(api_llm, "chat", lambda system, user, config, images=None: "\n".join(
        f"{i}\tLời {i}" for i in range(4)))
    monkeypatch.setattr(translate_mod, "gemini_headroom", _never)
    rows = translate(_anchors(4), [], GEMINI, tmp_path / "translate", TranslateConfig(part_units=2)).rows
    assert [row.text for row in rows] == [f"Lời {i}" for i in range(4)]


def _never(*args, **kwargs):
    raise AssertionError("API translation must not touch the agy workspace runner or quota oracle")


def test_pasted_key_whitespace_is_stripped_and_never_echoed(monkeypatch):
    seen = _mock(monkeypatch, _openai_reply("ok"))
    api_llm.chat("", "hi", replace(OPENAI, api_key="sk-secret-123456\n"))
    assert seen[0].headers["authorization"] == "Bearer sk-secret-123456"
    # h11 rejects a malformed header by echoing its value: that text must not leak
    def refuse(request):
        raise httpx.LocalProtocolError(f"Illegal header value {request.headers['authorization']!r}")

    _mock(monkeypatch, refuse)
    with pytest.raises(AgentError) as info:
        api_llm.chat("", "hi", replace(OPENAI, api_key=" sk-secret-123456\n"))
    assert "sk-secret-123456" not in str(info.value) and "***" in str(info.value)
    assert not info.value.retryable


def test_client_errors_are_not_retried_but_rate_limits_wait(monkeypatch):
    sleeps = []
    monkeypatch.setattr("backend.app.agents.runtime.time.sleep", sleeps.append)
    seen = _mock(monkeypatch, lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(AgentError, match="bad key"):
        complete("", "hi", replace(OPENAI, retries=3))
    assert len(seen) == 1 and sleeps == []

    replies = iter([httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}),
                    httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    seen = _mock(monkeypatch, lambda request: next(replies))
    assert complete("", "hi", replace(OPENAI, retries=3)) == "ok"
    assert len(seen) == 2 and sleeps == [7.0]


def test_gemini_models_keep_chat_capable_only_and_model_prefix_is_tolerated(monkeypatch):
    _mock(monkeypatch, lambda request: httpx.Response(200, json={"models": [
        {"name": "models/gemini-x", "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/text-embedding", "supportedGenerationMethods": ["embedContent"]}]}))
    assert api_llm.list_models(GEMINI) == ["gemini-x"]
    seen = _mock(monkeypatch, _gemini_reply("ok"))
    api_llm.chat("", "hi", replace(GEMINI, model="models/gemini-x"))
    assert seen[0].url.path == "/v1beta/models/gemini-x:generateContent"


def test_translate_api_error_bisects_like_a_timed_out_session(monkeypatch, tmp_path):
    def fake_chat(system, user, config, images=None):
        marked = [line for line in user.splitlines() if line.startswith("*")]
        if len(marked) > 2:
            raise AgentError("openai_compat HTTP 400: maximum context length exceeded", retryable=False)
        return "\n".join(f"{line[1:].split(chr(9))[0]}\tLời" for line in marked)

    monkeypatch.setattr(api_llm, "chat", fake_chat)
    result = translate(_anchors(4), [], OPENAI, tmp_path / "translate", TranslateConfig(min_part_units=1))
    assert [row.text for row in result.rows] == ["Lời"] * 4
    assert result.runs[0].returncode == -1 and "context length" in result.runs[0].stderr

    monkeypatch.setattr(api_llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AgentError("HTTP 401: bad key", retryable=False)))
    with pytest.raises(translate_mod.TranslateError, match="bad key"):
        translate(_anchors(2), [], OPENAI, tmp_path / "again", TranslateConfig(min_part_units=1, max_split_depth=1))
