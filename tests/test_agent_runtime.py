import json

import pytest

from backend.app.agents import AgentConfig, AgentError, runtime
from backend.app.agents.agy import build_agy_argv, format_timeout, parse_agy_output
from backend.app.agents.claude_code import build_claude_argv, parse_claude_output
from backend.app.agents.codex import build_codex_argv, codex_error
from backend.app.agents.command import build_command_argv
from backend.app.agents.workspace import build_workspace_argv


def test_format_timeout_renders_go_durations():
    assert format_timeout(300) == "5m0s"
    assert format_timeout(90) == "1m30s"
    assert format_timeout(0.4) == "0m1s"


def test_build_agy_argv_carries_model_effort_and_timeout():
    config = AgentConfig(bin="agy", model="gemini-3.8-flash-low", effort="low", timeout=120)
    argv = build_agy_argv("dịch đoạn này", config)
    assert argv[0] == "agy"
    assert argv[1] == "-p" and argv[2] == "dịch đoạn này"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--print-timeout") + 1] == "2m0s"
    assert argv[argv.index("--model") + 1] == "gemini-3.8-flash-low"
    assert argv[argv.index("--effort") + 1] == "low"


def test_build_agy_argv_omits_effort_when_unset():
    argv = build_agy_argv("hi", AgentConfig(bin="agy", model="gemini-3.8-flash-low"))
    assert "--effort" not in argv
    assert "--model" in argv


def test_parse_agy_output_reads_last_envelope_and_reports_errors():
    ok = json.dumps({"status": "SUCCESS", "response": "xin chào"})
    assert parse_agy_output(f"scanner chatter\n{ok}") == "xin chào"

    err = json.dumps({"status": "ERROR", "error": "You have exhausted your quota on this model."})
    with pytest.raises(AgentError, match="quota"):
        parse_agy_output(err)


def test_parse_agy_output_requires_envelope():
    with pytest.raises(AgentError, match="no result envelope"):
        parse_agy_output("plain text only")


def test_build_command_argv_substitutes_or_appends_prompt():
    argv = build_command_argv("acpx claude exec --format quiet {prompt}", "hello")
    assert argv == ["acpx", "claude", "exec", "--format", "quiet", "hello"]
    assert build_command_argv("codex exec --json", "hi") == ["codex", "exec", "--json", "hi"]


def test_run_once_dispatches_providers(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "run_agy", lambda prompt, config: calls.append(("agy", prompt)) or "A")
    monkeypatch.setattr(runtime, "run_command", lambda prompt, config: calls.append(("cmd", prompt)) or "C")

    assert runtime.run_once("p1", AgentConfig(provider="agy")) == "A"
    assert runtime.run_once("p2", AgentConfig(provider="command", command="tool {prompt}")) == "C"
    assert calls == [("agy", "p1"), ("cmd", "p2")]

    with pytest.raises(AgentError, match="requires agent_command"):
        runtime.run_once("p3", AgentConfig(provider="command"))


def test_complete_retries_then_succeeds(monkeypatch):
    state = {"n": 0}

    def fake_run_once(prompt, config):
        state["n"] += 1
        if state["n"] < 3:
            raise AgentError("transient")
        return f"ok:{prompt[:3]}"

    monkeypatch.setattr(runtime, "run_once", fake_run_once)
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
    assert runtime.complete("SYS", "USER", AgentConfig(retries=3)) == "ok:SYS"
    assert state["n"] == 3


def test_complete_raises_after_last_attempt(monkeypatch):
    monkeypatch.setattr(runtime, "run_once", lambda prompt, config: (_ for _ in ()).throw(AgentError("nope")))
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
    with pytest.raises(AgentError, match="after 2 attempts"):
        runtime.complete("s", "u", AgentConfig(retries=2))


def test_compose_prompt_keeps_system_then_user():
    text = runtime.compose_prompt("SYS", "USER")
    assert text.startswith("SYS") and text.endswith("USER")
    assert runtime.compose_prompt("", "USER") == "USER"


def test_build_claude_argv_one_shot_is_lean_and_tool_free():
    argv = build_claude_argv("dịch", AgentConfig(provider="claude_code", bin="claude", model="sonnet", effort="high"))
    assert argv[:3] == ["claude", "-p", "dịch"]
    assert argv[argv.index("--output-format") + 1] == "json"
    # the user's CLAUDE.md / MCP / plugins never ride along (measured ~283k tokens otherwise)
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--effort") + 1] == "high"


def test_build_claude_argv_workspace_allows_file_tools_only_and_resumes():
    config = AgentConfig(provider="claude_code", bin="claude")
    argv = build_claude_argv("p", config, workspace=True, resume=True)
    assert argv[argv.index("--tools") + 1] == "Read,Write,Edit,Glob,Grep"
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert "--continue" in argv and "--no-session-persistence" not in argv
    assert "--continue" not in build_claude_argv("p", config, workspace=True)


def test_parse_claude_output_reads_result_and_reports_errors():
    ok = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "xin chào"})
    assert parse_claude_output(f"noise\n{ok}") == "xin chào"
    bad = json.dumps({"type": "result", "subtype": "success", "is_error": True, "result": "Credit balance is too low"})
    with pytest.raises(AgentError, match="Credit balance"):
        parse_claude_output(bad)
    with pytest.raises(AgentError, match="no result object"):
        parse_claude_output("plain text")


def test_build_codex_argv_one_shot_and_workspace_resume(tmp_path):
    config = AgentConfig(provider="codex", bin="codex", model="gpt-5.5", effort="medium")
    argv = build_codex_argv("-dịch", config, last_message=tmp_path / "out.txt")
    assert argv[:2] == ["codex", "exec"]
    assert 'sandbox_mode="read-only"' in argv and "--ephemeral" in argv
    assert argv[argv.index("--output-last-message") + 1] == str(tmp_path / "out.txt")
    assert argv[argv.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="medium"' in argv
    assert argv[-2:] == ["--", "-dịch"]  # a prompt that looks like a flag stays the prompt

    resumed = build_codex_argv("p", AgentConfig(provider="codex", bin="codex"), workspace=True, resume=True)
    assert resumed[:4] == ["codex", "exec", "resume", "--last"]
    assert 'sandbox_mode="workspace-write"' in resumed and "--ephemeral" not in resumed


def test_codex_error_prefers_error_lines():
    stderr = "2026 INFO starting\nERROR: stream disconnected\nsome trailing note"
    assert codex_error(stderr, 1) == "agent exited 1: ERROR: stream disconnected"
    assert codex_error("", 2) == "agent exited 2: no output"


def test_run_once_dispatches_claude_and_codex(monkeypatch):
    monkeypatch.setattr(runtime, "run_claude", lambda prompt, config: f"claude:{prompt}")
    monkeypatch.setattr(runtime, "run_codex", lambda prompt, config: f"codex:{prompt}")
    assert runtime.run_once("a", AgentConfig(provider="claude_code")) == "claude:a"
    assert runtime.run_once("b", AgentConfig(provider="codex")) == "codex:b"


def test_workspace_argv_routes_claude_and_codex(tmp_path):
    claude = build_workspace_argv("p", AgentConfig(provider="claude_code", bin="claude"), tmp_path, resume=True)
    assert claude[0] == "claude" and "--continue" in claude
    codex = build_workspace_argv("p", AgentConfig(provider="codex", bin="codex"), tmp_path)
    assert codex[:2] == ["codex", "exec"] and 'sandbox_mode="workspace-write"' in codex


def test_agy_quota_oracle_is_never_asked_about_other_clis(monkeypatch):
    from backend.app.dubbing import translate as translate_mod

    monkeypatch.setattr(translate_mod, "gemini_headroom", lambda binary: pytest.fail(f"asked agy quota for {binary}"))
    config = translate_mod.TranslateConfig(max_parallel=3)
    for provider in ("claude_code", "codex"):
        assert translate_mod._parallelism(config, AgentConfig(provider=provider, bin=provider)) == 3
