from pathlib import Path

import backend.app.dubbing.translate as translate_mod
from backend.app.agents import AgentConfig
from backend.app.agents.workspace import WorkspaceRun
from backend.app.core.config import Settings
from backend.app.domain.jobs import JobInput, JobState, StageName
from backend.app.dubbing.models import Anchor
from backend.app.dubbing.translate import TranslateConfig, TranslateError, translate
from backend.app.pipeline.contracts import PipelineContext, PipelineStage, StageResult
from backend.app.services.job_service import JobService

QUOTA = 'AGY_ERROR: {"short_error":"RESOURCE_EXHAUSTED (code 429): Individual quota reached."}'


def _context(published: list) -> PipelineContext:
    context = PipelineContext(job_id="j", input_url="", job_dir=None, settings=None, config=None)
    context.on_progress = lambda live, urgent: published.append((live, urgent))
    return context


def test_a_new_step_drops_the_previous_counts_but_keeps_the_log_tail():
    published: list = []
    context = _context(published)
    context.report({"step": "Whisper"})
    context.report({"done": 3, "total": 9, "unit": "đoạn"})
    for index in range(7):
        context.log(f"line {index}")
    context.report({"step": "Chuẩn hoá"})

    assert [urgent for _live, urgent in published[:2]] == [True, False]  # counters are throttled
    live, urgent = published[-1]
    assert urgent
    assert live == {"step": "Chuẩn hoá", "logs": [f"line {index}" for index in range(2, 7)]}


def test_a_running_stage_shows_its_step_and_log_on_the_manifest_at_once(tmp_path):
    settings = Settings(root_dir=tmp_path, _env_file=None)
    settings.ensure_directories()
    service = JobService(settings)
    job_id = service.create(JobInput(url="https://example.com/v")).job_id
    seen: list = []

    class Asr(PipelineStage):
        name = StageName.asr

        def run(self, context):
            seen.append(service.get(job_id).progress)  # before the stage reports anything
            context.report({"done": 1, "total": 4, "unit": "đoạn"})
            context.report({"done": 2, "total": 4, "unit": "đoạn"})  # throttled: not saved yet
            context.report({"step": "Whisper"})
            context.log("asr-plan: full")
            seen.append(service.get(job_id).progress)
            return StageResult(name=self.name)

    service.pipeline.stages = [Asr()]
    service.run(job_id)

    assert seen[0]["stage"] == "asr" and seen[0]["started_at"]
    assert seen[1]["step"] == "Whisper"
    assert seen[1]["logs"] == ["asr-plan: full"]
    assert "done" not in seen[1]  # a new step starts without the previous step's counts
    assert service.get(job_id).progress is None


def _units(count: int) -> list[Anchor]:
    return [Anchor(i, float(i), i + 0.9, f"{i}-{i}", f"字{i}") for i in range(count)]


def test_an_exhausted_quota_stops_the_stage_instead_of_bisecting(monkeypatch, tmp_path):
    calls: list = []

    def exhausted(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        calls.append(workspace)
        return WorkspaceRun(1, "", QUOTA, 2.0)

    monkeypatch.setattr(translate_mod, "run_agent_workspace", exhausted)
    logs: list = []
    config = TranslateConfig(part_units=40, min_part_units=5)
    try:
        translate(_units(40), [], AgentConfig(), tmp_path / "t", config, log=logs.append)
    except TranslateError as exc:
        assert "quota" in str(exc)
    else:
        raise AssertionError("expected TranslateError")
    assert len(calls) == 1  # no follow-up turn, no bisection
    assert logs[-1].startswith("phần 1: 0/40 dòng")


def test_parallel_parts_report_units_done_and_skip_a_part_already_final(monkeypatch, tmp_path):
    units = _units(20)
    calls: list = []

    def writer(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        workspace = Path(workspace)
        calls.append(workspace.name)
        lines = (workspace / "input" / "units.tsv").read_text().splitlines()
        marked = [line.split("\t")[0].lstrip("*") for line in lines if line.startswith("*")]
        body = "".join(f"{i}\tLời {i}\n" for i in marked)
        (workspace / "out" / "draft.tsv").write_text(body)
        (workspace / "out" / "final.tsv").write_text(body)
        return WorkspaceRun(0, "", "", 1.0)

    monkeypatch.setattr(translate_mod, "run_agent_workspace", writer)
    monkeypatch.setattr(translate_mod, "gemini_headroom", lambda binary: None)
    config = TranslateConfig(part_units=10)
    reports: list = []
    translate(units, [], AgentConfig(), tmp_path / "t", config, report=reports.append)
    assert sorted(calls) == ["part0", "part1"]
    assert reports[-1] == {"done": 20, "total": 20, "unit": "dòng", "sessions": 0}

    calls.clear()
    result = translate(units, [], AgentConfig(), tmp_path / "t", config)  # a retry after a later stage failed
    assert calls == []  # both finals survive on disk: no new session
    assert [row.text for row in result.rows] == [f"Lời {i}" for i in range(20)]


def test_switching_agents_starts_a_fresh_session_instead_of_continuing(monkeypatch, tmp_path):
    calls: list = []

    def half(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        calls.append((config.provider, resume))
        (Path(workspace) / "out" / "draft.tsv").write_text("0\tLời 0\n")
        return WorkspaceRun(0, "", "", 1.0)

    monkeypatch.setattr(translate_mod, "run_agent_workspace", half)
    config = TranslateConfig(retries=0, min_part_units=10)
    for provider in ("agy", "claude_code", "claude_code"):
        try:
            translate(_units(2), [], AgentConfig(provider=provider), tmp_path / "t", config)
        except TranslateError:
            pass
    # the agy draft cannot be --continue'd by claude; claude's own session can
    assert calls == [("agy", False), ("claude_code", False), ("claude_code", True)]


def test_a_user_retry_runs_with_the_current_providers(tmp_path):
    settings = Settings(root_dir=tmp_path, _env_file=None)
    settings.ensure_directories()
    service = JobService(settings)
    manifest = service.create(JobInput(url="https://example.com/v"))
    manifest.config.settings_snapshot = {"stale": True}
    manifest.state = JobState.failed
    service.save(manifest)

    service.refresh_settings(manifest.job_id)

    snapshot = service.get(manifest.job_id).config.settings_snapshot
    assert "stale" not in snapshot and "tasks" in snapshot


def test_stopping_kills_an_agent_session_and_a_command_at_once(tmp_path):
    import threading
    import time

    import pytest

    from backend.app.agents import AgentError
    from backend.app.agents.workspace import run_agent_workspace
    from backend.app.infrastructure.process import ProcessError, run_command

    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(AgentError, match="canceled"):
        run_agent_workspace(tmp_path / "ws", "x", AgentConfig(provider="command", command="sleep 30"), timeout=60, cancel=cancel)
    assert time.monotonic() - started < 5

    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(ProcessError, match="canceled"):
        run_command(["sleep", "30"], cwd=tmp_path, cancel=cancel)
    assert time.monotonic() - started < 5


def test_a_stopped_translation_keeps_its_finished_parts_for_the_next_run(monkeypatch, tmp_path):
    import threading

    from backend.app.agents import AgentError

    units = _units(20)
    stop = threading.Event()
    calls: list = []

    def agent(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        workspace = Path(workspace)
        calls.append(workspace.name)
        if workspace.name == "part1" and not stop.is_set():
            stop.set()  # the user stops while part 2 is in flight
            raise AgentError("canceled", retryable=False)
        lines = (workspace / "input" / "units.tsv").read_text().splitlines()
        body = "".join(f"{line.split(chr(9))[0].lstrip('*')}\tLời\n" for line in lines if line.startswith("*"))
        (workspace / "out" / "draft.tsv").write_text(body)
        (workspace / "out" / "final.tsv").write_text(body)
        return WorkspaceRun(0, "", "", 1.0)

    monkeypatch.setattr(translate_mod, "run_agent_workspace", agent)
    monkeypatch.setattr(translate_mod, "gemini_headroom", lambda binary: None)
    config = TranslateConfig(part_units=10, max_parallel=1)
    try:
        translate(units, [], AgentConfig(), tmp_path / "t", config, cancel=stop)
    except AgentError:
        pass
    else:
        raise AssertionError("expected the stop to end the stage")
    assert calls == ["part0", "part1"]

    calls.clear()
    translate(units, [], AgentConfig(), tmp_path / "t", config)
    assert calls == ["part1"]  # part 1 was final on disk: only the stopped part 2 runs again


def test_stopping_a_running_job_cancels_the_stage_in_flight(tmp_path):
    import threading

    settings = Settings(root_dir=tmp_path, _env_file=None)
    settings.ensure_directories()
    service = JobService(settings)
    job_id = service.create(JobInput(url="https://example.com/v")).job_id
    reached = threading.Event()

    class Translate(PipelineStage):
        name = StageName.translate

        def run(self, context):
            reached.set()
            assert context.cancel.wait(5), "the stop request never reached the stage"
            context.check_cancel()
            return StageResult(name=self.name)

    service.pipeline.stages = [Translate()]
    worker = threading.Thread(target=service.run, args=(job_id,))
    worker.start()
    assert reached.wait(5)
    service.request_cancel(job_id)
    worker.join(5)

    manifest = service.get(job_id)
    assert manifest.state == JobState.canceled
    assert manifest.stages[-1].error == "canceled"
    assert manifest.progress is None and manifest.error is None
