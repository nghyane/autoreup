from pathlib import Path

import backend.app.dubbing.glossary as glossary_mod
import backend.app.dubbing.translate as translate_mod
from backend.app.dubbing.glossary import extract_glossary, format_glossary
from backend.app.agents import AgentConfig
from backend.app.dubbing.tsv_io import parse_tsv_rows
from backend.app.dubbing.models import Anchor, GlossaryEntry, SpeechRow
from backend.app.dubbing.units import UnitConfig, build_units
from backend.app.dubbing.models import AsrSegment
from backend.app.dubbing.translate import TranslateConfig, translate

LLM = AgentConfig()


def test_parse_tsv_rows_strips_fences_and_short_rows():
    raw = "```\n0\tXin chào\n1\tTạm biệt\nbroken\n```"
    rows = parse_tsv_rows(raw, expected_cols=2)
    assert rows == [["0", "Xin chào"], ["1", "Tạm biệt"]]


def test_parse_tsv_rows_joins_extra_columns():
    rows = parse_tsv_rows("0\ta\tb\tc", expected_cols=2)
    assert rows == [["0", "a\tb\tc"]]


def test_extract_glossary_parses_and_dedupes(monkeypatch):
    out = "罗瑞\tLa Thụy\tname\tnhân vật chính\n警察\tcảnh sát\tterm\t\n罗瑞\tLa Thụy\tname\tdup"
    monkeypatch.setattr(glossary_mod, "complete", lambda s, u, c: out)
    anchors = [Anchor(0, 0, 1, "0-0", "罗瑞是警察")]
    entries = extract_glossary(anchors, LLM, genre="hình sự")
    assert [e.term_zh for e in entries] == ["罗瑞", "警察"]
    assert "La Thụy" in format_glossary(entries)


def _rail_and_units():
    # 6 micro-segments -> 3 semantic units (2 segments each) via the v3 builder
    segs = [AsrSegment(i, i * 1.0, i * 1.0 + 0.9, f"字{i}") for i in range(6)]
    units = build_units(segs, UnitConfig(max_chars=4, min_chars=1))
    anchors = [Anchor(id=u.id, start=u.start, end=u.end, source_ids=u.segment_ids, source_zh=u.source_zh) for u in units]
    return segs, anchors


def _workspace_writer(texts, *, record=None, omit=()):
    """Fake agent run: writes out/draft.tsv + out/final.tsv, records the calls."""
    from backend.app.agents.workspace import WorkspaceRun

    def fake(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        ws = Path(workspace)
        (ws / "out").mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"{i}\t{t}" for i, t in texts.items() if i not in omit)
        (ws / "out" / "draft.tsv").write_text(body + "\n")
        (ws / "out" / "final.tsv").write_text(body + "\n")
        if record is not None:
            record.append({"prompt": prompt, "resume": resume, "timeout": timeout})
        return WorkspaceRun(0, "", "", 1.0)

    return fake


def test_workspace_translate_prepares_inputs_and_gates_edits(monkeypatch, tmp_path):
    segments, anchors = _rail_and_units()
    calls = []
    texts = {0: "Lời 0 giữ số 7", 1: "Lời 1 giữ số 7", 2: "Lời 2 giữ số 7"}
    monkeypatch.setattr(translate_mod, "run_agent_workspace", _workspace_writer(texts, record=calls))

    result = translate(anchors, [], LLM, tmp_path / "translate", TranslateConfig())
    rows = result.rows
    assert [r.anchor_id for r in rows] == [a.id for a in anchors]
    assert [r.anchor_start for r in rows] == [a.start for a in anchors]
    assert len(calls) == 1 and calls[0]["resume"] is False
    # the workspace is a real, inspectable project: skill + inputs, artifacts on disk
    ws = tmp_path / "translate"
    assert (ws / "SKILL.md").exists()
    units_tsv = (ws / "input" / "units.tsv").read_text()
    assert units_tsv.count("\n") == len(anchors)
    assert "glossary.tsv" in str(list((ws / "input").iterdir()))


def test_workspace_translate_gates_unsafe_edits(monkeypatch, tmp_path):
    segments, anchors = _rail_and_units()
    from backend.app.agents.workspace import WorkspaceRun

    def fake(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        ws = Path(workspace)
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "draft.tsv").write_text("0\tLời 0 giữ số 7\n1\tLời 1 giữ số 7\n2\tLời 2 giữ số 7\n")
        # unit 0: same length, keeps the number -> accepted
        # unit 1: drops the number "7" -> the code gate must reject it
        (ws / "out" / "final.tsv").write_text("0\tLời 0 sai số 7\n1\tLời 1 sai số\n2\tLời 2 giữ số 7\n")
        return WorkspaceRun(0, "", "", 1.0)

    monkeypatch.setattr(translate_mod, "run_agent_workspace", fake)
    rows = translate(anchors, [], LLM, tmp_path / "translate", TranslateConfig()).rows
    texts = {r.anchor_id: r.text for r in rows}
    assert texts[0] == "Lời 0 sai số 7"
    assert texts[1] == "Lời 1 giữ số 7"


def test_workspace_translate_follow_up_turn_for_missing_ids(monkeypatch, tmp_path):
    segments, anchors = _rail_and_units()
    calls = []

    def fake(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        from backend.app.agents.workspace import WorkspaceRun

        ws = Path(workspace)
        (ws / "out").mkdir(parents=True, exist_ok=True)
        omit = (2,) if not resume else ()
        body = "\n".join(f"{i}\tLời {i}" for i in range(3) if i not in omit)
        (ws / "out" / "final.tsv").write_text(body + "\n")
        calls.append(resume)
        return WorkspaceRun(0, "", "", 1.0)

    monkeypatch.setattr(translate_mod, "run_agent_workspace", fake)
    result = translate(anchors, [], LLM, tmp_path / "translate", TranslateConfig(retries=2))
    assert [r.text for r in result.rows] == ["Lời 0", "Lời 1", "Lời 2"]
    assert calls == [False, True]          # one follow-up turn, resume mode
    assert len(result.runs) == 2


def test_workspace_translate_raises_when_ids_stay_missing(monkeypatch, tmp_path):
    segments, anchors = _rail_and_units()
    monkeypatch.setattr(translate_mod, "run_agent_workspace", _workspace_writer({0: "Lời 0"}, omit=(1, 2)))
    try:
        translate(anchors, [], LLM, tmp_path / "translate", TranslateConfig(retries=1))
        assert False, "expected TranslateError"
    except translate_mod.TranslateError:
        pass


def test_translate_bisects_a_short_session_instead_of_retrying(monkeypatch, tmp_path):
    """A session that cannot finish its part is split in half, recursively."""
    from backend.app.agents.workspace import WorkspaceRun
    from backend.app.dubbing.models import Anchor

    units = [Anchor(id=i, start=i * 5.0, end=i * 5.0 + 4.0, source_ids=f"{i}-{i}", source_zh=f"字{i}") for i in range(60)]
    runs = {"n": 0}

    def fake(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        runs["n"] += 1
        # the input file marks the lines this session must translate
        marked = []
        for line in (ws / "input" / "units.tsv").read_text().splitlines():
            if line.startswith("*"):
                marked.append(line.split("\t")[0].lstrip("*"))
        (ws / "out").mkdir(exist_ok=True)
        if len(marked) <= 20:  # only small sessions are "finishable"
            (ws / "out" / "final.tsv").write_text("\n".join(f"{i}\tLời {i}" for i in marked) + "\n")
        else:
            (ws / "out" / "final.tsv").write_text("")
        return WorkspaceRun(0, "", "", 1.0)

    monkeypatch.setattr(translate_mod, "run_agent_workspace", fake)
    result = translate(units, [], LLM, tmp_path / "translate", TranslateConfig(part_units=60, min_part_units=25, max_split_depth=3))
    assert [r.text for r in result.rows] == [f"Lời {i}" for i in range(60)]
    assert any("bisecting" in log for log in result.logs)
    assert runs["n"] >= 7  # 1 + 2 + 4 sessions
