"""L4 — unit translation as ONE workspace agent run (F1 + LLM fundamentals).

An LLM translates best with the whole document in front of it, on units that carry a
complete thought, writing artifacts to disk it can re-read and fix. So this stage is
not a series of one-shot prompts: it prepares a workspace (``SKILL.md`` + ``input/``),
lets the agent run its full workflow there — read all → draft → self-review → final —
and then validates the FILES on disk. Model prose is never parsed for content.

The code still owns safety: every line the agent changed between draft and final goes
through ``gate_patch`` (numbers, negations, glossary terms, length), so a bad edit can
only ever be rejected, never smuggled in.

An LLM API connection has no file tools, so the same workspace is EMULATED: SKILL.md
becomes the system prompt, the input files travel inline, the fenced TSV reply is
written to ``out/draft.tsv`` + ``out/final.tsv`` — and from there the disk-first
validation, follow-up turns and bisection are exactly the agent path's.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from backend.app.agents import AgentConfig, AgentError, complete, is_api
from backend.app.agents.quota import gemini_headroom
from backend.app.agents.workspace import WorkspaceRun, run_agent_workspace

from .edit import EditConfig, gate_patch
from .models import Anchor, GlossaryEntry, SpeechRow
from .text import has_han, has_kana, sanitize_caption

_SKILL_PATH = Path(__file__).with_name("skills") / "translate" / "SKILL.md"
_AGENT_PROMPT = "Đọc SKILL.md và làm đầy đủ workflow trong đó với input/units.tsv."
_API_NOTE = """

## Chạy qua API (không có công cụ đọc/ghi file)

Lượt này bạn KHÔNG đọc/ghi được file: nội dung `input/units.tsv`, `input/glossary.tsv`,
`input/notes.md` được dán ngay trong tin nhắn. Làm đủ workflow ở trên trong đầu (đọc hết →
nháp → soát → bản cuối) rồi trả về DUY NHẤT nội dung `out/final.tsv` trong MỘT khối ```tsv —
mỗi dòng `ID<TAB>TEXT`, không giải thích gì thêm.
"""


# A session that failed on the account's quota: every further session fails the same
# way, so bisecting it only multiplies dead sessions (measured: one exhausted agy pool
# grew a 1021-unit job into 103 workspaces before the stage gave up).
_QUOTA_ERROR = re.compile(r"RESOURCE_EXHAUSTED|quota|usage limit|rate.?limit|\b429\b", re.IGNORECASE)


class TranslateError(RuntimeError):
    pass


class _Progress:
    """Live numbers for the stage: units translated so far and sessions in flight.

    Parts run on parallel threads; a unit counts once whichever (sub)session wrote it.
    ``stopped`` is set by the first quota failure so queued sessions do not start;
    ``cancel`` (the user stopped the job) also kills the sessions in flight."""

    def __init__(
        self,
        total: int,
        report: Callable[[dict], None] | None,
        log: Callable[[str], None] | None,
        cancel: threading.Event | None = None,
    ):
        self.total = total
        self.cancel = cancel
        self._report = report
        self._log = log
        self._lock = threading.Lock()
        self._done: set[int] = set()
        self.sessions = 0
        self.stopped: str | None = None

    def session(self, delta: int) -> None:
        with self._lock:
            self.sessions += delta
            self._publish()

    def finished(self, texts: dict[int, str], missing: list[int], label: str, part: list[Anchor], seconds: float) -> None:
        with self._lock:
            self._done.update(unit_id for unit_id in texts if unit_id not in missing)
            self._publish()
        self.log(f"{label}: {len(part) - len(missing)}/{len(part)} dòng ({seconds:.0f}s)")

    def log(self, line: str) -> None:
        if self._log is not None:
            self._log(line)

    def _publish(self) -> None:
        if self._report is not None:
            self._report({"done": len(self._done), "total": self.total, "unit": "dòng", "sessions": self.sessions})


@dataclass(slots=True)
class TranslateConfig:
    notes: str = ""            # per-video context, written to input/notes.md
    # A session (one agy run) is bounded by the CLI's print timeout AND by what the
    # agent can actually finish: measured, a 190-unit session = 6m17s, while a 404-unit
    # session produced ~40 units and then failed the stage. So: budget each session
    # well under the timeout, run sessions in parallel, and BISECT a session that
    # times out or comes back short instead of retrying it at the same size.
    timeout: float = 600.0     # wall-clock budget for ONE session
    retries: int = 1           # follow-up turns inside a session before bisecting
    part_units: int = 100      # measured capacity with ~2x headroom
    min_part_units: int = 5    # anywhere above this a short session is split again
    max_split_depth: int = 3   # bisections per part (effective floor: 100/2^3 units)
    max_parallel: int = 4      # quota-aware cap (see _parallelism)


@dataclass(slots=True)
class TranslateResult:
    rows: list[SpeechRow]
    runs: list[WorkspaceRun] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


def _parse_table(path: Path) -> dict[int, str]:
    """Read ``ID<TAB>TEXT`` rows from a file the agent wrote (tolerates fences/quotes)."""
    texts: dict[int, str] = {}
    if not path.exists():
        return texts
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().strip("|").strip()
        if not line or line.startswith(("```", "#")):
            continue
        cells = [cell.strip() for cell in line.split("\t")]
        if len(cells) < 2:
            continue
        key = cells[0].lstrip("*").strip()
        if not key.isdigit():
            continue
        text = sanitize_caption(cells[1])
        if text:
            texts[int(key)] = text
    return texts


def _provider(llm: AgentConfig) -> str:
    return (llm.provider or "agy").strip().lower()


def _can_resume(workspace: Path, llm: AgentConfig) -> bool:
    """``--continue`` resumes the cwd's last session OF THAT CLI: a workspace drafted by
    agy has no Claude Code session to continue, so switching agents starts fresh (the
    agent still sees out/draft.tsv on disk)."""
    stamp = workspace / "out" / ".agent"
    return (workspace / "out" / "draft.tsv").exists() and stamp.exists() and stamp.read_text(encoding="utf-8").strip() == _provider(llm)


def _quota_error(runs: list[WorkspaceRun]) -> str | None:
    for run in runs:
        if run.returncode != 0:
            detail = f"{run.stderr}\n{run.stdout}"
            if _QUOTA_ERROR.search(detail):
                lines = [line for line in detail.strip().splitlines() if _QUOTA_ERROR.search(line)]
                return (lines[0] if lines else detail.strip())[:300]
    return None


def _workspace_matches(workspace: Path, digest: str) -> bool:
    stamp = workspace / "input" / ".content"
    return stamp.exists() and stamp.read_text(encoding="utf-8").strip() == digest


def _prepare_workspace(
    workspace: Path, units: list[Anchor], glossary: list[GlossaryEntry], notes: str, marked: set[int]
) -> bool:
    """(Re)build the workspace; returns True when a previous run's artifacts survive.

    The agent writes its own helpers and partial drafts here (observed: it split a
    too-big job itself with part1.py..part5.py). Those are progress, not garbage, so
    the workspace is CONTENT-ADDRESSED: same inputs -> keep out/ and let the next
    session continue (``--continue``); different inputs -> rebuild from scratch."""
    units_text = "\n".join(
        f"{'*' if u.id in marked else ' '}{u.id}\t{u.start:.2f}\t{u.end:.2f}\t{u.source_zh}" for u in units
    ) + "\n"
    digest = hashlib.sha256(units_text.encode("utf-8")).hexdigest()[:16]
    keep = _workspace_matches(workspace, digest)
    if workspace.exists() and not keep:
        shutil.rmtree(workspace)
    (workspace / "input").mkdir(parents=True, exist_ok=True)
    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_SKILL_PATH, workspace / "SKILL.md")
    (workspace / "input" / ".content").write_text(digest, encoding="utf-8")
    (workspace / "input" / "units.tsv").write_text(units_text, encoding="utf-8")
    (workspace / "input" / "glossary.tsv").write_text(
        "\n".join(f"{g.term_zh}\t{g.term_vi}\t{g.kind}\t{g.note}" for g in glossary) + "\n", encoding="utf-8"
    )
    (workspace / "input" / "notes.md").write_text(notes or "(không có bối cảnh bổ sung)\n", encoding="utf-8")
    return keep


def _collect(workspace: Path, units: list[Anchor], glossary: list[GlossaryEntry]) -> dict[int, str]:
    """Draft -> final, with every edit gated (unsafe edits fall back to the draft)."""
    draft = _parse_table(workspace / "out" / "draft.tsv")
    final = _parse_table(workspace / "out" / "final.tsv")
    texts: dict[int, str] = {}
    for unit in units:
        candidate = final.get(unit.id) or draft.get(unit.id)
        if candidate is None:
            continue
        base = draft.get(unit.id)
        if base and base != candidate:
            gated = gate_patch(base, candidate, glossary, EditConfig())
            candidate = gated or base
        texts[unit.id] = candidate
    return texts


def _unusable(texts: dict[int, str], units: list[Anchor]) -> list[int]:
    return [
        unit.id
        for unit in units
        if not (text := texts.get(unit.id)) or has_han(text) or has_kana(text)
    ]


def _run_part(
    part: list[Anchor],
    all_units: list[Anchor],
    glossary: list[GlossaryEntry],
    llm: AgentConfig,
    workspace: Path,
    config: TranslateConfig,
    cancel: threading.Event | None = None,
) -> tuple[dict[int, str], list[WorkspaceRun], list[int]]:
    """One session (plus its bounded follow-up turns) for one part."""
    marked = {unit.id for unit in part}
    keep = _prepare_workspace(workspace, all_units, glossary, config.notes, marked)
    if keep and not _unusable(_parse_table(workspace / "out" / "final.tsv"), part):
        # a reviewed final from an earlier attempt (same inputs) needs no new session
        return _collect(workspace, part, glossary), [], []
    if is_api(llm):
        return _run_part_inline(part, glossary, llm, workspace, config)
    resume = keep and _can_resume(workspace, llm)
    (workspace / "out" / ".agent").write_text(_provider(llm), encoding="utf-8")
    runs = [run_agent_workspace(workspace, _AGENT_PROMPT, llm, timeout=config.timeout, resume=resume, cancel=cancel)]
    texts = _collect(workspace, part, glossary)
    missing = _unusable(texts, part)
    attempt = 0
    while missing and attempt < max(0, config.retries) and not _quota_error(runs):
        attempt += 1
        prompt = (
            f"out/final.tsv còn thiếu/sai {len(missing)} ID: {', '.join(str(i) for i in missing[:40])}. "
            "Bổ sung và sửa đúng định dạng ID<TAB>TEXT (tiếng Việt, KHÔNG Hán tự, không rỗng), "
            "kiểm tra ĐỦ mọi ID có dấu * trong input/units.tsv rồi ghi lại out/final.tsv."
        )
        runs.append(run_agent_workspace(workspace, prompt, llm, timeout=config.timeout / 2, resume=True, cancel=cancel))
        texts = _collect(workspace, part, glossary)
        missing = _unusable(texts, part)
    return texts, runs, missing


def _inline_inputs(workspace: Path) -> str:
    blocks = []
    for name in ("units.tsv", "glossary.tsv", "notes.md"):
        text = (workspace / "input" / name).read_text(encoding="utf-8").rstrip()
        blocks.append(f"### input/{name}\n```\n{text}\n```")
    return "\n\n".join(blocks)


def _write_rows(path: Path, part: list[Anchor], rows: dict[int, str]) -> None:
    path.write_text("".join(f"{u.id}\t{rows[u.id]}\n" for u in part if u.id in rows), encoding="utf-8")


def _run_part_inline(
    part: list[Anchor],
    glossary: list[GlossaryEntry],
    llm: AgentConfig,
    workspace: Path,
    config: TranslateConfig,
) -> tuple[dict[int, str], list[WorkspaceRun], list[int]]:
    """The workspace session emulated over an LLM API: one call, then follow-ups that
    ask only for the IDs still missing. Rows already on disk (same inputs) are kept."""
    out = workspace / "out"
    system = _SKILL_PATH.read_text(encoding="utf-8") + _API_NOTE
    inputs = _inline_inputs(workspace)
    texts = _collect(workspace, part, glossary)
    missing = _unusable(texts, part)
    runs: list[WorkspaceRun] = []
    while missing and len(runs) <= max(0, config.retries):
        if len(missing) == len(part):
            ask = "Dịch MỌI dòng có dấu * trong input/units.tsv."
        else:
            done = "\n".join(f"{u.id}\t{texts[u.id]}" for u in part if u.id in texts)
            ask = (
                f"out/final.tsv hiện tại:\n```tsv\n{done}\n```\n"
                f"Còn thiếu/sai {len(missing)} ID: {', '.join(str(i) for i in missing)}. "
                "Chỉ trả về các ID đó (tiếng Việt, KHÔNG Hán tự, không rỗng), giữ mạch và xưng hô như các dòng đã có."
            )
        # same budgets as the agent path: a full session, then half for each follow-up
        session = replace(llm, timeout=config.timeout if not runs else config.timeout / 2)
        started = time.monotonic()
        try:
            reply = complete(system, f"{inputs}\n\n{ask}", session)
        except AgentError as exc:
            # like a timed-out agent session: report it as a failed run so the part is
            # bisected (too-long output, timeouts), instead of failing the whole stage
            timed_out = "Timeout" in str(exc)
            runs.append(WorkspaceRun(-1, "", str(exc), round(time.monotonic() - started, 1), timed_out=timed_out))
            break
        runs.append(WorkspaceRun(0, reply, "", round(time.monotonic() - started, 1)))
        raw = out / f"reply-{len(runs)}.txt"
        raw.write_text(reply, encoding="utf-8")
        fresh = _parse_table(raw)
        texts.update({unit_id: fresh[unit_id] for unit_id in missing if unit_id in fresh})
        # no separate self-review pass over an API: draft == final, nothing to gate
        _write_rows(out / "draft.tsv", part, texts)
        _write_rows(out / "final.tsv", part, texts)
        texts = _collect(workspace, part, glossary)
        missing = _unusable(texts, part)
    return texts, runs, missing


def _parallelism(config: TranslateConfig, llm: AgentConfig) -> int:
    """How many sessions may run at once: part count, capped by measured quota headroom.

    Every agent call draws from the same 5-hour pool as the fit loop; when the oracle
    says the window is nearly spent, serialize instead of collecting 429s. An LLM API
    or another CLI (claude, codex) has its own limits, not the agy pool, so the oracle
    does not apply there."""
    if (llm.provider or "agy").strip().lower() not in {"agy", "antigravity"}:
        return config.max_parallel
    headroom = gemini_headroom(llm.bin)
    if headroom is None:
        return config.max_parallel
    if headroom < 0.3:
        return 1
    return max(1, config.max_parallel)


def _session(
    part: list[Anchor],
    all_units: list[Anchor],
    glossary: list[GlossaryEntry],
    llm: AgentConfig,
    workspace: Path,
    config: TranslateConfig,
    progress: _Progress,
    label: str,
    depth: int = 0,
) -> tuple[dict[int, str], list[WorkspaceRun], list[int], list[str]]:
    """Run one part; a session that comes back short is BISECTED, never retried as-is —
    unless it failed on the quota, which stops the whole stage."""
    if progress.cancel is not None and progress.cancel.is_set():
        raise TranslateError("canceled")
    if progress.stopped:
        raise TranslateError(f"agent hết quota: {progress.stopped}")
    progress.session(+1)
    started = time.monotonic()
    try:
        texts, runs, missing = _run_part(part, all_units, glossary, llm, workspace, config, progress.cancel)
    finally:
        progress.session(-1)
    progress.finished(texts, missing, label, part, time.monotonic() - started)
    quota = _quota_error(runs) if missing else None
    if quota:
        progress.stopped = progress.stopped or quota
        raise TranslateError(f"agent hết quota: {quota}")
    logs: list[str] = []
    if missing and len(part) > config.min_part_units and depth < config.max_split_depth:
        mid = len(part) // 2
        line = (
            f"translate: session left {len(missing)}/{len(part)} units at depth {depth} -> "
            f"bisecting into {mid}+{len(part) - mid}"
        )
        logs.append(line)
        progress.log(f"{label}: thiếu {len(missing)}/{len(part)} dòng -> chia đôi {mid}+{len(part) - mid}")
        left_texts, left_runs, left_missing, left_logs = _session(
            part[:mid], all_units, glossary, llm, workspace / "a", config, progress, f"{label}/a", depth + 1
        )
        right_texts, right_runs, right_missing, right_logs = _session(
            part[mid:], all_units, glossary, llm, workspace / "b", config, progress, f"{label}/b", depth + 1
        )
        texts.update(left_texts)
        texts.update(right_texts)
        runs += left_runs + right_runs
        missing = left_missing + right_missing
        logs += left_logs + right_logs
    return texts, runs, missing, logs


def translate(
    units: list[Anchor],
    glossary: list[GlossaryEntry],
    llm: AgentConfig,
    workspace: Path,
    config: TranslateConfig | None = None,
    report: Callable[[dict], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
) -> TranslateResult:
    """Translate in quota-bounded parallel sessions, each seeing the whole transcript.

    ``report`` receives live counts (units done, sessions in flight) and ``log`` one
    line per finished session, so a long stage shows how far it got. Setting ``cancel``
    stops the stage within a second; finished parts stay on disk for the next run."""
    config = config or TranslateConfig()
    if not units:
        return TranslateResult(rows=[])
    workspace = Path(workspace)
    size = max(1, config.part_units)
    parts = [units[i : i + size] for i in range(0, len(units), size)]
    progress = _Progress(len(units), report, log, cancel)

    outcomes: list[tuple[dict[int, str], list[WorkspaceRun], list[int], list[str]]] = []
    if len(parts) == 1:
        # short video: one session at the workspace root, no quota oracle needed
        parallel = 1
        outcomes.append(_session(parts[0], units, glossary, llm, workspace, config, progress, "phần 1"))
    else:
        parallel = min(len(parts), _parallelism(config, llm))
        progress.log(f"{len(units)} dòng -> {len(parts)} phần, {parallel} phiên song song ({_provider(llm)})")
        if parallel <= 1:
            for index, part in enumerate(parts):
                outcomes.append(_session(part, units, glossary, llm, workspace / f"part{index}", config, progress, f"phần {index + 1}"))
        else:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [
                    pool.submit(_session, part, units, glossary, llm, workspace / f"part{index}", config, progress, f"phần {index + 1}")
                    for index, part in enumerate(parts)
                ]
                outcomes = [future.result() for future in futures]

    texts: dict[int, str] = {}
    runs: list[WorkspaceRun] = []
    missing: list[int] = []
    logs: list[str] = [f"translate: {len(parts)} part(s), {parallel} parallel, {config.part_units} units/session"]
    for part_texts, part_runs, part_missing, part_logs in outcomes:
        texts.update(part_texts)
        runs += part_runs
        missing += part_missing
        logs += part_logs
    if missing:
        errors = [run.stderr for run in runs if run.returncode != 0 and run.stderr]
        detail = f" (last error: {errors[-1]})" if errors else ""
        raise TranslateError(f"workspace sessions left {len(missing)} unit(s) untranslated: {missing[:10]}{detail}")

    rows = [
        SpeechRow(
            sub_id=index,
            anchor_id=unit.id,
            anchor_start=unit.start,
            anchor_end=unit.end,
            source_ids=unit.source_ids,
            text=texts[unit.id],
        )
        for index, unit in enumerate(units)
    ]
    return TranslateResult(rows=rows, runs=runs, logs=logs)
