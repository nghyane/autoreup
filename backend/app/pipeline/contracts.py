import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from backend.app.domain.artifacts import Artifact
from backend.app.domain.jobs import StageName

_LIVE_LOG_LINES = 5  # log lines of the stage in flight shown while it runs


class StageCanceled(RuntimeError):
    """The user stopped the job while this stage was running."""


@dataclass
class StageResult:
    name: StageName
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


class PipelineStage:
    name: StageName

    def run(self, context: "PipelineContext") -> StageResult:
        raise NotImplementedError

    def hydrate(self, context: "PipelineContext") -> bool:
        """Reload this stage's outputs from disk into ``context.artifacts`` WITHOUT
        re-running it. Returns False when the artifacts are missing/unreadable — the
        stage (and everything after it) must then run for real. This is what lets a
        failed job RESUME at the broken stage instead of re-downloading everything."""
        return False


@dataclass
class PipelineContext:
    job_id: str
    input_url: str
    job_dir: object
    settings: object
    config: object
    artifacts: dict[str, object] = field(default_factory=dict)
    series_id: str | None = None
    source_video_id: str | None = None  # library link: keys the source-video cache  # library link: lets stages share per-series state (e.g. glossary)
    # Live progress of the stage in flight (see report/log); the job runner publishes
    # it on the manifest. Called as on_progress(live, urgent). None: nobody listens.
    on_progress: Callable[[dict, bool], None] | None = None
    live: dict = field(default_factory=dict)
    _live_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    # Set when the user stops the job: long work (agent sessions, Whisper, downloads,
    # TTS) checks it and stops mid-stage, keeping what it already wrote to disk.
    cancel: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise StageCanceled("canceled")

    def report(self, fields: dict) -> None:
        """Merge live numbers of the stage in flight — a ``step`` label, ``done``/``total``
        counts (``unit`` names what is counted), download bytes — and publish them. A None
        value drops that key; a new ``step`` drops the previous step's numbers and is
        published at once, while counters are throttled by the publisher. Thread-safe:
        parallel translate sessions report into one context. A new step is also a
        cancel point: a stopped job does not start its next step."""
        if "step" in fields:
            self.check_cancel()
        with self._live_lock:
            urgent = "step" in fields and fields["step"] != self.live.get("step")
            if urgent:
                self.live = {"logs": self.live["logs"]} if "logs" in self.live else {}
            for key, value in fields.items():
                if value is None:
                    self.live.pop(key, None)
                else:
                    self.live[key] = value
            self._publish(urgent)

    def log(self, line: str) -> None:
        """Show one line of the stage's log while it runs (only the last few are kept)."""
        with self._live_lock:
            self.live["logs"] = [*self.live.get("logs", []), line][-_LIVE_LOG_LINES:]
            self._publish(True)

    def reset_live(self) -> None:
        with self._live_lock:
            self.live = {}

    def _publish(self, urgent: bool) -> None:
        if self.on_progress is not None:
            self.on_progress(dict(self.live), urgent)
