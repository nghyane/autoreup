from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from .artifacts import Artifact


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"
    # Everything before the render is done; the package stage waits for the user to
    # review the cover boxes, caption, font and quality, then start it (POST /repack).
    review = "review"


class StageStatus(StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class StageName(StrEnum):
    download = "download"
    asr = "asr"
    glossary = "glossary"
    translate = "translate"
    # Retired: the LLM editor pass rewrote ~half the anchors and each rewrite was a
    # fresh artifact source. Kept only so manifests of old jobs still parse.
    edit = "edit"
    qa = "qa"
    voiceover = "voiceover"
    package = "package"


class JobInput(BaseModel):
    url: str
    notes: str = ""  # per-video context (cast/gender/register/terms) fed to translation
    series_id: str | None = None       # library link: which series this dub belongs to
    source_video_id: str | None = None  # library link: the episode this dub is for


class PipelineConfig(BaseModel):
    content_genre: str = ""  # optional hint; glossary is extracted data-driven per video
    content_notes: str = ""  # per-video context for the translator (e.g. "Tống Phù Lai là nam")
    max_chars: int = 60
    display_max_chars: int = 32
    display_max_lines: int = 1
    max_cps: float = 32.0
    target_cps: float = 22.0
    min_duration: float = 0.45
    max_duration: float = 4.8
    min_gap: float = 0.03
    snap_gap: float = 0.12
    tail_out: float = 0.28
    stages: list[StageName] = Field(default_factory=lambda: [stage for stage in StageName if stage is not StageName.edit])
    # Provider/model settings frozen when the job was created (secrets stripped — the
    # manifest is served). A server-restart resume runs with this; a retry the user asks
    # for re-freezes it from the current settings first (JobService.refresh_settings).
    settings_snapshot: dict | None = None


class StageRecord(BaseModel):
    name: StageName
    status: StageStatus = StageStatus.pending
    started_at: datetime | None = None
    ended_at: datetime | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    error: str | None = None


class JobManifest(BaseModel):
    schema_version: str
    job_id: str
    input: JobInput
    state: JobState = JobState.queued
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    artifacts: list[Artifact] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    # Live progress of the stage in flight: {"stage", "started_at", and what the stage
    # reports — "step" label, "done"/"total"/"unit" counts, download bytes/speed/eta,
    # the last "logs" lines}. Cleared when the stage ends.
    progress: dict | None = None
    # Set once the user approved the render in the editor; until then the pipeline
    # stops before the package stage in the ``review`` state.
    render_approved: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
