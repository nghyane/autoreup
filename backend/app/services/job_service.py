import json
import queue
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.app.core.config import Settings
from backend.app.domain.artifacts import ArtifactKind
from backend.app.domain.jobs import JobInput, JobManifest, JobState, PipelineConfig, StageName, StageRecord, StageStatus
from backend.app.pipeline.production import ProductionPipeline
from backend.app.settings import SettingsStore, strip_secrets
from backend.app.studio.providers import extract_url


class JobNotFoundError(KeyError):
    pass


def drop_missing_artifacts(manifest: JobManifest, data_dir: Path) -> JobManifest:
    """Terminal manifests can reference files that were replaced or cleaned up later
    (e.g. a stale hardsub render). Never advertise artifacts the API cannot serve —
    the UI picks the newest artifact of a kind and would otherwise play a dead URL."""
    if manifest.state in {JobState.running, JobState.queued}:
        return manifest
    kept = [artifact for artifact in manifest.artifacts if (data_dir / artifact.path).exists()]
    if len(kept) != len(manifest.artifacts):
        manifest.artifacts = kept
    return manifest


_PROGRESS_SAVE_INTERVAL = 1.0  # seconds; yt-dlp fires hooks many times per second


class JobBusyError(RuntimeError):
    """Raised when an operation (e.g. delete) needs the job to be stopped first."""


class JobService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline = ProductionPipeline(settings)
        self._lock = threading.Lock()
        self._cancel_requested: set[str] = set()
        # The cancel event of each job whose pipeline (or download prefetch) is running:
        # a stop request sets it and the stage in flight stops mid-way.
        self._cancel_events: dict[str, threading.Event] = {}
        # Bounded FIFO queue so picking many episodes at once doesn't spawn one
        # pipeline thread per job (ASR + ffmpeg + TTS × N oversubscribes the box and
        # crashes it). A small pool of workers drains the queue; the rest wait.
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._queued: set[str] = set()  # job_ids waiting or in flight, for de-dup
        self._workers_started = False
        # Per-job mutex: the download prefetcher and the pipeline worker may touch the
        # same job; the lock serializes them so the manifest is never written by two
        # threads and the worker simply waits out an in-flight prefetch.
        self._job_locks: dict[str, threading.Lock] = {}
        # Downloads are network-bound and shouldn't wait for a pipeline slot: while all
        # workers are busy dubbing, queued jobs prefetch their video here. Kept small so
        # Douyin doesn't see a burst of parallel scrapes (CAPTCHA risk).
        self._download_slots = threading.Semaphore(2)
        self._reconcile_orphans()

    def _job_lock(self, job_id: str) -> threading.Lock:
        with self._lock:
            return self._job_locks.setdefault(job_id, threading.Lock())

    def _reconcile_orphans(self) -> None:
        """A fresh process has no worker threads for jobs a previous run left mid-flight.
        Any job still marked running/queued on disk at startup was interrupted by the
        restart — RE-ENQUEUE it: the runner resumes at the first stage whose outputs
        don't hydrate from disk, so the finished work is kept and the user never has to
        notice the restart (no more permanent 'đang chạy' zombies, no manual retry)."""
        try:
            paths = list(self.settings.jobs_dir.glob("*/manifest.json"))
        except OSError:
            return
        for path in paths:
            try:
                manifest = JobManifest.model_validate_json(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError, OSError):
                continue
            if manifest.state not in {JobState.running, JobState.queued}:
                continue
            if manifest.stages and manifest.stages[-1].status == StageStatus.running:
                manifest.stages[-1].status = StageStatus.failed
                manifest.stages[-1].ended_at = datetime.now(timezone.utc)
                manifest.stages[-1].error = "interrupted (server restart)"
            manifest.state = JobState.queued
            manifest.error = None
            manifest.progress = None
            manifest.touch()
            self.save(manifest)
            self.start_background(manifest.job_id)

    def create(self, job_input: JobInput) -> JobManifest:
        job_id = uuid4().hex[:12]
        job_input = JobInput(
            url=extract_url(job_input.url),
            notes=job_input.notes,
            series_id=job_input.series_id,
            source_video_id=job_input.source_video_id,
        )
        manifest = JobManifest(
            schema_version=self.settings.schema_version,
            job_id=job_id,
            input=job_input,
            config=PipelineConfig(
                content_genre=self.settings.content_genre,
                content_notes=job_input.notes or self.settings.content_notes,
                max_chars=self.settings.max_chars,
                display_max_chars=self.settings.display_max_chars,
                display_max_lines=self.settings.display_max_lines,
                max_cps=self.settings.max_cps,
                target_cps=self.settings.target_cps,
                min_duration=self.settings.min_duration,
                max_duration=self.settings.max_duration,
                min_gap=self.settings.min_gap,
                snap_gap=self.settings.snap_gap,
                tail_out=self.settings.tail_out,
                settings_snapshot=strip_secrets(SettingsStore(self.settings).load()),
            ),
        )
        self.save(manifest)
        return manifest

    def refresh_settings(self, job_id: str) -> None:
        """Re-freeze a stopped job's provider settings from the current store, so a retry
        the user asks for runs with the providers set NOW (e.g. translation moved from an
        exhausted agy pool to Claude Code). A server-restart resume keeps the old snapshot."""
        manifest = self.get(job_id)
        if manifest.state not in {JobState.failed, JobState.canceled}:
            return
        manifest.config.settings_snapshot = strip_secrets(SettingsStore(self.settings).load())
        manifest.touch()
        self.save(manifest)

    def start_background(self, job_id: str) -> None:
        """Enqueue a job to run on the worker pool (FIFO, bounded concurrency).

        The download starts IMMEDIATELY on the prefetch pool — a queued job no longer
        sits idle waiting for a pipeline slot while the network does nothing."""
        with self._lock:
            if job_id in self._queued:
                return
            self._queued.add(job_id)
            self._ensure_workers()
        self._queue.put(job_id)
        threading.Thread(target=self._prefetch_download, args=(job_id,), name=f"prefetch-{job_id}", daemon=True).start()

    def _prefetch_download(self, job_id: str) -> None:
        """Run just the download stage for a still-queued job. Best-effort: any failure
        leaves the job untouched and the worker downloads as usual. When the worker gets
        to the job first (or is about to), the job lock + state check make this a no-op."""
        with self._download_slots:
            lock = self._job_lock(job_id)
            if not lock.acquire(blocking=False):
                return  # worker already owns the job
            try:
                manifest = self.get(job_id)
                if manifest.state != JobState.queued or manifest.stages:
                    return
                with self._lock:
                    if job_id in self._cancel_requested:
                        return
                context = self.pipeline.create_context(job_id, manifest.input, manifest.config)
                context.cancel = self._watch_cancel(job_id)
                stage = self.pipeline.stages[0]
                record = StageRecord(name=stage.name, status=StageStatus.running, started_at=datetime.now(timezone.utc))
                context.on_progress = self._progress_publisher(manifest, record)
                result = stage.run(context)
                record.status = StageStatus.succeeded
                record.ended_at = datetime.now(timezone.utc)
                record.artifacts = result.artifacts
                record.metrics = result.metrics
                record.logs = result.logs[-200:]
                manifest = self.get(job_id)  # re-read: keep any concurrent edits
                if manifest.state != JobState.queued or manifest.stages:
                    return
                manifest.stages.append(record)
                manifest.artifacts.extend(result.artifacts)
                manifest.metrics[result.name] = result.metrics
                manifest.progress = None
                manifest.touch()
                self.save(manifest)
            except Exception:
                # The worker will download normally; until then don't show a frozen bar.
                try:
                    manifest = self.get(job_id)
                    if manifest.progress is not None:
                        manifest.progress = None
                        self.save(manifest)
                except Exception:
                    pass
            finally:
                with self._lock:
                    self._cancel_events.pop(job_id, None)
                lock.release()

    def _ensure_workers(self) -> None:
        """Lazily spin up the worker pool on first use. Caller holds ``self._lock``."""
        if self._workers_started:
            return
        self._workers_started = True
        count = max(1, self.settings.max_concurrent_jobs)
        for i in range(count):
            threading.Thread(target=self._worker_loop, name=f"job-worker-{i}", daemon=True).start()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self.run(job_id)
            except Exception:
                # run() already records failure on the manifest; never let a worker die.
                pass
            finally:
                with self._lock:
                    self._queued.discard(job_id)
                self._queue.task_done()

    def run(self, job_id: str) -> None:
        with self._job_lock(job_id):
            cancel = self._watch_cancel(job_id)
            try:
                self._run_locked(job_id, cancel)
            finally:
                with self._lock:
                    self._cancel_events.pop(job_id, None)
                    if cancel.is_set():
                        self._cancel_requested.discard(job_id)

    def _watch_cancel(self, job_id: str) -> threading.Event:
        cancel = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = cancel
        return cancel

    def _run_locked(self, job_id: str, cancel: threading.Event) -> None:
        manifest = self.get(job_id)
        context = self.pipeline.create_context(job_id, manifest.input, manifest.config)
        context.cancel = cancel
        if self._take_cancel(job_id):
            return self._mark_canceled(manifest)

        # RESUME: keep the leading run of stages that already succeeded (previous
        # attempt or the download prefetcher) and whose outputs still hydrate from
        # disk. A failed translate no longer re-downloads the video — the pipeline
        # restarts at the broken stage.
        prior = {record.name: record for record in manifest.stages if record.status == StageStatus.succeeded}
        kept: list[StageRecord] = []
        for stage in self.pipeline.stages:
            record = prior.get(stage.name)
            if record is None or not stage.hydrate(context):
                break
            kept.append(record)
        manifest.stages = kept
        manifest.artifacts = [artifact for record in kept for artifact in record.artifacts]
        manifest.metrics = {str(record.name): record.metrics for record in kept}
        manifest.error = None
        manifest.progress = None
        manifest.state = JobState.running
        manifest.touch()
        self.save(manifest)
        remaining = self.pipeline.stages[len(kept):]

        try:
            for stage in remaining:
                # Cooperative cancellation: stop at the stage boundary. A stage
                # already in flight (ffmpeg/LLM) finishes first, then we bail.
                if self._take_cancel(job_id):
                    return self._mark_canceled(manifest)
                # The render burns the overlay into the deliverable: never pick the
                # cover boxes / caption / quality for the user — wait for their review.
                if stage.name == StageName.package and not manifest.render_approved:
                    return self._mark_review(manifest)
                record = StageRecord(name=stage.name, status=StageStatus.running, started_at=datetime.now(timezone.utc))
                manifest.stages.append(record)
                context.reset_live()
                context.on_progress = self._progress_publisher(manifest, record)
                manifest.progress = self._live_progress(record, {})
                manifest.touch()
                self.save(manifest)

                result = stage.run(context)
                manifest.progress = None
                record.status = StageStatus.succeeded
                record.ended_at = datetime.now(timezone.utc)
                record.artifacts = result.artifacts
                record.metrics = result.metrics
                record.logs = result.logs[-200:]
                manifest.artifacts.extend(result.artifacts)
                manifest.metrics[result.name] = result.metrics
                manifest.touch()
                self.save(manifest)

            manifest.state = JobState.succeeded
            manifest.touch()
            self.save(manifest)
        except Exception as exc:
            if cancel.is_set():
                # the stage stopped because the user asked; what it wrote stays for a resume
                return self._mark_canceled(manifest)
            if manifest.stages:
                manifest.stages[-1].status = StageStatus.failed
                manifest.stages[-1].ended_at = datetime.now(timezone.utc)
                manifest.stages[-1].error = str(exc)
            manifest.progress = None
            manifest.state = JobState.failed
            manifest.error = str(exc)
            manifest.touch()
            self.save(manifest)

    def _progress_publisher(self, manifest: JobManifest, record: StageRecord) -> Callable[[dict, bool], None]:
        """A stage's progress callback: put the latest live state on ``manifest`` and save
        it — at once when ``urgent`` (a new step, a log line), else at most once per
        interval (download hooks and counters fire many times per second). Stages call it
        under the context's live lock, and the stage's thread owns the job lock, so writing
        this in-memory manifest cannot race another writer."""
        last_save = 0.0

        def publish(live: dict, urgent: bool = False) -> None:
            nonlocal last_save
            manifest.progress = self._live_progress(record, live)
            now = time.monotonic()
            if not urgent and now - last_save < _PROGRESS_SAVE_INTERVAL:
                return
            last_save = now
            manifest.touch()
            self.save(manifest)

        return publish

    @staticmethod
    def _live_progress(record: StageRecord, live: dict) -> dict:
        started = record.started_at.isoformat() if record.started_at else None
        return {"stage": str(record.name), "started_at": started, **live}

    def request_cancel(self, job_id: str) -> JobManifest:
        """Ask a pending/running job to stop. No-op for already-finished jobs.

        The running thread owns the state transition (to ``canceled``); here we only
        record the request so a forced delete never races the pipeline thread."""
        manifest = self.get(job_id)
        if manifest.state in {JobState.succeeded, JobState.failed, JobState.canceled, JobState.review}:
            return manifest
        with self._lock:
            self._cancel_requested.add(job_id)
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()  # the stage in flight stops now, not at the next stage boundary
        return manifest

    def _take_cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._cancel_requested:
                self._cancel_requested.discard(job_id)
                return True
        return False

    def _mark_canceled(self, manifest: JobManifest) -> None:
        if manifest.stages and manifest.stages[-1].status == StageStatus.running:
            manifest.stages[-1].status = StageStatus.failed
            manifest.stages[-1].ended_at = datetime.now(timezone.utc)
            manifest.stages[-1].error = "canceled"
        manifest.state = JobState.canceled
        manifest.progress = None
        manifest.touch()
        self.save(manifest)

    def _mark_review(self, manifest: JobManifest) -> None:
        manifest.state = JobState.review
        manifest.progress = None
        manifest.touch()
        self.save(manifest)

    def approve_render(self, job_id: str) -> JobManifest:
        """The user reviewed the overlay: run (or re-run) the package stage with it."""
        lock = self._job_lock(job_id)
        if not lock.acquire(blocking=False):
            raise JobBusyError(job_id)  # a worker or the prefetcher owns the job
        try:
            manifest = self.get(job_id)
            if manifest.state == JobState.running:
                raise JobBusyError(job_id)
            if not manifest.render_approved:
                manifest.render_approved = True
                manifest.touch()
                self.save(manifest)
        finally:
            lock.release()
        self.start_background(job_id)
        return self.get(job_id)

    def delete(self, job_id: str) -> None:
        """Remove a job and all its artifacts. Refuses while it is still running."""
        manifest = self.get(job_id)  # raises JobNotFoundError
        if manifest.state == JobState.running:
            raise JobBusyError(job_id)
        with self._lock:
            self._cancel_requested.discard(job_id)
        # Drop the per-job artifact files in each data subdir (named "<job_id>...").
        # The shared tts/cache (keyed by text) lives in a "cache" subdir and is left
        # alone. Guard each path stays inside data_dir.
        data_dir = self.settings.data_dir.resolve()
        for sub in (self.settings.packages_dir, self.settings.videos_dir, self.settings.subtitles_dir, self.settings.tts_dir):
            for path in sub.glob(f"{job_id}*"):
                resolved = path.resolve()
                if data_dir not in resolved.parents:
                    continue
                if resolved.is_dir():
                    shutil.rmtree(resolved, ignore_errors=True)
                else:
                    resolved.unlink(missing_ok=True)
        shutil.rmtree(self.settings.jobs_dir / job_id, ignore_errors=True)

    def get(self, job_id: str) -> JobManifest:
        path = self.manifest_path(job_id)
        if not path.exists():
            raise JobNotFoundError(job_id)
        manifest = JobManifest.model_validate_json(path.read_text(encoding="utf-8"))
        return drop_missing_artifacts(manifest, self.settings.data_dir)

    def save(self, manifest: JobManifest) -> None:
        with self._lock:
            path = self.manifest_path(manifest.job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(path)

    def manifest_path(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id / "manifest.json"

    def artifact_path(self, job_id: str, kind: ArtifactKind) -> tuple[Path, str]:
        manifest = self.get(job_id)
        for artifact in reversed(manifest.artifacts):
            if artifact.kind == kind:
                path = (self.settings.data_dir / artifact.path).resolve()
                data_dir = self.settings.data_dir.resolve()
                if data_dir not in path.parents and path != data_dir:
                    raise FileNotFoundError("Artifact path escapes data directory")
                if not path.exists():
                    raise FileNotFoundError(path)
                return path, artifact.mime_type
        raise FileNotFoundError(kind)

    def list_recent(self, limit: int = 20) -> list[JobManifest]:
        # Newest first by CREATION time — a stable order that doesn't reshuffle when a
        # job updates (sorting by mtime made the list jump around as jobs ran; sorting
        # by the path string buried recent jobs because job_ids are random hex).
        manifests = []
        for path in self.settings.jobs_dir.glob("*/manifest.json"):
            try:
                manifests.append(JobManifest.model_validate_json(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, ValueError):
                continue
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests[:limit]
