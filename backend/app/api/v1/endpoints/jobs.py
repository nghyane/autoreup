import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.app.api.deps import get_job_service
from backend.app.domain.artifacts import ArtifactKind
from backend.app.domain.jobs import JobState
from backend.app.schemas.jobs import CreateJobRequest, JobResponse
from backend.app.services.job_service import JobBusyError, JobNotFoundError


router = APIRouter()


@router.post("", response_model=JobResponse, status_code=202)
def create_job(request: CreateJobRequest) -> JobResponse:
    service = get_job_service()
    manifest = service.create(request.to_job_input())
    service.start_background(manifest.job_id)
    return JobResponse(job=manifest)


@router.get("", response_model=list[JobResponse])
def list_jobs() -> list[JobResponse]:
    return [JobResponse(job=job) for job in get_job_service().list_recent()]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        return JobResponse(job=get_job_service().get(job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str) -> JobResponse:
    try:
        return JobResponse(job=get_job_service().request_cancel(job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    try:
        get_job_service().delete(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail="Job đang chạy — hãy Dừng trước khi xóa.") from exc


@router.get("/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    async def stream():
        last_payload = ""
        while True:
            try:
                manifest = get_job_service().get(job_id)
            except JobNotFoundError:
                yield "event: error\ndata: {\"error\": \"Job not found\"}\n\n"
                return
            payload = manifest.model_dump_json()
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if manifest.state in {JobState.succeeded, JobState.failed, JobState.canceled, JobState.review}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/{job_id}/artifacts/{kind}")
def download_artifact(job_id: str, kind: ArtifactKind) -> FileResponse:
    try:
        path, mime_type = get_job_service().artifact_path(job_id, kind)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    return FileResponse(path, media_type=mime_type, filename=path.name)


@router.get("/{job_id}/artifacts")
def list_artifacts(job_id: str) -> dict:
    try:
        manifest = get_job_service().get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return {"artifacts": [json.loads(artifact.model_dump_json()) for artifact in manifest.artifacts]}


# --- overlay editor: caption position + cover regions (blur boxes) ----------------


class CoverRegionIn(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)
    mode: str = "blur"
    strength: float = Field(default=1.0, gt=0.0, le=1.0)
    label: str = ""
    image: str = ""


class OverlayIn(BaseModel):
    caption_mode: str = "band"
    caption_y: float | None = None
    caption_scale: float = 1.0
    cover: list[CoverRegionIn] = Field(default_factory=list)
    font: str = ""
    bitrate: str = ""
    bed_volume: float | None = Field(default=None, ge=0.0, le=1.0)
    voice_volume: float = Field(default=1.0, ge=0.0, le=1.0)
    gap_volume: float | None = Field(default=None, ge=0.0, le=1.0)


def _overlay_context(job_id: str):
    """(job_dir, source video) or the right HTTP error."""
    from backend.app.api.deps import get_job_service

    service = get_job_service()
    try:
        service.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    video = service.settings.videos_dir / f"{job_id}.mp4"
    if not video.exists():
        raise HTTPException(status_code=409, detail="Video nguồn chưa có — chạy job trước đã.")
    return service.settings.jobs_dir / job_id, video


def _frame_info(video: Path) -> dict:
    """Output frame geometry for the editor: aspect, letterbox padding, duration.

    The editor MUST preview the output frame, not the raw source: for a portrait (or
    any non-canonical) source the render pads to the canonical size, and a box drawn on
    the raw frame would land somewhere else in the final video."""
    import cv2

    from backend.app.api.deps import get_job_service
    from backend.app.pipeline.production import _hardsub_config
    from backend.app.services.audio import probe_duration
    from backend.app.services.hardsub import overlay_geometry

    config = _hardsub_config(get_job_service().settings)
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    geometry = overlay_geometry(width, height, None, config)
    try:
        duration = round(probe_duration(video), 3)
    except RuntimeError:
        duration = 0.0
    return {
        "out": {"width": geometry.out_width, "height": geometry.out_height},
        "source": {"width": width, "height": height},
        "scale": geometry.scale,
        "pad": {"x": geometry.pad_x, "y": geometry.pad_y},
        "font_size": config.font_size,
        "duration": duration,
        "encode": _encode_info(video),
        "bed_volume": get_job_service().settings.dub_bed_volume,
        "gap_volume": get_job_service().settings.dub_gap_volume,
    }


def _encode_info(video: Path) -> dict:
    """The render quality tiers (measured, see BITRATE_PRESETS) and the source's own
    codec/rate, so the editor can label each choice with the file size it produces."""
    from backend.app.api.deps import get_job_service
    from backend.app.infrastructure.ffmpeg_encode import BITRATE_PRESETS, probe_video

    settings = get_job_service().settings
    probe = probe_video(video) or {}
    return {
        "source_codec": probe.get("codec", ""),
        "source_bitrate": probe.get("bit_rate", 0),
        "default": settings.video_bitrate,
        "audio_bitrate": 192_000,
        "presets": list(BITRATE_PRESETS),
    }


@router.get("/{job_id}/overlay")
def get_overlay(job_id: str) -> dict:
    """The job's overlay: the saved config (or the detector's default) + frame geometry."""
    from backend.app.services.overlay import default_overlay, load_overlay

    job_dir, video = _overlay_context(job_id)
    config = load_overlay(job_dir, job_id) or default_overlay(video)
    return {**config.to_json(), "frame": _frame_info(video)}


@router.put("/{job_id}/overlay")
def put_overlay(job_id: str, payload: OverlayIn) -> dict:
    """Persist the overlay (validated, fractions only). Re-render with /repack."""
    from backend.app.services.overlay import OverlayConfig, save_overlay

    job_dir, _video = _overlay_context(job_id)
    saved = save_overlay(job_dir, job_id, OverlayConfig.from_json(payload.model_dump()))
    return saved.to_json()


@router.post("/{job_id}/overlay/frame")
def overlay_frame(job_id: str, payload: OverlayIn, t: float = 1.0, text: str = "") -> Response:
    """One frame with the overlay applied — the editor's WYSIWYG check.

    Uses the SAME hardsub config the package stage builds (canonical output size,
    encoder-independent geometry), so the preview cannot disagree with the render."""
    from backend.app.pipeline.production import _hardsub_config
    from backend.app.services.hardsub import render_overlay_frame
    from backend.app.services.overlay import OverlayConfig

    from backend.app.api.deps import get_job_service

    _job_dir, video = _overlay_context(job_id)
    overlay = OverlayConfig.from_json(payload.model_dump())
    config = _hardsub_config(get_job_service().settings, overlay)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frame.jpg"
        render_overlay_frame(video, out, max(0.0, t), config, caption=text)
        return Response(content=out.read_bytes(), media_type="image/jpeg")


@router.post("/{job_id}/repack", response_model=JobResponse)
def repack_job(job_id: str) -> JobResponse:
    """Approve the render and run the package stage (picks up the saved overlay)."""
    service = get_job_service()
    try:
        return JobResponse(job=service.approve_render(job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail="Job đang chạy.") from exc
