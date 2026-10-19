"""Library REST API — series, episodes, exports. See docs/library-spec.md."""

from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query, Response

from backend.app.api.deps import get_library_service
from backend.app.library.models import DubState, Episode, Export, Series, SeriesPage, SeriesSource
from backend.app.library.service import LibraryError
from backend.app.schemas.library import (
    AddSourceRequest,
    AddVideoRequest,
    CreateExportRequest,
    CreateSeriesRequest,
    ImportChannelRequest,
    PreviewChannelRequest,
    SuggestExportRequest,
    TranslateRequest,
    UpdateSeriesRequest,
)

router = APIRouter()

# Episode/series covers are remote CDN images that browsers can't load cross-origin
# (Douyin checks Referer). We proxy them — restricted to known image hosts so this
# can't be turned into an open SSRF relay.
_THUMB_HOSTS = ("douyinpic.com", "pstatp.com", "bytecdn.com", "byteimg.com", "douyincdn.com", "hdslb.com", "bilivideo.com")
# Bilibili's CDN hotlink-guards too, and answers 403 to a douyin.com Referer.
_BILIBILI_THUMB_HOSTS = ("hdslb.com", "bilivideo.com")


@router.get("/thumb")
def proxy_thumbnail(url: str) -> Response:
    host = urlparse(url).netloc.lower()
    if not any(host.endswith(h) or host == h for h in _THUMB_HOSTS):
        raise HTTPException(status_code=400, detail="Host ảnh không được phép")
    is_bilibili = any(host.endswith(h) for h in _BILIBILI_THUMB_HOSTS)
    referer = "https://www.bilibili.com/" if is_bilibili else "https://www.douyin.com/"
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True,
                         headers={"Referer": referer, "User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Không tải được ảnh") from exc
    return Response(content=resp.content, media_type=resp.headers.get("content-type", "image/jpeg"),
                    headers={"Cache-Control": "public, max-age=86400"})


def _series_or_404(series_id: str) -> Series:
    series = get_library_service().get_series(series_id)
    if not series:
        raise HTTPException(status_code=404, detail="Series không tồn tại")
    return series


# ----- channel import (1 channel -> many series) --------------------------
@router.post("/preview-channel")
def preview_channel(request: PreviewChannelRequest) -> dict:
    return {"groups": get_library_service().preview_channel(request.url)}


@router.post("/import-channel")
def import_channel(request: ImportChannelRequest) -> dict:
    ids = get_library_service().import_channel(request.url, [g.model_dump() for g in request.groups])
    return {"series_ids": ids}


@router.post("/add-video")
def add_video(request: AddVideoRequest) -> dict:
    try:
        return get_library_service().add_video(request.url, request.series_id, request.title)
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ----- series -------------------------------------------------------------
@router.get("/series", response_model=SeriesPage)
def list_series(q: str = "", limit: int = Query(60, le=200), offset: int = 0) -> SeriesPage:
    items, total = get_library_service().list_series(q, limit, offset)
    return SeriesPage(items=items, total=total)


@router.post("/series", response_model=Series, status_code=201)
def create_series(request: CreateSeriesRequest) -> Series:
    return get_library_service().create_series(request.title, [s.model_dump() for s in request.sources])


@router.get("/series/{series_id}", response_model=Series)
def get_series(series_id: str) -> Series:
    return _series_or_404(series_id)


@router.patch("/series/{series_id}", response_model=Series)
def update_series(series_id: str, request: UpdateSeriesRequest) -> Series:
    _series_or_404(series_id)
    return get_library_service().update_series(series_id, **request.model_dump(exclude_none=True))


@router.delete("/series/{series_id}", status_code=204)
def delete_series(series_id: str) -> None:
    if not get_library_service().delete_series(series_id):
        raise HTTPException(status_code=404, detail="Series không tồn tại")


# ----- sources ------------------------------------------------------------
@router.post("/series/{series_id}/sources", response_model=SeriesSource, status_code=201)
def add_source(series_id: str, request: AddSourceRequest) -> SeriesSource:
    _series_or_404(series_id)
    return get_library_service().add_source(series_id, request.platform, request.url, request.label)


@router.delete("/series/{series_id}/sources/{source_id}", status_code=204)
def delete_source(series_id: str, source_id: str) -> None:
    if not get_library_service().delete_source(source_id):
        raise HTTPException(status_code=404, detail="Nguồn không tồn tại")


@router.post("/series/{series_id}/scan")
def scan_series(series_id: str) -> dict:
    _series_or_404(series_id)
    try:
        return get_library_service().scan(series_id)
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ----- episodes -----------------------------------------------------------
@router.get("/series/{series_id}/episodes", response_model=list[Episode])
def list_episodes(series_id: str, dub_state: DubState | None = None) -> list[Episode]:
    _series_or_404(series_id)
    return get_library_service().list_episodes(series_id, dub_state)


@router.post("/series/{series_id}/translate")
def translate(series_id: str, request: TranslateRequest) -> dict:
    _series_or_404(series_id)
    job_ids = get_library_service().translate(series_id, request.episode_ids)
    return {"jobs": job_ids}


@router.post("/series/{series_id}/translate-titles")
def translate_titles(series_id: str) -> dict:
    _series_or_404(series_id)
    return {"translated": get_library_service().translate_titles(series_id)}


@router.delete("/episodes/{episode_id}/dub", status_code=204)
def delete_episode_dub(episode_id: str) -> None:
    if not get_library_service().delete_episode_dub(episode_id):
        raise HTTPException(status_code=404, detail="Tập không tồn tại")


# ----- exports ------------------------------------------------------------
@router.post("/series/{series_id}/suggest-export")
def suggest_export(series_id: str, request: SuggestExportRequest) -> dict:
    _series_or_404(series_id)
    try:
        return get_library_service().suggest_export_meta(series_id, request.episode_ids)
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/series/{series_id}/exports", response_model=list[Export])
def list_exports(series_id: str) -> list[Export]:
    _series_or_404(series_id)
    return get_library_service().list_exports(series_id)


@router.post("/series/{series_id}/exports", response_model=Export, status_code=201)
def create_export(series_id: str, request: CreateExportRequest) -> Export:
    _series_or_404(series_id)
    try:
        return get_library_service().create_export(
            series_id, request.title, request.description, request.hashtags, request.episode_job_ids
        )
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/exports/{export_id}/render", response_model=Export)
def render_export(export_id: str) -> Export:
    export = get_library_service().start_render(export_id)
    if not export:
        raise HTTPException(status_code=404, detail="Bản ghép không tồn tại")
    return export


@router.get("/exports/{export_id}", response_model=Export)
def get_export(export_id: str) -> Export:
    export = get_library_service().get_export(export_id)
    if not export:
        raise HTTPException(status_code=404, detail="Bản ghép không tồn tại")
    return export


@router.delete("/exports/{export_id}", status_code=204)
def delete_export(export_id: str) -> None:
    if not get_library_service().delete_export(export_id):
        raise HTTPException(status_code=404, detail="Bản ghép không tồn tại")
