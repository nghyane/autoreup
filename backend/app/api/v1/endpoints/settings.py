"""/settings — provider connections + per-task bindings (secrets never leave masked)."""

from __future__ import annotations

import json
import os
import shlex
import shutil
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi import APIRouter, Body, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, ValidationError

from backend.app.agents import AgentError, api_llm
from backend.app.core.config import get_settings
from backend.app.services.tts_engines import TtsEngineError, audio_media_type, build_engine, list_voices as engine_voices, load_vieneu
from backend.app.settings import CLI_BINS, Connection, SettingsFileError, SettingsStore, TtsTask, agent_for

router = APIRouter()

_PREVIEW_TEXT = "Xin chào, đây là giọng đọc thử cho video lồng tiếng của bạn."
_PROBE_TIMEOUT = 15.0  # a model listing is quick; never wait the long agent timeout here
# `claude --model` aliases that always point at the newest model of each tier
_CLAUDE_MODELS = ["sonnet", "opus", "fable", "haiku"]


def get_settings_store() -> SettingsStore:
    return SettingsStore(get_settings())


def _connection(store: SettingsStore, connection_id: str) -> Connection:
    conn = store.load().connection(connection_id)
    if conn is None:
        raise HTTPException(status_code=404, detail=f"không có kết nối {connection_id!r}")
    return conn


# Voice services with nothing to configure: the Voice tab lists + previews them straight
# from the draft, before the new connection is saved.
_KEYLESS_TTS = ("edge", "vieneu")


def _tts_connection(store: SettingsStore, connection_id: str, kind: str) -> Connection:
    if kind in _KEYLESS_TTS:
        return Connection(id=connection_id or kind, type=kind)
    return _connection(store, connection_id)


@router.get("")
def read_settings() -> dict:
    try:
        return get_settings_store().public()
    except SettingsFileError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("")
def write_settings(payload: dict = Body(...)) -> dict:
    store = get_settings_store()
    try:
        app = store.update(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors(include_url=False, include_context=False, include_input=False)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return store.public(app)


@router.post("/connections/{connection_id}/test")
def test_connection(connection_id: str) -> dict:
    store = get_settings_store()
    conn = _connection(store, connection_id)
    try:
        ok, message = _probe(conn, store)
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never 500s
        ok, message = False, f"{type(exc).__name__}: {exc}"
    return {"ok": ok, "message": message}


@router.get("/connections/{connection_id}/models")
def list_models(connection_id: str) -> dict:
    store = get_settings_store()
    conn = _connection(store, connection_id)
    if conn.type == "claude_code":
        return {"models": _CLAUDE_MODELS}
    if conn.type == "codex":
        return {"models": _codex_models(conn)}
    if conn.type not in api_llm.API_PROVIDERS:
        return {"models": []}  # other CLI agents take free-form model names
    try:
        return {"models": _api_models(conn, store)}
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/tts/voices")
def list_voices(connection: str = "", kind: str = Query("", alias="type")) -> dict:
    store = get_settings_store()
    conn = _tts_connection(store, connection, kind)
    try:
        return {"voices": engine_voices(conn, store.env)}
    except TtsEngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class TtsPreview(BaseModel):
    """An unsaved draft is previewable: voice/model come from the request, the connection
    (with its real secrets) from the store — or, for a keyless ``type``, from nothing."""

    connection: str = ""
    type: str = ""
    voice: str = ""
    model: str = ""
    text: str = ""


@router.post("/tts/preview")
def preview_tts(payload: TtsPreview) -> Response:
    store = get_settings_store()
    conn = _tts_connection(store, payload.connection, payload.type)
    text = payload.text.strip()[:500] or _PREVIEW_TEXT
    try:
        engine = build_engine(TtsTask(connection=conn.id, voice=payload.voice, model=payload.model), conn, store.env)
        audio = engine.synthesize_batch([text])[0]
    except RuntimeError as exc:  # TtsEngineError, CapCut and HTTP-server failures alike
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(content=audio, media_type=audio_media_type(audio))


def _api_models(conn: Connection, store: SettingsStore) -> list[str]:
    return api_llm.list_models(replace(agent_for(conn, "", store.env), timeout=_PROBE_TIMEOUT))


def _codex_models(conn: Connection) -> list[str]:
    """Suggestions from the model list the Codex CLI itself caches (none if absent)."""
    home = Path(os.path.expanduser(conn.home)) / ".codex" if conn.home else Path(os.environ.get("CODEX_HOME") or "~/.codex").expanduser()
    try:
        cache = json.loads((home / "models_cache.json").read_text(encoding="utf-8"))
        slugs = [str(model.get("slug") or "") for model in cache.get("models", [])]
    except (OSError, ValueError, AttributeError):
        return []
    return [slug for slug in slugs if slug and "review" not in slug]


def _probe(conn: Connection, store: SettingsStore) -> tuple[bool, str]:
    """Cheap reachability check — never synthesizes audio (vieneu: loads the local model)."""
    if conn.type in api_llm.API_PROVIDERS:
        # listing models proves the endpoint answers AND the key is accepted
        try:
            models = _api_models(conn, store)
        except AgentError as exc:
            return False, str(exc)
        return True, f"kết nối được — {len(models)} model"
    if conn.type in CLI_BINS:
        binary = conn.bin or CLI_BINS[conn.type]
        found = shutil.which(binary)
        return (True, f"{binary}: {found}") if found else (False, f"không tìm thấy lệnh {binary}")
    if conn.type == "command":
        if "{prompt}" not in conn.command:
            return False, "lệnh phải chứa {prompt}"
        program = shlex.split(conn.command)[0]
        found = shutil.which(program)
        return (True, f"{program}: {found}") if found else (False, f"không tìm thấy lệnh {program}")
    if conn.type == "capcut":
        env = store.env
        session = Path(env.capcut_session_file) if env.capcut_session_file else env.private_dir / "capcut" / "capcut-session.json"
        if session.exists():
            return True, "có phiên CapCut"
        if conn.email and conn.password:
            return True, "chưa có phiên — sẽ đăng nhập bằng email/mật khẩu khi chạy"
        if conn.base_url:
            return _probe_http(conn.base_url)
        return False, "chưa có phiên, email/mật khẩu hay base_url dự phòng"
    if conn.type == "http_tts":
        if not conn.base_url:
            return False, "thiếu base_url"
        return _probe_http(conn.base_url)
    if conn.type == "vieneu":
        try:
            load_vieneu(store.env)
        except TtsEngineError as exc:
            return False, str(exc)
        return True, "đã tải model VieNeu trên máy này"
    if conn.type in ("edge", "elevenlabs"):
        # listing voices proves the package/endpoint works and (elevenlabs) the key is accepted
        try:
            voices = engine_voices(conn, store.env)
        except TtsEngineError as exc:
            return False, str(exc)
        return True, f"kết nối được — {len(voices)} giọng"
    return False, f"loại kết nối {conn.type} chưa được hỗ trợ ở bản này"


def _probe_http(url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(url, timeout=5.0)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        return False, f"{url}: {exc}"
    return True, f"{url}: HTTP {response.status_code}"


@router.get("/fonts")
def fonts() -> dict:
    """Caption fonts the hardsub can burn (Vietnamese-capable only) + the default."""
    from backend.app.services.fonts import default_font, list_fonts

    settings = get_settings()
    return {"default": str(default_font(settings)), "fonts": list_fonts(settings)}


@router.get("/images")
def overlay_images() -> dict:
    """The user's images the overlay editor can stamp on a render (served under /media)."""
    from backend.app.services.overlay_images import list_images

    return {"base": "/media/overlay_images", "images": list_images(get_settings())}


@router.post("/images")
async def upload_overlay_image(file: UploadFile = File(...)) -> dict:
    """Add an image (PNG/JPG/WebP…) to the library; stored as a PNG, alpha kept."""
    from backend.app.services.overlay_images import MAX_UPLOAD_BYTES, ImageUploadError, save_image

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        return save_image(get_settings(), file.filename or "", data)
    except ImageUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/fonts/file")
def font_file(path: str) -> Response:
    """One offered font's file, so the overlay editor previews the caption in it."""
    from backend.app.services.fonts import default_font, resolve_font

    settings = get_settings()
    font = resolve_font(settings, path)
    if font is None and path == str(default_font(settings)) and Path(path).is_file():
        font = Path(path)
    if font is None:
        raise HTTPException(status_code=404, detail="không có font này")
    media = "font/otf" if font.suffix.lower() == ".otf" else "font/ttf"
    return Response(content=font.read_bytes(), media_type=media, headers={"Cache-Control": "max-age=86400"})
