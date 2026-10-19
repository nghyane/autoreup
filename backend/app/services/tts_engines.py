"""TTS engines — one seam for every voice provider: ``synthesize_batch(texts) -> [audio]``.

``render()`` only sees this protocol, so the fit loop / timeline never care which service
spoke. ``cache_id`` keys the raw-clip cache together with voice + text: CapCut keeps
``capcut-native`` and the HTTP server keeps its ``base_url`` so existing caches stay valid;
the API engines use ``<type>:<model>``.

Errors are loud (``TtsEngineError``) — the only fallback is CapCut → its HTTP ``base_url``
when no CapCut session can be minted, as before. API keys travel in headers only and are
redacted from every error message (errors land in served job manifests).
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import shutil
import threading
import time
import wave
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import httpx

from backend.app.agents.api_llm import GEMINI_ROOT, _error_message, _retry_after
from backend.app.core.config import Settings
from backend.app.services.tts import HttpTtsProvider
from backend.app.settings.schema import Connection, TtsTask

ELEVENLABS_ROOT = "https://api.elevenlabs.io"
ELEVENLABS_DEFAULT_MODEL = "eleven_flash_v2_5"
# Engine defaults when the task leaves the voice empty. ElevenLabs voices belong to the
# account, so it has none: the user must pick one.
DEFAULT_VOICES = {
    "capcut": "BV421_vivn_streaming",
    "http_tts": "BV421_vivn_streaming",
    "edge": "vi-VN-HoaiMyNeural",
    "openai_compat": "nova",
    "gemini": "Kore",
    "vieneu": "Thiện Minh",
}
# CapCut's "Tiếng Việt" tone panel (``/artist/v1/panel/get_panel_info``): speaker →
# (title, effect_id). multi_platform needs the matching effect_id/resource_id (the same
# id) next to the speaker. The ElevenLabs-backed ones are flagged Pro by CapCut.
CAPCUT_CATALOG = {
    "BV421_vivn_streaming": ("Nguồn nhỏ ngọt ngào", "7252594014782755330"),
    "vi_female_huong": ("Giọng nữ phổ thông", "7264854897953083905"),
    "BV074_streaming": ("Giọng nữ dễ thương", "7102355709945188865"),
    "BV075_streaming": ("Giọng nam tự tin", "7102355803792740865"),
    "BV560_streaming": ("Anh Dũng", "7483736167565758992"),
    "BV562_streaming": ("Chí Mai", "7483736254694035984"),
    "pGapy9MNHCukzJtjavF0": ("Hà Nữ", "7491607926562361857"),
    "xPEfmymXC4WdBxGMznS7": ("Sài Nữ", "7491607853086544400"),
    "1d5Bb0SMBPB10Gx6iQeu": ("Duy Bắc", "7491608109043945985"),
    "9EE00wK5qV6tPtpQIxvy": ("Nam Trầm", "7493478233703191041"),
    "7hsfEc7irDn6E8br0qfw": ("Ly Nam", "7491608014181372417"),
    "ueSxRO0nLF1bj93J2hVt": ("Trung Caha", "7491607775621943824"),
}
_CAPCUT_PRO = {"pGapy9MNHCukzJtjavF0", "xPEfmymXC4WdBxGMznS7", "1d5Bb0SMBPB10Gx6iQeu",
               "9EE00wK5qV6tPtpQIxvy", "7hsfEc7irDn6E8br0qfw", "ueSxRO0nLF1bj93J2hVt"}
CAPCUT_VOICES = [
    {"id": speaker, "name": f"{title} (Pro)" if speaker in _CAPCUT_PRO else title}
    for speaker, (title, _effect) in CAPCUT_CATALOG.items()
]
OPENAI_VOICES = [{"id": v, "name": v} for v in (
    "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
)]
GEMINI_VOICES = [{"id": v, "name": f"{v} — {style}"} for v, style in (
    ("Zephyr", "Bright"), ("Puck", "Upbeat"), ("Charon", "Informative"), ("Kore", "Firm"),
    ("Fenrir", "Excitable"), ("Leda", "Youthful"), ("Orus", "Firm"), ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"), ("Autonoe", "Bright"), ("Enceladus", "Breathy"), ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"), ("Algieba", "Smooth"), ("Despina", "Smooth"), ("Erinome", "Clear"),
    ("Algenib", "Gravelly"), ("Rasalgethi", "Informative"), ("Laomedeia", "Upbeat"), ("Achernar", "Soft"),
    ("Alnilam", "Firm"), ("Schedar", "Even"), ("Gacrux", "Mature"), ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"), ("Zubenelgenubi", "Casual"), ("Vindemiatrix", "Gentle"), ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"), ("Sulafat", "Warm"),
)]

# VieNeu-TTS v3 Turbo presets (vieneu==3.8.3 ``list_preset_voices()``): listing them must
# not load the 1.2 GB model. ⭐ = the voices VieNeu itself recommends.
VIENEU_VOICES = [{"id": v, "name": f"{v} — {desc}"} for v, desc in (
    ("Thiện Minh", "Nam · Bắc · Kể chuyện ⭐"), ("Adam bựa", "Nam · Bắc · Tự nhiên ⭐"),
    ("Trúc Ly", "Nữ · Bắc · Tự nhiên ⭐"), ("Mai Anh", "Nữ · Bắc · Tin tức ⭐"),
    ("Hải Đăng", "Nam · Bắc · Tự nhiên ⭐"), ("Thùy Dung", "Nữ · Nam · Tin tức ⭐"),
    ("Thiền Tâm Đức", "Nam · Bắc · Kể chuyện ⭐"), ("Ngọc Huyền", "Nữ · Bắc · Tự nhiên ⭐"),
    ("Quang Sơn", "Nam · Trung · Tự nhiên ⭐"), ("Ngọc Trân", "Nữ · Trung · Tự nhiên ⭐"),
    ("Minh Đức", "Nam · Bắc · Tin tức"), ("Phạm Tuyên", "Nam · Bắc · Tự nhiên"),
    ("Thái Sơn", "Nam · Nam · Kể chuyện"), ("Xuân Vĩnh", "Nam · Bắc · Tự nhiên"),
    ("Thanh Bình", "Nam · Bắc · Kể chuyện"), ("Ngọc Linh", "Nữ · Bắc · Kể chuyện"),
    ("Đoan Trang", "Nữ · Bắc · Tự nhiên"), ("Thục Đoan", "Nữ · Nam · Kể chuyện"),
    ("Minh Triết", "Nam · Nam · Tin tức"), ("Mỹ Duyên", "Nữ · Nam · Đọc truyện"),
    ("Quỳnh Anh", "Nữ · Bắc · Đọc truyện"), ("Đức Trí", "Nam · Nam · Đọc truyện"),
    ("Kim Thanh", "Nữ · Nam · Đọc truyện"), ("Adam", "Nam · Nam · Tự nhiên"),
    ("Quốc Tuấn", "Nam · Bắc · Tự nhiên"),
)]
_VIENEU_SILENCE_SECONDS = 0.3  # a symbols-only cue ("♪", "*") — VieNeu has nothing to say

_TIMEOUT = 60.0     # one sentence of speech; Gemini TTS can take several seconds
_ATTEMPTS = 3
_REMOTE_WORKERS = 4  # cap parallel calls to third-party APIs (rate limits, unofficial edge)
_GEMINI_PCM_RATE = 24000


class TtsEngineError(RuntimeError):
    """``partial`` (same order as the texts) holds the clips that DID synthesize when a
    batch fails part-way, so the caller can cache what was already paid for."""

    def __init__(self, message: str, partial: list[bytes | None] | None = None):
        super().__init__(message)
        self.partial = partial or []


class TtsEngine(Protocol):
    cache_id: str
    voice: str

    def synthesize_batch(self, texts: list[str]) -> list[bytes]: ...


def _client(timeout: float) -> httpx.Client:
    """Factory seam: tests swap in an ``httpx.MockTransport``."""
    return httpx.Client(timeout=timeout)


def _redact(text: str, secret: str) -> str:
    for value in {secret, secret.strip()}:
        if value:
            text = text.replace(value, "***")
    return text


def _request(name: str, method: str, url: str, secret: str = "", **kwargs) -> httpx.Response:
    """HTTP call with retries on network errors, 5xx, 408 and 429; anything else fails at once."""
    last = ""
    for attempt in range(1, _ATTEMPTS + 1):
        wait = 0.7 * attempt
        try:
            with _client(_TIMEOUT) as client:
                response = client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeError) as exc:
            raise TtsEngineError(_redact(f"{name}: {type(exc).__name__}: {exc}", secret)) from exc
        else:
            status = response.status_code
            if status < 400:
                return response
            last = f"HTTP {status}: {_error_message(response)}"
            if status < 500 and status not in (408, 429):
                break
            wait = max(wait, min(30.0, _retry_after(response)))
        if attempt < _ATTEMPTS:
            time.sleep(wait)
    raise TtsEngineError(_redact(f"{name} {last}", secret))


def _parallel(fn: Callable[[str], bytes], texts: list[str], workers: int) -> list[bytes]:
    if not texts:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(texts)))) as pool:
        futures = [pool.submit(fn, text) for text in texts]
    results: list[bytes | None] = []
    failure: BaseException | None = None
    for future in futures:
        try:
            results.append(future.result())
        except Exception as exc:  # noqa: BLE001 - re-raised below with the partial results
            results.append(None)
            failure = failure or exc
    if failure is not None:
        raise TtsEngineError(str(failure), partial=results) from failure
    return results  # type: ignore[return-value]


def _audio(name: str, payload: bytes) -> bytes:
    if not payload:
        raise TtsEngineError(f"{name}: trả về audio rỗng")
    return payload


@dataclass
class CapCutEngine:
    """Native CapCut: one signed call per batch, batches run concurrently."""

    client: object
    voice: str
    batch_size: int = 8
    concurrency: int = 8
    rate: float = 1.0
    volume: int = 10
    cache_id: str = "capcut-native"

    def synthesize_batch(self, texts: list[str]) -> list[bytes]:
        from backend.app.services.capcut_tts import synthesize_many

        return synthesize_many(
            self.client, texts, batch_size=self.batch_size, concurrency=self.concurrency, rate=self.rate, volume=self.volume,
        )


@dataclass
class HttpEngine:
    """The self-hosted TTS server (``POST /v2/synthesize``), one request per text."""

    provider: HttpTtsProvider
    cache_id: str
    workers: int = 8

    @property
    def voice(self) -> str:
        return self.provider.speaker

    def synthesize_batch(self, texts: list[str]) -> list[bytes]:
        return _parallel(self.provider.synthesize, texts, self.workers)


@dataclass
class EdgeEngine:
    """Microsoft Edge read-aloud via the ``edge-tts`` package (unofficial endpoint, mp3)."""

    voice: str
    workers: int = _REMOTE_WORKERS
    cache_id: str = "edge"

    def __post_init__(self) -> None:
        self.voice = self.voice.strip() or DEFAULT_VOICES["edge"]

    def synthesize_batch(self, texts: list[str]) -> list[bytes]:
        edge_tts = _edge_module()
        return _parallel(lambda text: self._one(edge_tts, text), texts, self.workers)

    def _one(self, edge_tts, text: str) -> bytes:
        async def collect() -> bytes:
            audio = bytearray()
            async for chunk in edge_tts.Communicate(text, self.voice).stream():
                if chunk.get("type") == "audio":
                    audio.extend(chunk.get("data") or b"")
            return bytes(audio)

        last: Exception | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                return _audio("edge", asyncio.run(collect()))
            except Exception as exc:  # noqa: BLE001 - edge_tts raises many unrelated types
                last = exc
            if attempt < _ATTEMPTS:
                time.sleep(0.7 * attempt)
        raise TtsEngineError(f"edge: {type(last).__name__}: {last}")


def _edge_module():
    try:
        import edge_tts
    except ImportError as exc:
        raise TtsEngineError("edge: thiếu package edge-tts (pip install -r requirements.txt)") from exc
    return edge_tts


@dataclass
class OpenAiTtsEngine:
    """OpenAI-compatible ``POST {base_url}/audio/speech`` → mp3."""

    base_url: str
    api_key: str
    model: str
    voice: str
    workers: int = _REMOTE_WORKERS

    @property
    def cache_id(self) -> str:
        return f"openai_compat:{self.model}"

    def synthesize_batch(self, texts: list[str]) -> list[bytes]:
        return _parallel(self._one, texts, self.workers)

    def _one(self, text: str) -> bytes:
        key = self.api_key.strip()
        response = _request(
            "openai_compat", "POST", f"{self.base_url.strip().rstrip('/')}/audio/speech", key,
            headers={"Authorization": f"Bearer {key}"} if key else {},
            json={"model": self.model, "input": text, "voice": self.voice, "response_format": "mp3"},
        )
        return _audio("openai_compat", response.content)


@dataclass
class GeminiTtsEngine:
    """Gemini native TTS: ``generateContent`` with an AUDIO modality → raw PCM, wrapped as WAV."""

    api_key: str
    model: str
    voice: str
    base_url: str = ""
    workers: int = _REMOTE_WORKERS

    def __post_init__(self) -> None:
        self.model = self.model.strip().removeprefix("models/")
        self.voice = self.voice.strip() or DEFAULT_VOICES["gemini"]

    @property
    def cache_id(self) -> str:
        return f"gemini:{self.model}"

    def synthesize_batch(self, texts: list[str]) -> list[bytes]:
        return _parallel(self._one, texts, self.workers)

    def _one(self, text: str) -> bytes:
        key = self.api_key.strip()
        root = self.base_url.strip().rstrip("/") or GEMINI_ROOT
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}},
            },
        }
        response = _request("gemini", "POST", f"{root}/models/{quote(self.model, safe='')}:generateContent", key,
                            headers={"x-goog-api-key": key}, json=body)
        try:
            data = response.json()
        except ValueError as exc:
            raise TtsEngineError("gemini: phản hồi không phải JSON") from exc
        candidates = (data.get("candidates") if isinstance(data, dict) else None) or [{}]
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = (content or {}).get("parts") or []
        pcm = bytearray()
        rate = _GEMINI_PCM_RATE
        for part in parts:
            inline = (part.get("inlineData") or part.get("inline_data")) if isinstance(part, dict) else None
            if not inline or not inline.get("data"):
                continue
            try:
                pcm.extend(base64.b64decode(inline["data"]))
            except ValueError as exc:
                raise TtsEngineError("gemini: audio base64 không hợp lệ") from exc
            found = re.search(r"rate=(\d+)", str(inline.get("mimeType") or inline.get("mime_type") or ""))
            rate = int(found.group(1)) if found else rate
        if not pcm:
            raise TtsEngineError("gemini: phản hồi không có audio (model có hỗ trợ TTS không?)")
        return pcm_to_wav(bytes(pcm), rate)


def pcm_to_wav(pcm: bytes, rate: int = _GEMINI_PCM_RATE) -> bytes:
    """Wrap raw s16le mono PCM in a WAV header so ffmpeg can read it."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


@dataclass
class ElevenLabsEngine:
    """ElevenLabs ``POST /v1/text-to-speech/{voice}`` → mp3."""

    api_key: str
    model: str
    voice: str
    base_url: str = ""
    workers: int = _REMOTE_WORKERS

    def __post_init__(self) -> None:
        self.model = self.model.strip() or ELEVENLABS_DEFAULT_MODEL

    @property
    def cache_id(self) -> str:
        return f"elevenlabs:{self.model}"

    def synthesize_batch(self, texts: list[str]) -> list[bytes]:
        return _parallel(self._one, texts, self.workers)

    def _one(self, text: str) -> bytes:
        key = self.api_key.strip()
        root = self.base_url.strip().rstrip("/") or ELEVENLABS_ROOT
        response = _request(
            "elevenlabs", "POST", f"{root}/v1/text-to-speech/{quote(self.voice, safe='')}", key,
            headers={"xi-api-key": key}, params={"output_format": "mp3_44100_128"},
            json={"text": text, "model_id": self.model},
        )
        return _audio("elevenlabs", response.content)


class _MaterializedFetch:
    """Wraps VieNeu's artifact fetch so it returns a real folder of hardlinks instead of a HF
    snapshot folder: onnxruntime ≥1.24 refuses an ``.onnx`` whose external ``.data`` resolves
    outside the model's folder, and the HF hub's shared blob store gives every file its own
    folder. Hardlinks cost no disk; ``copy2`` only when ``root`` sits on another volume."""

    def __init__(self, original: Callable[..., Path], root: Path):
        self.original = original
        self.root = root

    def __call__(self, repo: str, files: list[str], subfolder: str | None) -> Path:
        snapshot = Path(self.original(repo, files, subfolder))
        target = self.root / repo.replace("/", "--") / (subfolder or "")
        target.mkdir(parents=True, exist_ok=True)
        for source in snapshot.iterdir():
            if not source.is_file():
                continue
            blob, link = source.resolve(), target / source.name
            if link.exists() and (os.path.samefile(link, blob) or _same_copy(link, blob)):
                continue
            link.unlink(missing_ok=True)  # the hub moved on to a newer blob
            try:
                os.link(blob, link)
            except OSError:
                shutil.copy2(blob, link)
        return target


def _same_copy(a: Path, b: Path) -> bool:
    sa, sb = a.stat(), b.stat()
    return sa.st_size == sb.st_size and sa.st_mtime_ns == sb.st_mtime_ns


_VIENEU_LOCK = threading.Lock()
_vieneu: object | None = None


def load_vieneu(env: Settings) -> None:
    """Load (first time: download) the local VieNeu model — the settings "Kiểm tra" button."""
    _vieneu_model(_vieneu_dir(env))


def _vieneu_model(models_dir: Path):
    """VieNeu-TTS v3 Turbo on ONNX/CPU, loaded once per process (~6 s, ~1.2 GB RAM)."""
    global _vieneu
    with _VIENEU_LOCK:
        if _vieneu is None:
            try:
                from vieneu import Vieneu
                from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            except ImportError as exc:
                raise TtsEngineError("vieneu: thiếu package vieneu (pip install -r requirements.txt)") from exc
            fetch = getattr(OnnxV3LiteEngine.__dict__.get("_fetch"), "__func__", None)
            if fetch is None:
                raise TtsEngineError("vieneu: phiên bản vieneu không hỗ trợ (cần vieneu==3.8.3 trong requirements.txt)")
            if not isinstance(fetch, _MaterializedFetch):
                OnnxV3LiteEngine._fetch = staticmethod(_MaterializedFetch(fetch, models_dir))
            try:
                _vieneu = Vieneu(backend="onnx")
            except Exception as exc:  # noqa: BLE001 - download / onnxruntime errors of every kind
                raise TtsEngineError(f"vieneu: không tải được model: {type(exc).__name__}: {exc}") from exc
        return _vieneu


@dataclass
class VieNeuEngine:
    """VieNeu-TTS on this machine → 48 kHz WAV. Clips run one after another: the model
    serializes inference itself and already spreads one clip over the CPU cores."""

    voice: str
    models_dir: Path
    cache_id: str = "vieneu:v3turbo"

    def __post_init__(self) -> None:
        self.voice = self.voice.strip() or DEFAULT_VOICES["vieneu"]

    def synthesize_batch(self, texts: list[str]) -> list[bytes]:
        if not texts:
            return []
        import numpy as np

        model = _vieneu_model(self.models_dir)
        clips: list[bytes | None] = []
        for text in texts:
            if not any(ch.isalnum() for ch in text):
                clips.append(pcm_to_wav(b"\0\0" * int(model.sample_rate * _VIENEU_SILENCE_SECONDS), model.sample_rate))
                continue
            try:
                # numpy on macOS Accelerate flags spurious matmul overflow/invalid warnings
                # inside VieNeu's sampler — the audio is fine; keep them out of job logs.
                with np.errstate(all="ignore"):
                    audio = np.asarray(model.infer(text, voice=self.voice), dtype=np.float32).reshape(-1)
                pcm = _audio("vieneu", (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
            except Exception as exc:  # noqa: BLE001 - re-raised with the clips already made
                message = f"{exc} — câu: {text[:80]!r}" if isinstance(exc, TtsEngineError) else f"vieneu: {type(exc).__name__}: {exc} — câu: {text[:80]!r}"
                raise TtsEngineError(message, partial=clips + [None] * (len(texts) - len(clips))) from exc
            clips.append(pcm_to_wav(pcm, model.sample_rate))
        return clips  # type: ignore[return-value]


def _vieneu_dir(env: Settings) -> Path:
    return env.data_dir / "models" / "vieneu"


def build_capcut_client(settings: Settings, conn: Connection, voice: str):
    """Native in-process CapCut TTS client, or None to fall back to the HTTP server.

    Self-contained: mints/refreshes its own CapCut session via email/password login
    (pure Python) when the session file is missing or stale, then signs + batches
    synthesis itself. Returns None when no session can be obtained (then the pipeline
    uses the connection's HTTP ``base_url`` as before)."""
    session_path = Path(settings.capcut_session_file) if settings.capcut_session_file else settings.private_dir / "capcut" / "capcut-session.json"

    stale = (not session_path.exists()) or (
        time.time() - session_path.stat().st_mtime > settings.capcut_session_max_age_days * 86400
    )
    if stale and conn.email and conn.password:
        try:
            from backend.app.services.capcut_login import LoginConfig, login

            login(LoginConfig(
                email=conn.email,
                password=conn.password,
                session_file=session_path,
                app_id=settings.capcut_app_id,
                web_url=settings.capcut_web_url,
                login_host=settings.capcut_login_host,
                fallback_login_host=settings.capcut_fallback_login_host,
                page_locale=settings.capcut_page_locale,
                locale=settings.capcut_locale,
                store_country=settings.capcut_store_country,
                user_agent=settings.capcut_user_agent,
            ))
        except Exception:
            pass  # fall back to an existing (possibly stale) session if present

    if not session_path.exists():
        return None
    from backend.app.services.capcut_tts import CapCutConfig, CapCutTtsClient

    # A catalog voice carries its own effect id; an unknown speaker keeps the env defaults.
    entry = CAPCUT_CATALOG.get(voice)
    voice_name, effect_id, resource_id = (entry[0], entry[1], entry[1]) if entry else (
        settings.capcut_voice_name, settings.capcut_effect_id, settings.capcut_resource_id)
    try:
        return CapCutTtsClient(CapCutConfig(
            session_file=session_path,
            edit_api=settings.capcut_edit_api,
            web_url=settings.capcut_web_url,
            app_id=settings.capcut_app_id,
            app_version=settings.capcut_app_version,
            platform_id=settings.capcut_platform_id,
            sign_version=settings.capcut_sign_version,
            sign_prefix=settings.capcut_sign_prefix,
            sign_suffix=settings.capcut_sign_suffix,
            sign_path_tail=settings.capcut_sign_path_tail,
            store_country=settings.capcut_store_country,
            user_agent=settings.capcut_user_agent,
            speaker=voice,
            effect_id=effect_id,
            resource_id=resource_id,
            voice_name=voice_name,
        ))
    except Exception:
        return None


def _http_engine(conn: Connection, voice: str, env: Settings) -> HttpEngine:
    return HttpEngine(HttpTtsProvider(conn.base_url, speaker=voice), cache_id=conn.base_url, workers=env.tts_workers)


def build_engine(
    task: TtsTask,
    conn: Connection | None,
    env: Settings,
    capcut_client: Callable[[Settings, Connection, str], object | None] = build_capcut_client,
) -> TtsEngine:
    """The engine for ``tasks.tts``. CapCut synthesizes natively and falls back to its HTTP
    ``base_url`` when no session can be minted; every other engine fails loudly when it is
    missing what it needs."""
    if conn is None:
        raise TtsEngineError("TTS: chưa chọn kết nối giọng đọc trong Cài đặt → Giọng đọc.")
    voice = task.voice.strip() or DEFAULT_VOICES.get(conn.type, "")
    model = task.model.strip()
    remote = min(env.tts_workers, _REMOTE_WORKERS)
    if conn.type == "capcut":
        client = capcut_client(env, conn, voice)
        if client is not None:
            return CapCutEngine(client, voice, batch_size=env.capcut_batch_size,
                                concurrency=env.capcut_batch_concurrency, rate=task.synth_rate)
        if conn.base_url:
            return _http_engine(conn, voice, env)
        raise TtsEngineError("TTS CapCut: no session (set email/password on the connection) and no HTTP fallback base_url.")
    if conn.type == "http_tts":
        if not conn.base_url:
            raise TtsEngineError("Missing TTS server URL. Set base_url on the TTS connection (or AUTOREUP_TTS_SERVER_URL).")
        return _http_engine(conn, voice, env)
    if conn.type == "edge":
        return EdgeEngine(voice, workers=remote)
    if conn.type == "vieneu":
        return VieNeuEngine(voice, _vieneu_dir(env))
    if conn.type == "openai_compat":
        if not conn.base_url.strip():
            raise TtsEngineError(f"TTS {conn.id}: thiếu base_url")
        if not model:
            raise TtsEngineError(f"TTS {conn.id}: chưa chọn model giọng đọc")
        return OpenAiTtsEngine(conn.base_url, conn.api_key, model, voice, workers=remote)
    if conn.type == "gemini":
        if not conn.api_key.strip():
            raise TtsEngineError(f"TTS {conn.id}: thiếu api_key")
        if not model:
            raise TtsEngineError(f"TTS {conn.id}: chưa chọn model giọng đọc (vd. model *-tts)")
        return GeminiTtsEngine(conn.api_key, model, voice, base_url=conn.base_url, workers=remote)
    if conn.type == "elevenlabs":
        if not conn.api_key.strip():
            raise TtsEngineError(f"TTS {conn.id}: thiếu api_key")
        if not voice:
            raise TtsEngineError(f"TTS {conn.id}: ElevenLabs cần chọn giọng (voice id của tài khoản)")
        return ElevenLabsEngine(conn.api_key, model, voice, base_url=conn.base_url, workers=remote)
    raise TtsEngineError(f"TTS: kết nối {conn.id} ({conn.type}) không đọc được giọng")


def list_voices(conn: Connection, env: Settings) -> list[dict]:
    """Voices a connection offers as ``[{id, name}]`` (edge: Vietnamese only)."""
    del env  # reserved for engines whose listing needs infra settings
    if conn.type in ("capcut", "http_tts"):
        return CAPCUT_VOICES
    if conn.type == "openai_compat":
        return OPENAI_VOICES
    if conn.type == "gemini":
        return GEMINI_VOICES
    if conn.type == "vieneu":
        return VIENEU_VOICES
    if conn.type == "edge":
        edge_tts = _edge_module()
        try:
            voices = asyncio.run(edge_tts.list_voices())
        except Exception as exc:  # noqa: BLE001
            raise TtsEngineError(f"edge: {type(exc).__name__}: {exc}") from exc
        return [
            {"id": v["ShortName"], "name": v.get("FriendlyName") or v["ShortName"]}
            for v in voices
            if str(v.get("Locale", "")).lower().startswith("vi")
        ]
    if conn.type == "elevenlabs":
        key = conn.api_key.strip()
        if not key:
            raise TtsEngineError(f"TTS {conn.id}: thiếu api_key")
        root = conn.base_url.strip().rstrip("/") or ELEVENLABS_ROOT
        try:
            voices = _request("elevenlabs", "GET", f"{root}/v1/voices", key, headers={"xi-api-key": key}).json()["voices"]
            return [{"id": v["voice_id"], "name": v.get("name") or v["voice_id"]} for v in voices if v.get("voice_id")]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise TtsEngineError("elevenlabs: danh sách giọng không hợp lệ") from exc
    return []


def audio_media_type(audio: bytes) -> str:
    if audio.startswith(b"RIFF"):
        return "audio/wav"
    if audio.startswith(b"OggS"):
        return "audio/ogg"
    return "audio/mpeg"


__all__ = [
    "CAPCUT_CATALOG",
    "CAPCUT_VOICES",
    "DEFAULT_VOICES",
    "CapCutEngine",
    "EdgeEngine",
    "ElevenLabsEngine",
    "GeminiTtsEngine",
    "HttpEngine",
    "OpenAiTtsEngine",
    "TtsEngine",
    "TtsEngineError",
    "VIENEU_VOICES",
    "VieNeuEngine",
    "audio_media_type",
    "build_capcut_client",
    "build_engine",
    "list_voices",
    "load_vieneu",
    "pcm_to_wav",
]
