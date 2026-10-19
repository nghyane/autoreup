"""TTS engines: wire format per provider, Gemini PCM→WAV, edge via edge_tts, build_engine
routing (capcut→http fallback), render caching through an engine, voice list + preview API."""

import base64
import io
import json
import subprocess
import sys
import tempfile
import types
import wave
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import backend.app.api.v1.endpoints.settings as settings_endpoint
import backend.app.services.tts_engines as engines
from backend.app.core.config import Settings
from backend.app.dubbing.models import SpeechRow
from backend.app.dubbing.render import RenderConfig, render
from backend.app.main import create_app
from backend.app.settings import Connection, SettingsStore, TtsTask


def _mock(monkeypatch, handler):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(engines, "_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(record), timeout=timeout))
    monkeypatch.setattr(engines.time, "sleep", lambda _s: None)
    return seen


def _env(root: Path) -> Settings:
    env = Settings(root_dir=root, _env_file=None)
    env.ensure_directories()
    return env


def _tone(seconds: float = 0.6) -> bytes:
    """A real (non-silent) WAV clip, the shape ffmpeg sees from a synth."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-f", "wav", "-"],
        check=True, capture_output=True,
    ).stdout


# ---- HTTP engines: request shape -------------------------------------------------


def test_openai_engine_posts_audio_speech_with_bearer(monkeypatch):
    seen = _mock(monkeypatch, lambda r: httpx.Response(200, content=b"MP3DATA"))
    engine = engines.OpenAiTtsEngine(base_url="https://api.x/v1/", api_key=" sk-key-1234 ", model="tts-1", voice="nova")
    assert engine.synthesize_batch(["xin chào", "tạm biệt"]) == [b"MP3DATA", b"MP3DATA"]
    request = seen[0]
    assert str(request.url) == "https://api.x/v1/audio/speech"
    assert request.headers["authorization"] == "Bearer sk-key-1234"
    assert json.loads(request.content) == {"model": "tts-1", "input": "xin chào", "voice": "nova", "response_format": "mp3"}
    assert engine.cache_id == "openai_compat:tts-1"


def test_elevenlabs_engine_posts_voice_path_and_default_model(monkeypatch):
    seen = _mock(monkeypatch, lambda r: httpx.Response(200, content=b"MP3"))
    engine = engines.ElevenLabsEngine(api_key="el-key", model="", voice="VOICE123")
    assert engine.synthesize_batch(["chào"]) == [b"MP3"]
    request = seen[0]
    assert request.url.path == "/v1/text-to-speech/VOICE123"
    assert request.url.params["output_format"] == "mp3_44100_128"
    assert request.headers["xi-api-key"] == "el-key"
    assert json.loads(request.content) == {"text": "chào", "model_id": "eleven_flash_v2_5"}
    assert engine.cache_id == "elevenlabs:eleven_flash_v2_5"


def test_gemini_engine_wraps_pcm_in_a_valid_wav(monkeypatch):
    pcm = (b"\x01\x00\xff\xff" * 2400)  # 4800 s16le frames
    reply = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm).decode()}}]}}]}
    seen = _mock(monkeypatch, lambda r: httpx.Response(200, json=reply))
    engine = engines.GeminiTtsEngine(api_key="AIza-k", model="models/gemini-tts", voice="")
    audio = engine.synthesize_batch(["xin chào"])[0]
    with wave.open(io.BytesIO(audio)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        assert wav.readframes(wav.getnframes()) == pcm
    request = seen[0]
    assert str(request.url) == f"{engines.GEMINI_ROOT}/models/gemini-tts:generateContent"
    assert request.headers["x-goog-api-key"] == "AIza-k"
    body = json.loads(request.content)
    assert body["contents"][0]["parts"][0]["text"] == "xin chào"
    assert body["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
    assert engine.cache_id == "gemini:gemini-tts"


def test_http_errors_are_clear_redacted_and_5xx_retried(monkeypatch):
    _mock(monkeypatch, lambda r: httpx.Response(401, json={"error": {"message": "bad key sk-key-1234"}}))
    engine = engines.OpenAiTtsEngine(base_url="https://api.x/v1", api_key="sk-key-1234", model="tts-1", voice="nova")
    with pytest.raises(engines.TtsEngineError) as err:
        engine.synthesize_batch(["a"])
    assert "401" in str(err.value) and "sk-key-1234" not in str(err.value)

    answers = iter([httpx.Response(503), httpx.Response(200, content=b"OK")])
    seen = _mock(monkeypatch, lambda r: next(answers))
    assert engine.synthesize_batch(["a"]) == [b"OK"]
    assert len(seen) == 2


def test_gemini_without_audio_part_is_an_error(monkeypatch):
    _mock(monkeypatch, lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "no"}]}}]}))
    with pytest.raises(engines.TtsEngineError):
        engines.GeminiTtsEngine(api_key="k", model="m", voice="Kore").synthesize_batch(["a"])


# ---- edge -------------------------------------------------------------------------


def _fake_edge(monkeypatch, calls: list):
    module = types.ModuleType("edge_tts")

    class Communicate:
        def __init__(self, text, voice, **_kw):
            calls.append((text, voice))

        async def stream(self):
            yield {"type": "audio", "data": b"AB"}
            yield {"type": "WordBoundary", "offset": 1}
            yield {"type": "audio", "data": b"CD"}

    async def list_voices(**_kw):
        return [
            {"ShortName": "vi-VN-HoaiMyNeural", "Locale": "vi-VN", "Gender": "Female", "FriendlyName": "HoaiMy"},
            {"ShortName": "en-US-AriaNeural", "Locale": "en-US", "Gender": "Female", "FriendlyName": "Aria"},
        ]

    module.Communicate = Communicate
    module.list_voices = list_voices
    monkeypatch.setitem(sys.modules, "edge_tts", module)


def test_edge_engine_collects_audio_chunks_with_default_voice(monkeypatch):
    calls: list = []
    _fake_edge(monkeypatch, calls)
    engine = engines.EdgeEngine(voice="")
    assert engine.synthesize_batch(["một", "hai"]) == [b"ABCD", b"ABCD"]
    assert sorted(calls) == [("hai", "vi-VN-HoaiMyNeural"), ("một", "vi-VN-HoaiMyNeural")]
    assert engine.cache_id == "edge"


def test_edge_engine_without_package_fails_loudly(monkeypatch):
    monkeypatch.setitem(sys.modules, "edge_tts", None)  # import raises ImportError
    with pytest.raises(engines.TtsEngineError, match="edge-tts"):
        engines.EdgeEngine(voice="x").synthesize_batch(["a"])


# ---- vieneu (local) ---------------------------------------------------------------


def _fake_vieneu(monkeypatch, calls: list, fail_on: str = ""):
    """``vieneu`` + its ONNX engine module, faked; the loaded-model singleton is reset."""
    import numpy as np

    class Model:
        sample_rate = 48_000

        def infer(self, text, voice=None):
            calls.append((text, voice))
            if text == fail_on:
                raise RuntimeError("boom")
            return np.full(480, 0.5, dtype=np.float32)

        def list_preset_voices(self):
            return [("⭐ Thiện Minh — Nam · Bắc · Phong cách kể chuyện", "Thiện Minh"), ("Ngọc Linh — Nữ · Bắc", "Ngọc Linh")]

    class OnnxV3LiteEngine:
        @staticmethod
        def _fetch(repo, files, subfolder):
            return Path("/nonexistent")

    package = types.ModuleType("vieneu")
    package.Vieneu = lambda **kw: calls.append(("load", kw)) or Model()
    lite = types.ModuleType("vieneu._v3_turbo_engine.onnx_runtime_lite")
    lite.OnnxV3LiteEngine = OnnxV3LiteEngine
    monkeypatch.setitem(sys.modules, "vieneu", package)
    monkeypatch.setitem(sys.modules, "vieneu._v3_turbo_engine", types.ModuleType("vieneu._v3_turbo_engine"))
    monkeypatch.setitem(sys.modules, "vieneu._v3_turbo_engine.onnx_runtime_lite", lite)
    monkeypatch.setattr(engines, "_vieneu", None)
    return OnnxV3LiteEngine


def test_vieneu_engine_returns_48k_wav_with_default_voice_and_loads_once(monkeypatch, tmp_path):
    calls: list = []
    lite = _fake_vieneu(monkeypatch, calls)
    engine = engines.VieNeuEngine(voice="", models_dir=tmp_path)
    clips = engine.synthesize_batch(["một", "hai"])
    engine.synthesize_batch(["ba"])
    assert engine.voice == "Thiện Minh" and engine.cache_id == "vieneu:v3turbo"
    assert [c for c in calls if c[0] == "load"] == [("load", {"backend": "onnx"})]
    assert [c for c in calls if c[0] != "load"] == [("một", "Thiện Minh"), ("hai", "Thiện Minh"), ("ba", "Thiện Minh")]
    with wave.open(io.BytesIO(clips[0])) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (48_000, 1, 2, 480)
    fetch = lite.__dict__["_fetch"].__func__
    assert isinstance(fetch, engines._MaterializedFetch) and fetch.root == tmp_path


def test_vieneu_engine_keeps_the_clips_made_before_a_failure(monkeypatch, tmp_path):
    _fake_vieneu(monkeypatch, [], fail_on="hai")
    with pytest.raises(engines.TtsEngineError, match="vieneu: RuntimeError: boom") as err:
        engines.VieNeuEngine(voice="Ngọc Linh", models_dir=tmp_path).synthesize_batch(["một", "hai", "ba"])
    assert err.value.partial[0].startswith(b"RIFF") and err.value.partial[1:] == [None, None]


def test_vieneu_without_package_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vieneu", None)
    monkeypatch.setattr(engines, "_vieneu", None)
    with pytest.raises(engines.TtsEngineError, match="thiếu package vieneu"):
        engines.VieNeuEngine(voice="x", models_dir=tmp_path).synthesize_batch(["a"])


def test_vieneu_routes_and_lists_voices_without_loading_the_model(monkeypatch, tmp_path):
    calls: list = []
    _fake_vieneu(monkeypatch, calls)
    env = _env(tmp_path)
    conn = Connection(id="vn", type="vieneu")
    engine = engines.build_engine(TtsTask(), conn, env)
    assert isinstance(engine, engines.VieNeuEngine) and engine.voice == "Thiện Minh"
    assert engine.models_dir == env.data_dir / "models" / "vieneu"
    voices = engines.list_voices(conn, env)
    assert len(voices) == 25 and voices[0]["id"] == engines.DEFAULT_VOICES["vieneu"]
    assert {"Ngọc Linh", "Quỳnh Anh"} <= {v["id"] for v in voices}
    assert calls == []  # listing never loads the model


def test_vieneu_symbols_only_text_is_short_silence_not_an_error(monkeypatch, tmp_path):
    calls: list = []
    _fake_vieneu(monkeypatch, calls)
    clip, spoken = engines.VieNeuEngine(voice="", models_dir=tmp_path).synthesize_batch(["♪ ~ *", "Chào"])
    with wave.open(io.BytesIO(clip)) as wav:
        assert wav.getnframes() == int(48_000 * 0.3) and set(wav.readframes(wav.getnframes())) == {0}
    assert spoken.startswith(b"RIFF") and [c[0] for c in calls if c[0] != "load"] == ["Chào"]


def test_vieneu_model_load_failure_is_an_engine_error(monkeypatch, tmp_path):
    _fake_vieneu(monkeypatch, [])

    def broken(**_kw):
        raise OSError("disk full")

    monkeypatch.setattr(sys.modules["vieneu"], "Vieneu", broken)
    with pytest.raises(engines.TtsEngineError, match="không tải được model: OSError: disk full"):
        engines.load_vieneu(_env(tmp_path))
    assert engines._vieneu is None  # the next call retries


def test_materialized_fetch_hardlinks_split_blobs_into_one_folder(tmp_path):
    """HF shared blobs put model.onnx and its model.data in different folders; onnxruntime
    only accepts them side by side."""
    snapshot = tmp_path / "hub" / "snapshots" / "rev" / "onnx"
    snapshot.mkdir(parents=True)
    blobs = {}
    for name, shard in (("model.onnx", "aa"), ("model.data", "bb")):
        blob = tmp_path / "hub" / "blobs" / shard / name.replace(".", "_")
        blob.parent.mkdir(parents=True)
        blob.write_bytes(name.encode())
        (snapshot / name).symlink_to(blob)
        blobs[name] = blob
    seen: list = []
    fetch = engines._MaterializedFetch(lambda *args: seen.append(args) or snapshot, tmp_path / "models")

    folder = fetch("org/repo", ["model.onnx"], "onnx")
    assert folder == tmp_path / "models" / "org--repo" / "onnx" and seen == [("org/repo", ["model.onnx"], "onnx")]
    for name, blob in blobs.items():
        assert not (folder / name).is_symlink() and (folder / name).samefile(blob)

    newer = tmp_path / "hub" / "blobs" / "cc" / "model_data"
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b"v2")
    (snapshot / "model.data").unlink()
    (snapshot / "model.data").symlink_to(newer)
    assert fetch("org/repo", [], "onnx") == folder and (folder / "model.data").read_bytes() == b"v2"


# ---- build_engine -----------------------------------------------------------------


def test_build_engine_routes_by_type_and_keeps_legacy_cache_ids(tmp_path):
    env = _env(tmp_path)
    client = object()
    capcut = Connection(id="capcut", type="capcut", base_url="http://tts:8080")

    native = engines.build_engine(TtsTask(connection="capcut", voice=""), capcut, env, capcut_client=lambda *_: client)
    assert isinstance(native, engines.CapCutEngine) and native.cache_id == "capcut-native"
    assert native.client is client and native.voice == "BV421_vivn_streaming"

    fallback = engines.build_engine(TtsTask(voice="v"), capcut, env, capcut_client=lambda *_: None)
    assert isinstance(fallback, engines.HttpEngine) and fallback.cache_id == "http://tts:8080"
    with pytest.raises(engines.TtsEngineError):
        engines.build_engine(TtsTask(), Connection(id="capcut", type="capcut"), env, capcut_client=lambda *_: None)

    http = engines.build_engine(TtsTask(voice="v"), Connection(id="t", type="http_tts", base_url="http://x/"), env)
    assert isinstance(http, engines.HttpEngine) and http.cache_id == "http://x/"
    with pytest.raises(engines.TtsEngineError):
        engines.build_engine(TtsTask(), Connection(id="t", type="http_tts"), env)

    edge = engines.build_engine(TtsTask(), Connection(id="e", type="edge"), env)
    assert isinstance(edge, engines.EdgeEngine) and edge.voice == "vi-VN-HoaiMyNeural"

    gemini = engines.build_engine(TtsTask(model="g-tts"), Connection(id="g", type="gemini", api_key="k"), env)
    assert isinstance(gemini, engines.GeminiTtsEngine) and gemini.voice == "Kore"

    openai = engines.build_engine(TtsTask(model="tts-1"), Connection(id="o", type="openai_compat", base_url="http://o"), env)
    assert isinstance(openai, engines.OpenAiTtsEngine) and openai.voice == "nova"

    for task, conn in (
        (TtsTask(), None),
        (TtsTask(), Connection(id="o", type="openai_compat", base_url="http://o")),  # no model
        (TtsTask(model="m"), Connection(id="g", type="gemini")),  # no key
        (TtsTask(), Connection(id="el", type="elevenlabs", api_key="k")),  # no voice
        (TtsTask(), Connection(id="a", type="agy")),  # not a TTS connection
    ):
        with pytest.raises(engines.TtsEngineError):
            engines.build_engine(task, conn, env)


# ---- render through an engine ------------------------------------------------------


class _CountingEngine:
    cache_id = "fake:engine"
    voice = "v"

    def __init__(self):
        self.calls: list[list[str]] = []
        self.audio = _tone()

    def synthesize_batch(self, texts):
        self.calls.append(list(texts))
        return [self.audio for _ in texts]


def test_render_synthesizes_through_engine_and_reuses_its_cache():
    rows = [SpeechRow(0, 0, 0.0, 2.0, "0-0", "Xin chào các bạn"), SpeechRow(1, 1, 2.0, 4.0, "1-1", "Hẹn gặp lại")]
    engine = _CountingEngine()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = RenderConfig(provider_url="ignored-when-engine-given", speaker="v", workers=2)
        first = render(rows, config, root / "work", root / "cache", engine=engine)
        assert [len(c) for c in engine.calls] == [2]  # one batch for every missing clip
        assert len(first.timings) == 2 and all(t.tts_duration > 0.2 for t in first.timings)
        render(rows, config, root / "work", root / "cache", engine=engine)
        assert len(engine.calls) == 1  # second render: all clips cached, engine not called


def test_render_caches_the_clips_that_succeeded_before_failing():
    rows = [SpeechRow(0, 0, 0.0, 2.0, "0-0", "Một"), SpeechRow(1, 1, 2.0, 4.0, "1-1", "Hai")]
    tone = _tone()

    def flaky(text):
        if text == "Hai":
            raise engines.TtsEngineError("HTTP 429: quota")
        return tone

    class Flaky:
        cache_id = "fake:flaky"
        voice = "v"

        def synthesize_batch(self, texts):
            return engines._parallel(flaky, texts, 2)

    engine = _CountingEngine()
    engine.cache_id = "fake:flaky"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = RenderConfig(provider_url="x", workers=2)
        with pytest.raises(engines.TtsEngineError, match="429"):
            render(rows, config, root / "w", root / "c", engine=Flaky())
        render(rows, config, root / "w", root / "c", engine=engine)
    assert engine.calls == [["Hai"]]  # "Một" was cached by the failed run


def test_capcut_and_http_cache_keys_match_the_pre_engine_layout(tmp_path):
    """Existing caches stay valid: same provider_url + speaker as before engines existed."""
    from backend.app.services.tts import TtsBlock, raw_audio_path

    env = _env(tmp_path)
    block = TtsBlock("0", 0.0, 1.0, "xin chào")
    for conn, client, legacy_url in (
        (Connection(id="capcut", type="capcut"), object(), "capcut-native"),
        (Connection(id="t", type="http_tts", base_url="http://tts:8080"), None, "http://tts:8080"),
    ):
        engine = engines.build_engine(TtsTask(voice=env.tts_voice), conn, env, capcut_client=lambda *_a, c=client: c)
        args = dict(speed=1.0, pitch=1.0, volume=1.0, text_mode="expressive")
        assert raw_audio_path(tmp_path, engine.cache_id, engine.voice, block, **args) == raw_audio_path(
            tmp_path, legacy_url, env.tts_voice, block, **args
        )


def test_render_legacy_synth_client_still_goes_native():
    class Client:
        def __init__(self):
            self.batches = []
            self.audio = _tone()

        def synthesize_batch(self, texts, rate=1.0, volume=10.0):
            self.batches.append(list(texts))
            return [self.audio for _ in texts]

    client = Client()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        render([SpeechRow(0, 0, 0.0, 2.0, "0-0", "Một")], RenderConfig(provider_url="capcut-native"),
               root / "w", root / "c", synth_client=client)
    assert client.batches == [["Một"]]


# ---- voices ------------------------------------------------------------------------


def test_list_voices_per_engine(monkeypatch, tmp_path):
    env = _env(tmp_path)
    assert engines.list_voices(Connection(id="c", type="capcut"), env)[0]["id"] == "BV421_vivn_streaming"
    assert any(v["id"] == "Kore" for v in engines.list_voices(Connection(id="g", type="gemini"), env))
    assert any(v["id"] == "nova" for v in engines.list_voices(Connection(id="o", type="openai_compat"), env))

    _fake_edge(monkeypatch, [])
    edge = engines.list_voices(Connection(id="e", type="edge"), env)
    assert [v["id"] for v in edge] == ["vi-VN-HoaiMyNeural"]  # Vietnamese voices only

    capcut = engines.list_voices(Connection(id="c", type="capcut"), env)
    assert len(capcut) == len(engines.CAPCUT_CATALOG) and any(v["id"] == "BV560_streaming" for v in capcut)

    seen = _mock(monkeypatch, lambda r: httpx.Response(200, json={"voices": [{"voice_id": "abc", "name": "Linh"}]}))
    assert engines.list_voices(Connection(id="el", type="elevenlabs", api_key="el-k"), env) == [{"id": "abc", "name": "Linh"}]
    assert seen[0].url.path == "/v1/voices" and seen[0].headers["xi-api-key"] == "el-k"


def test_capcut_client_sends_the_chosen_voice_effect_id(tmp_path):
    """multi_platform rejects a speaker paired with another voice's effect id."""
    session = tmp_path / "capcut-session.json"
    session.write_text(json.dumps({"deviceId": "d", "tdid": "t", "cookies": [{"name": "sid", "value": "x", "domain": ".capcut.com"}]}))
    env = Settings(root_dir=tmp_path, capcut_session_file=str(session), _env_file=None)
    conn = Connection(id="capcut", type="capcut")

    picked = engines.build_capcut_client(env, conn, "BV560_streaming").config
    assert (picked.speaker, picked.voice_name) == ("BV560_streaming", "Anh Dũng")
    assert picked.effect_id == picked.resource_id == "7483736167565758992"

    unknown = engines.build_capcut_client(env, conn, "custom_speaker").config
    assert (unknown.effect_id, unknown.resource_id, unknown.voice_name) == (env.capcut_effect_id, env.capcut_resource_id, env.capcut_voice_name)


# ---- API ---------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path):
    store = SettingsStore(_env(tmp_path))
    with patch.object(settings_endpoint, "get_settings_store", return_value=store):
        yield TestClient(create_app()), store


def _add_connection(client, conn: dict) -> None:
    data = client.get("/api/v1/settings").json()
    data["connections"].append(conn)
    assert client.put("/api/v1/settings", json=data).status_code == 200


def test_voices_endpoint_lists_engine_voices(api):
    client, _store = api
    _add_connection(client, {"id": "gm", "type": "gemini", "api_key": "AIza-secret"})
    voices = client.get("/api/v1/settings/tts/voices", params={"connection": "gm"}).json()["voices"]
    assert any(v["id"] == "Kore" for v in voices)
    assert client.get("/api/v1/settings/tts/voices", params={"connection": "nope"}).status_code == 404


def test_preview_returns_audio_and_reports_engine_errors(api, monkeypatch):
    client, _store = api
    _add_connection(client, {"id": "gm", "type": "gemini", "api_key": "AIza-secret"})
    captured = {}

    class Fake:
        cache_id = "gemini:m"

        def synthesize_batch(self, texts):
            captured["texts"] = texts
            return [b"RIFF\x00\x00\x00\x00WAVEfmt "]

    def fake_build(task, conn, env, **_kw):
        captured["task"], captured["conn"] = task, conn
        return Fake()

    monkeypatch.setattr(settings_endpoint, "build_engine", fake_build)
    response = client.post("/api/v1/settings/tts/preview", json={"connection": "gm", "voice": "Puck", "model": "m", "text": "thử"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert captured["texts"] == ["thử"]
    assert (captured["task"].voice, captured["task"].model) == ("Puck", "m")
    assert captured["conn"].api_key == "AIza-secret"  # real key from the store, not the mask

    def failing(*_a, **_kw):
        raise engines.TtsEngineError("gemini HTTP 400: model sai")

    monkeypatch.setattr(settings_endpoint, "build_engine", failing)
    bad = client.post("/api/v1/settings/tts/preview", json={"connection": "gm"})
    assert bad.status_code == 502 and "model sai" in bad.json()["detail"]


def test_keyless_voice_services_list_and_preview_before_saving(api, monkeypatch):
    """VieNeu / Edge have nothing to configure: the Voice tab works on the unsaved draft."""
    client, _store = api
    voices = client.get("/api/v1/settings/tts/voices", params={"connection": "c-new", "type": "vieneu"}).json()["voices"]
    assert len(voices) == 25 and voices[0]["id"] == "Thiện Minh"
    captured = {}

    class Fake:
        def synthesize_batch(self, texts):
            return [engines.pcm_to_wav(b"\0\0" * 10, 48_000)]

    def fake_build(task, conn, env, **_kw):
        captured["conn"], captured["voice"] = conn, task.voice
        return Fake()

    monkeypatch.setattr(settings_endpoint, "build_engine", fake_build)
    response = client.post("/api/v1/settings/tts/preview", json={"connection": "c-new", "type": "vieneu", "voice": "Ngọc Linh"})
    assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
    assert (captured["conn"].id, captured["conn"].type, captured["voice"]) == ("c-new", "vieneu", "Ngọc Linh")
    # a service that needs a key still needs its saved connection
    assert client.get("/api/v1/settings/tts/voices", params={"connection": "c-new", "type": "gemini"}).status_code == 404
