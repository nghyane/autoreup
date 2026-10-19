"""OCR rail with a pluggable recognizer: LLM vision batching/parsing, the shared
change-detection path, and the ``tasks.ocr`` binding."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import backend.app.agents.api_llm as api_llm
from backend.app.agents import AgentConfig, AgentError, complete
from backend.app.core.config import Settings
from backend.app.dubbing import ocr_subs
from backend.app.dubbing.ocr_subs import extract_ocr_segments, llm_recognizer
from backend.app.settings import CHAT_TASKS, AppSettings, Connection, ocr_agent

VISION = AgentConfig(provider="gemini", model="gemini-x", api_key="AIza-secret-999", retries=1)


def _crops(n: int) -> list[np.ndarray]:
    return [np.full((20, 60), i % 256, dtype=np.uint8) for i in range(n)]


def _fake_chat(monkeypatch, reply_for):
    """Patch the API chat; ``reply_for(images)`` builds each reply. Returns the calls."""
    calls: list[dict] = []

    def chat(system, user, config, images=None):
        calls.append({"system": system, "user": user, "config": config, "images": list(images or [])})
        return reply_for(list(images or []))

    monkeypatch.setattr(api_llm, "chat", chat)
    return calls


def test_llm_recognizer_batches_and_maps_indexes(monkeypatch):
    calls = _fake_chat(monkeypatch, lambda images: json.dumps({str(i + 1): f"字{i + 1}" for i in range(len(images))}))
    texts = llm_recognizer(VISION)(_crops(45))
    assert [len(call["images"]) for call in calls] == [20, 20, 5]
    assert all(img.startswith(b"\x89PNG") for call in calls for img in call["images"])
    assert all(call["config"] is VISION for call in calls)
    assert len(texts) == 45
    assert texts[0] == "字1" and texts[19] == "字20" and texts[20] == "字1" and texts[44] == "字5"


def test_llm_recognizer_tolerates_fences_and_missing_or_bad_indexes(monkeypatch):
    reply = 'Kết quả:\n```json\n{"1": "第一句", "3": "第三句", "9": "lạc", "x": "rác", "2": null}\n```'
    _fake_chat(monkeypatch, lambda images: reply)
    assert llm_recognizer(VISION)(_crops(3)) == ["第一句", "", "第三句"]


def test_llm_recognizer_ignores_braces_in_chatter_after_the_json(monkeypatch):
    _fake_chat(monkeypatch, lambda images: '{"1": "好"} — lưu ý: ảnh 2 không có {phụ đề}')
    assert llm_recognizer(VISION)(_crops(2)) == ["好", ""]


def test_llm_recognizer_retries_a_transient_api_error(monkeypatch):
    import backend.app.agents.runtime as runtime

    attempts: list[int] = []

    def flaky(images):
        attempts.append(len(images))
        if len(attempts) == 1:
            raise AgentError("gemini HTTP 429: quota", retryable=True)
        return '{"1": "好"}'

    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    _fake_chat(monkeypatch, flaky)
    assert llm_recognizer(replace(VISION, retries=2))(_crops(1)) == ["好"]
    assert attempts == [1, 1]


def test_complete_refuses_images_for_cli_agents():
    with pytest.raises(AgentError):
        complete("s", "u", AgentConfig(provider="agy", retries=1), images=[b"\x89PNG"])


def test_llm_recognizer_unparseable_reply_raises(monkeypatch):
    _fake_chat(monkeypatch, lambda images: "không đọc được")
    with pytest.raises(AgentError):
        llm_recognizer(VISION)(_crops(2))


def test_llm_recognizer_empty_input_makes_no_call(monkeypatch):
    calls = _fake_chat(monkeypatch, lambda images: "{}")
    assert llm_recognizer(VISION)([]) == []
    assert calls == []


def _caption_video(path: Path, fps: int = 10) -> tuple[int, int, int, int]:
    """Gray video: 1.5s blank, caption A (left) to 3.5s, caption B (right) to 5.5s, blank to 6.5s.

    Captions are white glyphs with a black outline — what the text mask looks for."""
    import cv2

    width, height = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()

    def frame(caption: tuple[str, int] | None):
        img = np.full((height, width, 3), 128, dtype=np.uint8)
        if caption:
            text, x = caption
            for color, thickness in (((0, 0, 0), 9), ((255, 255, 255), 3)):
                cv2.putText(img, text, (x, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, thickness, cv2.LINE_AA)
        return img

    timeline = [(1.5, None), (2.0, ("HHH", 60)), (2.0, ("MMM", 175)), (1.0, None)]
    for seconds, caption in timeline:
        for _ in range(int(seconds * fps)):
            writer.write(frame(caption))
    writer.release()
    return 165, 55, width, height


def test_extract_ocr_segments_uses_injected_recognizer_once():
    seen: list[list[np.ndarray]] = []

    def recognizer(crops):
        seen.append(crops)
        return ["甲", "乙"][: len(crops)]

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "caps.avi"
        band = _caption_video(video)
        segments = extract_ocr_segments(video, band, recognizer=recognizer)

    assert len(seen) == 1 and len(seen[0]) == 2
    assert all(crop.ndim == 2 and crop.shape[1] == band[2] for crop in seen[0])
    assert [seg.text_zh for seg in segments] == ["甲", "乙"]
    assert [seg.id for seg in segments] == [1, 2]
    first, second = segments
    assert first.start == pytest.approx(1.5, abs=0.3) and first.end == pytest.approx(3.5, abs=0.3)
    assert second.start == pytest.approx(3.5, abs=0.3) and second.end == pytest.approx(5.5, abs=0.3)


def test_extract_ocr_segments_drops_empty_reads():
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "caps.avi"
        band = _caption_video(video)
        segments = extract_ocr_segments(video, band, recognizer=lambda crops: ["", "乙"])
    assert [(seg.id, seg.text_zh) for seg in segments] == [(1, "乙")]


def test_extract_ocr_segments_rejects_a_recognizer_returning_the_wrong_count():
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "caps.avi"
        band = _caption_video(video)
        with pytest.raises(ValueError):
            extract_ocr_segments(video, band, recognizer=lambda crops: ["甲"])


def test_extract_ocr_segments_without_vision_or_recognizer_returns_none(monkeypatch):
    monkeypatch.setattr(ocr_subs, "vision_recognizer", lambda config: None)
    assert extract_ocr_segments(Path("/nonexistent.mp4"), (0, 10, 10, 10)) is None


def _app(recognizer: str, connection: str, conn_type: str = "gemini") -> AppSettings:
    chat = {"connection": "agent", "model": "m-chat"}
    return AppSettings(
        connections=[
            Connection(id="agent", type="agy"),
            Connection(id="vis", type=conn_type, api_key="k", base_url="https://api.example/v1"),
        ],
        tasks={name: chat for name in CHAT_TASKS} | {
            "ocr": {"recognizer": recognizer, "connection": connection, "model": "m-vision"},
        },
    )


def test_ocr_agent_resolves_vision_connection():
    with tempfile.TemporaryDirectory() as tmp:
        env = Settings(root_dir=Path(tmp), _env_file=None)
        config = ocr_agent(_app("llm", "vis"), env)
    assert (config.provider, config.model, config.api_key) == ("gemini", "m-vision", "k")
    assert config.timeout == env.agent_timeout and config.retries == env.agent_retries


def test_ocr_agent_rejects_missing_or_non_vision_connection():
    with tempfile.TemporaryDirectory() as tmp:
        env = Settings(root_dir=Path(tmp), _env_file=None)
        with pytest.raises(ValueError):
            ocr_agent(_app("llm", "gone"), env)
        with pytest.raises(ValueError):
            ocr_agent(_app("llm", "vis", conn_type="edge"), env)
    with pytest.raises(ValueError):
        _app("llm", "gone").validate_references()
    with pytest.raises(ValueError):
        _app("llm", "vis", conn_type="capcut").validate_references()
    _app("vision", "").validate_references()  # Vision needs no connection


def test_extract_ocr_segments_keeps_time_on_a_420_video_with_an_odd_crop_height():
    """A 4:2:0 source rounds an odd crop height down to even; read at the asked-for
    height, every frame is short a row and captions drift early over a long video."""
    import cv2
    import subprocess

    width, height, fps = 320, 240, 10
    with tempfile.TemporaryDirectory() as tmp:
        raw, video = Path(tmp) / "caps.avi", Path(tmp) / "caps.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        blank = np.full((height, width, 3), 128, dtype=np.uint8)
        caption = blank.copy()
        for color, thickness in (((0, 0, 0), 9), ((255, 255, 255), 3)):
            cv2.putText(caption, "HHH", (60, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, thickness, cv2.LINE_AA)
        for img, seconds in ((blank, 60), (caption, 2), (blank, 1)):
            for _ in range(seconds * fps):
                writer.write(img)
        writer.release()
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
            check=True,
        )
        band = (165, 55, width, height)  # padded crop is 75 rows: odd
        segments = extract_ocr_segments(video, band, recognizer=lambda crops: ["甲"] * len(crops))

    assert [seg.text_zh for seg in segments] == ["甲"]
    assert segments[0].start == pytest.approx(60.0, abs=0.3)
    assert segments[0].end == pytest.approx(62.0, abs=0.3)
