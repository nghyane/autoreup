import shutil
import subprocess
from pathlib import Path

import pytest

import backend.app.services.hardsub as hardsub
from backend.app.infrastructure.packaging import package_outputs
from backend.app.services.hardsub import HardsubConfig, _build_cover_chain


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.parametrize("height", [2, 3, 4, 40, 41, 74, 80, 81])
def test_a_cover_box_of_any_height_renders(height):
    # an 80px subtitle band used to ask for chroma radius 20 and kill the whole render
    chain = _build_cover_chain([(0, 900, 1920, height, "blur", 1.0)], HardsubConfig(blur=20))
    completed = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=1920x1080", "-frames:v", "1",
         "-filter_complex", f"[0:v]format=yuv420p[src];{chain.replace('[0:v]', '[src]', 1)}[cov]null[out]",
         "-map", "[out]", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_a_failed_render_fails_the_package_instead_of_passing_without_a_video(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("Command failed (234): ffmpeg ... boxblur")

    monkeypatch.setattr(hardsub, "build_hardsub", broken)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    data = tmp_path / "data"
    (data / "subtitles").mkdir(parents=True)
    subtitle = data / "subtitles" / "ep.vi.display.json"
    subtitle.write_text('[{"start": 0.0, "end": 1.0, "text": "Xin chào"}]')
    voice = data / "voice.m4a"
    voice.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="boxblur"):
        package_outputs(Path("v.mp4"), subtitle, data / "ep.srt", data / "packages", data, voice_track=voice)
