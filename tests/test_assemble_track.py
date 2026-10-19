"""assemble_track: clips land in their slots with afade-style edges, and a long episode
with many lines assembles in seconds (one amix over every clip ran at ~0.5x realtime)."""

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from backend.app.services.audio import _fade_gain, _placed_clip, assemble_track

RATE = 44100


def _tone(path: Path, seconds: float, frequency: int = 440) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}:sample_rate={RATE}",
         "-ac", "2", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def _decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(RATE), "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4")


def _rms(samples: np.ndarray, start: float, end: float) -> float:
    return float(np.sqrt(np.mean(samples[int(start * RATE):int(end * RATE)] ** 2)))


def test_fade_curves_match_afade():
    x = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    assert np.allclose(_fade_gain(x, 1.0, "tri"), [0, 0.25, 0.5, 1, 1])
    assert np.allclose(_fade_gain(x, 1.0, "qsin"), np.sin(np.clip(x, 0, 1) * np.pi / 2))
    assert np.allclose(_fade_gain(np.array([-0.1]), 1.0, "tri"), [0])  # past the end of a fade-out


def test_clip_is_faded_and_cut_at_its_duration(tmp_path):
    tone = _tone(tmp_path / "t.wav", 1.0)
    clip = _placed_clip(tone, 0.5, 0.05, "tri", 0.05, "qsin")
    source = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(tone), "-f", "f32le", "pipe:1"],
                                          capture_output=True, check=True).stdout, dtype="<f4").reshape(-1, 2)
    assert clip.shape == (int(0.5 * RATE), 2) and clip.dtype == np.float32
    assert abs(clip[0]).max() < 1e-6 and abs(clip[-1]).max() < 0.01  # both edges fade to silence
    inner = slice(int(0.06 * RATE), int(0.44 * RATE))
    assert np.array_equal(clip[inner], source[inner])  # untouched between the fades
    head = int(0.05 * RATE)
    live = np.abs(source[:head, 0]) > 1e-3
    gain = clip[:head, 0][live] / source[:head, 0][live]
    assert np.allclose(gain, (np.arange(head) / RATE / 0.05)[live], atol=1e-4)  # linear "tri" fade-in


def test_clips_land_in_their_slots_with_silence_between(tmp_path):
    tone = _tone(tmp_path / "t.wav", 1.0)
    out = tmp_path / "track.m4a"
    assemble_track([(1.0, tone, 1.0, False), (3.0, tone, 0.5, False)], out, total_duration=5.0)
    track = _decode(out)
    assert abs(len(track) / RATE - 5.0) < 0.05
    assert _rms(track, 1.1, 1.9) > 0.08 and _rms(track, 3.1, 3.4) > 0.08
    for start, end in ((0.1, 0.9), (2.1, 2.9), (3.6, 4.9)):  # gaps + the cut tail of the 2nd clip
        assert _rms(track, start, end) < 0.002


def test_glued_neighbours_crossfade_without_a_dip_or_bump(tmp_path):
    # different pitches = uncorrelated, which is what an equal-power crossfade keeps level
    first, second = _tone(tmp_path / "a.wav", 1.0, 440), _tone(tmp_path / "b.wav", 1.0, 1000)
    out = tmp_path / "track.m4a"
    # 2nd clip glued: starts 60 ms before the 1st ends, equal-power qsin crossfade
    assemble_track([(0.5, first, 1.0, False), (1.44, second, 1.0, True)], out, total_duration=3.0)
    track = _decode(out)
    steady = _rms(track, 0.7, 1.3)
    assert 0.85 * steady < _rms(track, 1.44, 1.50) < 1.15 * steady


def test_many_lines_over_a_long_track_assemble_quickly(tmp_path):
    tone = _tone(tmp_path / "t.wav", 1.5)
    placements = [(i * 3.0, tone, 1.5, False) for i in range(300)]  # 300 lines over 15 minutes
    out = tmp_path / "track.m4a"
    started = time.monotonic()
    assemble_track(placements, out, total_duration=905.0)
    assert time.monotonic() - started < 30
    track = _decode(out)
    assert abs(len(track) / RATE - 905.0) < 0.1
    assert _rms(track, 897.1, 898.4) > 0.08 and _rms(track, 898.6, 899.9) < 0.002  # the last line is in place


def test_failed_decode_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError, match="decode"):
        assemble_track([(0.0, bad, 1.0, False)], tmp_path / "out.m4a", total_duration=2.0)
