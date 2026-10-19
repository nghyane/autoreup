"""In-process ffmpeg helpers for voice rendering.

Ported from the prototype synth script so the pipeline never shells out to a
``scripts/`` file. Pure ffmpeg/ffprobe mechanics: probe duration, tempo-fit a
clip to a target window, render silence, and concat-assemble a timed track.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def run_command(args: list[str]) -> list[str]:
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    logs: list[str] = []
    if completed.stdout.strip():
        logs.extend(completed.stdout.strip().splitlines())
    if completed.stderr.strip():
        logs.extend(completed.stderr.strip().splitlines())
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(args)}\n" + "\n".join(logs[-20:]))
    return logs


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if missing:
        raise RuntimeError(f"Missing required audio tool(s): {', '.join(missing)}")


def ffmpeg_available() -> bool:
    return not [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]


DUB_DUCK = "sidechaincompress=threshold=0.03:ratio=6:attack=25:release=420:makeup=1"
DUB_CARVE = "equalizer=f=2500:t=q:w=1.0:g=-3"


def dub_mix_filter(bed: str, voice: str, out: str, bed_volume: float, duck: bool = True, carve_db: float = -3.0, voice_volume: float = 1.0, bed_gain: str | None = None) -> str:
    """Dub mix graph. ``bed``/``voice``/``out`` are stream labels ('[0:a:0]' …).

    Keep the original bed alive and DUCK it under narration (sidechain 6:1, slow
    attack/release) instead of playing it at a constant low level; a 2.5 kHz dip on
    the bed buys intelligibility so the bed can stay loud, and music/SFX breathe back
    in the gaps.

    ``bed_gain`` (a mono gain-envelope stream, see ``render_gap_gain``) replaces the
    constant ``bed_volume``: the envelope already holds the bed level under speech and
    the louder level in the gaps."""
    stable = out.strip("[]")
    carve = f",equalizer=f=2500:t=q:w=1.0:g={carve_db:.1f}" if carve_db else ""
    if bed_gain:
        # pan, not aformat: aformat's mono->stereo upmix lowers each side by 3 dB
        bed_chain = (
            f"{bed}aresample=44100,aformat=channel_layouts=stereo[{stable}_raw];"
            f"{bed_gain}aresample=44100,pan=stereo|c0=c0|c1=c0[{stable}_gain];"
            f"[{stable}_raw][{stable}_gain]amultiply{carve}[{stable}_bed];"
        )
    else:
        bed_chain = f"{bed}aresample=44100,volume={bed_volume:.3f}{carve}[{stable}_bed];"
    if duck:
        return (
            bed_chain
            + f"{voice}aresample=44100,volume={voice_volume:.3f},asplit=2[{stable}_voc][{stable}_sc];"
            f"[{stable}_bed][{stable}_sc]{DUB_DUCK}[{stable}_ducked];"
            f"[{stable}_ducked][{stable}_voc]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,volume=0.95,alimiter=limit=0.95{out}"
        )
    return (
        bed_chain
        + f"{voice}aresample=44100,volume={voice_volume:.3f}[{stable}_voc];"
        f"[{stable}_bed][{stable}_voc]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,volume=0.95,alimiter=limit=0.95{out}"
    )


# Gap boost: wherever nobody speaks — neither the source (Silero VAD) nor the dub —
# the original audio (music, SFX) comes up from the bed level to a louder one.
# Source speech counts too: where a dub line ends before the original line does, a
# boost would bring the original voice up in the clear.
GAP_PAD = 0.3          # speech keeps the bed level this long around it (s)
GAP_MIN = 1.5          # quieter gaps than this stay at the bed level (no pumping)
GAP_RAMP = 0.4         # bed <-> gap fade, inside the gap (s)
GAIN_RATE = 1000       # gain-envelope sample rate; ffmpeg resamples it to the mix rate
VOICE_FLOOR_DB = -45.0  # the dub track counts as speaking above this frame RMS
_VOICE_RATE = 8000
_VOICE_FRAME = 0.02


def voice_regions(voice_track: Path) -> list[tuple[float, float]]:
    """Spans where the dub track is audible (its gaps are rendered as silence).

    Streams the decode in one-minute blocks: a whole 2 h track at 8 kHz would be ~230 MB."""
    frame = int(_VOICE_RATE * _VOICE_FRAME)
    floor = 10 ** (VOICE_FLOOR_DB / 20)
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(voice_track), "-f", "f32le", "-ac", "1", "-ar", str(_VOICE_RATE), "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    loud: list[np.ndarray] = []
    block = frame * 3000  # one minute of 20 ms frames
    pending = np.zeros(0, dtype=np.float32)
    while True:
        chunk = decoder.stdout.read(block * 4)
        if not chunk:
            break
        pending = np.concatenate([pending, np.frombuffer(chunk, dtype="<f4")])
        whole = len(pending) // frame * frame
        frames = pending[:whole].reshape(-1, frame)
        loud.append(np.sqrt(np.mean(frames * frames, axis=1)) > floor)
        pending = pending[whole:]
    stderr = decoder.stderr.read().decode(errors="replace")
    if decoder.wait() != 0:
        raise RuntimeError(f"Command failed ({decoder.returncode}): decode {voice_track}\n{stderr[-2000:]}")
    active = np.concatenate(loud) if loud else np.zeros(0, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(np.int8), [0]])))
    return [(start * _VOICE_FRAME, end * _VOICE_FRAME) for start, end in zip(edges[::2], edges[1::2])]


def speech_gaps(regions: list[tuple[float, float]], duration: float, pad: float = GAP_PAD, min_gap: float = GAP_MIN) -> list[tuple[float, float]]:
    """The spans of ``[0, duration]`` at least ``min_gap`` long that no padded speech
    region touches."""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(regions):
        if start - pad - cursor >= min_gap:
            gaps.append((cursor, start - pad))
        cursor = max(cursor, end + pad)
    if duration - cursor >= min_gap:
        gaps.append((cursor, duration))
    return gaps


def gap_gain_envelope(gaps: list[tuple[float, float]], duration: float, bed: float, gap: float, ramp: float = GAP_RAMP, rate: int = GAIN_RATE) -> np.ndarray:
    """Per-sample bed gain: ``bed`` under speech, ``gap`` in the gaps, a raised-cosine
    ``ramp`` inside each gap edge (none at the very start/end of the video)."""
    envelope = np.full(int(round(duration * rate)) + 1, bed, dtype=np.float32)
    for start, end in gaps:
        lo, hi = int(round(start * rate)), min(len(envelope), int(round(end * rate)))
        t = np.arange(lo, hi, dtype=np.float64) / rate
        rise = np.clip((t - start) / ramp, 0.0, 1.0) if start > 0 else 1.0
        fall = np.clip((end - t) / ramp, 0.0, 1.0) if end < duration else 1.0
        weight = 0.5 - 0.5 * np.cos(np.pi * np.minimum(rise, fall))
        envelope[lo:hi] = bed + (gap - bed) * weight
    return envelope


def render_gap_gain(video: Path, voice_track: Path, output_path: Path, bed: float, gap: float) -> dict | None:
    """Write the bed's gain envelope (mono 16-bit wav at ``GAIN_RATE``) for
    ``dub_mix_filter(bed_gain=...)``. ``None`` = no boost: the source VAD is missing
    (never boost blind — the gaps could hold the original voice) or nothing qualifies."""
    import wave

    from backend.app.dubbing.onset import _regions_from_probs, _speech_probs

    packed = _speech_probs(video)
    if packed is None:
        return None
    duration = probe_duration(video)
    gaps = speech_gaps(_regions_from_probs(*packed) + voice_regions(voice_track), duration)
    if not gaps:
        return None
    # hold the last level past the bed's end: amultiply stops at the shorter input
    envelope = np.pad(gap_gain_envelope(gaps, duration, bed, gap), (0, 5 * GAIN_RATE), mode="edge")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(GAIN_RATE)
        out.writeframes((np.clip(envelope, 0.0, 1.0) * 32767).astype("<i2").tobytes())
    return {"gaps": len(gaps), "seconds": round(float(sum(end - start for start, end in gaps)), 1)}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    return max(0.001, float(completed.stdout.strip()))


def atempo_chain(tempo: float) -> str:
    """ffmpeg atempo only accepts 0.5-2.0; chain factors to reach any tempo."""
    parts: list[float] = []
    tempo = max(0.001, tempo)
    while tempo > 2.0:
        parts.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        parts.append(0.5)
        tempo /= 0.5
    parts.append(tempo)
    return ",".join(f"atempo={part:.6f}" for part in parts)


def align_audio(
    raw_path: Path, output_path: Path, target_duration: float, min_tempo: float, max_tempo: float, fixed_tempo: float | None = None
) -> dict:
    """Tempo-fit a raw clip toward ``target_duration``.

    With ``fixed_tempo`` set, every clip is played at exactly that tempo — one
    stable speaking rate across the whole dub — and timing is fitted by the text
    repair loop instead of by varying the tempo per phrase. Otherwise the tempo is
    clamped to ``[min_tempo, max_tempo]`` to fit each clip's window.
    """
    raw_duration = probe_duration(raw_path)
    if fixed_tempo:
        tempo = fixed_tempo
    else:
        tempo = clamp(raw_duration / max(0.001, target_duration), min_tempo, max_tempo)
    playback_duration = raw_duration / tempo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish: this output is a SHARED cache entry and concurrent jobs read it.
    # MP4 writes its moov atom last, so a reader catching a direct write mid-encode
    # sees a corrupt file ("moov atom not found"). Encode to a WRITER-UNIQUE temp name
    # and rename — readers only ever see complete files, and two jobs rendering the
    # same entry at once each build their own temp (last rename wins, both valid).
    part = output_path.with_name(f"{output_path.stem}.{os.getpid()}-{threading.get_ident()}.part{output_path.suffix}")
    run_command([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(raw_path),
        "-filter:a", f"aresample=44100,{atempo_chain(tempo)}",
        "-ac", "2", "-c:a", "aac", "-b:a", "192k",
        str(part),
    ])
    part.replace(output_path)
    return {
        "raw_duration": round(raw_duration, 3),
        "target_duration": round(target_duration, 3),
        "tempo": round(tempo, 4),
        "playback_duration": round(playback_duration, 3),
        "timing_status": "fits" if playback_duration <= target_duration + 0.05 else "overflow",
        "overflow_ms": max(0, int(round((playback_duration - target_duration) * 1000))),
    }


# Voice mastering chain ported from the proven probe pipeline — what made the
# old voice-over pleasant to listen to: de-rumble, tame mud at 220 Hz, lift
# presence (3.2 kHz) and air (7.6 kHz), even out levels, then limit.
VOICE_MASTER_CHAIN = (
    "highpass=f=70,"
    "equalizer=f=250:t=q:w=1.0:g=-0.8,"    # gentle low-mud cut (too much = thin/metal)
    "equalizer=f=3200:t=q:w=1.4:g=1.2,"    # presence for a sweet voice
    "equalizer=f=10000:t=q:w=1.2:g=1.0,"   # air — a female voice needs the top end
    "lowpass=f=15000,"                      # was 13k: cutting lower dulls it
    "acompressor=threshold=-22dB:ratio=1.6:attack=20:release=240:makeup=1.2,"  # slow attack: no pumping
    "alimiter=limit=0.95"
)


def master_voice(input_path: Path, output_path: Path, pitch: float = 1.0) -> None:
    """Apply the voice mastering chain so the dub is clear and easy to listen to.

    ``pitch`` > 1.0 brightens a too-deep voice WITHOUT changing tempo/duration:
    resample up (raises pitch + speed) then atempo back down by the same factor."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chain = f"aresample=44100,{VOICE_MASTER_CHAIN}"
    if abs(pitch - 1.0) > 1e-3:
        shift = f"asetrate=44100*{pitch:.4f},aresample=44100,{atempo_chain(1.0 / pitch)}"
        chain = f"{shift},{chain}"
    run_command([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_path),
        "-filter:a", chain,
        "-ac", "2", "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ])


def speed_clip(path: Path, rate: float) -> None:
    """Speed up (or slow) a clip in place by ``rate`` via atempo. Needed because the
    CapCut synth IGNORES its ``rate`` parameter (verified: 1.0 and 1.4 give identical
    audio), so the only way to change the speaking rate is to time-stretch locally."""
    if abs(rate - 1.0) < 1e-3:
        return
    tmp = path.with_name(path.stem + ".sp" + path.suffix)
    run_command(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-filter:a", atempo_chain(rate), str(tmp)])
    if tmp.exists() and tmp.stat().st_size > 512:
        tmp.replace(path)


def trim_silence(path: Path, threshold_db: float = -50.0, keep: float = 0.08) -> None:
    """Strip leading/trailing silence from a synthesized clip so it starts and ends on
    speech. CapCut adds ~0.1-0.2s of head silence; left in, the clip (placed at the
    source time) and its caption both lead the actual voice by that much. ``keep``
    leaves a hair of padding so the first/last phoneme isn't clipped."""
    tmp = path.with_name(path.stem + ".trim" + path.suffix)
    chain = (
        f"silenceremove=start_periods=1:start_silence={keep}:start_threshold={threshold_db}dB:detection=peak,"
        f"areverse,"
        f"silenceremove=start_periods=1:start_silence={keep}:start_threshold={threshold_db}dB:detection=peak,"
        f"areverse"
    )
    run_command(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-af", chain, str(tmp)])
    if tmp.exists() and tmp.stat().st_size > 512:
        tmp.replace(path)


def render_silence(path: Path, duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{max(0.001, duration):.3f}",
        "-c:a", "aac", "-b:a", "192k",
        str(path),
    ])


_MIX_RATE = 44100
_MIX_WINDOW_SECONDS = 30.0
_MIX_PREFETCH = 16


def _fade_gain(position: np.ndarray, length: float, curve: str) -> np.ndarray:
    """ffmpeg ``afade`` gains for 0..1 progress: ``tri`` is linear, ``qsin`` a quarter sine."""
    x = np.clip(position / length, 0.0, 1.0)
    return np.sin(x * (np.pi / 2)) if curve == "qsin" else x


def _placed_clip(path: Path, duration: float, fin: float, fin_curve: str, fout: float, fout_curve: str) -> np.ndarray:
    """One clip as float32 stereo at ``_MIX_RATE``, faded like ``afade`` in/out and cut at
    ``duration`` (an ``afade`` out silences everything after its end)."""
    completed = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(_MIX_RATE), "pipe:1"],
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): decode {path}\n{completed.stderr.decode(errors='replace')[-2000:]}")
    samples = np.frombuffer(completed.stdout, dtype="<f4").reshape(-1, 2)
    samples = samples[: max(0, int(round(duration * _MIX_RATE)))].copy()
    t = np.arange(len(samples), dtype=np.float64) / _MIX_RATE
    gain = _fade_gain(t, fin, fin_curve) * _fade_gain(duration - t, fout, fout_curve)
    samples *= gain[:, None].astype(np.float32)
    return samples


def assemble_track(
    placements: list[tuple[float, Path, float, bool]],
    output_path: Path,
    total_duration: float | None = None,
    crossfade: float = 0.06,
) -> None:
    """Render the planned placements into one track.

    ``placements`` are ``(place_start, clip_path, duration, glued)`` — the timeline
    stage owns the planning (run-level tempo, breaths, run-boundary anchoring).
    Each clip's edges are faded (equal-power ``qsin`` where it overlaps a glued
    neighbour, a short declick otherwise) and summed into its slot, so the matched
    crossfades stay at unity while gaps between runs stay silent.

    The sum runs in numpy over 30-second windows streamed into a single ffmpeg encode:
    work grows with the clips' own length. (One ``amix`` over every clip summed all
    inputs for every frame of the track — a 2.4 h episode with 1875 lines ran at
    ~0.5x realtime.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not placements:
        render_silence(output_path, max(0.1, total_duration or 0.1))
        return

    cursor = max((place + duration for place, _path, duration, _glued in placements), default=0.0)
    track_end = max(cursor, total_duration or 0.0)
    total = int(round(max(0.1, track_end) * _MIX_RATE))

    declick = 0.008
    n = len(placements)
    jobs: list[tuple[int, tuple]] = []
    for index, (place, path, duration, glued_head) in enumerate(placements):
        glued_tail = index + 1 < n and placements[index + 1][3]
        fin = min(crossfade if glued_head else declick, duration / 2)
        fout = min(crossfade if glued_tail else declick, duration / 2)
        # the planned millisecond, as adelay took it (the amix graph landed within ±1 sample)
        start = int(max(0, int(round(place * 1000))) * _MIX_RATE / 1000 + 0.5)
        jobs.append((start, (Path(path), duration, fin, "qsin" if glued_head else "tri", fout, "qsin" if glued_tail else "tri")))
    jobs.sort(key=lambda job: job[0])

    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(_MIX_RATE), "-ac", "2", "-i", "pipe:0",
         "-c:a", "aac", "-b:a", "192k", str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    window = int(_MIX_WINDOW_SECONDS * _MIX_RATE)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            ahead: deque = deque()
            queued = 0

            def fill() -> None:
                nonlocal queued
                while queued < len(jobs) and len(ahead) < _MIX_PREFETCH:
                    start, args = jobs[queued]
                    ahead.append((start, pool.submit(_placed_clip, *args)))
                    queued += 1

            active: list[tuple[int, np.ndarray]] = []
            position = 0
            while position < total:
                end = min(position + window, total)
                fill()
                while ahead and ahead[0][0] < end:
                    start, future = ahead.popleft()
                    active.append((start, future.result()))
                    fill()
                block = np.zeros((end - position, 2), dtype=np.float32)
                for start, samples in active:
                    lo, hi = max(start, position), min(start + len(samples), end)
                    if lo < hi:
                        block[lo - position:hi - position] += samples[lo - start:hi - start]
                active = [(start, samples) for start, samples in active if start + len(samples) > end]
                encoder.stdin.write(block.tobytes())
                position = end
        encoder.stdin.close()
    except BrokenPipeError:
        pass  # the encoder died: its own error is reported below
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    stderr = encoder.stderr.read().decode(errors="replace")
    if encoder.wait() != 0:
        raise RuntimeError(f"Command failed ({encoder.returncode}): encode {output_path}\n{stderr[-2000:]}")
