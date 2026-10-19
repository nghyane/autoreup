"""Voice render + measure (v3): one clip per caption row, placed at its picture.

Placement is the timeline's per-row re-anchoring (``timeline.plan_rows``): every
clip starts at its own caption window, so the dub cannot drift away from the
picture and no clamps are needed. The voice is only ever sped up (<= ceiling) and
only as much as needed to fit the free time before the next caption.

This module measures and reports — it never hides an overflow. The voiceover stage
grades the measured durations against the source speech (VAD) and runs the fit
loop (repair/expand) on the rows that need it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from backend.app.services.audio import align_audio, assemble_track, master_voice, probe_duration, require_ffmpeg, speed_clip, trim_silence
from backend.app.services.tts import (
    HttpTtsProvider,
    TtsBlock,
    clean_tts_text,
    raw_audio_path,
    to_provider_scale,
)
from .models import SpeechRow
from .text import strip_explicit_turn_markers
from .timeline import RowPlan, TimelineConfig, plan_rows, summarize

if TYPE_CHECKING:
    from collections.abc import Callable

    from backend.app.services.capcut_tts import CapCutTtsClient
    from backend.app.services.tts_engines import TtsEngine


_SYNTH_CHUNK = 64  # clips per engine call before they are written to the cache


@dataclass(slots=True)
class RenderConfig:
    provider_url: str
    speaker: str = "BV421_vivn_streaming"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    text_mode: str = "natural"
    timeout: float = 30.0
    attempts: int = 3
    batch_size: int = 8
    batch_concurrency: int = 8
    synth_rate: float = 1.0
    workers: int = 8
    crossfade: float = 0.06
    # timeline (per-row placement, see timeline.TimelineConfig)
    lead: float = 0.15
    tempo_ceiling: float = 1.18
    overflow_tolerance: float = 0.4


@dataclass(slots=True)
class RowTiming:
    sub_id: int
    source_start: float
    source_end: float
    place_start: float
    free_after: float
    tts_duration: float
    tempo: float
    overflow_ms: int


@dataclass(slots=True)
class RenderResult:
    voice_track: Path | None
    timings: list[RowTiming] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    # (place_start, clip_path, playback_duration, glued)
    placements: list[tuple[float, Path, float, bool]] = field(default_factory=list)
    plans: list[RowPlan] = field(default_factory=list)
    total_duration: float = 0.0


def assemble_voice(result: RenderResult, output_audio: Path, work_dir: Path, crossfade: float = 0.06, pitch: float = 1.0) -> Path:
    """Assemble the placed clips smoothly, then master the track (EQ/compress/limit)."""
    output_audio = Path(output_audio)
    work_dir = Path(work_dir)
    unmastered = work_dir / "voice.unmastered.m4a"
    assemble_track(result.placements, unmastered, total_duration=result.total_duration, crossfade=crossfade)
    master_voice(unmastered, output_audio, pitch=pitch)
    result.voice_track = output_audio
    result.summary["voice_track"] = str(output_audio)
    return output_audio


def _blocks_from_rows(rows: list[SpeechRow], text_mode: str) -> tuple[list[TtsBlock], list[SpeechRow]]:
    """One TTS block per caption row (empty texts skipped); returns (blocks, rows)."""
    blocks: list[TtsBlock] = []
    usable: list[SpeechRow] = []
    for row in rows:
        text = clean_tts_text(strip_explicit_turn_markers(row.text), text_mode=text_mode)
        if not text:
            continue
        blocks.append(
            TtsBlock(
                block_id=str(row.sub_id),
                start=round(row.anchor_start, 3),
                end=round(row.anchor_end, 3),
                text=text,
            )
        )
        usable.append(row)
    return blocks, usable


def _legacy_engine(config: RenderConfig, synth_client) -> "TtsEngine":
    """Engine for callers that still pass a CapCut client (or nothing → the HTTP server)."""
    from backend.app.services.tts_engines import CapCutEngine, HttpEngine

    if synth_client is not None:
        return CapCutEngine(
            synth_client,
            config.speaker,
            batch_size=config.batch_size,
            concurrency=config.batch_concurrency,
            rate=config.synth_rate,
            volume=to_provider_scale(config.volume),
            cache_id=config.provider_url,
        )
    provider = HttpTtsProvider(
        config.provider_url,
        speaker=config.speaker,
        speed=config.speed,
        pitch=config.pitch,
        volume=config.volume,
        timeout=config.timeout,
        attempts=config.attempts,
    )
    return HttpEngine(provider, cache_id=config.provider_url, workers=config.workers)


def render(
    speech_rows: list[SpeechRow],
    config: RenderConfig,
    work_dir: Path,
    cache_dir: Path,
    synth_client: "CapCutTtsClient | None" = None,
    engine: "TtsEngine | None" = None,
    on_synth: "Callable[[int, int], None] | None" = None,
) -> RenderResult:
    """Synthesize every row and place it at its caption; report measured timings.

    Every synth goes through ``engine`` (its ``cache_id`` keys the raw-clip cache); a
    legacy ``synth_client`` becomes a CapCut engine. Does NOT assemble the final track
    (the fit loop re-measures first); call :func:`assemble_voice` once timings converge.
    ``on_synth(done, total)`` follows the clips synthesized in this call (cache misses).
    """
    require_ffmpeg()
    blocks, usable = _blocks_from_rows(speech_rows, config.text_mode)
    if not blocks:
        return RenderResult(voice_track=None, summary={"blocks": 0, "rows": 0})

    if engine is None:
        engine = _legacy_engine(config, synth_client)
    cache_dir = Path(cache_dir)
    work_dir = Path(work_dir)

    def raw_path_for(block: TtsBlock) -> Path:
        return raw_audio_path(
            cache_dir, engine.cache_id, engine.voice, block,
            speed=config.synth_rate, pitch=config.pitch, volume=config.volume, text_mode=config.text_mode,
        )

    def store_raw(raw_path: Path, audio: bytes) -> None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = raw_path.with_name(f"{raw_path.stem}.{os.getpid()}-{threading.get_ident()}.build{raw_path.suffix}")
        tmp.write_bytes(audio)
        trim_silence(tmp)
        speed_clip(tmp, config.synth_rate)
        tmp.replace(raw_path)

    # 0) synthesize only the clips not cached yet, in chunks so a late failure keeps
    # everything already paid for
    todo = [(raw_path_for(b), b) for b in blocks]
    todo = [(rp, b) for rp, b in todo if not (rp.exists() and rp.stat().st_size > 1024)]
    for start in range(0, len(todo), _SYNTH_CHUNK):
        chunk = todo[start : start + _SYNTH_CHUNK]
        try:
            audios = engine.synthesize_batch([b.text for _, b in chunk])
        except RuntimeError as exc:
            # keep the clips that did synthesize (TtsEngineError.partial), then fail loudly
            done = [(rp, audio) for (rp, _b), audio in zip(chunk, getattr(exc, "partial", None) or []) if audio]
            with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
                list(pool.map(lambda item: store_raw(*item), done))
            raise
        if len(audios) != len(chunk):
            raise RuntimeError(f"TTS engine returned {len(audios)} clips for {len(chunk)} texts")
        with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
            list(pool.map(store_raw, [rp for rp, _ in chunk], audios))
        if on_synth is not None:
            on_synth(start + len(chunk), len(todo))

    def ensure_raw(block: TtsBlock) -> Path:
        raw_path = raw_path_for(block)
        if not (raw_path.exists() and raw_path.stat().st_size > 1024):
            raise RuntimeError(f"TTS produced no audio for block {block.block_id}")
        return raw_path

    # 1) raw synth (parallel) and raw durations
    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        raw_paths = list(pool.map(ensure_raw, blocks))
        raw_durations = list(pool.map(probe_duration, raw_paths))

    # 2) per-row plan: re-anchor to each caption, speed up only to fit the free time
    plans = plan_rows(
        usable,
        {index: duration for index, duration in enumerate(raw_durations)},
        TimelineConfig(
            lead=config.lead,
            tempo_ceiling=config.tempo_ceiling,
            overflow_tolerance=config.overflow_tolerance,
        ),
    )

    # 3) tempo-fit each clip with its own factor (cached per clip+tempo)
    def fit_block(index: int) -> tuple[int, float, Path, float, bool]:
        raw_path = raw_paths[index]
        plan = plans[index]
        aligned_key = hashlib.sha256(f"{raw_path.stem}|{round(plan.tempo, 4)}|{config.synth_rate}".encode()).hexdigest()[:24]
        aligned = cache_dir / "aligned" / f"{aligned_key}.m4a"
        playback: float | None = None
        if aligned.exists() and aligned.stat().st_size > 1024:
            try:
                playback = probe_duration(aligned)
            except RuntimeError:
                aligned.unlink(missing_ok=True)
        if playback is None:
            timing = align_audio(raw_path, aligned, raw_durations[index] / plan.tempo, min_tempo=1.0, max_tempo=1.0, fixed_tempo=plan.tempo)
            playback = float(timing["playback_duration"])
        return (index, plan.place_start, aligned, playback, False)

    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        fitted = list(pool.map(fit_block, range(len(blocks))))

    placements = [(place, path, playback, glued) for _index, place, path, playback, glued in fitted]
    timings = [
        RowTiming(
            sub_id=plans[index].sub_id,
            source_start=plans[index].source_start,
            source_end=plans[index].source_end,
            place_start=place,
            free_after=plans[index].free_after,
            tts_duration=round(playback, 3),
            tempo=plans[index].tempo,
            overflow_ms=int(round(plans[index].overflow_seconds * 1000)),
        )
        for index, place, _path, playback, _glued in fitted
    ]
    total_duration = max((row.anchor_end for row in usable), default=0.1)
    summary = {
        "blocks": len(blocks),
        "rows": len(timings),
        "max_overflow_ms": max((t.overflow_ms for t in timings), default=0),
        "overflow_rows": sum(1 for t in timings if t.overflow_ms > 0),
        "timeline": summarize(plans),
        "voice_track": None,
        "total_duration": round(total_duration, 3),
    }
    return RenderResult(
        voice_track=None,
        timings=timings,
        summary=summary,
        placements=placements,
        plans=plans,
        total_duration=total_duration,
    )
