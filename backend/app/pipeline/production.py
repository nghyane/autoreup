"""Production dubbing pipeline — in-process, no subprocess to prototype scripts.

Stage order::

    download -> asr -> glossary -> translate -> qa -> voiceover -> package

Each stage calls :mod:`backend.app.dubbing` functions and passes typed objects
through ``context.artifacts``. TSV/JSON files are written only as inspectable
artifacts. The single external adapter still shelled out to is the ASR worker
client (an external-service adapter, like yt-dlp/ffmpeg).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.domain.artifacts import Artifact, ArtifactKind
from backend.app.domain.jobs import JobInput, PipelineConfig, StageName
from backend.app.dubbing.coverage import (
    CoverageConfig,
    dedupe_overlaps,
    fill_coverage_gaps,
    find_speech_holes,
    fuse_rails,
    plan_uncovered_asr,
    speech_coverage,
    speech_seconds,
)
from backend.app.dubbing.normalize import normalize_segments
from backend.app.dubbing.units import build_units
from backend.app.dubbing.display import build_display_cues, write_subtitles
from backend.app.dubbing.glossary import extract_glossary, format_glossary
from backend.app.agents import AgentConfig
from backend.app.dubbing.onset import _speech_probs, refine_segments
from backend.app.dubbing.models import (
    ANCHOR_FIELDS,
    SEGMENT_FIELDS,
    SPEECH_FIELDS,
    Anchor,
    AsrSegment,
    GlossaryEntry,
    SpeechRow,
)
from backend.app.dubbing.qa import qa_report
from backend.app.dubbing.rail import segments_from_srt
from backend.app.dubbing.render import RenderConfig, assemble_voice, render
from backend.app.dubbing.repair import RepairConfig, expand, repair
from backend.app.dubbing.translate import TranslateConfig, translate
from backend.app.dubbing.tsv_io import read_tsv, write_json, write_tsv
from backend.app.infrastructure.downloader import YtDlpDownloader
from backend.app.infrastructure.packaging import package_outputs
from backend.app.infrastructure.process import run_command
from backend.app.pipeline.contracts import PipelineContext, PipelineStage, StageResult
from backend.app.services.tts_engines import TtsEngine, build_engine
from backend.app.services.tts_engines import build_capcut_client as _build_capcut_client
from backend.app.settings import app_settings_for, chat_agent, ocr_agent, tts_binding
from backend.app.settings.schema import AsrTask, Connection, TtsTask

_ASR_CHUNK_LINE = re.compile(r"ASR (\d+)/(\d+)$")
_VOICEOVER_REPAIR_ROUNDS = 3  # converge; each round only re-synthesizes what changed


class ProductionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.stages: list[PipelineStage] = [
            DownloadStage(),
            ASRStage(),
            GlossaryStage(),
            TranslateStage(),
            QAStage(),
            VoiceoverStage(),
            PackageStage(),
        ]

    def create_context(self, job_id: str, job_input: JobInput, config: PipelineConfig) -> PipelineContext:
        job_dir = self.settings.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return PipelineContext(
            job_id=job_id,
            input_url=job_input.url,
            job_dir=job_dir,
            settings=self.settings,
            config=config,
            artifacts={},
            series_id=job_input.series_id,
            source_video_id=job_input.source_video_id,
        )


def _translate_agent(context: PipelineContext, task: str = "translate") -> AgentConfig:
    """The agent bound to a chat task (translate, glossary, normalize) in the job's settings."""
    return chat_agent(app_settings_for(context), task, context.settings)


def _polish_agent(context: PipelineContext) -> AgentConfig:
    return chat_agent(app_settings_for(context), "repair", context.settings)


def _load_overlay(context: PipelineContext):
    """The job's editable overlay (caption position + cover regions), or None."""
    from backend.app.services.overlay import load_overlay

    return load_overlay(context.job_dir, context.job_id)


def _hardsub_config(settings: Settings, overlay=None):
    from backend.app.services.fonts import default_font, resolve_font
    from backend.app.services.hardsub import HardsubConfig
    from backend.app.services.overlay_images import images_dir
    from backend.app.infrastructure.ffmpeg_encode import video_encode_args, canonical_scale_filter

    font = resolve_font(settings, getattr(overlay, "font", "")) or default_font(settings)
    scale_filter = canonical_scale_filter(settings)
    bed_volume = getattr(overlay, "bed_volume", None)
    gap_volume = getattr(overlay, "gap_volume", None)
    return HardsubConfig(
        font_path=str(font),
        font_size=settings.hardsub_font_size,
        blur=settings.hardsub_blur,
        band_pad=settings.hardsub_band_pad,
        darken=settings.hardsub_darken,
        bed_volume=settings.dub_bed_volume if bed_volume is None else bed_volume,
        voice_volume=getattr(overlay, "voice_volume", 1.0),
        gap_volume=settings.dub_gap_volume if gap_volume is None else gap_volume,
        images_dir=str(images_dir(settings)),
        duck=settings.dub_duck,
        carve_db=settings.dub_carve_db,
        video_codec_args=tuple(video_encode_args(settings, getattr(overlay, "bitrate", ""))),
        scale_filter=scale_filter,
        # The caption is rendered in the OUTPUT space. Without a canonical scale the
        # output IS the source, so the overlay must stay source-sized (0 = source).
        out_width=settings.video_width if scale_filter else 0,
        out_height=settings.video_height if scale_filter else 0,
        sharpen=settings.video_sharpen,
        overlay=overlay,
    )


def _tts_engine(settings: Settings, conn: Connection | None, task: TtsTask) -> TtsEngine:
    """The engine for ``tasks.tts`` (CapCut native → its HTTP ``base_url`` fallback)."""
    return build_engine(task, conn, settings, capcut_client=_build_capcut_client)


class DownloadStage(PipelineStage):
    name = StageName.download

    def hydrate(self, context: PipelineContext) -> bool:
        video = context.settings.videos_dir / f"{context.job_id}.mp4"
        if not (video.exists() and video.stat().st_size > 0):
            return False
        context.artifacts["video"] = video
        return True

    def run(self, context: PipelineContext) -> StageResult:
        settings: Settings = context.settings
        # Reuse an already-downloaded copy of the same source: a retry/requeue of a
        # 48-min episode burned 43 min on a repeat download (Douyin throttles), and the
        # bytes do not change. Keyed by the source video id; hard link when possible.
        source_id = context.source_video_id
        cached = settings.videos_dir / f"src-{source_id}.mp4" if source_id else None
        from backend.app.services.audio import probe_duration

        cached_duration = None
        if cached is not None and cached.exists() and cached.stat().st_size > 0:
            try:
                cached_duration = probe_duration(cached)  # a truncated cache must not ship
            except RuntimeError:
                cached.unlink(missing_ok=True)
                cached = None
        if cached is not None and cached_duration:
            video = settings.videos_dir / f"{context.job_id}.mp4"
            if not video.exists():
                try:
                    os.link(cached, video)
                except OSError:
                    shutil.copyfile(cached, video)
            context.artifacts["video"] = video
            thumb = _make_thumbnail(video, settings.videos_dir / f"{context.job_id}.thumb.jpg")
            artifacts = [_artifact(ArtifactKind.video, video, settings.data_dir)]
            if thumb is not None:
                artifacts.append(_artifact(ArtifactKind.thumbnail, thumb, settings.data_dir))
            return StageResult(
                name=self.name,
                artifacts=artifacts,
                metrics={"reused_cached_video": True, "duration": cached_duration},
                logs=[f"download: reused cached source video {cached.name}"],
            )
        def on_bytes(fields: dict) -> None:
            context.check_cancel()  # raised inside yt-dlp's hook: aborts, the .part stays
            context.report(fields)

        result = _download_source(settings, context.input_url, context.job_id, on_bytes)
        video = Path(result["path"])
        if cached is not None and video.exists() and not cached.exists():
            try:
                os.link(video, cached)
            except OSError:
                shutil.copyfile(video, cached)
        context.artifacts["video"] = video
        artifacts = [_artifact(ArtifactKind.video, video, settings.data_dir)]
        thumb = _make_thumbnail(video, settings.videos_dir / f"{context.job_id}.thumb.jpg")
        if thumb is not None:
            artifacts.append(_artifact(ArtifactKind.thumbnail, thumb, settings.data_dir))
        return StageResult(
            name=self.name,
            artifacts=artifacts,
            metrics={
                "title": result.get("title"),
                "duration": result.get("duration"),
                "thumbnail_url": result.get("thumbnail_url"),
                "uploader": result.get("uploader"),
                "webpage_url": result.get("webpage_url"),
                "cdn_host": result.get("cdn_host"),
            },
        )


class ASRStage(PipelineStage):
    name = StageName.asr

    def hydrate(self, context: PipelineContext) -> bool:
        settings: Settings = context.settings
        segments_tsv = settings.subtitles_dir / f"{context.job_id}.segments.tsv"
        anchors_tsv = settings.subtitles_dir / f"{context.job_id}.anchors.tsv"
        if not (segments_tsv.exists() and anchors_tsv.exists()):
            return False
        try:
            segments = [AsrSegment.from_row(r) for r in read_tsv(segments_tsv)]
            anchors = [Anchor.from_row(r) for r in read_tsv(anchors_tsv)]
        except (ValueError, KeyError):
            return False
        if not segments or not anchors:
            return False
        context.artifacts["segments"] = segments
        context.artifacts["anchors"] = anchors
        return True

    def run(self, context: PipelineContext) -> StageResult:
        settings: Settings = context.settings
        app = app_settings_for(context)
        asr = app.tasks.asr
        if not asr.worker_url and _asr_uses_worker(asr.backend):
            raise RuntimeError("Missing ASR worker URL. Set it in Cài đặt → Phụ đề & lời thoại (or AUTOREUP_ASR_WORKER_URL).")
        logs: list[str] = []

        def note(line: str) -> None:
            logs.append(line)
            context.log(line)

        video = Path(context.artifacts["video"])
        from backend.app.dubbing.onset import _speech_probs, tighten_to_speech
        from backend.app.services.audio import probe_duration

        context.report({"step": "Tìm đoạn có giọng nói (VAD)"})
        packed = _speech_probs(video)
        context.artifacts["vad_probs"] = packed
        duration = probe_duration(video)

        # Captions first (they ARE the script for reup content): OCR the burned-in
        # subtitle band. When they cover the video, the whole-video Whisper pass is
        # redundant — the gap-fill ladder below re-ASRs only the caption-less speech,
        # which is where the cost of a full pass (minutes per hour of video) is wasted.
        ocr_stats: dict = {"ocr_segments": 0, "ocr_coverage": 0.0}
        ocr_segments = None
        if asr.ocr_first:
            from backend.app.dubbing.ocr_subs import detect_band, extract_ocr_segments, llm_recognizer, ocr_coverage

            context.report({"step": "Đọc phụ đề có sẵn (OCR)"})
            try:
                band = detect_band(video)
                recognizer = None  # default reader: macOS Vision
                if app.tasks.ocr.recognizer == "llm":
                    recognizer = llm_recognizer(ocr_agent(app, settings))
                    if band is None:
                        # Vision found no band (or isn't installed): the LLM can still
                        # read a band located by the pixel heuristic.
                        from backend.app.services.hardsub import HardsubConfig, detect_sub_band

                        band = detect_sub_band(video, HardsubConfig())
                ocr_segments = extract_ocr_segments(video, band, recognizer=recognizer) if band else None
            except Exception as exc:  # noqa: BLE001 - OCR is optional; ASR stands alone
                ocr_segments, band = None, None
                note(f"ocr-subs failed: {exc}")
            if ocr_segments and len(ocr_segments) >= 5:
                coverage = ocr_coverage(ocr_segments, duration)
                ocr_stats = {"ocr_segments": len(ocr_segments), "ocr_coverage": round(coverage, 3)}

        # The ASR's actual job: transcribe the VAD speech the captions do NOT cover.
        # No captions at all -> that set is all speech -> the plain full pass. With
        # captions, the cheapest mechanism is chosen by a MEASURED cost model (gluing
        # the uncovered ranges vs transcribing the whole video; see coverage.py) — the
        # uncovered set comes from VAD, never from a coverage threshold.
        if ocr_segments and packed is not None:
            spans = [(seg.start, seg.end) for seg in ocr_segments]
            holes = find_speech_holes(spans, packed, min_seconds=0.5, merge_gap=1.2)
            uncovered_ranges = [(max(0.0, a - 0.2), min(duration, b + 0.2)) for a, b in holes]
            uncovered = sum(speech_seconds(packed, a, b) for a, b in uncovered_ranges)
            plan = plan_uncovered_asr(uncovered, len(uncovered_ranges), duration)
            note(
                f"asr-plan: {uncovered:.0f}s of speech uncovered in {len(uncovered_ranges)} range(s) -> {plan}"
            )
            segments: list[AsrSegment] = list(ocr_segments)
            if plan == "full":
                output_srt = settings.subtitles_dir / f"{context.job_id}.asr.zh.srt"
                logs += _run_asr(context, output_srt, asr)
                asr_segments = segments_from_srt(output_srt)
                total = len(asr_segments)
                asr_segments, tightened = tighten_to_speech(asr_segments, packed)
                if tightened:
                    note(f"vad-tighten: dropped {tightened} music/hallucination segments ({len(asr_segments)} of {total} left)")
                segments, fuse_stats = fuse_rails(segments, asr_segments)
                note(
                    f"ocr-subs: {len(ocr_segments)} captions; kept {fuse_stats['asr_kept']} ASR segments, "
                    f"dropped {fuse_stats['asr_overlapped']} overlapped"
                )
            elif plan == "glue":
                context.report({"step": f"Nhận dạng lời ngoài phụ đề ({len(uncovered_ranges)} đoạn)"})
                try:
                    from backend.app.dubbing.coverage import clamp_to_ranges, transcribe_ranges

                    recovered = transcribe_ranges(
                        video, uncovered_ranges, asr.worker_url, asr.model,
                        workers=settings.asr_workers, chunk_seconds=settings.asr_chunk_seconds,
                        backend=asr.backend,
                        scripts_dir=settings.scripts_dir, python_bin=sys.executable,
                    )
                    recovered = clamp_to_ranges(recovered, uncovered_ranges)
                    segments = sorted([*segments, *recovered], key=lambda seg: (seg.start, seg.end))
                    segments = [AsrSegment(id=index, start=s.start, end=s.end, text_zh=s.text_zh) for index, s in enumerate(segments)]
                    note(f"asr-glue: +{len(recovered)} segments from {len(uncovered_ranges)} range(s)")
                except Exception as exc:  # noqa: BLE001 - best-effort; the ladder follows
                    note(f"asr-glue failed: {exc}")
        else:
            output_srt = settings.subtitles_dir / f"{context.job_id}.asr.zh.srt"
            logs += _run_asr(context, output_srt, asr)
            segments = segments_from_srt(output_srt)
            # Whisper timestamps lead the real words on noisy content (music/SFX) and it
            # hallucinates lines inside music-only passages — align to VAD before use.
            total = len(segments)
            segments, tightened = tighten_to_speech(segments, packed)
            if tightened:
                note(f"vad-tighten: dropped {tightened} music/hallucination segments ({len(segments)} of {total} left)")

        if ocr_segments:
            segments, fuse_stats = fuse_rails(ocr_segments, segments)
            note(
                f"ocr-subs: {len(ocr_segments)} captions; "
                f"kept {fuse_stats['asr_kept']} ASR segments, dropped {fuse_stats['asr_overlapped']} overlapped"
            )

        context.report({"step": "Căn thời gian theo giọng"})
        segments, moved, dropped = refine_segments(segments, video, packed=packed)
        note(f"vad-align: moved {moved} starts, dropped {dropped} hallucinated of {len(segments)} segments")

        context.report({"step": "Nghe lại khoảng trống lời"})
        segments, gap_stats = fill_coverage_gaps(
            video=video,
            segments=segments,
            duration=duration,
            worker_url=asr.worker_url,
            model=asr.model,
            packed=packed,
            config=CoverageConfig(
                enabled=settings.asr_gap_fill,
                min_gap=settings.asr_gap_min_seconds,
                speech_floor=settings.asr_gap_speech_floor,
            ),
        )
        if gap_stats["gaps"]:
            note(
                f"gap-fill: {gap_stats['filled']}/{gap_stats['gaps']} speech gaps re-ASR'd "
                f"({gap_stats['seconds']}s) -> +{gap_stats['recovered']} segments, skipped {gap_stats['skipped']} lyric/artifact, "
                f"unrecovered {gap_stats['unrecovered']}s"
            )
        before_dedupe = len(segments)
        segments = dedupe_overlaps(segments)
        if len(segments) != before_dedupe:
            note(f"rail-dedupe: {before_dedupe} -> {len(segments)} segments after overlap trim")
        context.report({"step": f"Chuẩn hoá {len(segments)} câu (LLM)"})
        segments, normalize_metrics = normalize_segments(segments, _translate_agent(context, "normalize"))
        note(f"normalize: patched {normalize_metrics['patched']} of {normalize_metrics['suspicious']} suspicious segments")
        covered, speech_total, ratio = speech_coverage([(s.start, s.end) for s in segments], packed)
        if speech_total > 0:
            note(f"coverage: {covered:.0f}/{speech_total:.0f}s of speech texted ({ratio*100:.0f}%)")
            if ratio < 0.9:
                note("coverage WARNING: unmatched speech remains")
        units = build_units(segments)
        anchors = [Anchor(id=u.id, start=u.start, end=u.end, source_ids=u.segment_ids, source_zh=u.source_zh) for u in units]
        context.artifacts["segments"] = segments
        context.artifacts["anchors"] = anchors

        segments_tsv = settings.subtitles_dir / f"{context.job_id}.segments.tsv"
        anchors_tsv = settings.subtitles_dir / f"{context.job_id}.anchors.tsv"
        write_tsv(segments_tsv, [s.to_row() for s in segments], SEGMENT_FIELDS)
        write_tsv(anchors_tsv, [a.to_row() for a in anchors], ANCHOR_FIELDS)
        return StageResult(
            name=self.name,
            artifacts=[_artifact(ArtifactKind.source_json, anchors_tsv, settings.data_dir)],
            metrics={"segments": len(segments), "units": len(anchors), "speech_coverage": round(ratio, 3), **normalize_metrics, **gap_stats, **ocr_stats},
            logs=logs,
        )


class GlossaryStage(PipelineStage):
    name = StageName.glossary

    def hydrate(self, context: PipelineContext) -> bool:
        json_path = context.settings.subtitles_dir / f"{context.job_id}.glossary.json"
        if not json_path.exists():
            return False
        try:
            rows = json.loads(json_path.read_text(encoding="utf-8"))
            context.artifacts["glossary"] = [GlossaryEntry(**row) for row in rows]
        except (json.JSONDecodeError, TypeError, OSError):
            return False
        return True  # an empty glossary is a valid extraction result

    def run(self, context: PipelineContext) -> StageResult:
        settings: Settings = context.settings
        anchors: list[Anchor] = context.artifacts["anchors"]

        # Episodes of one series share a cast: the FIRST dubbed episode's glossary
        # becomes the series "show bible" and later episodes just load it. That keeps
        # names/pronouns identical across episodes (each episode re-extracting risked
        # a different Hán-Việt spelling per episode) and drops an ~12s LLM call from
        # every episode after the first. Delete the series file to force a re-extract.
        series_path = _series_glossary_path(settings, context.series_id)
        reused = False
        if series_path is not None and series_path.exists():
            context.report({"step": "Dùng thuật ngữ chung của series"})
            reused = True
            rows = json.loads(series_path.read_text(encoding="utf-8"))
            glossary = [GlossaryEntry(**row) for row in rows]
        else:
            context.report({"step": f"Lập thuật ngữ từ {len(anchors)} câu (LLM)"})
            glossary = extract_glossary(anchors, _translate_agent(context, "glossary"), genre=context.config.content_genre, notes=context.config.content_notes)
            rows = [{"term_zh": g.term_zh, "term_vi": g.term_vi, "kind": g.kind, "note": g.note} for g in glossary]
            if series_path is not None and glossary:
                write_json(series_path, rows)

        context.log(f"{len(glossary)} thuật ngữ: {_count_kinds(glossary)}")
        context.artifacts["glossary"] = glossary
        tsv_path = settings.subtitles_dir / f"{context.job_id}.glossary.tsv"
        json_path = settings.subtitles_dir / f"{context.job_id}.glossary.json"
        write_tsv(tsv_path, [{k.upper(): v for k, v in row.items()} for row in rows], ["TERM_ZH", "TERM_VI", "KIND", "NOTE"])
        write_json(json_path, rows)
        return StageResult(
            name=self.name,
            artifacts=[_artifact(ArtifactKind.glossary, json_path, settings.data_dir)],
            metrics={"terms": len(glossary), "by_kind": _count_kinds(glossary), "reused_series_glossary": reused},
            logs=[format_glossary(glossary)[:2000]],
        )


def _series_glossary_path(settings: Settings, series_id: str | None) -> Path | None:
    if not series_id:
        return None
    return settings.subtitles_dir / f"series-{series_id}.glossary.json"


class TranslateStage(PipelineStage):
    name = StageName.translate

    def hydrate(self, context: PipelineContext) -> bool:
        path = context.settings.subtitles_dir / f"{context.job_id}.speech.tsv"
        if not path.exists():
            return False
        try:
            rows = [SpeechRow.from_row(r) for r in read_tsv(path)]
        except (ValueError, KeyError):
            return False
        if not rows:
            return False
        context.artifacts["speech_rows"] = rows
        return True

    def run(self, context: PipelineContext) -> StageResult:
        settings: Settings = context.settings
        anchors: list[Anchor] = context.artifacts["anchors"]
        segments: list[AsrSegment] = context.artifacts["segments"]
        glossary: list[GlossaryEntry] = context.artifacts.get("glossary", [])
        context.report({"done": 0, "total": len(anchors), "unit": "dòng"})
        result = translate(
            anchors, glossary, _translate_agent(context), context.job_dir / "translate",
            TranslateConfig(
                notes=context.config.content_notes,
                timeout=settings.translate_timeout,
                retries=settings.translate_retries,
            ),
            report=context.report,
            log=context.log,
            cancel=context.cancel,
        )
        speech_rows = result.rows
        context.artifacts["speech_rows"] = speech_rows
        path = settings.subtitles_dir / f"{context.job_id}.speech.tsv"
        write_tsv(path, [r.to_row() for r in speech_rows], SPEECH_FIELDS)
        runs = [(run.seconds, run.timed_out) for run in result.runs]
        return StageResult(
            name=self.name,
            artifacts=[_artifact(ArtifactKind.subtitle_json, path, settings.data_dir)],
            metrics={"speech_rows": len(speech_rows), "captions": len(segments), "workspace_runs": len(result.runs)},
            logs=[f"workspace runs (seconds, timed_out): {runs}"],
        )


class QAStage(PipelineStage):
    name = StageName.qa

    def hydrate(self, context: PipelineContext) -> bool:
        settings: Settings = context.settings
        subtitle_json = settings.subtitles_dir / f"{context.job_id}.vi.display.json"
        subtitle_srt = settings.subtitles_dir / f"{context.job_id}.vi.display.srt"
        if not (subtitle_json.exists() and subtitle_srt.exists()):
            return False
        try:
            cues = json.loads(subtitle_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        context.artifacts["display_cues"] = cues
        context.artifacts["subtitle_json"] = subtitle_json
        context.artifacts["subtitle_srt"] = subtitle_srt
        return True

    def run(self, context: PipelineContext) -> StageResult:
        settings: Settings = context.settings
        anchors: list[Anchor] = context.artifacts["anchors"]
        segments: list[AsrSegment] = context.artifacts["segments"]
        speech_rows: list[SpeechRow] = context.artifacts["speech_rows"]
        context.report({"step": f"Dựng và kiểm tra {len(speech_rows)} dòng phụ đề"})
        cues = build_display_cues(
            speech_rows, segments, settings.min_duration,
            max_chars=settings.subtitle_max_line_chars, lead=settings.subtitle_lead,
        )
        report = qa_report(speech_rows, anchors, cues, max_display_chars=settings.subtitle_max_line_chars)

        subtitle_json = settings.subtitles_dir / f"{context.job_id}.vi.display.json"
        subtitle_srt = settings.subtitles_dir / f"{context.job_id}.vi.display.srt"
        write_subtitles(cues, subtitle_json, subtitle_srt)
        qa_json = settings.subtitles_dir / f"{context.job_id}.qa.json"
        write_json(qa_json, report)

        context.artifacts["display_cues"] = cues
        context.artifacts["subtitle_json"] = subtitle_json
        context.artifacts["subtitle_srt"] = subtitle_srt
        if report["summary"]["issues"]:
            raise RuntimeError(f"QA failed: {report['summary']['issue_detail']}")
        return StageResult(
            name=self.name,
            artifacts=[
                _artifact(ArtifactKind.subtitle_json, subtitle_json, settings.data_dir),
                _artifact(ArtifactKind.subtitle_srt, subtitle_srt, settings.data_dir),
                _artifact(ArtifactKind.qa_report, qa_json, settings.data_dir),
            ],
            metrics=report["summary"] | {"coverage": report["coverage"]},
        )


class VoiceoverStage(PipelineStage):
    name = StageName.voiceover

    def hydrate(self, context: PipelineContext) -> bool:
        settings: Settings = context.settings
        if not app_settings_for(context).tasks.tts.enabled:
            return True  # stage would have been a no-op anyway
        voice = settings.tts_dir / f"{context.job_id}.voice.m4a"
        if not (voice.exists() and voice.stat().st_size > 0):
            return False
        context.artifacts["voice_track"] = voice
        return True

    def run(self, context: PipelineContext) -> StageResult:
        settings: Settings = context.settings
        app = app_settings_for(context)
        tts, conn = tts_binding(app)
        if not tts.enabled:
            return StageResult(name=self.name, metrics={"enabled": False}, logs=["TTS disabled; enable it in Cài đặt → Giọng đọc"])
        engine = _tts_engine(settings, conn, tts)

        glossary: list[GlossaryEntry] = context.artifacts.get("glossary", [])
        anchors: list[Anchor] = context.artifacts.get("anchors", [])
        speech_rows: list[SpeechRow] = context.artifacts["speech_rows"]
        segments: list[AsrSegment] = context.artifacts.get("segments", [])
        seg_by_id = {s.id: s for s in segments}
        # The fit-loop prompts show SOURCE_ZH per ROW, so this map is keyed by ROW id.
        # Keying it by rail segment id fed the LLM a different caption's line and it
        # rewrote the Vietnamese to match the wrong source ("nội dung dịch sai hoàn toàn").
        anchor_zh = {a.id: a.source_zh for a in anchors}
        sources: dict[int, str] = {}
        for row in speech_rows:
            lo, _, hi = row.source_ids.partition("-")
            text = ""
            if lo.strip().isdigit() and hi.strip().isdigit() and int(lo) <= int(hi):
                text = "".join(seg_by_id[i].text_zh for i in range(int(lo), int(hi) + 1) if i in seg_by_id)
            sources[row.sub_id] = text or anchor_zh.get(row.anchor_id, "")
        render_config = RenderConfig(
            provider_url=engine.cache_id,
            speaker=engine.voice,
            text_mode=settings.tts_text_mode,
            crossfade=settings.tts_crossfade,
            workers=settings.tts_workers,
            batch_size=settings.capcut_batch_size,
            batch_concurrency=settings.capcut_batch_concurrency,
            synth_rate=tts.synth_rate,
            lead=settings.tts_lead,
            tempo_ceiling=settings.tts_tempo_ceiling,
            overflow_tolerance=settings.tts_overflow_tolerance,
        )
        work_dir = context.job_dir / "tts"
        cache_dir = settings.tts_cache_dir
        output_audio = settings.tts_dir / f"{context.job_id}.voice.m4a"

        # ONE budget, measured and consistent with what the translator was told: the
        # caption's own window. The VAD speech is evidence only — it distinguishes a
        # genuine source pause (keep the dub short) from dropped content (complete it).
        packed = context.artifacts.get("vad_probs")
        if packed is None:
            packed = _speech_probs(Path(context.artifacts["video"]))
        window_of: dict[int, float] = {}
        vad_of: dict[int, float] = {}
        for index, row in enumerate(speech_rows):
            limit = speech_rows[index + 1].anchor_start if index + 1 < len(speech_rows) else row.anchor_end + 0.5
            window_of[row.sub_id] = round(max(0.1, row.anchor_end - row.anchor_start), 3)
            vad_of[row.sub_id] = round(speech_seconds(packed, row.anchor_start - 0.2, max(limit, row.anchor_end)), 3)

        def _sync_score(candidate) -> tuple[float, float]:
            """Lower is better: total overflow seconds, then worst lateness."""
            overflow = sum(t.overflow_ms for t in candidate.timings) / 1000.0
            late = max((max(0.0, t.place_start - t.source_start) for t in candidate.timings), default=0.0)
            return (round(overflow, 3), round(late, 3))

        def synth_step(label: str):
            context.report({"step": label})
            def counted(done: int, total: int) -> None:
                context.check_cancel()  # between synth chunks: clips already made stay cached
                context.report({"done": done, "total": total, "unit": "câu TTS"})

            return counted

        logs: list[str] = []

        def note(line: str) -> None:
            logs.append(line)
            context.log(line)

        result = render(speech_rows, render_config, work_dir, cache_dir, engine=engine, on_synth=synth_step("Đọc giọng (TTS)"))
        note(f"render round 0: {result.summary}")
        repair_cfg = RepairConfig()
        best_score = _sync_score(result)
        best_state = (speech_rows, result)
        rounds = 0
        while rounds < _VOICEOVER_REPAIR_ROUNDS:
            overflow = {
                t.sub_id: (t.free_after, t.tts_duration)
                for t in result.timings
                if t.free_after > 0 and t.tts_duration > t.free_after + render_config.overflow_tolerance
            }
            # Underfill = the dub is far shorter than the caption while the SOURCE was
            # actually speaking through it (a real pause is not dropped content).
            # The budget for "how long should this line be" is the SOURCE SPEECH (VAD):
            # the dub replaces the performance, not the caption's display time.
            underfull = {
                t.sub_id: (vad_of.get(t.sub_id, 0.0), t.tts_duration)
                for t in result.timings
                if vad_of.get(t.sub_id, 0.0) >= 0.8
                and t.tts_duration < vad_of[t.sub_id] * settings.tts_expand_floor
            }
            if not overflow and not underfull:
                break
            changed = 0
            context.report({"step": f"Vòng chỉnh {rounds + 1}/{_VOICEOVER_REPAIR_ROUNDS}: sửa câu dài/ngắn (LLM)"})
            if overflow:
                speech_rows, n = repair(speech_rows, sources, glossary, overflow, _polish_agent(context), repair_cfg)
                changed += n
                note(f"round {rounds + 1}: shortened {n} over-budget rows")
            if underfull:
                speech_rows, n = expand(speech_rows, sources, glossary, underfull, _polish_agent(context), repair_cfg)
                changed += n
                note(f"round {rounds + 1}: completed {n} under-filled rows")
            rounds += 1
            if not changed:
                break
            candidate = render(
                speech_rows, render_config, work_dir, cache_dir, engine=engine,
                on_synth=synth_step(f"Vòng chỉnh {rounds}/{_VOICEOVER_REPAIR_ROUNDS}: đọc lại câu đã sửa"),
            )
            # Accept a round only when the MEASURED sync does not get worse: a later
            # expand that fixes a pause but overflows the next line must not ship
            # (measured: round 1 hit 0 overflow / 0.22s late, rounds 2-3 turned it into
            # 10.7s overflow and 15.9s late — the loop kept the worst state).
            score = _sync_score(candidate)
            note(f"render round {rounds}: {candidate.summary}")
            if score <= best_score:
                best_score, best_state = score, (speech_rows, candidate)
                note(f"round {rounds} kept (sync score {score})")
            else:
                note(f"round {rounds} rejected (sync score {score} > {best_score}); keeping the better render")
            result = candidate
        speech_rows, result = best_state
        note(f"final render kept: sync score {best_score}")

        # Assemble the final track once, after timings have converged.
        context.report({"step": "Ghép track giọng"})
        assemble_voice(result, output_audio, work_dir, crossfade=render_config.crossfade, pitch=settings.tts_pitch_shift)
        context.artifacts["speech_rows"] = speech_rows
        context.artifacts["timings"] = result.timings

        # The fit loop rewrote rows AFTER the QA stage wrote the subtitles: rebuild the
        # cues from the FINAL text, or the screen shows the pre-fit translation while
        # the voice reads the post-fit one ("sub không khớp với voice"). The windows are
        # the captions', so only the text (and its reflow) changes here.
        if rounds:
            voice_windows = {t.sub_id: (t.place_start, t.place_start + t.tts_duration) for t in result.timings}
            cues = build_display_cues(
                speech_rows, segments, settings.min_duration,
                max_chars=settings.subtitle_max_line_chars, lead=settings.subtitle_lead,
                voice_windows=voice_windows,
            )
            write_subtitles(cues, Path(context.artifacts["subtitle_json"]), Path(context.artifacts["subtitle_srt"]))
            context.artifacts["display_cues"] = cues
            note(f"subtitles rebuilt from the fitted rows ({len(cues)} cues)")
        speech_tsv = settings.subtitles_dir / f"{context.job_id}.speech.tsv"
        write_tsv(speech_tsv, [r.to_row() for r in speech_rows], SPEECH_FIELDS)

        manifest = settings.tts_dir / f"{context.job_id}.tts.json"
        write_json(manifest, {"summary": result.summary, "timings": [asdict(t) for t in result.timings]})
        artifacts = []
        if result.voice_track:
            context.artifacts["voice_track"] = result.voice_track
            artifacts.append(_artifact(ArtifactKind.voice_track, result.voice_track, settings.data_dir))
        artifacts.append(_artifact(ArtifactKind.tts_manifest, manifest, settings.data_dir))
        return StageResult(name=self.name, artifacts=artifacts, metrics={"enabled": True, "repair_rounds": rounds, **result.summary}, logs=logs)


class PackageStage(PipelineStage):
    name = StageName.package

    def run(self, context: PipelineContext) -> StageResult:
        settings: Settings = context.settings
        voice_track = context.artifacts.get("voice_track")
        video = Path(context.artifacts["video"])
        overlay = _load_overlay(context)
        hardsub_config = _hardsub_config(settings, overlay)
        context.report({"step": "Ghép video, giọng và phụ đề (ffmpeg)"})
        context.log(
            f"font {Path(hardsub_config.font_path).name} · {_encode_summary(settings, video, getattr(overlay, 'bitrate', ''))}"
            f" · âm gốc {hardsub_config.bed_volume:.0%} (khi không thoại {hardsub_config.gap_volume:.0%})"
            f" · lồng tiếng {hardsub_config.voice_volume:.0%}"
        )
        artifacts, logs = package_outputs(
            video=video,
            subtitle_json=Path(context.artifacts["subtitle_json"]),
            subtitle_srt=Path(context.artifacts["subtitle_srt"]),
            packages_dir=settings.packages_dir,
            root_data_dir=settings.data_dir,
            voice_track=Path(voice_track) if voice_track else None,
            bed_volume=settings.dub_bed_volume,
        duck=settings.dub_duck,
        carve_db=settings.dub_carve_db,
            hardsub_config=hardsub_config,
        )
        return StageResult(name=self.name, artifacts=artifacts, logs=logs)


def _encode_summary(settings: Settings, video: Path, bitrate: str = "") -> str:
    """"gốc av1 441 kbit/s → h264 882 kbit/s": why the render is the size it is."""
    from backend.app.infrastructure.ffmpeg_encode import probe_video, target_bitrate

    probe = probe_video(video) or {}
    source = f"gốc {probe.get('codec') or '?'} {probe.get('bit_rate', 0) // 1000} kbit/s → " if probe else ""
    return f"{source}h264 {target_bitrate(settings, bitrate) // 1000} kbit/s"


def _artifact(kind: ArtifactKind, path: Path, data_dir: Path) -> Artifact:
    mime = {
        ArtifactKind.video: "video/mp4",
        ArtifactKind.source_json: "text/tab-separated-values",
        ArtifactKind.subtitle_json: "application/json",
        ArtifactKind.subtitle_srt: "application/x-subrip",
        ArtifactKind.qa_report: "application/json",
        ArtifactKind.glossary: "application/json",
        ArtifactKind.pipeline_audit: "text/tab-separated-values",
        ArtifactKind.tts_manifest: "application/json",
        ArtifactKind.voice_track: "audio/mp4",
        ArtifactKind.thumbnail: "image/jpeg",
    }.get(kind, "application/octet-stream")
    return Artifact(
        kind=kind,
        path=str(path.relative_to(data_dir)),
        mime_type=mime,
        size_bytes=path.stat().st_size if path.exists() else None,
    )


def _make_thumbnail(video: Path, out: Path) -> Path | None:
    """Grab a small poster frame from the video for the recent-jobs list. Best-effort:
    a thumbnail is cosmetic, so any ffmpeg hiccup just skips it."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(video),
         "-frames:v", "1", "-vf", "scale=320:-2", str(out)],
        capture_output=True, text=True,
    )
    return out if completed.returncode == 0 and out.exists() else None


def _download_source(settings: Settings, url: str, job_id: str, on_progress=None) -> dict:
    """Douyin goes through the headless-browser extractor; everything else via yt-dlp."""
    from backend.app.services.cookie_vault import CookieVault
    from backend.app.studio.providers import detect_platform

    vault = CookieVault(settings.provider_cookies_dir)
    platform = detect_platform(url)
    if platform.value == "douyin":
        # Signed web API first (in-process, no browser); fall back to the headless
        # browser only when the API is risk-controlled (empty response).
        from backend.app.infrastructure.douyin.extractor import DouyinApiExtractor
        from backend.app.infrastructure.douyin.api_client import DouyinApiError, DouyinDrmError

        try:
            return DouyinApiExtractor(settings.videos_dir, vault).download(url, job_id)
        except DouyinDrmError:
            raise  # encrypted playlet — the browser sniffs the same DRM stream, no point
        except DouyinApiError:
            from backend.app.infrastructure.browser_extractor import DouyinBrowserExtractor

            extractor = DouyinBrowserExtractor(settings.videos_dir, vault, settings.root_dir, settings.browser_path, settings.node_bin)
            if extractor.available():
                return extractor.download(url, job_id)

    cookies_file = (vault.netscape_file(platform.value) if platform.value in {"douyin", "bilibili"} else None)
    return YtDlpDownloader(settings.videos_dir).download(
        url, str(cookies_file) if cookies_file else settings.cookies_file, job_id, on_progress=on_progress
    )


def _count_kinds(glossary: list[GlossaryEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in glossary:
        key = entry.kind or "term"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _asr_uses_worker(backend: str) -> bool:
    """Whether ASR goes to the Cloudflare worker — mirrors ``pick_backend`` in
    scripts/asr_worker_client.py: ``auto`` runs locally when mlx-qwen3-asr or
    mlx-whisper is installed, so only the worker backend needs a worker URL."""
    if backend != "auto":
        return backend == "worker"
    return importlib.util.find_spec("mlx_qwen3_asr") is None and importlib.util.find_spec("mlx_whisper") is None


def _run_asr(context: PipelineContext, output_srt: Path, asr: AsrTask) -> list[str]:
    settings: Settings = context.settings
    script = settings.scripts_dir / "asr_worker_client.py"
    command = [
        sys.executable, str(script), str(context.artifacts["video"]),
        "--backend", asr.backend,
        "--worker-url", asr.worker_url,
        "--output", str(output_srt),
        "--language", "auto",  # dialogue may be Chinese OR Japanese (mixed anime audio)
        "--model", asr.model,
        "--workers", str(settings.asr_workers),
        "--chunk-seconds", str(settings.asr_chunk_seconds),
        "--max-chars", "18",
    ]
    context.report({"step": "Nhận dạng lời (ASR)"})

    def on_line(line: str) -> None:
        # asr_worker_client prints "ASR <done>/<total>" as each chunk comes back
        match = _ASR_CHUNK_LINE.match(line.strip())
        if match:
            context.report({"done": int(match[1]), "total": int(match[2]), "unit": "đoạn"})

    return run_command(command, cwd=settings.root_dir, on_line=on_line, cancel=context.cancel)
