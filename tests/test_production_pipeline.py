"""Contract tests for the in-process production pipeline.

No subprocess, no external services: we drive the stages with prebuilt context
artifacts and stub the LLM/render boundaries.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.core.config import Settings
from backend.app.domain.jobs import PipelineConfig, StageName
from backend.app.dubbing.models import Anchor, AsrSegment, SpeechRow
from backend.app.pipeline.contracts import PipelineContext
from backend.app.pipeline.production import (
    ProductionPipeline,
    QAStage,
    TranslateStage,
    VoiceoverStage,
)


def _context(root: Path) -> PipelineContext:
    # Hermetic: ignore any developer .env (e.g. TTS toggles) so the test is deterministic.
    settings = Settings(root_dir=root, _env_file=None, tts_enabled=False, tts_server_url=None)
    settings.ensure_directories()
    anchors = [Anchor(0, 0.0, 2.0, "0-1", "源一"), Anchor(1, 2.0, 4.0, "2-3", "源二")]
    segments = [
        AsrSegment(0, 0.0, 1.0, "源一"),
        AsrSegment(1, 1.0, 2.0, "一"),
        AsrSegment(2, 2.0, 3.0, "源二"),
        AsrSegment(3, 3.0, 4.0, "二"),
    ]
    return PipelineContext(
        job_id="job",
        input_url="https://example.com/video",
        job_dir=settings.jobs_dir / "job",
        settings=settings,
        config=PipelineConfig(display_max_chars=settings.display_max_chars, min_duration=settings.min_duration),
        artifacts={"anchors": anchors, "segments": segments, "glossary": [], "video": root / "v.mp4"},
    )


class ProductionPipelineTest(unittest.TestCase):
    def test_stage_order(self):
        pipeline = ProductionPipeline(Settings())
        self.assertEqual(
            [s.name for s in pipeline.stages],
            [
                StageName.download, StageName.asr, StageName.glossary, StageName.translate,
                StageName.qa, StageName.voiceover, StageName.package,
            ],
        )

    def test_translate_then_qa_produces_covered_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp))
            segments = ctx.artifacts["segments"]

            from backend.app.agents.workspace import WorkspaceRun

            def fake_run(workspace, prompt, config, *, timeout=None, resume=False, cancel=None):
                # the real agent writes artifacts; the stub does the same
                ws = Path(workspace)
                (ws / "out").mkdir(parents=True, exist_ok=True)
                body = "\n".join(f"{a.id}\tLời unit {a.id}" for a in ctx.artifacts["anchors"])
                (ws / "out" / "final.tsv").write_text(body + "\n")
                return WorkspaceRun(0, "", "", 1.0)

            with patch("backend.app.dubbing.translate.run_agent_workspace", side_effect=fake_run):
                TranslateStage().run(ctx)
            # the workspace is a real project the agent can inspect/re-run
            self.assertTrue((ctx.job_dir / "translate" / "SKILL.md").exists())
            self.assertTrue((ctx.job_dir / "translate" / "input" / "units.tsv").exists())
            rows = ctx.artifacts["speech_rows"]
            # every unit translated, each row exactly on its unit's window
            self.assertEqual([r.anchor_id for r in rows], [a.id for a in ctx.artifacts["anchors"]])
            self.assertEqual([r.anchor_start for r in rows], [a.start for a in ctx.artifacts["anchors"]])

            result = QAStage().run(ctx)
            self.assertEqual(result.metrics["issues"], 0)
            self.assertTrue(Path(ctx.artifacts["subtitle_srt"]).exists())

    def test_glossary_reused_from_series_file_without_llm(self):
        import json

        from backend.app.pipeline.production import GlossaryStage, _series_glossary_path

        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp))
            ctx.series_id = "series1"
            series_path = _series_glossary_path(ctx.settings, "series1")
            series_path.write_text(json.dumps([
                {"term_zh": "罗瑞", "term_vi": "La Thụy", "kind": "name", "note": "nam"},
            ]), encoding="utf-8")

            def explode(*args, **kwargs):
                raise AssertionError("series glossary present -> the LLM must NOT be called")

            with patch("backend.app.dubbing.glossary.complete", side_effect=explode):
                result = GlossaryStage().run(ctx)
            self.assertTrue(result.metrics["reused_series_glossary"])
            self.assertEqual([g.term_vi for g in ctx.artifacts["glossary"]], ["La Thụy"])

    def test_stages_hydrate_from_disk_for_resume(self):
        import json

        from backend.app.dubbing.models import ANCHOR_FIELDS, SEGMENT_FIELDS, SPEECH_FIELDS, AsrSegment, SpeechRow
        from backend.app.dubbing.tsv_io import write_tsv
        from backend.app.pipeline.production import ASRStage, DownloadStage, GlossaryStage

        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp))
            settings = ctx.settings

            # nothing on disk yet -> every stage refuses to hydrate
            self.assertFalse(DownloadStage().hydrate(ctx))
            self.assertFalse(ASRStage().hydrate(ctx))
            self.assertFalse(TranslateStage().hydrate(ctx))

            (settings.videos_dir / "job.mp4").write_bytes(b"x" * 10)
            segments = [AsrSegment(0, 0.0, 1.0, "字"), AsrSegment(1, 1.0, 2.0, "字")]
            write_tsv(settings.subtitles_dir / "job.segments.tsv", [s.to_row() for s in segments], SEGMENT_FIELDS)
            anchors = ctx.artifacts["anchors"]
            write_tsv(settings.subtitles_dir / "job.anchors.tsv", [a.to_row() for a in anchors], ANCHOR_FIELDS)
            (settings.subtitles_dir / "job.glossary.json").write_text(json.dumps([
                {"term_zh": "罗", "term_vi": "La", "kind": "name", "note": ""},
            ]), encoding="utf-8")
            rows = [SpeechRow(0, 0, 0.0, 2.0, "0-1", "Lời thoại")]
            write_tsv(settings.subtitles_dir / "job.speech.tsv", [r.to_row() for r in rows], SPEECH_FIELDS)

            self.assertTrue(DownloadStage().hydrate(ctx))
            self.assertTrue(ASRStage().hydrate(ctx))
            self.assertTrue(GlossaryStage().hydrate(ctx))
            self.assertTrue(TranslateStage().hydrate(ctx))
            self.assertEqual([r.text for r in ctx.artifacts["speech_rows"]], ["Lời thoại"])
            self.assertEqual(len(ctx.artifacts["segments"]), 2)

    def test_voiceover_rebuilds_subtitles_from_fitted_rows(self):
        """The fit loop rewrites rows after QA wrote the cues; the burned subtitles
        must show the FINAL text — otherwise the screen and the voice disagree."""
        import backend.app.pipeline.production as prod
        from backend.app.dubbing.models import AsrSegment
        from backend.app.dubbing.render import RenderResult, RowTiming

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(root_dir=root, _env_file=None, tts_enabled=True, tts_server_url="http://x")
            settings.ensure_directories()
            segments = [AsrSegment(0, 0.0, 1.0, "源一"), AsrSegment(1, 1.0, 2.0, "一"),
                        AsrSegment(2, 2.0, 3.0, "源二"), AsrSegment(3, 3.0, 4.0, "二")]
            anchors = [Anchor(0, 0.0, 2.0, "0-1", "源一一"), Anchor(1, 2.0, 4.0, "2-3", "源二二")]
            ctx = PipelineContext(
                job_id="job",
                input_url="u",
                job_dir=settings.jobs_dir / "job",
                settings=settings,
                config=PipelineConfig(display_max_chars=settings.display_max_chars, min_duration=settings.min_duration),
                artifacts={"anchors": anchors, "segments": segments, "glossary": [], "video": root / "v.mp4"},
            )
            ctx.artifacts["speech_rows"] = [
                SpeechRow(0, 0, 0.0, 2.0, "0-1", "Lời cũ một"),
                SpeechRow(1, 1, 2.0, 4.0, "2-3", "Lời cũ hai"),
            ]
            # the QA stage gates on unit coverage: both units have a row on their window
            QAStage().run(ctx)  # writes the pre-fit subtitles + cue artifacts

            # row 0 overruns its 2s window by 1.35s -> the fit loop must shorten it
            timings = [
                RowTiming(0, 0.0, 2.0, 0.0, 2.0, 3.5, 1.0, 1500),
                RowTiming(1, 2.0, 4.0, 3.5, -1.0, 0.5, 1.0, 0),
            ]
            fake_render = RenderResult(
                voice_track=None, timings=timings,
                summary={"blocks": 2, "rows": 2}, placements=[], total_duration=4.0,
            )

            captured = {}

            def fake_repair(rows, sources, glossary, overflow, llm, cfg):
                # the prompt must show each row its OWN unit's source — keying by rail
                # segment id instead of row id fed the LLM a different line and
                # produced completely wrong rewrites
                captured.update(sources)
                return [SpeechRow(0, 0, 0.0, 2.0, "0-1", "Lời MỚI một"), rows[1]], 1

            with patch.object(prod, "render", return_value=fake_render), patch.object(
                prod, "repair", side_effect=fake_repair
            ), patch.object(prod, "assemble_voice"), patch.object(
                prod, "_build_capcut_client", return_value=None
            ), patch.object(prod, "_polish_agent", return_value=None):
                VoiceoverStage().run(ctx)

            self.assertEqual(captured[0], "源一一")  # row 0's own unit source, not segments[0]
            self.assertEqual(captured[1], "源二二")
            cues = json.loads(Path(ctx.artifacts["subtitle_json"]).read_text())
            self.assertEqual([c["text"] for c in cues], ["Lời MỚI một", "Lời cũ hai"])

    def test_voiceover_keeps_the_best_measured_round(self):
        """A later fit round that makes the measured sync WORSE must be rolled back."""
        import backend.app.pipeline.production as prod
        from backend.app.agents.workspace import WorkspaceRun  # noqa: F401
        from backend.app.dubbing.models import AsrSegment
        from backend.app.dubbing.render import RenderResult, RowTiming

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(root_dir=root, _env_file=None, tts_enabled=True, tts_server_url="http://x")
            settings.ensure_directories()
            segments = [AsrSegment(0, 0.0, 1.0, "源一"), AsrSegment(1, 1.0, 2.0, "一"),
                        AsrSegment(2, 2.0, 3.0, "源二"), AsrSegment(3, 3.0, 4.0, "二")]
            anchors = [Anchor(0, 0.0, 2.0, "0-1", "源一一"), Anchor(1, 2.0, 4.0, "2-3", "源二二")]
            ctx = PipelineContext(
                job_id="job", input_url="u", job_dir=settings.jobs_dir / "job", settings=settings,
                config=PipelineConfig(display_max_chars=settings.display_max_chars, min_duration=settings.min_duration),
                artifacts={"anchors": anchors, "segments": segments, "glossary": [], "video": root / "v.mp4"},
            )
            ctx.artifacts["speech_rows"] = [
                SpeechRow(0, 0, 0.0, 2.0, "0-1", "Lời cũ một"),
                SpeechRow(1, 1, 2.0, 4.0, "2-3", "Lời cũ hai"),
            ]
            QAStage().run(ctx)

            # round 0: one row overruns a little; round 1 (after repair): perfect;
            # round 2 (after an expand): much worse -> the loop must ship round 1
            renders = [
                RenderResult(None, timings=[RowTiming(0, 0.0, 2.0, 0.0, 2.0, 2.6, 1.0, 600),
                                            RowTiming(1, 2.0, 4.0, 2.6, 1.4, 0.5, 1.0, 0)],
                             summary={"blocks": 2, "rows": 2}, placements=[], total_duration=4.0),
                RenderResult(None, timings=[RowTiming(0, 0.0, 2.0, 0.0, 2.0, 2.5, 1.0, 100),
                                            RowTiming(1, 2.0, 4.0, 2.5, 1.5, 0.5, 1.0, 0)],
                             summary={"blocks": 2, "rows": 2}, placements=[], total_duration=4.0),
                RenderResult(None, timings=[RowTiming(0, 0.0, 2.0, 0.0, 2.0, 9.0, 1.0, 7000),
                                            RowTiming(1, 2.0, 4.0, 9.0, -5.0, 0.5, 1.0, 0)],
                             summary={"blocks": 2, "rows": 2}, placements=[], total_duration=4.0),
            ]
            calls = {"n": 0}

            def fake_render(*args, **kwargs):
                result = renders[min(calls["n"], len(renders) - 1)]
                calls["n"] += 1
                return result

            def fake_repair(rows, sources, glossary, overflow, llm, cfg):
                return rows, 1  # pretend every round changes the text

            with patch.object(prod, "render", side_effect=fake_render), patch.object(
                prod, "repair", side_effect=fake_repair
            ), patch.object(prod, "expand", side_effect=fake_repair), patch.object(
                prod, "assemble_voice"
            ), patch.object(prod, "_build_capcut_client", return_value=None), patch.object(
                prod, "_polish_agent", return_value=None
            ):
                out = VoiceoverStage().run(ctx)
            # round 1 kept, not round 2's 7000ms overrun
            self.assertEqual(max(t.overflow_ms for t in ctx.artifacts["timings"]), 100)
            self.assertIn("keeping the better render", "\n".join(out.logs))

    def test_voiceover_skipped_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp))  # tts_enabled defaults to False
            ctx.artifacts["speech_rows"] = [SpeechRow(0, 0, 0.0, 2.0, "0-1", "Lời")]
            result = VoiceoverStage().run(ctx)
            self.assertEqual(result.metrics, {"enabled": False})


if __name__ == "__main__":
    unittest.main()


class DownloadCacheTest(unittest.TestCase):
    def test_download_stage_reuses_cached_source_video(self):
        """A retry of a long episode must not re-download: the bytes are cached by id."""
        import backend.app.pipeline.production as prod
        from backend.app.domain.jobs import JobInput
        from backend.app.pipeline.production import DownloadStage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(root_dir=root, _env_file=None)
            settings.ensure_directories()
            # a previous job already downloaded this source (a real tiny mp4)
            cached = settings.videos_dir / "src-4863edb7fc33.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.3",
                 "-pix_fmt", "yuv420p", str(cached)],
                check=True,
            )
            ctx = PipelineContext(
                job_id="job2", input_url="https://example.com/v", job_dir=settings.jobs_dir / "job2",
                settings=settings, config=PipelineConfig(),
                artifacts={},
            )
            ctx.source_video_id = "4863edb7fc33"

            def explode(*args, **kwargs):
                raise AssertionError("cached video present -> must not download")

            with patch.object(prod, "_download_source", side_effect=explode):
                result = DownloadStage().run(ctx)
            self.assertTrue(result.metrics["reused_cached_video"])
            self.assertTrue(ctx.artifacts["video"].exists())
            self.assertEqual(ctx.artifacts["video"].read_bytes(), cached.read_bytes())
            # and the cache was seeded on a fresh download in the normal path

    def test_asr_worker_url_only_required_for_the_worker_backend(self):
        from backend.app.pipeline import production as prod

        self.assertTrue(prod._asr_uses_worker("worker"))
        self.assertFalse(prod._asr_uses_worker("local"))
        self.assertFalse(prod._asr_uses_worker("qwen"))
        # auto runs locally when mlx-qwen3-asr or mlx-whisper is importable, else the worker
        with patch.object(prod.importlib.util, "find_spec", return_value=object()):
            self.assertFalse(prod._asr_uses_worker("auto"))
        with patch.object(prod.importlib.util, "find_spec", return_value=None):
            self.assertTrue(prod._asr_uses_worker("auto"))
