import sys
import types
from unittest.mock import patch

from backend.app.core.config import Settings
from backend.app.domain.jobs import JobInput, JobState, StageName
from backend.app.infrastructure import bilibili_cdn
from backend.app.infrastructure.downloader import YtDlpDownloader, _ProgressRelay
from backend.app.pipeline.contracts import PipelineStage, StageResult
from backend.app.services.job_service import JobService


class FakeThrottled(Exception):
    pass


def _fake_yt_dlp(captured: dict, videos_dir, hook_events: list[dict], raw_formats=None, throttle_first: int = 0):
    """yt_dlp + yt_dlp.utils modules whose YoutubeDL records its options, fires the given
    progress events and writes the output file — no network. The first
    ``throttle_first`` downloads raise ThrottledDownload, recording the URLs they used."""
    captured.setdefault("download_urls", [])

    class FakeYDL:
        def __init__(self, options):
            captured.update(options)
            self.params = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download, process=True):
            raw = {"title": "t", "duration": 8666, "formats": [dict(f) for f in raw_formats or []]}
            return self.process_ie_result(raw, download) if process else raw

        def process_ie_result(self, raw, download):
            captured["download_urls"].append(([f["url"] for f in raw["formats"]], self.params.get("throttledratelimit")))
            if len(captured["download_urls"]) <= throttle_first:
                raise FakeThrottled()
            for event in hook_events:
                for hook in captured.get("progress_hooks", []):
                    hook(dict(event))
            (videos_dir / "vid.mp4").write_bytes(b"x")
            return raw

        def prepare_filename(self, info):
            return str(videos_dir / "vid.mp4")

    return {
        "yt_dlp": types.SimpleNamespace(YoutubeDL=FakeYDL),
        "yt_dlp.utils": types.SimpleNamespace(ThrottledDownload=FakeThrottled),
    }


def test_download_retries_a_dropped_connection_and_reports_progress(tmp_path):
    captured: dict = {}
    events = [
        {"status": "downloading", "filename": "v.f1.mp4", "downloaded_bytes": 100, "total_bytes": 400, "speed": 50.0, "eta": 6},
        {"status": "finished", "filename": "v.f1.mp4", "downloaded_bytes": 400, "total_bytes": 400},
    ]
    seen: list[dict] = []
    with patch.dict(sys.modules, _fake_yt_dlp(captured, tmp_path, events)):
        YtDlpDownloader(tmp_path).download("https://www.bilibili.com/video/BV1", video_id="vid", on_progress=seen.append)

    # library yt-dlp defaults to ZERO retries; one dropped read must not fail a 2h download
    assert captured["retries"] >= 10
    assert captured["fragment_retries"] >= 10
    assert captured["retry_sleep_functions"]["http"](0) > 0
    assert captured["socket_timeout"] <= 30  # a stalled read must become a retry
    assert seen[0] == {"downloaded_bytes": 100, "total_bytes": 400, "speed": 50.0, "eta": 6}
    assert seen[-1]["downloaded_bytes"] == 400 and seen[-1]["speed"] is None


def test_progress_sums_the_separate_video_and_audio_files():
    seen: list[dict] = []
    relay = _ProgressRelay(seen.append)
    relay({"status": "finished", "filename": "v.f100026.mp4", "downloaded_bytes": 465, "total_bytes": 465})
    relay({"status": "downloading", "filename": "v.f30280.m4a", "downloaded_bytes": 5, "total_bytes_estimate": 35, "speed": 1.0, "eta": 30})
    assert seen[-1]["downloaded_bytes"] == 470
    assert seen[-1]["total_bytes"] == 500
    relay({"status": "downloading", "filename": "v.f9.mp4", "downloaded_bytes": 1})  # size unknown
    assert seen[-1]["total_bytes"] is None


def test_job_publishes_stage_progress_and_clears_it_when_the_stage_ends(tmp_path):
    settings = Settings(root_dir=tmp_path, _env_file=None)
    settings.ensure_directories()
    service = JobService(settings)
    job_id = service.create(JobInput(url="https://example.com/v")).job_id
    on_disk_mid_stage: list = []

    class SlowDownload(PipelineStage):
        name = StageName.download

        def run(self, context):
            context.report({"downloaded_bytes": 10, "total_bytes": 100, "speed": 5.0, "eta": 18})
            on_disk_mid_stage.append(service.get(job_id).progress)
            return StageResult(name=self.name)

    service.pipeline.stages = [SlowDownload()]
    service.run(job_id)

    [mid] = on_disk_mid_stage
    assert mid.pop("started_at")  # the stage's start, for the elapsed clock
    assert mid == {"stage": "download", "downloaded_bytes": 10, "total_bytes": 100, "speed": 5.0, "eta": 18}
    final = service.get(job_id)
    assert final.state == JobState.succeeded
    assert final.progress is None


AKAMAI = "upos-hz-mirrorakam.akamaized.net"


def _raw(host: str) -> dict:
    sig = "/upgcxcode/26/v.m4s?deadline=1&upsig=abc"
    return {"formats": [
        {"format_id": "100026", "vcodec": "av01", "tbr": 900, "url": f"https://{host}{sig}"},
        {"format_id": "30280", "vcodec": "none", "acodec": "mp4a", "url": f"https://{host}/a.m4s?upsig=abc"},
    ]}


def _fake_speeds(speeds: dict):
    return lambda url, client: speeds.get(url.split("/")[2], 0.0)


def test_bilibili_streams_move_to_the_fastest_mirror():
    speeds = {AKAMAI: 40_000.0, "upos-sz-mirrorali.bilivideo.com": 2_400_000.0}
    with patch.object(bilibili_cdn, "probe_speed", side_effect=_fake_speeds(speeds)):
        raw = _raw(AKAMAI)
        host, _ = bilibili_cdn.retarget_formats(raw)
    assert host == "upos-sz-mirrorali.bilivideo.com"
    # video AND audio move; the signed path+query is kept verbatim
    assert raw["formats"][0]["url"] == "https://upos-sz-mirrorali.bilivideo.com/upgcxcode/26/v.m4s?deadline=1&upsig=abc"
    assert raw["formats"][1]["url"].startswith("https://upos-sz-mirrorali.bilivideo.com/a.m4s")


def test_bilibili_keeps_its_own_host_unless_a_mirror_is_clearly_faster():
    speeds = {AKAMAI: 2_000_000.0, "upos-sz-mirrorali.bilivideo.com": 2_200_000.0}
    with patch.object(bilibili_cdn, "probe_speed", side_effect=_fake_speeds(speeds)):
        raw = _raw(AKAMAI)
        host, _ = bilibili_cdn.retarget_formats(raw)
    assert host == AKAMAI
    assert raw["formats"][0]["url"].startswith(f"https://{AKAMAI}/")


def test_non_bilibili_formats_are_left_alone():
    raw = {"formats": [{"format_id": "1", "url": "https://example.com/v.mp4"}]}
    assert bilibili_cdn.retarget_formats(raw) is None


def test_a_bilibili_host_that_throttles_mid_download_is_swapped_for_another(tmp_path):
    captured: dict = {}
    ali, hw = "upos-sz-mirrorali.bilivideo.com", "upos-sz-mirrorhw.bilivideo.com"
    speeds = {AKAMAI: 40_000.0, ali: 2_400_000.0, hw: 1_600_000.0}
    modules = _fake_yt_dlp(captured, tmp_path, [], raw_formats=_raw(AKAMAI)["formats"], throttle_first=1)
    with patch.dict(sys.modules, modules), patch.object(bilibili_cdn, "probe_speed", side_effect=_fake_speeds(speeds)):
        result = YtDlpDownloader(tmp_path).download("https://www.bilibili.com/video/BV1", video_id="vid")

    (first_urls, first_floor), (second_urls, second_floor) = captured["download_urls"]
    assert first_urls[0].startswith(f"https://{ali}/") and first_floor
    assert second_urls[0].startswith(f"https://{hw}/")  # ali slowed down: never re-picked
    assert result["cdn_host"] == hw
