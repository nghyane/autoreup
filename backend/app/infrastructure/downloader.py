from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

# yt-dlp used as a LIBRARY retries nothing (the CLI's "10 retries" default lives in its
# option parser): one dropped connection killed a 2h24m Bilibili download 6 MB in. Each
# retry resumes the .part file, and a long download over a flaky link can drop many
# times, so the budget is generous; 4xx errors are never retried by yt-dlp anyway.
_RETRIES = 50
_RETRY_SLEEP_CAP = 15.0  # seconds; backoff 1, 2, 4, 8, 15, 15, ...
_SOCKET_TIMEOUT = 15
# Bilibili: a host whose speed stays under this floor (yt-dlp averages it per
# connection, for 3s) is abandoned for a re-probed one; the .part file resumes.
_BILIBILI_SPEED_FLOOR = 200 * 1024  # bytes/s
_BILIBILI_HOST_SWITCHES = 5  # then finish on whatever host we have


def _retry_sleep(n: int) -> float:
    return min(2.0 ** n, _RETRY_SLEEP_CAP)


class _ProgressRelay:
    """Turn yt-dlp's per-file progress hooks into one overall progress dict.

    A DASH source (Bilibili) downloads video and audio as separate files, so bytes are
    summed across files; the total grows when the next file starts."""

    def __init__(self, on_progress: Callable[[dict], None]):
        self.on_progress = on_progress
        self.files: dict[str, tuple[int, int | None]] = {}

    def __call__(self, d: dict) -> None:
        if d.get("status") not in {"downloading", "finished"}:
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes") or 0
        self.files[d.get("filename") or ""] = (int(downloaded), int(total) if total else None)
        totals = [t for _, t in self.files.values()]
        self.on_progress({
            "downloaded_bytes": sum(b for b, _ in self.files.values()),
            "total_bytes": sum(totals) if all(totals) else None,
            "speed": d.get("speed") if d.get("status") == "downloading" else None,
            "eta": d.get("eta") if d.get("status") == "downloading" else None,
        })


class DownloadError(RuntimeError):
    pass


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    modal_id = parse_qs(parsed.query).get("modal_id", [None])[0]
    if parsed.netloc.endswith("douyin.com") and modal_id:
        return f"https://www.douyin.com/video/{modal_id}"
    return url


class YtDlpDownloader:
    def __init__(self, videos_dir: Path):
        self.videos_dir = videos_dir

    def download(
        self,
        url: str,
        cookies_file: str | None = None,
        video_id: str | None = None,
        cookie_header: str | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        try:
            import yt_dlp
        except ImportError as exc:
            raise DownloadError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

        normalized = normalize_url(url)
        video_id = video_id or uuid4().hex[:12]
        output_template = str(self.videos_dir / f"{video_id}.%(ext)s")
        options: dict = {
            "outtmpl": output_template,
            "format": "bv*+ba/best",
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": _RETRIES,
            "fragment_retries": _RETRIES,
            "retry_sleep_functions": {"http": _retry_sleep, "fragment": _retry_sleep},
            # A stalled read (no bytes) surfaces as a retryable error after this many
            # seconds instead of hanging the download.
            "socket_timeout": _SOCKET_TIMEOUT,
        }
        if on_progress is not None:
            options["progress_hooks"] = [_ProgressRelay(on_progress)]
        if cookies_file:
            options["cookiefile"] = cookies_file
        if cookie_header:
            options["http_headers"] = {
                "Cookie": cookie_header,
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Referer": "https://www.douyin.com/",
            }

        cdn_host = None
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                if urlparse(normalized).netloc.endswith("bilibili.com"):
                    info, cdn_host = self._download_bilibili(ydl, normalized)
                else:
                    info = ydl.extract_info(normalized, download=True)
                filename = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
                if not filename.exists():
                    candidates = sorted(self.videos_dir.glob(f"{video_id}.*"))
                    if not candidates:
                        raise DownloadError("Download finished but output file was not found")
                    filename = candidates[0]
        except Exception as exc:
            raise DownloadError(str(exc)) from exc

        return {
            "id": video_id,
            "title": info.get("title") or video_id,
            "duration": info.get("duration"),
            "thumbnail_url": info.get("thumbnail"),
            "uploader": info.get("uploader") or info.get("creator") or info.get("channel"),
            "path": filename,
            "webpage_url": info.get("webpage_url") or normalized,
            "cdn_host": cdn_host,
        }

    @staticmethod
    def _download_bilibili(ydl, url: str) -> tuple[dict, str | None]:
        """Extract, move the streams to the fastest CDN host, download. When that host
        throttles mid-download, re-extract (fresh signed URLs), re-probe without it and
        resume the same .part file there."""
        from yt_dlp.utils import ThrottledDownload

        from backend.app.infrastructure.bilibili_cdn import retarget_formats

        slow_hosts: set[str] = set()
        for attempt in range(_BILIBILI_HOST_SWITCHES + 1):
            raw = ydl.extract_info(url, download=False, process=False)
            picked = retarget_formats(raw, frozenset(slow_hosts)) if raw.get("_type", "video") == "video" else None
            host = picked[0] if picked else None
            last_try = host is None or attempt == _BILIBILI_HOST_SWITCHES
            ydl.params["throttledratelimit"] = None if last_try else _BILIBILI_SPEED_FLOOR
            try:
                return ydl.process_ie_result(raw, download=True), host
            except ThrottledDownload:
                slow_hosts.add(host)
        raise AssertionError("unreachable: the last attempt runs without a speed floor")
