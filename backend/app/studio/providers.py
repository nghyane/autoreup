import hashlib
import json
import re
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from backend.app.services.cookie_vault import CookieVault
from backend.app.studio.models import Platform, SourceVideo

_URL_RE = re.compile(r"https?://[^\s　<>\"']+", re.IGNORECASE)
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def extract_url(text: str) -> str:
    """Pull the first URL out of pasted text (e.g. a Douyin share blob).

    Douyin's "copy" produces text like:
        4.33 复制打开抖音…【…】… https://v.douyin.com/XXXX/ :0pm 08/11 w@F.Hv
    We return just the URL; if none is found, the trimmed input is returned.
    """
    match = _URL_RE.search(text or "")
    if not match:
        return (text or "").strip()
    return match.group(0).rstrip(".,;:!?)】》>\"'")


def canonical_source_url(url: str) -> str:
    """Stable form of a source URL for storing/matching. Bilibili share links carry
    volatile tracking params (?spm_id_from…&vd_source…) — without stripping them a
    re-scan of the same collection would never match its existing series."""
    url = extract_url(url)
    parsed = urlparse(url)
    if "bilibili.com" in parsed.netloc and "/video/" in parsed.path:
        bvid = _extract_video_id(url)
        if bvid:
            return f"https://www.bilibili.com/video/{bvid}"
    mix_id = _extract_douyin_mix_id(url)
    if mix_id:
        return f"https://www.douyin.com/collection/{mix_id}"
    return url


def detect_platform(url: str) -> Platform:
    host = urlparse(url).netloc.lower()
    if "douyin.com" in host:
        return Platform.douyin
    if "bilibili.com" in host or url.startswith("BV"):
        return Platform.bilibili
    if "youtube.com" in host or "youtu.be" in host:
        return Platform.youtube
    return Platform.unknown


class SourceDiscovery:
    def __init__(self, cookie_vault: CookieVault, settings=None):
        self.cookie_vault = cookie_vault
        self.settings = settings  # needed for the headless Douyin channel lister

    def discover(self, url: str, limit: int = 12) -> list[SourceVideo]:
        url = extract_url(url)
        platform = detect_platform(url)
        if platform == Platform.bilibili and self._is_bilibili_space(url):
            return self._discover_bilibili_space(url, limit)
        if platform == Platform.douyin and self._is_douyin_user(url):
            return self._discover_douyin_user(url, limit)
        return [self._discover_single(url, platform)]

    def _discover_douyin_user(self, url: str, limit: int) -> list[SourceVideo]:
        """List a Douyin creator's posts. Signed web API first (no browser); fall back
        to the headless browser only when the API is risk-controlled."""
        if self.settings is None:
            raise RuntimeError("SourceDiscovery cần settings để liệt kê kênh Douyin.")

        sec_uid = _extract_sec_uid(url)
        page_url = f"https://www.douyin.com/user/{sec_uid}" if sec_uid else extract_url(url)
        posts = self._list_douyin_posts(page_url, limit)
        videos: list[SourceVideo] = []
        for post in posts:
            vid = str(post.get("id") or "")
            if not vid:
                continue
            videos.append(SourceVideo(
                id=self._id_for(Platform.douyin, vid),
                platform=Platform.douyin,
                platform_video_id=vid,
                source_url=f"https://www.douyin.com/video/{vid}",
                title=post.get("desc") or f"Video Douyin {vid}",
                thumbnail_url=post.get("cover"),
                duration=post.get("duration"),
            ))
        return videos

    def _list_douyin_posts(self, page_url: str, limit: int) -> list[dict]:
        from backend.app.infrastructure.douyin.api_client import DouyinApiError
        from backend.app.infrastructure.douyin.extractor import DouyinApiExtractor

        try:
            return DouyinApiExtractor(self.settings.videos_dir, self.cookie_vault).list_user_posts(page_url, limit)
        except DouyinApiError:
            from backend.app.infrastructure.browser_extractor import DouyinBrowserExtractor

            extractor = DouyinBrowserExtractor(
                self.settings.videos_dir, self.cookie_vault, self.settings.root_dir,
                self.settings.browser_path, self.settings.node_bin,
            )
            return extractor.list_user_posts(page_url, limit)

    def _discover_single(self, url: str, platform: Platform) -> SourceVideo:
        if platform == Platform.bilibili:
            return self._discover_bilibili_video(url)
        if platform == Platform.douyin:
            return self._discover_douyin(url)
        info = self._extract_info(url, flat=False, platform=platform)
        return self._video_from_info(info, url, platform)

    def _discover_douyin(self, url: str) -> SourceVideo:
        # Short share links (v.douyin.com/XXXX) aren't matched by yt-dlp's Douyin
        # extractor, so it falls back to the generic one and Douyin returns 444.
        # Resolve the redirect to the canonical /video/{id} URL first.
        resolved = self._resolve_redirect(url) or url
        video_id = _extract_video_id(resolved) or _extract_video_id(url)
        try:
            info = self._extract_info(resolved, flat=False, platform=Platform.douyin)
            return self._video_from_info(info, resolved, Platform.douyin)
        except Exception as exc:
            # Degrade gracefully: a minimal card the user can still queue/dub.
            return SourceVideo(
                id=self._id_for(Platform.douyin, video_id or resolved),
                platform=Platform.douyin,
                platform_video_id=video_id or resolved,
                source_url=resolved,
                title=f"Video Douyin {video_id}" if video_id else "Video Douyin",
                error=f"Chưa lấy được metadata ({_short_reason(exc)}). Có thể cần cookie Douyin; vẫn có thể thử dựng.",
            )

    @staticmethod
    def _resolve_redirect(url: str) -> str | None:
        try:
            req = Request(url, headers={"User-Agent": _BROWSER_UA})
            with urlopen(req, timeout=15) as response:
                return response.geturl()
        except Exception:
            return None

    def _discover_douyin_mix(self, url: str, limit: int) -> tuple[str, list[SourceVideo]] | None:
        mix_id = _extract_douyin_mix_id(url)
        if not mix_id:
            return None
        from backend.app.infrastructure.douyin.api_client import DouyinApiClient, DouyinApiError

        cookie = self.cookie_vault.read_header("douyin") or ""
        try:
            mix_name, episodes = DouyinApiClient(cookie).mix_episodes(mix_id, limit=limit)
        except DouyinApiError:
            return None
        videos: list[SourceVideo] = []
        for ep in episodes:
            videos.append(SourceVideo(
                id=self._id_for(Platform.douyin, ep["id"]),
                platform=Platform.douyin,
                platform_video_id=ep["id"],
                source_url=f"https://www.douyin.com/video/{ep['id']}",
                title=ep.get("desc") or (f"Tập {ep['episode']}" if ep.get("episode") else ep["id"]),
                thumbnail_url=ep.get("cover"),
                duration=ep.get("duration"),
                error="Tập trả phí — cần mua trên Douyin" if ep.get("paid") else None,
            ))
        if not videos:
            return None
        return mix_name or "Bộ Douyin", videos

    def _bilibili_view(self, bvid: str) -> dict:
        """Bilibili's public view API — carries full metadata AND the ``ugc_season``
        collection (合集) the video belongs to, when it belongs to one.
        Uses the ``wbi/`` route: the legacy ``/x/web-interface/view`` is risk-controlled
        (HTTP 412) for non-browser clients, while ``wbi/view`` answers unsigned."""
        api_url = f"https://api.bilibili.com/x/web-interface/wbi/view?bvid={bvid}"
        req = Request(api_url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"})
        with urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode())
        if payload.get("code") != 0:
            raise RuntimeError(payload.get("message") or "Bilibili metadata failed")
        return payload.get("data") or {}

    def discover_collection(self, url: str, limit: int = 300) -> tuple[str, list[SourceVideo]] | None:
        """Expand a playlist URL into ``(series_title, episodes)``:
        * Douyin 短剧/合集 (``/share/playlet/detail/<mix_id>`` or ``?object_id=``);
        * Bilibili 合集 (``ugc_season``) or multi-part video (``pages``).
        Returns None for a plain single video."""
        url = extract_url(url)
        if detect_platform(url) == Platform.douyin:
            return self._discover_douyin_mix(url, limit)
        if detect_platform(url) != Platform.bilibili or "/video/" not in urlparse(url).path:
            return None
        bvid = _extract_video_id(url)
        if not bvid:
            return None
        data = self._bilibili_view(bvid)

        season = data.get("ugc_season") or {}
        episodes = [ep for sec in (season.get("sections") or []) for ep in (sec.get("episodes") or [])]
        if episodes:
            videos: list[SourceVideo] = []
            for ep in episodes[:limit]:
                arc = ep.get("arc") or {}
                ep_bvid = ep.get("bvid")
                if not ep_bvid:
                    continue
                videos.append(SourceVideo(
                    id=self._id_for(Platform.bilibili, ep_bvid),
                    platform=Platform.bilibili,
                    platform_video_id=ep_bvid,
                    source_url=f"https://www.bilibili.com/video/{ep_bvid}",
                    title=ep.get("title") or arc.get("title") or ep_bvid,
                    thumbnail_url=arc.get("pic"),
                    duration=arc.get("duration"),
                ))
            if videos:
                return season.get("title") or "Bộ Bilibili", videos

        pages = data.get("pages") or []
        if len(pages) > 1:
            videos = []
            for page in pages[:limit]:
                num = page.get("page")
                if not num:
                    continue
                part = str(page.get("part") or "").strip()
                # uploaders often name parts "1", "2"… — normalize to a readable episode title
                title = f"Tập {num}" if (not part or part == str(num)) else f"Tập {num}: {part}"
                videos.append(SourceVideo(
                    id=self._id_for(Platform.bilibili, f"{bvid}_p{num}"),
                    platform=Platform.bilibili,
                    platform_video_id=f"{bvid}_p{num}",  # unique per part for episode dedup
                    source_url=f"https://www.bilibili.com/video/{bvid}?p={num}",
                    title=title,
                    thumbnail_url=page.get("first_frame") or data.get("pic"),
                    duration=page.get("duration"),
                ))
            if videos:
                return data.get("title") or "Bộ Bilibili", videos
        return None

    def _discover_bilibili_video(self, url: str) -> SourceVideo:
        bvid = _extract_video_id(url) or url.strip()
        data = self._bilibili_view(bvid)
        owner = data.get("owner") or {}
        return SourceVideo(
            id=self._id_for(Platform.bilibili, data.get("bvid") or bvid),
            platform=Platform.bilibili,
            platform_video_id=data.get("bvid") or bvid,
            source_url=f"https://www.bilibili.com/video/{data.get('bvid') or bvid}",
            title=data.get("title") or bvid,
            thumbnail_url=data.get("pic"),
            duration=data.get("duration"),
            uploader=owner.get("name"),
        )

    def _discover_bilibili_space(self, url: str, limit: int) -> list[SourceVideo]:
        playlist = self._extract_info(url, flat=True, platform=Platform.bilibili, playlist_end=limit)
        videos = []
        for entry in (playlist.get("entries") or [])[:limit]:
            bvid = entry.get("id") or entry.get("url")
            if not bvid:
                continue
            try:
                info = self._extract_info(f"https://www.bilibili.com/video/{bvid}", flat=False, platform=Platform.bilibili)
                videos.append(self._video_from_info(info, f"https://www.bilibili.com/video/{bvid}", Platform.bilibili))
            except Exception:
                videos.append(SourceVideo(
                    id=self._id_for(Platform.bilibili, bvid),
                    platform=Platform.bilibili,
                    platform_video_id=bvid,
                    source_url=f"https://www.bilibili.com/video/{bvid}",
                    title=entry.get("title") or bvid,
                    error="Không lấy được metadata chi tiết, vẫn có thể thử tải sau.",
                ))
        return videos

    def _extract_info(self, url: str, *, flat: bool, platform: Platform, playlist_end: int | None = None) -> dict:
        import yt_dlp

        headers = self._headers_for(platform)
        options: dict = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": "in_playlist" if flat else False,
            "noplaylist": not flat,
            "http_headers": headers,
        }
        if platform.value in {"douyin", "bilibili"}:
            cookie_file = self.cookie_vault.netscape_file(platform.value)
            if cookie_file:
                options["cookiefile"] = str(cookie_file)
        if playlist_end:
            options["playlistend"] = playlist_end
        with yt_dlp.YoutubeDL(options) as ydl:
            return ydl.extract_info(url, download=False)

    def _headers_for(self, platform: Platform) -> dict:
        headers = {"User-Agent": _BROWSER_UA}
        cookie = self.cookie_vault.read_header(platform.value) if platform in {Platform.douyin, Platform.bilibili} else None
        if cookie:
            headers["Cookie"] = cookie
        if platform == Platform.bilibili:
            headers["Referer"] = "https://www.bilibili.com/"
        if platform == Platform.douyin:
            headers["Referer"] = "https://www.douyin.com/"
        return headers

    @staticmethod
    def _video_from_info(info: dict, fallback_url: str, platform: Platform) -> SourceVideo:
        platform_id = str(info.get("id") or _extract_video_id(fallback_url) or fallback_url)
        thumbnails = info.get("thumbnails") or []
        thumbnail = info.get("thumbnail") or next((item.get("url") for item in reversed(thumbnails) if item.get("url")), None)
        return SourceVideo(
            id=SourceDiscovery._id_for(platform, platform_id),
            platform=platform,
            platform_video_id=platform_id,
            source_url=info.get("webpage_url") or fallback_url,
            title=info.get("title") or platform_id,
            thumbnail_url=thumbnail,
            duration=info.get("duration"),
            uploader=info.get("uploader") or info.get("channel"),
        )

    @staticmethod
    def _id_for(platform: Platform, platform_video_id: str) -> str:
        digest = hashlib.sha1(f"{platform}:{platform_video_id}".encode()).hexdigest()[:14]
        return f"{platform}-{digest}"

    @staticmethod
    def _is_bilibili_space(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.netloc.endswith("bilibili.com") and parsed.path.startswith("/") and parsed.path.split("/")[1:2] != ["video"] and "space.bilibili.com" in parsed.netloc

    @staticmethod
    def _is_douyin_user(url: str) -> bool:
        parsed = urlparse(url)
        return "douyin.com" in parsed.netloc and parsed.path.startswith("/user/")


def _short_reason(exc: Exception) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return text[:120]


def _extract_douyin_mix_id(url: str) -> str | None:
    """The mix/playlet id from a Douyin 短剧 link: /share/playlet/detail/<id> or
    ?object_id=<id>, and /collection/<id>."""
    parsed = urlparse(url)
    if "douyin.com" not in parsed.netloc and "iesdouyin.com" not in parsed.netloc:
        return None
    m = re.search(r"/(?:playlet/detail|collection|mix)/(\d{15,})", parsed.path)
    if m:
        return m.group(1)
    oid = parse_qs(parsed.query).get("object_id", [None])[0]
    return oid if oid and oid.isdigit() else None


def _extract_sec_uid(url: str) -> str | None:
    parsed = urlparse(url)
    if "douyin.com" in parsed.netloc and parsed.path.startswith("/user/"):
        return parsed.path.rstrip("/").split("/")[-1] or None
    return None


def _extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "bilibili.com" in parsed.netloc and "/video/" in parsed.path:
        return parsed.path.rstrip("/").split("/")[-1]
    if "douyin.com" in parsed.netloc and "/video/" in parsed.path:
        return parsed.path.rstrip("/").split("/")[-1]
    modal_id = parse_qs(parsed.query).get("modal_id", [None])[0]
    if modal_id:
        return modal_id
    # Resolved Douyin share links carry a long numeric id somewhere in the path
    # (e.g. /share/video/7234567890123456789/).
    if "douyin.com" in parsed.netloc:
        digits = re.findall(r"\d{15,}", parsed.path)
        if digits:
            return digits[0]
    return None
