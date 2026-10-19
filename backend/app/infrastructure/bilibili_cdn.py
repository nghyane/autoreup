"""Pick the fastest Bilibili CDN host for a video's DASH streams.

Bilibili hands overseas viewers its Akamai host (upos-hz-mirrorakam.akamaized.net),
and yt-dlp downloads ``baseUrl`` as-is. Measured from Vietnam on one 2h24m episode:
Akamai crawled at ~30-46 KB/s and kept dropping, while the same signed URL on
upos-sz-mirrorali served 2.4 MB/s. The signature lives in the query string, not the
host, so the path+query can be replayed on any ``upos-sz-mirror*`` host — the trick
BBDown (--upos-host) and Bili23-Downloader (CDN.replace) use. Each candidate reads a
short range deep in the file and the fastest wins; the original host stays a candidate.
A host's speed swings over minutes (Akamai probed 6 MB/s once, 46 KB/s earlier), so the
downloader re-probes — avoiding hosts that already slowed down — when one throttles.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlparse

import httpx

# Bili23-Downloader's default node list (mainland + overseas mirrors). Akamai is not a
# valid REWRITE target (403 for any URL signed for another host), only an original.
MIRROR_HOSTS = (
    "upos-sz-mirrorali.bilivideo.com",
    "upos-sz-mirroralib.bilivideo.com",
    "upos-sz-mirrorhw.bilivideo.com",
    "upos-sz-mirrorhwb.bilivideo.com",
    "upos-sz-mirror08c.bilivideo.com",
    "upos-sz-mirrorcos.bilivideo.com",
    "upos-sz-mirrorcosb.bilivideo.com",
    "upos-sz-mirrorcoso1.bilivideo.com",
    "upos-sz-mirroraliov.bilivideo.com",
    "upos-sz-mirrorcosov.bilivideo.com",
)
_HEADERS = {"Referer": "https://www.bilibili.com", "User-Agent": "Mozilla/5.0"}
# Read past the first MBs: a slow host can still burst the start of a file (the stuck
# download got its first ~20 MB quickly, then fell to tens of KB/s).
_PROBE_OFFSET = 32 * 1024 * 1024
_PROBE_BYTES = 1024 * 1024
_PROBE_SECONDS = 5.0  # read budget per host; probes run in parallel
# Wall-clock budget for the whole round. A host whose name resolves to many IPs can
# spend the connect timeout on EACH (mirrorali: 24s) — such a straggler scores 0
# and is left to finish on its own thread instead of holding the download back.
_ROUND_SECONDS = 7.0
_MIN_GAIN = 1.5  # switch only when clearly faster than the host Bilibili chose


def _is_bilibili_media(url: str) -> bool:
    host = urlparse(url).netloc
    return host.endswith((".bilivideo.com", ".bilivideo.cn", ".akamaized.net")) and "/mallxcodeboss/" not in url


def with_host(url: str, host: str) -> str:
    return urlparse(url)._replace(netloc=host).geturl()


def _probe(url: str) -> float:
    # One client per probe: a straggler still running after the round closes only its own.
    with httpx.Client(timeout=httpx.Timeout(2.0, connect=3.0), follow_redirects=True) as client:
        return probe_speed(url, client)


def probe_speed(url: str, client: httpx.Client, offset: int = _PROBE_OFFSET) -> float:
    """Bytes/s reading a short range of ``url`` at ``offset``; 0.0 when unusable."""
    start = time.monotonic()
    got = 0
    try:
        headers = {**_HEADERS, "Range": f"bytes={offset}-{offset + _PROBE_BYTES - 1}"}
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and offset:
                return probe_speed(url, client, 0)  # file shorter than the offset
            if response.status_code != 206:
                return 0.0
            # No chunk_size: a sized iter_bytes waits to fill each chunk, which on a
            # 7 KB/s host blew the deadline (one probe round took 32s).
            for chunk in response.iter_bytes():
                got += len(chunk)
                if got >= _PROBE_BYTES or time.monotonic() - start > _PROBE_SECONDS:
                    break
    except httpx.HTTPError:
        return 0.0
    return got / max(time.monotonic() - start, 1e-3)


def fastest_host(url: str, avoid: frozenset[str] = frozenset()) -> tuple[str, dict[str, float]]:
    """Probe ``url`` on its own host and every mirror in parallel; return the host to
    use and the measured speeds. Keeps the original unless a mirror is clearly faster.
    Hosts in ``avoid`` (they throttled earlier in this download) are not candidates."""
    original = urlparse(url).netloc
    hosts = [h for h in dict.fromkeys([original, *MIRROR_HOSTS]) if h not in avoid]
    if not hosts:
        return original, {}
    pool = ThreadPoolExecutor(len(hosts), thread_name_prefix="cdn-probe")
    futures = {host: pool.submit(_probe, with_host(url, host)) for host in hosts}
    wait(futures.values(), timeout=_ROUND_SECONDS)
    pool.shutdown(wait=False, cancel_futures=True)
    speeds = {host: (f.result() if f.done() else 0.0) for host, f in futures.items()}
    best = max(hosts, key=lambda h: speeds[h])
    if original in avoid or not speeds.get(original):
        return (best if speeds[best] else original), speeds
    if best != original and speeds[best] > speeds[original] * _MIN_GAIN:
        return best, speeds
    return original, speeds


def retarget_formats(ie_result: dict, avoid: frozenset[str] = frozenset()) -> tuple[str, dict[str, float]] | None:
    """Rewrite every Bilibili stream URL in an UNPROCESSED yt-dlp result onto the
    fastest host (in place). Returns (host, speeds), or None when there was nothing
    to probe (not a single video, or no Bilibili CDN URLs)."""
    formats = [f for f in ie_result.get("formats") or [] if f.get("url") and _is_bilibili_media(f["url"])]
    if not formats:
        return None
    probe = max(formats, key=lambda f: (f.get("vcodec") not in (None, "none"), f.get("tbr") or 0))
    host, speeds = fastest_host(probe["url"], avoid)
    if host != urlparse(probe["url"]).netloc:
        for fmt in formats:
            fmt["url"] = with_host(fmt["url"], host)
    return host, speeds
