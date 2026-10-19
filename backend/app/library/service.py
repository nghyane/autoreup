"""Library service — the business logic tying the store to discovery, jobs, and render.

Endpoints call this; it owns the SQLite store, reuses SourceDiscovery to scan channel
links into episodes, JobService to dub them, and ffmpeg to concat dubbed episodes into
one export video. Job state is reconciled lazily into each episode's cached dub_state.
"""

import json
import shutil
import subprocess
import threading
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.domain.artifacts import ArtifactKind
from backend.app.domain.jobs import JobInput, JobState, StageStatus
from backend.app.agents import AgentConfig, complete
from backend.app.library.db import LibraryDB
from backend.app.library.grouping import Group, cluster_into_series, episode_number, show_tags
from backend.app.library.models import DubState, Episode, Export, ExportState, Series
from backend.app.library.store import LibraryStore, _new_id
from backend.app.services.cookie_vault import CookieVault
from backend.app.services.job_service import JobService
from backend.app.settings import SettingsStore, chat_agent
from backend.app.studio.models import Platform
from backend.app.studio.providers import SourceDiscovery, canonical_source_url, detect_platform

# Walk the whole channel when scanning (bounded by has_more/timeout in the lister),
# not just the newest page — otherwise older episodes of a show are silently missed.
_CHANNEL_SCAN_LIMIT = 500

# Maps a dubbing job's state to the episode dub_state we cache.
_JOB_TO_DUB = {
    JobState.queued: DubState.translating,
    JobState.running: DubState.translating,
    JobState.review: DubState.review,
    JobState.succeeded: DubState.dubbed,
    JobState.failed: DubState.failed,
    JobState.canceled: DubState.none,
}


class LibraryError(RuntimeError):
    pass


class LibraryService:
    def __init__(self, settings: Settings, job_service: JobService):
        self.settings = settings
        self.jobs = job_service
        self.store = LibraryStore(LibraryDB(settings.data_dir / "library.db"))
        self.discovery = SourceDiscovery(CookieVault(settings.provider_cookies_dir), settings)

    @property
    def exports_dir(self) -> Path:
        return self.settings.data_dir / "exports"

    # ----- series ---------------------------------------------------------
    def create_series(self, title: str, sources: list[dict] | None = None) -> Series:
        series = self.store.create_series(title)
        for src in sources or []:
            self.store.add_source(series.id, Platform(src["platform"]), src["url"], src.get("label"))
        return self.store.get_series(series.id) or series

    def get_series(self, series_id: str) -> Series | None:
        self._reconcile(series_id)
        return self.store.get_series(series_id)

    def list_series(self, q: str = "", limit: int = 60, offset: int = 0):
        return self.store.list_series(q, limit, offset)

    def update_series(self, series_id: str, **changes) -> Series | None:
        return self.store.update_series(series_id, **changes)

    def delete_series(self, series_id: str) -> bool:
        """Cascade: remove every dubbing job + its files, every export file, then the
        series row (its episodes/sources/exports cascade via the FK)."""
        series = self.store.get_series(series_id)
        if not series:
            return False
        for ep in self.store.list_episodes(series_id):
            if ep.dub_job_id:
                try:
                    self.jobs.delete(ep.dub_job_id)
                except Exception:
                    pass
        for export in self.store.list_exports(series_id):
            self._remove_export_file(export)
        return self.store.delete_series(series_id)

    # ----- sources --------------------------------------------------------
    def add_source(self, series_id: str, platform: str, url: str, label: str | None = None):
        return self.store.add_source(series_id, Platform(platform), url, label)

    def delete_source(self, source_id: str) -> bool:
        return self.store.delete_source(source_id)

    def scan(self, series_id: str) -> dict:
        """Re-scan the series' channel sources for NEW episodes. A channel holds several
        shows, so we cluster the discovered videos and import only the cluster that
        belongs to THIS series (the one overlapping its existing episodes / matching its
        title) — other shows on the same channel are left for their own series."""
        series = self.store.get_series(series_id)
        if not series:
            raise LibraryError("Series không tồn tại")
        known = {e.platform_video_id for e in self.store.list_episodes(series_id)}
        added = 0
        for source in series.sources:
            videos = self.discovery.discover(source.url, limit=_CHANNEL_SCAN_LIMIT)
            groups = cluster_into_series(videos) if len(videos) > 1 else [Group(title=series.title, videos=videos)]
            picked = _match_group(series.title, known, groups)
            if picked:
                episodes = [self._episode_from_source(series_id, v) for v in picked.videos]
                added += self.store.upsert_episodes(series_id, episodes)
            self.store.mark_source_scanned(source.id)
        if not series.cover_url:
            cover = next((e.thumbnail_url for e in self.store.list_episodes(series_id) if e.thumbnail_url), None)
            if cover:
                self.store.update_series(series_id, cover_url=cover)
        self._translate_titles_bg(series_id)
        total = len(self.store.list_episodes(series_id))
        return {"added": added, "total": total}

    def add_video(self, url: str, series_id: str | None = None, title: str | None = None) -> dict:
        """Single-video intake: discover one video and add it as an episode — into an
        existing series, or a fresh 1-episode series. Returns the series + episode ids."""
        videos = self.discovery.discover(url, limit=1)
        if not videos:
            raise LibraryError("Không lấy được video từ link này")
        video = videos[0]
        if series_id and self.store.get_series(series_id, with_sources=False):
            sid = series_id
        else:
            series = self.store.create_series(title or video.title or "Video lẻ")
            sid = series.id
            if video.thumbnail_url:
                self.store.update_series(sid, cover_url=video.thumbnail_url)
        self.store.upsert_episodes(sid, [self._episode_from_source(sid, video)])
        episode = next(
            (e for e in self.store.list_episodes(sid) if e.platform_video_id == str(video.platform_video_id)),
            None,
        )
        self._translate_titles_bg(sid)
        return {"series_id": sid, "episode_id": episode.id if episode else None}

    # ----- channel import (1 channel -> many series) ----------------------
    def preview_channel(self, url: str) -> list[dict]:
        """Discover a channel and cluster its videos into proposed series (not yet
        persisted). Each group is matched to an EXISTING series of this channel (so a
        re-scan syncs instead of duplicating), with a count of how many episodes are new.
        LLM assists with naming/clustering when the heuristic is unsure."""
        url = canonical_source_url(url)
        # A Bilibili collection / multi-part video IS one series with an authoritative
        # title — no clustering or LLM naming needed, just its episode list.
        collection = self.discovery.discover_collection(url)
        if collection:
            title, videos = collection
            groups = [Group(title=title, videos=videos)]
        else:
            videos = self.discovery.discover(url, limit=_CHANNEL_SCAN_LIMIT)
            groups = cluster_into_series(videos) if len(videos) > 1 else [Group(title="", videos=videos)]
            groups = self._refine_with_llm(groups, videos)
        existing = self._existing_for_channel(url)
        platform = detect_platform(url).value
        out: list[dict] = []
        for group in groups:
            gids = {str(v.platform_video_id) for v in group.videos}
            match = self._match_existing(group, existing)
            known = match[2] if match else set()
            new_count = len(gids - known)
            out.append({
                "title": group.title,
                "title_vi": None,
                "platform": platform,
                "series_id": match[0] if match else None,
                "existing_title": match[1] if match else None,
                "new_count": new_count,
                "total": len(group.videos),
                "episodes": [self._video_dict(v) for v in group.videos],
            })
        self._fill_group_titles_vi(out)
        return out

    def _fill_group_titles_vi(self, groups: list[dict]) -> None:
        """Give each proposed series a Vietnamese title so the scan review is readable
        (the cluster titles are raw Chinese). One LLM call for the whole list;
        best-effort — on any hiccup the review simply shows the original titles."""
        pending = [(i, g["title"]) for i, g in enumerate(groups) if g.get("title")]
        if not pending:
            return
        lines = "\n".join(f"{i}\t{title}" for i, title in pending)
        system = (
            "Dịch tên bộ phim/series sang tiếng Việt: TỰ NHIÊN, ngắn gọn, cuốn hút; BỎ hashtag và ký hiệu "
            'thừa. Chỉ trả JSON: [{"i": chỉ_số, "vi": "tên tiếng Việt"}]'
        )
        try:
            raw = complete(system, f"Danh sách (chỉ_số<TAB>tên):\n{lines}", self._agent())
            data = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        except Exception:
            return
        for item in data:
            i, vi = item.get("i"), (item.get("vi") or "").strip()
            if isinstance(i, int) and 0 <= i < len(groups) and vi:
                groups[i]["title_vi"] = vi

    def import_channel(self, url: str, groups: list[dict]) -> list[str]:
        """Sync chosen groups: a group carrying ``series_id`` upserts new episodes into
        that existing series; otherwise a new series is created (channel attached as a
        source). Idempotent — re-scanning a channel adds new episodes + new shows without
        duplicating. Episode payloads come from the preview, so no extra network call."""
        url = canonical_source_url(url)
        platform = detect_platform(url)
        touched: list[str] = []
        for grp in groups:
            episodes_in = grp.get("episodes") or []
            if not episodes_in:
                continue
            series_id = grp.get("series_id")
            if series_id and self.store.get_series(series_id, with_sources=False):
                self.store.update_series(series_id, title=grp.get("title") or None)
            else:
                series = self.store.create_series(grp.get("title") or "Bộ mới")
                self.store.add_source(series.id, platform, url)
                series_id = series.id
            episodes = [self._episode_from_dict(series_id, platform, e) for e in episodes_in]
            self.store.upsert_episodes(series_id, episodes)
            if not (self.store.get_series(series_id, with_sources=False).cover_url):
                cover = next((e.get("thumbnail_url") for e in episodes_in if e.get("thumbnail_url")), None)
                if cover:
                    self.store.update_series(series_id, cover_url=cover)
            self._translate_titles_bg(series_id)
            touched.append(series_id)
        return touched

    def _existing_for_channel(self, url: str) -> list[tuple[str, str, set[str]]]:
        """(series_id, title, known platform_video_ids) for series already on this channel."""
        out: list[tuple[str, str, set[str]]] = []
        for sid in self.store.series_ids_for_source_url(url):
            series = self.store.get_series(sid, with_sources=False)
            if not series:
                continue
            ids = {e.platform_video_id for e in self.store.list_episodes(sid)}
            out.append((sid, series.title, ids))
        return out

    @staticmethod
    def _match_existing(group: Group, existing: list[tuple[str, str, set[str]]]) -> tuple[str, str, set[str]] | None:
        """Match a fresh cluster to an existing series by episode-id overlap, else title."""
        gids = {str(v.platform_video_id) for v in group.videos}
        best, best_overlap = None, 0
        for entry in existing:
            overlap = len(gids & entry[2])
            if overlap > best_overlap:
                best, best_overlap = entry, overlap
        if best:
            return best
        return next((e for e in existing if e[1] and group.title and (e[1] == group.title)), None)

    # ----- episodes -------------------------------------------------------
    def list_episodes(self, series_id: str, dub_state: DubState | None = None) -> list[Episode]:
        self._reconcile(series_id)
        episodes = self.store.list_episodes(series_id, dub_state)
        for ep in episodes:
            if ep.dub_job_id and ep.dub_state in {DubState.translating, DubState.failed}:
                self._attach_dub_progress(ep)
        return episodes

    def _attach_dub_progress(self, ep: Episode) -> None:
        """Fill an episode's transient dub_stage/index/total/error from its job manifest so
        the UI can show 'bước X/8 · <stage>' while translating and the reason if it failed."""
        try:
            m = self.jobs.get(ep.dub_job_id)
        except Exception:
            return
        ep.dub_stage_total = len(m.config.stages) if m.config and m.config.stages else 8
        if m.state == JobState.failed:
            failed = next((s for s in reversed(m.stages) if s.status == StageStatus.failed), None)
            ep.dub_stage = failed.name if failed else None
            ep.dub_stage_index = len(m.stages)
            ep.dub_error = m.error or (failed.error if failed else None)
        elif m.progress and not m.stages:
            # A queued job prefetching its video: the download is already under way.
            ep.dub_stage = m.progress.get("stage")
            ep.dub_stage_index = 1
            ep.dub_progress = m.progress
        elif not m.stages:
            ep.dub_stage = "queued"
            ep.dub_stage_index = 0
        else:
            ep.dub_stage = m.stages[-1].name
            ep.dub_stage_index = len(m.stages)
            if m.progress and m.progress.get("stage") == str(m.stages[-1].name):
                ep.dub_progress = m.progress

    def translate(self, series_id: str, episode_ids: list[str]) -> list[str]:
        """Dub each episode (one job). A FAILED job is resumed in place — the pipeline
        restarts at the broken stage and reuses the video/ASR/translation already on
        disk (re-downloading everything on every retry wasted minutes per episode),
        with the providers set now, not the ones frozen when the job was created.
        Only a previously SUCCEEDED dub is replaced from scratch. Returns the job ids."""
        job_ids: list[str] = []
        for episode_id in episode_ids:
            ep = self.store.get_episode(episode_id)
            if not ep or ep.series_id != series_id:
                continue
            job_id = self._resumable_job_id(ep)
            if job_id is None:
                if ep.dub_job_id:
                    try:
                        self.jobs.delete(ep.dub_job_id)
                    except Exception:
                        pass
                job_id = self.jobs.create(JobInput(url=ep.source_url, series_id=series_id, source_video_id=ep.id)).job_id
            else:
                self.jobs.refresh_settings(job_id)
            self.jobs.start_background(job_id)
            self.store.set_episode_dub(ep.id, job_id, DubState.translating)
            job_ids.append(job_id)
        return job_ids

    def _resumable_job_id(self, ep) -> str | None:
        """The episode's existing job id when it can simply be re-run (failed/canceled
        attempt for the SAME video), else None to start a fresh job."""
        if not ep.dub_job_id:
            return None
        try:
            manifest = self.jobs.get(ep.dub_job_id)
        except Exception:
            return None
        if manifest.state not in (JobState.failed, JobState.canceled):
            return None
        if manifest.input.source_video_id and manifest.input.source_video_id != ep.id:
            return None
        return manifest.job_id

    def delete_episode_dub(self, episode_id: str) -> bool:
        """Drop an episode's dub (job + files) and reset it to not-translated."""
        ep = self.store.get_episode(episode_id)
        if not ep:
            return False
        if ep.dub_job_id:
            try:
                self.jobs.delete(ep.dub_job_id)
            except Exception:
                pass
        self.store.set_episode_dub(episode_id, None, DubState.none)
        return True

    # ----- exports --------------------------------------------------------
    def create_export(self, series_id: str, title: str, description: str, hashtags: list[str], episode_job_ids: list[str]) -> Export:
        if not episode_job_ids:
            raise LibraryError("Chọn ít nhất một tập để ghép")
        return self.store.create_export(series_id, title, description, hashtags, episode_job_ids)

    def get_export(self, export_id: str) -> Export | None:
        return self.store.get_export(export_id)

    def list_exports(self, series_id: str) -> list[Export]:
        return self.store.list_exports(series_id)

    def delete_export(self, export_id: str) -> bool:
        export = self.store.get_export(export_id)
        if export:
            self._remove_export_file(export)
        return self.store.delete_export(export_id)

    def start_render(self, export_id: str) -> Export | None:
        export = self.store.get_export(export_id)
        if not export:
            return None
        self.store.update_export(export_id, state=ExportState.rendering.value, error=None)
        threading.Thread(target=self._render, args=(export_id,), daemon=True).start()
        return self.store.get_export(export_id)

    # ----- internals ------------------------------------------------------
    def _render(self, export_id: str) -> None:
        export = self.store.get_export(export_id)
        if not export:
            return
        try:
            inputs = [self._hardsub_path(job_id) for job_id in export.episode_job_ids]
            inputs = [p for p in inputs if p is not None]
            if not inputs:
                raise LibraryError("Không tìm thấy video đã lồng tiếng cho các tập đã chọn")
            out = self.exports_dir / export.series_id / f"{export_id}.mp4"
            work = self.exports_dir / export.series_id / f"{export_id}.work"
            _concat_videos(inputs, out, work, self.settings)
            rel = str(out.relative_to(self.settings.data_dir))
            self.store.update_export(export_id, state=ExportState.done.value, output_path=rel,
                                     duration=_probe_duration(out))
        except Exception as exc:
            self.store.update_export(export_id, state=ExportState.failed.value, error=str(exc))

    def _hardsub_path(self, job_id: str) -> Path | None:
        try:
            manifest = self.jobs.get(job_id)
        except Exception:
            return None
        for art in reversed(manifest.artifacts):
            if art.kind == ArtifactKind.hardsub_mp4:
                path = self.settings.data_dir / art.path
                return path if path.exists() else None
        return None

    def _remove_export_file(self, export: Export) -> None:
        if export.output_path:
            path = (self.settings.data_dir / export.output_path)
            if path.exists():
                path.unlink(missing_ok=True)
        work = self.exports_dir / export.series_id / f"{export.id}.work"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)

    def _reconcile(self, series_id: str) -> None:
        """Refresh each episode's dub_state from its job's CURRENT state.

        Includes finished episodes: a package re-render (the editor's "Render lại video")
        runs the same job again, so a cached 'dubbed' episode must go back to
        'translating' — otherwise the UI shows nothing while the render runs and never
        polls for the result."""
        for ep in self.store.list_episodes(series_id):
            if not ep.dub_job_id:
                continue
            try:
                manifest = self.jobs.get(ep.dub_job_id)
            except Exception:
                continue
            mapped = _JOB_TO_DUB.get(manifest.state, DubState.translating)
            if mapped != ep.dub_state:
                self.store.set_episode_dub(ep.id, ep.dub_job_id, mapped)

    def _episode_from_source(self, series_id: str, video) -> Episode:
        return Episode(
            id=_new_id(),
            series_id=series_id,
            platform=video.platform,
            platform_video_id=str(video.platform_video_id),
            source_url=video.source_url,
            title=video.title,
            thumbnail_url=video.thumbnail_url,
            duration=video.duration,
            uploader=video.uploader,
        )

    def _episode_from_dict(self, series_id: str, platform: Platform, data: dict) -> Episode:
        return Episode(
            id=_new_id(),
            series_id=series_id,
            platform=Platform(data.get("platform") or platform),
            platform_video_id=str(data["platform_video_id"]),
            source_url=data["source_url"],
            title=data.get("title") or "",
            thumbnail_url=data.get("thumbnail_url"),
            duration=data.get("duration"),
            uploader=data.get("uploader"),
        )

    @staticmethod
    def _video_dict(video) -> dict:
        return {
            "platform": video.platform.value,
            "platform_video_id": str(video.platform_video_id),
            "source_url": video.source_url,
            "title": video.title,
            "thumbnail_url": video.thumbnail_url,
            "duration": video.duration,
            "uploader": video.uploader,
        }

    # ----- title translation (bilingual display) --------------------------
    def translate_titles(self, series_id: str) -> int:
        """Batch-translate untranslated episode titles to Vietnamese (kept alongside the
        original for bilingual display). Best-effort — LLM hiccup leaves titles as-is."""
        pending = [e for e in self.store.list_episodes(series_id) if not e.title_vi]
        if not pending:
            return 0
        lines = "\n".join(f"{i}\t{e.title}" for i, e in enumerate(pending))
        system = (
            "Dịch tiêu đề tập phim sang tiếng Việt: TỰ NHIÊN, ngắn gọn, cuốn hút; GIỮ số tập "
            "(vd '第37集'→'Tập 37'); BỎ hashtag và ký hiệu thừa. Chỉ trả JSON: "
            '[{"i": chỉ_số, "vi": "tiêu đề tiếng Việt"}]'
        )
        try:
            raw = complete(system, f"Danh sách (chỉ_số<TAB>tiêu_đề):\n{lines}", self._agent())
            data = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        except Exception:
            return 0
        count = 0
        for item in data:
            i, vi = item.get("i"), (item.get("vi") or "").strip()
            if isinstance(i, int) and 0 <= i < len(pending) and vi:
                self.store.set_episode_title_vi(pending[i].id, vi)
                count += 1
        return count

    def _translate_titles_bg(self, series_id: str) -> None:
        threading.Thread(target=lambda: self.translate_titles(series_id), daemon=True).start()

    # ----- export metadata suggestion (YouTube title/description) ----------
    def suggest_export_meta(self, series_id: str, episode_ids: list[str]) -> dict:
        """Ask the LLM for a catchy YouTube title + description + hashtags for the
        compilation, from the series name and the (Vietnamese) titles of its episodes."""
        series = self.store.get_series(series_id)
        eps = [self.store.get_episode(eid) for eid in episode_ids]
        eps = [e for e in eps if e and e.series_id == series_id]
        if not series or not eps:
            raise LibraryError("Chọn tập đã dịch để gợi ý tiêu đề")
        listing = "\n".join(f"{i+1}. {e.title_vi or e.title}" for i, e in enumerate(eps))
        system = (
            "Bạn là chuyên gia đặt tiêu đề YouTube cho kênh THUYẾT MINH phim/truyện Việt (kiểu triệu view). "
            "Từ tên bộ và danh sách các tập trong MỘT video tổng hợp, hãy tạo:\n"
            "- title: tiêu đề CUỐN HÚT, gây tò mò, chuẩn YouTube, ≤100 ký tự (có thể nêu 'Tập X-Y' hoặc 'Trọn bộ');\n"
            "- description: 2-4 câu tóm nội dung hấp dẫn + 1 câu kêu gọi đăng ký/theo dõi;\n"
            "- hashtags: 5-8 hashtag tiếng Việt liên quan, mỗi tag KHÔNG có dấu cách.\n"
            'Chỉ trả JSON: {"title": "...", "description": "...", "hashtags": ["#...", "#..."]}'
        )
        user = f"Tên bộ: {series.title}\nCác tập trong video (theo thứ tự):\n{listing}"
        try:
            raw = complete(system, user, self._agent())
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception as exc:
            raise LibraryError("Không tạo được gợi ý (agent lỗi)") from exc
        tags = data.get("hashtags") or []
        return {
            "title": (data.get("title") or "").strip(),
            "description": (data.get("description") or "").strip(),
            "hashtags": [t if str(t).startswith("#") else f"#{t}" for t in tags if str(t).strip()],
        }

    # ----- LLM-assisted naming / clustering -------------------------------
    def _agent(self) -> AgentConfig:
        """The agent bound to the naming task (current settings; env when never saved)."""
        return chat_agent(SettingsStore(self.settings).load(), "naming", self.settings)

    def _refine_with_llm(self, groups: list[Group], videos: list) -> list[Group]:
        """Use the LLM only when the heuristic is unsure: many titles lack a "第X集"
        number (can't trust the reset boundary → re-cluster), or a group has no show
        hashtag to name it (→ ask for a name). Best-effort: any failure keeps heuristic."""
        if not videos:
            return groups
        missing_ratio = sum(1 for v in videos if episode_number(v.title) is None) / len(videos)
        weak = [g for g in groups if not any(show_tags(v.title) for v in g.videos)]
        if missing_ratio <= 0.3 and not weak:
            return groups
        try:
            if missing_ratio > 0.3:
                return self._llm_cluster(videos) or groups
            for group in weak:
                name = self._llm_name(group.videos)
                if name:
                    group.title = name
            return groups
        except Exception:
            return groups

    def _llm_cluster(self, videos: list) -> list[Group] | None:
        lines = "\n".join(f"{i}\t{v.title}" for i, v in enumerate(videos))
        system = (
            "Bạn phân loại video của một kênh reup thành các BỘ phim. Một bộ là một câu chuyện "
            "liên tục, đánh số tập (第X集). Sắp các tập trong bộ theo đúng thứ tự. Đặt tên bộ ngắn "
            "gọn theo nội dung/tag. Chỉ trả JSON: "
            '[{"name":"...","episodes":[chỉ_số_theo_thứ_tự_tập]}]'
        )
        raw = complete(system, f"Danh sách (chỉ_số<TAB>tiêu_đề):\n{lines}", self._agent())
        data = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        groups: list[Group] = []
        for grp in data:
            picked = [videos[i] for i in grp.get("episodes", []) if isinstance(i, int) and 0 <= i < len(videos)]
            if picked:
                groups.append(Group(title=str(grp.get("name") or "Bộ mới").strip(), videos=picked))
        return groups or None

    def _llm_name(self, videos: list) -> str | None:
        titles = "\n".join(v.title for v in videos[:8])
        # Grounding guard: titles like "第72集" carry no content and the model used to
        # INVENT a show name from thin air ("Tình Yêu Không Hẹn Mà Đến" for a survival
        # story). It must name only from words that exist, or decline.
        system = (
            "Đặt MỘT tên bộ phim ngắn gọn (≤6 từ) CHỈ DÙNG thông tin có thật trong các tiêu đề tập. "
            "Nếu tiêu đề chỉ là số tập, không có nội dung gì — trả về đúng chuỗi KHONG. "
            "Chỉ trả về tên, không giải thích."
        )
        name = complete(system, titles, self._agent()).strip().splitlines()[0].strip()
        if not name or name.upper() in {"KHONG", "KHÔNG"}:
            return None
        return name[:60]


def _match_group(series_title: str, known_ids: set[str], groups: list[Group]) -> Group | None:
    """Pick the cluster that belongs to a series: most overlap with its known episodes,
    else a title match, else (single cluster) that one."""
    if not groups:
        return None
    if len(groups) == 1:
        return groups[0]
    best, best_overlap = None, 0
    for group in groups:
        overlap = sum(1 for v in group.videos if str(v.platform_video_id) in known_ids)
        if overlap > best_overlap:
            best, best_overlap = group, overlap
    if best:
        return best
    title = (series_title or "").strip()
    return next((g for g in groups if g.title and (g.title in title or title in g.title)), None)


def _concat_videos(inputs: list[Path], output: Path, work_dir: Path, settings) -> None:
    """Concat episodes into one video, re-encoding as little as possible.

    Hardsub now emits canonical 1080p/30fps/h264 + aac, so in the common case every clip
    already shares a spec and we stream-copy them straight together (no second encode).
    Only if a clip is off-spec (e.g. a legacy hardsub) do we normalise that clip first."""
    if not shutil.which("ffmpeg"):
        raise LibraryError("Cần ffmpeg để ghép video")
    output.parent.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    from backend.app.infrastructure.ffmpeg_encode import video_encode_args, canonical_scale_filter
    target = (settings.video_width, settings.video_height, "h264")
    scale_filter = canonical_scale_filter(settings)

    clips: list[Path] = []
    for index, src in enumerate(inputs):
        if _probe_video_params(src) == target:
            clips.append(src)  # already canonical -> copy as-is
            continue
        norm = work_dir / f"{index:03d}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", scale_filter,
             *video_encode_args(settings),
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(norm)],
            check=True, capture_output=True, text=True,
        )
        clips.append(norm)

    list_file = work_dir / "concat.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in clips), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(output)],
        check=True, capture_output=True, text=True,
    )
    shutil.rmtree(work_dir, ignore_errors=True)


def _probe_video_params(path: Path) -> tuple[int, int, str] | None:
    """(width, height, codec) of the first video stream, or None on failure.

    Read fields by name (key=value) — ffprobe's CSV order is not the requested order."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        fields = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        return int(fields["width"]), int(fields["height"]), fields["codec_name"]
    except Exception:
        return None


def _probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return round(float(out), 2)
    except Exception:
        return None
